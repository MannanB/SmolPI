import copy
import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from transformers import LlamaConfig, SmolVLMConfig

from core.config import SmolPIConfig
from core.types import Observation
from models.base import BaseModel
from models.factory import register_model
from models.smolpi.smolvlm import SmolVLMWithExpertModel

# thanks openpi / physical intelligence / pi0 for a lot of this code


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def sample_noise(shape, device):
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )


def sample_time(bsize, device):
    time_beta = sample_beta(1.5, 1.0, bsize, device)
    time = time_beta * 0.99 + 0.01
    return time.to(dtype=torch.float32, device=device)


PADDING_MARKER = -1
CONTINUE_ATTENTION_BLOCK = 0
START_NEW_ATTENTION_BLOCK = 1


def build_blockwise_attention_mask(attention_block_markers):
    """Build a prefix-LM attention mask and position IDs.

    Each marker describes how its token relates to the preceding token:

    * ``-1``: padding token
    * ``0``: continue the current attention block
    * ``1``: start a new attention block

    Tokens within a block attend bidirectionally to one another. They can also
    attend every earlier block, but never a later block. Padding is excluded as
    both a query and a key.
    """
    if attention_block_markers.ndim != 2:
        raise ValueError(
            "attention_block_markers must have shape [batch, sequence], "
            f"got {tuple(attention_block_markers.shape)}"
        )

    valid_token_mask = attention_block_markers != PADDING_MARKER

    # Padding must not increment the cumulative block number.
    block_starts = attention_block_markers.masked_fill(~valid_token_mask, CONTINUE_ATTENTION_BLOCK)
    block_ids = torch.cumsum(block_starts, dim=1)

    query_block_ids = block_ids[:, :, None]
    key_block_ids = block_ids[:, None, :]
    can_attend = key_block_ids <= query_block_ids

    valid_query_and_key = valid_token_mask[:, :, None] & valid_token_mask[:, None, :]
    attention_mask = can_attend & valid_query_and_key

    position_ids = torch.cumsum(valid_token_mask.long(), dim=1) - 1
    position_ids = position_ids.masked_fill(~valid_token_mask, 0)

    return attention_mask, position_ids


@register_model("smolpi")
class SmolPI(BaseModel):
    def __init__(self, config: SmolPIConfig):
        super().__init__()
        self.config = config

        vlm_config = SmolVLMConfig.from_pretrained(
            config.smolvlm_id
        )  # HuggingFaceTB/SmolVLM-256M-Instruct
        action_expert_config = LlamaConfig.from_pretrained(
            config.action_expert_id
        )  # HuggingFaceTB/SmolLM2-135M

        self.smolvlm_with_expert = SmolVLMWithExpertModel(
            config.smolvlm_id,
            vlm_config,
            action_expert_config,
            precision=config.precision,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.hidden_size)
        self.action_out_proj = nn.Linear(action_expert_config.hidden_size, config.action_dim)

        self.state_proj = nn.Linear(config.action_dim, action_expert_config.hidden_size)
        self.action_time_mlp_in = nn.Linear(
            2 * action_expert_config.hidden_size, action_expert_config.hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            action_expert_config.hidden_size, action_expert_config.hidden_size
        )

        self.dtype = config.precision
        self.gradient_checkpointing_enable()

        torch.set_float32_matmul_precision("high")
        # self.sample_actions = torch.compile(self.sample_actions, dynamic=False)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.smolvlm_with_expert.smolvlm.model.text_model.gradient_checkpointing = True
        self.smolvlm_with_expert.smolvlm.model.vision_model.gradient_checkpointing = True
        self.smolvlm_with_expert.action_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.smolvlm_with_expert.smolvlm.model.text_model.gradient_checkpointing = False
        self.smolvlm_with_expert.smolvlm.model.vision_model.gradient_checkpointing = False
        self.smolvlm_with_expert.action_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _to_additive_attention_mask(self, attention_mask):
        """Convert [batch, query, key] visibility to a transformer mask."""
        attention_mask = attention_mask[:, None, :, :]
        return torch.where(attention_mask, 0.0, -2.3819763e38)

    def preprocess_observations(
        self,
        observations: list[Observation],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if not observations:
            raise ValueError("observations cannot be empty")

        processor = self.processor
        processor.image_processor.do_image_splitting = False

        device = next(self.parameters()).device
        batch_size = len(observations)

        # Build prompts with one image token per camera.
        batch_messages = [
            [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image"} for _ in observation.env.images]
                        + [{"type": "text", "text": observation.prompt.strip()}]
                    ),
                }
            ]
            for observation in observations
        ]

        batch_prompts = processor.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        image_prompt = (
            processor.fake_image_token
            + processor.global_image_tag
            + processor.image_token * processor.image_seq_len
            + processor.fake_image_token
        )

        batch_prompts = [
            prompt.replace(processor.image_token, image_prompt) for prompt in batch_prompts
        ]

        batch_inputs = dict(
            processor.tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
            )
        )

        num_images = [len(observation.env.images) for observation in observations]

        if any(count == 0 for count in num_images):
            raise ValueError("Every observation must contain at least one image")

        if any(count > 2 for count in num_images):
            raise ValueError("At most two camera images are supported")

        max_num_images = max(num_images)
        image_size = processor.image_processor.max_image_size["longest_edge"]

        pixel_values = torch.zeros(
            batch_size,
            max_num_images,
            3,
            image_size,
            image_size,
            dtype=torch.float32,
        )

        pixel_attention_mask = torch.zeros(
            batch_size,
            max_num_images,
            image_size,
            image_size,
            dtype=torch.bool,
        )

        for batch_index, observation in enumerate(observations):
            for image_index, image in enumerate(observation.env.images):
                image = image.detach().cpu()

                if image.ndim != 3:
                    raise ValueError(
                        f"Expected a 3D RGB image, received shape {tuple(image.shape)}"
                    )

                # Accept both CHW and HWC RGB tensors.
                if image.shape[0] == 3:
                    pass
                elif image.shape[-1] == 3:
                    image = image.permute(2, 0, 1)
                else:
                    raise ValueError(f"Expected an RGB image, received shape {tuple(image.shape)}")

                image = image.to(torch.float32)

                image = TF.resize(
                    image,
                    [image_size, image_size],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )

                minimum = image.amin().item()
                maximum = image.amax().item()

                # Normalize RGB values to [-1, 1].
                if minimum >= 0.0 and maximum <= 1.0:
                    image = image.mul(2.0).sub(1.0)
                elif minimum >= -1.0 and maximum <= 1.0:
                    # Already normalized.
                    pass
                else:
                    image = image.clamp(0.0, 255.0).div(127.5).sub(1.0)

                pixel_values[batch_index, image_index] = image
                pixel_attention_mask[batch_index, image_index] = True

        robot_state = torch.stack(
            [
                observation.env.robot_state.detach().cpu().to(torch.float32)
                for observation in observations
            ]
        )

        batch_inputs["pixel_values"] = pixel_values
        batch_inputs["pixel_attention_mask"] = pixel_attention_mask

        if device.type == "cuda":
            batch_inputs = {key: value.pin_memory() for key, value in batch_inputs.items()}
            robot_state = robot_state.pin_memory()

        batch_inputs = {
            key: value.to(device=device, non_blocking=True) for key, value in batch_inputs.items()
        }
        robot_state = robot_state.to(device=device, non_blocking=True)

        return batch_inputs, robot_state

    def embed_prefix(self, inputs):
        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.smolvlm_with_expert.embed_language_tokens(lang_tokens)
            return lang_emb.to(inputs.input_ids.device)

        lang_emb = self._apply_checkpoint(lang_embed_func, inputs.input_ids)

        image_hidden_states = self.smolvlm_with_expert.embed_image(
            inputs.pixel_values, inputs.pixel_attention_mask
        )

        embs = self.smolvlm_with_expert.smolvlm.model.inputs_merger(
            input_ids=inputs.input_ids,
            inputs_embeds=lang_emb,
            image_hidden_states=image_hidden_states,
        )

        valid_prefix_tokens = inputs.attention_mask.bool()  # [batch, prefix]

        prefix_block_markers = torch.where(
            valid_prefix_tokens,
            torch.zeros_like(inputs.attention_mask, dtype=torch.long),
            torch.full_like(inputs.attention_mask, PADDING_MARKER, dtype=torch.long),
        )

        return embs, prefix_block_markers

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Action Expert processing."""
        embs = []
        attention_block_markers = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        # Embed state
        def state_proj_func(state):
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)

        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]

        # Set attention masks so that image and language inputs do not attend to state or actions
        attention_block_markers.append(START_NEW_ATTENTION_BLOCK)

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        # Apply MLP layers
        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)  # swish == silu
            return self.action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize = action_time_emb.shape[0]

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        attention_block_markers += [START_NEW_ATTENTION_BLOCK]
        attention_block_markers += [CONTINUE_ATTENTION_BLOCK] * (self.config.action_horizon - 1)

        embs = torch.cat(embs, dim=1)
        attention_block_markers = torch.tensor(
            attention_block_markers, dtype=torch.long, device=embs.device
        )
        attention_block_markers = attention_block_markers[None, :].expand(bsize, -1)

        return embs, attention_block_markers

    def forward(self, processed_inputs, states, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = sample_noise(actions.shape, actions.device)

        if time is None:
            time = sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_block_markers = self.embed_prefix(processed_inputs)

        v_t = self._predict_velocity_from_prefix(
            prefix_embs,
            prefix_block_markers,
            states,
            x_t,
            time,
        )
        return F.mse_loss(u_t, v_t, reduction="none")

    def bc_loss(self, observations, actual_action):
        """Compute the behavior cloning loss (batch_size x num_steps x num_motors)"""
        processed_inputs, states = self.preprocess_observations(observations)
        return self.forward(processed_inputs, states, actual_action)

    def _predict_velocity_from_prefix(
        self,
        prefix_embs,
        prefix_block_markers,
        state,
        x_t,
        time,
    ):
        """Predict velocity through the shared training and sampling path."""
        suffix_embs, suffix_block_markers = self.embed_suffix(state, x_t, time)

        suffix_embs = suffix_embs.to(dtype=self.dtype)
        prefix_embs = prefix_embs.to(dtype=self.dtype)

        attention_block_markers = torch.cat([prefix_block_markers, suffix_block_markers], dim=1)

        attention_mask, position_ids = build_blockwise_attention_mask(attention_block_markers)

        # Prepare attention masks
        additive_attention_mask = self._to_additive_attention_mask(attention_mask)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, additive_attention_mask, position_ids):
            (_, suffix_out), _ = self.smolvlm_with_expert.forward(
                attention_mask=additive_attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            additive_attention_mask,
            position_ids,
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        return self._apply_checkpoint(action_out_proj_func, suffix_out)

    @torch.no_grad()
    def sample_actions(
        self,
        observations,
        noise=None,
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""

        processed_inputs, states = self.preprocess_observations(observations)

        bsize = states.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = sample_noise(actions_shape, self.device)

        # print(noise.shape, sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device).shape)

        prefix_embs, prefix_block_markers = self.embed_prefix(processed_inputs)

        valid_prefix_tokens = prefix_block_markers != PADDING_MARKER
        prefix_attention_mask, prefix_position_ids = build_blockwise_attention_mask(
            prefix_block_markers
        )

        # Compute the image/language KV cache once and reuse it for every flow step.
        additive_prefix_attention_mask = self._to_additive_attention_mask(prefix_attention_mask)
        _, past_key_values = self.smolvlm_with_expert.forward(
            attention_mask=additive_prefix_attention_mask,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = 1.0 / self.config.num_flow_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=self.device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        for step in range(self.config.num_flow_steps):
            time = torch.full(
                (bsize,),
                1.0 - step * dt,
                device=self.device,
                dtype=torch.float32,
            )

            step_key_values = copy.deepcopy(past_key_values)

            v_t = self.denoise_step(
                states,
                valid_prefix_tokens,
                step_key_values,
                x_t,
                time,
            )

            x_t = x_t - dt * v_t

        return x_t

    def denoise_step(
        self,
        state,
        valid_prefix_tokens,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one cached denoising step to the noisy action chunk."""
        suffix_embs, suffix_block_markers = self.embed_suffix(state, x_t, timestep)
        suffix_embs = suffix_embs.to(dtype=self.dtype)

        batch_size, suffix_len = suffix_embs.shape[:2]
        suffix_attention_mask, _ = build_blockwise_attention_mask(suffix_block_markers)
        prefix_attention_mask = valid_prefix_tokens[:, None, :].expand(batch_size, suffix_len, -1)
        attention_mask = torch.cat([prefix_attention_mask, suffix_attention_mask], dim=2)
        additive_attention_mask = self._to_additive_attention_mask(attention_mask)

        prefix_lengths = valid_prefix_tokens.sum(dim=1, keepdim=True)
        suffix_offsets = torch.arange(suffix_len, device=suffix_embs.device, dtype=torch.long)[
            None, :
        ]
        position_ids = prefix_lengths + suffix_offsets

        outputs_embeds, _ = self.smolvlm_with_expert.forward(
            attention_mask=additive_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
