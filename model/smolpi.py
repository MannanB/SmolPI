import logging
import math
import copy
from typing import Literal

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from transformers import LlamaConfig, SmolVLMConfig

from pydantic import BaseModel, ConfigDict

from model.smolvlm import SmolVLMWithExpertModel


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
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
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
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=torch.float32, device=device)

def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks

class SmolPIConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    smolvlm_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    action_expert_id: str = "HuggingFaceTB/SmolLM2-135M"
    train_vlm_with_lora: bool = True
    vlm_lora_rank: int = 8
    vlm_lora_alpha: int = 16
    vlm_lora_dropout: float = 0.05
    vlm_lora_train_layer_fraction: float = 0.10
    vlm_lora_layer_selection: Literal["first", "last"] = "last"
    vlm_lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
    )
    action_dim: int = 16
    action_horizon: int = 10
    precision: torch.dtype = torch.float16
    pytorch_compile_mode: str | None = None

class Observation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    state: torch.Tensor

class SmolPI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        vlm_config = SmolVLMConfig.from_pretrained(config.smolvlm_id) # HuggingFaceTB/SmolVLM-256M-Instruct
        action_expert_config = LlamaConfig.from_pretrained(config.action_expert_id) # HuggingFaceTB/SmolLM2-135M
        
        self.smolvlm_with_expert = SmolVLMWithExpertModel(
            config.smolvlm_id,
            vlm_config,
            action_expert_config,
            precision=config.precision,
            use_vlm_lora=config.train_vlm_with_lora,
            vlm_lora_rank=config.vlm_lora_rank,
            vlm_lora_alpha=config.vlm_lora_alpha,
            vlm_lora_dropout=config.vlm_lora_dropout,
            vlm_lora_target_modules=config.vlm_lora_target_modules,
            vlm_lora_train_layer_fraction=config.vlm_lora_train_layer_fraction,
            vlm_lora_layer_selection=config.vlm_lora_layer_selection,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.hidden_size)
        self.action_out_proj = nn.Linear(action_expert_config.hidden_size, config.action_dim)

        self.state_proj = nn.Linear(config.action_dim, action_expert_config.hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.hidden_size, action_expert_config.hidden_size)
        self.action_time_mlp_out = nn.Linear(action_expert_config.hidden_size, action_expert_config.hidden_size)
        
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

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        # observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )
    
    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer"""
        embs = []
        pad_masks = []
        att_masks = []
        # print(images[0].shape, img_masks[0].shape, lang_tokens.shape, lang_masks.shape, len(images), len(img_masks))

        # Process images
        # print(img_masks[0].shape)
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.smolvlm_with_expert.embed_image(img)
            
            img_emb = self._apply_checkpoint(image_embed_func, img)
            # print("xxx", img_emb.shape, img_mask.shape, img.shape)
            bsize, num_img_embs = img_emb.shape[:2]
            if img_mask.ndim == 0:
                img_mask = img_mask.unsqueeze(0)

            embs.append(img_emb)
            # print(img_mask.shape)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.smolvlm_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        # print(embs.shape, pad_masks.shape, att_masks.shape)
        return embs, pad_masks, att_masks
    
    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Action Expert processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        # Embed state
        def state_proj_func(state):
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)

        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
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

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks
    
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        if noise is None:
            noise = sample_noise(actions.shape, actions.device)

        if time is None:
            time = sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        suffix_embs = suffix_embs.to(dtype=self.dtype)
        prefix_embs = prefix_embs.to(dtype=self.dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids):
            (_, suffix_out), _ = self.smolvlm_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    def denoise_step_from_observation(self, observation, noisy_actions, timestep) -> Tensor:
        """Same v_theta as denoise_step, but recomputes prefix context for training."""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, noisy_actions, timestep)

        suffix_embs = suffix_embs.to(dtype=self.dtype)
        prefix_embs = prefix_embs.to(dtype=self.dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids):
            (_, suffix_out), _ = self.smolvlm_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        return self._apply_checkpoint(action_out_proj_func, suffix_out)

    def ddpo_log_probs(
        self,
        observation,
        noisy_actions: Tensor,
        next_noisy_actions: Tensor,
        timesteps: Tensor,
        num_steps: int,
        transition_std: float | Tensor = 0.1,
    ) -> Tensor:
        """Recompute log p_theta(x_{t-1} | x_t, c) for DDPO/PPO updates."""
        if noisy_actions.ndim != 4:
            raise ValueError(f"noisy_actions must be [batch, denoise_steps, horizon, action_dim], got {noisy_actions.shape}")

        dt = -1.0 / float(num_steps)
        log_probs = []
        for step in range(noisy_actions.shape[1]):
            x_t = noisy_actions[:, step]
            x_prev = next_noisy_actions[:, step]
            time = timesteps[:, step]
            v_t = self.denoise_step_from_observation(observation, x_t, time)
            mean = x_t + dt * v_t
            log_probs.append(self._ddpo_gaussian_log_prob(x_prev, mean, transition_std))
        return torch.stack(log_probs, dim=1)

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        return_ddpo_data=False,
        transition_std: float = 0.1,
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        # self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.smolvlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        noisy_actions = []
        next_noisy_actions = []
        timesteps = []
        log_probs = []
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            # Llama attention updates the cache when provided, so clone the
            # prefix cache each step to keep key/value length stable.
            step_key_values = copy.deepcopy(past_key_values)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                step_key_values,
                x_t,
                expanded_time,
            )

            mean = x_t + dt * v_t
            if return_ddpo_data:
                if transition_std <= 0:
                    raise ValueError("DDPO sampling requires transition_std > 0")
                sampled_x = mean + transition_std * sample_noise(mean.shape, device)
                noisy_actions.append(x_t.detach())
                next_noisy_actions.append(sampled_x.detach())
                timesteps.append(expanded_time.detach())
                log_probs.append(self._ddpo_gaussian_log_prob(sampled_x, mean, transition_std).detach())
                x_t = sampled_x
            else:
                x_t = mean
            time += dt
        if return_ddpo_data:
            return x_t, {
                "noisy_actions": torch.stack(noisy_actions, dim=1),
                "next_noisy_actions": torch.stack(next_noisy_actions, dim=1),
                "timesteps": torch.stack(timesteps, dim=1),
                "old_log_probs": torch.stack(log_probs, dim=1),
                "transition_std": torch.tensor(float(transition_std), device=device, dtype=torch.float32),
            }
        return x_t

    @staticmethod
    def _ddpo_gaussian_log_prob(value: Tensor, mean: Tensor, std: float | Tensor) -> Tensor:
        std = torch.as_tensor(std, device=value.device, dtype=value.dtype)
        var = std.square()
        log_scale = torch.log(std)
        log_prob = -0.5 * ((value - mean).square() / var + 2.0 * log_scale + math.log(2.0 * math.pi))
        return log_prob.flatten(1).sum(dim=1)

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        # self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.smolvlm_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
