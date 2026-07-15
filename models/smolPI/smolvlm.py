
from typing import Literal

import torch
from torch import nn
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import SmolVLMConfig
from transformers import SmolVLMForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.llama import modeling_llama


class SmolVLMWithExpertModel(nn.Module):
    def __init__(self, smolvlm_id, vlm_config_hf, action_expert_config_hf, \
                 precision: Literal["bfloat16", "float16", "float32"] | torch.dtype = "bfloat16"):
        super().__init__()

        self.smolvlm = SmolVLMForConditionalGeneration.from_pretrained(smolvlm_id, config=vlm_config_hf).to(precision)

        self.action_expert = LlamaForCausalLM(config=action_expert_config_hf)

        self.action_expert.model.embed_tokens = None
        self.action_expert.lm_head = None

        self.freeze_smolvlm()
        self.to_bfloat16_for_selected_params(precision)
        self.keep_action_expert_params_float32()

    def freeze_smolvlm(self):
        """Keep the base SmolVLM fixed while training the action expert."""
        self.smolvlm.eval()
        for param in self.smolvlm.parameters():
            param.requires_grad = False

    def _module_layer_key(self, module_name: str) -> tuple[str, int] | None:
        parts = module_name.split(".")
        for idx, part in enumerate(parts[:-1]):
            if part == "layers" and parts[idx + 1].isdigit():
                return ".".join(parts[: idx + 1]), int(parts[idx + 1])
        return None

    def train(self, mode: bool = True):
        super().train(mode)
        self.smolvlm.eval()
        return self

    def keep_action_expert_params_float32(self):
        """Keep expert master weights and Adam state precise under autocast."""
        for param in self.action_expert.parameters():
            if param.requires_grad:
                param.data = param.data.to(dtype=torch.float32)

    def to_bfloat16_for_selected_params(
        self, precision: Literal["bfloat16", "float16", "float32"] | torch.dtype = "bfloat16"
    ):
        if isinstance(precision, torch.dtype):
            if precision == torch.bfloat16:
                precision = "bfloat16"
            elif precision == torch.float16:
                precision = "float16"
            elif precision == torch.float32:
                precision = "float32"
            else:
                raise ValueError(f"Invalid precision dtype: {precision}")

        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float16":
            self.to(dtype=torch.float16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "model.vision_model.embeddings.patch_embedding.weight",
            "model.vision_model.embeddings.patch_embedding.bias",
            "model.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.text_model.norm",
            "action_expert.model.norm",
            "model.norm",
            "layer_norm"
        ]


        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor | None):
        return self.smolvlm.model.get_image_features(pixel_values, pixel_attention_mask).to(pixel_values.device)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.smolvlm.model.get_input_embeddings()(tokens).to(tokens.device)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
    ):
        if inputs_embeds[1] is None:
            prefix_output = self.smolvlm.model.text_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.action_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.smolvlm.model.text_model, self.action_expert.model]
            num_layers = self.smolvlm.config.text_config.num_hidden_layers

            total_seq_len = inputs_embeds[0].shape[1] + inputs_embeds[1].shape[1]
            batch_size = inputs_embeds[0].shape[0]
            hidden_size = self.smolvlm.config.text_config.hidden_size

            rope_input = torch.zeros(
                batch_size,
                total_seq_len,
                hidden_size,
                device=inputs_embeds[0].device,
                dtype=inputs_embeds[0].dtype,
            )
            position_embeddings = models[0].rotary_emb(rope_input, position_ids)

            def compute_layer_complete(layer_idx, layer_inputs, layer_attention_mask):
                query_states = []
                key_states = []
                value_states = []
                input_lengths = []
                residuals = []

                for i, hidden_states in enumerate(layer_inputs):
                    layer = models[i].layers[layer_idx]

                    residual = hidden_states
                    residuals.append(residual)
                    input_lengths.append(hidden_states.shape[1])

                    normed_hidden_states = layer.input_layernorm(hidden_states)

                    target_dtype = layer.self_attn.q_proj.weight.dtype
                    if normed_hidden_states.dtype != target_dtype:
                        normed_hidden_states = normed_hidden_states.to(dtype=target_dtype)

                    input_shape = normed_hidden_states.shape[:-1]
                    head_dim = layer.self_attn.head_dim

                    num_heads = layer.self_attn.config.num_attention_heads
                    num_kv_heads = layer.self_attn.config.num_key_value_heads

                    query_shape = (*input_shape, num_heads, head_dim)
                    kv_shape = (*input_shape, num_kv_heads, head_dim)

                    query_state = layer.self_attn.q_proj(normed_hidden_states).view(query_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(normed_hidden_states).view(kv_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(normed_hidden_states).view(kv_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                cos, sin = position_embeddings
                query_states, key_states = modeling_llama.apply_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                )

                att_layer = models[0].layers[layer_idx].self_attn

                att_output, _ = modeling_llama.eager_attention_forward(
                    att_layer,
                    query_states,
                    key_states,
                    value_states,
                    layer_attention_mask,
                    scaling=att_layer.scaling,
                    dropout=0.0 if not self.training else att_layer.attention_dropout,
                )

                head_dim = att_layer.head_dim
                num_heads = att_layer.config.num_attention_heads
                att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)

                outputs_embeds = []
                start_pos = 0

                for i, hidden_states in enumerate(layer_inputs):
                    layer = models[i].layers[layer_idx]

                    end_pos = start_pos + input_lengths[i]
                    layer_att_output = att_output[:, start_pos:end_pos]
                    start_pos = end_pos

                    if layer_att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        layer_att_output = layer_att_output.to(layer.self_attn.o_proj.weight.dtype)

                    # Correct Llama residual: original hidden_states + attention output
                    hidden_states = residuals[i] + layer.self_attn.o_proj(layer_att_output)

                    # Correct MLP residual
                    residual = hidden_states
                    mlp_in = layer.post_attention_layernorm(hidden_states)
                    mlp_out = layer.mlp(mlp_in)
                    hidden_states = residual + mlp_out

                    outputs_embeds.append(hidden_states)

                return outputs_embeds

            for layer_idx in range(num_layers):
                inputs_embeds = compute_layer_complete(layer_idx, inputs_embeds, attention_mask)

            prefix_output = models[0].norm(inputs_embeds[0])
            suffix_output = models[1].norm(inputs_embeds[1])
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
