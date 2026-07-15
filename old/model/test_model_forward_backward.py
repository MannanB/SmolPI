from pathlib import Path
import os
import sys

import torch

# Allow running this file directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.smolpi import Observation, SmolPI, SmolPIConfig


def _get_image_shape(model: SmolPI) -> tuple[int, int, int]:
    vision_cfg = model.smolvlm_with_expert.smolvlm.config.vision_config
    num_channels = int(getattr(vision_cfg, "num_channels", 3))
    image_size = getattr(vision_cfg, "image_size", 384)

    if isinstance(image_size, (tuple, list)):
        height, width = int(image_size[0]), int(image_size[1])
    else:
        height = width = int(image_size)

    # Keep test runtime practical while still using the real model path.
    height = min(height, 64)
    width = min(width, 64)
    return num_channels, height, width


def _make_observation(model: SmolPI, device: torch.device, batch_size: int) -> Observation:
    num_channels, height, width = _get_image_shape(model)
    text_cfg = model.smolvlm_with_expert.smolvlm.config.text_config
    vocab_size = int(getattr(text_cfg, "vocab_size", 8192))
    prompt_len = 8

    return Observation(
        images={
            "front": torch.randn(
                batch_size, num_channels, height, width, device=device, dtype=torch.float32
            )
        },
        image_masks={"front": torch.ones(batch_size, device=device, dtype=torch.bool)},
        tokenized_prompt=torch.randint(
            low=0, high=vocab_size, size=(batch_size, prompt_len), device=device, dtype=torch.long
        ),
        tokenized_prompt_mask=torch.ones(batch_size, prompt_len, device=device, dtype=torch.bool),
        state=torch.randn(batch_size, model.config.action_dim, device=device, dtype=torch.float32),
    )


def main() -> None:
    torch.manual_seed(0)
    requested_device = os.environ.get("SMOLPI_TEST_DEVICE", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SMOLPI_TEST_DEVICE=cuda was requested but CUDA is not available.")
    device = torch.device(requested_device)

    cfg = SmolPIConfig()
    if device.type == "cpu" and cfg.precision in (torch.float16, torch.bfloat16):
        cfg.precision = torch.float32

    print(f"Using device: {device}")
    print(f"Loading models: {cfg.smolvlm_id} + {cfg.action_expert_id}")
    model = SmolPI(cfg).to(device)
    model.train()

    batch_size = 1
    observation = _make_observation(model, device, batch_size)
    actions = torch.randn(
        batch_size, cfg.action_horizon, cfg.action_dim, device=device, dtype=torch.float32
    )

    with torch.autocast(device_type=device.type, dtype=cfg.precision):

        print("Running forward pass...")
        loss_per_dim = model(observation, actions)
        if loss_per_dim.shape != actions.shape:
            raise RuntimeError(f"Unexpected loss shape: {loss_per_dim.shape} != {actions.shape}")
        if not torch.isfinite(loss_per_dim).all():
            raise RuntimeError("Forward pass produced non-finite values.")

        loss = loss_per_dim.mean()
        print(f"Forward pass OK. Mean loss: {loss.item():.6f}")

        print("Running backward pass...")
        model.zero_grad(set_to_none=True)
        loss.backward()

        grad = model.action_out_proj.weight.grad
        if grad is None:
            raise RuntimeError("Backward pass failed: action_out_proj.weight.grad is None.")
        if not torch.isfinite(grad).all():
            raise RuntimeError("Backward pass produced non-finite gradients.")

        print("Backward pass OK.")


if __name__ == "__main__":
    main()
