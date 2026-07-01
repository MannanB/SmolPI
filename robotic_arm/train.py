from __future__ import annotations

import argparse
from contextlib import nullcontext
import pickle
from pathlib import Path
import sys
from typing import Any

import torch
import tqdm
from transformers import AutoProcessor
import tensorflow as tf
import tensorflow_datasets as tfds

# Support both `python -m robotic_arm.train` and `python robotic_arm/train.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.smolpi import Observation, SmolPI, SmolPIConfig
from robotic_arm.config import Config


BRIDGE_IMAGE_KEYS = ("image_0", "image_1", "image_2", "image_3")


def normalize_image_batch(images: torch.Tensor) -> torch.Tensor:
    images = images.to(dtype=torch.float32)
    if images.numel() == 0:
        return images
    if float(images.max()) > 1.0:
        images = images / 255.0
    if float(images.min()) >= 0.0:
        images = (images - 0.5) / 0.5
    return images


def _format_bridge_prompts(instructions: Any, processor: Any) -> dict[str, torch.Tensor]:
    prompts = []
    for instruction in instructions:
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        instruction = str(instruction).strip()
        prompts.append(
            processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": instruction}]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return processor.tokenizer(
        prompts, padding=True, truncation=True, return_tensors="pt"
    )


def bridge_batch_to_torch(
    batch: dict[str, Any], device: torch.device, processor: Any
) -> tuple[Observation, torch.Tensor]:
    """Convert one already-batched RLDS result to the SmolPI batch layout."""
    images: dict[str, torch.Tensor] = {}
    image_masks: dict[str, torch.Tensor] = {}
    all_image_masks = []
    batch_size = len(batch["language_instruction"])

    for key in BRIDGE_IMAGE_KEYS:
        image = torch.tensor(batch[key])
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(f"Bridge {key} must be BHWC RGB, got {tuple(image.shape)}")
        valid = image.reshape(batch_size, -1).ne(0).any(dim=1)
        all_image_masks.append(valid)
        if not valid.any():
            # Avoid running the vision tower for a dummy camera absent from
            # every observation in this batch.
            continue
        images[key] = normalize_image_batch(image.permute(0, 3, 1, 2).contiguous()).to(
            device=device, non_blocking=True
        )
        image_masks[key] = valid.to(device=device, dtype=torch.bool, non_blocking=True)

    if not torch.stack(all_image_masks, dim=1).any(dim=1).all():
        raise ValueError("A Bridge observation has no valid camera images")

    tokens = _format_bridge_prompts(batch["language_instruction"], processor)
    observation = Observation(
        images=images,
        image_masks=image_masks,
        tokenized_prompt=tokens["input_ids"].to(device=device, dtype=torch.long),
        tokenized_prompt_mask=tokens["attention_mask"].to(device=device, dtype=torch.bool),
        state=torch.tensor(batch["state"], device=device, dtype=torch.float32),
    )
    actions = torch.tensor(batch["actions"], device=device, dtype=torch.float32)
    return observation, actions


def build_bridge_dataset(
    dataset_path: Path,
    *,
    split: str,
    action_horizon: int,
    batch_size: int,
    shuffle_buffer: int,
):
    builder = tfds.builder_from_directory(dataset_path)
    episodes = builder.as_dataset(split=split, shuffle_files=True)

    def episode_to_windows(episode):
        steps = episode["steps"]
        action_windows = (
            steps.map(
                lambda step: step["action"],
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .window(action_horizon, shift=1, drop_remainder=True)
            .flat_map(lambda window: window.batch(action_horizon, drop_remainder=True))
        )

        def make_transition(step, actions):
            observation = step["observation"]
            return {
                **{key: observation[key] for key in BRIDGE_IMAGE_KEYS},
                "state": observation["state"],
                "language_instruction": step["language_instruction"],
                "actions": actions,
            }
        return tf.data.Dataset.zip((steps, action_windows)).map(
            make_transition, num_parallel_calls=tf.data.AUTOTUNE
        )

    transitions = episodes.interleave(
        episode_to_windows,
        cycle_length=8,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    if shuffle_buffer > 0:
        transitions = transitions.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    return transitions.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    precision = torch.float16 if torch.cuda.is_available() else torch.float32
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=7, action_horizon=1, precision=precision),
        use_8bit_adam=True
    )
    policy = SmolPI(cfg.smolpi).to(cfg.device)
    trainable_params = (parameter for parameter in policy.parameters() if parameter.requires_grad)
    if cfg.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    optimizer.zero_grad(set_to_none=True)

    processor = AutoProcessor.from_pretrained(cfg.smolpi.smolvlm_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    dataset = build_bridge_dataset(
        cfg.dataset,
        split=cfg.split,
        action_horizon=cfg.smolpi.action_horizon,
        batch_size=cfg.batch_size,
        shuffle_buffer=cfg.shuffle_buffer,
    )
    amp_context = (
        lambda: torch.autocast(device_type=cfg.device.type, dtype=cfg.smolpi.precision)
        if cfg.device.type != "cpu"
        else nullcontext()
    )

    policy.train()
    losses: list[float] = []
    accumulated_batches = 0
    try:
        for epoch in range(cfg.epochs):
            epoch_dataset = dataset.take(cfg.max_batches) if cfg.max_batches else dataset
            progress = tqdm.tqdm(
                epoch_dataset.as_numpy_iterator(), desc=f"Epoch {epoch + 1}/{cfg.epochs}", unit="batch", total=cfg.max_batches)
            for batch in progress:
                observation, actions = bridge_batch_to_torch(batch, cfg.device, processor)
                if actions.shape[1:] != (
                    cfg.smolpi.action_horizon,
                    cfg.smolpi.action_dim,
                ):
                    raise ValueError(
                        "Unexpected Bridge action shape: "
                        f"{tuple(actions.shape)}; expected [batch, "
                        f"{cfg.smolpi.action_horizon}, {cfg.smolpi.action_dim}]"
                    )

                with amp_context():
                    loss = policy(observation, actions).mean()
                    (loss / cfg.grad_accum_steps).backward()

                accumulated_batches += 1
                if accumulated_batches == cfg.grad_accum_steps:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_batches = 0

                loss_value = loss.detach().item()
                losses.append(loss_value)
                progress.set_postfix(loss=f"{loss_value:.4f}")

    finally:
        torch.save(policy.state_dict(), "smolpi_bridge.pth")
        with open("training_metrics.pkl", "wb") as metrics_file:
            pickle.dump({"loss": losses}, metrics_file)


if __name__ == "__main__":
    main()
