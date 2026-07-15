from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm
from transformers import AutoProcessor
from environment import MujocoEnvironment, get_vision_input_shape
from model.smolpi import SmolPI, Observation, SmolPIConfig
from two_wheeled.config import Config
import os, pickle

def normalize_image_batch(images: torch.Tensor) -> torch.Tensor:
    if images.numel() > 0 and float(images.min()) >= 0.0 and float(images.max()) <= 1.0:
        return (images - 0.5) / 0.5
    return images

def prompt_tokens_for_observation(observation: Observation, processor) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = processor.tokenizer
    token_ids = observation.tokenized_prompt.squeeze(0).detach().cpu().to(dtype=torch.long)
    token_mask = observation.tokenized_prompt_mask.squeeze(0).detach().cpu().to(dtype=torch.bool)
    valid_ids = token_ids[token_mask] if token_mask.shape == token_ids.shape else token_ids

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    already_formatted = valid_ids.numel() > 0 and (
        int(valid_ids[0]) == tokenizer.bos_token_id or image_token_id in valid_ids.tolist()
    )
    if already_formatted:
        return valid_ids, torch.ones_like(valid_ids, dtype=torch.bool)

    prompt = tokenizer.decode(valid_ids, skip_special_tokens=True).strip()
    formatted_prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
    return tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0).to(dtype=torch.bool)

def stack_observations(observations: list[Observation], device: torch.device, processor) -> tuple[Observation, dict[str, torch.Tensor], torch.Tensor]:
    # print([obs.images["front"].shape for obs in observations])
    # print([obs.image_masks["front"].shape for obs in observations])

    images = torch.stack([s.images["front"] for s in observations], dim=0).squeeze(1).to(device=device, dtype=torch.float32)
    images = normalize_image_batch(images)
    image_masks = torch.stack([s.image_masks["front"] for s in observations], dim=0).squeeze(1).to(device=device, dtype=torch.bool)
    prompt_tokens = [prompt_tokens_for_observation(s, processor) for s in observations]
    tokenized_prompt = pad_sequence(
        [tokens for tokens, _ in prompt_tokens],
        batch_first=True,
        padding_value=0,
    ).to(device=device, dtype=torch.long)
    tokenized_prompt_mask = pad_sequence(
        [mask for _, mask in prompt_tokens],
        batch_first=True,
        padding_value=False,
    ).to(device=device, dtype=torch.bool)
    state = torch.stack([s.state for s in observations], dim=0).squeeze(1).to(device=device, dtype=torch.float32)

    obs = Observation(
        images={"front": images},
        image_masks={"front": image_masks},
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        state=state,
    )

    # print(obs.images["front"].shape, obs.image_masks["front"].shape, obs.tokenized_prompt.shape, obs.tokenized_prompt_mask.shape, obs.state.shape)

    return obs

from two_wheeled.objectives import *
EVAL_OBJECTIVES = [FaceRedPlatformRewardModel, MoveForwardRewardModel, MoveToGreenPlatformRewardModel]

def main():
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=2, action_horizon=5, precision=torch.float16),
        cam_sim_output_path="vids/sft_eval_sim.mp4",
        cam_omni_output_path="vids/sft_eval_omni.mp4",
    )

    policy = SmolPI(cfg.smolpi).to(cfg.device)
    trainable_params = (p for p in policy.parameters() if p.requires_grad)
    if cfg.use_8bit_adam_sft:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer.zero_grad()

    env = MujocoEnvironment(cfg, *get_vision_input_shape(policy))

    policy.train()
    processor = AutoProcessor.from_pretrained(cfg.smolpi.smolvlm_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    data_files = os.listdir("./data/shuffled_chunks/")
    # data_files = [f for f in data_files if f.startswith("synthetic_samples_shuffled_") and f.endswith(".pkl")]
    # data_files = ["synthetic_samples_FaceRedPlatformRewardModel.pkl"]
    
    if cfg.device.type != "cpu" and cfg.smolpi.precision in (torch.float16, torch.bfloat16):
        amp_ctx = torch.autocast(device_type=cfg.device.type, dtype=cfg.smolpi.precision)
    else:
        amp_ctx = nullcontext()

    metrics = {objective.__name__: [] for objective in EVAL_OBJECTIVES}
    metrics["loss"] = []
    steps = 0
    try:
        for epoch in range(cfg.epochs):
            for file in tqdm.tqdm(list(data_files), desc=f"Epoch {epoch+1}/{cfg.epochs}", unit="file"):
                # can't load all at once due to memory restrictions
                with open(f'./data/shuffled_chunks/{file}', "rb") as f:
                    samples = pickle.load(f)
                
                num_samples = len(samples)
                num_batches = (num_samples + cfg.batch_size - 1) // cfg.batch_size

                pbar = tqdm.tqdm(range(num_batches), desc="Batches", unit="batch", leave=False)
                for batch in pbar:
                    batch_samples = samples[batch*cfg.batch_size : (batch+1)*cfg.batch_size]

                    observations = []
                    actions = []

                    for sample in batch_samples:
                        obs = sample["observation"]
                        action = sample["action"]
                        observations.append(obs)
                        action_tensor = torch.as_tensor(action, dtype=torch.float32)
                        if action_tensor.ndim == 1:
                            action_tensor = action_tensor.unsqueeze(0)
                        actions.append(action_tensor)

                    observations_tensor = stack_observations(observations, device=cfg.device, processor=processor)

                    actions_tensor = torch.stack(actions).to(cfg.device)
                
                    if actions_tensor.shape[1] != cfg.smolpi.action_horizon:
                        raise ValueError(
                            f"Sample action horizon {actions_tensor.shape[1]} does not match cfg.smolpi.action_horizon "
                            f"{cfg.smolpi.action_horizon}; regenerate synthetic data with the current horizon."
                        )
                    with amp_ctx:
                        loss = policy(observations_tensor, actions_tensor).mean()
                        scaled_loss = loss / cfg.grad_accum_steps

                        scaled_loss.backward()

                    should_step = (batch + 1) % cfg.grad_accum_steps == 0 or (batch + 1) == num_batches
                    if should_step:
                        optimizer.step()
                        optimizer.zero_grad()
                    metrics["loss"].append(loss.item())
                    pbar.set_postfix({"loss": loss.item()})
                # todo: eval
                # print(steps, steps % 4)
            tqdm.tqdm.write(f"Evaluating policy at epoch {epoch+1}, file {steps}/{len(data_files)}...")
            policy.eval()
            for objective in tqdm.tqdm(EVAL_OBJECTIVES, desc="Evaluating Objectives", leave=False):
                all_samples = env.rollout(
                    policy=policy,
                    reward_models=[objective(cfg) for _ in range(cfg.num_parallel_rollouts)],
                    write_to_video=True,
                )
                episode_returns = [
                    sum(s["reward"] for s in all_samples if s["env_idx"] == env_idx)
                    for env_idx in range(cfg.num_parallel_rollouts)
                ]
                mean_episode_return = float(np.mean(episode_returns))
                metrics[objective.__name__].append(mean_episode_return)
                tqdm.tqdm.write(f"File: {file}, Objective: {objective.__name__}, Mean Episode Return: {mean_episode_return:.2f}")
            policy.train()
    finally:
        try:
            env.close()
        finally:
            # save
            torch.save(policy.state_dict(), "smolpi_sft_final.pth")
            with open("sft_training_metrics.pkl", "wb") as f:
                pickle.dump(metrics, f)

if __name__ == "__main__":
    main()
