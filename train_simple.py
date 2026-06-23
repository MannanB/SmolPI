from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import tqdm
from transformers import AutoProcessor
from environment import MujocoEnvironment
from model.smolpi import SmolPI, Observation, SmolPIConfig
from config import Config
import random
import os, pickle



class SingleTaskModel(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        self.cnn = nn.Sequential(
            # [B, 3, 512, 512] -> [B, 32, 256, 256]
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            # -> [B, 64, 128, 128]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # -> [B, 128, 64, 64]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # -> [B, 256, 32, 32]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # -> [B, 256, 8, 8]
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, hidden_dim),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 2),
            nn.Tanh()
        )

    def forward(self, img: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = self.cnn(img)
        x = self.flatten(x)
        x = self.head(torch.concat((x, state), dim=1))

        # Scale tanh output from [-1, 1] to [-3, 3]
        return 3.0 * x

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

def action_from_red(image, center_threshold=0.15, min_red_pixels=100):
    """
    image: [3, H, W] RGB tensor, either 0–1 or 0–255.

    Returns:
        [0, 0]   if red is near the center
        [2, -2]  if red is left of center
        [-2, 2]  if red is right of center or missing
    """
    image = torch.as_tensor(image, dtype=torch.float32)

    # Remove an optional leading dimension: [1, 3, H, W] -> [3, H, W]
    if image.ndim == 4:
        image = image.squeeze(0)

    # Normalize 0–255 images to 0–1
    if image.max() > 1.5:
        image = image / 255.0

    r, g, b = image[0], image[1], image[2]

    # Red detection:
    # - allows darker red caused by shadows
    # - requires red to dominate green and blue
    # - avoids most pink/white regions
    red_mask = (
        (r > 0.18)
        & (r > g * 1.35)
        & (r > b * 1.35)
        & ((r - g) > 0.08)
        & ((r - b) > 0.08)
    )

    red_locations = torch.nonzero(red_mask, as_tuple=False)

    if red_locations.shape[0] < min_red_pixels:
        return torch.tensor([[-2.0, 2.0]], dtype=torch.float32), red_mask

    # red_locations columns are [y, x]
    red_center_x = red_locations[:, 1].float().mean()
    image_center_x = image.shape[2] / 2

    # Fraction of image width considered "middle-ish"
    threshold_pixels = image.shape[2] * center_threshold

    if red_center_x < image_center_x - threshold_pixels:
        return torch.tensor([[-2.0, 2.0]], dtype=torch.float32), red_mask

    if red_center_x > image_center_x + threshold_pixels:
        return torch.tensor([[2.0, -2.0]], dtype=torch.float32), red_mask

    return torch.tensor([[0.0, 0.0]], dtype=torch.float32), red_mask

from objectives import *
EVAL_OBJECTIVES = [FaceRedPlatformRewardModel]

def main():
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=2, action_horizon=1, precision=torch.float32),
        cam_sim_output_path="vids/sft_eval_sim.mp4",
        cam_omni_output_path="vids/sft_eval_omni.mp4",
    )

    policy = SingleTaskModel(256).to(cfg.device, dtype=cfg.smolpi.precision)
    policy.train()
    trainable_params = (p for p in policy.parameters() if p.requires_grad)
    print("Params:", sum(p.numel() for p in policy.parameters()))
    if cfg.use_8bit_adam_sft:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer.zero_grad()

    env = MujocoEnvironment(cfg, 512, 512)

    processor = AutoProcessor.from_pretrained(cfg.smolpi.smolvlm_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    data_files = ["synthetic_samples_MoveToRedPlatformRewardModel.pkl"]
    
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
                with open(f'./data/{file}', "rb") as f:
                    samples = pickle.load(f)
                
                num_samples = len(samples)
                num_batches = (num_samples + cfg.batch_size - 1) // cfg.batch_size
                random.shuffle(samples)

                pbar = tqdm.tqdm(range(num_batches), desc="Batches", unit="batch", leave=False)
                for batch in pbar:
                    batch_samples = samples[batch*cfg.batch_size : (batch+1)*cfg.batch_size]

                    imgs = []
                    states = []
                    actions = []
                    actions2 = []

                    for sample in batch_samples:
                        obs = sample["observation"]
                        action = sample["action"]
                        # states.append(obs.state)
                        action_tensor = torch.as_tensor(action, dtype=torch.float32)
                        # if action_tensor.ndim == 1:
                        #     action_tensor = action_tensor.unsqueeze(0)\
                        # action_tensor, red_mask = action_from_red(
                        #     obs.images["front"][0],
                        #     center_threshold=0.15,
                        #     min_red_pixels=100,
                        # )
                        
                        actions.append(action_tensor)
                        imgs.append(obs.images["front"][0])

                        # is_blue = torch.rand(1).item() < 0.5

                        # img = torch.zeros((3, 255, 255), dtype=torch.float32)

                        # if is_blue:
                        #     img[2, :, :] = 1.0  # RGB: blue channel
                        #     action_tensor = torch.tensor([[1.0, 1.0]])
                        # else:
                        #     img[0, :, :] = 1.0  # RGB: red channel
                        #     action_tensor = torch.tensor([[-1.0, -1.0]])

                        state = torch.zeros(2, dtype=torch.float32)

                        # imgs.append(img)
                        states.append(state)
                        # actions.append(action_tensor)

                    observations_tensor = torch.stack(imgs, dim=0).squeeze(1).to(device=cfg.device, dtype=torch.float32)
                    states_tensor = torch.stack(states, dim=0).squeeze(1).to(device=cfg.device, dtype=torch.float32)

                    actions_tensor = torch.stack(actions).to(cfg.device)



                    with amp_ctx:
                        policy_out = policy(observations_tensor, states_tensor)
                        loss = F.mse_loss(policy_out, actions_tensor.squeeze(1))


                        scaled_loss = loss / cfg.grad_accum_steps

                        scaled_loss.backward()

                    if epoch == 9:
                        print(policy(observations_tensor, states_tensor)[:5])
                        input()
                        print("\n\n\n----")
                        print(actions_tensor.squeeze(1)[:5])
                        input()

                    should_step = (batch + 1) % cfg.grad_accum_steps == 0 or (batch + 1) == num_batches
                    if should_step:
                        optimizer.step()
                        optimizer.zero_grad()
                    metrics["loss"].append(loss.item())
                    pbar.set_postfix({"loss": loss.item()})
                # todo: eval
                # print(steps, steps % 4)
                # if steps % 1  == 0:
                #     tqdm.tqdm.write(f"Evaluating policy at epoch {epoch+1}, file {steps}/{len(data_files)}...")
                #     print("wtf")
                #     policy.eval()
                #     print("moved to eval")

                #     # MoveForwardRewardModel, MoveToBluePlatformRewardModel, FacePinkPlatformRewardModel, SpinCounterClockwiseRewardModel
                #     print("hey")
                #     for objective in tqdm.tqdm([FaceRedPlatformRewardModel], desc="Evaluating Objectives", leave=False):
                #         all_samples = env.rollout(
                #             policy=policy,
                #             reward_models=[objective(cfg) for _ in range(cfg.num_parallel_rollouts)],
                #             write_to_video=True,
                #         )
                #         episode_returns = [
                #             sum(s["reward"] for s in all_samples if s["env_idx"] == env_idx)
                #             for env_idx in range(cfg.num_parallel_rollouts)
                #         ]
                #         mean_episode_return = float(np.mean(episode_returns))
                #         metrics[objective.__name__].append(mean_episode_return)
                #         tqdm.tqdm.write(f"File: {file}, Objective: {objective.__name__}, Mean Episode Return: {mean_episode_return:.2f}")
                #     policy.train()
                steps += 1
    finally:
        try:
            env.close()
        finally:
            # save
            torch.save(policy.state_dict(), "test_sft_final.pth")
            with open("sft_training_metrics.pkl", "wb") as f:
                pickle.dump(metrics, f)

if __name__ == "__main__":
    main()
