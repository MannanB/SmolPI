from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm
from environment import MujocoEnvironment
from model.smolpi import SmolPI, Observation, SmolPIConfig
from config import Config
import os, pickle

def stack_observations(observations: list[Observation], device: torch.device) -> tuple[Observation, dict[str, torch.Tensor], torch.Tensor]:
    # print([obs.images["front"].shape for obs in observations])
    # print([obs.image_masks["front"].shape for obs in observations])

    images = torch.stack([s.images["front"] for s in observations], dim=0).squeeze(1).to(device=device, dtype=torch.float32)
    image_masks = torch.stack([s.image_masks["front"] for s in observations], dim=0).squeeze(1).to(device=device, dtype=torch.bool)
    tokenized_prompt = pad_sequence(
        [s.tokenized_prompt.squeeze(0) for s in observations],
        batch_first=True,
        padding_value=0,
    ).to(device=device, dtype=torch.long)
    tokenized_prompt_mask = pad_sequence(
        [s.tokenized_prompt_mask.squeeze(0) for s in observations],
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

from objectives import *
EVAL_OBJECTIVES = [MoveForwardRewardModel, MoveToBluePlatformRewardModel, FacePinkPlatformRewardModel, SpinCounterClockwiseRewardModel]

def main():
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=2, action_horizon=1, precision=torch.float16),
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

    policy.train()

    data_files = os.listdir("./data/shuffled_chunks/")
    data_files = [f for f in data_files if f.startswith("synthetic_samples_shuffled_") and f.endswith(".pkl")]

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
                        actions.append(torch.tensor(action, dtype=torch.float32))

                    observations_tensor = stack_observations(observations, device=cfg.device)

                    actions_tensor = torch.stack(actions).unsqueeze(1).to(cfg.device) # todo: action horizon > 1
                    with amp_ctx:
                        loss = policy(observations_tensor, actions_tensor).mean()

                        loss.backward()

                    if batch % cfg.grad_accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                    pbar.set_postfix({"loss": loss.item()})
                    metrics["loss"].append(loss.item())
                # todo: eval
                # print(steps, steps % 4)
                if steps % 4  == 0:
                    tqdm.tqdm.write(f"Evaluating policy at epoch {epoch+1}, file {steps}/{len(data_files)}...")
                    policy.eval()
                    env = MujocoEnvironment(cfg)
                    for objective in tqdm.tqdm([MoveForwardRewardModel, MoveToBluePlatformRewardModel, FacePinkPlatformRewardModel, SpinCounterClockwiseRewardModel], desc="Evaluating Objectives", leave=False):
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
                    env.close()
                    del env
                    policy.train()
                steps += 1
    finally:
        # save
        torch.save(policy.state_dict(), "smolpi_sft_final.pth")
        with open("sft_training_metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)

if __name__ == "__main__":
    main()
