import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from model.smolpi import SmolPI, Observation, SmolPIConfig
from environment import MujocoEnvironment
from contextlib import nullcontext
from objectives import make_random_objective
from config import Config


class ReplayBuffer:
    def __init__(self, capacity: int):

        self.capacity = capacity
        self.samples: list[dict] = []

    def add(self, samples: list[dict]) -> None:
        self.samples.extend(samples)
        overflow = len(self.samples) - self.capacity
        if overflow > 0:
            del self.samples[:overflow]

    def shuffled_samples(self) -> list[dict]:
        samples = list(self.samples)
        np.random.shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)


def stack_rollout_batch(samples: list[dict], device: torch.device) -> tuple[Observation, dict[str, torch.Tensor], torch.Tensor]:
    if not samples:
        raise ValueError("cannot stack an empty sample list")

    images = torch.stack([s["image"] for s in samples], dim=0).to(device=device, dtype=torch.float32)
    image_masks = torch.stack([s["image_mask"] for s in samples], dim=0).to(device=device, dtype=torch.bool)
    tokenized_prompt = pad_sequence(
        [s["prompt_ids"].to(dtype=torch.long) for s in samples],
        batch_first=True,
        padding_value=0,
    ).to(device=device, dtype=torch.long)
    tokenized_prompt_mask = pad_sequence(
        [s["prompt_mask"].to(dtype=torch.bool) for s in samples],
        batch_first=True,
        padding_value=False,
    ).to(device=device, dtype=torch.bool)
    state = torch.stack([s["state"] for s in samples], dim=0).to(device=device, dtype=torch.float32)
    actions = {
        "final_actions": torch.stack([s["actions"] for s in samples], dim=0).to(device=device, dtype=torch.float32),
        "noisy_actions": torch.stack([s["noisy_actions"] for s in samples], dim=0).to(device=device, dtype=torch.float32),
        "next_noisy_actions": torch.stack([s["next_noisy_actions"] for s in samples], dim=0).to(device=device, dtype=torch.float32),
        "timesteps": torch.stack([s["timesteps"] for s in samples], dim=0).to(device=device, dtype=torch.float32),
        "old_log_probs": torch.stack([s["old_log_probs"] for s in samples], dim=0).to(device=device, dtype=torch.float32),
        "transition_std": torch.tensor(float(samples[0]["transition_std"]), device=device, dtype=torch.float32),
    }
    rewards = torch.tensor([s["reward"] for s in samples], device=device, dtype=torch.float32)

    obs = Observation(
        images={"front": images},
        image_masks={"front": image_masks},
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        state=state,
    )

    return obs, actions, rewards


def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    advantages = rewards - rewards.mean()
    reward_std = rewards.std(unbiased=False)
    if torch.isfinite(reward_std) and reward_std > 1e-6:
        advantages = advantages / (reward_std + 1e-6)
    return advantages.clamp(-5.0, 5.0).detach()


def DDPO_minibatch_loss(
    policy: SmolPI,
    cfg: Config,
    observations: Observation,
    actions: dict[str, torch.Tensor],
    advantages: torch.Tensor,
):
    if not isinstance(actions, dict):
        raise ValueError("Exact DDPO_update requires the action dict returned by stack_rollout_batch after DDPO rollout collection")

    noisy_actions = actions["noisy_actions"].to(dtype=torch.float32)
    device = noisy_actions.device
    next_noisy_actions = actions["next_noisy_actions"].to(device=device, dtype=torch.float32)
    timesteps = actions["timesteps"].to(device=device, dtype=torch.float32)
    old_log_probs = actions["old_log_probs"].to(device=device, dtype=torch.float32).detach()
    transition_std = actions["transition_std"]
    advantages = advantages.to(device=device, dtype=torch.float32).detach()
    observations = Observation(
        images={k: v.to(device=device, dtype=torch.float32) for k, v in observations.images.items()},
        image_masks={k: v.to(device=device, dtype=torch.bool) for k, v in observations.image_masks.items()},
        tokenized_prompt=observations.tokenized_prompt.to(device=device, dtype=torch.long),
        tokenized_prompt_mask=observations.tokenized_prompt_mask.to(device=device, dtype=torch.bool),
        state=observations.state.to(device=device, dtype=torch.float32),
    )

    obs_batch = next(iter(observations.images.values())).shape[0]
    target_batch = min(obs_batch, noisy_actions.shape[0], advantages.shape[0])
    if obs_batch != target_batch:
        observations = Observation(
            images={k: v[:target_batch] for k, v in observations.images.items()},
            image_masks={k: v[:target_batch] for k, v in observations.image_masks.items()},
            tokenized_prompt=observations.tokenized_prompt[:target_batch],
            tokenized_prompt_mask=observations.tokenized_prompt_mask[:target_batch],
            state=observations.state[:target_batch],
        )
    if noisy_actions.shape[0] != target_batch:
        noisy_actions = noisy_actions[:target_batch]
        next_noisy_actions = next_noisy_actions[:target_batch]
        timesteps = timesteps[:target_batch]
        old_log_probs = old_log_probs[:target_batch]
    if advantages.shape[0] != target_batch:
        advantages = advantages[:target_batch]

    if advantages.ndim != 1:
        raise ValueError(f"advantages must be [batch], got {advantages.shape}")
    if noisy_actions.ndim != 4:
        raise ValueError(f"noisy_actions must be [batch, denoise_steps, horizon, action_dim], got {noisy_actions.shape}")
    if noisy_actions.shape[0] != advantages.shape[0]:
        raise ValueError(f"actions batch ({noisy_actions.shape[0]}) and advantages batch ({advantages.shape[0]}) must match")

    precision = cfg.smolpi.precision
    if device.type != "cpu" and precision in (torch.float16, torch.bfloat16):
        amp_ctx = torch.autocast(device_type=device.type, dtype=precision)
    else:
        amp_ctx = nullcontext()

    with amp_ctx:
        new_log_probs = policy.ddpo_log_probs(
            observations,
            noisy_actions,
            next_noisy_actions,
            timesteps,
            num_steps=noisy_actions.shape[1],
            transition_std=transition_std,
        )
        log_ratio = (new_log_probs - old_log_probs).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)
        clipped_ratio = ratio.clamp(0.8, 1.2)
        advantages = advantages[:, None].expand_as(new_log_probs)
        surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        loss = -surrogate.mean() + cfg.kl_coef * approx_kl

    if not torch.isfinite(loss):
        raise RuntimeError("DDPO_minibatch_loss produced a non-finite loss")

    return loss, {
        "loss": float(loss.detach().item()),
        "mean_advantage": float(advantages.mean().detach().item()),
        "approx_kl": float(approx_kl.detach().item()),
        "clip_fraction": float(((ratio - 1.0).abs() > 0.2).to(torch.float32).mean().detach().item()),
    }


def DDPO_update(
    policy: SmolPI,
    cfg: Config,
    optimizer: torch.optim.Optimizer,
    samples: list[dict],
):
    if not samples:
        raise ValueError("DDPO_update requires at least one sample")

    rewards = torch.tensor([s["reward"] for s in samples], dtype=torch.float32)
    advantages = compute_advantages(rewards)
    total_samples = min(len(samples), 200)

    policy.train()
    optimizer.zero_grad(set_to_none=True)

    loss_sum = 0.0
    kl_sum = 0.0
    clip_sum = 0.0
    advantage_sum = 0.0
    num_minibatches = 0

    for start in range(0, total_samples, cfg.batch_size):
        end = min(start + cfg.batch_size, total_samples)
        batch_size = end - start
        weight = batch_size / total_samples

        minibatch_observations, minibatch_actions, _ = stack_rollout_batch(samples[start:end], cfg.device)
        minibatch_advantages = advantages[start:end].to(device=cfg.device)

        loss, minibatch_metrics = DDPO_minibatch_loss(
            policy,
            cfg,
            minibatch_observations,
            minibatch_actions,
            minibatch_advantages,
        )
        (loss * weight).backward()

        loss_sum += minibatch_metrics["loss"] * weight
        kl_sum += minibatch_metrics["approx_kl"] * weight
        clip_sum += minibatch_metrics["clip_fraction"] * weight
        advantage_sum += minibatch_metrics["mean_advantage"] * weight
        num_minibatches += 1

    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": loss_sum,
        "mean_reward": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "min_reward": float(rewards.min().item()),
        "max_reward": float(rewards.max().item()),
        "mean_advantage": advantage_sum,
        "approx_kl": kl_sum,
        "clip_fraction": clip_sum,
        "num_train_samples": int(total_samples),
        "num_minibatches": num_minibatches,
    }

def main():
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=2, action_horizon=5, precision=torch.float16)
    )
    env = MujocoEnvironment(cfg)

    policy = SmolPI(cfg.smolpi).to(cfg.device)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay)
    policy.eval()

    update_pbar = tqdm(range(cfg.num_updates), desc="DDPO Updates", unit="update")
    replay_buffer = ReplayBuffer(cfg.replay_buffer_capacity)

    past_metrics = []

    try:
        for update in update_pbar:
            objectives = [make_random_objective(cfg) for _ in range(cfg.num_parallel_rollouts)]
            all_samples = env.rollout(
                policy=policy,
                reward_models=objectives,
                write_to_video=True,
            )
            episode_returns = [
                sum(s["reward"] for s in all_samples if s["env_idx"] == env_idx)
                for env_idx in range(cfg.num_parallel_rollouts)
            ]
            tqdm.write(
                "collected {} parallel rollouts with {} samples and mean episode return {:.4f}".format(
                    cfg.num_parallel_rollouts,
                    len(all_samples),
                    float(np.mean(episode_returns)),
                )
            )

            replay_buffer.add(all_samples)
            train_samples = replay_buffer.shuffled_samples()
            metrics = DDPO_update(policy, cfg, optimizer, train_samples)
            mean_episode_return = float(np.mean(episode_returns))
            # print(
            #     f"update={update} samples={len(all_samples)} "
            #     f"episode_return={mean_episode_return:.4f} "
            #     f"reward={metrics['mean_reward']:.4f} reward_std={metrics['reward_std']:.4f} "
            #     f"reward_range=[{metrics['min_reward']:.4f},{metrics['max_reward']:.4f}] loss={metrics['loss']:.6f} "
            #     f"kl={metrics['approx_kl']:.6f} clip={metrics['clip_fraction']:.3f}"
            # )
            update_pbar.set_postfix({
                "reward": metrics["mean_reward"],
                "reward_std": metrics["reward_std"],
                "episode_return": mean_episode_return,
                # "train_samples": metrics["num_train_samples"],
                "batches": metrics["num_minibatches"],
                # "replay": len(replay_buffer),
                "kl": metrics["approx_kl"],
                "clip": metrics["clip_fraction"],
            })

            past_metrics.append({
                "update": update,
                "samples": len(all_samples),
                "replay_size": len(replay_buffer),
                "episode_return": mean_episode_return,
                **metrics,
            })
    finally:
        env.close()
        torch.save({
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "past_metrics": past_metrics,
        }, "ddpo_training.pt")
        import pickle
        with open("past_metrics.pkl", "wb") as f:
            pickle.dump(past_metrics, f)


if __name__ == "__main__":
    main()
