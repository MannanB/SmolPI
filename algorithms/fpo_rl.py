import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tqdm
import wandb
from torch import nn

from algorithms.base import BaseAlgorithm
from algorithms.factory import register_algorithm
from core.config import FPORLConfig
from core.types import EnvObservation, Observation
from environments.base import BaseEnvironment
from models.base import BaseModel


@dataclass
class Transition:
    observation: Observation
    action: torch.Tensor
    reward: float = 0.0
    done: bool = False
    value: float = 0.0
    advantage: float = 0.0
    ret: float = 0.0


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state.float()).squeeze(-1)


def compute_gae(
    transitions: list[Transition],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> None:
    next_value = last_value
    next_advantage = 0.0
    for transition in reversed(transitions):
        nonterminal = 0.0 if transition.done else 1.0
        delta = transition.reward + gamma * next_value * nonterminal - transition.value
        transition.advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        transition.ret = transition.advantage + transition.value
        next_value = transition.value
        next_advantage = transition.advantage


@register_algorithm("FPORL")
class FPORLTrainer(BaseAlgorithm):
    def __init__(
        self,
        config: FPORLConfig,
        policy: BaseModel,
        device: torch.device,
        wandb_run: wandb.Run | None = None,
        environment: BaseEnvironment | None = None,
    ) -> None:
        if environment is None:
            raise ValueError("FPORL requires an environment")

        self.config = config
        self.policy = policy
        self.device = device
        self.wandb_run = wandb_run
        self.environment = environment
        self.value_net = ValueNet(policy.config.action_dim, config.value_hidden_dim).to(device)
        self.policy_optimizer = torch.optim.AdamW(
            (parameter for parameter in policy.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_net.parameters(),
            lr=config.value_learning_rate,
        )
        self.update = 0

    def _amp_context(self):
        if not self.config.use_amp or self.device.type == "cpu":
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.policy.config.precision)

    def _copy_observation(self, observation: EnvObservation) -> Observation:
        return Observation(
            env=EnvObservation(
                images=[image.detach().cpu().clone() for image in observation.images],
                robot_state=observation.robot_state.detach().cpu().clone(),
            ),
            prompt=self.config.prompt,
        )

    def _states(self, observations: list[Observation]) -> torch.Tensor:
        return torch.stack(
            [observation.env.robot_state for observation in observations],
        ).to(self.device)

    def _collect_rollouts(self, exploration_std: float) -> list[Transition]:
        transitions: list[Transition] = []

        for _ in range(self.config.episodes_per_update):
            episode_transitions: list[list[Transition]] = []
            get_action = self._get_rollout_action(
                episode_transitions,
                exploration_std,
            )

            self.policy.eval()
            self.value_net.eval()
            rewards = self.environment.rollout(
                self.config.rollout_seconds,
                self.policy.config.action_horizon,
                get_action,
            )

            if len(rewards) != max((len(items) for items in episode_transitions), default=0):
                raise RuntimeError("Rollout rewards do not match collected transitions")

            for step, step_rewards in enumerate(rewards):
                if len(step_rewards) != len(episode_transitions):
                    raise RuntimeError("Rollout reward batch does not match environment batch")
                for env_index, reward in enumerate(step_rewards):
                    transition = episode_transitions[env_index][step]
                    transition.reward = self.config.reward_scale * float(reward)
                    transition.done = step == len(rewards) - 1

            for items in episode_transitions:
                compute_gae(
                    items,
                    last_value=0.0,
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                )
                transitions.extend(items)

        return transitions

    def _get_rollout_action(
        self,
        episode_transitions: list[list[Transition]],
        exploration_std: float,
    ):
        def get_action(env_observations: list[EnvObservation]) -> torch.Tensor:
            observations = [self._copy_observation(item) for item in env_observations]
            if not episode_transitions:
                episode_transitions.extend([] for _ in observations)

            with torch.no_grad(), self._amp_context():
                actions = self.policy.sample_actions(observations).float()
                values = self.value_net(self._states(observations)).float()

            noise = torch.randn_like(actions) * exploration_std
            actions = torch.clamp(
                actions + noise,
                -self.config.action_clip,
                self.config.action_clip,
            )

            for index, observation in enumerate(observations):
                episode_transitions[index].append(
                    Transition(
                        observation=observation,
                        action=actions[index].detach().cpu(),
                        value=float(values[index].detach().cpu()),
                    )
                )
            return actions

        return get_action

    def _advantage_weights(
        self,
        transitions: list[Transition],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_advantages = torch.tensor(
            [transition.advantage for transition in transitions],
            dtype=torch.float32,
        )
        mean = raw_advantages.mean()
        std = raw_advantages.std(unbiased=False).clamp_min(1e-6)
        advantages = (raw_advantages - mean) / std
        weights = torch.exp(advantages / self.config.awr_temperature).clamp(
            max=self.config.max_awr_weight
        )

        if self.config.advantage_filter == "positive":
            weights = torch.where(advantages > 0.0, weights, torch.zeros_like(weights))
        elif self.config.advantage_filter == "top_quantile":
            threshold = torch.quantile(advantages, self.config.advantage_quantile)
            weights = torch.where(advantages >= threshold, weights, torch.zeros_like(weights))
        elif self.config.advantage_filter != "all":
            raise ValueError(f"Unsupported advantage filter {self.config.advantage_filter!r}")

        returns = torch.tensor(
            [transition.ret for transition in transitions],
            dtype=torch.float32,
        )
        if self.config.value_target_clip > 0:
            returns = returns.clamp(
                -self.config.value_target_clip,
                self.config.value_target_clip,
            )
        return raw_advantages, weights, returns

    def _train_rollouts(self, transitions: list[Transition]) -> dict[str, float]:
        self.policy.train()
        self.value_net.train()
        advantages, weights, returns = self._advantage_weights(transitions)
        policy_losses = []
        value_losses = []

        progress = tqdm.tqdm(
            total=self.config.train_epochs * math.ceil(len(transitions) / self.config.batch_size),
            desc=f"Update {self.update}/{self.config.epochs}",
            unit="batch",
        )

        try:
            for _ in range(self.config.train_epochs):
                indices = torch.randperm(len(transitions))
                for start in range(0, len(indices), self.config.batch_size):
                    batch_indices = indices[start : start + self.config.batch_size].tolist()
                    batch = [transitions[index] for index in batch_indices]
                    observations = [transition.observation for transition in batch]
                    actions = torch.stack([transition.action for transition in batch]).to(
                        self.device
                    )
                    batch_weights = weights[batch_indices].to(self.device)
                    batch_returns = returns[batch_indices].to(self.device)

                    with self._amp_context():
                        flow_loss = self.policy.bc_loss(observations, actions).mean(dim=(1, 2))
                        weight_sum = batch_weights.sum()
                        if float(weight_sum.detach().cpu()) <= 1e-6:
                            policy_loss = flow_loss.mean()
                        else:
                            policy_loss = (batch_weights * flow_loss).sum() / weight_sum

                    self.policy_optimizer.zero_grad(set_to_none=True)
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.policy_optimizer.step()

                    predicted_values = self.value_net(self._states(observations))
                    value_loss = nn.functional.huber_loss(
                        predicted_values,
                        batch_returns,
                        delta=self.config.value_huber_delta,
                    )
                    self.value_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.value_net.parameters(),
                        self.config.value_max_grad_norm,
                    )
                    self.value_optimizer.step()

                    policy_losses.append(float(policy_loss.detach().cpu()))
                    value_losses.append(float(value_loss.detach().cpu()))
                    progress.update(1)
                    progress.set_postfix(
                        policy_loss=f"{policy_losses[-1]:.4f}",
                        value_loss=f"{value_losses[-1]:.4f}",
                    )
        finally:
            progress.close()

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std(unbiased=False)),
            "weight_mean": float(weights.mean()),
            "active_weight_fraction": float((weights > 0).float().mean()),
            "return_target_mean": float(returns.mean()),
            "return_target_std": float(returns.std(unbiased=False)),
        }

    def train(self, loader=None) -> None:
        try:
            for update in range(1, self.config.epochs + 1):
                self.update = update
                transitions = self._collect_rollouts(self.config.exploration_std)
                if not transitions:
                    raise RuntimeError("Collected no rollout transitions")
                metrics = self._train_rollouts(transitions)
                metrics["update"] = self.update
                metrics["mean_return"] = float(
                    np.mean([transition.ret for transition in transitions])
                )

                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {f"train/{key}": value for key, value in metrics.items()},
                        step=self.update,
                    )

                if (
                    self.config.checkpoint_every_steps > 0
                    and self.update % self.config.checkpoint_every_steps == 0
                ):
                    self.save_checkpoint(
                        self.config.checkpoint_dir / f"update_{self.update:08d}.pt"
                    )
        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()

    def evaluate(self) -> dict[str, float]:
        transitions = self._collect_rollouts(0.0)
        if not transitions:
            return {"mean_return": 0.0}
        return {
            "mean_return": float(np.mean([transition.ret for transition in transitions])),
            "mean_reward": float(np.mean([transition.reward for transition in transitions])),
        }

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value_net.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "value_optimizer_state_dict": self.value_optimizer.state_dict(),
                "update": self.update,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self.update = checkpoint["update"]
