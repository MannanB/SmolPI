import math
from contextlib import nullcontext
from pathlib import Path

import torch
import tqdm
import wandb

from algorithms.base import BaseAlgorithm
from algorithms.factory import register_algorithm
from core.config import BehaviorCloningConfig
from environments.base import BaseEnvironment
from models.base import BaseModel


@register_algorithm("BehaviorCloning")
class BehaviorCloningTrainer(BaseAlgorithm):
    def __init__(
        self,
        config: BehaviorCloningConfig,
        policy: BaseModel,
        device: torch.device,
        wandb_run: wandb.Run | None = None,
        environment: BaseEnvironment | None = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.device = device

        trainable_parameters = (
            parameter for parameter in policy.parameters() if parameter.requires_grad
        )

        if config.use_8bit_adam:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(
                trainable_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler = self._create_scheduler()
        self.wandb_run = wandb_run

        self.global_batch = 0
        self.optimizer_step = 0

    def _create_scheduler(self):
        batches_per_epoch = self.config.max_batches_per_epoch

        if batches_per_epoch is None:
            raise ValueError("max_batches_per_epoch is required for cosine scheduling")

        total_steps = max(1, self.config.epochs * batches_per_epoch // self.config.grad_accum_steps)

        warmup_steps = min(self.config.warmup_steps, total_steps)
        decay_steps = max(1, total_steps - warmup_steps)

        def lr_multiplier(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps

            progress = min(
                max((step - warmup_steps) / decay_steps, 0.0),
                1.0,
            )

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

            return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_multiplier,
        )

    def _amp_context(self):
        if not self.config.use_amp or self.device.type == "cpu":
            return nullcontext()

        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        )

    def train(self, loader: torch.utils.data.DataLoader) -> None:
        self.policy.train()
        accumulated_batches = 0

        try:
            for epoch in range(self.config.epochs):
                progress = tqdm.tqdm(
                    loader,
                    desc=f"Epoch {epoch + 1}/{self.config.epochs}",
                    total=self.config.max_batches_per_epoch,
                    unit="batch",
                )

                for raw_batch in progress:
                    if (
                        self.config.max_batches_per_epoch is not None
                        and accumulated_batches >= self.config.max_batches_per_epoch
                    ):
                        break

                    batch = self.batch_preprocessor(raw_batch)

                    with self._amp_context():
                        loss = self.policy(batch).mean()
                        scaled_loss = loss / self.config.grad_accum_steps

                    scaled_loss.backward()

                    accumulated_batches += 1
                    self.global_batch += 1
                    grad_norm = None

                    if accumulated_batches % self.config.grad_accum_steps == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(),
                            self.config.max_grad_norm,
                        )

                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        self.optimizer_step += 1

                    loss_value = loss.detach().item()

                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {
                                "train/loss": loss_value,
                                "train/learning_rate": (self.optimizer.param_groups[0]["lr"]),
                                "train/global_batch": self.global_batch,
                            },
                            step=self.global_batch,
                        )

                    if (
                        self.config.checkpoint_every_steps > 0
                        and self.global_batch % self.config.checkpoint_every_steps == 0
                    ):
                        self.save_checkpoint(
                            self.config.checkpoint_dir / f"batch_{self.global_batch:08d}.pt",
                            epoch=epoch,
                        )

                    progress.set_postfix(
                        loss=f"{loss_value:.4f}",
                        lr=(f"{self.optimizer.param_groups[0]['lr']:.2e}"),
                        grad_norm=(f"{float(grad_norm):.3f}" if grad_norm is not None else "-"),
                    )

                accumulated_batches = 0

        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "global_batch": self.global_batch,
                "optimizer_step": self.optimizer_step,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_batch = checkpoint["global_batch"]
        self.optimizer_step = checkpoint["optimizer_step"]
