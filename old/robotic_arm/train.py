import sys
import math
import importlib
import pickle
from contextlib import nullcontext
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # hack

import torch
import tqdm
from transformers import AutoProcessor

from model.smolpi import Observation, SmolPI, SmolPIConfig, sample_noise
from robotic_arm.config import Config

from data import build_bridge_dataset, bridge_batch_to_torch

def prepare_next_batch(iterator, device, processor):
    try:
        batch = next(iterator)
    except StopIteration:
        return None
    return bridge_batch_to_torch(batch, device, processor)

def background_batches(batches, device, processor, buffer_size):
    iterator = iter(batches)
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [
            pool.submit(prepare_next_batch, iterator, device, processor)
            for _ in range(buffer_size)
        ]
        while futures:
            batch = futures.pop(0).result()
            if batch is None:
                return
            futures.append(pool.submit(prepare_next_batch, iterator, device, processor))
            yield batch

def build_lr_scheduler(optimizer, cfg):
    if cfg.max_batches <= 0:
        raise ValueError("max_batches must be positive to use cosine LR decay")
    if not 0.0 <= cfg.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    total_steps = max(
        1,
        (cfg.epochs * cfg.max_batches) // cfg.grad_accum_steps,
    )
    warmup_steps = min(cfg.warmup_steps, total_steps)
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_multiplier(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)

def init_wandb(cfg):
    if not cfg.use_wandb:
        return None
    try:
        import wandb
    except ImportError as error:
        raise ImportError("Install wandb or set use_wandb=False") from error

    return wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config={
            "smolvlm_id": cfg.smolpi.smolvlm_id,
            "action_expert_id": cfg.smolpi.action_expert_id,
            "action_dim": cfg.smolpi.action_dim,
            "action_horizon": cfg.smolpi.action_horizon,
            "precision": str(cfg.smolpi.precision),
            "epochs": cfg.epochs,
            "learning_rate": cfg.lr,
            "batch_size": cfg.batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "effective_batch_size": cfg.batch_size * cfg.grad_accum_steps,
            "warmup_steps": cfg.warmup_steps,
            "max_batches": cfg.max_batches,
            "weight_decay": cfg.weight_decay,
        },
    )

def save_checkpoint(
    policy: SmolPI,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: Path,
    *,
    epoch: int,
    batch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "batch": batch,
            "action_horizon": policy.config.action_horizon,
        },
        path,
    )

def train(policy: SmolPI, dataset: torch.utils.data.IterableDataset, cfg: Config, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:

    if cfg.checkpoint_every_batches < 0 or cfg.video_every_batches < 0:
        raise ValueError("Batch artifact intervals must be non-negative")

    amp_context = nullcontext
    if cfg.device.type != "cpu":
        amp_context = lambda: torch.autocast(device_type=cfg.device.type, dtype=cfg.smolpi.precision)

    processor = AutoProcessor.from_pretrained(cfg.smolpi.smolvlm_id)
    processor.image_processor.do_image_splitting = False
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    wandb_run = init_wandb(cfg)
    policy.train()
    losses: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []
    conditioning_gaps: list[float] = []
    accumulated_batches = 0
    total_batches = 0
    try:
        for epoch in range(cfg.epochs):
            epoch_dataset = dataset.take(cfg.max_batches) if cfg.max_batches else dataset
            progress = tqdm.tqdm(
                epoch_dataset.as_numpy_iterator(), desc=f"Epoch {epoch + 1}/{cfg.epochs}", unit="batch", total=cfg.max_batches)

            for observation, actions in background_batches(
                progress, cfg.device, processor, cfg.prefetch_batches
            ):
                wandb_metrics = {}
                with amp_context():
                    loss = policy(observation, actions).mean()
                    (loss / cfg.grad_accum_steps).backward()

                accumulated_batches += 1
                total_batches += 1

                if accumulated_batches == cfg.grad_accum_steps:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_batches = 0

                if cfg.checkpoint_every_batches > 0 and total_batches % cfg.checkpoint_every_batches == 0:
                    checkpoint_path = (
                        Path(cfg.checkpoint_dir)
                        / f"smolpi_bridge_batch_{total_batches:08d}.pth"
                    )
                    save_checkpoint(
                        policy,
                        optimizer,
                        scheduler,
                        checkpoint_path,
                        epoch=epoch + 1,
                        batch=total_batches,
                    )
                    tqdm.tqdm.write(f"Saved checkpoint: {checkpoint_path}")

                if cfg.video_every_batches > 0 and total_batches % cfg.video_every_batches == 0:
                    video_path = (
                        Path(cfg.video_dir)
                        / f"smolpi_bridge_pick_red_box_batch_{total_batches:08d}.mp4"
                    )
                    try_module = importlib.import_module("robotic_arm.try")
                    try_module.record_policy(
                        policy,
                        video_path,
                        processor=processor,
                    )
                    tqdm.tqdm.write(f"Saved trial video: {video_path}")

              
                loss_value = loss.detach().item()
                losses.append(loss_value)
                if wandb_run is not None:
                    wandb_metrics.update({
                        "train/loss": loss_value,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    })
                    wandb_run.log(wandb_metrics, step=total_batches)

                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    grad_norm=f"{float(grad_norm):.3f}"
                    if accumulated_batches == 0
                    else "-",
                )

    finally:
        torch.save(policy.state_dict(), "smolpi_bridge.pth")
        with open("training_metrics.pkl", "wb") as metrics_file:
            pickle.dump(
                {
                    "loss": losses,
                    "eval_step": eval_steps,
                    "eval_loss": eval_losses,
                    "conditioning_gap": conditioning_gaps,
                },
                metrics_file,
            )
        if wandb_run is not None:
            wandb_run.finish()

def main() -> None:
    precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=7, action_horizon=1, precision=precision),
        use_8bit_adam=False,
        use_wandb=True,
        wandb_project="smolpi",
        wandb_run_name="bridge-10k",
    )
    policy = SmolPI(cfg.smolpi).to(cfg.device)
    trainable_params = (parameter for parameter in policy.parameters() if parameter.requires_grad)
    if cfg.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    dataset = build_bridge_dataset(
        cfg.dataset,
        stats_path=Path("action_stats.npz"),
        split=cfg.split,
        action_horizon=cfg.smolpi.action_horizon,
        batch_size=cfg.batch_size,
        shuffle_buffer=cfg.shuffle_buffer,
    )

    scheduler = build_lr_scheduler(optimizer, cfg)


    if cfg.resume_checkpoint is not None:
        checkpoint = torch.load(cfg.resume_checkpoint)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint["batch"]

        print(f"Resumed from checkpoint: {cfg.resume_checkpoint} (epoch {start_epoch}, batch {start_batch})")

    train(policy, dataset, cfg, optimizer, scheduler)


if __name__ == "__main__":
    main()
