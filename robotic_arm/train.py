import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # hack

from contextlib import nullcontext
import pickle

import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoProcessor

from model.smolpi import Observation, SmolPI, SmolPIConfig
from robotic_arm.config import Config

from data import build_bridge_dataset, bridge_batch_to_torch

def train(policy: SmolPI, dataset: torch.utils.data.IterableDataset, cfg: Config, optimizer: torch.optim.Optimizer):

    amp_context = nullcontext()
    if cfg.device.type != "cpu":
        amp_context = lambda: torch.autocast(device_type=cfg.device.type, dtype=cfg.smolpi.precision)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            (step + 1) / max(1, cfg.warmup_steps), 1.0
        ) if cfg.warmup_steps > 0 else 1.0,
    )

    processor = AutoProcessor.from_pretrained(cfg.smolpi.smolvlm_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    policy.train()
    losses: list[float] = []
    accumulated_batches = 0
    time_until_eval = 0
    try:
        for epoch in range(cfg.epochs):
            epoch_dataset = dataset.take(cfg.max_batches) if cfg.max_batches else dataset
            progress = tqdm.tqdm(
                epoch_dataset.as_numpy_iterator(), desc=f"Epoch {epoch + 1}/{cfg.epochs}", unit="batch", total=cfg.max_batches)

            for batch in progress:
                observation, actions = bridge_batch_to_torch(batch, cfg.device, processor)

                with amp_context():
                    loss = policy(observation, actions).mean()
                    (loss / cfg.grad_accum_steps).backward()

                accumulated_batches += 1

                if accumulated_batches == cfg.grad_accum_steps:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_batches = 0

                if time_until_eval % 250 == 0:
                    # check a sample
                    policy.eval()
                    with torch.no_grad():
                        with amp_context():
                            action_chunk = policy.sample_actions(
                                cfg.device,
                                observation,
                                num_steps=cfg.flow_steps,
                            )

                            loss2 = F.mse_loss(action_chunk, actions).mean().item()
                        tqdm.tqdm.write(f"Eval Sample loss: {loss2:.4f}")
                    policy.train()
                time_until_eval += 1

                loss_value = loss.detach().item()
                losses.append(loss_value)
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
            pickle.dump({"loss": losses}, metrics_file)

def main() -> None:
    precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=7, action_horizon=5, precision=precision),
        use_8bit_adam=False
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

    train(policy, dataset, cfg, optimizer)


if __name__ == "__main__":
    main()
