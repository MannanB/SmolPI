import sys
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # hack

from contextlib import nullcontext
import pickle

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

def train(policy: SmolPI, dataset: torch.utils.data.IterableDataset, cfg: Config, optimizer: torch.optim.Optimizer):

    amp_context = nullcontext()
    if cfg.device.type != "cpu":
        amp_context = lambda: torch.autocast(device_type=cfg.device.type, dtype=cfg.smolpi.precision)

    scheduler = build_lr_scheduler(optimizer, cfg)

    processor = AutoProcessor.from_pretrained(cfg.smolpi.smolvlm_id)

    processor.image_processor.do_image_splitting = False
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    policy.train()
    losses: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []
    conditioning_gaps: list[float] = []
    accumulated_batches = 0
    time_until_eval = 0
    try:
        for epoch in range(cfg.epochs):
            epoch_dataset = dataset.take(cfg.max_batches) if cfg.max_batches else dataset
            progress = tqdm.tqdm(
                epoch_dataset.as_numpy_iterator(), desc=f"Epoch {epoch + 1}/{cfg.epochs}", unit="batch", total=cfg.max_batches)

            for observation, actions in background_batches(
                progress, cfg.device, processor, cfg.prefetch_batches
            ):

                with amp_context():
                    loss = policy(observation, actions).mean()
                    (loss / cfg.grad_accum_steps).backward()

                accumulated_batches += 1

                if accumulated_batches == cfg.grad_accum_steps:
                    # grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_batches = 0

                if time_until_eval % 250 == 0:
                    sample_count = min(2, actions.shape[0])
                    sample_inputs = type(observation.processed_inputs)(
                        {key: value[:sample_count] for key, value in observation.processed_inputs.items()}
                    )
                    sample_observation = Observation(
                        processed_inputs=sample_inputs,
                        state=observation.state[:sample_count],
                    )
                    policy.eval()
                    with torch.no_grad():
                        with amp_context():
                            sample_actions = actions[:sample_count]
                            eval_noise = sample_noise(sample_actions.shape, cfg.device)
                            eval_time = torch.full(
                                (sample_count,), 0.9, device=cfg.device, dtype=torch.float32
                            )
                            eval_loss = policy(
                                sample_observation,
                                sample_actions,
                                noise=eval_noise,
                                time=eval_time,
                            ).mean()

                            if sample_count > 1:
                                shuffled_inputs = type(sample_inputs)(
                                    {key: value.flip(0) for key, value in sample_inputs.items()}
                                )
                                shuffled_observation = Observation(
                                    processed_inputs=shuffled_inputs,
                                    state=sample_observation.state.flip(0),
                                )
                                shuffled_loss = policy(
                                    shuffled_observation,
                                    sample_actions,
                                    noise=eval_noise,
                                    time=eval_time,
                                ).mean()
                                conditioning_gap = (shuffled_loss - eval_loss).item()
                            else:
                                conditioning_gap = float("nan")

                        eval_steps.append(time_until_eval)
                        eval_losses.append(eval_loss.item())
                        conditioning_gaps.append(conditioning_gap)
                        tqdm.tqdm.write(
                            f"Eval flow loss: {eval_loss.item():.4f}, "
                            f"conditioning gap: {conditioning_gap:+.4f}"
                        )
                    policy.train()
                time_until_eval += 1

                loss_value = loss.detach().item()
                losses.append(loss_value)
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    # grad_norm=f"{float(grad_norm):.3f}"
                    # if accumulated_batches == 0
                    # else "-",
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
