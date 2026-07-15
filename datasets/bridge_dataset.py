import numpy as np
import tensorflow as tf
from pathlib import Path

from core.config import BridgeDatasetConfig
from datasets.factory import register_dataloader

tf.config.set_visible_devices([], "GPU")  # Disable GPU for TF to avoid memory conflicts with PyTorch

import tqdm
from typing import Any
import tensorflow_datasets as tfds

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

BRIDGE_IMAGE_KEYS = ("image_0", "image_1") #, "image_2", "image_3")

def compute_action_stats(episodes, action_dim):
    actions = episodes.flat_map(
        lambda episode: episode["steps"].map(
            lambda step: tf.cast(step["action"], tf.float32)
        )
    ).batch(50)

    count = 0
    total = np.zeros(action_dim, dtype=np.float64)
    total_sq = np.zeros(action_dim, dtype=np.float64)

    for batch in tqdm.tqdm(
        actions.as_numpy_iterator(),
        desc="Computing action stats",
        unit="batch",
    ):
        batch = batch.astype(np.float64)
        count += len(batch)
        total += batch.sum(axis=0)
        total_sq += np.square(batch).sum(axis=0)

    print(f"Processed {count} actions for stats computation.")

    mean = total / count
    variance = total_sq / count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    
    print("Action mean:", mean)
    print("Action std: ", std)

    return {
        "count": count,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def load_or_compute_action_stats(episodes: tf.data.Dataset, action_dim: int, stats_path: Path):
    if stats_path.exists():
        loaded = np.load(stats_path)
        print("Loaded action statistics", "mean:", loaded["mean"], "std:", loaded["std"])  
        return {
            "count": loaded["count"],
            "mean": loaded["mean"],
            "std": loaded["std"],
        }

    print("Computing action statistics...")

    stats = compute_action_stats(episodes, action_dim=action_dim)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path,**stats)

    print("Action mean:", stats["mean"])
    print("Action std: ", stats["std"])

    return stats

def build_bridge_dataset(
    dataset_path: Path,
    stats_path: Path,
    split: str,
    action_horizon: int,
    batch_size: int,
    shuffle_buffer: int,
):
    builder = tfds.builder_from_directory(dataset_path)
    episodes = builder.as_dataset(split=split, shuffle_files=True)
    episodes = episodes.filter(
        lambda episode: episode["episode_metadata"]["has_language"]
    )

    stats_episodes = (
        episodes
        # .shuffle(5_000, seed=42, reshuffle_each_iteration=False) shuffling hangs
        .take(1000)  
    )

    action_stats = load_or_compute_action_stats(stats_episodes, action_dim=7, stats_path=stats_path)

    action_mean = tf.constant(action_stats["mean"], dtype=tf.float32)
    action_std = tf.constant(action_stats["std"], dtype=tf.float32)

    def normalize_action(action):
        normalized = (tf.cast(action, tf.float32) - action_mean) / tf.maximum(action_std, 1e-6)
        # Stops extreme dataset outliers from dominating training.
        return tf.clip_by_value(normalized, -5.0, 5.0)

    def episode_to_windows(episode):
        steps = episode["steps"]
        action_windows = (
            steps.map(
                lambda step: normalize_action(step["action"]),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .window(action_horizon, shift=1, drop_remainder=True)
            .flat_map(lambda window: window.batch(action_horizon, drop_remainder=True))
        )

        def make_transition(step, actions):
            observation = step["observation"]
            return {
                **{key: observation[key] for key in BRIDGE_IMAGE_KEYS},
                "state": tf.cast(observation["state"], tf.float32),
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


class BridgeDataset(IterableDataset):
    def __init__(
        self,
        config: BridgeDatasetConfig,
        action_horizon: int,

    ) -> None:
        self.config = config
        self.action_horizon = action_horizon

    def __iter__(self) -> Iterator[dict[str, object]]:
        tf_dataset = build_bridge_dataset(
            dataset_path=self.config.dataset_path,
            stats_path=self.config.stats_path,
            split=self.config.split,
            action_horizon=self.action_horizon,
            batch_size=self.config.batch_size,
            shuffle_buffer=self.config.shuffle_buffer,
        )

        for batch in tf_dataset.as_numpy_iterator():
            yield {
                "image_0": torch.from_numpy(batch["image_0"]),
                "image_1": torch.from_numpy(batch["image_1"]),
                "state": torch.from_numpy(batch["state"]).float(),
                "actions": torch.from_numpy(batch["actions"]).float(),
                # Keep strings as Python/NumPy objects until tokenization.
                "language_instruction": batch["language_instruction"],
            }

@register_dataloader("Bridge")
def create_bridge_data_loader(config: BridgeDatasetConfig, action_horizon: int) -> torch.utils.data.DataLoader:
    dataset = BridgeDataset(config=config, action_horizon=action_horizon)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # Already batched in the dataset.
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
    )