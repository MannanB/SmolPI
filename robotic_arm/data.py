from platform import processor

import numpy as np
import torch
import tensorflow as tf
from pathlib import Path

import tqdm

from model.smolpi import Observation
from typing import Any

import tensorflow_datasets as tfds


BRIDGE_IMAGE_KEYS = ("image_0", "image_1") #, "image_2", "image_3")


def bridge_batch_to_torch(
    batch: dict[str, Any], device: torch.device, processor: Any
) -> tuple[Observation, torch.Tensor]:

    batch_size = len(batch["language_instruction"])

    batch_messages = [
        [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in range(2)] + # up to two cams for memory
                    [{"type": "text", "text": str(batch["language_instruction"][i]).strip()}],
            }
        ] for i in range(batch_size)
    ]

    images1 = torch.tensor(batch["image_0"])
    images2 = torch.tensor(batch["image_1"]) 
    # theres also image_2 and image_3 but it aint worth it

    images = [
        [images1[i], images2[i]]
        for i in range(images1.shape[0])
    ]

    batch_prompts = processor.apply_chat_template(batch_messages, add_generation_prompt=False)
    batch_inputs = processor(text=batch_prompts, images=images, padding=True, return_tensors="pt")
    batch_inputs = batch_inputs.to(device=device, non_blocking=True)

    observation = Observation(
        images=images,
        prompts=batch["language_instruction"],
        processed_inputs=batch_inputs,
        state=torch.tensor(batch["state"], device=device, dtype=torch.float32),
    )
    actions = torch.tensor(batch["actions"], device=device, dtype=torch.float32)
    return observation, actions

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

    stats_episodes = (
        episodes
        # .shuffle(5_000, seed=42, reshuffle_each_iteration=False) shuffling hangs
        .take(500)  
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
                "state": normalize_action(observation["state"]),
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