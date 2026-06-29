from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_DIR = Path(r"./data/bridge_dataset")

builder = tfds.builder_from_directory(DATASET_DIR)

print(builder.info)
print(builder.info.splits)
print(builder.info.features)

train_ds = builder.as_dataset(
    split="train",
    shuffle_files=True,
)

# Each top-level item is one robot episode.
for episode in train_ds.take(1):
    metadata = episode["episode_metadata"]
    steps = episode["steps"]

    print("Episode ID:", metadata["episode_id"].numpy())
    print("Original path:", metadata["file_path"].numpy().decode())
    print("Has language:", metadata["has_language"].numpy())

    for step in steps.take(3):
        print("Instruction:", step["language_instruction"].numpy().decode())
        print("Action:", step["action"].numpy())
        print("State:", step["observation"]["state"].numpy())
        print("Image shape:", step["observation"]["image_0"].shape)