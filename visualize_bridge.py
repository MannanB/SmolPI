"""Render and play one complete episode from the local Bridge RLDS dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
import tensorflow_datasets as tfds


DEFAULT_DATASET = Path(__file__).parent / "data" / "bridge_dataset"
CANVAS_SIZE = (1280, 720)  # width, height
ACTION_LABELS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip")
STATE_LABELS = ("x", "y", "z", "roll", "pitch", "yaw", "grip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"TFDS dataset directory (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--split", default="train", help="TFDS split to read.")
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Zero-based episode index within the split.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Playback and output frame rate (Bridge was collected at 5 Hz).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("bridge_sample.mp4"),
        help="Output MP4 path (default: bridge_sample.mp4 beside this script).",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Create the MP4 without opening a playback window.",
    )
    return parser.parse_args()


def decode_text(value: object) -> str:
    value = value.numpy() if hasattr(value, "numpy") else value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_episode(
    dataset_dir: Path, split: str, episode_index: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if episode_index < 0:
        raise ValueError("episode-index must be non-negative")

    builder = tfds.builder_from_directory(dataset_dir)
    if split not in builder.info.splits:
        available = ", ".join(builder.info.splits)
        raise ValueError(f"Unknown split {split!r}; available splits: {available}")
    split_size = builder.info.splits[split].num_examples
    if episode_index >= split_size:
        raise IndexError(
            f"episode-index must be between 0 and {split_size - 1} for {split!r}"
        )

    dataset = builder.as_dataset(split=split, shuffle_files=False)
    episode = next(iter(dataset.skip(episode_index).take(1)))
    raw_metadata = episode["episode_metadata"]
    metadata: dict[str, object] = {
        key: decode_text(value)
        if value.dtype == "string"
        else value.numpy().item()
        for key, value in raw_metadata.items()
    }

    steps: list[dict[str, object]] = []
    for raw_step in episode["steps"]:
        observation = raw_step["observation"]
        images = {
            key: value.numpy()
            for key, value in observation.items()
            if key.startswith("image_")
        }
        steps.append(
            {
                "action": raw_step["action"].numpy(),
                "state": observation["state"].numpy(),
                "images": images,
                "instruction": decode_text(raw_step["language_instruction"]),
                "reward": float(raw_step["reward"].numpy()),
                "is_first": bool(raw_step["is_first"].numpy()),
                "is_last": bool(raw_step["is_last"].numpy()),
                "is_terminal": bool(raw_step["is_terminal"].numpy()),
            }
        )
    if not steps:
        raise ValueError(f"Episode {episode_index} contains no steps")
    return metadata, steps


def populated_cameras(steps: list[dict[str, object]]) -> list[str]:
    names = sorted(steps[0]["images"])
    return [
        name
        for name in names
        if any(np.any(step["images"][name]) for step in steps)
    ]


def fit_square(image: np.ndarray, size: int) -> np.ndarray:
    interpolation = cv2.INTER_AREA if image.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interpolation)


def draw_camera_row(
    canvas: np.ndarray, images: dict[str, np.ndarray], cameras: list[str]
) -> None:
    if not cameras:
        cv2.putText(
            canvas,
            "No populated camera streams",
            (420, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        return

    slot_width = CANVAS_SIZE[0] // len(cameras)
    image_size = min(420, slot_width - 12)
    for index, camera in enumerate(cameras):
        image = cv2.cvtColor(images[camera], cv2.COLOR_RGB2BGR)
        image = fit_square(image, image_size)
        x = index * slot_width + (slot_width - image_size) // 2
        y = 82 + (420 - image_size) // 2
        canvas[y : y + image_size, x : x + image_size] = image
        cv2.rectangle(canvas, (x, y), (x + image_size, y + 34), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            camera,
            (x + 10, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_vector_chart(
    canvas: np.ndarray,
    values: np.ndarray,
    labels: tuple[str, ...],
    title: str,
    left: int,
    top: int,
    width: int,
) -> None:
    cv2.putText(
        canvas,
        title,
        (left, top + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    baseline = top + 115
    chart_height = 66
    slot_width = width // len(values)
    scale = max(1.0, float(np.max(np.abs(values))))
    cv2.line(canvas, (left, baseline), (left + width, baseline), (130, 130, 130), 1)

    for index, (label, value) in enumerate(zip(labels, values)):
        center = left + index * slot_width + slot_width // 2
        bar_height = int(np.clip(value / scale, -1.0, 1.0) * chart_height)
        end_y = baseline - bar_height
        color = (235, 170, 60) if value >= 0 else (65, 120, 235)
        cv2.rectangle(canvas, (center - 18, baseline), (center + 18, end_y), color, -1)
        cv2.putText(
            canvas,
            label,
            (center - 24, top + 142),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{value:+.2f}",
            (center - 28, top + 164),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def render_frame(
    step: dict[str, object],
    cameras: list[str],
    episode_index: int,
    episode_id: object,
    step_index: int,
    total_steps: int,
) -> np.ndarray:
    canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), 20, dtype=np.uint8)
    instruction = str(step["instruction"])
    cv2.putText(
        canvas,
        f"Bridge episode {episode_index} | dataset episode ID {episode_id}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        instruction[:110] if instruction else "No language instruction",
        (20, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (185, 220, 255),
        1,
        cv2.LINE_AA,
    )
    status = f"step {step_index + 1}/{total_steps} | reward {step['reward']:.0f}"
    if step["is_terminal"]:
        status += " | terminal"
    cv2.putText(
        canvas,
        status,
        (980, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    draw_camera_row(canvas, step["images"], cameras)
    draw_vector_chart(
        canvas,
        np.asarray(step["action"]),
        ACTION_LABELS,
        "Action (7-D)",
        left=30,
        top=510,
        width=570,
    )
    draw_vector_chart(
        canvas,
        np.asarray(step["state"]),
        STATE_LABELS,
        "Robot state (7-D)",
        left=680,
        top=510,
        width=570,
    )

    progress = int(1240 * (step_index + 1) / total_steps)
    cv2.rectangle(canvas, (20, 704), (1260, 714), (65, 65, 65), -1)
    cv2.rectangle(canvas, (20, 704), (20 + progress, 714), (70, 210, 130), -1)
    return canvas


def write_video(
    steps: list[dict[str, object]],
    cameras: list[str],
    metadata: dict[str, object],
    episode_index: int,
    output: Path,
    fps: float,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, CANVAS_SIZE
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {output}")
    try:
        for index, step in enumerate(steps):
            writer.write(
                render_frame(
                    step,
                    cameras,
                    episode_index,
                    metadata["episode_id"],
                    index,
                    len(steps),
                )
            )
    finally:
        writer.release()


def play_video(video_file: Path, fps: float) -> None:
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open rendered video: {video_file}")
    window = "Bridge sample (Q/Esc to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    delay_ms = max(1, round(1000 / fps))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imshow(window, frame)
            if cv2.waitKey(delay_ms) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    metadata, steps = load_episode(dataset_dir, args.split, args.episode_index)
    cameras = populated_cameras(steps)
    output = args.output.expanduser().resolve()
    write_video(steps, cameras, metadata, args.episode_index, output, args.fps)

    instruction = str(steps[0]["instruction"]) or "<none>"
    print(f"Episode index: {args.episode_index} ({len(steps)} steps)")
    print(f"Episode ID: {metadata['episode_id']}")
    print(f"Instruction: {instruction}")
    print(f"Populated cameras: {', '.join(cameras) if cameras else 'none'}")
    print(f"Duration: {len(steps) / args.fps:.2f} seconds at {args.fps:g} FPS")
    print(f"Saved video to {output}")
    if not args.no_play:
        play_video(output, args.fps)


if __name__ == "__main__":
    main()
