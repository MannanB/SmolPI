"""Compare a Bridge sample, its WX250s replay, and SmolPI one-step predictions."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import mujoco
import numpy as np
import torch
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.smolpi import Observation, SmolPI, SmolPIConfig
from robotic_arm.simulate_sample import (
    ArmReplay,
    DEFAULT_DATASET,
    DEFAULT_SCENE,
    PANEL_SIZE,
    choose_camera,
    euler_xyz_to_rotation_matrix,
    load_episode,
)


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "smolpi_bridge.pth"


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image.copy()).to(device=device, dtype=torch.float32) / 255.0
    return ((tensor - 0.5) / 0.5).permute(2, 0, 1).contiguous().unsqueeze(0)


def load_policy(checkpoint_path: Path, device: torch.device) -> SmolPI:
    precision = torch.float16 if device.type == "cuda" else torch.float32
    policy = SmolPI(
        SmolPIConfig(action_dim=7, action_horizon=5, precision=precision)
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        checkpoint = checkpoint["policy_state_dict"]
    policy.load_state_dict(checkpoint)
    return policy.eval()


def make_observation(
    step: dict[str, object],
    processor,
    device: torch.device,
) -> Observation:
    images = {}
    image_masks = {}
    for camera in sorted(step["images"]):
        image = step["images"][camera]
        if not np.any(image):
            continue
        images[camera] = image_to_tensor(image, device)
        image_masks[camera] = torch.ones(1, dtype=torch.bool, device=device)
    if not images:
        raise ValueError("Sample step has no populated camera image")

    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": str(step["instruction"])}],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = processor.tokenizer(prompt, return_tensors="pt", truncation=True)
    return Observation(
        images=images,
        image_masks=image_masks,
        tokenized_prompt=tokens["input_ids"].to(device=device, dtype=torch.long),
        tokenized_prompt_mask=tokens["attention_mask"].to(
            device=device, dtype=torch.bool
        ),
        state=torch.from_numpy(np.asarray(step["state"], dtype=np.float32))
        .to(device)
        .unsqueeze(0),
    )


class PredictedArm(ArmReplay):
    """Render a one-step action from the ground-truth arm pose."""

    def copy_pose_from(self, replay: ArmReplay) -> None:
        self.data.qpos[:] = replay.data.qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = replay.data.ctrl
        self.commanded_arm_qpos = self.data.qpos[self.arm_qpos].copy()
        self.previous_arm_qpos = self.commanded_arm_qpos.copy()
        mujoco.mj_forward(self.model, self.data)

    def apply_prediction(self, action: np.ndarray) -> None:
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f"Invalid predicted action: {action}")
        current_position = self.data.site_xpos[self.ee_site_id].copy()
        current_rotation = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        target_position = current_position + np.clip(action[:3], -0.04, 0.04)
        delta_rotation = euler_xyz_to_rotation_matrix(
            np.clip(action[3:6], -0.25, 0.25)
        )
        target_rotation = delta_rotation @ current_rotation
        predicted_qpos = self.solve_target(
            target_position,
            target_rotation,
            regularize=True,
        )
        self.previous_arm_qpos = self.commanded_arm_qpos.copy()
        self.commanded_arm_qpos = predicted_qpos
        self.data.qpos[self.arm_qpos] = predicted_qpos
        self.data.ctrl[:6] = predicted_qpos
        gripper = 0.037 if action[6] >= 0.5 else 0.015
        self.data.qpos[6] = gripper
        self.data.qpos[7] = -gripper
        self.data.ctrl[6] = gripper
        mujoco.mj_forward(self.model, self.data)


def fit_panel(image: np.ndarray, *, rgb: bool) -> np.ndarray:
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.resize(image, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)


def draw_frame(
    sample_rgb: np.ndarray,
    replay_bgr: np.ndarray,
    prediction_bgr: np.ndarray,
    *,
    camera: str,
    instruction: str,
    recorded_action: np.ndarray,
    predicted_action: np.ndarray,
    step_index: int,
    total_steps: int,
) -> np.ndarray:
    frame = np.concatenate(
        [
            fit_panel(sample_rgb, rgb=True),
            fit_panel(replay_bgr, rgb=False),
            fit_panel(prediction_bgr, rgb=False),
        ],
        axis=1,
    )
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (3 * PANEL_SIZE, 74), (0, 0, 0), -1)
    cv2.rectangle(
        overlay,
        (0, PANEL_SIZE - 86),
        (3 * PANEL_SIZE, PANEL_SIZE),
        (0, 0, 0),
        -1,
    )
    frame = cv2.addWeighted(overlay, 0.74, frame, 0.26, 0)
    headings = (
        f"Bridge sample: {camera}",
        "Ground-truth motion replay",
        "SmolPI one-step prediction",
    )
    for index, heading in enumerate(headings):
        cv2.putText(
            frame,
            heading,
            (index * PANEL_SIZE + 14, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
    cv2.putText(
        frame,
        instruction[:140],
        (14, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (190, 225, 255),
        1,
    )
    recorded_text = "recorded:  " + " ".join(
        f"{value:+.3f}" for value in recorded_action
    )
    predicted_text = "predicted: " + " ".join(
        f"{value:+.3f}" for value in predicted_action
    )
    cv2.putText(
        frame,
        recorded_text,
        (14, PANEL_SIZE - 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (225, 225, 225),
        1,
    )
    cv2.putText(
        frame,
        predicted_text,
        (14, PANEL_SIZE - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (120, 220, 255),
        1,
    )
    cv2.putText(
        frame,
        f"step {step_index + 1}/{total_steps}",
        (3 * PANEL_SIZE - 150, PANEL_SIZE - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (100, 225, 145),
        1,
    )
    return frame


def play_video(path: Path, fps: float) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imshow("Bridge / replay / SmolPI prediction", frame)
            if cv2.waitKey(max(1, round(1000 / fps))) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def run(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("fps must be greater than zero")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    steps = load_episode(args.dataset.expanduser().resolve(), args.split, args.episode_index)
    if args.max_steps is not None:
        steps = steps[: args.max_steps]
    camera = choose_camera(steps, args.camera)
    initial_state = np.asarray(steps[0]["state"], dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint.expanduser().resolve(), device)
    processor = AutoProcessor.from_pretrained(policy.config.smolvlm_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ground_truth = ArmReplay(args.scene)
    predicted = PredictedArm(args.scene)
    ground_truth.configure_trajectory(initial_state, args.tool_pitch_degrees)
    predicted.configure_trajectory(initial_state, args.tool_pitch_degrees)
    simulation_steps = max(
        1, round(1.0 / args.fps / ground_truth.model.opt.timestep)
    )

    output = args.output
    if output is None:
        output = Path("vids") / f"bridge_wx250s_predictions_{args.episode_index}.mp4"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (3 * PANEL_SIZE, PANEL_SIZE),
    )
    if not writer.isOpened():
        ground_truth.close()
        predicted.close()
        raise RuntimeError(f"Could not open video writer for {output}")

    try:
        with torch.inference_mode():
            for index, step in enumerate(steps):
                ground_truth.track_state(
                    np.asarray(step["state"], dtype=np.float64),
                    initial_state,
                    args.motion_scale,
                    simulation_steps,
                )
                observation = make_observation(step, processor, device)
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=policy.config.precision)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with amp_context:
                    action_chunk = policy.sample_actions(
                        device,
                        observation,
                        num_steps=args.flow_steps,
                    )
                predicted_action = action_chunk[0, 0].float().cpu().numpy()
                predicted.copy_pose_from(ground_truth)
                predicted.apply_prediction(predicted_action)
                writer.write(
                    draw_frame(
                        step["images"][camera],
                        ground_truth.render(),
                        predicted.render(),
                        camera=camera,
                        instruction=str(step["instruction"]),
                        recorded_action=np.asarray(step["action"]),
                        predicted_action=predicted_action,
                        step_index=index,
                        total_steps=len(steps),
                    )
                )
                print(f"step {index + 1}/{len(steps)}", end="\r", flush=True)
    finally:
        writer.release()
        ground_truth.close()
        predicted.close()
    print()
    print(f"recorded {output}")
    if args.play:
        play_video(output, args.fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--flow-steps", type=int, default=12)
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--tool-pitch-degrees", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--play", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
