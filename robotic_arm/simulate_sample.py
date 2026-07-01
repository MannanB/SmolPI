"""Replay a Bridge sample on WX250s beside the recorded camera stream."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
import tensorflow_datasets as tfds


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "bridge_dataset"
DEFAULT_SCENE = Path(__file__).resolve().parents[1] / "world" / "wx_replay.xml"
ARM_JOINTS = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)
HOME_QPOS = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.037, -0.037])
HOME_CTRL = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.037])
PANEL_SIZE = 512


def decode_text(value: object) -> str:
    value = value.numpy() if hasattr(value, "numpy") else value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_episode(dataset_dir: Path, split: str, episode_index: int) -> list[dict[str, object]]:
    if episode_index < 0:
        raise ValueError("episode-index must be non-negative")
    builder = tfds.builder_from_directory(dataset_dir)
    if split not in builder.info.splits:
        raise ValueError(f"Unknown split {split!r}; available: {', '.join(builder.info.splits)}")
    split_size = builder.info.splits[split].num_examples
    if episode_index >= split_size:
        raise IndexError(f"episode-index must be less than {split_size}")

    dataset = builder.as_dataset(split=split, shuffle_files=False)
    episode = next(iter(dataset.skip(episode_index).take(1)))
    steps = []
    for raw_step in episode["steps"]:
        observation = raw_step["observation"]
        steps.append(
            {
                "action": raw_step["action"].numpy(),
                "state": observation["state"].numpy(),
                "images": {
                    key: value.numpy()
                    for key, value in observation.items()
                    if key.startswith("image_")
                },
                "instruction": decode_text(raw_step["language_instruction"]),
            }
        )
    if not steps:
        raise ValueError(f"Episode {episode_index} contains no steps")
    return steps


def choose_camera(steps: list[dict[str, object]], requested: str) -> str:
    cameras = sorted(steps[0]["images"])
    populated = [
        camera
        for camera in cameras
        if any(np.any(step["images"][camera]) for step in steps)
    ]
    if not populated:
        raise ValueError("Episode has no populated camera streams")
    if requested == "auto":
        return "image_0" if "image_0" in populated else populated[0]
    if requested not in populated:
        raise ValueError(f"Camera {requested!r} is unavailable; populated: {', '.join(populated)}")
    return requested


def euler_xyz_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def scale_rotation(rotation: np.ndarray, scale: float) -> np.ndarray:
    """Scale a relative rotation without introducing Euler-angle artifacts."""
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    if quaternion[0] < 0:
        quaternion *= -1
    vector_norm = np.linalg.norm(quaternion[1:])
    if vector_norm < 1e-10:
        return np.eye(3)
    angle = 2.0 * np.arctan2(vector_norm, quaternion[0]) * scale
    axis = quaternion[1:] / vector_norm
    scaled_quaternion = np.concatenate([[np.cos(angle / 2.0)], axis * np.sin(angle / 2.0)])
    result = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(result, scaled_quaternion)
    return result.reshape(3, 3)


class ArmReplay:
    def __init__(self, scene_path: Path) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene_path.resolve()))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=PANEL_SIZE, height=PANEL_SIZE)
        self.ee_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        joint_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
        )
        self.arm_qpos = self.model.jnt_qposadr[joint_ids]
        self.arm_ranges = self.model.jnt_range[joint_ids].copy()
        self.commanded_arm_qpos = HOME_QPOS[:6].copy()
        self.previous_arm_qpos = HOME_QPOS[:6].copy()
        self.reset()

    def _id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Replay scene is missing {name!r}")
        return object_id

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = HOME_QPOS
        self.data.ctrl[:] = HOME_CTRL
        mujoco.mj_forward(self.model, self.data)
        self.commanded_arm_qpos = HOME_QPOS[:6].copy()
        self.previous_arm_qpos = HOME_QPOS[:6].copy()
        self.home_position = self.data.site_xpos[self.ee_site_id].copy()
        self.source_initial_rotation = np.eye(3)
        self.replay_initial_rotation = np.eye(3)

    def configure_trajectory(
        self, initial_state: np.ndarray, tool_pitch_degrees: float
    ) -> None:
        self.source_initial_rotation = euler_xyz_to_rotation_matrix(initial_state[3:6])
        # Bridge's tool frame has a fixed offset: near-zero recorded rotation
        # corresponds to a gripper approaching downward. WX250s local +X is its
        # approach axis, so +90 degrees about world Y points that axis at -Z.
        self.replay_initial_rotation = euler_xyz_to_rotation_matrix(
            np.array([0.0, np.deg2rad(tool_pitch_degrees), 0.0])
        )
        arm_qpos = self.solve_target(
            self.home_position,
            self.replay_initial_rotation,
            regularize=False,
        )
        self.commanded_arm_qpos = arm_qpos
        self.previous_arm_qpos = arm_qpos.copy()
        self.data.qpos[self.arm_qpos] = arm_qpos
        self.data.ctrl[:6] = arm_qpos

        normalized_position = float(initial_state[6])
        position = 0.015 + np.clip(normalized_position, 0.0, 1.0) * (0.037 - 0.015)
        self.data.qpos[6] = position
        self.data.qpos[7] = -position
        self.data.ctrl[6] = position
        mujoco.mj_forward(self.model, self.data)

    def solve_target(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        regularize: bool,
    ) -> np.ndarray:
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel[:] = self.data.qvel
        # Seed from the previous command, not the lagging physics state. This
        # keeps redundant wrist solutions on the same kinematic branch.
        self.ik_data.qpos[self.arm_qpos] = self.commanded_arm_qpos
        mujoco.mj_forward(self.model, self.ik_data)
        seed_qpos = self.commanded_arm_qpos.copy()

        target_position = np.clip(target_position, [0.08, -0.35, 0.01], [0.54, 0.35, 0.50])
        target_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(target_quaternion, target_rotation.reshape(-1))

        def residual(qpos: np.ndarray) -> np.ndarray:
            self.ik_data.qpos[self.arm_qpos] = qpos
            mujoco.mj_forward(self.model, self.ik_data)
            current_quaternion = np.empty(4, dtype=np.float64)
            rotation_error = np.empty(3, dtype=np.float64)
            mujoco.mju_mat2Quat(
                current_quaternion, self.ik_data.site_xmat[self.ee_site_id]
            )
            mujoco.mju_subQuat(rotation_error, target_quaternion, current_quaternion)
            terms = [
                20.0 * (target_position - self.ik_data.site_xpos[self.ee_site_id]),
                2.0 * rotation_error,
            ]
            if regularize:
                terms.extend(
                    [
                        0.8 * (qpos - seed_qpos),
                        0.8
                        * (qpos - 2.0 * seed_qpos + self.previous_arm_qpos),
                    ]
                )
            return np.concatenate(terms)

        result = least_squares(
            residual,
            seed_qpos,
            bounds=(self.arm_ranges[:, 0], self.arm_ranges[:, 1]),
            max_nfev=100,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        return result.x

    def track_state(
        self,
        state: np.ndarray,
        initial_state: np.ndarray,
        motion_scale: float,
        simulation_steps: int,
    ) -> None:
        position_delta = (state[:3] - initial_state[:3]) * motion_scale
        source_rotation = euler_xyz_to_rotation_matrix(state[3:6])
        relative_rotation = source_rotation @ self.source_initial_rotation.T
        relative_rotation = scale_rotation(relative_rotation, motion_scale)
        target_rotation = relative_rotation @ self.replay_initial_rotation
        next_qpos = self.solve_target(
            self.home_position + position_delta,
            target_rotation,
            regularize=True,
        )
        self.previous_arm_qpos = self.commanded_arm_qpos.copy()
        self.commanded_arm_qpos = next_qpos
        self.data.ctrl[:6] = self.commanded_arm_qpos
        self.data.ctrl[6] = 0.015 + np.clip(state[6], 0.0, 1.0) * (0.037 - 0.015)
        for _ in range(simulation_steps):
            mujoco.mj_step(self.model, self.data)

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="replay_cam")
        return cv2.cvtColor(np.asarray(self.renderer.render()), cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self.renderer.close()


def make_frame(
    bridge_rgb: np.ndarray,
    simulation_bgr: np.ndarray,
    *,
    camera: str,
    instruction: str,
    action: np.ndarray,
    step_index: int,
    total_steps: int,
) -> np.ndarray:
    bridge_bgr = cv2.cvtColor(bridge_rgb, cv2.COLOR_RGB2BGR)
    bridge_bgr = cv2.resize(bridge_bgr, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)
    frame = np.concatenate([bridge_bgr, simulation_bgr], axis=1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (2 * PANEL_SIZE, 70), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, PANEL_SIZE - 58), (2 * PANEL_SIZE, PANEL_SIZE), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    cv2.putText(frame, f"Bridge: {camera}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    cv2.putText(frame, "MuJoCo: WX250s motion replay", (PANEL_SIZE + 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    cv2.putText(frame, instruction[:115], (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 225, 255), 1)
    action_text = "action: " + " ".join(f"{value:+.3f}" for value in action)
    cv2.putText(frame, action_text, (14, PANEL_SIZE - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (230, 230, 230), 1)
    cv2.putText(frame, f"step {step_index + 1}/{total_steps}", (2 * PANEL_SIZE - 145, PANEL_SIZE - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 220, 140), 1)
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
            cv2.imshow("Bridge / WX250s sample replay", frame)
            if cv2.waitKey(max(1, round(1000 / fps))) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def run(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("fps must be greater than zero")
    dataset_dir = args.dataset.expanduser().resolve()
    steps = load_episode(dataset_dir, args.split, args.episode_index)
    camera = choose_camera(steps, args.camera)
    initial_state = np.asarray(steps[0]["state"], dtype=np.float64)

    output = args.output
    if output is None:
        output = Path("vids") / f"bridge_wx250s_sample_{args.episode_index}.mp4"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (2 * PANEL_SIZE, PANEL_SIZE),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {output}")

    replay = ArmReplay(args.scene)
    replay.configure_trajectory(initial_state, args.tool_pitch_degrees)
    simulation_steps = max(1, round(1.0 / args.fps / replay.model.opt.timestep))
    try:
        for index, step in enumerate(steps):
            replay.track_state(
                np.asarray(step["state"], dtype=np.float64),
                initial_state,
                args.motion_scale,
                simulation_steps,
            )
            writer.write(
                make_frame(
                    step["images"][camera],
                    replay.render(),
                    camera=camera,
                    instruction=str(step["instruction"]),
                    action=np.asarray(step["action"]),
                    step_index=index,
                    total_steps=len(steps),
                )
            )
    finally:
        writer.release()
        replay.close()

    print(f"episode={args.episode_index} steps={len(steps)} camera={camera}")
    print(f"instruction={steps[0]['instruction']!r}")
    print(f"recorded {output}")
    if args.play:
        play_video(output, args.fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--camera", default="auto", help="Bridge camera name or 'auto'")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument(
        "--tool-pitch-degrees",
        type=float,
        default=80.0,
        help="Initial WX250s tool pitch; 80 points down without the 90-degree singularity.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--play", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
