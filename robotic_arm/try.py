from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")

import cv2
import mujoco
import numpy as np
import torch
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.smolpi import Observation, SmolPI, SmolPIConfig


ARM_JOINTS = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)
# Preserve the original end-effector position while pitching the tool's +X
# approach axis 45 degrees downward instead of parallel to the floor.
HOME_QPOS = np.array([0.0, -0.712953, 0.501707, 0.0, 0.996644, 0.0, 0.015, -0.015])
HOME_CTRL = np.array([0.0, -0.712953, 0.501707, 0.0, 0.996644, 0.0, 0.015])


@dataclass(frozen=True)
class TaskSpec:
    prompt: str
    output_name: str
    # Object positions make each task reproducible instead of depending on
    # whatever layout happens to be encoded in the scene file.
    object_xy: dict[str, tuple[float, float]]


TASKS = {
    "pick-red": TaskSpec(
        prompt="pick up the red box",
        output_name="smolpi_bridge_wx250s_pick_red_box.mp4",
        object_xy={
            "red_box": (0.34, 0.00),
            "blue_box": (0.34, -0.08),
            "green_box": (0.34, 0.08),
        },
    ),    
    "push-red": TaskSpec(
        prompt="push the red box",
        output_name="smolpi_bridge_wx250s_push_red_box.mp4",
        object_xy={
            "blue_box": (0.27, -0.10),
            "green_box": (0.43, -0.10),
            "red_box": (0.34, 0.10),
        },
    ),
    "push-blue-to-green": TaskSpec(
        prompt="push the blue box next to the green box",
        output_name="smolpi_bridge_wx250s_push_blue_to_green.mp4",
        # Blue and green share a clear lane along the table's x axis. The red
        # box is moved out of that lane so success requires controlled pushing,
        # not obstacle avoidance or grasping.
        object_xy={
            "blue_box": (0.27, -0.10),
            "green_box": (0.43, -0.10),
            "red_box": (0.34, 0.10),
        },
    ),
}


class ActionNormalizer:
    def __init__(self, stats_path: Path) -> None:
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing action statistics: {stats_path}")
        with np.load(stats_path) as stats:
            self.mean = np.asarray(stats["mean"], dtype=np.float32)
            self.std = np.maximum(
                np.asarray(stats["std"], dtype=np.float32), 1e-6
            )
        if self.mean.shape != (7,) or self.std.shape != (7,):
            raise ValueError(
                f"Expected 7D action statistics, got "
                f"mean={self.mean.shape}, std={self.std.shape}"
            )

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.mean) / self.std, -5.0, 5.0)

    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def rotation_matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = np.arctan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


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


class Wx250sTrial:
    def __init__(
        self,
        scene_path: Path,
        output_path: Path,
        *,
        width: int,
        height: int,
        fps: int,
        show: bool,
        task: TaskSpec,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene_path.resolve()))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.width = width
        self.height = height
        self.fps = fps
        self.show = show
        self.task = task

        self.ee_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.worktop_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "worktop")
        self.arm_joint_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
        )
        self.arm_qpos = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_dofs = self.model.jnt_dofadr[self.arm_joint_ids]
        self.arm_ranges = self.model.jnt_range[self.arm_joint_ids].copy()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (3 * width, height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")
        self.reset()

    def _id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Scene is missing required MuJoCo object {name!r}")
        return object_id

    def reset(self) -> None:
        # mj_resetData preserves the free box's XML pose. Applying the old robot
        # keyframe would reset that free joint to the world origin.
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:8] = HOME_QPOS
        self.data.ctrl[:] = HOME_CTRL
        mujoco.mj_forward(self.model, self.data)
        self.work_surface_z = (
            self.data.geom_xpos[self.worktop_id, 2]
            + self.model.geom_size[self.worktop_id, 2]
        )
        for body_name, (x, y) in self.task.object_xy.items():
            joint_id = self._id(
                mujoco.mjtObj.mjOBJ_JOINT, f"{body_name}_freejoint"
            )
            geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, f"{body_name}_geom")
            qpos_address = self.model.jnt_qposadr[joint_id]
            object_z = self.work_surface_z + self.model.geom_size[geom_id, 2]
            self.data.qpos[qpos_address : qpos_address + 3] = (x, y, object_z)
        mujoco.mj_forward(self.model, self.data)

    def render_camera(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return np.asarray(self.renderer.render()).copy()

    def render_views(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.render_camera("scene_cam"),
            self.render_camera("scene_cam_2"),
            self.render_camera("wrist_cam"),
        )

    def write_frame(
        self,
        scene_rgb: np.ndarray,
        scene_2_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
    ) -> bool:
        scene_bgr = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR)
        scene_2_bgr = cv2.cvtColor(scene_2_rgb, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(scene_bgr, "scene camera", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(scene_2_bgr, "scene camera 2", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(wrist_bgr, "wrist camera", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        composite = np.concatenate([scene_bgr, scene_2_bgr, wrist_bgr], axis=1)
        cv2.putText(
            composite,
            f'prompt: "{self.task.prompt}"   t={self.data.time:05.2f}s',
            (12, self.height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        self.writer.write(composite)
        if self.show:
            cv2.imshow("SmolPI Bridge - WX250s", composite)
            return cv2.waitKey(1) & 0xFF not in (ord("q"), 27)
        return True

    def bridge_state(self) -> np.ndarray:
        position = self.data.site_xpos[self.ee_site_id].copy()
        rotation = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        euler = rotation_matrix_to_euler_xyz(rotation)
        gripper = np.clip((self.data.qpos[6] - 0.015) / (0.037 - 0.015), 0.0, 1.0)
        return np.concatenate([position, euler, [gripper]]).astype(np.float32)

    def observation(
        self,
        scene_rgb: np.ndarray,
        scene_2_rgb: np.ndarray,
        processor,
        device: torch.device,
    ) -> Observation:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": self.task.prompt},
                ],
            }
        ]
        formatted_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        processed_inputs = processor(
            text=[formatted_prompt],
            images=[[scene_rgb, scene_2_rgb]],
            padding=True,
            return_tensors="pt",
        )
        processed_inputs = processed_inputs.to(device=device)

        return Observation(
            processed_inputs=processed_inputs,
            state=torch.from_numpy(self.bridge_state()).to(device=device).unsqueeze(0),
        )

    def solve_ik(self, action: np.ndarray) -> np.ndarray:
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel[:] = self.data.qvel
        mujoco.mj_forward(self.model, self.ik_data)

        start_position = self.ik_data.site_xpos[self.ee_site_id].copy()
        start_rotation = self.ik_data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        target_position = start_position + np.clip(action[:3], -0.04, 0.04)
        target_position = np.clip(
            target_position,
            [0.10, -0.32, self.work_surface_z + 0.005],
            [0.52, 0.32, 0.48],
        )
        target_euler = rotation_matrix_to_euler_xyz(start_rotation) + np.clip(
            action[3:6], -0.25, 0.25
        )
        target_matrix = euler_xyz_to_rotation_matrix(target_euler)
        target_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(target_quaternion, target_matrix.reshape(-1))

        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        current_quaternion = np.empty(4, dtype=np.float64)
        rotation_error = np.empty(3, dtype=np.float64)

        for _ in range(30):
            current_position = self.ik_data.site_xpos[self.ee_site_id]
            current_matrix = self.ik_data.site_xmat[self.ee_site_id]
            mujoco.mju_mat2Quat(current_quaternion, current_matrix)
            mujoco.mju_subQuat(rotation_error, target_quaternion, current_quaternion)
            error = np.concatenate([target_position - current_position, rotation_error])
            if np.linalg.norm(error[:3]) < 5e-4 and np.linalg.norm(error[3:]) < 2e-3:
                break

            mujoco.mj_jacSite(
                self.model,
                self.ik_data,
                jacobian_position,
                jacobian_rotation,
                self.ee_site_id,
            )
            jacobian = np.vstack([jacobian_position, jacobian_rotation])[:, self.arm_dofs]
            damping = 0.025
            update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(6), error
            )
            update_norm = np.linalg.norm(update)
            if update_norm > 0.18:
                update *= 0.18 / update_norm
            qpos = self.ik_data.qpos[self.arm_qpos] + update
            self.ik_data.qpos[self.arm_qpos] = np.clip(
                qpos, self.arm_ranges[:, 0], self.arm_ranges[:, 1]
            )
            mujoco.mj_forward(self.model, self.ik_data)

        return self.ik_data.qpos[self.arm_qpos].copy()

    def apply_bridge_action(self, action: np.ndarray) -> None:
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f"Invalid Bridge action: shape={action.shape}, value={action}")
        self.data.ctrl[:6] = self.solve_ik(action)
        self.data.ctrl[6] = 0.037 if action[6] >= 0.5 else 0.015

    def close(self) -> None:
        self.writer.release()
        self.renderer.close()
        if self.show:
            cv2.destroyAllWindows()


def load_policy(checkpoint_path: Path, device: torch.device) -> SmolPI:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    precision = torch.float16 if device.type == "cuda" else torch.float32
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    action_horizon = (
        int(checkpoint.get("action_horizon", 5))
        if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint
        else 5
    )
    policy = SmolPI(
        SmolPIConfig(
            action_dim=7,
            action_horizon=action_horizon,
            precision=precision,
        )
    ).to(device)
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        checkpoint = checkpoint["policy_state_dict"]
    policy.load_state_dict(checkpoint)
    return policy.eval()


def run(
    args: argparse.Namespace,
    policy: SmolPI | None = None,
    processor=None,
) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if policy is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = load_policy(args.checkpoint, device)
    else:
        device = next(policy.parameters()).device
        policy.eval()
    normalizer = ActionNormalizer(args.action_stats)
    if processor is None:
        processor = AutoProcessor.from_pretrained(policy.config.smolvlm_id)
        processor.image_processor.do_image_splitting = False
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    task = TASKS[args.task]
    output = args.output or Path("vids") / task.output_name
    trial = Wx250sTrial(
        args.scene,
        output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        show=args.show,
        task=task,
    )
    sim_steps_per_action = max(
        1, round(1.0 / args.control_hz / float(trial.model.opt.timestep))
    )
    next_frame_time = 0.0
    keep_running = True

    execution_mode = "first action only" #if args.first_action_only else "full action chunk"
    print(
        f"device={device} task={args.task!r} prompt={task.prompt!r} output={output} "
        f"execution={execution_mode}"
    )
    try:
        with torch.inference_mode():
            while trial.data.time < args.duration and keep_running:
                scene_rgb, scene_2_rgb, wrist_rgb = trial.render_views()
                observation = trial.observation(
                    scene_rgb,
                    scene_2_rgb,
                    processor,
                    device,
                )
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=policy.config.precision)
                    if device.type == "cuda"
                    else nullcontext()
                )
                # print(observation.state.shape)
                with amp_context:
                    action_chunk = policy.sample_actions(
                        device,
                        observation,
                        num_steps=args.flow_steps,
                        # noise=observation.state.unsqueeze(1)
                    )
                normalized_actions = action_chunk[0].float().cpu().numpy()
                actions = normalizer.unnormalize(normalized_actions)
                if args.first_action_only:
                    actions = actions[:1]
                print(f"t={trial.data.time:05.2f} actions={np.round(actions, 3).tolist()}")

                for action in actions:
                    trial.apply_bridge_action(action)
                    for _ in range(sim_steps_per_action):
                        mujoco.mj_step(trial.model, trial.data)
                        if trial.data.time + 1e-9 >= next_frame_time:
                            scene_rgb, scene_2_rgb, wrist_rgb = trial.render_views()
                            keep_running = trial.write_frame(
                                scene_rgb, scene_2_rgb, wrist_rgb
                            )
                            next_frame_time += 1.0 / args.fps
                        if trial.data.time >= args.duration or not keep_running:
                            break
                    if trial.data.time >= args.duration or not keep_running:
                        break
    finally:
        trial.close()
    print(f"recorded {output.resolve()}")


def record_policy(
    policy: SmolPI,
    output: Path,
    *,
    processor=None,
    action_stats: Path = Path("action_stats.npz"),
    scene: Path = Path("world/wx_scene.xml"),
    task: str = "pick-red",
) -> None:
    """Record one of the configured trials for an in-memory policy."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASKS)}")
    was_training = policy.training
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        run(
            argparse.Namespace(
                scene=scene,
                task=task,
                checkpoint=None,
                action_stats=action_stats,
                output=output,
                duration=10.0,
                control_hz=5.0,
                flow_steps=24,
                first_action_only=True,
                fps=30,
                width=512,
                height=512,
                seed=0,
                show=False,
            ),
            policy=policy,
            processor=processor,
        )
    finally:
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        policy.train(was_training)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and record SmolPI Bridge on WX250s")
    parser.add_argument("--scene", type=Path, default=Path("world/wx_scene.xml"))
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="pick-red",
        help="Manipulation task to run.",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("smolpi_bridge.pth"))
    parser.add_argument(
        "--action-stats", type=Path, default=Path("action_stats.npz")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (defaults to a task-specific filename).",
    )
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--control-hz", type=float, default=5.0)
    parser.add_argument("--flow-steps", type=int, default=14)
    parser.add_argument(
        "--first-action-only",
        action="store_true",
        help="Execute only the first action from each generated chunk, then replan.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="Show the two camera feeds while recording")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
