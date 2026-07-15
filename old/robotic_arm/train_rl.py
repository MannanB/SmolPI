from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
import sys
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import cv2
import numpy as np
import torch
import tqdm
from torch import nn
from transformers import AutoProcessor
from transformers.tokenization_utils_base import BatchEncoding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.smolpi import Observation, SmolPI, SmolPIConfig


trial_module = importlib.import_module("robotic_arm.try")
ActionNormalizer = trial_module.ActionNormalizer
TASKS = trial_module.TASKS
Wx250sTrial = trial_module.Wx250sTrial


class ArmEnvView:
    def __init__(self, env: BatchedWx250sArm, env_idx: int) -> None:
        self.env = env
        self.env_idx = env_idx
        self.model = env.model
        self.ee_site_id = env.ee_site_id

    @property
    def data(self):
        return self.env.datas[self.env_idx]


def body_position(trial: Wx250sTrial, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(trial.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Scene is missing body {body_name!r}")
    return trial.data.xpos[body_id].copy()


class ArmReward:
    def __init__(self, trial: Wx250sTrial) -> None:
        self.previous_score = self.score(trial)

    def score(self, trial: Wx250sTrial) -> float:
        raise NotImplementedError

    def update(self, trial: Wx250sTrial) -> tuple[float, bool]:
        score = self.score(trial)
        reward = score - self.previous_score
        self.previous_score = score
        return reward, self.done(trial)

    def done(self, trial: Wx250sTrial) -> bool:
        return False

    def metrics(self, trial: Wx250sTrial) -> dict[str, float]:
        return {
            "score": float(self.score(trial)),
            "success": float(self.done(trial)),
        }


class PickRedReward(ArmReward):
    def __init__(self, trial: Wx250sTrial) -> None:
        self.initial_z = float(body_position(trial, "red_box")[2])
        super().__init__(trial)

    def score(self, trial: Wx250sTrial) -> float:
        red = body_position(trial, "red_box")
        ee = trial.data.site_xpos[trial.ee_site_id]
        lift = max(0.0, float(red[2] - self.initial_z))
        approach = -float(np.linalg.norm(red - ee))
        bonus = 1.0 if lift > 0.06 else 0.0
        return 20.0 * lift + approach + bonus

    def done(self, trial: Wx250sTrial) -> bool:
        return float(body_position(trial, "red_box")[2] - self.initial_z) > 0.08

    def metrics(self, trial: Wx250sTrial) -> dict[str, float]:
        red = body_position(trial, "red_box")
        ee = trial.data.site_xpos[trial.ee_site_id]
        lift = max(0.0, float(red[2] - self.initial_z))
        return {
            "score": float(self.score(trial)),
            "success": float(self.done(trial)),
            "red_lift": lift,
            "ee_red_dist": float(np.linalg.norm(red - ee)),
            "red_z": float(red[2]),
        }


class PushRedReward(ArmReward):
    def __init__(self, trial: Wx250sTrial) -> None:
        self.initial_xy = body_position(trial, "red_box")[:2]
        super().__init__(trial)

    def score(self, trial: Wx250sTrial) -> float:
        xy = body_position(trial, "red_box")[:2]
        progress_x = float(xy[0] - self.initial_xy[0])
        lateral_drift = abs(float(xy[1] - self.initial_xy[1]))
        return 5.0 * progress_x - 0.5 * lateral_drift

    def done(self, trial: Wx250sTrial) -> bool:
        return float(body_position(trial, "red_box")[0] - self.initial_xy[0]) > 0.12

    def metrics(self, trial: Wx250sTrial) -> dict[str, float]:
        xy = body_position(trial, "red_box")[:2]
        progress_x = float(xy[0] - self.initial_xy[0])
        lateral_drift = abs(float(xy[1] - self.initial_xy[1]))
        return {
            "score": float(self.score(trial)),
            "success": float(self.done(trial)),
            "red_push_x": progress_x,
            "red_lateral_drift": lateral_drift,
        }


class PushBlueToGreenReward(ArmReward):
    def score(self, trial: Wx250sTrial) -> float:
        blue = body_position(trial, "blue_box")[:2]
        green = body_position(trial, "green_box")[:2]
        return -4.0 * float(np.linalg.norm(blue - green))

    def done(self, trial: Wx250sTrial) -> bool:
        blue = body_position(trial, "blue_box")[:2]
        green = body_position(trial, "green_box")[:2]
        return float(np.linalg.norm(blue - green)) < 0.07

    def metrics(self, trial: Wx250sTrial) -> dict[str, float]:
        blue = body_position(trial, "blue_box")[:2]
        green = body_position(trial, "green_box")[:2]
        distance = float(np.linalg.norm(blue - green))
        return {
            "score": float(self.score(trial)),
            "success": float(self.done(trial)),
            "blue_green_dist": distance,
        }


REWARD_BY_TASK = {
    "pick-red": PickRedReward,
    "push-red": PushRedReward,
    "push-blue-to-green": PushBlueToGreenReward,
}


@dataclass
class Transition:
    processed_inputs: dict[str, torch.Tensor]
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    done: bool
    value: float
    advantage: float = 0.0
    ret: float = 0.0


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state.float()).squeeze(-1)


def parse_precision(value: str) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported precision {value!r}")


def detach_observation(observation: Observation) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    processed_inputs = {
        key: value.detach().cpu()
        for key, value in observation.processed_inputs.items()
        if torch.is_tensor(value)
    }
    state = observation.state.detach().cpu()
    return processed_inputs, state


def batch_observations(transitions: list[Transition], device: torch.device) -> Observation:
    keys = transitions[0].processed_inputs.keys()
    processed_inputs = BatchEncoding(
        {
            key: torch.cat([transition.processed_inputs[key] for transition in transitions], dim=0).to(device)
            for key in keys
        }
    )
    state = torch.cat([transition.state for transition in transitions], dim=0).to(device)
    return Observation(processed_inputs=processed_inputs, state=state)


def observation_row(observation: Observation, row_idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    processed_inputs = {
        key: value[row_idx : row_idx + 1].detach().cpu()
        for key, value in observation.processed_inputs.items()
        if torch.is_tensor(value)
    }
    state = observation.state[row_idx : row_idx + 1].detach().cpu()
    return processed_inputs, state


def compute_gae(
    transitions: list[Transition],
    *,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> None:
    next_value = last_value
    next_advantage = 0.0
    for transition in reversed(transitions):
        nonterminal = 0.0 if transition.done else 1.0
        delta = transition.reward + gamma * next_value * nonterminal - transition.value
        transition.advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        transition.ret = transition.advantage + transition.value
        next_value = transition.value
        next_advantage = transition.advantage


def load_policy(args: argparse.Namespace, device: torch.device) -> SmolPI:
    policy = SmolPI(
        SmolPIConfig(
            action_dim=7,
            action_horizon=1,
            precision=parse_precision(args.precision),
        )
    ).to(device)
    if args.checkpoint is not None and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
            checkpoint = checkpoint["policy_state_dict"]
        policy.load_state_dict(checkpoint)
        print(f"Loaded policy checkpoint: {args.checkpoint}")
    return policy


def make_amp_context(device: torch.device, precision: torch.dtype):
    if device.type != "cpu" and precision in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type=device.type, dtype=precision)
    return nullcontext()


def format_metrics(metrics: dict[str, float | int]) -> dict[str, str]:
    formatted = {
        "return": f"{metrics['mean_return']:.3f}",
        "raw": f"{metrics['mean_raw_return']:.4f}",
        "roll": f"{metrics['rolling_return']:.3f}",
        "len": f"{metrics['mean_length']:.1f}",
        "ploss": f"{metrics['policy_loss']:.3f}",
        "vloss": f"{metrics['value_loss']:.3f}",
        "vstd": f"{metrics['return_target_std']:.2f}",
        "w": f"{metrics['weight_mean']:.2f}",
        "aw": f"{metrics['active_weight_fraction']:.2f}",
    }
    if "train_success" in metrics:
        formatted["succ"] = f"{metrics['train_success']:.2f}"
    if "eval_mean_return" in metrics:
        formatted["eval"] = f"{metrics['eval_mean_return']:.3f}"
    return formatted


def append_jsonl(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(row, sort_keys=True) + "\n")


def flush_metrics(path: Path, metrics: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as metrics_file:
        pickle.dump(metrics, metrics_file)


def mean_metric_dict(prefix: str, rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        f"{prefix}_{key}": float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
    }


class BatchedWx250sArm:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        task,
        num_envs: int,
        record_path: Path | None,
    ) -> None:
        self.args = args
        self.task = task
        self.num_envs = num_envs
        self.model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.ik_datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.renderer = mujoco.Renderer(self.model, width=args.width, height=args.height)
        self.width = args.width
        self.height = args.height
        self.fps = args.fps

        self.ee_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.worktop_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "worktop")
        self.arm_joint_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in trial_module.ARM_JOINTS]
        )
        self.arm_qpos = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_dofs = self.model.jnt_dofadr[self.arm_joint_ids]
        self.arm_ranges = self.model.jnt_range[self.arm_joint_ids].copy()
        self.views = [ArmEnvView(self, env_idx) for env_idx in range(num_envs)]

        self.writer = None
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                str(record_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (3 * args.width, args.height),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {record_path}")

        for env_idx in range(num_envs):
            self.reset(env_idx)

    def _id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Scene is missing required MuJoCo object {name!r}")
        return object_id

    def reset(self, env_idx: int) -> None:
        data = self.datas[env_idx]
        mujoco.mj_resetData(self.model, data)
        data.qpos[:8] = trial_module.HOME_QPOS
        data.ctrl[:] = trial_module.HOME_CTRL
        mujoco.mj_forward(self.model, data)
        work_surface_z = (
            data.geom_xpos[self.worktop_id, 2]
            + self.model.geom_size[self.worktop_id, 2]
        )
        for body_name, (x, y) in self.task.object_xy.items():
            joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"{body_name}_freejoint")
            geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, f"{body_name}_geom")
            qpos_address = self.model.jnt_qposadr[joint_id]
            object_z = work_surface_z + self.model.geom_size[geom_id, 2]
            data.qpos[qpos_address : qpos_address + 3] = (x, y, object_z)
        mujoco.mj_forward(self.model, data)

    def render_camera(self, env_idx: int, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.datas[env_idx], camera=camera)
        return np.asarray(self.renderer.render()).copy()

    def render_views(self, env_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.render_camera(env_idx, "scene_cam"),
            self.render_camera(env_idx, "scene_cam_2"),
            self.render_camera(env_idx, "wrist_cam"),
        )

    def write_frame(self, scene_rgb: np.ndarray, scene_2_rgb: np.ndarray, wrist_rgb: np.ndarray, time: float) -> None:
        if self.writer is None:
            return
        scene_bgr = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR)
        scene_2_bgr = cv2.cvtColor(scene_2_rgb, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(scene_bgr, "scene camera", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(scene_2_bgr, "scene camera 2", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(wrist_bgr, "wrist camera", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        composite = np.concatenate([scene_bgr, scene_2_bgr, wrist_bgr], axis=1)
        cv2.putText(
            composite,
            f'prompt: "{self.task.prompt}"   t={time:05.2f}s',
            (12, self.height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        self.writer.write(composite)

    def bridge_state(self, env_idx: int) -> np.ndarray:
        data = self.datas[env_idx]
        position = data.site_xpos[self.ee_site_id].copy()
        rotation = data.site_xmat[self.ee_site_id].reshape(3, 3)
        euler = trial_module.rotation_matrix_to_euler_xyz(rotation)
        gripper = np.clip((data.qpos[6] - 0.015) / (0.037 - 0.015), 0.0, 1.0)
        return np.concatenate([position, euler, [gripper]]).astype(np.float32)

    def observation(self, env_indices: list[int], processor, device: torch.device) -> Observation:
        formatted_prompts = []
        image_groups = []
        states = []
        for env_idx in env_indices:
            scene_rgb = self.render_camera(env_idx, "scene_cam")
            scene_2_rgb = self.render_camera(env_idx, "scene_cam_2")
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
            formatted_prompts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            image_groups.append([scene_rgb, scene_2_rgb])
            states.append(self.bridge_state(env_idx))

        processed_inputs = processor(
            text=formatted_prompts,
            images=image_groups,
            padding=True,
            return_tensors="pt",
        )
        processed_inputs = processed_inputs.to(device=device)
        return Observation(
            processed_inputs=processed_inputs,
            state=torch.from_numpy(np.stack(states)).to(device=device),
        )

    def solve_ik(self, env_idx: int, action: np.ndarray) -> np.ndarray:
        data = self.datas[env_idx]
        ik_data = self.ik_datas[env_idx]
        ik_data.qpos[:] = data.qpos
        ik_data.qvel[:] = data.qvel
        mujoco.mj_forward(self.model, ik_data)

        work_surface_z = (
            data.geom_xpos[self.worktop_id, 2]
            + self.model.geom_size[self.worktop_id, 2]
        )
        start_position = ik_data.site_xpos[self.ee_site_id].copy()
        start_rotation = ik_data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        target_position = start_position + np.clip(action[:3], -0.04, 0.04)
        target_position = np.clip(
            target_position,
            [0.10, -0.32, work_surface_z + 0.005],
            [0.52, 0.32, 0.48],
        )
        target_euler = trial_module.rotation_matrix_to_euler_xyz(start_rotation) + np.clip(
            action[3:6], -0.25, 0.25
        )
        target_matrix = trial_module.euler_xyz_to_rotation_matrix(target_euler)
        target_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(target_quaternion, target_matrix.reshape(-1))

        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        current_quaternion = np.empty(4, dtype=np.float64)
        rotation_error = np.empty(3, dtype=np.float64)

        for _ in range(30):
            current_position = ik_data.site_xpos[self.ee_site_id]
            current_matrix = ik_data.site_xmat[self.ee_site_id]
            mujoco.mju_mat2Quat(current_quaternion, current_matrix)
            mujoco.mju_subQuat(rotation_error, target_quaternion, current_quaternion)
            error = np.concatenate([target_position - current_position, rotation_error])
            if np.linalg.norm(error[:3]) < 5e-4 and np.linalg.norm(error[3:]) < 2e-3:
                break

            mujoco.mj_jacSite(
                self.model,
                ik_data,
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
            qpos = ik_data.qpos[self.arm_qpos] + update
            ik_data.qpos[self.arm_qpos] = np.clip(
                qpos, self.arm_ranges[:, 0], self.arm_ranges[:, 1]
            )
            mujoco.mj_forward(self.model, ik_data)

        return ik_data.qpos[self.arm_qpos].copy()

    def apply_bridge_action(self, env_idx: int, action: np.ndarray) -> None:
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f"Invalid Bridge action: shape={action.shape}, value={action}")
        data = self.datas[env_idx]
        data.ctrl[:6] = self.solve_ik(env_idx, action)
        data.ctrl[6] = 0.037 if action[6] >= 0.5 else 0.015

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.renderer.close()


def collect_episodes_batched(
    *,
    args: argparse.Namespace,
    policy: SmolPI,
    value_net: ValueNet,
    processor,
    normalizer: ActionNormalizer,
    device: torch.device,
    num_episodes: int,
    update: int,
    exploration_std: float,
    store_transitions: bool,
    phase: str,
    record_path: Path | None,
) -> tuple[list[Transition], list[float], list[float], list[int], list[dict[str, float]]]:
    task = TASKS[args.task]
    all_transitions: list[Transition] = []
    all_returns: list[float] = []
    all_raw_returns: list[float] = []
    all_lengths: list[int] = []
    all_metrics: list[dict[str, float]] = []
    sim_steps_per_action = None
    expected_actions = max(1, math.ceil(args.duration * args.control_hz))
    remaining = num_episodes
    episode_offset = 0

    progress_bar = None
    if args.progress:
        progress_bar = tqdm.tqdm(
            total=num_episodes * expected_actions,
            desc=f"{phase} u{update:04d}",
            unit="act",
            leave=False,
        )

    policy.eval()
    value_net.eval()
    try:
        while remaining > 0:
            num_envs = min(args.parallel_envs, remaining)
            wave_record_path = record_path if episode_offset == 0 else None
            env = BatchedWx250sArm(
                args,
                task=task,
                num_envs=num_envs,
                record_path=wave_record_path,
            )
            if sim_steps_per_action is None:
                sim_steps_per_action = max(
                    1,
                    round(1.0 / args.control_hz / float(env.model.opt.timestep)),
                )
            reward_models = [REWARD_BY_TASK[args.task](env.views[env_idx]) for env_idx in range(num_envs)]
            transitions_by_env: list[list[Transition]] = [[] for _ in range(num_envs)]
            returns = [0.0 for _ in range(num_envs)]
            raw_returns = [0.0 for _ in range(num_envs)]
            lengths = [0 for _ in range(num_envs)]
            active = [True for _ in range(num_envs)]
            next_frame_time = 0.0

            try:
                while any(active):
                    active_indices = [
                        env_idx
                        for env_idx, is_active in enumerate(active)
                        if is_active and env.datas[env_idx].time < args.duration
                    ]
                    if not active_indices:
                        break

                    observation = env.observation(active_indices, processor, device)
                    with torch.no_grad(), make_amp_context(device, policy.config.precision):
                        base_actions = policy.sample_actions(
                            device,
                            observation,
                            num_steps=args.flow_steps,
                        )[:, 0].float()
                        values = value_net(observation.state).float()

                    noise = torch.randn_like(base_actions) * exploration_std
                    normalized_actions = torch.clamp(base_actions + noise, -5.0, 5.0)
                    bridge_actions = normalizer.unnormalize(normalized_actions.cpu().numpy())

                    stored_rows: dict[int, tuple[dict[str, torch.Tensor], torch.Tensor]] = {}
                    if store_transitions:
                        for row_idx, env_idx in enumerate(active_indices):
                            stored_rows[env_idx] = observation_row(observation, row_idx)

                    for row_idx, env_idx in enumerate(active_indices):
                        env.apply_bridge_action(env_idx, bridge_actions[row_idx])

                    for _ in range(sim_steps_per_action):
                        for env_idx in active_indices:
                            if active[env_idx]:
                                mujoco.mj_step(env.model, env.datas[env_idx])
                        if wave_record_path is not None and active[0] and env.datas[0].time + 1e-9 >= next_frame_time:
                            scene_rgb, scene_2_rgb, wrist_rgb = env.render_views(0)
                            env.write_frame(scene_rgb, scene_2_rgb, wrist_rgb, env.datas[0].time)
                            next_frame_time += 1.0 / args.fps
                        if all(env.datas[env_idx].time >= args.duration for env_idx in active_indices):
                            break

                    for row_idx, env_idx in enumerate(active_indices):
                        if not active[env_idx]:
                            continue
                        raw_reward, done = reward_models[env_idx].update(env.views[env_idx])
                        reward = args.reward_scale * raw_reward
                        raw_returns[env_idx] += raw_reward
                        returns[env_idx] += reward
                        lengths[env_idx] += 1
                        is_terminal = done or env.datas[env_idx].time >= args.duration
                        if store_transitions:
                            processed_inputs, state = stored_rows[env_idx]
                            transitions_by_env[env_idx].append(
                                Transition(
                                    processed_inputs=processed_inputs,
                                    state=state,
                                    action=normalized_actions[row_idx].detach().cpu().view(1, 1, -1),
                                    reward=float(reward),
                                    done=bool(is_terminal),
                                    value=float(values[row_idx].detach().cpu()),
                                )
                            )
                        active[env_idx] = not is_terminal

                    if progress_bar is not None:
                        progress_bar.update(len(active_indices))
                        progress_bar.set_postfix(
                            {
                                "return": f"{np.mean(returns):.3f}",
                                "raw": f"{np.mean(raw_returns):.4f}",
                                "active": sum(active),
                            }
                        )

                for env_idx in range(num_envs):
                    if store_transitions:
                        compute_gae(
                            transitions_by_env[env_idx],
                            last_value=0.0,
                            gamma=args.gamma,
                            gae_lambda=args.gae_lambda,
                        )
                        all_transitions.extend(transitions_by_env[env_idx])
                    episode_metrics = reward_models[env_idx].metrics(env.views[env_idx])
                    episode_metrics.update(
                        {
                            "return": returns[env_idx],
                            "raw_return": raw_returns[env_idx],
                            "length": float(lengths[env_idx]),
                            "exploration_std": exploration_std,
                        }
                    )
                    all_returns.append(returns[env_idx])
                    all_raw_returns.append(raw_returns[env_idx])
                    all_lengths.append(lengths[env_idx])
                    all_metrics.append(episode_metrics)
            finally:
                env.close()

            remaining -= num_envs
            episode_offset += num_envs
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return all_transitions, all_returns, all_raw_returns, all_lengths, all_metrics


def collect_episode(
    *,
    args: argparse.Namespace,
    policy: SmolPI,
    value_net: ValueNet,
    processor,
    normalizer: ActionNormalizer,
    device: torch.device,
    record_path: Path | None,
    update: int,
    episode_idx: int,
    exploration_std: float,
    store_transitions: bool,
    phase: str,
) -> tuple[list[Transition], float, float, int, dict[str, float]]:
    task = TASKS[args.task]
    output_path = record_path or (Path(args.video_dir) / "_latest_unrecorded.mp4")
    trial = Wx250sTrial(
        args.scene,
        output_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        show=False,
        task=task,
    )
    reward_model = REWARD_BY_TASK[args.task](trial)
    sim_steps_per_action = max(1, round(1.0 / args.control_hz / float(trial.model.opt.timestep)))
    next_frame_time = 0.0
    transitions: list[Transition] = []
    episode_return = 0.0
    raw_episode_return = 0.0
    done = False
    expected_actions = max(1, math.ceil(args.duration * args.control_hz))
    step_bar = None
    if args.progress and args.step_progress:
        step_bar = tqdm.tqdm(
            total=expected_actions,
            desc=f"{phase} u{update:04d} ep{episode_idx + 1:02d} movements",
            unit="act",
            leave=False,
        )

    policy.eval()
    value_net.eval()
    try:
        while trial.data.time < args.duration and not done:
            scene_rgb, scene_2_rgb, _ = trial.render_views()
            observation = trial.observation(scene_rgb, scene_2_rgb, processor, device)

            with torch.no_grad(), make_amp_context(device, policy.config.precision):
                base_action = policy.sample_actions(
                    device,
                    observation,
                    num_steps=args.flow_steps,
                )[0, 0].float()
                value = float(value_net(observation.state).item())

            noise = torch.randn_like(base_action) * exploration_std
            normalized_action = torch.clamp(base_action + noise, -5.0, 5.0)
            bridge_action = normalizer.unnormalize(normalized_action.cpu().numpy())

            if store_transitions:
                processed_inputs, state = detach_observation(observation)
            trial.apply_bridge_action(bridge_action)
            for _ in range(sim_steps_per_action):
                mujoco.mj_step(trial.model, trial.data)
                if record_path is not None and trial.data.time + 1e-9 >= next_frame_time:
                    scene_rgb, scene_2_rgb, wrist_rgb = trial.render_views()
                    trial.write_frame(scene_rgb, scene_2_rgb, wrist_rgb)
                    next_frame_time += 1.0 / args.fps
                if trial.data.time >= args.duration:
                    break

            raw_reward, done = reward_model.update(trial)
            reward = args.reward_scale * raw_reward
            raw_episode_return += raw_reward
            episode_return += reward
            if step_bar is not None:
                step_bar.update(1)
                step_bar.set_postfix(
                    {
                        "return": f"{episode_return:.3f}",
                        "raw": f"{raw_episode_return:.4f}",
                        "reward": f"{reward:.3f}",
                        "t": f"{trial.data.time:.2f}",
                    }
                )
            if store_transitions:
                transitions.append(
                    Transition(
                        processed_inputs=processed_inputs,
                        state=state,
                        action=normalized_action.detach().cpu().view(1, 1, -1),
                        reward=float(reward),
                        done=done,
                        value=value,
                    )
                )

        last_value = 0.0
        if store_transitions and transitions and not transitions[-1].done and trial.data.time < args.duration:
            scene_rgb, scene_2_rgb, _ = trial.render_views()
            observation = trial.observation(scene_rgb, scene_2_rgb, processor, device)
            with torch.no_grad():
                last_value = float(value_net(observation.state).item())

        if store_transitions:
            compute_gae(
                transitions,
                last_value=last_value,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
        episode_length = len(transitions) if store_transitions else round(trial.data.time * args.control_hz)
        episode_metrics = reward_model.metrics(trial)
        episode_metrics.update(
            {
                "return": episode_return,
                "raw_return": raw_episode_return,
                "length": float(episode_length),
                "exploration_std": exploration_std,
            }
        )
        return transitions, episode_return, raw_episode_return, episode_length, episode_metrics
    finally:
        if step_bar is not None:
            step_bar.close()
        trial.close()


def train_on_rollouts(
    *,
    args: argparse.Namespace,
    policy: SmolPI,
    value_net: ValueNet,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    device: torch.device,
) -> dict[str, float]:
    policy.train()
    value_net.train()

    raw_advantages = torch.tensor([transition.advantage for transition in transitions], dtype=torch.float32)
    raw_advantage_mean = raw_advantages.mean()
    raw_advantage_std = raw_advantages.std(unbiased=False).clamp_min(1e-6)
    advantages = (raw_advantages - raw_advantage_mean) / raw_advantage_std
    weights = torch.exp(advantages / args.awr_temperature).clamp(max=args.max_awr_weight)
    if args.advantage_filter == "positive":
        weights = torch.where(advantages > 0.0, weights, torch.zeros_like(weights))
    elif args.advantage_filter == "top_quantile":
        threshold = torch.quantile(advantages, args.advantage_quantile)
        weights = torch.where(advantages >= threshold, weights, torch.zeros_like(weights))
    returns = torch.tensor([transition.ret for transition in transitions], dtype=torch.float32)
    unclipped_return_mean = returns.mean()
    unclipped_return_std = returns.std(unbiased=False)
    if args.value_target_clip > 0:
        returns = returns.clamp(-args.value_target_clip, args.value_target_clip)
    clipped_return_mean = returns.mean()
    clipped_return_std = returns.std(unbiased=False)
    active_weight_fraction = float((weights > 0.0).float().mean())

    policy_losses: list[float] = []
    value_losses: list[float] = []
    minibatches_per_epoch = math.ceil(len(transitions) / args.batch_size)
    train_bar = None
    if args.progress:
        train_bar = tqdm.tqdm(
            total=args.train_epochs * minibatches_per_epoch,
            desc="train minibatches",
            unit="batch",
            leave=False,
        )

    try:
        for epoch in range(args.train_epochs):
            indices = torch.randperm(len(transitions))
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size].tolist()
                batch = [transitions[index] for index in batch_indices]
                observation = batch_observations(batch, device)
                action = torch.cat([transition.action for transition in batch], dim=0).to(device)
                batch_weights = weights[batch_indices].to(device)
                batch_returns = returns[batch_indices].to(device)

                with make_amp_context(device, policy.config.precision):
                    flow_loss = policy(observation, action).mean(dim=(1, 2))
                    weight_sum = batch_weights.sum()
                    if float(weight_sum.detach().cpu()) <= 1e-6:
                        policy_loss = flow_loss.mean()
                    else:
                        policy_loss = (batch_weights * flow_loss).sum() / weight_sum

                policy_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                policy_optimizer.step()

                predicted_values = value_net(observation.state)
                value_loss = nn.functional.huber_loss(
                    predicted_values,
                    batch_returns,
                    delta=args.value_huber_delta,
                )
                value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), args.value_max_grad_norm)
                value_optimizer.step()

                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                if train_bar is not None:
                    train_bar.update(1)
                    train_bar.set_postfix(
                        {
                            "epoch": epoch + 1,
                            "ploss": f"{policy_losses[-1]:.3f}",
                            "vloss": f"{value_losses[-1]:.3f}",
                            "w": f"{float(batch_weights.mean().detach().cpu()):.2f}",
                        }
                    )
    finally:
        if train_bar is not None:
            train_bar.close()

    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else math.nan,
        "value_loss": float(np.mean(value_losses)) if value_losses else math.nan,
        "advantage_mean": float(raw_advantage_mean),
        "advantage_std": float(raw_advantage_std),
        "weight_mean": float(weights.mean()),
        "active_weight_fraction": active_weight_fraction,
        "return_target_mean": float(clipped_return_mean),
        "return_target_std": float(clipped_return_std),
        "unclipped_return_target_mean": float(unclipped_return_mean),
        "unclipped_return_target_std": float(unclipped_return_std),
    }


def save_checkpoint(
    path: Path,
    *,
    policy: SmolPI,
    value_net: ValueNet,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    update: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "value_state_dict": value_net.state_dict(),
            "policy_optimizer_state_dict": policy_optimizer.state_dict(),
            "value_optimizer_state_dict": value_optimizer.state_dict(),
            "update": update,
            "action_horizon": policy.config.action_horizon,
            "args": vars(args),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Movement-level RL fine-tuning for the SmolPI WX250s robotic arm."
    )
    parser.add_argument("--scene", type=Path, default=Path("world/wx_scene.xml"))
    parser.add_argument("--task", choices=sorted(REWARD_BY_TASK), default="pick-red")
    parser.add_argument("--checkpoint", type=Path, default=Path("smolpi_bridge_rl_best1.pth"))
    parser.add_argument("--action-stats", type=Path, default=Path("action_stats.npz"))
    parser.add_argument("--output-checkpoint", type=Path, default=Path("smolpi_bridge_rl.pth"))
    parser.add_argument("--best-checkpoint", type=Path, default=Path("smolpi_bridge_rl_best.pth"))
    parser.add_argument("--metrics-output", type=Path, default=Path("rl_training_metrics.pkl"))
    parser.add_argument("--metrics-jsonl", type=Path, default=Path("rl_training_metrics.jsonl"))
    parser.add_argument("--video-dir", type=Path, default=Path("vids/rl"))
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument(
        "--parallel-envs",
        type=int,
        default=4,
        help="Number of parallel MuJoCo MjData rollouts to run per batched policy step.",
    )
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--control-hz", type=float, default=5.0)
    parser.add_argument("--flow-steps", type=int, default=14)
    parser.add_argument("--exploration-std", type=float, default=0.5)
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=50.0,
        help="Multiplier applied to raw environment rewards before GAE/value/policy updates.",
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--awr-temperature", type=float, default=1.0)
    parser.add_argument("--max-awr-weight", type=float, default=20.0)
    parser.add_argument(
        "--advantage-filter",
        choices=("positive", "top_quantile", "all"),
        default="positive",
        help="Which rollout actions the policy update is allowed to imitate.",
    )
    parser.add_argument(
        "--advantage-quantile",
        type=float,
        default=0.6,
        help="Quantile cutoff when --advantage-filter=top_quantile.",
    )
    parser.add_argument("--policy-lr", type=float, default=3e-5)
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument(
        "--value-target-clip",
        type=float,
        default=10.0,
        help="Clip critic return targets to +/- this value. Set <=0 to disable.",
    )
    parser.add_argument("--value-huber-delta", type=float, default=1.0)
    parser.add_argument("--value-max-grad-norm", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many evals without best-return improvement. 0 disables it.",
    )
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--video-every", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars and live metric postfixes.",
    )
    parser.add_argument(
        "--step-progress",
        action="store_true",
        help="Show a nested progress bar for movement steps inside each episode.",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float16", "bfloat16"),
        default="float16" if torch.cuda.is_available() else "float32",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task not in TASKS:
        raise ValueError(f"Unknown task {args.task!r}; choose from {sorted(TASKS)}")
    if args.episodes_per_update <= 0:
        raise ValueError("--episodes-per-update must be positive")
    if args.parallel_envs <= 0:
        raise ValueError("--parallel-envs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.train_epochs <= 0:
        raise ValueError("--train-epochs must be positive")
    if args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive")
    if args.exploration_std < 0:
        raise ValueError("--exploration-std must be non-negative")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be non-negative")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive")
    if not 0.0 <= args.advantage_quantile <= 1.0:
        raise ValueError("--advantage-quantile must be between 0 and 1")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be non-negative")
    if args.value_huber_delta <= 0:
        raise ValueError("--value-huber-delta must be positive")
    if args.value_max_grad_norm <= 0:
        raise ValueError("--value-max-grad-norm must be positive")

    args.video_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_jsonl.write_text("", encoding="utf-8")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = load_policy(args, args.device)
    value_net = ValueNet(state_dim=7).to(args.device)
    policy_optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=args.policy_lr,
        weight_decay=args.weight_decay,
    )
    value_optimizer = torch.optim.AdamW(value_net.parameters(), lr=args.value_lr)

    normalizer = ActionNormalizer(args.action_stats)
    processor = AutoProcessor.from_pretrained(policy.config.smolvlm_id)
    processor.image_processor.do_image_splitting = False
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    metrics: list[dict[str, float | int]] = []
    recent_returns: deque[float] = deque(maxlen=max(1, args.rolling_window))
    best_eval_return = -float("inf")
    evals_without_improvement = 0
    update_iter = range(1, args.updates + 1)
    update_bar = None
    if args.progress:
        update_bar = tqdm.tqdm(update_iter, desc="RL updates", unit="update")
        update_iter = update_bar
    try:
        for update in update_iter:
            record_path = (
                args.video_dir / f"{args.task}_update_{update:05d}.mp4"
                if args.video_every > 0 and update % args.video_every == 0
                else None
            )
            rollout_batch, returns, raw_returns, lengths, train_episode_metrics = collect_episodes_batched(
                args=args,
                policy=policy,
                value_net=value_net,
                processor=processor,
                normalizer=normalizer,
                device=args.device,
                num_episodes=args.episodes_per_update,
                update=update,
                exploration_std=args.exploration_std,
                store_transitions=True,
                phase="train",
                record_path=record_path,
            )

            if not rollout_batch:
                raise RuntimeError("Collected no transitions; check duration and scene setup")

            train_metrics = train_on_rollouts(
                args=args,
                policy=policy,
                value_net=value_net,
                policy_optimizer=policy_optimizer,
                value_optimizer=value_optimizer,
                transitions=rollout_batch,
                device=args.device,
            )
            recent_returns.extend(returns)
            rolling_return = float(np.mean(recent_returns))
            update_metrics = {
                "update": update,
                "mean_return": float(np.mean(returns)),
                "mean_raw_return": float(np.mean(raw_returns)),
                "rolling_return": rolling_return,
                "mean_length": float(np.mean(lengths)),
                "num_transitions": float(len(rollout_batch)),
                **mean_metric_dict("train", train_episode_metrics),
                **train_metrics,
            }
            if args.eval_every > 0 and update % args.eval_every == 0:
                _, eval_returns, eval_raw_returns, eval_lengths, eval_episode_metrics = collect_episodes_batched(
                    args=args,
                    policy=policy,
                    value_net=value_net,
                    processor=processor,
                    normalizer=normalizer,
                    device=args.device,
                    num_episodes=args.eval_episodes,
                    update=update,
                    exploration_std=0.0,
                    store_transitions=False,
                    phase="eval",
                    record_path=None,
                )
                update_metrics.update(
                    {
                        "eval_mean_return": float(np.mean(eval_returns)),
                        "eval_mean_raw_return": float(np.mean(eval_raw_returns)),
                        "eval_mean_length": float(np.mean(eval_lengths)),
                        **mean_metric_dict("eval", eval_episode_metrics),
                    }
                )
                if update_metrics["eval_mean_return"] > best_eval_return + args.early_stop_min_delta:
                    best_eval_return = update_metrics["eval_mean_return"]
                    evals_without_improvement = 0
                    save_checkpoint(
                        args.best_checkpoint,
                        policy=policy,
                        value_net=value_net,
                        policy_optimizer=policy_optimizer,
                        value_optimizer=value_optimizer,
                        update=update,
                        args=args,
                    )
                    if args.progress:
                        tqdm.tqdm.write(
                            f"Saved best eval checkpoint: {args.best_checkpoint} "
                            f"(eval_return={best_eval_return:.4f})"
                        )
                    else:
                        print(
                            f"Saved best eval checkpoint: {args.best_checkpoint} "
                            f"(eval_return={best_eval_return:.4f})",
                            flush=True,
                        )
                else:
                    evals_without_improvement += 1
                    if args.early_stop_patience > 0 and evals_without_improvement >= args.early_stop_patience:
                        update_metrics["early_stop"] = 1
            metrics.append(update_metrics)
            append_jsonl(args.metrics_jsonl, update_metrics)
            flush_metrics(args.metrics_output, metrics)
            message = (
                "update={update} return={mean_return:.4f} raw={mean_raw_return:.4f} "
                "rolling={rolling_return:.4f} "
                "len={mean_length:.1f} transitions={num_transitions:.0f} "
                "policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} "
                "vtarget={return_target_mean:.3f}/{return_target_std:.3f} "
                "adv={advantage_mean:.4f}/{advantage_std:.4f} weight={weight_mean:.2f} "
                "active_w={active_weight_fraction:.2f} "
                "success={train_success:.2f}"
            ).format(**update_metrics)
            if "eval_mean_return" in update_metrics:
                message += (
                    " eval_return={eval_mean_return:.4f} eval_raw={eval_mean_raw_return:.4f} "
                    "eval_success={eval_success:.2f}"
                ).format(**update_metrics)
            if update_bar is not None:
                update_bar.set_postfix(format_metrics(update_metrics))
                tqdm.tqdm.write(message)
            else:
                print(message, flush=True)

            if args.save_every > 0 and update % args.save_every == 0:
                save_checkpoint(
                    args.output_checkpoint,
                    policy=policy,
                    value_net=value_net,
                    policy_optimizer=policy_optimizer,
                    value_optimizer=value_optimizer,
                    update=update,
                    args=args,
                )
                if args.progress:
                    tqdm.tqdm.write(f"Saved checkpoint: {args.output_checkpoint}")
                else:
                    print(f"Saved checkpoint: {args.output_checkpoint}", flush=True)
            if update_metrics.get("early_stop", 0) == 1:
                if args.progress:
                    tqdm.tqdm.write(
                        f"Early stopping after {evals_without_improvement} evals without improvement. "
                        f"Best eval_return={best_eval_return:.4f} at {args.best_checkpoint}"
                    )
                else:
                    print(
                        f"Early stopping after {evals_without_improvement} evals without improvement. "
                        f"Best eval_return={best_eval_return:.4f} at {args.best_checkpoint}",
                        flush=True,
                    )
                break
    finally:
        if update_bar is not None:
            update_bar.close()
        save_checkpoint(
            args.output_checkpoint,
            policy=policy,
            value_net=value_net,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            update=len(metrics),
            args=args,
        )
        flush_metrics(args.metrics_output, metrics)


if __name__ == "__main__":
    main()
