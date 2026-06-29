import torch
import tqdm 
from two_wheeled.config import Config
from environment import MujocoEnvironment, get_vision_input_shape

import mujoco
import numpy as np

from model.smolpi import SmolPI, SmolPIConfig
from two_wheeled.objectives import *

import os
import glob
import pickle
import random
from collections import defaultdict

import pickle

POSSIBLE_ACTIONS = [-2.0, -1.3, -0.6, 0.0, 0.6, 1.3, 2.0]
FIXED_FORWARD_ACTION = np.array([2.0, 2.0], dtype=np.float32)
FIXED_BACKWARD_ACTION = np.array([-2.0, -2.0], dtype=np.float32)
FIXED_SPIN_COUNTER_CLOCKWISE_ACTION = np.array([-2.0, 2.0], dtype=np.float32)
FIXED_SPIN_CLOCKWISE_ACTION = np.array([2.0, -2.0], dtype=np.float32)
SEARCH_TURN_ACTION = np.array([-2.0, 2.0], dtype=np.float32)
STOP_ACTION = np.array([0.0, 0.0], dtype=np.float32)

MIN_TARGET_PIXEL_FRACTION = 0.0003
FACE_CENTER_DEADBAND = 0.12
MOVE_CENTER_DEADBAND = 0.18
MIN_TURN_TORQUE = 0.7
MAX_TORQUE = 2.0
FACE_TURN_GAIN = 2.6
MOVE_TURN_GAIN = 2.2
MOVE_FORWARD_TORQUE = 1.5


def fixed_synthetic_action(reward_model: RewardModel) -> np.ndarray | None:
    if isinstance(reward_model, MoveBackwardRewardModel):
        return FIXED_BACKWARD_ACTION
    if isinstance(reward_model, MoveForwardRewardModel):
        return FIXED_FORWARD_ACTION
    if isinstance(reward_model, SpinCounterClockwiseRewardModel):
        return FIXED_SPIN_COUNTER_CLOCKWISE_ACTION
    if isinstance(reward_model, SpinClockwiseRewardModel):
        return FIXED_SPIN_CLOCKWISE_ACTION
    return None

def make_synthetic_env(cfg: Config, dummy_policy: SmolPI | None = None) -> MujocoEnvironment:
    if dummy_policy is None:
        return MujocoEnvironment(cfg, 512, 512)

    obs_height, obs_width = get_vision_input_shape(dummy_policy)

    return MujocoEnvironment(cfg, obs_width, obs_height)

def _image_to_rgb01(image) -> np.ndarray:
    image = torch.as_tensor(image, dtype=torch.float32).detach().cpu()
    if image.ndim == 4:
        image = image.squeeze(0)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected image shape [3,H,W] or [1,3,H,W], got {tuple(image.shape)}")

    rgb = image.numpy()
    if float(rgb.min()) < 0.0:
        rgb = (rgb + 1.0) * 0.5
    elif float(rgb.max()) > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)

def _target_color_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    r, g, b = rgb
    if color == "red":
        return (
            (r > 0.18)
            & (r > g * 1.35)
            & (r > b * 1.35)
            & ((r - g) > 0.08)
            & ((r - b) > 0.08)
        )
    if color == "blue":
        return (
            (b > 0.18)
            & (b > r * 1.25)
            & (b > g * 1.15)
            & ((b - r) > 0.08)
            & ((b - g) > 0.04)
        )
    if color == "green":
        return (
            (g > 0.16)
            & (g > r * 1.25)
            & (g > b * 1.25)
            & ((g - r) > 0.05)
            & ((g - b) > 0.05)
        )
    if color == "pink":
        return (
            (r > 0.25)
            & (b > 0.18)
            & (r > g * 1.25)
            & (b > g * 1.15)
            & ((r - b) < 0.45)
        )
    raise ValueError(f"unsupported platform color: {color}")

def _detect_platform(observation, color: str) -> dict[str, float] | None:
    rgb = _image_to_rgb01(observation.images["front"][0])
    mask = _target_color_mask(rgb, color)
    height, width = mask.shape
    locations = np.argwhere(mask)
    min_pixels = max(64, int(width * height * MIN_TARGET_PIXEL_FRACTION))
    if locations.shape[0] < min_pixels:
        return None

    ys = locations[:, 0]
    xs = locations[:, 1]
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    return {
        "error_x": (center_x - (width * 0.5)) / (width * 0.5),
        "center_y": center_y / float(height),
        "area_fraction": float(locations.shape[0]) / float(width * height),
        "bottom": float(ys.max()) / float(height),
    }

def _turn_only_action(error_x: float, deadband: float) -> np.ndarray:
    if abs(error_x) <= deadband:
        return STOP_ACTION.copy()

    torque = np.clip(abs(error_x) * FACE_TURN_GAIN, MIN_TURN_TORQUE, MAX_TORQUE)
    if error_x < 0.0:
        return np.array([-torque, torque], dtype=np.float32)
    return np.array([torque, -torque], dtype=np.float32)

def _move_toward_action(error_x: float) -> np.ndarray:
    if abs(error_x) <= MOVE_CENTER_DEADBAND:
        return FIXED_FORWARD_ACTION.copy()

    turn = np.clip(abs(error_x) * MOVE_TURN_GAIN, MIN_TURN_TORQUE, MAX_TORQUE)
    forward = MOVE_FORWARD_TORQUE * max(0.0, 1.0 - min(abs(error_x), 1.0))
    if error_x < 0.0:
        action = np.array([forward - turn, forward + turn], dtype=np.float32)
    else:
        action = np.array([forward + turn, forward - turn], dtype=np.float32)
    return np.clip(action, -MAX_TORQUE, MAX_TORQUE).astype(np.float32)

def supervised_action_from_observation(
    reward_model: RewardModel,
    observation,
) -> np.ndarray | None:
    fixed_action = fixed_synthetic_action(reward_model)
    if fixed_action is not None:
        return fixed_action.copy()

    if isinstance(reward_model, FacePlatformRewardModel):
        detection = _detect_platform(observation, reward_model.color)
        if detection is None:
            return SEARCH_TURN_ACTION.copy()
        return _turn_only_action(detection["error_x"], FACE_CENTER_DEADBAND)

    if isinstance(reward_model, MoveToPlatformRewardModel):
        detection = _detect_platform(observation, reward_model.color)
        if detection is None:
            return SEARCH_TURN_ACTION.copy()
        return _move_toward_action(detection["error_x"])

    return None

def make_action_grid() -> np.ndarray:
    values = POSSIBLE_ACTIONS
    left, right = np.meshgrid(values, values, indexing="ij")
    return np.stack([left.ravel(), right.ravel()], axis=1)

def choose_next_action(
    environment: MujocoEnvironment,
    reward_model: RewardModel,
    action_grid: np.ndarray,
    steps_per_control: int,
) -> tuple[np.ndarray, float]:
    best_action = action_grid[0]
    best_reward = -float("inf")

    for action in action_grid:
        reward = environment.score_candidate(reward_model, action, steps_per_control, env_idx=0)
        if reward > best_reward:
            best_reward = reward
            best_action = action

    return best_action.copy(), best_reward

        
def gen_synthetic_data(
    cfg: Config,
    env: MujocoEnvironment,
    objective_cls: RewardModel,
    dummy_policy: SmolPI | None = None,
    seed: int | None = None,
    write_to_camera: bool = False,
):
    if seed is not None:
        np.random.seed(seed)
    if len(env.datas) != 1:
        raise ValueError("gen_synthetic_data is the single-env generator; use a cfg/env with num_parallel_rollouts=1")

    reward_model = objective_cls(cfg)

    pre_face_reward_model = None
    if isinstance(reward_model, MoveToPlatformRewardModel):
        # grid search wont perform a "human like" rollout since it has outside info of where the platform is
        # instead we face the platform first and then move to platform
        pre_face_reward_model = FaceTargetUntilInViewRewardModel(cfg, target=reward_model.target)
    

    env.reset_episode_via_reward_model([reward_model])
    if pre_face_reward_model is not None:
        pre_face_reward_model.init_rollout(env.datas[0])

    fixed_action = fixed_synthetic_action(reward_model)
    action_grid = make_action_grid()

    data = env.datas[0]
    mujoco.mj_forward(env.model, data) # ensure first observation is created

    sim_steps = int(env.cfg.sim_duration_sec / env.model.opt.timestep)
    steps_per_control = max(1, int(1 / env.cfg.control_freq_hz / env.model.opt.timestep))
    steps_per_frame_omni = int(1 / env.cfg.cam_omni_fps / env.model.opt.timestep)
    # print(steps_per_frame_omni)
    num_chunks = sim_steps // (steps_per_control * env.cfg.smolpi.action_horizon)

    samples = []


    steps_since_completed_task = 0
    steps_since_stuck = 0

    for chunk_step in tqdm.tqdm(range(num_chunks), desc="Generating Synthetic Data", unit="chunk", leave=False):
        observation = env.make_observations(
            [reward_model.prompt],
            torch.device("cpu"),
        )
        supervised_action = supervised_action_from_observation(reward_model, observation)
        
        action_chunk = []
        chunk_reward = 0.0
        end_episode = False

        for horizon_step in range(env.cfg.smolpi.action_horizon):
            if fixed_action is not None:
                rollout_action = fixed_action.copy()
            elif pre_face_reward_model is not None and not pre_face_reward_model._in_view(env.datas[0]):
                rollout_action, _ = choose_next_action(
                    env,
                    pre_face_reward_model,
                    action_grid,
                    steps_per_control,
                )
            else:
                rollout_action, _ = choose_next_action(
                    env,
                    reward_model,
                    action_grid,
                    steps_per_control,
                )

            action_label = rollout_action if supervised_action is None else supervised_action
            action_chunk.append(action_label.astype(np.float32, copy=True))
            env.datas[0].ctrl[0] = float(rollout_action[0])
            env.datas[0].ctrl[1] = float(rollout_action[1])

            for i in range(steps_per_control):
                mujoco.mj_step(env.model, env.datas[0])

                if write_to_camera:
                    step = (
                        (chunk_step * env.cfg.smolpi.action_horizon * steps_per_control)
                        + (horizon_step * steps_per_control)
                        + i
                    )
                    if step % steps_per_frame_omni == 0:
                        omni_rgb, omni_bgr = env.render_camera(env.omni_cam_renderer, "omniscient_cam", env_idx=0)
                        env.omni_cam_video.write(omni_bgr)

            if fixed_action is None:
                reward = reward_model.update(env.datas[0])
                chunk_reward += float(reward)
                if reward_model.has_completed_task(env.datas[0]):
                    steps_since_completed_task += 1
                else:
                    steps_since_completed_task = 0

            # if reward < 1e-3: # if we're not making progress, consider the episode "stuck" and end it to avoid generating redundant data
            #     steps_since_stuck += 1
            # else:
            #     steps_since_stuck = 0

            if pre_face_reward_model is not None:
                pre_face_reward_model.update(env.datas[0])

            if steps_since_completed_task > 10: # if we've completed the task for 11 steps, end the episode to avoid generating redundant data
                if write_to_camera:
                    print("Completed task, ending episode early to avoid redundant data")
                end_episode = True
                break

            if steps_since_stuck > 8: # if we've been stuck for 9 steps, end the episode to avoid generating redundant data
                if write_to_camera:
                    print("Stuck for too long, ending episode early to avoid redundant data")
                end_episode = True
                break

        if len(action_chunk) == env.cfg.smolpi.action_horizon:
            samples.append({
                "observation": observation,
                "action": np.stack(action_chunk, axis=0),
                "reward": chunk_reward,
            })

        if end_episode:
            break

    if write_to_camera:
        env.close()
        
    return samples

def bulk_gen_synthetic_data(
    cfg: Config,
    dummy_policy: SmolPI | None,
    objective_classes: list[RewardModel],
    num_episodes_per_class: int
):
    all_samples = []
    env = make_synthetic_env(cfg, dummy_policy)

    try:
        pbar = tqdm.tqdm(list(enumerate(objective_classes)), desc="Objective Classes", leave=True)
        for class_idx, objective_cls in pbar:
            pbar.set_description(f"Generating data for objective {class_idx+1}/{len(objective_classes)}: {objective_cls.__name__}")
            for episode in tqdm.tqdm(range(num_episodes_per_class[class_idx]), desc=f"Episode", leave=False):
                episode_seed = (class_idx * num_episodes_per_class[class_idx]) + episode
                samples = gen_synthetic_data(
                    cfg,
                    env,
                    objective_cls,
                    dummy_policy,
                    seed=episode_seed,
                )
                all_samples.extend(samples)
            # write to file and clear samples
            with open(f"data/synthetic_samples_{objective_cls.__name__}.pkl", "wb") as f:
                pickle.dump(all_samples, f)
            all_samples.clear()
            del samples
    finally:
        # save all_samples
        with open(f"data/synthetic_samples_interrupted.pkl", "wb") as f:
            pickle.dump(all_samples, f)
    return all_samples

def try_one_sample():
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=2, action_horizon=5, precision=torch.float16),
        num_parallel_rollouts=1,
        cam_omni_output_path="vids/omni_synth_test.mp4",
        cam_sim_output_path="vids/sim_synth_test.mp4",
    )
    objective_cls = MoveToRedPlatformRewardModel
    dummy_policy = SmolPI(cfg.smolpi).to(cfg.device)

    env = make_synthetic_env(cfg, dummy_policy)
    samples = gen_synthetic_data(cfg, env, objective_cls, dummy_policy, seed=42, write_to_camera=True)
    print(f"Generated {len(samples)} samples for objective {objective_cls.__name__}.")


NUM_SHUFFLE_FILES = 20

def append_pickle(path, obj):
    with open(path, "ab") as f:
        pickle.dump(obj, f)

def load_appended_pickles(path):
    items = []

    with open(path, "rb") as f:
        while True:
            try:
                items.extend(pickle.load(f))
            except EOFError:
                break

    return items

def distribute_and_shuffle_generated_data():
    out_dir = "data/shuffled_chunks"
    os.makedirs(out_dir, exist_ok=True)

    shuffle_files = [
        f"{out_dir}/synthetic_samples_shuffled_{i}.pkl"
        for i in range(NUM_SHUFFLE_FILES)
    ]

    # clear old outputs
    for path in shuffle_files:
        if os.path.exists(path):
            os.remove(path)

    source_files = [
        f for f in glob.glob("data/synthetic_samples_*.pkl")
        if "shuffled" not in f
    ]

    # distribute each class/type file across all shuffle files
    for file in source_files:
        print(f"Reading {file}")

        with open(file, "rb") as f:
            samples = pickle.load(f)

        random.shuffle(samples)

        n = len(samples)
        part_size = (n + NUM_SHUFFLE_FILES - 1) // NUM_SHUFFLE_FILES

        for i, out_file in enumerate(shuffle_files):
            start = i * part_size
            end = min(start + part_size, n)
            part = samples[start:end]
            if part:
                append_pickle(out_file, part)

        del samples

    # now shuffle each output file independently
    for path in shuffle_files:
        print(f"Final shuffling {path}")

        samples = load_appended_pickles(path)
        random.shuffle(samples)

        with open(path, "wb") as f:
            pickle.dump(samples, f)

        del samples

if __name__ == "__main__":
    # try_one_sample()
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=2, action_horizon=5, precision=torch.float16),
        num_parallel_rollouts=1,
        cam_omni_output_path="vids/omni_synth_test.mp4",
        cam_sim_output_path="vids/sim_synth_test.mp4",
    )
    
    objective_classes = OBJECTIVE_CLASSES
    dummy_policy = SmolPI(cfg.smolpi).to("cpu")

    samples = bulk_gen_synthetic_data(cfg, dummy_policy, objective_classes, num_episodes_per_class=[10,10,10,10,60,60,60,60,30,30,30,30,60])
    print(f"Generated {len(samples)} samples.")
    print("Distributing and shuffling generated data...")
    distribute_and_shuffle_generated_data()
