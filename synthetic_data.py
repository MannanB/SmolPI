import torch
import tqdm 
from config import Config
from environment import MujocoEnvironment

import mujoco
import numpy as np

from model.smolpi import SmolPI, SmolPIConfig
from objectives import *

import os
import glob
import pickle
import random
from collections import defaultdict

import pickle

POSSIBLE_ACTIONS = [-2.2, -1.1, 0.0, 1.1, 2.2]

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

    reward_model = objective_cls(cfg)

    pre_face_reward_model = None
    if isinstance(reward_model, MoveToPlatformRewardModel):
        # grid search wont perform a "human like" rollout since it has outside info of where the platform is
        # instead we face the platform first and then move to platform
        pre_face_reward_model = FaceTargetUntilInViewRewardModel(cfg, target=reward_model.target)
    

    env.reset_episode_via_reward_model([reward_model])
    if pre_face_reward_model is not None:
        pre_face_reward_model.init_rollout(env.datas[0])

    action_grid = make_action_grid()

    for data in env.datas:
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
            dummy_policy,
            [reward_model.prompt],
            "cpu",
        )
        
        if pre_face_reward_model is not None and not pre_face_reward_model._in_view(env.datas[0]):
            action, _ = choose_next_action(
                env,
                pre_face_reward_model,
                action_grid,
                steps_per_control,
            )
        else:   
            action, _ = choose_next_action(
                env,
                reward_model,
                action_grid,
                steps_per_control,
            )

        data.ctrl[0] = float(action[0])
        data.ctrl[1] = float(action[1])

        for i in range(steps_per_control):
            mujoco.mj_step(env.model, env.datas[0])

            if write_to_camera:
                step = (chunk_step * steps_per_control) + i
                if step % steps_per_frame_omni == 0:
                    omni_rgb, omni_bgr = env.render_camera(env.omni_cam_renderer, "omniscient_cam", env_idx=0)
                    env.omni_cam_video.write(omni_bgr)

        reward = reward_model.update(env.datas[0])
        if reward_model.has_completed_task(env.datas[0]):
            steps_since_completed_task += 1
        else:
            steps_since_completed_task = 0

        if reward < 1e-3: # if we're not making progress, consider the episode "stuck" and end it to avoid generating redundant data
            steps_since_stuck += 1
        else:
            steps_since_stuck = 0

        if steps_since_completed_task > 10: # if we've completed the task for 11 steps, end the episode to avoid generating redundant data
            if write_to_camera:
                print("Completed task, ending episode early to avoid redundant data")
            break

        if steps_since_stuck > 8: # if we've been stuck for 9 steps, end the episode to avoid redundant data
            if write_to_camera:
                print("Stuck for too long, ending episode early to avoid redundant data")
            break


        samples.append({
            "observation": observation,
            "action": action,
            "reward": float(reward),
        })

        if pre_face_reward_model is not None:
            pre_face_reward_model.update(env.datas[0])

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
    env = MujocoEnvironment(cfg)

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
        smolpi=SmolPIConfig(action_dim=2, action_horizon=1, precision=torch.float16),
        num_parallel_rollouts=1,
        cam_omni_output_path="vids/omni_synth_test.mp4",
        cam_sim_output_path="vids/sim_synth_test.mp4",
    )
    objective_cls = MoveToPinkPlatformRewardModel
    dummy_policy = SmolPI(cfg.smolpi).to(cfg.device)

    samples = gen_synthetic_data(cfg, objective_cls, dummy_policy, seed=42, write_to_camera=True)
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
    # cfg = Config(
    #     smolpi=SmolPIConfig(action_dim=2, action_horizon=1, precision=torch.float16),
    #     num_parallel_rollouts=1,
    #     cam_omni_output_path="vids/omni_synth_test.mp4",
    #     cam_sim_output_path="vids/sim_synth_test.mp4",
    # )
    # objective_classes = OBJECTIVE_CLASSES
    # dummy_policy = SmolPI(cfg.smolpi).to("cpu")

    # samples = bulk_gen_synthetic_data(cfg, dummy_policy, objective_classes, num_episodes_per_class=[10,10,10,10,60,60,60,60,30,30,30,30,60])
    # print(f"Generated {len(samples)} samples.")
    distribute_and_shuffle_generated_data()
