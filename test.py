import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from model.smolpi import SmolPI, Observation, SmolPIConfig
from environment import MujocoEnvironment, get_vision_input_shape
from contextlib import nullcontext
from objectives import *
from config import Config

OBJECTIVE = MoveToBluePlatformRewardModel

def batch_test(file=None):
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=2, action_horizon=5, precision=torch.float16),
        cam_sim_output_path=f"vids/front_cam_vla_batch_test_{file}.mp4",
        cam_omni_output_path=f"vids/omniscient_vla_batch_test_{file}.mp4",
        sim_duration_sec=5,
    )

    policy = SmolPI(cfg.smolpi).to(cfg.device)
    env = MujocoEnvironment(cfg, *get_vision_input_shape(policy))

    if file is not None:
        # checkpoint = torch.load("./ddpo_training.pt", map_location=cfg.device)
        # policy.load_state_dict(checkpoint["policy_state_dict"])
        checkpoint = torch.load("./"+file, map_location=cfg.device)
        policy.load_state_dict(checkpoint)

    policy.eval()

    for objective in [MoveToPinkPlatformRewardModel, MoveToRedPlatformRewardModel, MoveToBluePlatformRewardModel, FaceGreenPlatformRewardModel, FacePinkPlatformRewardModel]:
        all_samples = env.rollout(
            policy=policy,
            reward_models=[objective(cfg) for _ in range(cfg.num_parallel_rollouts)],
            write_to_video=True,
        )

        episode_returns = [
            sum(s["reward"] for s in all_samples if s["env_idx"] == env_idx)
            for env_idx in range(cfg.num_parallel_rollouts)
        ]
        mean_episode_return = float(np.mean(episode_returns))
        print(f"Objective: {objective.__name__}, Mean Episode Return: {mean_episode_return:.2f}")
    env.close()

def single_test():
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=2, action_horizon=1, precision=torch.float16),
        num_parallel_rollouts=1,
        cam_sim_output_path="vids/front_cam_vla_test_rollout.mp4",
        cam_omni_output_path="vids/omniscient_vla_test_rollout.mp4",
        sim_duration_sec=8,
    )
    env = MujocoEnvironment(cfg)

    policy = SmolPI(cfg.smolpi).to(cfg.device)
    # checkpoint = torch.load("./ddpo_training.pt", map_location=cfg.device)
    # policy.load_state_dict(checkpoint["policy_state_dict"])
    checkpoint = torch.load("./smolpi_sft_final.pth", map_location=cfg.device)
    policy.load_state_dict(checkpoint)

    policy.eval()


    all_samples = env.rollout(
        policy=policy,
        reward_models=[OBJECTIVE(cfg)],
        write_to_video=True,
    )
    episode_returns = [
        sum(s["reward"] for s in all_samples if s["env_idx"] == env_idx)
        for env_idx in range(cfg.num_parallel_rollouts)
    ]

    mean_episode_return = float(np.mean(episode_returns))
    print(f"Mean Episode Return: {mean_episode_return:.2f}")

    env.close()


if __name__ == "__main__":
    print("-- Running All Objectives Test (SFT) --")
    batch_test(file="smolpi_sft_final.pth")
    # print("-- Running All Objectives Test (SFT+RL) --")
    # batch_test(file="ddpo_training.pt")
    print("\n-- Running Baseline Test --")
    batch_test(file=None)
    # print("-- Running Single Objective Test --")
    # single_test()