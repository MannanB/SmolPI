from contextlib import nullcontext

import cv2, os

from transformers import AutoTokenizer
from model.smolpi import Observation, SmolPI, SmolPIConfig

import mujoco
import torch
import numpy as np
import tqdm

XML = ""
with open("world/platform_world.xml", "r", encoding="utf-8") as f:
    XML = f.read()

PROMPT = "Drive forward"
CAM_SIM_WIDTH, CAM_SIM_HEIGHT = 640, 480
CAM_OMNI_WIDTH, CAM_OMNI_HEIGHT = 1280, 960
CAM_SIM_FPS = 30
CAM_OMNI_FPS = 60
SIM_DURATION_SEC = 4
CONTROL_FREQ_HZ = 25 # Control frequency for the policy (e.g., 10 Hz means the policy outputs new actions every 0.1 seconds).
NUM_EPISODES_PER_UPDATE = 2 # Number of parallel episodes to run for each policy update (if doing training).
REWARD_SCALE = 0.5
NUM_UPDATES = 1000
KL_COEF = 0.02

def reset_episode(model: mujoco.MjModel, data: mujoco.MjData):
    mujoco.mj_resetData(model, data)
    data.qpos[0] = np.random.uniform(-0.5, 0.5)
    data.qpos[1] = np.random.uniform(-0.5, 0.5)
    mujoco.mj_forward(model, data)

def make_writer(path, width, height, fps):
    return cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def render_camera(renderer: mujoco.Renderer, data: mujoco.MjData, camera_name: str):
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb, bgr

def get_vision_input_shape(policy: SmolPI) -> tuple[int, int, int]:
    vision_cfg = policy.smolvlm_with_expert.smolvlm.config.vision_config
    channels = int(getattr(vision_cfg, "num_channels", 3))
    image_size = getattr(vision_cfg, "image_size", 384)
    if isinstance(image_size, (tuple, list)):
        h, w = int(image_size[0]), int(image_size[1])
    else:
        h = w = int(image_size)
    return channels, h, w


def frame_to_tesnor(rgb_frame: np.ndarray, height: int, width: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(rgb_frame, (width, height), interpolation=cv2.INTER_AREA)
    frame = torch.from_numpy(resized).to(device=device, dtype=torch.float32) / 255.0
    frame = frame.permute(2, 0, 1).contiguous()
    return frame.unsqueeze(0)


def make_observation(
    policy: SmolPI,
    tokenizer: AutoTokenizer,
    prompt: str,
    rgb_frame: np.ndarray,
    wheel_speeds: np.ndarray,
    device: torch.device,
) -> Observation:
    _, h, w = get_vision_input_shape(policy)
    image = frame_to_tesnor(rgb_frame, h, w, device)

    tokens = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = tokens["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = tokens["attention_mask"].to(device=device, dtype=torch.bool)

    state = torch.as_tensor(wheel_speeds, device=device, dtype=torch.float32).unsqueeze(0)

    return Observation(
        images={"front": image},
        image_masks={"front": torch.ones(1, device=device, dtype=torch.bool)},
        tokenized_prompt=input_ids,
        tokenized_prompt_mask=attention_mask,
        state=state,
    )


def get_wheel_speed_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    left_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
    right_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")

    left_dof = model.jnt_dofadr[left_joint_id]
    right_dof = model.jnt_dofadr[right_joint_id]

    return np.array([data.qvel[left_dof], data.qvel[right_dof]], dtype=np.float32)


def stack_rollout_batch(samples: list[dict], device: torch.device, batch_size: int) -> tuple[Observation, dict[str, torch.Tensor], torch.Tensor]:
    images = torch.stack([s["image"] for s in samples], dim=0).to(device=device, dtype=torch.float32)
    image_masks = torch.stack([s["image_mask"] for s in samples], dim=0).to(device=device, dtype=torch.bool)
    tokenized_prompt = torch.stack([s["prompt_ids"] for s in samples], dim=0).to(device=device, dtype=torch.long)
    tokenized_prompt_mask = torch.stack([s["prompt_mask"] for s in samples], dim=0).to(device=device, dtype=torch.bool)
    state = torch.stack([s["state"] for s in samples], dim=0).to(device=device, dtype=torch.float32)
    actions = {
        "final_actions": torch.stack([s["actions"] for s in samples], dim=0).to(device=device, dtype=torch.float32),
        "noisy_actions": torch.stack([s["noisy_actions"] for s in samples], dim=0).to(device=device, dtype=torch.float32),
        "next_noisy_actions": torch.stack([s["next_noisy_actions"] for s in samples], dim=0).to(device=device, dtype=torch.float32),
        "timesteps": torch.stack([s["timesteps"] for s in samples], dim=0).to(device=device, dtype=torch.float32),
        "old_log_probs": torch.stack([s["old_log_probs"] for s in samples], dim=0).to(device=device, dtype=torch.float32),
        "transition_std": torch.tensor(float(samples[0]["transition_std"]), device=device, dtype=torch.float32),
    }
    rewards = torch.tensor([s["reward"] for s in samples], device=device, dtype=torch.float32)

    obs = Observation(
        images={"front": images},
        image_masks={"front": image_masks},
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        state=state,
    )

    return obs, actions, rewards

def rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: SmolPIConfig,
    policy: SmolPI,
    tokenizer: AutoTokenizer,
    cam_renderer: mujoco.Renderer,
    omni_cam_renderer: mujoco.Renderer,
    device: torch.device,
    write_to_video: bool = False,
    cam_video: cv2.VideoWriter = None,
    omni_cam_video: cv2.VideoWriter = None,

):
    
    mujoco.mj_forward(model, data) # ensure first observation is created

    cam_frames = []
    omni_frames = []

    sim_steps = int(SIM_DURATION_SEC / model.opt.timestep)
    steps_per_frame = int(1 / CAM_SIM_FPS / model.opt.timestep)
    steps_per_frame_omni = int(1 / CAM_OMNI_FPS / model.opt.timestep)
    steps_per_control = max(1, int(1 / CONTROL_FREQ_HZ / model.opt.timestep))

    num_chunks = sim_steps // (steps_per_control * cfg.action_horizon)

    samples = []

    previous_x = float(data.qpos[0])
    policy.eval()

    for chunk_step in range(num_chunks):
        front_rgb, _ = render_camera(cam_renderer, data, "front_cam")
        wheel_speeds = get_wheel_speed_state(model, data)
        observation = make_observation(policy, tokenizer, PROMPT, front_rgb, wheel_speeds, device)

        if cfg.precision in (torch.float16, torch.bfloat16):
            amp_ctx = torch.autocast(device_type=device.type, dtype=cfg.precision)
        else:
            amp_ctx = nullcontext()
            
        with amp_ctx:
            action_chunk, ddpo_data = policy.sample_actions(
                device,
                observation,
                num_steps=10,
                transition_std=0.25,
                return_ddpo_data=True,
            )
        action_chunk_np = action_chunk[0].to(dtype=torch.float32).cpu().numpy()

        for ctrl_step in range(action_chunk_np.shape[0]):
            torque_cmd = action_chunk_np[ctrl_step]
            torque_cmd = np.clip(torque_cmd, -5.0, 5.0)

            data.ctrl[0] = float(torque_cmd[0])
            data.ctrl[1] = float(torque_cmd[1])

            for i in range(steps_per_control):
                mujoco.mj_step(model, data)

                step = chunk_step * action_chunk_np.shape[0] * steps_per_control + ctrl_step * steps_per_control + i

                if step % steps_per_frame == 0 and write_to_video and cam_video is not None:
                    front_rgb, front_bgr = render_camera(cam_renderer, data, "front_cam")
                    cam_frames.append(front_rgb.copy())
                    cam_video.write(front_bgr)

                if step % steps_per_frame_omni == 0 and write_to_video and omni_cam_video is not None:
                    omni_rgb, omni_bgr = render_camera(omni_cam_renderer, data, "omniscient_cam")
                    omni_frames.append(omni_rgb.copy())
                    omni_cam_video.write(omni_bgr)
            
        current_x = float(data.qpos[0])
        reward = REWARD_SCALE * (current_x - previous_x)
        previous_x = current_x

        samples.append(
            {
                "image": observation.images["front"][0].detach().cpu(),
                "image_mask": observation.image_masks["front"][0].detach().cpu(),
                "prompt_ids": observation.tokenized_prompt[0].detach().cpu(),
                "prompt_mask": observation.tokenized_prompt_mask[0].detach().cpu(),
                "state": observation.state[0].detach().cpu(),
                "actions": torch.from_numpy(action_chunk_np),
                "noisy_actions": ddpo_data["noisy_actions"][0].detach().cpu(),
                "next_noisy_actions": ddpo_data["next_noisy_actions"][0].detach().cpu(),
                "timesteps": ddpo_data["timesteps"][0].detach().cpu(),
                "old_log_probs": ddpo_data["old_log_probs"][0].detach().cpu(),
                "transition_std": float(ddpo_data["transition_std"].detach().cpu()),
                "reward": reward,
            }
        )
        
    return samples

def DDPO_update(
    policy: SmolPI,
    optimizer: torch.optim.Optimizer,
    observations: Observation,
    actions: dict[str, torch.Tensor],
    rewards: torch.Tensor,
):
    if not isinstance(actions, dict):
        raise ValueError("Exact DDPO_update requires the action dict returned by stack_rollout_batch after DDPO rollout collection")

    noisy_actions = actions["noisy_actions"].to(dtype=torch.float32)
    device = noisy_actions.device
    next_noisy_actions = actions["next_noisy_actions"].to(device=device, dtype=torch.float32)
    timesteps = actions["timesteps"].to(device=device, dtype=torch.float32)
    old_log_probs = actions["old_log_probs"].to(device=device, dtype=torch.float32).detach()
    transition_std = actions["transition_std"]
    rewards = rewards.to(device=device, dtype=torch.float32)
    observations = Observation(
        images={k: v.to(device=device, dtype=torch.float32) for k, v in observations.images.items()},
        image_masks={k: v.to(device=device, dtype=torch.bool) for k, v in observations.image_masks.items()},
        tokenized_prompt=observations.tokenized_prompt.to(device=device, dtype=torch.long),
        tokenized_prompt_mask=observations.tokenized_prompt_mask.to(device=device, dtype=torch.bool),
        state=observations.state.to(device=device, dtype=torch.float32),
    )

    obs_batch = next(iter(observations.images.values())).shape[0]
    target_batch = min(obs_batch, noisy_actions.shape[0], rewards.shape[0])
    if obs_batch != target_batch:
        observations = Observation(
            images={k: v[:target_batch] for k, v in observations.images.items()},
            image_masks={k: v[:target_batch] for k, v in observations.image_masks.items()},
            tokenized_prompt=observations.tokenized_prompt[:target_batch],
            tokenized_prompt_mask=observations.tokenized_prompt_mask[:target_batch],
            state=observations.state[:target_batch],
        )
    if noisy_actions.shape[0] != target_batch:
        noisy_actions = noisy_actions[:target_batch]
        next_noisy_actions = next_noisy_actions[:target_batch]
        timesteps = timesteps[:target_batch]
        old_log_probs = old_log_probs[:target_batch]
    if rewards.shape[0] != target_batch:
        rewards = rewards[:target_batch]

    if rewards.ndim != 1:
        raise ValueError(f"rewards must be [batch], got {rewards.shape}")
    if noisy_actions.ndim != 4:
        raise ValueError(f"noisy_actions must be [batch, denoise_steps, horizon, action_dim], got {noisy_actions.shape}")
    if noisy_actions.shape[0] != rewards.shape[0]:
        raise ValueError(f"actions batch ({noisy_actions.shape[0]}) and rewards batch ({rewards.shape[0]}) must match")

    advantages = rewards - rewards.mean()
    reward_std = rewards.std(unbiased=False)
    if torch.isfinite(reward_std) and reward_std > 1e-6:
        advantages = advantages / (reward_std + 1e-6)
    advantages = advantages.clamp(-5.0, 5.0).detach()

    precision = getattr(getattr(policy, "config", None), "precision", torch.float32)
    if device.type != "cpu" and precision in (torch.float16, torch.bfloat16):
        amp_ctx = torch.autocast(device_type=device.type, dtype=precision)
    else:
        amp_ctx = nullcontext()

    policy.train()
    optimizer.zero_grad(set_to_none=True)

    with amp_ctx:
        new_log_probs = policy.ddpo_log_probs(
            observations,
            noisy_actions,
            next_noisy_actions,
            timesteps,
            num_steps=noisy_actions.shape[1],
            transition_std=transition_std,
        )
        log_ratio = (new_log_probs - old_log_probs).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)
        clipped_ratio = ratio.clamp(0.8, 1.2)
        advantages = advantages[:, None].expand_as(new_log_probs)
        surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        loss = -surrogate.mean() + KL_COEF * approx_kl

    if not torch.isfinite(loss):
        raise RuntimeError("DDPO_update produced a non-finite loss")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "mean_reward": float(rewards.mean().detach().item()),
        "reward_std": float(rewards.std(unbiased=False).detach().item()),
        "min_reward": float(rewards.min().detach().item()),
        "max_reward": float(rewards.max().detach().item()),
        "mean_advantage": float(advantages.mean().detach().item()),
        "approx_kl": float(approx_kl.detach().item()),
        "clip_fraction": float(((ratio - 1.0).abs() > 0.2).to(torch.float32).mean().detach().item()),
    }

os.environ.setdefault("MUJOCO_GL", "glfw")

def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    reset_episode(model, data)

    cfg = SmolPIConfig(action_dim=2, action_horizon=5, precision=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and cfg.precision in (torch.float16, torch.bfloat16):
        cfg.precision = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(cfg.smolvlm_id)
    policy = SmolPI(cfg).to(device)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-5, weight_decay=1e-4)
    policy.eval()

    cam_renderer = mujoco.Renderer(model, height=CAM_SIM_HEIGHT, width=CAM_SIM_WIDTH)
    omni_cam_renderer = mujoco.Renderer(model, height=CAM_OMNI_HEIGHT, width=CAM_OMNI_WIDTH)

    cam_video = make_writer("vids/front_cam_vla_rollout.mp4", CAM_SIM_WIDTH, CAM_SIM_HEIGHT, CAM_SIM_FPS)
    omni_cam_video = make_writer("vids/omniscient_vla_rollout.mp4", CAM_OMNI_WIDTH, CAM_OMNI_HEIGHT, CAM_OMNI_FPS)

    update_pbar = tqdm.tqdm(range(NUM_UPDATES), desc="DDPO Updates", unit="update")
    # episode_pbar = tqdm.tqdm(range(NUM_EPISODES_PER_UPDATE), desc="Episodes per Update", unit="episode", leave=False)

    past_rewards = []

    try:
        for update in update_pbar:
            all_samples = []
            episode_returns = []
            for episode in range(NUM_EPISODES_PER_UPDATE):
                samples = rollout(
                    model=model,
                    data=data,
                    cfg=cfg,
                    policy=policy,
                    tokenizer=tokenizer,
                    cam_renderer=cam_renderer,
                    omni_cam_renderer=omni_cam_renderer,
                    device=device,
                    write_to_video=True,
                    cam_video=cam_video,
                    omni_cam_video=omni_cam_video,
                )
                print(len(samples), "samples collected")

                all_samples.extend(samples)
                episode_returns.append(sum(s["reward"] for s in samples))
                reset_episode(model, data)
            observations, actions, rewards = stack_rollout_batch(all_samples, device, batch_size=len(all_samples))
            metrics = DDPO_update(policy, optimizer, observations, actions, rewards)
            mean_episode_return = float(np.mean(episode_returns))
            print(
                f"update={update} samples={len(all_samples)} "
                f"episode_return={mean_episode_return:.4f} "
                f"reward={metrics['mean_reward']:.4f} reward_std={metrics['reward_std']:.4f} "
                f"reward_range=[{metrics['min_reward']:.4f},{metrics['max_reward']:.4f}] loss={metrics['loss']:.6f} "
                f"kl={metrics['approx_kl']:.6f} clip={metrics['clip_fraction']:.3f}"
            )
            past_rewards.append(mean_episode_return)
    finally:
        cam_video.release()
        omni_cam_video.release()
        cam_renderer.close()
        omni_cam_renderer.close() 
        import pickle
        with open("vla_rollout_rewards.pkl", "wb") as f:
            pickle.dump(past_rewards, f)
        # save the model
        torch.save(policy.state_dict(), "vla_rollout_policy.pt")

if __name__ == "__main__":
    main()
