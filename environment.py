from contextlib import nullcontext
import os

import cv2

from transformers import AutoTokenizer
from model.smolpi import Observation, SmolPI

import mujoco
import torch
import numpy as np

from config import Config
from objectives import RewardModel

os.environ.setdefault("MUJOCO_GL", "glfw")
    

def make_writer(path, width, height, fps):
    return cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

def get_vision_input_shape(policy: SmolPI) -> tuple[int, int, int]:
    vision_cfg = policy.smolvlm_with_expert.smolvlm.config.vision_config
    channels = int(getattr(vision_cfg, "num_channels", 3))
    image_size = getattr(vision_cfg, "image_size", 384)
    if isinstance(image_size, (tuple, list)):
        h, w = int(image_size[0]), int(image_size[1])
    else:
        h = w = int(image_size)
    return channels, h, w

def frame_to_tensor(rgb_frame: np.ndarray, height: int, width: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(rgb_frame, (width, height), interpolation=cv2.INTER_AREA)
    frame = torch.from_numpy(resized).to(device=device, dtype=torch.float32) / 255.0
    frame = frame.permute(2, 0, 1).contiguous()
    return frame.unsqueeze(0)

class MujocoEnvironment:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        xml_str = ""
        with open(cfg.world_xml_path, "r") as f:
            xml_str = f.read()

        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.datas = [mujoco.MjData(self.model) for _ in range(cfg.num_parallel_rollouts)]

        self.reset_episode()

        left_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
        right_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")
        self.left_dof = self.model.jnt_dofadr[left_joint_id]
        self.right_dof = self.model.jnt_dofadr[right_joint_id]

        self.cam_renderer = mujoco.Renderer(self.model, width=cfg.cam_sim_width, height=cfg.cam_sim_height)
        self.omni_cam_renderer = mujoco.Renderer(self.model, width=cfg.cam_omni_width, height=cfg.cam_omni_height)

        self.cam_video = make_writer(cfg.cam_sim_output_path, cfg.cam_sim_width, cfg.cam_sim_height, cfg.cam_sim_fps)
        self.omni_cam_video = make_writer(cfg.cam_omni_output_path, cfg.cam_omni_width, cfg.cam_omni_height, cfg.cam_omni_fps)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.smolpi.smolvlm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def reset_episode(self, randomize_position: bool = True, randomize_rotation: bool = True):
        for data in self.datas:
            mujoco.mj_resetData(self.model, data)
            if randomize_position:
                data.qpos[0] = np.random.uniform(-0.5, 0.5)
                data.qpos[1] = np.random.uniform(-0.5, 0.5)
            if randomize_rotation:
                data.qpos[2] = np.random.uniform(-np.pi, np.pi)
            mujoco.mj_forward(self.model, data)

    def reset_episode_via_reward_model(self, reward_models: list[RewardModel]):
        if len(reward_models) != len(self.datas):
            raise ValueError(f"expected {len(self.datas)} reward models, got {len(reward_models)}")
        for i, data in enumerate(self.datas):
            mujoco.mj_resetData(self.model, data)
            reward_models[i].init_rollout(data)
            mujoco.mj_forward(self.model, data)    

    def render_camera(self, renderer: mujoco.Renderer, camera_name: str, env_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
        data = self.datas[env_idx]
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return rgb, bgr
    
    def get_wheel_speed_state(self, env_idx: int) -> np.ndarray:
        data = self.datas[env_idx]
        return np.array([data.qvel[self.left_dof], data.qvel[self.right_dof]], dtype=np.float32)

    def make_observations(
        self,
        policy: SmolPI,
        prompts: list[str],
        device: torch.device,
        env_indices: list[int] | None = None,
    ) -> Observation:
        if env_indices is None:
            env_indices = list(range(len(prompts)))
        if len(prompts) != len(env_indices):
            raise ValueError(f"got {len(prompts)} prompts for {len(env_indices)} environment indices")

        _, h, w = get_vision_input_shape(policy)
        images = []
        states = []
        for env_idx in env_indices:
            rgb_frame, _ = self.render_camera(self.cam_renderer, "front_cam", env_idx)
            images.append(frame_to_tensor(rgb_frame, h, w, device))
            states.append(torch.as_tensor(self.get_wheel_speed_state(env_idx), device=device, dtype=torch.float32))

        tokens = self.tokenizer(prompts, return_tensors="pt", truncation=True, padding=True)
        input_ids = tokens["input_ids"].to(device=device, dtype=torch.long)
        attention_mask = tokens["attention_mask"].to(device=device, dtype=torch.bool)
        image_batch = torch.cat(images, dim=0)
        state_batch = torch.stack(states, dim=0)

        return Observation(
            images={"front": image_batch},
            image_masks={"front": torch.ones(len(env_indices), device=device, dtype=torch.bool)},
            tokenized_prompt=input_ids,
            tokenized_prompt_mask=attention_mask,
            state=state_batch,
        )

    def make_observation(
        self,
        policy: SmolPI,
        prompt: str,
        device: torch.device,
        env_idx: int = 0,
    ) -> Observation:
        return self.make_observations(policy, [prompt], device, [env_idx])

    def rollout(
        self,
        policy: SmolPI,
        reward_models: list[RewardModel],
        write_to_video: bool = False,
    ):
        
        self.reset_episode_via_reward_model(reward_models)
        
        for data in self.datas:
            mujoco.mj_forward(self.model, data) # ensure first observation is created

        cam_frames = []
        omni_frames = []

        sim_steps = int(self.cfg.sim_duration_sec / self.model.opt.timestep)
        steps_per_frame = int(1 / self.cfg.cam_sim_fps / self.model.opt.timestep)
        steps_per_frame_omni = int(1 / self.cfg.cam_omni_fps / self.model.opt.timestep)
        steps_per_control = max(1, int(1 / self.cfg.control_freq_hz / self.model.opt.timestep))

        num_chunks = sim_steps // (steps_per_control * self.cfg.smolpi.action_horizon)

        samples = []

        policy.eval()

        for chunk_step in range(num_chunks):
            observation = self.make_observations(
                policy,
                [reward_model.prompt for reward_model in reward_models],
                self.cfg.device,
            )
            

            if self.cfg.device.type != "cpu" and self.cfg.smolpi.precision in (torch.float16, torch.bfloat16):
                amp_ctx = torch.autocast(device_type=self.cfg.device.type, dtype=self.cfg.smolpi.precision)
            else:
                amp_ctx = nullcontext()
                
            with amp_ctx:
                action_chunk, ddpo_data = policy.sample_actions(
                    self.cfg.device,
                    observation,
                    num_steps=self.cfg.flow_steps,
                    transition_std=self.cfg.flow_std,
                    return_ddpo_data=True,
                )
            action_chunk_np = action_chunk.to(dtype=torch.float32).cpu().numpy()

            for ctrl_step in range(action_chunk_np.shape[1]):
                for env_idx, data in enumerate(self.datas):
                    torque_cmd = np.clip(action_chunk_np[env_idx, ctrl_step], -5.0, 5.0)
                    
                    data.ctrl[0] = float(torque_cmd[0])
                    data.ctrl[1] = float(torque_cmd[1])

                    # print(f"Env {env_idx}, Step {chunk_step * self.cfg.smolpi.action_horizon + ctrl_step}: Action: {torque_cmd}")

                for i in range(steps_per_control):
                    for data in self.datas:
                        mujoco.mj_step(self.model, data)

                    step = chunk_step * action_chunk_np.shape[1] * steps_per_control + ctrl_step * steps_per_control + i

                    if step % steps_per_frame == 0 and write_to_video and self.cam_video is not None:
                        front_rgb, front_bgr = self.render_camera(self.cam_renderer, "front_cam", env_idx=0)
                        cam_frames.append(front_rgb.copy())
                        self.cam_video.write(front_bgr)

                    if step % steps_per_frame_omni == 0 and write_to_video and self.omni_cam_video is not None:
                        omni_rgb, omni_bgr = self.render_camera(self.omni_cam_renderer, "omniscient_cam", env_idx=0)
                        omni_frames.append(omni_rgb.copy())
                        self.omni_cam_video.write(omni_bgr)

            for env_idx, (data, reward_model) in enumerate(zip(self.datas, reward_models, strict=True)):
                reward = reward_model.update(data)
                # print(f"Env {env_idx}, Chunk {chunk_step}: Reward: {reward:.4f}")
                samples.append(
                    {
                        "env_idx": env_idx,
                        "prompt": reward_model.prompt,
                        "image": observation.images["front"][env_idx].detach().cpu(),
                        "image_mask": observation.image_masks["front"][env_idx].detach().cpu(),
                        "prompt_ids": observation.tokenized_prompt[env_idx].detach().cpu(),
                        "prompt_mask": observation.tokenized_prompt_mask[env_idx].detach().cpu(),
                        "state": observation.state[env_idx].detach().cpu(),
                        "actions": torch.from_numpy(action_chunk_np[env_idx]),
                        "noisy_actions": ddpo_data["noisy_actions"][env_idx].detach().cpu(),
                        "next_noisy_actions": ddpo_data["next_noisy_actions"][env_idx].detach().cpu(),
                        "timesteps": ddpo_data["timesteps"][env_idx].detach().cpu(),
                        "old_log_probs": ddpo_data["old_log_probs"][env_idx].detach().cpu(),
                        "transition_std": float(ddpo_data["transition_std"].detach().cpu()),
                        "reward": reward,
                    }
                )
            
        return samples
    
    def close(self):
        if self.cam_video is not None:
            self.cam_video.release()
        if self.omni_cam_video is not None:
            self.omni_cam_video.release()
        
        self.cam_renderer.close()
        self.omni_cam_renderer.close()
