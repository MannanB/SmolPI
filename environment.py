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
        self.data = mujoco.MjData(self.model)

        self.reset_episode(self.model, self.data)

        self.cam_renderer = mujoco.Renderer(self.model, width=cfg.cam_sim_width, height=cfg.cam_sim_height)
        self.omni_cam_renderer = mujoco.Renderer(self.model, width=cfg.cam_omni_width, height=cfg.cam_omni_height)

        self.cam_video = make_writer(cfg.cam_sim_output_path, cfg.cam_sim_width, cfg.cam_sim_height, cfg.cam_sim_fps)
        self.omni_cam_video = make_writer(cfg.cam_omni_output_path, cfg.cam_omni_width, cfg.cam_omni_height, cfg.cam_omni_fps)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.smolpi.smolvlm_id)

    def reset_episode(self, randomize_position: bool = True, randomize_rotation: bool = True):
        mujoco.mj_resetData(self.model, self.data)
        if randomize_position:
            self.data.qpos[0] = np.random.uniform(-0.5, 0.5)
            self.data.qpos[1] = np.random.uniform(-0.5, 0.5)
        if randomize_rotation:
            self.data.qpos[2] = np.random.uniform(-np.pi, np.pi)
        mujoco.mj_forward(self.model, self.data)

    def reset_episode_via_reward_model(self, reward_model: RewardModel):
        mujoco.mj_resetData(self.model, self.data)
        reward_model.init_rollout(self.data)
        mujoco.mj_forward(self.model, self.data)

    def render_camera(self, renderer: mujoco.Renderer, camera_name: str):
        renderer.update_scene(self.data, camera=camera_name)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return rgb, bgr
    
    def get_wheel_speed_state(self) -> np.ndarray:
        left_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
        right_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")

        left_dof = self.model.jnt_dofadr[left_joint_id]
        right_dof = self.model.jnt_dofadr[right_joint_id]

        return np.array([self.data.qvel[left_dof], self.data.qvel[right_dof]], dtype=np.float32)

    def make_observation(
        self,
        policy: SmolPI,
        prompt: str,
        rgb_frame: np.ndarray,
        device: torch.device,
    ) -> Observation:
        _, h, w = get_vision_input_shape(policy)
        image = frame_to_tensor(rgb_frame, h, w, device)

        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = tokens["input_ids"].to(device=device, dtype=torch.long)
        attention_mask = tokens["attention_mask"].to(device=device, dtype=torch.bool)

        wheel_speeds = self.get_wheel_speed_state()
        state = torch.as_tensor(wheel_speeds, device=device, dtype=torch.float32).unsqueeze(0)

        return Observation(
            images={"front": image},
            image_masks={"front": torch.ones(1, device=device, dtype=torch.bool)},
            tokenized_prompt=input_ids,
            tokenized_prompt_mask=attention_mask,
            state=state,
        )

    def rollout(
        self,
        policy: SmolPI,
        reward_model: RewardModel,
        write_to_video: bool = False,
    ):
        
        self.reset_episode_via_reward_model(reward_model)
        
        mujoco.mj_forward(self.model, self.data) # ensure first observation is created

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
            front_rgb, _ = self.render_camera(self.cam_renderer, "front_cam")
            observation = self.make_observation(policy, reward_model.prompt, front_rgb, self.cfg.device)

            if self.cfg.smolpi.precision in (torch.float16, torch.bfloat16):
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
            action_chunk_np = action_chunk[0].to(dtype=torch.float32).cpu().numpy()

            for ctrl_step in range(action_chunk_np.shape[0]):
                torque_cmd = action_chunk_np[ctrl_step]
                torque_cmd = np.clip(torque_cmd, -5.0, 5.0)

                self.data.ctrl[0] = float(torque_cmd[0])
                self.data.ctrl[1] = float(torque_cmd[1])

                for i in range(steps_per_control):
                    mujoco.mj_step(self.model, self.data)

                    step = chunk_step * action_chunk_np.shape[0] * steps_per_control + ctrl_step * steps_per_control + i

                    if step % steps_per_frame == 0 and write_to_video and self.cam_video is not None:
                        front_rgb, front_bgr = self.render_camera(self.cam_renderer, "front_cam")
                        cam_frames.append(front_rgb.copy())
                        self.cam_video.write(front_bgr)

                    if step % steps_per_frame_omni == 0 and write_to_video and self.omni_cam_video is not None:
                        omni_rgb, omni_bgr = self.render_camera(self.omni_cam_renderer, "omniscient_cam")
                        omni_frames.append(omni_rgb.copy())
                        self.omni_cam_video.write(omni_bgr)
                
            reward = reward_model.update(self.data)

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
    
    def close(self):
        if self.cam_video is not None:
            self.cam_video.release()
        if self.omni_cam_video is not None:
            self.omni_cam_video.release()
        
        self.cam_renderer.close()
        self.omni_cam_renderer.close()