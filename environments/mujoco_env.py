import os
from abc import ABC, abstractmethod
from collections.abc import Callable

import cv2
import mujoco
import numpy as np
import torch

from core.config import MujocoEnvConfig
from core.types import EnvObservation
from environments.base import BaseEnvironment, make_writer
from environments.factory import register_environment

os.environ.setdefault("MUJOCO_GL", "glfw")


def frame_to_tensor(rgb_frame: np.ndarray) -> torch.Tensor:
    frame = torch.from_numpy(rgb_frame).to("cpu")
    return frame


@register_environment("Mujoco")
class MujocoEnvironment(BaseEnvironment, ABC):
    def __init__(self, config: MujocoEnvConfig, control_freq_hz: float):
        self.config = config

        xml_str = ""
        with open(self.config.xml_path) as f:
            xml_str = f.read()

        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.datas = [mujoco.MjData(self.model) for _ in range(self.config.num_parallel_envs)]

        self.ctrl_freq = control_freq_hz
        self.steps_per_control = max(1, int(1 / self.ctrl_freq / self.model.opt.timestep))

        self.obs_renderers = [
            mujoco.Renderer(self.model, width=cam.width, height=cam.height)
            for cam in self.config.observation_cams
        ]

        # TODO: rename for clarity? these are only for rendering to mp4, not observations
        self.cam_renderers = [
            mujoco.Renderer(self.model, width=cam.width, height=cam.height)
            for cam in self.config.render_cams
        ]
        self.cam_writers = [
            make_writer(self.config.vid_output_dir / cam.name, cam.width, cam.height, cam.fps)
            for cam in self.config.render_cams
        ]
        self.steps_per_frames = [
            int(1 / cam.fps / self.model.opt.timestep) for cam in self.cam_renderers
        ]

        self.render_cameras = False

    @abstractmethod
    def reset_data(self, data: mujoco.MjData):
        # for specific mujoco envs to apply random start positions
        ...

    def reset(self):
        for data in self.datas:
            mujoco.mj_resetData(self.model, data)
            self.reset_data(data)
            mujoco.mj_forward(self.model, data)

    @abstractmethod
    def control_robot(self, action: torch.Tensor, data: mujoco.MjData): ...

    def collect_camera(self, data: mujoco.MjData) -> list[torch.Tensor]:
        images = []
        for i in range(len(self.obs_renderers)):
            self.obs_renderers[i].update_scene(data, camera=self.config.observation_cams[i].name)
            rgb = self.observation_cams[i].render()
            images.append(frame_to_tensor(rgb))
        return images

    @abstractmethod
    def get_robot_state(self, data: mujoco.MjData) -> torch.Tensor: ...

    def get_observations(self) -> list[EnvObservation]:
        observations = []
        for i in range(len(self.datas)):
            camera_obs = self.collect_camera(self.data[i])
            state = self.get_robot_state(self.data[i])
            observations.append(EnvObservation(images=camera_obs, robot_state=state))
        return observations

    def start_camera_rendering(self):
        self.render_cameras = True

    def end_camera_rendering(self):
        for renderer in self.cam_renderers:
            renderer.close()
        for writer in self.cam_writers:
            writer.release()
        self.render_cameras = False

    def render_camera_frame(self, data: mujoco.MjData, step_n: int):
        for i in range(len(self.cam_renderers)):
            if step_n % self.steps_per_frames[i] == 0:
                self.cam_renderers[i].update_scene(data, camera=self.config.render_cams[i].name)
                rgb = self.cam_renderers[i].render()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self.cam_writers[i].write(bgr)

    @abstractmethod
    def get_reward(self, data: mujoco.MjData) -> float: ...

    def step(
        self,
        actions: list[torch.Tensor] | torch.Tensor,
        env_idx: int = -1,
        step_n: int = None,
    ) -> EnvObservation | list[EnvObservation]:
        if env_idx == -1:
            observations = []
            for i in range(len(self.datas)):
                observations.append(self.step(actions[i], i))
            return observations

        self.control_robot(actions, self.datas[env_idx])
        camera_obs = self.collect_camera(self.data[env_idx])
        state = self.get_robot_state(self.data[env_idx])

        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.datas[env_idx])
            if self.render_cameras:
                self.render_camera_frame(self.datas[env_idx], step_n)
            step_n += 1

        return EnvObservation(images=camera_obs, robot_state=state)

    def rollout(self, seconds: float, action_horizon: int, get_action: Callable):
        self.reset()

        sim_steps = int(seconds / self.model.opt.timestep)

        num_chunks = sim_steps // (self.steps_per_control * action_horizon)

        obs = self.get_observations()

        rewards = []

        for chunk in range(num_chunks):
            actions = get_action(obs)
            for i in range(action_horizon):
                obs = self.step(
                    actions[:, i],
                    step_n=chunk * action_horizon + i * self.steps_per_control,
                )

            rewards.append([self.get_reward(data) for data in self.datas])
        return rewards

    def close(self):
        self.end_camera_rendering()
        for renderer in self.obs_renderers:
            renderer.close()
