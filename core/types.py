from dataclasses import dataclass

import torch


@dataclass
class EnvObservation:
    images: list[torch.Tensor]
    robot_state: torch.Tensor


@dataclass
class Observation:
    env: EnvObservation
    prompt: str


@dataclass
class Camera:
    name: str
    width: int
    height: int
    fps: int
