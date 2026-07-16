from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from core.types import EnvObservation

import cv2


def make_writer(path, width, height, fps):
    return cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def step(self, action): ...

    @abstractmethod
    def rollout(
        self,
        seconds: float,
        action_horizon: int,
        get_action: Callable[[list[EnvObservation]], torch.Tensor],
    ) -> list[list[float]]: ...

    @abstractmethod
    def close(self): ...
