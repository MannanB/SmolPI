from abc import ABC, abstractmethod

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
    def rollout(self, seconds, action_horizon): ...

    @abstractmethod
    def close(self): ...
