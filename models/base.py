from abc import ABC, abstractmethod

import torch


class BaseModel(torch.nn.Module, ABC):
    @abstractmethod
    def preprocess_observations(self, observations): ...

    @abstractmethod
    def sample_actions(self, observations): ...

    @abstractmethod
    def bc_loss(self, observations, actual_action): ...
