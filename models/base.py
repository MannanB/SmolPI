from abc import ABC, abstractmethod

import torch

class BaseModel(ABC):
    @abstractmethod
    def preprocess_observations(self, observations):
        ...

    @abstractmethod
    def sample_actions(self, observation):
        ...

    @abstractmethod
    def bc_loss(self, observation, actual_action):
        ...