from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    @abstractmethod
    def train(self, dataloader):
        ...

    @abstractmethod
    def save_checkpoint(self, path: str):
        ...
    
    @abstractmethod
    def load_checkpoint(self, path: str):
        ...

    @abstractmethod
    def evaluate(self):
        ...