import torch

from core.config import DatasetConfig

DATASET_REGISTRY: dict[str, type] = {}


def register_dataloader(name: str):
    def decorator(func):
        DATASET_REGISTRY[name] = func
        return func

    return decorator


def create_dataloader(cfg: DatasetConfig) -> torch.utils.data.Dataloader:
    dataset_func = DATASET_REGISTRY.get(cfg.name)
    if dataset_func is None:
        raise ValueError(f"Dataset '{cfg.name}' is not registered.")
    return dataset_func(cfg)
