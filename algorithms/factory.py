from algorithms.base import BaseAlgorithm
from core.config import AlgorithmConfig

ALGORITHM_REGISTRY: dict[str, type] = {}


def register_algorithm(name: str):
    def decorator(cls):
        ALGORITHM_REGISTRY[name] = cls
        return cls

    return decorator


def create_algorithm(cfg: AlgorithmConfig, *args, **kwargs) -> BaseAlgorithm:
    algorithm_cls = ALGORITHM_REGISTRY.get(cfg.name)
    if algorithm_cls is None:
        raise ValueError(f"Algorithm '{cfg.name}' is not registered.")
    return algorithm_cls(cfg, *args, **kwargs)
