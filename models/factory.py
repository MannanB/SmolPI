from core.config import ModelConfig
from algorithms.base import BaseModel

MODEL_REGISTRY: dict[str, type] = {}

def register_model(name: str):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def create_model(cfg: ModelConfig) -> BaseModel:
    model_cls = MODEL_REGISTRY.get(cfg.name)
    if model_cls is None:
        raise ValueError(f"Model '{cfg.name}' is not registered.")
    return model_cls(cfg)
