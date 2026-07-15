from core.config import EnvironmentConfig
from environments.base import BaseEnvironment

ENVIRONMENT_REGISTRY: dict[str, type] = {}

def register_environment(name: str):
    def decorator(cls):
        ENVIRONMENT_REGISTRY[name] = cls
        return cls
    return decorator

def create_environment(cfg: EnvironmentConfig) -> BaseEnvironment:
    env_cls = ENVIRONMENT_REGISTRY.get(cfg.name)
    if env_cls is None:
        raise ValueError(f"Environment '{cfg.name}' is not registered.")
    return env_cls(cfg)
