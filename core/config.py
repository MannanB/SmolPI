from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from core.types import Camera


@dataclass
class EnvironmentConfig:
    name: str


@dataclass
class MujocoEnvConfig(EnvironmentConfig):
    name: ClassVar[str] = "Mujoco"
    xml_path: str

    observation_cams: list[Camera]
    render_cams: list[Camera]
    render_env: int = 0
    vid_output_dir: Path = Path("./vids")

    num_parallel_envs: int = 1


@dataclass
class RealRobotConfig(EnvironmentConfig):
    name: ClassVar[str] = "RealRobot"
    ip: str


@dataclass
class ModelConfig:
    name: str
    action_dim: int
    action_horizon: int

    precision: torch.dtype = torch.float16


@dataclass
class SmolPIConfig(ModelConfig):
    name: ClassVar[str] = "SmolPI"
    smolvlm_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    action_expert_id: str = "HuggingFaceTB/SmolLM2-135M"
    num_flow_steps: int = 10


@dataclass
class DatasetConfig:
    name: str
    dataset_path: Path
    split: str
    batch_size: int

    prefetch_factor: int = 2
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class BridgeDatasetConfig(DatasetConfig):
    name: ClassVar[str] = "Bridge"
    stats_path: Path
    split: str
    shuffle_buffer: int


@dataclass
class AlgorithmConfig:  # shared algo traits
    name: str
    epochs: int = 10
    max_batches_per_epoch: int | None = None

    checkpoint_every_steps: int = 1_000
    checkpoint_dir: Path = Path("checkpoints")

    learning_rate: float = 1e-4


@dataclass
class BehaviorCloningConfig(AlgorithmConfig):
    name: ClassVar[str] = "BehaviorCloning"

    weight_decay: float = 1e-4
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0

    warmup_steps: int = 500
    min_lr_ratio: float = 0.1

    use_amp: bool = True
    use_8bit_adam: bool = False


@dataclass
class FPORLConfig(AlgorithmConfig):
    name: ClassVar[str] = "FPORL"

    prompt: str = ""
    episodes_per_update: int = 8
    rollout_seconds: float = 8.0
    train_epochs: int = 2
    batch_size: int = 8

    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 50.0
    exploration_std: float = 0.5

    awr_temperature: float = 1.0
    max_awr_weight: float = 20.0
    advantage_filter: str = "positive"
    advantage_quantile: float = 0.6

    value_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    value_max_grad_norm: float = 0.5
    value_hidden_dim: int = 256
    value_huber_delta: float = 1.0
    value_target_clip: float = 10.0
    action_clip: float = 5.0

    use_amp: bool = True


@dataclass
class RunConfig:
    model: ModelConfig
    dataset: DatasetConfig
    algorithm: AlgorithmConfig
    environment: EnvironmentConfig

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_wandb: bool = False
    wandb_project: str = "smolpi"
    wandb_run_name: str | None = None


def load_config(config_path: str) -> RunConfig:
    import yaml

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    model_cfg = ModelConfig(**config_dict["model"])
    dataset_cfg = DatasetConfig(**config_dict["dataset"])
    algorithm_cfg = AlgorithmConfig(**config_dict["algorithm"])
    env_cfg = EnvironmentConfig(**config_dict["environment"])

    return RunConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        algorithm=algorithm_cfg,
        environment=env_cfg,
        device=torch.device(
            config_dict.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        ),
        use_wandb=config_dict.get("use_wandb", False),
        wandb_project=config_dict.get("wandb_project", "robot-training"),
        wandb_run_name=config_dict.get("wandb_run_name"),
    )
