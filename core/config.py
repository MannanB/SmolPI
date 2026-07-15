from dataclasses import dataclass
from pathlib import Path
from core.types import Camera
import torch

@dataclass
class MujocoEnvConfig:
    xml_path: str

    observation_cams: list[Camera]
    render_cams: list[Camera]
    render_env: int = 0
    vid_output_dir: Path = Path("./vids")

    num_parallel_envs: int = 1


@dataclass
class RealRobotConifg:
    ip: str


@dataclass
class SmolPIConfig:
    smolvlm_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    action_expert_id: str = "HuggingFaceTB/SmolLM2-135M"

    action_dim: int = 16
    action_horizon: int = 10
    num_flow_steps: int = 10

    

    precision: torch.dtype = torch.float16
    pytorch_compile_mode: str | None = None
