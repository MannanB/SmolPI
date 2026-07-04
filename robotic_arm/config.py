from pydantic import BaseModel, ConfigDict
import torch
from model.smolpi import SmolPIConfig



class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    smolpi: SmolPIConfig

    # model hyper params
    flow_steps: int = 10
    flow_std: float = 0.05
    kl_coef: float = 0.05

    # sft specific configs
    epochs: int = 1
    lr: float = 3e-4
    batch_size: int = 12
    grad_accum_steps: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = .01
    use_8bit_adam: bool = False

    dataset: str = "data/bridge_dataset"
    split: str = "train"
    shuffle_buffer: int = 2048
    max_batches: int = 12000

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
