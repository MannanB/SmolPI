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
    batch_size: int = 16
    grad_accum_steps: int = 2
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = .01
    use_8bit_adam: bool = False

    dataset: str = "data/bridge_dataset"
    split: str = "train"
    shuffle_buffer: int = 2048
    prefetch_batches: int = 8
    max_batches: int = 20000

    # Periodic artifacts. Set either interval to 0 to disable it.
    checkpoint_every_batches: int = 1000
    video_every_batches: int = 1000
    checkpoint_dir: str = "checkpoints"
    video_dir: str = "vids"

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
