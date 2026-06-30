from pydantic import BaseModel, ConfigDict
import torch
from model.smolpi import SmolPIConfig

'''    parser.add_argument("--dataset", type=Path, default=Path("data/bridge_dataset"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--shuffle-buffer", type=int, default=2048)
    parser.add_argument("--max-batches", type=int, default=None)'''

class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    smolpi: SmolPIConfig

    # model hyper params
    flow_steps: int = 24
    flow_std: float = 0.05
    kl_coef: float = 0.05

    # sft specific configs
    epochs: int = 10
    lr: float = 1e-4
    batch_size: int = 32
    grad_accum_steps: int = 12
    weight_decay: float = 1e-4
    use_8bit_adam: bool = False

    dataset: str = "data/bridge_dataset"
    split: str = "train"
    shuffle_buffer: int = 2048
    max_batches: int = 10000

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
