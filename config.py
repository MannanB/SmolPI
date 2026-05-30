from pydantic import BaseModel, ConfigDict
import torch
from model.smolpi import SmolPIConfig

# PROMPT = "Drive forward"
# CAM_SIM_WIDTH, CAM_SIM_HEIGHT = 640, 480
# CAM_OMNI_WIDTH, CAM_OMNI_HEIGHT = 1280, 960
# CAM_SIM_FPS = 30
# CAM_OMNI_FPS = 60
# SIM_DURATION_SEC = 4
# CONTROL_FREQ_HZ = 25 # Control frequency for the policy (e.g., 10 Hz means the policy outputs new actions every 0.1 seconds).
# NUM_EPISODES_PER_UPDATE = 2 # Number of parallel episodes to run for each policy update (if doing training).
# REWARD_SCALE = 0.5
# NUM_UPDATES = 1000
# KL_COEF = 0.02


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    smolpi: SmolPIConfig

    world_xml_path: str = "world/platform_world.xml"
    cam_sim_width: int = 640
    cam_sim_height: int = 480
    cam_omni_width: int = 1280
    cam_omni_height: int = 960
    cam_sim_fps: int = 30
    cam_omni_fps: int = 60
    cam_sim_output_path: str = "vids/front_cam_vla_rollout.mp4"
    cam_omni_output_path: str = "vids/omniscient_vla_rollout.mp4"

    sim_duration_sec: int = 8
    control_freq_hz: int = 20
    num_episodes_per_update: int = 10
    num_parallel_rollouts: int = 10
    num_updates: int = 1000
    batch_size: int = 40
    replay_buffer_capacity: int = 1000

    flow_steps: int = 10
    flow_std: float = 0.25
    kl_coef: float = 0.02

    lr: float = 1e-5
    weight_decay: float = 1e-4

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
