from environments.base import BaseEnvironment
from environments.factory import register_environment
from core.config import RealRobotConfig
from core.types import EnvObservation

# placeholder for real robot (wip)

@register_environment("RealRobot")
class RealRobotEnv(BaseEnvironment):
    def __init__(self, config: RealRobotConfig):
        self.config = config


