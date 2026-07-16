from core.config import RealRobotConfig
from environments.base import BaseEnvironment
from environments.factory import register_environment

# placeholder for real robot (wip)


@register_environment("RealRobot")
class RealRobotEnv(BaseEnvironment):
    def __init__(self, config: RealRobotConfig):
        self.config = config
