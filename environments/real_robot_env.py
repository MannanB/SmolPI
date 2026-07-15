from environments.base import BaseEnvironment
from core.config import RealRobotEnvConfig

# placeholder for real robot (wip)

class RealRobotEnv(BaseEnvironment):
    def __init__(self, config: RealRobotEnvConfig):
        self.config = config


