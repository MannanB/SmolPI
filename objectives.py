# different prompts, with different reward functions

import math

import numpy as np


PLATFORM_TARGETS = {
    "red": np.array([1.4, 1.4], dtype=np.float32),
    "blue": np.array([1.4, -1.4], dtype=np.float32),
    "green": np.array([-1.4, 1.4], dtype=np.float32),
    "pink": np.array([-1.4, -1.4], dtype=np.float32),
}

ROBOT_Z = 0.18
PLATFORM_RADIUS = 0.35
DEFAULT_RESET_HALF_EXTENT = 0.7
PLATFORM_RESET_HALF_EXTENT = 2.15


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_to_quat(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * yaw
    return np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float64)


def _quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _position(data) -> np.ndarray:
    return np.array([float(data.qpos[0]), float(data.qpos[1])], dtype=np.float32)


def _yaw(data) -> float:
    if data.qpos.shape[0] >= 7:
        return _quat_to_yaw(data.qpos[3:7])
    if data.qpos.shape[0] >= 3:
        return float(data.qpos[2])
    return 0.0


def _heading(data) -> np.ndarray:
    yaw = _yaw(data)
    return np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)


def _set_pose(data, xy: np.ndarray, yaw: float) -> None:
    data.qpos[0] = float(xy[0])
    data.qpos[1] = float(xy[1])

    if data.qpos.shape[0] >= 7:
        data.qpos[2] = ROBOT_Z
        data.qpos[3:7] = _yaw_to_quat(yaw)
    elif data.qpos.shape[0] >= 3:
        data.qpos[2] = yaw

    data.qvel[:] = 0.0
    if hasattr(data, "ctrl"):
        data.ctrl[:] = 0.0


def _random_xy(half_extent: float, min_distance_from: np.ndarray | None = None, min_distance: float = 0.0) -> np.ndarray:
    for _ in range(100):
        xy = np.random.uniform(-half_extent, half_extent, size=2).astype(np.float32)
        if min_distance_from is None or np.linalg.norm(xy - min_distance_from) >= min_distance:
            return xy
    return np.random.uniform(-half_extent, half_extent, size=2).astype(np.float32)


def _reset_random_pose(
    data,
    half_extent: float = DEFAULT_RESET_HALF_EXTENT,
    min_distance_from: np.ndarray | None = None,
    min_distance: float = 0.0,
) -> None:
    xy = _random_xy(half_extent, min_distance_from=min_distance_from, min_distance=min_distance)
    yaw = float(np.random.uniform(-math.pi, math.pi))
    _set_pose(data, xy, yaw)


class RewardModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prompt = "Drive forward"

    def init_rollout(self, data):
        _reset_random_pose(data)

    def update(self, data):
        reward = 0.0
        return reward


class MoveForwardRewardModel(RewardModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.initial_pos = None
        self.initial_heading = None
        self.previous_progress = 0.0
        self.prompt = "Drive forward"

    def init_rollout(self, data):
        _reset_random_pose(data)
        self.initial_pos = _position(data)
        self.initial_heading = _heading(data)
        self.previous_progress = 0.0

    def update(self, data):
        progress = float(np.dot(_position(data) - self.initial_pos, self.initial_heading))
        reward = progress - self.previous_progress
        self.previous_progress = progress
        return reward


class MoveBackwardRewardModel(MoveForwardRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.prompt = "Drive backward"

    def update(self, data):
        progress = -float(np.dot(_position(data) - self.initial_pos, self.initial_heading))
        reward = progress - self.previous_progress
        self.previous_progress = progress
        return reward


class SpinRewardModel(RewardModel):
    def __init__(self, cfg, direction: float, prompt: str):
        super().__init__(cfg)
        self.direction = direction
        self.previous_yaw = 0.0
        self.prompt = prompt

    def init_rollout(self, data):
        _reset_random_pose(data)
        self.previous_yaw = _yaw(data)

    def update(self, data):
        current_yaw = _yaw(data)
        yaw_delta = _wrap_angle(current_yaw - self.previous_yaw)
        translation_penalty = 0.03 * float(np.linalg.norm(data.qvel[:2])) if data.qvel.shape[0] >= 2 else 0.0
        self.previous_yaw = current_yaw
        return self.direction * yaw_delta - translation_penalty


class SpinCounterClockwiseRewardModel(SpinRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg, direction=1.0, prompt="Spin counter-clockwise in place")


class SpinClockwiseRewardModel(SpinRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg, direction=-1.0, prompt="Spin clockwise in place")


class MoveToPlatformRewardModel(RewardModel):
    def __init__(self, cfg, color: str):
        super().__init__(cfg)
        self.color = color
        self.target = PLATFORM_TARGETS[color]
        self.previous_distance = 0.0
        self.prompt = f"Drive to the {color} platform"

    def init_rollout(self, data):
        _reset_random_pose(
            data,
            half_extent=PLATFORM_RESET_HALF_EXTENT,
            min_distance_from=self.target,
            min_distance=0.7,
        )
        self.previous_distance = float(np.linalg.norm(_position(data) - self.target))

    def update(self, data):
        distance = float(np.linalg.norm(_position(data) - self.target))
        progress_reward = self.previous_distance - distance
        platform_bonus = 0.0
        if distance <= PLATFORM_RADIUS:
            platform_bonus = 0.25 * (1.0 - distance / PLATFORM_RADIUS)
        self.previous_distance = distance
        return 2.0 * progress_reward + platform_bonus


class MoveToRedPlatformRewardModel(MoveToPlatformRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg, "red")


class MoveToGreenPlatformRewardModel(MoveToPlatformRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg, "green")


class MoveToPinkPlatformRewardModel(MoveToPlatformRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg, "pink")


class MoveToBluePlatformRewardModel(MoveToPlatformRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg, "blue")


class FacePlatformRewardModel(RewardModel):
    def __init__(self, cfg, color: str):
        super().__init__(cfg)
        self.color = color
        self.target = PLATFORM_TARGETS[color]
        self.previous_score = 0.0
        self.start_pos = None
        self.prompt = f"Turn to face the {color} platform"

    def init_rollout(self, data):
        _reset_random_pose(
            data,
            half_extent=PLATFORM_RESET_HALF_EXTENT,
            min_distance_from=self.target,
            min_distance=0.7,
        )
        self.start_pos = _position(data)
        self.previous_score = self._score(data)

    def _score(self, data) -> float:
        delta = self.target - _position(data)
        bearing = math.atan2(float(delta[1]), float(delta[0]))
        error = abs(_wrap_angle(bearing - _yaw(data)))
        return math.cos(error)

    def update(self, data):
        score = self._score(data)
        reward = score - self.previous_score
        reward += 0.05 * max(score, 0.0)
        reward -= 0.02 * float(np.linalg.norm(_position(data) - self.start_pos))
        self.previous_score = score
        return reward


class FaceRedPlatformRewardModel(FacePlatformRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg, "red")


class FaceGreenPlatformRewardModel(FacePlatformRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg, "green")


class FacePinkPlatformRewardModel(FacePlatformRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg, "pink")


class FaceBluePlatformRewardModel(FacePlatformRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg, "blue")


class MoveToCenterRewardModel(RewardModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.previous_distance = 0.0
        self.prompt = "Drive to the center between the four colored platforms"

    def init_rollout(self, data):
        _reset_random_pose(data, half_extent=PLATFORM_RESET_HALF_EXTENT, min_distance_from=self.target, min_distance=0.8)
        self.previous_distance = float(np.linalg.norm(_position(data) - self.target))

    def update(self, data):
        distance = float(np.linalg.norm(_position(data) - self.target))
        reward = 1.5 * (self.previous_distance - distance)
        if distance <= 0.3:
            reward += 0.2 * (1.0 - distance / 0.3)
        self.previous_distance = distance
        return reward


class MoveToAnyPlatformRewardModel(RewardModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.previous_distance = 0.0
        self.prompt = "Drive onto any colored platform"

    def init_rollout(self, data):
        _reset_random_pose(data, half_extent=PLATFORM_RESET_HALF_EXTENT)
        self.previous_distance = self._nearest_distance(data)

    def _nearest_distance(self, data) -> float:
        xy = _position(data)
        return float(min(np.linalg.norm(xy - target) for target in PLATFORM_TARGETS.values()))

    def update(self, data):
        distance = self._nearest_distance(data)
        reward = 2.0 * (self.previous_distance - distance)
        if distance <= PLATFORM_RADIUS:
            reward += 0.25 * (1.0 - distance / PLATFORM_RADIUS)
        self.previous_distance = distance
        return reward


class OrbitPlatformsRewardModel(RewardModel):
    def __init__(self, cfg, direction: float, prompt: str):
        super().__init__(cfg)
        self.direction = direction
        self.previous_angle = 0.0
        self.prompt = prompt

    def init_rollout(self, data):
        _reset_random_pose(data, half_extent=PLATFORM_RESET_HALF_EXTENT, min_distance_from=np.zeros(2), min_distance=0.9)
        pos = _position(data)
        self.previous_angle = math.atan2(float(pos[1]), float(pos[0]))

    def update(self, data):
        pos = _position(data)
        angle = math.atan2(float(pos[1]), float(pos[0]))
        angular_progress = self.direction * _wrap_angle(angle - self.previous_angle)
        radius = float(np.linalg.norm(pos))
        radius_penalty = 0.02 * abs(radius - 1.9)
        self.previous_angle = angle
        return angular_progress - radius_penalty


class OrbitPlatformsCounterClockwiseRewardModel(OrbitPlatformsRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg, direction=1.0, prompt="Drive around the outside of the platforms counter-clockwise")


class OrbitPlatformsClockwiseRewardModel(OrbitPlatformsRewardModel):
    def __init__(self, cfg):
        super().__init__(cfg, direction=-1.0, prompt="Drive around the outside of the platforms clockwise")


SpinCCWRewardModel = SpinCounterClockwiseRewardModel
SpinCWRewardModel = SpinClockwiseRewardModel

OBJECTIVE_CLASSES = [
    # MoveForwardRewardModel,
    # MoveBackwardRewardModel,
    # SpinCounterClockwiseRewardModel,
    # SpinClockwiseRewardModel,
    MoveToRedPlatformRewardModel,
    MoveToGreenPlatformRewardModel,
    MoveToPinkPlatformRewardModel,
    MoveToBluePlatformRewardModel,
    # FaceRedPlatformRewardModel,
    # FaceGreenPlatformRewardModel,
    # FacePinkPlatformRewardModel,
    # FaceBluePlatformRewardModel,
    # MoveToCenterRewardModel,
    # MoveToAnyPlatformRewardModel,
    # OrbitPlatformsCounterClockwiseRewardModel,
    # OrbitPlatformsClockwiseRewardModel,
]


def make_random_objective(cfg) -> RewardModel:
    return np.random.choice(OBJECTIVE_CLASSES)(cfg)
