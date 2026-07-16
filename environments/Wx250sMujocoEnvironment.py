import mujoco
import numpy as np
import torch

from environments.factory import register_environment
from environments.mujoco_env import MujocoEnvironment

ARM_JOINTS = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)

HOME_QPOS = np.array(
    [0.0, -0.712953, 0.501707, 0.0, 0.996644, 0.0, 0.015, -0.015],
    dtype=np.float64,
)

HOME_CTRL = np.array(
    [0.0, -0.712953, 0.501707, 0.0, 0.996644, 0.0, 0.015],
    dtype=np.float64,
)

TASK_OBJECT_POSITIONS = {
    "pick-red": {
        "red_box": (0.34, 0.00),
        "blue_box": (0.34, -0.08),
        "green_box": (0.34, 0.08),
    },
    "push-red": {
        "blue_box": (0.27, -0.10),
        "green_box": (0.43, -0.10),
        "red_box": (0.34, 0.10),
    },
    "push-blue-to-green": {
        "blue_box": (0.27, -0.10),
        "green_box": (0.43, -0.10),
        "red_box": (0.34, 0.10),
    },
}


def rotation_matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))

    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = np.arctan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0

    return np.array([roll, pitch, yaw], dtype=np.float64)


def euler_xyz_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array(
        [
            [
                cy * cp,
                cy * sp * sr - sy * cr,
                cy * sp * cr + sy * sr,
            ],
            [
                sy * cp,
                sy * sp * sr + cy * cr,
                sy * sp * cr - cy * sr,
            ],
            [
                -sp,
                cp * sr,
                cp * cr,
            ],
        ],
        dtype=np.float64,
    )


@register_environment("Wx250s")
class Wx250sEnvironment(MujocoEnvironment):
    """
    WX250s MuJoCo environment using Bridge-style actions:

        action[:3]  = end-effector position delta
        action[3:6] = end-effector Euler-angle delta
        action[6]   = gripper command in [0, 1]
    """

    @property
    def data(self) -> list[mujoco.MjData]:
        """
        Compatibility alias for MujocoEnvironment methods that reference
        self.data instead of self.datas.
        """
        return self.datas

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)

        if object_id < 0:
            raise ValueError(f"WX250s scene is missing required MuJoCo object {name!r}")

        return object_id

    def _initialize_robot_metadata(self) -> None:
        if hasattr(self, "_wx250s_initialized"):
            return

        self.ee_site_id = self._id(
            mujoco.mjtObj.mjOBJ_SITE,
            "ee_site",
        )
        self.worktop_id = self._id(
            mujoco.mjtObj.mjOBJ_GEOM,
            "worktop",
        )

        self.arm_joint_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in ARM_JOINTS],
            dtype=np.int32,
        )

        self.arm_qpos_addresses = self.model.jnt_qposadr[self.arm_joint_ids].copy()
        self.arm_dof_addresses = self.model.jnt_dofadr[self.arm_joint_ids].copy()
        self.arm_joint_ranges = self.model.jnt_range[self.arm_joint_ids].copy()

        self.ik_data = mujoco.MjData(self.model)
        self._wx250s_initialized = True

    def _get_work_surface_z(self, data: mujoco.MjData) -> float:
        return float(data.geom_xpos[self.worktop_id, 2] + self.model.geom_size[self.worktop_id, 2])

    def _get_task_name(self) -> str:
        return getattr(self.config, "task", "pick-red")

    def _reset_objects(self, data: mujoco.MjData) -> None:
        task_name = self._get_task_name()

        if task_name not in TASK_OBJECT_POSITIONS:
            raise ValueError(
                f"Unknown WX250s task {task_name!r}. "
                f"Expected one of {sorted(TASK_OBJECT_POSITIONS)}."
            )

        work_surface_z = self._get_work_surface_z(data)

        for body_name, (x, y) in TASK_OBJECT_POSITIONS[task_name].items():
            joint_id = self._id(
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{body_name}_freejoint",
            )
            geom_id = self._id(
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{body_name}_geom",
            )

            qpos_address = self.model.jnt_qposadr[joint_id]
            object_z = work_surface_z + self.model.geom_size[geom_id, 2]

            data.qpos[qpos_address : qpos_address + 3] = (
                x,
                y,
                object_z,
            )

    def reset_data(self, data: mujoco.MjData) -> None:
        self._initialize_robot_metadata()

        if self.model.nq < len(HOME_QPOS):
            raise ValueError(
                f"WX250s model has nq={self.model.nq}, "
                f"but the home configuration requires at least "
                f"{len(HOME_QPOS)} qpos values."
            )

        if self.model.nu < len(HOME_CTRL):
            raise ValueError(
                f"WX250s model has nu={self.model.nu}, "
                f"but the home configuration requires at least "
                f"{len(HOME_CTRL)} controls."
            )

        data.qpos[: len(HOME_QPOS)] = HOME_QPOS
        data.ctrl[: len(HOME_CTRL)] = HOME_CTRL

        mujoco.mj_forward(self.model, data)
        self._reset_objects(data)

    def _solve_ik(
        self,
        action: np.ndarray,
        data: mujoco.MjData,
    ) -> np.ndarray:
        self._initialize_robot_metadata()

        self.ik_data.qpos[:] = data.qpos
        self.ik_data.qvel[:] = data.qvel
        self.ik_data.ctrl[:] = data.ctrl

        mujoco.mj_forward(self.model, self.ik_data)

        start_position = self.ik_data.site_xpos[self.ee_site_id].copy()
        start_rotation = self.ik_data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

        target_position = start_position + np.clip(
            action[:3],
            -0.04,
            0.04,
        )

        work_surface_z = self._get_work_surface_z(self.ik_data)
        target_position = np.clip(
            target_position,
            [0.10, -0.32, work_surface_z + 0.005],
            [0.52, 0.32, 0.48],
        )

        start_euler = rotation_matrix_to_euler_xyz(start_rotation)
        target_euler = start_euler + np.clip(
            action[3:6],
            -0.25,
            0.25,
        )

        target_rotation = euler_xyz_to_rotation_matrix(target_euler)
        target_quaternion = np.empty(4, dtype=np.float64)

        mujoco.mju_mat2Quat(
            target_quaternion,
            target_rotation.reshape(-1),
        )

        jacobian_position = np.zeros(
            (3, self.model.nv),
            dtype=np.float64,
        )
        jacobian_rotation = np.zeros(
            (3, self.model.nv),
            dtype=np.float64,
        )
        current_quaternion = np.empty(4, dtype=np.float64)
        rotation_error = np.empty(3, dtype=np.float64)

        for _ in range(30):
            current_position = self.ik_data.site_xpos[self.ee_site_id]
            current_rotation = self.ik_data.site_xmat[self.ee_site_id]

            mujoco.mju_mat2Quat(
                current_quaternion,
                current_rotation,
            )
            mujoco.mju_subQuat(
                rotation_error,
                target_quaternion,
                current_quaternion,
            )

            error = np.concatenate(
                [
                    target_position - current_position,
                    rotation_error,
                ]
            )

            if np.linalg.norm(error[:3]) < 5e-4 and np.linalg.norm(error[3:]) < 2e-3:
                break

            mujoco.mj_jacSite(
                self.model,
                self.ik_data,
                jacobian_position,
                jacobian_rotation,
                self.ee_site_id,
            )

            jacobian = np.vstack([jacobian_position, jacobian_rotation])[:, self.arm_dof_addresses]

            damping = 0.025
            system = jacobian @ jacobian.T + damping**2 * np.eye(6)

            update = jacobian.T @ np.linalg.solve(
                system,
                error,
            )

            update_norm = np.linalg.norm(update)
            if update_norm > 0.18:
                update *= 0.18 / update_norm

            next_qpos = self.ik_data.qpos[self.arm_qpos_addresses] + update

            self.ik_data.qpos[self.arm_qpos_addresses] = np.clip(
                next_qpos,
                self.arm_joint_ranges[:, 0],
                self.arm_joint_ranges[:, 1],
            )

            mujoco.mj_forward(self.model, self.ik_data)

        return self.ik_data.qpos[self.arm_qpos_addresses].copy()

    def control_robot(
        self,
        action: torch.Tensor,
        data: mujoco.MjData,
    ) -> None:
        self._initialize_robot_metadata()

        action_np = action.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(-1)

        if action_np.shape != (7,):
            raise ValueError(f"Expected a 7D Bridge action, got {action_np.shape}.")

        if not np.isfinite(action_np).all():
            raise ValueError(f"WX250s action contains non-finite values: {action_np}")

        data.ctrl[:6] = self._solve_ik(action_np, data)
        data.ctrl[6] = 0.037 if action_np[6] >= 0.5 else 0.015

    def get_robot_state(
        self,
        data: mujoco.MjData,
    ) -> torch.Tensor:
        self._initialize_robot_metadata()

        position = data.site_xpos[self.ee_site_id].copy()
        rotation = data.site_xmat[self.ee_site_id].reshape(3, 3)

        euler = rotation_matrix_to_euler_xyz(rotation)

        gripper = np.clip(
            (data.qpos[6] - 0.015) / (0.037 - 0.015),
            0.0,
            1.0,
        )

        state = np.concatenate(
            [
                position,
                euler,
                [gripper],
            ]
        ).astype(np.float32)

        return torch.from_numpy(state)
