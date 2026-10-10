"""Independent C++ QP inverse kinematics for the Fairino wheel-legged dual-arm robot."""

from .runtime import _preload_native_dependencies

_preload_native_dependencies()

from .chassis import (
    ChassisCommand,
    ChassisController,
    ChassisLimits,
    base_goal_for_hands,
    body_twist,
    integrate_twist,
    planar_matrix,
    robot_to_world,
    world_to_robot,
)
from .solver import (
    ARM_JOINTS,
    ASSET_ROOT,
    DEFAULT_BASE_POSE,
    DEFAULT_LEFT_FRAME,
    DEFAULT_RIGHT_FRAME,
    DEFAULT_URDF,
    FairinoConfigurationError,
    FairinoIKError,
    FairinoQPModel,
    HEAD_JOINT,
    IKOptions,
    IKResult,
    JOINT_GROUPS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    TORSO_JOINTS,
    URDF_BASE_POSE,
    WHEEL_JOINTS,
    available_qp_solvers,
    solve_dual_arm_ik,
)

__version__ = "0.1.0"

__all__ = [
    "ChassisCommand",
    "ChassisController",
    "ChassisLimits",
    "base_goal_for_hands",
    "body_twist",
    "integrate_twist",
    "planar_matrix",
    "robot_to_world",
    "world_to_robot",
    "ARM_JOINTS",
    "ASSET_ROOT",
    "DEFAULT_BASE_POSE",
    "DEFAULT_LEFT_FRAME",
    "DEFAULT_RIGHT_FRAME",
    "DEFAULT_URDF",
    "FairinoConfigurationError",
    "FairinoIKError",
    "FairinoQPModel",
    "HEAD_JOINT",
    "IKOptions",
    "IKResult",
    "JOINT_GROUPS",
    "LEFT_ARM_JOINTS",
    "RIGHT_ARM_JOINTS",
    "TORSO_JOINTS",
    "URDF_BASE_POSE",
    "WHEEL_JOINTS",
    "available_qp_solvers",
    "solve_dual_arm_ik",
]
