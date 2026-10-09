"""Thin Python interfaces for the shared C++ Pinocchio / eiquadprog IK kernel."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np


ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "fairino"
DEFAULT_URDF = ASSET_ROOT / "fairino_wheel_legged_robot.urdf"

# The exported URDF names every joint jN.  These groups are this package's
# convention; the C++ kernel uses the same names and order.
LEFT_ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6", "j7")
RIGHT_ARM_JOINTS = ("j8", "j9", "j10", "j11", "j12", "j13", "j14")
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
# Root first: lower leg, knee, hip, waist, body yaw.
TORSO_JOINTS = ("j20", "j19", "j18", "j16", "j15")
HEAD_JOINT = "j17"
WHEEL_JOINTS = ("j21", "j22", "j23", "j24", "j25", "j26", "j27", "j28")
JOINT_GROUPS = {
    "left_arm": LEFT_ARM_JOINTS,
    "right_arm": RIGHT_ARM_JOINTS,
    "torso": TORSO_JOINTS,
    "head": (HEAD_JOINT,),
    "wheels": WHEEL_JOINTS,
}

DEFAULT_LEFT_FRAME = "left_waist2_Link"
DEFAULT_RIGHT_FRAME = "right_waist2_Link"

# The SolidWorks-exported base_link faces -y with the left arm on +x.  The
# public world frame turns its axes 90 degrees clockwise about z (seen from
# above) so that x points forward, y left and z up; the origin is unchanged.
# Pass ``base_pose=URDF_BASE_POSE`` to work in raw base_link coordinates.
DEFAULT_BASE_POSE = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
DEFAULT_BASE_POSE.setflags(write=False)
URDF_BASE_POSE = np.eye(4)
URDF_BASE_POSE.setflags(write=False)


class FairinoIKError(RuntimeError):
    """Base class for Fairino IK errors."""


class FairinoConfigurationError(FairinoIKError, ValueError):
    """Raised when the model, target pose, or options are invalid."""


@dataclass
class IKOptions:
    """Native QP and fixed-joint settings for :meth:`FairinoQPModel.solve_ik`.

    Pose targets are homogeneous matrices in the model's world frame: by
    default x forward, y left, z up, with the origin at ``base_link``
    (see ``DEFAULT_BASE_POSE``).
    Position tolerances and costs use metres; orientation tolerances and costs
    use radians.  ``posture_target`` follows the model's public order (left
    seven, right seven, then the torso when it is active).  When posture
    regularization is enabled without an explicit target, the seed becomes the
    target so successive solves favour continuous solutions.

    In arm-only mode ``torso_position`` (``TORSO_JOINTS`` order) locks the
    torso; it changes where the arms are mounted, so FK, IK and the full
    configuration all use it.  With ``FairinoQPModel(active_torso=True)`` the
    torso comes from ``q`` instead and ``torso_position`` is ignored.
    ``head_position`` is always locked.
    """

    max_iterations: int = 500
    qp_max_iterations: int = 100
    body_avoidance: bool = True
    body_activation_distance: float = 0.04
    body_min_distance: float = 0.01
    body_link_radius: float = 0.06
    body_avoidance_weight: float = 4.0
    body_avoidance_gain: float = 1.0
    tolerance: float = 1e-5
    position_tolerance: float | None = None
    orientation_tolerance: float | None = None

    dt: float = 0.02
    damping: float = 1e-5
    position_cost: float = 1.0
    orientation_cost: float = 1.0
    frame_gain: float = 1.0
    step_size: float = 1.0
    max_step_norm: float = 0.15
    # The URDF velocity limits are all zero; these speeds (rad/s) replace them.
    default_velocity_limit: float = float(np.pi)
    torso_velocity_limit: float = 0.5
    # Active-torso mode: try the arms alone for this many iterations so the
    # torso stays put whenever the arms can reach; then free the torso.  0
    # frees it from the start.  torso_step_weight is an extra cost on torso
    # steps in that phase (it slows convergence near arm singularities).
    arms_first_iterations: int = 50
    torso_step_weight: float = 0.0

    posture_weight: float = 0.0
    posture_gain: float = 1.0
    posture_target: np.ndarray | None = None

    qp_solver: str | None = None

    # Locked joint values (torso only in arm-only mode, head always).
    torso_position: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    head_position: float = 0.0

    def validate(self) -> None:
        """Validate values in place and normalize array-like fields."""

        for name, minimum in (("max_iterations", 1), ("qp_max_iterations", 1),
                              ("arms_first_iterations", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < minimum:
                raise FairinoConfigurationError(f"{name} must be an integer >= {minimum}")
            setattr(self, name, int(value))

        non_negative = (
            "damping",
            "torso_step_weight",
            "position_cost",
            "orientation_cost",
            "posture_weight",
        )
        for name in non_negative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise FairinoConfigurationError(f"{name} must be finite and non-negative")
            setattr(self, name, value)

        positive = ("tolerance", "dt", "max_step_norm", "default_velocity_limit",
                    "torso_velocity_limit")
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise FairinoConfigurationError(f"{name} must be finite and positive")
            setattr(self, name, value)

        if self.position_cost == 0.0 and self.orientation_cost == 0.0:
            raise FairinoConfigurationError(
                "position_cost and orientation_cost cannot both be zero"
            )
        for name in ("position_tolerance", "orientation_tolerance"):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value) or value <= 0.0:
                raise FairinoConfigurationError(f"{name} must be finite and positive")
            setattr(self, name, value)

        for name in ("frame_gain", "posture_gain", "step_size"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise FairinoConfigurationError(f"{name} must be in (0, 1]")
            setattr(self, name, value)

        if not isinstance(self.body_avoidance, bool):
            raise FairinoConfigurationError("body_avoidance must be a bool")
        for name in (
            "body_activation_distance",
            "body_min_distance",
            "body_link_radius",
            "body_avoidance_weight",
            "body_avoidance_gain",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise FairinoConfigurationError(f"{name} must be finite and non-negative")
            setattr(self, name, value)

        if self.qp_solver is not None:
            self.qp_solver = str(self.qp_solver).strip().lower()
            if not self.qp_solver:
                raise FairinoConfigurationError("qp_solver cannot be empty")
        if self.qp_solver not in (None, "eiquadprog"):
            raise FairinoConfigurationError("Only the native C++ eiquadprog backend is supported")

        torso = np.asarray(self.torso_position, dtype=float).reshape(-1)
        if torso.shape != (len(TORSO_JOINTS),) or not np.isfinite(torso).all():
            raise FairinoConfigurationError(
                f"torso_position must contain {len(TORSO_JOINTS)} finite values "
                f"in {TORSO_JOINTS} order"
            )
        self.torso_position = tuple(float(value) for value in torso)
        head = float(self.head_position)
        if not np.isfinite(head):
            raise FairinoConfigurationError("head_position must be finite")
        self.head_position = head

        if self.posture_target is not None:
            posture = np.asarray(self.posture_target, dtype=float).reshape(-1)
            if posture.shape not in ((14,), (19,)) or not np.isfinite(posture).all():
                raise FairinoConfigurationError(
                    "posture_target must be a finite vector with the model's nq values"
                )
            self.posture_target = posture.copy()


@dataclass
class IKResult:
    """Solution and diagnostics returned by C++ constrained QP IK."""

    q: np.ndarray
    success: bool
    iterations: int
    error: float
    left_position_error: float
    left_orientation_error: float
    right_position_error: float
    right_orientation_error: float
    status: str = ""
    message: str = ""
    solver: str = ""
    qp_iterations: int = 0
    qp_active_constraints: int = 0
    solve_time_ms: float = 0.0
    body_collision_free: bool = True
    body_min_distance: float = float("inf")
    body_nearest_segment: str = ""
    full_q: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float).reshape(-1).copy()
        if self.full_q is not None:
            self.full_q = np.asarray(self.full_q, dtype=float).reshape(-1).copy()

    @property
    def position_error(self) -> float:
        return float(max(self.left_position_error, self.right_position_error))

    @property
    def orientation_error(self) -> float:
        return float(max(self.left_orientation_error, self.right_orientation_error))

    @property
    def residual(self) -> float:
        return float(self.error)

    def as_dict(self) -> dict[str, Any]:
        return {
            "q": self.q.tolist(),
            "success": bool(self.success),
            "iterations": int(self.iterations),
            "error": float(self.error),
            "position_error": self.position_error,
            "orientation_error": self.orientation_error,
            "left_position_error": float(self.left_position_error),
            "left_orientation_error": float(self.left_orientation_error),
            "right_position_error": float(self.right_position_error),
            "right_orientation_error": float(self.right_orientation_error),
            "status": self.status,
            "message": self.message,
            "solver": self.solver,
            "qp_iterations": self.qp_iterations,
            "qp_active_constraints": self.qp_active_constraints,
            "solve_time_ms": self.solve_time_ms,
            "body_collision_free": self.body_collision_free,
            "body_min_distance": self.body_min_distance,
            "body_nearest_segment": self.body_nearest_segment,
        }


def _native_module():
    from .runtime import _preload_native_dependencies

    _preload_native_dependencies()
    try:
        from . import _native
    except ImportError as exc:
        raise FairinoIKError(
            "C++ IK extension unavailable. The original native-loader error was: "
            f"{exc}. Reinstall the project after activating the intended environment "
            "with `pip install -e .`, and make sure its native library directories "
            "are visible in LD_LIBRARY_PATH."
        ) from exc
    return _native


def available_qp_solvers() -> tuple[str, ...]:
    _native_module()
    return ("eiquadprog",)


def _native_options(options: IKOptions | None):
    options = IKOptions() if options is None else options
    if not isinstance(options, IKOptions):
        raise TypeError("options must be IKOptions")
    options.validate()
    native = _native_module().Options()
    for name in options.__dataclass_fields__:
        if not hasattr(native, name):
            continue
        value = getattr(options, name)
        if name == "posture_target":
            value = [] if value is None else np.asarray(value, dtype=float).reshape(-1).tolist()
        elif name in ("position_tolerance", "orientation_tolerance"):
            value = 0.0 if value is None else value
        elif name == "torso_position":
            value = list(value)
        setattr(native, name, value)
    return native


def _pose_values(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (4, 4):
        raise FairinoConfigurationError(f"{name} must be a 4x4 matrix")
    # SO(3), finite and homogeneous checks are performed in C++.
    return array.reshape(-1).tolist()


def _batch_pose_values(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise FairinoConfigurationError(f"{name} must have shape (batch, 4, 4)")
    if not np.isfinite(array).all():
        raise FairinoConfigurationError(f"{name} must contain finite values")
    return [array[index].reshape(-1).tolist() for index in range(array.shape[0])]


def _q_values(value, nq, name="q"):
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (nq,) or not np.isfinite(array).all():
        raise FairinoConfigurationError(f"{name} must be a finite {nq}-element vector")
    return array.tolist()


class FairinoQPModel:
    """Dual-arm facade. FK, Jacobians and QP iteration run in C++.

    By default the IK moves the 14 arm joints (``q = [left 7, right 7]``).
    ``active_torso=True`` adds the torso (``q = [left 7, right 7, torso 5]``
    with the torso in ``TORSO_JOINTS`` order) so it can bend and turn to help
    the arms reach.
    """

    left_arm_joints = LEFT_ARM_JOINTS
    right_arm_joints = RIGHT_ARM_JOINTS
    torso_joints = TORSO_JOINTS
    head_joint = HEAD_JOINT

    def __init__(self, urdf_path=DEFAULT_URDF, *, left_frame=DEFAULT_LEFT_FRAME,
                 right_frame=DEFAULT_RIGHT_FRAME, base_pose=None, active_torso=False):
        """``base_pose`` maps base_link into the world frame; ``None`` uses
        :data:`DEFAULT_BASE_POSE` (x forward, y left, z up)."""
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"Fairino URDF does not exist: {self.urdf_path}")
        self.left_frame, self.right_frame = left_frame, right_frame
        self.base_pose = np.asarray(
            DEFAULT_BASE_POSE if base_pose is None else base_pose, dtype=float
        ).copy()
        try:
            self._native = _native_module().Model(str(self.urdf_path), left_frame, right_frame,
                                                  _pose_values(self.base_pose, "base_pose"),
                                                  bool(active_torso))
        except ValueError as exc:
            raise FairinoConfigurationError(str(exc)) from exc

    @property
    def nq(self): return self._native.nq

    @property
    def nv(self): return self._native.nv

    @property
    def full_nq(self): return self._native.full_nq

    @property
    def full_nv(self): return self._native.full_nv

    @property
    def active_torso(self): return self._native.active_torso

    @property
    def joint_names(self): return tuple(self._native.joint_names)

    @property
    def joint_limits(self): return np.asarray(self._native.joint_limits)

    @property
    def torso_limits(self): return np.asarray(self._native.torso_limits)

    @property
    def head_limits(self): return np.asarray(self._native.head_limits)

    def split_q(self, q):
        """Return ``(left, right, torso)``; ``torso`` is ``None`` in arm-only mode."""
        q = np.asarray(_q_values(q, self.nq))
        return q[:7], q[7:14], (q[14:] if self.active_torso else None)

    @property
    def velocity_limits(self):
        """Joint speed limits used with default :class:`IKOptions`."""
        return self.effective_velocity_limits()

    def effective_velocity_limits(self, options=None):
        """Per-joint speed bound (rad/s) the QP uses for ``options``."""
        try:
            return np.asarray(self._native.velocity_limits(_native_options(options)))
        except ValueError as exc:
            raise FairinoConfigurationError(str(exc)) from exc

    @property
    def frame_names(self): return tuple(self._native.frame_names)

    @property
    def joint_coordinates(self): return dict(self._native.joint_coordinates)

    def full_configuration(self, q, options=None):
        try:
            return np.asarray(self._native.full_configuration(_q_values(q, self.nq), _native_options(options)))
        except ValueError as exc:
            raise FairinoConfigurationError(str(exc)) from exc

    def forward_kinematics(self, q, *, side="left", options=None):
        try:
            return np.asarray(self._native.forward_kinematics(
                _q_values(q, self.nq), str(side).lower(), _native_options(options))).reshape(4, 4)
        except ValueError as exc:
            raise FairinoConfigurationError(str(exc)) from exc

    def end_effector_poses(self, q, options=None):
        placements = self.frame_placements(q, options)
        return placements[self.left_frame], placements[self.right_frame]

    def frame_placements(self, q, options=None):
        try:
            placements = self._native.frame_placements(_q_values(q, self.nq), _native_options(options))
        except ValueError as exc:
            raise FairinoConfigurationError(str(exc)) from exc
        return {name: np.asarray(pose).reshape(4, 4) for name, pose in placements.items()}

    def solve_ik(self, left_target, right_target, q_seed=None, options=None, *, dt=None, solver=None):
        options = IKOptions() if options is None else options
        if not isinstance(options, IKOptions):
            raise TypeError("options must be IKOptions")
        overrides = {}
        if dt is not None: overrides["dt"] = dt
        if solver is not None: overrides["qp_solver"] = solver
        options = replace(options, **overrides)
        native_options = _native_options(options)
        seed = [0.0] * self.nq if q_seed is None else _q_values(q_seed, self.nq, "q_seed")
        try:
            native_result = self._native.solve_ik(_pose_values(left_target, "left_target"),
                _pose_values(right_target, "right_target"), seed, native_options)
        except ValueError as exc:
            raise FairinoConfigurationError(str(exc)) from exc
        return self._result_from_native(native_result)

    def _result_from_native(self, native_result):
        values = {name: getattr(native_result, name) for name in IKResult.__dataclass_fields__}
        return IKResult(**values)

    def solve_ik_batch(self, left_targets, right_targets, q_seeds=None,
                       options=None, *, dt=None, solver=None):
        """Solve a batch of independent dual-arm targets with the native QP.

        Each target still runs the same constrained QP iteration as
        :meth:`solve_ik`; the batch entry point only amortizes validation,
        model locking, and the Python/C++ boundary.
        """
        options = IKOptions() if options is None else options
        if not isinstance(options, IKOptions):
            raise TypeError("options must be IKOptions")
        overrides = {}
        if dt is not None:
            overrides["dt"] = dt
        if solver is not None:
            overrides["qp_solver"] = solver
        options = replace(options, **overrides)
        native_options = _native_options(options)
        left = _batch_pose_values(left_targets, "left_targets")
        right = _batch_pose_values(right_targets, "right_targets")
        if len(left) != len(right) or not left:
            raise FairinoConfigurationError(
                "left_targets and right_targets must have the same non-zero batch size"
            )
        if q_seeds is None:
            seeds = [[0.0] * self.nq for _ in left]
        else:
            seed_array = np.asarray(q_seeds, dtype=float)
            if seed_array.shape != (len(left), self.nq) or not np.isfinite(seed_array).all():
                raise FairinoConfigurationError(
                    f"q_seeds must have shape ({len(left)}, {self.nq}) and contain finite values"
                )
            seeds = seed_array.tolist()
        try:
            native_results = self._native.solve_ik_batch(
                left, right, seeds, native_options
            )
        except ValueError as exc:
            raise FairinoConfigurationError(str(exc)) from exc
        return [self._result_from_native(result) for result in native_results]


def solve_dual_arm_ik(left_target, right_target, q_seed=None, *, urdf_path=DEFAULT_URDF,
                      left_frame=DEFAULT_LEFT_FRAME, right_frame=DEFAULT_RIGHT_FRAME,
                      base_pose=None, options=None, dt=None, solver=None):
    return FairinoQPModel(urdf_path, left_frame=left_frame, right_frame=right_frame,
                          base_pose=base_pose).solve_ik(left_target, right_target, q_seed,
                                                        options, dt=dt, solver=solver)


__all__ = ["ARM_JOINTS", "LEFT_ARM_JOINTS", "RIGHT_ARM_JOINTS", "TORSO_JOINTS",
           "HEAD_JOINT", "WHEEL_JOINTS", "JOINT_GROUPS", "ASSET_ROOT", "DEFAULT_URDF",
           "DEFAULT_LEFT_FRAME", "DEFAULT_RIGHT_FRAME", "DEFAULT_BASE_POSE",
           "URDF_BASE_POSE", "IKOptions", "IKResult",
           "FairinoQPModel", "FairinoIKError", "FairinoConfigurationError",
           "available_qp_solvers", "solve_dual_arm_ik"]
