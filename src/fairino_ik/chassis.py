"""Omnidirectional chassis control with body-frame (vx, vy, rz) commands.

The chassis is deliberately kept out of the IK: the IK only solves the arms
(and optionally the torso) in the robot frame.  This module

* integrates and differentiates planar poses ``(x, y, yaw)`` exactly on SE(2),
* turns a goal pose or a teleop request into a saturated, acceleration-limited
  ``(vx, vy, rz)`` command for the chassis controller, and
* converts hand targets between the odometry (world) frame and the robot
  frame, so world-fixed targets can be handed to the IK while the robot drives.

``vx``/``vy`` are m/s along the robot's x (forward) and y (left) axes and
``rz`` is rad/s about z, all in the robot frame (``base_link`` with the
package's x-forward convention).  Poses are ``(x, y, yaw)`` of that frame in
the odometry frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def _pose3(value, name: str = "pose") -> np.ndarray:
    pose = np.asarray(value, dtype=float).reshape(-1)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite (x, y, yaw) triple")
    return pose


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def planar_matrix(pose: Sequence[float]) -> np.ndarray:
    """4x4 homogeneous transform of a planar pose ``(x, y, yaw)``."""

    x, y, yaw = _pose3(pose)
    matrix = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    matrix[:2, :2] = [[c, -s], [s, c]]
    matrix[:2, 3] = [x, y]
    return matrix


def _v_matrix(angle: float) -> np.ndarray:
    # Left Jacobian of SE(2): maps a body twist to the chord it travels.
    if abs(angle) < 1e-9:
        return np.eye(2)
    a = math.sin(angle) / angle
    b = (1.0 - math.cos(angle)) / angle
    return np.array([[a, -b], [b, a]])


def integrate_twist(pose: Sequence[float], twist: Sequence[float], dt: float) -> np.ndarray:
    """Pose after holding the body twist ``(vx, vy, rz)`` for ``dt`` seconds."""

    start = _pose3(pose)
    vx, vy, rz = _pose3(twist, "twist")
    angle = rz * dt
    chord = _v_matrix(angle) @ np.array([vx * dt, vy * dt])
    c, s = math.cos(start[2]), math.sin(start[2])
    world = np.array([[c, -s], [s, c]]) @ chord
    return np.array([start[0] + world[0], start[1] + world[1], _wrap(start[2] + angle)])


def relative_pose(pose_from: Sequence[float], pose_to: Sequence[float]) -> np.ndarray:
    """``pose_to`` expressed in the frame of ``pose_from``."""

    start, end = _pose3(pose_from), _pose3(pose_to)
    c, s = math.cos(start[2]), math.sin(start[2])
    local = np.array([[c, s], [-s, c]]) @ (end[:2] - start[:2])
    return np.array([local[0], local[1], _wrap(end[2] - start[2])])


def body_twist(
    pose_from: Sequence[float],
    pose_to: Sequence[float],
    dt: float,
    *,
    point: Sequence[float] = (0.0, 0.0),
) -> tuple[float, float, float]:
    """Constant body twist that moves ``pose_from`` onto ``pose_to`` in ``dt``.

    The linear part is the velocity of ``point`` (robot frame, default the
    robot origin); pass the chassis controller's reference point if it uses
    another one.
    """

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    delta = relative_pose(pose_from, pose_to)
    displacement = np.linalg.solve(_v_matrix(delta[2]), delta[:2])
    vx, vy = displacement / dt
    rz = delta[2] / dt
    px, py = (float(value) for value in point)
    return float(vx - rz * py), float(vy + rz * px), float(rz)


def world_to_robot(target_world: np.ndarray, base_pose: Sequence[float]) -> np.ndarray:
    """Express a 4x4 world (odometry) pose in the robot frame at ``base_pose``."""

    return np.linalg.inv(planar_matrix(base_pose)) @ np.asarray(target_world, dtype=float)


def robot_to_world(target_robot: np.ndarray, base_pose: Sequence[float]) -> np.ndarray:
    """Express a 4x4 robot-frame pose in the world (odometry) frame."""

    return planar_matrix(base_pose) @ np.asarray(target_robot, dtype=float)


def base_goal_for_hands(
    left_world: np.ndarray,
    right_world: np.ndarray,
    left_reference: np.ndarray,
    right_reference: np.ndarray,
) -> np.ndarray:
    """Base pose that puts two world hand targets where the reference hand
    poses (robot frame, e.g. the home posture's FK) are.

    Only the planar parts are used: the hands' midpoint fixes the position and
    the right-to-left hand direction fixes the yaw.  The IK then handles the
    remaining offset, height and orientation with the arms and torso.
    """

    lw, rw = np.asarray(left_world)[:2, 3], np.asarray(right_world)[:2, 3]
    lr, rr = np.asarray(left_reference)[:2, 3], np.asarray(right_reference)[:2, 3]
    world_dir, reference_dir = lw - rw, lr - rr
    if np.linalg.norm(world_dir) < 1e-6 or np.linalg.norm(reference_dir) < 1e-6:
        yaw = 0.0
    else:
        yaw = math.atan2(world_dir[1], world_dir[0]) - math.atan2(reference_dir[1], reference_dir[0])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    translation = 0.5 * (lw + rw) - rotation @ (0.5 * (lr + rr))
    return np.array([translation[0], translation[1], _wrap(yaw)])


@dataclass
class ChassisLimits:
    """Velocity (m/s, m/s, rad/s) and acceleration (m/s², m/s², rad/s²)
    limits of the chassis command.  An acceleration limit of 0 disables it."""

    max_vx: float = 0.5
    max_vy: float = 0.5
    max_wz: float = 0.5
    max_ax: float = 1.0
    max_ay: float = 1.0
    max_aw: float = 1.0

    def validate(self) -> None:
        for name in ("max_vx", "max_vy", "max_wz"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_ax", "max_ay", "max_aw"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass
class ChassisCommand:
    """One chassis command; ``reached`` is set when a goal is within tolerance."""

    vx: float
    vy: float
    rz: float
    reached: bool = False

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.rz)


class ChassisController:
    """Saturated, acceleration-limited (vx, vy, rz) commands for an omni base.

    :meth:`command` drives towards a goal pose with a proportional law on the
    exact SE(2) error, so the robot follows a constant-curvature arc and
    arrives with the goal heading.  :meth:`teleop` passes a requested twist
    through the same limits (e.g. for a joystick).  Velocity saturation
    scales all three axes together, keeping the commanded direction.
    """

    def __init__(
        self,
        limits: ChassisLimits | None = None,
        *,
        gain: Sequence[float] = (2.0, 2.0, 2.0),
        position_tolerance: float = 0.01,
        yaw_tolerance: float = 0.01,
    ) -> None:
        self.limits = ChassisLimits() if limits is None else limits
        self.limits.validate()
        self.gain = _pose3(gain, "gain")
        if (self.gain <= 0.0).any():
            raise ValueError("gain must be positive")
        if position_tolerance <= 0.0 or yaw_tolerance <= 0.0:
            raise ValueError("tolerances must be positive")
        self.position_tolerance = float(position_tolerance)
        self.yaw_tolerance = float(yaw_tolerance)
        self._last = np.zeros(3)

    @property
    def last_command(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._last)

    def reset(self) -> None:
        """Forget the previous command (the chassis is known to be at rest)."""
        self._last = np.zeros(3)

    def stop(self, dt: float) -> ChassisCommand:
        """Decelerate towards zero within the acceleration limits."""
        return self.teleop((0.0, 0.0, 0.0), dt)

    def teleop(self, twist: Sequence[float], dt: float) -> ChassisCommand:
        """Limit a requested body twist and return the command to send."""
        command = self._limit(_pose3(twist, "twist"), dt)
        return ChassisCommand(*command)

    def command(self, pose: Sequence[float], goal: Sequence[float], dt: float) -> ChassisCommand:
        """Command towards ``goal`` from the measured ``pose`` (odometry)."""
        delta = relative_pose(pose, goal)
        if np.hypot(delta[0], delta[1]) <= self.position_tolerance and abs(delta[2]) <= self.yaw_tolerance:
            command = self._limit(np.zeros(3), dt)
            return ChassisCommand(*command, reached=not command.any())
        # Twist that would reach the goal in unit time, scaled by the gains.
        error = np.append(np.linalg.solve(_v_matrix(delta[2]), delta[:2]), delta[2])
        desired = self.gain * error
        # Do not overshoot within one cycle.
        desired = np.where(np.abs(desired * dt) > np.abs(error), error / dt, desired)
        return ChassisCommand(*self._limit(desired, dt))

    def _limit(self, desired: np.ndarray, dt: float) -> np.ndarray:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive")
        limits = self.limits
        maximum = np.array([limits.max_vx, limits.max_vy, limits.max_wz])
        command = desired / max(1.0, float(np.max(np.abs(desired) / maximum)))
        acceleration = np.array([limits.max_ax, limits.max_ay, limits.max_aw])
        enabled = acceleration > 0.0
        step = np.where(enabled, acceleration * dt, np.inf)
        change = command - self._last
        # Scale the change together as well, so ramps stay straight.
        ratio = float(np.max(np.where(enabled, np.abs(change) / step, 0.0)))
        if ratio > 1.0:
            command = self._last + change / ratio
        self._last = command
        return command


__all__ = [
    "ChassisCommand",
    "ChassisController",
    "ChassisLimits",
    "base_goal_for_hands",
    "body_twist",
    "integrate_twist",
    "planar_matrix",
    "relative_pose",
    "robot_to_world",
    "world_to_robot",
]
