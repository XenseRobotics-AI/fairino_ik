"""Chassis (vx, vy, rz) control, kept separate from the IK."""

from __future__ import annotations

import numpy as np
import pytest

from fairino_ik import (
    ChassisController,
    ChassisLimits,
    FairinoQPModel,
    IKOptions,
    base_goal_for_hands,
    body_twist,
    integrate_twist,
    planar_matrix,
    robot_to_world,
    world_to_robot,
)

HOME = np.radians([-30, 45, 0, 60, 0, 0, 0, 30, 45, 0, 60, 0, 0, 0, 30, 60, 30, 0, 0])


def test_ik_has_no_base_coordinates() -> None:
    model = FairinoQPModel(active_torso=True)
    assert model.nq == 19
    assert not hasattr(model, "active_base")


def test_integrate_and_body_twist_are_inverse() -> None:
    start = np.array([1.0, 2.0, 0.3])
    twist = (0.4, -0.2, 0.6)
    end = integrate_twist(start, twist, 0.5)
    # Fine Euler integration of the same constant body twist.
    pose = start.copy()
    for _ in range(20000):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        pose += 0.5 / 20000 * np.array([c * 0.4 + s * 0.2, s * 0.4 - c * 0.2, 0.6])
    assert np.allclose(end, pose, atol=1e-4)
    assert np.allclose(body_twist(start, end, 0.5), twist)
    vx, vy, rz = body_twist(start, end, 0.5, point=(0.2, 0.1))
    assert np.allclose((vx, vy, rz), (0.4 - 0.6 * 0.1, -0.2 + 0.6 * 0.2, 0.6))
    with pytest.raises(ValueError):
        body_twist(start, end, 0.0)


def test_world_robot_conversion() -> None:
    target = np.eye(4)
    target[:3, 3] = [2.0, 1.0, 1.2]
    base = (1.0, 0.5, np.pi / 2)
    local = world_to_robot(target, base)
    assert np.allclose(local[:3, 3], [0.5, -1.0, 1.2])
    assert np.allclose(robot_to_world(local, base), target)
    assert np.allclose(planar_matrix(base) @ local, target)


def test_teleop_saturates_together_and_ramps() -> None:
    controller = ChassisController(ChassisLimits(max_vx=0.5, max_vy=0.25, max_wz=0.5,
                                                 max_ax=1.0, max_ay=1.0, max_aw=1.0))
    first = controller.teleop((1.0, 1.0, 0.0), 0.02)
    # The change is limited to 1 m/s^2 * 0.02 s and keeps its direction.
    assert np.isclose(max(abs(first.vx), abs(first.vy)), 0.02)
    assert np.isclose(first.vx, first.vy)
    for _ in range(100):
        command = controller.teleop((1.0, 1.0, 0.0), 0.02)
    # Saturated by vy: both axes scaled by the same factor.
    assert np.allclose(command.as_tuple(), (0.25, 0.25, 0.0))
    for _ in range(100):
        command = controller.stop(0.02)
    assert command.as_tuple() == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("goal", [(2.0, 0.0, 0.0), (0.3, 1.0, np.pi / 2), (-1.0, 0.5, -2.5)])
def test_goal_controller_converges_within_limits(goal) -> None:
    limits = ChassisLimits()
    controller = ChassisController(limits)
    pose = np.zeros(3)
    dt = 0.02
    commands = []
    for _ in range(3000):
        command = controller.command(pose, goal, dt)
        commands.append(command.as_tuple())
        if command.reached:
            break
        pose = integrate_twist(pose, command.as_tuple(), dt)
    commands = np.abs(np.asarray(commands))
    assert command.reached
    assert np.hypot(*(pose[:2] - goal[:2])) <= controller.position_tolerance
    assert abs((pose[2] - goal[2] + np.pi) % (2 * np.pi) - np.pi) <= controller.yaw_tolerance
    assert (commands.max(axis=0) <= np.array([limits.max_vx, limits.max_vy, limits.max_wz]) + 1e-12).all()


def test_base_goal_for_hands_recovers_a_planar_motion() -> None:
    model = FairinoQPModel(active_torso=True)
    left, right = model.end_effector_poses(HOME)
    motion = np.array([1.5, -0.7, 0.9])
    goal = base_goal_for_hands(robot_to_world(left, motion), robot_to_world(right, motion),
                               left, right)
    assert np.allclose(goal, motion)


def test_ik_holds_world_targets_while_the_chassis_drives() -> None:
    """Chassis and IK run side by side: odometry -> robot-frame targets -> IK."""
    model = FairinoQPModel()
    options = IKOptions(torso_position=tuple(HOME[14:]), posture_weight=1.0,
                        posture_target=HOME[:14])
    left_world, right_world = model.end_effector_poses(HOME[:14], options)
    controller = ChassisController()
    pose = np.zeros(3)
    q = HOME[:14].copy()
    dt = 0.02
    for _ in range(40):  # drive forward and turn a little
        command = controller.teleop((0.3, 0.0, 0.2), dt)
        pose = integrate_twist(pose, command.as_tuple(), dt)
        result = model.solve_ik(world_to_robot(left_world, pose),
                                world_to_robot(right_world, pose), q, options)
        assert result.success, result.as_dict()
        q = result.q
    assert pose[0] > 0.05
    left_actual = robot_to_world(model.forward_kinematics(q, side="left", options=options), pose)
    assert np.allclose(left_actual, left_world, atol=1e-4)
