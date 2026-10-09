"""Numerical and asset regressions for the C++ QP Fairino solver."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from fairino_ik import (
    ARM_JOINTS,
    DEFAULT_URDF,
    DEFAULT_BASE_POSE,
    FairinoConfigurationError,
    FairinoQPModel,
    HEAD_JOINT,
    IKOptions,
    JOINT_GROUPS,
    TORSO_JOINTS,
    URDF_BASE_POSE,
    available_qp_solvers,
)
from fairino_ik.runtime import sanitized_environment


def _small_goal(model: FairinoQPModel, options: IKOptions):
    q_goal = np.radians(
        [4.0, -4.0, 3.0, 2.0, -2.0, 2.0, 3.0, -4.0, 4.0, -3.0, -2.0, 2.0, -2.0, 3.0]
    )
    q_goal = np.clip(
        q_goal,
        model.joint_limits[:, 0] + 1e-5,
        model.joint_limits[:, 1] - 1e-5,
    )
    return (
        q_goal,
        model.forward_kinematics(q_goal, side="left", options=options),
        model.forward_kinematics(q_goal, side="right", options=options),
    )


def test_packaged_model_and_meshes_exist() -> None:
    assert DEFAULT_URDF.is_file()
    root = ET.parse(DEFAULT_URDF).getroot()
    mesh_paths = {
        (DEFAULT_URDF.parent / mesh.attrib["filename"]).resolve()
        for mesh in root.findall(".//mesh")
    }
    assert mesh_paths
    assert all(path.is_file() for path in mesh_paths)


def test_joint_groups_cover_every_movable_urdf_joint() -> None:
    root = ET.parse(DEFAULT_URDF).getroot()
    movable = {
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib["type"] != "fixed"
    }
    grouped = [name for names in JOINT_GROUPS.values() for name in names]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == movable


def test_ros_runtime_paths_are_filtered() -> None:
    environment, removed = sanitized_environment(
        {
            "PYTHONPATH": "/opt/ros/humble/lib/python3.10/site-packages:/work/src",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/opt/ros/humble/lib",
            "AMENT_PREFIX_PATH": "/opt/ros/humble",
        }
    )
    assert environment["PYTHONPATH"] == "/work/src"
    assert environment["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64"
    assert "AMENT_PREFIX_PATH" not in environment
    assert len(removed) == 3


def test_model_mapping_and_limits() -> None:
    model = FairinoQPModel()
    assert model.nq == model.nv == 14
    # Four continuous wheel drives use (cos, sin) coordinates.
    assert model.full_nq == 32 and model.full_nv == 28
    assert model.joint_names == ARM_JOINTS
    assert model.left_frame == "left_waist2_Link"
    assert model.right_frame == "right_waist2_Link"
    assert model.joint_limits.shape == (14, 2)
    assert np.isfinite(model.joint_limits).all()
    assert model.torso_limits.shape == (len(TORSO_JOINTS), 2)
    assert model.head_limits.shape == (2,)
    # The URDF velocity limits are zero, so the option fallback applies.
    assert np.allclose(model.velocity_limits, np.pi)
    assert np.allclose(
        model.effective_velocity_limits(IKOptions(default_velocity_limit=2.0)), 2.0
    )
    assert {"left_flange", "right_flange"} <= set(model.frame_names)


def test_flange_frames_extend_the_wrist_links() -> None:
    model = FairinoQPModel()
    placements = model.frame_placements(np.zeros(14))
    for side in ("left", "right"):
        wrist = placements[f"{side}_waist2_Link"]
        flange = placements[f"{side}_flange"]
        assert np.allclose(np.linalg.inv(wrist) @ flange[:, 3], [-0.114, 0, 0, 1])
    # Zero pose: both arms stretched sideways at shoulder height, left on +y.
    assert np.allclose(placements["left_flange"][:3, 3], [0.326, 0.9635, 1.205], atol=1e-3)
    assert np.allclose(placements["right_flange"][:3, 3], [0.326, -0.9635, 1.205], atol=1e-3)


def test_default_world_frame_is_x_forward_y_left() -> None:
    world = FairinoQPModel()
    urdf = FairinoQPModel(base_pose=URDF_BASE_POSE)
    assert np.allclose(world.base_pose, DEFAULT_BASE_POSE)
    # Elbows bent 90 degrees: both hands in front of the robot.
    q = np.radians([0, 45, 0, 90, 0, 45, 0] * 2)
    left, right = world.end_effector_poses(q)
    raw_left, raw_right = urdf.end_effector_poses(q)
    assert np.allclose(left, DEFAULT_BASE_POSE @ raw_left)
    assert left[0, 3] > 0.5 and right[0, 3] > 0.5
    # The exported arm bases are mirrored only to ~0.1 mm.
    assert np.isclose(left[1, 3], -right[1, 3], atol=1e-3) and left[1, 3] > 0.0
    # In raw base_link coordinates the same hands are on -y, left arm on +x.
    assert raw_left[1, 3] < -0.5 and raw_left[0, 3] > 0.0


def test_torso_and_head_are_locked_but_move_the_arms() -> None:
    model = FairinoQPModel()
    q = np.radians([10, 20, -5, 40, 5, 10, 0, -10, -20, 5, 40, -5, -10, 0])
    torso = (0.3, 0.8, -0.4, 0.2, 0.5)
    options = IKOptions(torso_position=torso, head_position=0.4)

    full_q = model.full_configuration(q, options)
    coordinates = model.joint_coordinates
    assert full_q.shape == (32,)
    assert np.allclose([full_q[coordinates[name]] for name in TORSO_JOINTS], torso)
    assert np.isclose(full_q[coordinates[HEAD_JOINT]], 0.4)
    assert np.allclose([full_q[coordinates[name]] for name in ARM_JOINTS], q)

    upright = model.frame_placements(q)
    moved = model.frame_placements(q, options)
    body_motion = moved["body_Link"] @ np.linalg.inv(upright["body_Link"])
    for frame in (model.left_frame, model.right_frame):
        assert np.allclose(moved[frame], body_motion @ upright[frame])
    assert np.allclose(
        model.forward_kinematics(q, side="left", options=options), moved[model.left_frame]
    )

    left, right = model.end_effector_poses(q, options)
    result = model.solve_ik(left, right, np.zeros(14), options)
    assert result.success, result.as_dict()
    assert np.allclose(model.forward_kinematics(result.q, side="right", options=options),
                       right, atol=5e-5)


def test_fk_to_ik_roundtrip() -> None:
    assert "eiquadprog" in available_qp_solvers()
    model = FairinoQPModel()
    options = IKOptions(max_iterations=300, tolerance=2e-5, qp_solver="eiquadprog")
    _, left_target, right_target = _small_goal(model, options)
    result = model.solve_ik(left_target, right_target, np.zeros(14), options)

    assert result.success, result.as_dict()
    assert result.status == "converged"
    assert result.solver == "eiquadprog"
    assert result.q.shape == (14,)
    assert result.full_q is not None and result.full_q.shape == (32,)
    assert np.allclose(result.full_q, model.full_configuration(result.q, options))
    assert result.position_error <= options.tolerance
    assert result.orientation_error <= options.tolerance

    left_actual, right_actual = model.end_effector_poses(result.q, options)
    assert np.linalg.norm(left_actual[:3, 3] - left_target[:3, 3]) < 5e-5
    assert np.linalg.norm(right_actual[:3, 3] - right_target[:3, 3]) < 5e-5


def test_random_reachable_targets_converge() -> None:
    model = FairinoQPModel()
    limits = model.joint_limits
    rng = np.random.default_rng(7)
    solved = attempted = 0
    while attempted < 30:
        goal = rng.uniform(0.6 * limits[:, 0], 0.6 * limits[:, 1])
        seed = np.clip(goal + rng.normal(0.0, 0.3, 14), limits[:, 0], limits[:, 1])
        left, right = model.end_effector_poses(goal)
        probe = IKOptions(max_iterations=1)
        if (model.solve_ik(left, right, goal, probe).body_min_distance < 0.02 or
                model.solve_ik(left, right, seed, probe).body_min_distance < 0.01):
            continue
        attempted += 1
        solved += model.solve_ik(left, right, seed).success
    assert solved == attempted


def test_body_avoidance_keeps_the_arm_out_of_the_torso() -> None:
    model = FairinoQPModel()
    seed = np.zeros(14)
    seed[[3, 10]] = 1.2
    left, right = model.end_effector_poses(seed)
    inside = left.copy()
    inside[:3, 3] = [0.326, 0.0, 1.10]  # inside body_Link

    guarded = model.solve_ik(inside, right, seed)
    assert not guarded.success
    assert guarded.body_collision_free
    assert guarded.body_min_distance >= IKOptions().body_min_distance - 1e-6
    assert "body_Link" in guarded.body_nearest_segment

    unguarded = model.solve_ik(inside, right, seed, IKOptions(body_avoidance=False))
    assert unguarded.body_collision_free
    assert np.isinf(unguarded.body_min_distance)
    assert unguarded.left_position_error < guarded.left_position_error


def test_pose_and_dimension_validation() -> None:
    model = FairinoQPModel()
    target = model.forward_kinematics(np.zeros(14), side="left")
    bad_rotation = target.copy()
    bad_rotation[0, 0] = 2.0
    with pytest.raises(FairinoConfigurationError):
        model.solve_ik(bad_rotation, target, np.zeros(14))
    with pytest.raises(FairinoConfigurationError):
        model.forward_kinematics(np.zeros(7))
    with pytest.raises(FairinoConfigurationError, match="torso_position"):
        model.forward_kinematics(np.zeros(14), options=IKOptions(torso_position=(0.0,) * 4))
    with pytest.raises(FairinoConfigurationError, match="default_velocity_limit"):
        model.forward_kinematics(np.zeros(14), options=IKOptions(default_velocity_limit=0.0))
