"""Arm + torso (19-DoF) IK regressions."""

from __future__ import annotations

import numpy as np
import pytest

from fairino_ik import ARM_JOINTS, FairinoConfigurationError, FairinoQPModel, IKOptions, TORSO_JOINTS

READY_ARMS = np.radians([0, 45, 0, 90, 0, 45, 0] * 2)
READY = np.concatenate((READY_ARMS, np.zeros(5)))


@pytest.fixture(scope="module")
def model() -> FairinoQPModel:
    return FairinoQPModel(active_torso=True)


def _shifted(model, q, offset):
    left, right = model.end_effector_poses(q)
    left[:3, 3] += offset
    right[:3, 3] += offset
    return left, right


def test_layout_and_limits(model) -> None:
    assert model.active_torso
    assert model.nq == model.nv == 19
    assert model.joint_names == ARM_JOINTS + TORSO_JOINTS
    assert model.joint_limits.shape == (19, 2)
    assert np.allclose(model.joint_limits[14:], model.torso_limits)
    assert np.allclose(model.velocity_limits[14:], IKOptions().torso_velocity_limit)
    left, right, torso = model.split_q(np.arange(19.0))
    assert left[0] == 0 and right[0] == 7 and torso[0] == 14
    with pytest.raises(FairinoConfigurationError, match="19"):
        model.forward_kinematics(np.zeros(14))


def test_fk_matches_arm_only_model_with_locked_torso(model) -> None:
    torso = np.array([0.3, 0.8, -0.4, 0.2, 0.5])
    arm_only = FairinoQPModel()
    locked = IKOptions(torso_position=tuple(torso), torso_step_weight=0.0)
    q = np.concatenate((READY_ARMS, torso))
    for side in ("left", "right"):
        assert np.allclose(
            model.forward_kinematics(q, side=side),
            arm_only.forward_kinematics(READY_ARMS, side=side, options=locked),
        )
    full_q = model.full_configuration(q)
    coordinates = model.joint_coordinates
    assert np.allclose([full_q[coordinates[name]] for name in TORSO_JOINTS], torso)


def test_arms_first_keeps_the_torso_still_when_the_arms_reach(model) -> None:
    left, right = _shifted(model, READY, [0.0, 0.0, -0.1])
    result = model.solve_ik(left, right, READY)
    assert result.success, result.as_dict()
    assert np.array_equal(result.q[14:], READY[14:])


@pytest.mark.parametrize("offset", [[0.2, 0.0, -0.5], [0.3, 0.0, -0.7], [0.55, 0.0, 0.0]])
def test_torso_extends_the_reach(model, offset) -> None:
    left, right = _shifted(model, READY, offset)
    arm_only = FairinoQPModel().solve_ik(left, right, READY_ARMS)
    assert not arm_only.success

    result = model.solve_ik(left, right, READY)
    assert result.success, result.as_dict()
    assert np.abs(result.q[14:]).max() > np.radians(5.0)
    assert result.body_collision_free
    assert np.all(result.q >= model.joint_limits[:, 0] - 1e-9)
    assert np.all(result.q <= model.joint_limits[:, 1] + 1e-9)
    assert np.allclose(model.forward_kinematics(result.q, side="left"), left, atol=1e-4)


def test_random_reachable_targets_converge(model) -> None:
    limits = model.joint_limits
    rng = np.random.default_rng(5)
    probe = IKOptions(max_iterations=1)
    solved = attempted = 0
    while attempted < 40:
        goal = rng.uniform(np.minimum(0.6 * limits[:, 0], limits[:, 0]), 0.6 * limits[:, 1])
        goal[14:16] = rng.uniform(0.0, [0.6, 1.2])  # j20/j19 start at their lower limit
        seed = np.clip(goal + rng.normal(0.0, 0.2, 19), limits[:, 0], limits[:, 1])
        left, right = model.end_effector_poses(goal)
        if (model.solve_ik(left, right, goal, probe).body_min_distance < 0.02 or
                model.solve_ik(left, right, seed, probe).body_min_distance < 0.01):
            continue
        attempted += 1
        solved += model.solve_ik(left, right, seed).success
    assert solved == attempted


def test_one_step_obeys_torso_speed_limit(model) -> None:
    left, right = _shifted(model, READY, [0.3, 0.0, -0.7])
    options = IKOptions(max_iterations=1, arms_first_iterations=0, dt=0.01)
    result = model.solve_ik(left, right, READY, options)
    step = np.abs(result.q - READY)
    assert np.all(step <= model.effective_velocity_limits(options) * options.dt + 1e-9)
    assert np.any(step[14:] > 0.0)


def test_hands_stay_out_of_the_fixed_chassis(model) -> None:
    left, right = model.end_effector_poses(READY)
    left[:3, 3] = [0.45, 0.15, 0.05]
    right[:3, 3] = [0.45, -0.15, 0.05]
    min_distance = IKOptions().body_min_distance
    result = model.solve_ik(left, right, READY)
    assert not result.success
    assert result.body_collision_free, result.as_dict()
    assert "base_link" in result.body_nearest_segment
    # Clearance re-evaluated independently at the returned configuration.
    check = model.solve_ik(*model.end_effector_poses(result.q), result.q,
                           IKOptions(max_iterations=1))
    assert check.body_min_distance >= min_distance - 1e-5


def test_torso_moves_out_of_the_way_of_a_target_inside_the_upright_body(model) -> None:
    left, right = model.end_effector_poses(READY)
    left[:3, 3] = [0.326, 0.0, 1.10]  # inside body_Link while upright
    assert not FairinoQPModel().solve_ik(left, right, READY_ARMS).success
    result = model.solve_ik(left, right, READY)
    assert result.success, result.as_dict()
    assert result.body_min_distance >= IKOptions().body_min_distance - 1e-5
    assert np.abs(result.q[14:]).max() > np.radians(5.0)


def test_batch_matches_single_solves(model) -> None:
    left, right = _shifted(model, READY, [0.2, 0.0, -0.5])
    single = model.solve_ik(left, right, READY)
    batch = model.solve_ik_batch(np.stack([left, left]), np.stack([right, right]),
                                 np.stack([READY, READY]))
    assert all(result.success for result in batch)
    assert np.array_equal(batch[0].q, single.q)
    assert batch[1].iterations == single.iterations


HOME = np.radians([-30, 45, 0, 60, 0, 0, 0, 30, 45, 0, 60, 0, 0, 0, 30, 60, 30, 0, 0])


def test_home_posture_is_recovered_from_a_perturbed_seed(model) -> None:
    left, right = model.end_effector_poses(HOME)
    options = IKOptions(posture_weight=1.0, posture_target=HOME)
    rng = np.random.default_rng(0)
    for _ in range(3):
        seed = np.clip(HOME + rng.normal(0.0, 0.25, 19), model.joint_limits[:, 0],
                       model.joint_limits[:, 1])
        plain = model.solve_ik(left, right, seed)
        homed = model.solve_ik(left, right, seed, options)
        assert plain.success and homed.success, homed.as_dict()
        assert np.abs(plain.q - HOME).max() > np.radians(5.0)
        assert np.allclose(homed.q, HOME, atol=1e-4)
        assert homed.error < 1e-6
        assert homed.posture_residual < 1e-4


def test_home_posture_does_not_bias_the_hands_and_returns_after_a_loop(model) -> None:
    left, right = model.end_effector_poses(HOME)
    options = IKOptions(posture_weight=1.0, posture_target=HOME)
    q = HOME.copy()
    for t in np.linspace(0.0, 2.0 * np.pi, 40):
        offset = 0.1 * np.array([np.sin(t), 1.0 - np.cos(t), -np.sin(t)])
        target_left, target_right = left.copy(), right.copy()
        target_left[:3, 3] += offset
        target_right[:3, 3] += offset
        result = model.solve_ik(target_left, target_right, q, options)
        assert result.success and result.error < 1e-5, result.as_dict()
        q = result.q
    assert np.allclose(q, HOME, atol=1e-4)


def test_arm_only_home_posture(model) -> None:
    arm_only = FairinoQPModel()
    home = HOME[:14]
    locked = dict(torso_position=tuple(HOME[14:]))
    left, right = arm_only.end_effector_poses(home, IKOptions(**locked))
    seed = home + np.radians([10, -10, 15, 5, -10, 10, 5] * 2)
    result = arm_only.solve_ik(left, right, seed,
                               IKOptions(posture_weight=1.0, posture_target=home, **locked))
    assert result.success
    assert np.allclose(result.q, home, atol=1e-4)
