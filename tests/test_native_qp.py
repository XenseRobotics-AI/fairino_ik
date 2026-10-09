"""Offline native-kernel regressions: no robot connection is made."""
import builtins
import numpy as np
import pytest

from fairino_ik import FairinoConfigurationError, FairinoQPModel, IKOptions


def test_no_python_solver_or_kinematics_import(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"pink", "qpsolvers", "daqp", "pinocchio"}:
            raise AssertionError(f"Low-level Python dependency imported: {name}")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    model = FairinoQPModel()
    q = np.zeros(14)
    goal = np.array([.01, -.01, .01, -.01, .01, .01, -.01] * 2)
    left, right = model.end_effector_poses(goal)
    result = model.solve_ik(left, right, q)
    assert result.success, result.as_dict()
    assert result.solver == "eiquadprog"
    assert result.qp_iterations > 0
    assert result.solve_time_ms > 0


def test_one_qp_step_obeys_urdf_speed_and_position_bounds():
    model = FairinoQPModel()
    # Zero is a stretched (singular) arm; bend both elbows first.
    seed = np.zeros(14)
    seed[[3, 10]] = 1.0
    left, right = model.end_effector_poses(seed)
    left[2, 3] += .3
    right[2, 3] += .3
    options = IKOptions(max_iterations=1, dt=1e-4, body_avoidance=False)
    result = model.solve_ik(left, right, seed, options)
    assert not result.success
    assert result.qp_active_constraints > 0
    assert np.all(np.abs(result.q - seed) <= model.velocity_limits * options.dt + 1e-9)
    assert np.linalg.norm(result.q - seed) <= options.max_step_norm + 1e-9
    assert np.all(result.q >= model.joint_limits[:, 0] - 1e-9)
    assert np.all(result.q <= model.joint_limits[:, 1] + 1e-9)


def test_qp_iteration_limit_does_not_apply_an_unverified_step():
    model = FairinoQPModel()
    left, right = model.end_effector_poses(np.zeros(14))
    left[0, 3] += .3
    result = model.solve_ik(left, right, np.zeros(14),
                           IKOptions(qp_max_iterations=1, body_avoidance=False))
    assert not result.success
    assert result.status == "qp_max_iterations"
    assert np.array_equal(result.q, np.zeros(14))


@pytest.mark.parametrize("body_avoidance", [False, True])
def test_batch_entrypoint_preserves_the_constrained_qp(body_avoidance):
    model = FairinoQPModel()
    options = IKOptions(max_iterations=300, body_avoidance=body_avoidance)
    seed = np.zeros(14)
    goal = np.array([.01, -.01, .01, -.01, .01, .01, -.01] * 2)
    left, right = model.end_effector_poses(goal, options)

    single = model.solve_ik(left, right, seed, options)
    batch = model.solve_ik_batch(
        np.stack([left, left]), np.stack([right, right]),
        np.stack([seed, seed]), options,
    )

    assert single.success, single.as_dict()
    assert len(batch) == 2
    assert all(result.success for result in batch)
    assert all(result.solver == "eiquadprog" for result in batch)
    assert all(result.qp_iterations > 0 for result in batch)
    assert np.array_equal(batch[0].q, single.q)
    assert batch[0].iterations == single.iterations
    assert batch[0].qp_iterations == single.qp_iterations
    assert np.allclose(batch[1].full_q, model.full_configuration(batch[1].q, options))


def test_world_base_pose_and_flange_frame():
    base = np.eye(4)
    base[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    base[:3, 3] = [.3, .1, .5]
    local = FairinoQPModel(left_frame="left_flange", right_frame="right_flange",
                           base_pose=np.eye(4))
    world = FairinoQPModel(left_frame="left_flange", right_frame="right_flange", base_pose=base)
    options = IKOptions(torso_position=(.1, .3, -.2, .1, .2))
    goal = [.01, -.01, .01, -.01, .01, .01, -.01] * 2
    a, b = local.end_effector_poses(goal, options)
    left, right = world.end_effector_poses(goal, options)
    assert np.allclose(left, base @ a)
    assert np.allclose(right, base @ b)
    result = world.solve_ik(left, right, np.zeros(14), options)
    assert result.success, result.as_dict()
    fk = world.forward_kinematics(result.q, side="left", options=options)
    assert np.allclose(fk, left, atol=2e-5)


def test_backend_options_are_not_silently_ignored():
    model = FairinoQPModel()
    left, right = model.end_effector_poses(np.zeros(14))
    with pytest.raises(FairinoConfigurationError, match="eiq"):
        model.solve_ik(left, right, options=IKOptions(qp_solver="daqp"))


@pytest.mark.parametrize("max_iterations,qp_max_iterations", [(1, 100), (500, 1)])
def test_batch_preserves_step_bounds_and_qp_failure(max_iterations, qp_max_iterations):
    model = FairinoQPModel()
    left, right = model.end_effector_poses(np.zeros(14))
    left[0, 3] += .3
    options = IKOptions(max_iterations=max_iterations,
                        qp_max_iterations=qp_max_iterations,
                        dt=1e-4, body_avoidance=False)
    single = model.solve_ik(left, right, options=options)
    batch = model.solve_ik_batch([left], [right], options=options)[0]
    assert not batch.success
    assert np.array_equal(batch.q, single.q)
    assert batch.status == single.status
    assert batch.qp_active_constraints == single.qp_active_constraints
    assert np.all(np.abs(batch.q) <= model.velocity_limits * options.dt + 1e-9)


def test_native_result_contains_full_configuration_without_extra_call(monkeypatch):
    model = FairinoQPModel()
    options = IKOptions(torso_position=(.1, .2, .3, -.1, .4), head_position=.5)
    left, right = model.end_effector_poses(np.zeros(14), options)
    expected = model.full_configuration(np.zeros(14), options)
    def unexpected(*args, **kwargs):
        raise AssertionError("extra full_configuration call")
    monkeypatch.setattr(model, "full_configuration", unexpected)
    assert np.array_equal(model.solve_ik(left, right, options=options).full_q, expected)
    assert np.array_equal(model.solve_ik_batch([left], [right], options=options)[0].full_q, expected)


def test_batch_input_validation():
    model = FairinoQPModel()
    left, right = model.end_effector_poses(np.zeros(14))
    with pytest.raises(FairinoConfigurationError, match="non-zero"):
        model.solve_ik_batch(np.empty((0,4,4)), np.empty((0,4,4)))
    with pytest.raises(FairinoConfigurationError, match="batch size"):
        model.solve_ik_batch([left], [right, right])
    with pytest.raises(FairinoConfigurationError, match="q_seeds"):
        model.solve_ik_batch([left], [right], np.zeros((1,7)))
    with pytest.raises(FairinoConfigurationError):
        model.solve_ik_batch([left], [np.full((4,4),np.nan)])
    bad = left.copy()
    bad[0,0] = 2.
    with pytest.raises(FairinoConfigurationError, match="rotation"):
        model.solve_ik_batch([bad], [right])
