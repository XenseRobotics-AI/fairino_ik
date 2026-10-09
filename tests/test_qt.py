"""Offscreen smoke test for the independent slider-based Qt application."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from fairino_ik import FairinoQPModel, IKOptions
from fairino_ik.qt import FairinoIKWindow, _load_qt


def test_qt_window_joint_sliders_and_qp_solve() -> None:
    qt = _load_qt()
    model = FairinoQPModel()
    options = IKOptions(
        max_iterations=150,
        tolerance=2e-5,
        qp_solver="eiquadprog",
    )
    app = QApplication.instance() or QApplication([])
    window = FairinoIKWindow.create(
        qt,
        model,
        options,
        initial_q=np.zeros(14),
        visualizer=None,
        qp_solver="eiquadprog",
        smooth_time=0.0,
    )
    window.show()
    app.processEvents()

    window.left_joints.controls[0].set_value(1.0, emit=True)
    window._apply_fk_from_sliders("left")
    app.processEvents()
    assert np.isclose(window.goal_q[0], np.deg2rad(1.0), atol=2e-4)

    q_goal = window.goal_q.copy()
    q_goal[0] = np.deg2rad(2.0)
    window.left_pose.set_pose(
        model.forward_kinematics(q_goal, side="left", options=options)
    )
    window._solve("left")
    app.processEvents()

    assert window.goal_q.shape == (14,)
    assert np.isfinite(window.goal_q).all()
    assert "success=True" in window.ik_status.text(), window.ik_status.text()
    assert window.qp_solver_combo.currentData() == "eiquadprog"

    # Moving the locked torso keeps the arm joints and re-targets both arms.
    q_before = window.goal_q.copy()
    window.torso_joints.controls[4].set_value(20.0)
    window._apply_torso_from_sliders()
    app.processEvents()
    assert np.isclose(window.options.torso_position[4], np.deg2rad(20.0), atol=2e-4)
    assert np.array_equal(window.goal_q, q_before)
    assert np.allclose(
        window.left_pose.pose()[:3, 3],
        model.forward_kinematics(q_before, side="left", options=window.options)[:3, 3],
        atol=1e-3,
    )
    window.close()


def test_rotation_sliders_have_no_gimbal_lock() -> None:
    """A vertical tool (pitch = 90 deg) keeps three independent rotations."""
    qt = _load_qt()
    app = QApplication.instance() or QApplication([])
    from fairino_ik.qt import _PosePanel, _pose_from_xyz_rpy

    panel = _PosePanel(qt, "target")
    reference = _pose_from_xyz_rpy([0.5, 0.2, 1.0, 0.3, np.pi / 2, 0.2])
    panel.set_pose(reference)
    assert np.allclose(panel.pose(), reference)

    axes = []
    for index, name in enumerate(("rot x", "rot y", "rot z")):
        panel.controls[name].set_value(5.0)
        delta = panel.pose()[:3, :3] @ reference[:3, :3].T
        # Each slider alone is an exact rotation about one fixed world axis.
        expected = np.zeros(3)
        expected[index] = 1.0
        angle = np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
        axis = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0],
                         delta[1, 0] - delta[0, 1]]) / (2.0 * np.sin(angle))
        assert np.isclose(angle, np.radians(5.0))
        assert np.allclose(axis, expected)
        axes.append(axis)
        panel.controls[name].set_value(0.0)
    assert np.linalg.matrix_rank(np.array(axes)) == 3
    app.processEvents()


def test_qt_active_torso_solves_beyond_arm_reach() -> None:
    qt = _load_qt()
    app = QApplication.instance() or QApplication([])
    model = FairinoQPModel(active_torso=True)
    initial = np.concatenate((np.radians([0, 45, 0, 90, 0, 45, 0] * 2), np.zeros(5)))
    window = FairinoIKWindow.create(
        qt, model, IKOptions(), initial_q=initial, visualizer=None, smooth_time=0.0,
    )
    window.show()
    app.processEvents()
    assert window.goal_q.shape == (19,)

    for panel in (window.left_pose, window.right_pose):
        target = panel.pose()
        target[:3, 3] += [0.2, 0.0, -0.5]
        panel.set_pose(target)
    window._solve("both")
    app.processEvents()
    assert "success=True" in window.ik_status.text(), window.ik_status.text()
    assert np.abs(window.goal_q[14:]).max() > np.radians(5.0)
    # The torso sliders follow the IK solution.
    assert np.allclose(window.torso_joints.q()[:5], window.goal_q[14:], atol=2e-4)

    # Dragging a torso slider moves the torso inside q, not the locked options.
    window.torso_joints.controls[4].set_value(15.0)
    window._apply_torso_from_sliders()
    assert np.isclose(window.goal_q[18], np.deg2rad(15.0), atol=2e-4)
    assert window.options.torso_position == (0.0,) * 5
    window.close()
