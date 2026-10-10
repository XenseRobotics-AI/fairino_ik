#!/usr/bin/env python3
"""Slider-based Qt control panel for C++ QP Fairino dual-arm IK.

The FK tab contains seven joint sliders for each arm plus the torso/head
joints, which stay locked during IK.  The IK tab contains independent XYZ and
rotation sliders and separate left/right/both solve buttons.  Joint angles are
shown in degrees.

Target orientation is not edited as absolute roll/pitch/yaw: those angles lock
at pitch = +-90 deg, which is exactly where a vertical tool lands.  The rotation
sliders instead give a rotation vector about the fixed world x/y/z axes,
applied to the orientation captured at the last FK sync.  That map has no
singularity inside the slider range.

Examples::

    python examples/dual_arm_qt.py
    python examples/dual_arm_qt.py --no-meshcat
    python examples/dual_arm_qt.py                     # arms + torso (default)
    python examples/dual_arm_qt.py --no-active-torso   # arms only, torso locked
    python examples/dual_arm_qt.py --chassis           # plus (vx, vy, rz) chassis
    python examples/dual_arm_qt.py \
        --home-q 0 45 0 90 0 45 0  0 45 0 90 0 45 0  0 0 0 0 0

Without --home-q/--initial-q the window starts at
-30 45 0 60 0 0 0  30 45 0 60 0 0 0 (arms) with the torso at 30 60 30 0 0 deg.
The initial configuration is also the home posture: the redundant joints are
pulled towards it in the null space of the hand targets (``--posture-weight 0``
disables this).  "Set home" stores the current joints as the new home.

For a headless construction check, use ``QT_QPA_PLATFORM=offscreen`` together
with ``--no-show``.  A normal desktop session is required for interaction.

``--chassis`` adds a chassis panel driven by (vx, vy, rz) commands.  The
chassis is not part of the IK: the arm base (base_link) rides on the chassis,
the IK targets stay in base_link, and driving never changes the arm or torso
joints.  The chassis only moves the whole robot in the odometry frame.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np


from .chassis import (
    ChassisController,
    ChassisLimits,
    integrate_twist,
    planar_matrix,
)
from .solver import (
    DEFAULT_LEFT_FRAME,
    DEFAULT_RIGHT_FRAME,
    DEFAULT_URDF,
    FairinoQPModel,
    HEAD_JOINT,
    IKOptions,
    TORSO_JOINTS,
    URDF_BASE_POSE,
    available_qp_solvers,
)
from .runtime import reexec_without_ros


def _pose_from_xyz_rpy(values: Iterable[float]) -> np.ndarray:
    """Build a base-frame pose from x, y, z, roll, pitch, yaw (radians)."""

    x, y, z, roll, pitch, yaw = (float(value) for value in values)
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rz @ ry @ rx
    pose[:3, 3] = [x, y, z]
    return pose


def _rpy_from_rotation(rotation: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(rotation, dtype=float)
    pitch = float(np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0)))
    cp = float(np.cos(pitch))
    if abs(cp) > 1e-8:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-matrix[0, 1], matrix[1, 1]))
    return roll, pitch, yaw


def _rotation_from_vector(vector: Iterable[float]) -> np.ndarray:
    """Rodrigues' formula: the rotation by ``|vector|`` about its direction."""

    vector = np.asarray(vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def _pose_values(pose: np.ndarray) -> list[float]:
    roll, pitch, yaw = _rpy_from_rotation(np.asarray(pose)[:3, :3])
    return [
        float(pose[0, 3]),
        float(pose[1, 3]),
        float(pose[2, 3]),
        roll,
        pitch,
        yaw,
    ]


def _load_qt() -> dict[str, Any]:
    """Load optional Qt symbols lazily."""

    try:
        from PySide6.QtCore import QSignalBlocker, Qt, QTimer, QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QDoubleSpinBox,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QSlider,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "This example requires PySide6. Install it in the active environment."
        ) from exc
    return locals()


def _available_qp_solvers() -> list[str]:
    """Return installed native C++ backends, using native eiquadprog."""

    available = sorted(available_qp_solvers())
    if not available:
        raise RuntimeError("Native QP extension is unavailable")
    preferred = [name for name in ("eiquadprog",) if name in available]
    return preferred + [name for name in available if name not in preferred]


class SliderControl:
    """Floating-point slider with a synchronized numeric spin box."""

    def __init__(
        self,
        qt: dict[str, Any],
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int,
        single_step: float,
        suffix: str,
    ) -> None:
        if not minimum < maximum:
            raise ValueError(f"invalid slider range for {label}: {minimum}, {maximum}")
        self._qt = qt
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._slider_steps = 20_000
        self._callbacks: list[Callable[[float], None]] = []
        self.name_label = qt["QLabel"](label)
        self.name_label.setMinimumWidth(118)
        self.slider = qt["QSlider"](qt["Qt"].Orientation.Horizontal)
        self.slider.setRange(0, self._slider_steps)
        self.slider.setTracking(True)
        self.spin_box = qt["QDoubleSpinBox"]()
        self.spin_box.setRange(self._minimum, self._maximum)
        self.spin_box.setDecimals(decimals)
        self.spin_box.setSingleStep(single_step)
        self.spin_box.setSuffix(suffix)
        self.spin_box.setKeyboardTracking(True)
        self.spin_box.setMinimumWidth(125)
        layout = qt["QHBoxLayout"]()
        layout.setContentsMargins(0, 1, 0, 1)
        layout.addWidget(self.name_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin_box)
        self.widget = qt["QWidget"]()
        self.widget.setLayout(layout)
        self.slider.valueChanged.connect(self._on_slider)
        self.spin_box.valueChanged.connect(self._on_spin)
        self.set_value(value)

    def connect(self, callback: Callable[[float], None]) -> None:
        self._callbacks.append(callback)

    def _position_from_value(self, value: float) -> int:
        ratio = (value - self._minimum) / (self._maximum - self._minimum)
        return int(round(float(np.clip(ratio, 0.0, 1.0)) * self._slider_steps))

    def _value_from_position(self, position: int) -> float:
        ratio = float(position) / self._slider_steps
        return self._minimum + ratio * (self._maximum - self._minimum)

    def _emit(self, value: float) -> None:
        for callback in tuple(self._callbacks):
            callback(float(value))

    def _on_slider(self, position: int) -> None:
        blocker = self._qt["QSignalBlocker"](self.spin_box)
        value = self._value_from_position(position)
        self.spin_box.setValue(value)
        del blocker
        self._emit(value)

    def _on_spin(self, value: float) -> None:
        blocker = self._qt["QSignalBlocker"](self.slider)
        self.slider.setValue(self._position_from_value(float(value)))
        del blocker
        self._emit(float(value))

    def value(self) -> float:
        return float(self.spin_box.value())

    def set_value(self, value: float, *, emit: bool = False) -> None:
        value = float(np.clip(value, self._minimum, self._maximum))
        slider_blocker = self._qt["QSignalBlocker"](self.slider)
        spin_blocker = self._qt["QSignalBlocker"](self.spin_box)
        self.slider.setValue(self._position_from_value(value))
        self.spin_box.setValue(value)
        del slider_blocker, spin_blocker
        if emit:
            self._emit(value)


class _JointPanel:
    def __init__(
        self,
        qt: dict[str, Any],
        title: str,
        joint_names: Iterable[str],
        limits: np.ndarray,
        labels: Iterable[str] | None = None,
    ) -> None:
        self.qt = qt
        self.controls: list[SliderControl] = []
        group = qt["QGroupBox"](title)
        layout = qt["QVBoxLayout"](group)
        joint_names = tuple(joint_names)
        labels = (
            tuple(f"J{index + 1} {name}" for index, name in enumerate(joint_names))
            if labels is None else tuple(labels)
        )
        for label, limit in zip(labels, limits, strict=True):
            control = SliderControl(
                qt,
                label,
                math.degrees(float(limit[0])),
                math.degrees(float(limit[1])),
                0.0,
                decimals=2,
                single_step=0.1,
                suffix=" deg",
            )
            self.controls.append(control)
            layout.addWidget(control.widget)
        self.widget = group

    def connect(self, callback: Callable[[float], None]) -> None:
        for control in self.controls:
            control.connect(callback)

    def set_q(self, q: Iterable[float]) -> None:
        for control, value in zip(self.controls, np.degrees(list(q)), strict=True):
            control.set_value(float(value))

    def q(self) -> np.ndarray:
        return np.radians([control.value() for control in self.controls])


class _PosePanel:
    """Absolute XYZ plus a world-axis rotation offset from a reference.

    ``pose()`` is ``exp([rx, ry, rz]) @ R_ref``.  ``set_pose`` stores the new
    reference and zeroes the rotation sliders, so a single rotation slider
    always turns the target about one fixed world axis.
    """

    SPECS = (
        ("x", -1.5, 1.5, 4, 0.001, " m"),
        ("y", -1.5, 1.5, 4, 0.001, " m"),
        ("z", 0.0, 2.2, 4, 0.001, " m"),
        ("rot x", -180.0, 180.0, 2, 0.5, " deg"),
        ("rot y", -180.0, 180.0, 2, 0.5, " deg"),
        ("rot z", -180.0, 180.0, 2, 0.5, " deg"),
    )

    def __init__(self, qt: dict[str, Any], title: str) -> None:
        self.qt = qt
        self.controls: dict[str, SliderControl] = {}
        self._reference = np.eye(3)
        group = qt["QGroupBox"](title)
        layout = qt["QVBoxLayout"](group)
        for name, minimum, maximum, decimals, step, suffix in self.SPECS:
            control = SliderControl(
                qt, name, minimum, maximum, 0.0,
                decimals=decimals, single_step=step, suffix=suffix,
            )
            self.controls[name] = control
            layout.addWidget(control.widget)
        self.rpy_label = qt["QLabel"]()
        self.rpy_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.rpy_label)
        self.widget = group
        self.connect(lambda _value: self._update_rpy_label())
        self._update_rpy_label()

    def connect(self, callback: Callable[[float], None]) -> None:
        for control in self.controls.values():
            control.connect(callback)

    def set_pose(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=float)
        self._reference = pose[:3, :3].copy()
        for name, value in zip(("x", "y", "z"), pose[:3, 3], strict=True):
            self.controls[name].set_value(float(value))
        for name in ("rot x", "rot y", "rot z"):
            self.controls[name].set_value(0.0)
        self._update_rpy_label()

    def pose(self) -> np.ndarray:
        offset = np.radians([self.controls[name].value() for name in ("rot x", "rot y", "rot z")])
        pose = np.eye(4)
        pose[:3, :3] = _rotation_from_vector(offset) @ self._reference
        pose[:3, 3] = [self.controls[name].value() for name in ("x", "y", "z")]
        return pose

    def _update_rpy_label(self) -> None:
        # Read-only display; near pitch = +-90 deg roll and yaw are ambiguous.
        roll, pitch, yaw = np.degrees(_rpy_from_rotation(self.pose()[:3, :3]))
        self.rpy_label.setText(
            f"target RPY (display only): {roll:7.2f} {pitch:7.2f} {yaw:7.2f} deg"
        )


# Start (and default home) pose in degrees: left j1..j7, right j8..j14 and
# torso j20, j19, j18, j16, j15.  Zero would be singular (every arm stretched
# with j1/j3/j5 collinear), so the arms start bent with the torso leaning in.
_START_ARMS_DEG = (-30.0, 45.0, 0.0, 60.0, 0.0, 0.0, 0.0, 30.0, 45.0, 0.0, 60.0, 0.0, 0.0, 0.0)
_START_TORSO_DEG = (30.0, 60.0, 30.0, 0.0, 0.0)


_TORSO_LABELS = (
    "j20 lower leg",
    "j19 knee",
    "j18 hip",
    "j16 waist",
    "j15 body yaw",
    "j17 head yaw",
)


class FairinoIKWindow:
    """FK/IK slider window for the two serial Fairino arms."""

    @staticmethod
    def create(
        qt: dict[str, Any],
        model: FairinoQPModel,
        options: IKOptions,
        *,
        initial_q: np.ndarray,
        visualizer: object | None = None,
        ik_update_hz: float = 30.0,
        smooth_time: float = 0.12,
        qp_solver: str = "eiquadprog",
        chassis: ChassisController | None = None,
    ):
        class _Window(qt["QMainWindow"]):
            def __init__(self) -> None:
                super().__init__()
                self.qt = qt
                self.model = model
                self.options = options
                self.visualizer = visualizer
                self.ik_update_hz = float(ik_update_hz)
                self.smooth_time = float(smooth_time)
                self.qp_solver = str(qp_solver)
                self._syncing = False
                self._pending_sides: set[str] = set()
                self.goal_q = np.clip(
                    np.asarray(initial_q, dtype=float).reshape(model.nq),
                    model.joint_limits[:, 0], model.joint_limits[:, 1],
                )
                if self.options.posture_weight > 0.0 and self.options.posture_target is None:
                    self.options = replace(self.options, posture_target=self.goal_q.copy())
                self.display_q = self.goal_q.copy()
                self._animation_start = self.display_q.copy()
                self._animation_started = time.monotonic()
                # Chassis odometry pose (x, y, yaw); stays zero without --chassis.
                self.chassis = chassis
                # Chassis odometry pose; it only places the robot in MeshCat.
                self.base = np.zeros(3)
                self.setWindowTitle("Fairino C++ QP Dual-Arm FK / IK")
                self.resize(1200, 900)
                self._build_ui()
                self._reset_from_q()

                self.fk_timer = qt["QTimer"](self)
                self.fk_timer.setSingleShot(True)
                self.fk_timer.setInterval(20)
                self.fk_timer.timeout.connect(self._apply_fk_from_sliders)
                self.torso_timer = qt["QTimer"](self)
                self.torso_timer.setSingleShot(True)
                self.torso_timer.setInterval(20)
                self.torso_timer.timeout.connect(self._apply_torso_from_sliders)
                self.chassis_timer = qt["QTimer"](self)
                self.chassis_timer.setInterval(max(1, round(1000.0 / self.ik_update_hz)))
                self.chassis_timer.timeout.connect(self._chassis_step)
                if self.chassis is not None:
                    self.chassis_timer.start()
                self.ik_timer = qt["QTimer"](self)
                self.ik_timer.setSingleShot(True)
                self.ik_timer.setInterval(max(1, round(1000.0 / self.ik_update_hz)))
                self.ik_timer.timeout.connect(self._solve_pending)
                self.render_timer = qt["QTimer"](self)
                self.render_timer.setInterval(16)
                self.render_timer.timeout.connect(self._render)
                self.render_timer.start()

            def _build_ui(self) -> None:
                central = self.qt["QWidget"]()
                root = self.qt["QVBoxLayout"](central)
                title = self.qt["QLabel"]("Fairino 轮腿双臂 FK / IK 滑动条控制")
                title.setStyleSheet("font-size: 20px; font-weight: 600;")
                root.addWidget(title)
                root.addWidget(self.qt["QLabel"](f"URDF: {model.urdf_path}"))
                mesh_row = self.qt["QHBoxLayout"]()
                self.mesh_label = self.qt["QLabel"](
                    f"MeshCat: {visualizer.url}" if visualizer is not None else "MeshCat: disabled"
                )
                mesh_row.addWidget(self.mesh_label, 1)
                self.open_mesh_button = self.qt["QPushButton"]("Open MeshCat")
                self.open_mesh_button.setEnabled(visualizer is not None)
                self.open_mesh_button.clicked.connect(self._open_meshcat)
                mesh_row.addWidget(self.open_mesh_button)
                root.addLayout(mesh_row)

                solver_row = self.qt["QHBoxLayout"]()
                solver_row.addWidget(self.qt["QLabel"]("IK backend: C++ QP"))
                solver_row.addStretch(1)
                solver_row.addWidget(self.qt["QLabel"]("QP solver:"))
                self.qp_solver_combo = self.qt["QComboBox"]()
                for solver_name in _available_qp_solvers():
                    self.qp_solver_combo.addItem(solver_name, solver_name)
                selected_index = self.qp_solver_combo.findData(self.qp_solver)
                if selected_index < 0:
                    selected_index = 0
                self.qp_solver_combo.setCurrentIndex(selected_index)
                self.qp_solver = str(self.qp_solver_combo.currentData())
                self.qp_solver_combo.currentIndexChanged.connect(
                    self._on_qp_solver_changed
                )
                solver_row.addWidget(self.qp_solver_combo)
                root.addLayout(solver_row)

                tabs = self.qt["QTabWidget"]()
                fk_page = self.qt["QWidget"]()
                fk_layout = self.qt["QVBoxLayout"](fk_page)
                fk_layout.addWidget(self.qt["QLabel"](
                    "FK：拖动左右臂关节滑条；Apply FK 会更新两侧末端目标。"
                    + ("躯干参与 IK 求解，头部锁定；拖动躯干滑条后两侧目标随之同步。"
                       if model.active_torso else
                       "躯干/头部在 IK 中锁定，改变后两侧目标随之同步。")
                ))
                joint_row = self.qt["QHBoxLayout"]()
                limits = np.asarray(model.joint_limits, dtype=float)
                self.left_joints = _JointPanel(qt, "Left arm joints", model.left_arm_joints, limits[:7])
                self.right_joints = _JointPanel(qt, "Right arm joints", model.right_arm_joints, limits[7:14])
                torso_limits = np.vstack((model.torso_limits, model.head_limits))
                self.torso_joints = _JointPanel(
                    qt,
                    "Torso (solved by IK) / head (locked)" if model.active_torso
                    else "Torso / head (locked during IK)",
                    TORSO_JOINTS + (HEAD_JOINT,), torso_limits, _TORSO_LABELS,
                )
                self.torso_joints.set_q(self._torso_values() + (self.options.head_position,))
                joint_row.addWidget(self.left_joints.widget)
                joint_row.addWidget(self.right_joints.widget)
                fk_layout.addLayout(joint_row)
                fk_layout.addWidget(self.torso_joints.widget)
                if self.chassis is not None:
                    root.addWidget(self._build_chassis_panel())
                fk_buttons = self.qt["QHBoxLayout"]()
                apply_left = self.qt["QPushButton"]("Apply left FK")
                apply_right = self.qt["QPushButton"]("Apply right FK")
                apply_both = self.qt["QPushButton"]("Apply both FK")
                apply_left.clicked.connect(lambda: self._apply_fk_from_sliders("left"))
                apply_right.clicked.connect(lambda: self._apply_fk_from_sliders("right"))
                apply_both.clicked.connect(self._apply_fk_from_sliders)
                fk_buttons.addWidget(apply_left)
                fk_buttons.addWidget(apply_right)
                fk_buttons.addWidget(apply_both)
                fk_layout.addLayout(fk_buttons)
                self.fk_status = self.qt["QLabel"]()
                self.fk_status.setWordWrap(True)
                self.fk_status.setStyleSheet("font-family: monospace;")
                fk_layout.addWidget(self.fk_status)
                fk_layout.addStretch(1)
                tabs.addTab(fk_page, "FK - joint sliders")

                ik_page = self.qt["QWidget"]()
                ik_layout = self.qt["QVBoxLayout"](ik_page)
                ik_layout.addWidget(self.qt["QLabel"](
                    "IK：拖动左/右 XYZ 和旋转滑条；目标相对机械臂基座 base_link（随底盘移动），x 前 / y 左 / z 上，平移单位米。"
                    "rot x/y/z 是绕世界固定轴、相对上次同步姿态的旋转（度），没有万向锁。"
                ))
                pose_row = self.qt["QHBoxLayout"]()
                self.left_pose = _PosePanel(qt, "Left arm target")
                self.right_pose = _PosePanel(qt, "Right arm target")
                pose_row.addWidget(self.left_pose.widget)
                pose_row.addWidget(self.right_pose.widget)
                ik_layout.addLayout(pose_row)
                self.left_pose.connect(lambda _value: self._schedule_ik("left"))
                self.right_pose.connect(lambda _value: self._schedule_ik("right"))
                ik_buttons = self.qt["QHBoxLayout"]()
                solve_left = self.qt["QPushButton"]("Solve left arm")
                solve_right = self.qt["QPushButton"]("Solve right arm")
                solve_both = self.qt["QPushButton"]("Solve both arms")
                sync = self.qt["QPushButton"]("Sync targets from current FK")
                home_buttons = self.qt["QHBoxLayout"]()
                set_home = self.qt["QPushButton"]("Set home = current joints")
                go_home = self.qt["QPushButton"]("Go home")
                set_home.clicked.connect(self._set_home)
                go_home.clicked.connect(self._go_home)
                home_buttons.addWidget(set_home)
                home_buttons.addWidget(go_home)
                self.home_label = self.qt["QLabel"]()
                self.home_label.setWordWrap(True)
                self.home_label.setStyleSheet("font-family: monospace;")
                solve_left.clicked.connect(lambda: self._solve("left"))
                solve_right.clicked.connect(lambda: self._solve("right"))
                solve_both.clicked.connect(lambda: self._solve("both"))
                sync.clicked.connect(self._reset_from_q)
                ik_buttons.addWidget(solve_left)
                ik_buttons.addWidget(solve_right)
                ik_buttons.addWidget(solve_both)
                ik_buttons.addWidget(sync)
                ik_layout.addLayout(ik_buttons)
                ik_layout.addLayout(home_buttons)
                ik_layout.addWidget(self.home_label)
                self._update_home_label()
                self.ik_status = self.qt["QLabel"]()
                self.ik_status.setWordWrap(True)
                self.ik_status.setStyleSheet("font-family: monospace;")
                ik_layout.addWidget(self.ik_status)
                ik_layout.addStretch(1)
                tabs.addTab(ik_page, "IK - XYZ / rotation sliders")
                root.addWidget(tabs, 1)
                reset = self.qt["QPushButton"]("Reset targets / sliders")
                reset.clicked.connect(self._reset_from_q)
                root.addWidget(reset)
                self.setCentralWidget(central)
                self.left_joints.connect(lambda _value: self._schedule_fk())
                self.right_joints.connect(lambda _value: self._schedule_fk())
                self.torso_joints.connect(lambda _value: self._schedule_torso())

            def _build_chassis_panel(self):
                limits = self.chassis.limits
                group = self.qt["QGroupBox"](
                    "Chassis (vx, vy, rz) - not part of the IK; base_link and the arms ride on it")
                layout = self.qt["QVBoxLayout"](group)
                self.chassis_controls: list[SliderControl] = []
                for label, limit, suffix in (
                    ("vx forward", limits.max_vx, " m/s"),
                    ("vy left", limits.max_vy, " m/s"),
                    ("rz turn", limits.max_wz, " rad/s"),
                ):
                    control = SliderControl(qt, label, -limit, limit, 0.0, decimals=3,
                                            single_step=0.01, suffix=suffix)
                    self.chassis_controls.append(control)
                    layout.addWidget(control.widget)
                row = self.qt["QHBoxLayout"]()
                stop = self.qt["QPushButton"]("Stop chassis")
                stop.clicked.connect(self._stop_chassis)
                reset = self.qt["QPushButton"]("Reset odometry")
                reset.clicked.connect(self._reset_odometry)
                row.addWidget(stop)
                row.addWidget(reset)
                layout.addLayout(row)
                self.chassis_label = self.qt["QLabel"]()
                self.chassis_label.setStyleSheet("font-family: monospace;")
                layout.addWidget(self.chassis_label)
                self._update_chassis_label((0.0, 0.0, 0.0))
                return group

            def _update_chassis_label(self, command) -> None:
                vx, vy, rz = command
                x, y, yaw = self.base
                self.chassis_label.setText(
                    f"command: vx={vx:+.3f} m/s  vy={vy:+.3f} m/s  rz={rz:+.3f} rad/s\n"
                    f"odometry: x={x:+.3f} m  y={y:+.3f} m  yaw={math.degrees(yaw):+.1f} deg"
                )

            def _stop_chassis(self) -> None:
                for control in self.chassis_controls:
                    control.set_value(0.0)

            def _reset_odometry(self) -> None:
                self._stop_chassis()
                self.chassis.reset()
                self.base = np.zeros(3)
                self._sync_targets()
                self._update_view(self.display_q)
                self._update_chassis_label((0.0, 0.0, 0.0))

            def _chassis_step(self) -> None:
                """One chassis cycle: limit the request and integrate odometry.

                The arm base rides on the chassis, so the arm/torso joints and
                the base_link IK targets are left untouched.
                """
                dt = 1.0 / self.ik_update_hz
                request = [control.value() for control in self.chassis_controls]
                command = self.chassis.teleop(request, dt).as_tuple()
                self._update_chassis_label(command)
                if not any(command):
                    return
                self.base = integrate_twist(self.base, command, dt)
                self._update_view(self.display_q)

            def _update_view(self, q: np.ndarray) -> None:
                if self.visualizer is not None:
                    try:
                        self.visualizer.update(
                            q, self.options,
                            planar_matrix(self.base) if self.chassis is not None else None,
                        )
                    except Exception as exc:  # pragma: no cover
                        self.mesh_label.setText(f"MeshCat update failed: {exc}")

            def _render(self) -> None:
                if self.smooth_time <= 0.0:
                    self.display_q = self.goal_q.copy()
                else:
                    progress = float(np.clip(
                        (time.monotonic() - self._animation_started) / self.smooth_time,
                        0.0, 1.0,
                    ))
                    smooth = progress * progress * progress * (10.0 + progress * (-15.0 + 6.0 * progress))
                    self.display_q = self._animation_start + smooth * (self.goal_q - self._animation_start)
                self._update_view(self.display_q)

            def _set_goal(self, q: np.ndarray, *, immediate: bool = False) -> None:
                q = np.asarray(q, dtype=float).reshape(model.nq)
                if immediate or self.smooth_time <= 0.0:
                    self.goal_q = q.copy()
                    self.display_q = q.copy()
                    self._animation_start = q.copy()
                else:
                    self._animation_start = self.display_q.copy()
                    self.goal_q = q.copy()
                    self._animation_started = time.monotonic()
                if immediate:
                    self._animation_started = time.monotonic()
                self._sync_joint_sliders()
                self._update_view(self.display_q)

            def _sync_joint_sliders(self) -> None:
                self._syncing = True
                try:
                    self.left_joints.set_q(self.goal_q[:7])
                    self.right_joints.set_q(self.goal_q[7:14])
                    if model.active_torso:
                        self.torso_joints.set_q(
                            self._torso_values() + (self.options.head_position,)
                        )
                finally:
                    self._syncing = False

            def _torso_values(self) -> tuple[float, ...]:
                if model.active_torso:
                    return tuple(float(value) for value in self.goal_q[14:])
                return tuple(self.options.torso_position)

            def _reset_from_q(self) -> None:
                self._set_goal(self.goal_q, immediate=True)
                self._sync_targets()
                self.fk_status.setText("Targets synchronized from current joint sliders.")
                self.ik_status.setText("Ready. Drag a target slider or press a solve button.")

            def _sync_targets(self) -> None:
                self._syncing = True
                try:
                    self.left_pose.set_pose(model.forward_kinematics(self.goal_q, side="left", options=self.options))
                    self.right_pose.set_pose(model.forward_kinematics(self.goal_q, side="right", options=self.options))
                finally:
                    self._syncing = False

            def _schedule_torso(self) -> None:
                if self._syncing:
                    return
                self.torso_timer.start()

            def _apply_torso_from_sliders(self) -> None:
                values = self.torso_joints.q()
                torso = tuple(float(value) for value in values[:-1])
                self.options = replace(self.options, head_position=float(values[-1]))
                if model.active_torso:
                    q = self.goal_q.copy()
                    q[14:] = torso
                    self._set_goal(q, immediate=True)
                else:
                    self.options = replace(self.options, torso_position=torso)
                # The arms keep their joint angles and ride along with the
                # torso, so the IK targets follow the new FK poses.
                self._sync_targets()
                self._update_view(self.display_q)
                self.fk_status.setText(
                    "Torso/head at [deg] = "
                    + np.array2string(np.degrees(values), precision=2, suppress_small=True)
                )

            def _schedule_fk(self, _value: float | None = None) -> None:
                if self._syncing:
                    return
                self.ik_timer.stop()
                self.fk_timer.start()

            def _apply_fk_from_sliders(self, side: str | None = None) -> None:
                self.fk_timer.stop()
                q = self.goal_q.copy()
                if side != "right":
                    q[:7] = self.left_joints.q()
                if side != "left":
                    q[7:14] = self.right_joints.q()
                q = np.clip(q, model.joint_limits[:, 0], model.joint_limits[:, 1])
                self._set_goal(q, immediate=True)
                self._syncing = True
                try:
                    if side != "right":
                        self.left_pose.set_pose(model.forward_kinematics(q, side="left", options=self.options))
                    if side != "left":
                        self.right_pose.set_pose(model.forward_kinematics(q, side="right", options=self.options))
                finally:
                    self._syncing = False
                self.fk_status.setText(
                    "FK applied: q [deg] = "
                    + np.array2string(np.degrees(q), precision=2, suppress_small=True)
                )

            def _schedule_ik(self, side: str) -> None:
                if self._syncing:
                    return
                self._pending_sides.add(side)
                self.fk_timer.stop()
                if not self.ik_timer.isActive():
                    self.ik_timer.start()
                self.ik_status.setText("Solving latest target...")

            def _on_qp_solver_changed(self, _index: int) -> None:
                selected = self.qp_solver_combo.currentData()
                if not selected:
                    return
                self.qp_solver = str(selected)
                self.ik_status.setText(f"C++ QP solver selected: {self.qp_solver}")

            def _solve_pending(self) -> None:
                sides = self._pending_sides or {"both"}
                self._pending_sides.clear()
                if sides == {"left"}:
                    self._solve("left")
                elif sides == {"right"}:
                    self._solve("right")
                else:
                    self._solve("both")

            def _solve(self, side: str, *, immediate: bool = False) -> None:
                try:
                    left_target = self.left_pose.pose()
                    right_target = self.right_pose.pose()
                    if side == "left":
                        right_target = model.forward_kinematics(self.goal_q, side="right", options=self.options)
                    elif side == "right":
                        left_target = model.forward_kinematics(self.goal_q, side="left", options=self.options)
                    result = model.solve_ik(
                        left_target,
                        right_target,
                        self.goal_q,
                        self.options,
                        dt=1.0 / self.ik_update_hz,
                        solver=self.qp_solver,
                    )
                    if np.isfinite(result.q).all():
                        self._set_goal(result.q, immediate=immediate)
                    torso_text = (
                        f"torso [deg] = {np.array2string(np.degrees(result.q[14:]), precision=2, suppress_small=True)}\n"
                        if model.active_torso else ""
                    )
                    self.ik_status.setText(
                        torso_text +
                        f"{side} solve: {result.status}, success={result.success}, "
                        f"iterations={result.iterations}\n"
                        f"position={result.position_error:.3e} m, "
                        f"orientation={result.orientation_error:.3e} rad, "
                        f"torso clearance={result.body_min_distance:.3f} m\n"
                        f"arms [deg] = {np.array2string(np.degrees(result.q[:14]), precision=2, suppress_small=True)}"
                    )
                    self.ik_status.setStyleSheet(
                        "font-family: monospace; color: #1b7f2a;"
                        if result.success else "font-family: monospace; color: #b00020;"
                    )
                except Exception as exc:  # pragma: no cover
                    self.ik_status.setText(f"IK failed: {exc}")
                    self.ik_status.setStyleSheet("font-family: monospace; color: #b00020;")

            def _update_home_label(self) -> None:
                if self.options.posture_weight <= 0.0:
                    self.home_label.setText("Home posture: disabled (--posture-weight 0)")
                    return
                home = self.options.posture_target
                self.home_label.setText(
                    "Home posture (null-space pull) [deg] = "
                    + np.array2string(np.degrees(home), precision=1, suppress_small=True)
                )

            def _set_home(self) -> None:
                self.options = replace(
                    self.options,
                    posture_target=self.goal_q.copy(),
                    posture_weight=self.options.posture_weight or 1.0,
                )
                self._update_home_label()

            def _go_home(self) -> None:
                if self.options.posture_target is None:
                    return
                self._set_goal(np.asarray(self.options.posture_target, dtype=float))
                self._sync_targets()
                self.ik_status.setText("Moved to the home posture.")

            def _open_meshcat(self) -> None:
                if self.visualizer is not None:
                    qt["QDesktopServices"].openUrl(qt["QUrl"](str(self.visualizer.url)))

        return _Window()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--left-frame", default=DEFAULT_LEFT_FRAME)
    parser.add_argument("--right-frame", default=DEFAULT_RIGHT_FRAME)
    parser.add_argument(
        "--urdf-base-frame", action="store_true",
        help="use raw base_link axes (forward is -y) instead of x forward",
    )
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--damping", type=float, default=1e-5)
    parser.add_argument(
        "--posture-weight", "--posture-cost", dest="posture_weight", type=float, default=1.0,
        help="null-space pull towards the home posture (0 disables)",
    )
    parser.add_argument("--posture-gain", type=float, default=1.0)
    parser.add_argument(
        "--home-q", type=float, nargs="+", metavar="DEG", default=None,
        help="home posture in degrees: 14 arm joints, or 19 with the torso "
        + " ".join(TORSO_JOINTS) + "; also the initial pose unless --initial-q is given",
    )
    parser.add_argument(
        "--velocity-limit", type=float, default=float(np.pi),
        help="arm joint speed in rad/s (the URDF limits are zero)",
    )
    parser.add_argument(
        "--torso", type=float, nargs=len(TORSO_JOINTS), metavar="RAD",
        default=None,
        help="initial torso joints " + " ".join(TORSO_JOINTS)
        + " in radians (default: 30 60 30 0 0 deg)",
    )
    parser.add_argument("--head", type=float, default=0.0, help="initial head yaw j17 (radians)")
    parser.add_argument("--no-body-avoidance", action="store_true")
    parser.add_argument(
        "--active-torso", action=argparse.BooleanOptionalAction, default=True,
        help="let the IK move the torso with the arms (default); --no-active-torso "
        "locks it at --torso",
    )
    parser.add_argument(
        "--chassis", action="store_true",
        help="add a chassis panel driven by (vx, vy, rz); the chassis is not part of the IK",
    )
    parser.add_argument(
        "--chassis-max-velocity", type=float, nargs=3, metavar=("VX", "VY", "RZ"),
        default=(0.5, 0.5, 0.5), help="chassis velocity limits in m/s, m/s, rad/s",
    )
    parser.add_argument(
        "--chassis-max-acceleration", type=float, nargs=3, metavar=("AX", "AY", "AW"),
        default=(1.0, 1.0, 1.0), help="chassis acceleration limits (0 disables an axis limit)",
    )
    parser.add_argument(
        "--arms-first-iterations", type=int, default=50,
        help="iterations with the torso fixed before it is freed (0: free at once)",
    )
    parser.add_argument("--torso-step-weight", type=float, default=0.0)
    parser.add_argument(
        "--solver",
        default="eiquadprog",
        help="native C++ backend shown initially in the GUI (default: eiquadprog)",
    )
    parser.add_argument("--ik-update-hz", type=float, default=30.0)
    parser.add_argument("--smooth-time", type=float, default=0.12)
    parser.add_argument("--initial-q", type=float, nargs=14, metavar="DEG", default=None)
    parser.add_argument("--no-meshcat", action="store_true")
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    qt = _load_qt()
    if args.torso is None:
        args.torso = [float(value) for value in np.radians(_START_TORSO_DEG)]
    try:
        model = FairinoQPModel(
            args.urdf, left_frame=args.left_frame, right_frame=args.right_frame,
            base_pose=URDF_BASE_POSE if args.urdf_base_frame else None,
            active_torso=args.active_torso,
        )
        available_solvers = _available_qp_solvers()
        if args.solver not in available_solvers:
            raise ValueError(
                f"QP solver {args.solver!r} is unavailable; installed: "
                + ", ".join(available_solvers)
            )
        options = IKOptions(
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            damping=args.damping,
            posture_weight=args.posture_weight,
            posture_gain=args.posture_gain,
            dt=1.0 / args.ik_update_hz,
            default_velocity_limit=args.velocity_limit,
            body_avoidance=not args.no_body_avoidance,
            torso_step_weight=args.torso_step_weight,
            arms_first_iterations=args.arms_first_iterations,
            qp_solver=args.solver,
            torso_position=tuple(args.torso),
            head_position=args.head,
        )
        options.validate()
        if not np.isfinite(args.ik_update_hz) or args.ik_update_hz <= 0:
            raise ValueError("--ik-update-hz must be positive")
        if not np.isfinite(args.smooth_time) or args.smooth_time < 0:
            raise ValueError("--smooth-time must be non-negative")
        torso = np.asarray(args.torso, dtype=float)
        home_q = None
        if args.home_q is not None:
            home_deg = np.asarray(args.home_q, dtype=float)
            if home_deg.shape not in ((14,), (19,)) or not np.isfinite(home_deg).all():
                raise ValueError("--home-q must contain 14 or 19 finite degree values")
            if home_deg.shape == (19,):
                torso = np.radians(home_deg[14:])
            home_q = np.radians(home_deg[:14])
        if args.initial_q is not None:
            initial_q = np.radians(np.asarray(args.initial_q, dtype=float))
        else:
            initial_q = home_q if home_q is not None else np.radians(_START_ARMS_DEG)
        if initial_q.shape != (14,) or not np.isfinite(initial_q).all():
            raise ValueError("--initial-q must contain 14 finite degree values")
        if model.active_torso:
            initial_q = np.concatenate((initial_q, torso))
            if home_q is not None:
                home_q = np.concatenate((home_q, torso))
        else:
            options = replace(options, torso_position=tuple(float(value) for value in torso))
        initial_q = np.clip(initial_q, model.joint_limits[:, 0], model.joint_limits[:, 1])
        if home_q is not None and options.posture_weight > 0.0:
            home_q = np.clip(home_q, model.joint_limits[:, 0], model.joint_limits[:, 1])
            options = replace(options, posture_target=home_q)
    except Exception as exc:
        print(f"Could not initialize Fairino model: {exc}", file=sys.stderr)
        return 1

    visualizer = None
    if not args.no_meshcat:
        try:
            from .visualizer import Visualizer

            visualizer = Visualizer(
                model, args.urdf, None, open_browser=not args.no_open_browser, root_node="fairino"
            )
            print(f"MeshCat: {visualizer.url}")
        except Exception as exc:  # optional visualizer must not block the GUI
            print(f"MeshCat disabled: {exc}", file=sys.stderr)

    app = qt["QApplication"].instance() or qt["QApplication"](sys.argv)
    chassis = None
    if args.chassis:
        vx, vy, wz = args.chassis_max_velocity
        ax, ay, aw = args.chassis_max_acceleration
        chassis = ChassisController(ChassisLimits(vx, vy, wz, ax, ay, aw))
    window = FairinoIKWindow.create(
        qt, model, options, initial_q=initial_q, visualizer=visualizer,
        ik_update_hz=args.ik_update_hz, smooth_time=args.smooth_time,
        qp_solver=args.solver, chassis=chassis,
    )
    window.show()
    if args.no_show:
        app.processEvents()
        window.close()
        return 0
    return int(app.exec())


def main() -> int:
    reexec_without_ros("fairino_ik.qt")
    try:
        return run(_parser().parse_args())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
