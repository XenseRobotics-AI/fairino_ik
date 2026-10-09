#!/usr/bin/env python3
"""Solve a pair of Fairino end-effector pose targets with native C++ QP.

The command operates on the fourteen serial arm joints (left j1..j7 followed
by right j8..j14).  The torso (j20, j19, j18, j16, j15), head (j17) and wheels
remain fixed; ``--torso`` and ``--head`` choose where they are locked.  With
``--active-torso`` the IK also moves the torso and ``--torso`` is its seed.
``--left-frame``/``--right-frame`` can be changed to ``left_flange`` and
``right_flange`` when the flange is the desired tool reference.

Targets use the world frame x forward, y left, z up (base_link turned 90
degrees about z); ``--urdf-base-frame`` switches to raw base_link axes.

Examples::

    python examples/dual_arm_ik.py
    python examples/dual_arm_ik.py \\
        --left-target 0.626 0.2354 0.905 -0.7854 0 1.5708 \\
        --right-target 0.626 -0.2354 0.905 0.7854 0 -1.5708 \\
        --q 0 30 0 60 0 30 0  0 30 0 60 0 30 0
    python examples/dual_arm_ik.py --torso 0 0 0 0 0.3
    python examples/dual_arm_ik.py --active-torso \\
        --left-target 0.7507 0.2354 0.5038 0.2618 0 1.5708 \\
        --right-target 0.7507 -0.2354 0.5038 -0.2618 0 -1.5708 \\
        --q 0 45 0 90 0 45 0  0 45 0 90 0 45 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .solver import (
    DEFAULT_LEFT_FRAME,
    DEFAULT_RIGHT_FRAME,
    DEFAULT_URDF,
    FairinoQPModel,
    IKOptions,
    TORSO_JOINTS,
    URDF_BASE_POSE,
)
from .runtime import reexec_without_ros


def _rpy_pose(xyz_rpy: list[float]) -> np.ndarray:
    if len(xyz_rpy) != 6:
        raise ValueError("a pose target must contain x y z roll pitch yaw")
    x, y, z, roll, pitch, yaw = map(float, xyz_rpy)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = rz @ ry @ rx
    pose[:3, 3] = [x, y, z]
    return pose


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--left-frame", default=DEFAULT_LEFT_FRAME)
    parser.add_argument("--right-frame", default=DEFAULT_RIGHT_FRAME)
    parser.add_argument(
        "--urdf-base-frame", action="store_true",
        help="use raw base_link axes (forward is -y) instead of x forward",
    )
    parser.add_argument(
        "--left-target", type=float, nargs=6, metavar="VALUE",
        help="left x y z roll pitch yaw in the world frame (metres/radians)",
    )
    parser.add_argument(
        "--right-target", type=float, nargs=6, metavar="VALUE",
        help="right x y z roll pitch yaw in the world frame (metres/radians)",
    )
    parser.add_argument(
        "--q", type=float, nargs=14, metavar="DEG",
        help="fourteen arm seed angles in degrees, left j1..j7 then right j8..j14",
    )
    parser.add_argument(
        "--torso", type=float, nargs=len(TORSO_JOINTS), metavar="RAD",
        default=[0.0] * len(TORSO_JOINTS),
        help="locked torso joints " + " ".join(TORSO_JOINTS) + " (radians)",
    )
    parser.add_argument("--head", type=float, default=0.0, help="locked head yaw j17 (radians)")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--damping", type=float, default=1e-5)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-step-norm", type=float, default=0.15)
    parser.add_argument(
        "--posture-weight", "--posture-cost", dest="posture_weight", type=float, default=0.0,
        help="null-space pull towards --home-q (or the seed); 0 disables",
    )
    parser.add_argument(
        "--home-q", type=float, nargs="+", metavar="DEG", default=None,
        help="home posture in degrees with the model's nq values (14, or 19 with --active-torso); "
        "enables --posture-weight 1 unless it is set",
    )
    parser.add_argument(
        "--velocity-limit", type=float, default=float(np.pi),
        help="arm joint speed in rad/s (the URDF limits are zero)",
    )
    parser.add_argument("--no-body-avoidance", action="store_true")
    parser.add_argument(
        "--active-torso", action="store_true",
        help="let the IK move the torso too (19 DoF); --torso is then the seed",
    )
    parser.add_argument(
        "--arms-first-iterations", type=int, default=50,
        help="iterations with the torso fixed before it is freed (0: free at once)",
    )
    parser.add_argument("--torso-step-weight", type=float, default=0.0)
    parser.add_argument("--solver", default="eiquadprog", help="native C++ backend")
    parser.add_argument("--print-full-q", action="store_true")
    return parser


def main() -> int:
    reexec_without_ros("fairino_ik.cli")
    args = _parser().parse_args()
    model = FairinoQPModel(
        args.urdf,
        left_frame=args.left_frame,
        right_frame=args.right_frame,
        base_pose=URDF_BASE_POSE if args.urdf_base_frame else None,
        active_torso=args.active_torso,
    )
    q_seed = (
        np.zeros(14, dtype=float)
        if args.q is None
        else np.radians(np.asarray(args.q, dtype=float))
    )
    if model.active_torso:
        q_seed = np.concatenate((q_seed, np.asarray(args.torso, dtype=float)))
    options = IKOptions(
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        damping=args.damping,
        dt=args.dt,
        max_step_norm=args.max_step_norm,
        posture_weight=args.posture_weight,
        default_velocity_limit=args.velocity_limit,
        body_avoidance=not args.no_body_avoidance,
        torso_step_weight=args.torso_step_weight,
        arms_first_iterations=args.arms_first_iterations,
        qp_solver=args.solver,
        torso_position=tuple(args.torso),
        head_position=args.head,
    )
    if args.home_q is not None:
        options.posture_target = np.radians(np.asarray(args.home_q, dtype=float))
        if options.posture_weight <= 0.0:
            options.posture_weight = 1.0
    if args.left_target is None and args.right_target is None:
        # Make the default command a real native QP solve instead of a zero-error
        # no-op by generating two nearby, reachable targets with FK.
        demo_q = np.radians(
            [4, -4, 3, 2, -2, 2, 3, -4, 4, -3, -2, 2, -2, 3]
            + ([2, 3, -2, 1, 4] if model.active_torso else [])
        )
        demo_q = np.clip(
            demo_q,
            model.joint_limits[:, 0] + 1e-5,
            model.joint_limits[:, 1] - 1e-5,
        )
        left_target = model.forward_kinematics(
            demo_q, side="left", options=options
        )
        right_target = model.forward_kinematics(
            demo_q, side="right", options=options
        )
    else:
        left_target = (
            model.forward_kinematics(q_seed, side="left", options=options)
            if args.left_target is None
            else _rpy_pose(args.left_target)
        )
        right_target = (
            model.forward_kinematics(q_seed, side="right", options=options)
            if args.right_target is None
            else _rpy_pose(args.right_target)
        )
    result = model.solve_ik(left_target, right_target, q_seed, options)
    print(f"URDF: {model.urdf_path}")
    print(f"frames: left={model.left_frame}, right={model.right_frame}")
    if model.active_torso:
        print(f"mode: arms + torso ({model.nq} DoF), head j17: {options.head_position}")
    else:
        print(f"torso {TORSO_JOINTS} [rad]: {options.torso_position}, head j17: {options.head_position}")
    print(f"backend: C++ QP / {result.solver}")
    print(f"success: {result.success} ({result.status})")
    print(f"iterations: {result.iterations}")
    print(f"position error [m]: {result.position_error:.8g}")
    print(f"orientation error [rad]: {result.orientation_error:.8g}")
    print(
        "left error [m, rad]: "
        f"{result.left_position_error:.8g}, {result.left_orientation_error:.8g}"
    )
    print(
        "right error [m, rad]: "
        f"{result.right_position_error:.8g}, {result.right_orientation_error:.8g}"
    )
    print(f"total error: {result.error:.8g}")
    if options.body_avoidance:
        print(
            f"torso clearance [m]: {result.body_min_distance:.4f} "
            f"({result.body_nearest_segment})"
        )
    print("arms q [deg]:", np.array2string(np.degrees(result.q[:14]), precision=6))
    if model.active_torso:
        print(f"torso {TORSO_JOINTS} [deg]:", np.array2string(np.degrees(result.q[14:]), precision=6))
    if options.posture_weight > 0.0:
        print(f"posture residual [rad]: {result.posture_residual:.3g}")
    if result.message:
        print(f"message: {result.message}")
    if args.print_full_q and result.full_q is not None:
        print("full Pinocchio q:", np.array2string(result.full_q, precision=8))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
