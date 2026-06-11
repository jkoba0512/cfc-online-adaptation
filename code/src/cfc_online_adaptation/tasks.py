"""Center-out reaching task and minimum-jerk target trajectories (§6.3).

Endpoint-space minimum-jerk paths are generated, mapped to joint space by
inverse kinematics, and differentiated numerically to obtain qdot* and qddot*.
These desired trajectories drive both the feedforward CfC inverse model input
and the PD feedback term in the FEL control loop.

Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arm import TwoLinkArm


def minimum_jerk_scalar(t: np.ndarray | float, T: float) -> np.ndarray:
    """Minimum-jerk position fraction s(t) in [0, 1] over duration T.

    s(t) = 10 (t/T)^3 - 15 (t/T)^4 + 6 (t/T)^5, clamped outside [0, T].
    """
    tau = np.clip(np.asarray(t, dtype=float) / T, 0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def minimum_jerk_trajectory(
    p0: np.ndarray, p1: np.ndarray, T: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-jerk straight-line path from p0 to p1.

    Returns (times, positions) where positions has shape (n_steps, dim).
    Times are sampled at 0, dt, ..., T inclusive.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    n_steps = int(round(T / dt)) + 1
    times = np.linspace(0.0, T, n_steps)
    s = minimum_jerk_scalar(times, T)  # (n_steps,)
    positions = p0[None, :] + s[:, None] * (p1 - p0)[None, :]
    return times, positions


def center_out_targets(
    center_xy: np.ndarray, radius: float = 0.10, n_targets: int = 8
) -> np.ndarray:
    """Endpoint targets evenly spaced on a circle of given radius.

    Returns array of shape (n_targets, 2). Angles start at 0 rad (+x axis).
    """
    center_xy = np.asarray(center_xy, dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, n_targets, endpoint=False)
    offsets = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    return center_xy[None, :] + offsets


@dataclass
class Reach:
    """One center-out reach in joint space."""

    target_xy: np.ndarray
    times: np.ndarray  # (n_steps,)
    q_des: np.ndarray  # (n_steps, 2)
    qdot_des: np.ndarray  # (n_steps, 2)
    qddot_des: np.ndarray  # (n_steps, 2)


def joint_space_reaches(
    arm: TwoLinkArm,
    center_q: np.ndarray,
    radius: float = 0.10,
    T: float = 0.5,
    dt: float = 0.005,
    n_targets: int = 8,
    elbow: str = "down",
) -> list[Reach]:
    """Build n_targets center-out reaches as joint-space desired trajectories.

    The endpoint follows a minimum-jerk straight line from the center to each
    target; each sampled point is mapped to q* by IK (fixed elbow), and
    qdot*/qddot* are obtained by numerical differentiation (np.gradient, dt).
    """
    center_q = np.asarray(center_q, dtype=float)
    center_xy = arm.fk(center_q)
    targets = center_out_targets(center_xy, radius=radius, n_targets=n_targets)

    reaches: list[Reach] = []
    for target_xy in targets:
        times, ee_path = minimum_jerk_trajectory(center_xy, target_xy, T, dt)
        q_des = np.array([arm.inverse_kinematics(p, elbow=elbow) for p in ee_path])
        qdot_des = np.gradient(q_des, dt, axis=0)
        qddot_des = np.gradient(qdot_des, dt, axis=0)
        reaches.append(
            Reach(
                target_xy=target_xy,
                times=times,
                q_des=q_des,
                qdot_des=qdot_des,
                qddot_des=qddot_des,
            )
        )
    return reaches
