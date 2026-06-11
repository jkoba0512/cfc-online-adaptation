"""Trajectory error metrics for the reaching task (§7).

Perpendicular deviation from the ideal straight line (center -> target) is the
classic force-field readout used by Shadmehr & Mussa-Ivaldi; here it is kept
signed (sign of the cross product) so the direction of the deflection can be
logged. Joint-space and endpoint tracking errors summarise overall accuracy.
"""

from __future__ import annotations

import numpy as np


def signed_perpendicular_deviation(
    ee_traj: np.ndarray, start_xy: np.ndarray, target_xy: np.ndarray
) -> np.ndarray:
    """Signed perpendicular distance of each endpoint sample from start->target.

    Positive = left of the start->target direction (2D cross product > 0).
    Returns an array of shape (n_steps,).
    """
    ee_traj = np.asarray(ee_traj, dtype=float)
    start_xy = np.asarray(start_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    d = target_xy - start_xy
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return np.zeros(ee_traj.shape[0])
    u = d / norm
    rel = ee_traj - start_xy[None, :]
    # 2D cross product rel x u-direction -> signed distance
    return rel[:, 0] * u[1] - rel[:, 1] * u[0]


def max_abs_perpendicular_deviation(
    ee_traj: np.ndarray, start_xy: np.ndarray, target_xy: np.ndarray
) -> tuple[float, float]:
    """Return (max |perp deviation|, signed value at that point)."""
    perp = signed_perpendicular_deviation(ee_traj, start_xy, target_xy)
    i = int(np.argmax(np.abs(perp)))
    return float(np.abs(perp[i])), float(perp[i])


def endpoint_error(ee_final: np.ndarray, target_xy: np.ndarray) -> float:
    """Euclidean distance of the final endpoint from the target."""
    return float(np.linalg.norm(np.asarray(ee_final) - np.asarray(target_xy)))


def joint_tracking_rmse(q_traj: np.ndarray, q_des: np.ndarray) -> float:
    """RMS joint-space tracking error over the trajectory (rad)."""
    q_traj = np.asarray(q_traj, dtype=float)
    q_des = np.asarray(q_des, dtype=float)
    return float(np.sqrt(np.mean(np.sum((q_traj - q_des) ** 2, axis=1))))


def endpoint_tracking_rmse(ee_traj: np.ndarray, ee_des: np.ndarray) -> float:
    """RMS endpoint tracking error over the trajectory (m)."""
    ee_traj = np.asarray(ee_traj, dtype=float)
    ee_des = np.asarray(ee_des, dtype=float)
    return float(np.sqrt(np.mean(np.sum((ee_traj - ee_des) ** 2, axis=1))))
