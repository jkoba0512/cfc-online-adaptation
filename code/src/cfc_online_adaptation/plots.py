"""Paper-grade-adjacent confirmation plots for Phase 1 (Step 5D).

All functions take already-extracted arrays / dicts and write a PNG. matplotlib
is used with the Agg backend so the script runs headless. ``styles``
dicts (per condition: color / ls / alpha) let the caller distinguish the stable
methods from the unstable RLS-forgetting negative result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_STYLE = {"color": None, "ls": "-", "alpha": 1.0}


def _style(styles, name):
    s = dict(DEFAULT_STYLE)
    if styles and name in styles:
        s.update(styles[name])
    return s


def plot_learning_curves(
    curves_by_name: dict, b0_ref_mm: float, path: Path, styles: dict | None = None
) -> None:
    """B-phase mean max perpendicular deviation (mm) vs cycle per condition."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, b_perp_mm in curves_by_name.items():
        s = _style(styles, name)
        ax.plot(range(1, len(b_perp_mm) + 1), b_perp_mm, marker=".",
                label=name, color=s["color"], ls=s["ls"], alpha=s["alpha"])
    ax.axhline(b0_ref_mm, ls=":", c="k", lw=1.2, label=f"B0 curl ({b0_ref_mm:.1f} mm)")
    ax.set_xlabel("force-field exposure cycle")
    ax.set_ylabel("mean max perpendicular deviation (mm)")
    ax.set_title("Force-field adaptation learning curves")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_after_effect_bars(ae_by_name: dict, path: Path) -> None:
    """C-catch mean signed perp dev (mm); mirror after-effect = negative."""
    names = list(ae_by_name.keys())
    vals = [ae_by_name[n] for n in names]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    colors = ["tab:blue" if v < 0 else "tab:gray" for v in vals]
    ax.bar(names, vals, color=colors)
    ax.axhline(0, c="k", lw=1.0)
    ymin = min(vals + [0])
    ax.text(0.02, 0.06, "mirror after-effect (opposite to curl) = negative",
            transform=ax.transAxes, fontsize=8, color="tab:blue")
    ax.set_ylim(ymin * 1.25 if ymin < 0 else -1, max(vals + [1]) * 1.25 + 1)
    ax.set_ylabel("C-catch mean signed perp dev (mm)")
    ax.set_title("After-effect by condition")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_trajectories_example(traj: dict, dir_idx: int, path: Path, prefix: str = "lms") -> None:
    """Endpoint paths for one direction across A/B_first/B_last/C_catch."""
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = [("A baseline", f"{prefix}_A"), ("B first", f"{prefix}_B_first"),
              ("B last", f"{prefix}_B_last"), ("C catch", f"{prefix}_C_catch")]
    ee_des = traj[f"{prefix}_A_ee_des"][dir_idx]
    ax.plot(ee_des[:, 0], ee_des[:, 1], "k--", lw=1, label="desired")
    for lbl, key in labels:
        ee = traj[f"{key}_ee"][dir_idx]
        ax.plot(ee[:, 0], ee[:, 1], label=lbl)
    ax.scatter(*ee_des[0], c="k", s=25, zorder=5)
    ax.scatter(*ee_des[-1], c="r", s=25, zorder=5, label="target")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Endpoint trajectories (direction {dir_idx}, LMS)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_j_norm(jnorm_by_name: dict, path: Path, logy: bool = True,
                styles: dict | None = None) -> None:
    """Readout weight norm ||J|| per protocol cycle (log y by default)."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, jn in jnorm_by_name.items():
        s = _style(styles, name)
        ax.plot(range(len(jn)), jn, marker=".", label=name,
                color=s["color"], ls=s["ls"], alpha=s["alpha"])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("protocol cycle (A, B..., C)")
    ax.set_ylabel("||J||")
    ax.set_title("Readout weight norm over the protocol")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def signed_perp_curve(arr: np.ndarray) -> np.ndarray:  # small helper kept for reuse
    return np.asarray(arr, dtype=float)
