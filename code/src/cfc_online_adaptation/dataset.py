"""Supervised dataset for offline CfC inverse-model training (§3, Step 2A).

Teacher-forced inputs and analytic null-field inverse-dynamics teacher torques
are built from minimum-jerk center-out reaches:

    x_t        = [q*_t, qdot*_t, qddot*_t, q*_t, qdot*_t]        (dim 10)
                  (teacher-forced: actual q_t := q*_t, qdot_t := qdot*_t)
    tau_teach  = arm.inverse_dynamics(q*_t, qdot*_t, qddot*_t)   (dim 2, null field)

The dataset is augmented over a few movement durations and radii so a held-out
validation split is meaningful; the split is reproducible from a fixed seed.
Returns numpy arrays of shape (n_seq, time, dim); the trainer moves them to torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arm import TwoLinkArm
from .tasks import joint_space_reaches

INPUT_DIM = 10
TARGET_DIM = 2


@dataclass
class InverseDataset:
    """Sequence dataset for inverse-model supervised training."""

    x: np.ndarray  # (n_seq, time, 10)
    tau: np.ndarray  # (n_seq, time, 2)
    meta: list[dict]  # per-sequence: target_xy, radius, T

    def __len__(self) -> int:
        return self.x.shape[0]


def build_reach_sequences(
    arm: TwoLinkArm,
    center_q: np.ndarray,
    radii=(0.10,),
    durations=(0.5,),
    dt: float = 0.005,
    n_targets: int = 8,
    elbow: str = "down",
) -> InverseDataset:
    """Build teacher-forced (x, tau_teach) sequences over reaches.

    All sequences share the same length (so they stack into a dense array);
    duration/dt are chosen to keep step counts equal across `durations` only
    when durations are equal, so by default a single duration is used and
    radii provide the augmentation. When multiple durations are given, each
    duration group is returned padded to the max length is NOT done — instead
    we require equal step counts; mismatched lengths raise.
    """
    center_q = np.asarray(center_q, dtype=float)
    xs: list[np.ndarray] = []
    taus: list[np.ndarray] = []
    meta: list[dict] = []
    length = None
    for T in durations:
        reaches = joint_space_reaches(
            arm, center_q, radius=radii[0], T=T, dt=dt, n_targets=n_targets, elbow=elbow
        )
        n_steps = reaches[0].q_des.shape[0]
        if length is None:
            length = n_steps
        for radius in radii:
            reaches = joint_space_reaches(
                arm, center_q, radius=radius, T=T, dt=dt, n_targets=n_targets, elbow=elbow
            )
            for r in reaches:
                if r.q_des.shape[0] != length:
                    raise ValueError(
                        "all reaches must share a step count; vary radii not durations,"
                        " or keep durations equal"
                    )
                x = np.concatenate(
                    [r.q_des, r.qdot_des, r.qddot_des, r.q_des, r.qdot_des], axis=1
                )  # (time, 10) teacher-forced
                tau = np.array(
                    [
                        arm.inverse_dynamics(q, qd, qdd)
                        for q, qd, qdd in zip(r.q_des, r.qdot_des, r.qddot_des)
                    ]
                )  # (time, 2)
                xs.append(x)
                taus.append(tau)
                meta.append({"target_xy": r.target_xy, "radius": radius, "T": T})
    return InverseDataset(
        x=np.stack(xs).astype(np.float64),
        tau=np.stack(taus).astype(np.float64),
        meta=meta,
    )


def train_val_split(
    ds: InverseDataset, val_frac: float = 0.25, seed: int = 0
) -> tuple[InverseDataset, InverseDataset]:
    """Reproducible train/validation split over sequences (seeded permutation)."""
    n = len(ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    def take(idx):
        return InverseDataset(
            x=ds.x[idx], tau=ds.tau[idx], meta=[ds.meta[i] for i in idx]
        )

    return take(train_idx), take(val_idx)
