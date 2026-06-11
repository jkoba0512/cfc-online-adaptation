"""Closed-loop feedback-error-learning control loop (§1.2, Step 2C).

Runs one center-out reach under the FEL controller

    tau   = tau_ff + tau_fb
    tau_ff = J z_t            (CfC inverse model, stepped online)
    tau_fb = Kp (q* - q) + Kd (qdot* - qdot)

The CfC core is stepped one timestep at a time, carrying its liquid state, so
the model sees the *actual* (q, qdot) fed back through x_t. No online readout
adaptation here (that is Step 3); this is the B0 / control substrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .arm import TwoLinkArm
from .core import CfCInverseModel
from .tasks import Reach


@dataclass
class ClosedLoopResult:
    target_xy: np.ndarray
    t: np.ndarray  # (T,)
    q: np.ndarray  # (T, 2) actual joint angles
    qdot: np.ndarray  # (T, 2)
    q_des: np.ndarray  # (T, 2)
    ee: np.ndarray  # (T, 2) actual endpoint
    ee_des: np.ndarray  # (T, 2) desired endpoint
    tau: np.ndarray  # (T, 2)
    tau_ff: np.ndarray  # (T, 2)
    tau_fb: np.ndarray  # (T, 2)
    finite: bool


def run_closed_loop(
    arm: TwoLinkArm,
    model: CfCInverseModel,
    reach: Reach,
    Kp: np.ndarray,
    Kd: np.ndarray,
    dt: float = 0.005,
    field_b: float = 0.0,
    device: str = "cpu",
) -> ClosedLoopResult:
    """Simulate one reach under FEL control (no online adaptation)."""
    Kp = np.asarray(Kp, dtype=float)
    Kd = np.asarray(Kd, dtype=float)
    n_steps = reach.q_des.shape[0]

    q = reach.q_des[0].copy()
    qdot = reach.qdot_des[0].copy()
    hx = None

    log_q = np.zeros((n_steps, 2))
    log_qdot = np.zeros((n_steps, 2))
    log_ee = np.zeros((n_steps, 2))
    log_tau = np.zeros((n_steps, 2))
    log_ff = np.zeros((n_steps, 2))
    log_fb = np.zeros((n_steps, 2))
    finite = True

    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            qd = reach.q_des[i]
            qdotd = reach.qdot_des[i]
            qddotd = reach.qddot_des[i]
            x = np.concatenate([qd, qdotd, qddotd, q, qdot])  # (10,)
            xt = torch.from_numpy(x.astype(np.float32)).view(1, 1, -1).to(device)
            tau_ff_t, _z, hx = model(xt, hx)
            tau_ff = tau_ff_t.view(-1).cpu().numpy().astype(np.float64)

            tau_fb = Kp @ (qd - q) + Kd @ (qdotd - qdot)
            tau = tau_ff + tau_fb

            log_q[i] = q
            log_qdot[i] = qdot
            log_ee[i] = arm.fk(q)
            log_tau[i] = tau
            log_ff[i] = tau_ff
            log_fb[i] = tau_fb

            if not np.all(np.isfinite(tau)):
                finite = False
                break

            q, qdot = arm.rk4_step(q, qdot, tau, dt, field_b=field_b)
            if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qdot))):
                finite = False
                # record the diverged sample tail as the last valid index+1
                if i + 1 < n_steps:
                    log_q[i + 1 :] = log_q[i]
                    log_ee[i + 1 :] = log_ee[i]
                break

    ee_des = np.array([arm.fk(qd) for qd in reach.q_des])
    return ClosedLoopResult(
        target_xy=np.asarray(reach.target_xy, dtype=float),
        t=reach.times,
        q=log_q,
        qdot=log_qdot,
        q_des=reach.q_des,
        ee=log_ee,
        ee_des=ee_des,
        tau=log_tau,
        tau_ff=log_ff,
        tau_fb=log_fb,
        finite=finite,
    )
