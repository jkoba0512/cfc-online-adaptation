"""Adaptive closed-loop FEL control with online readout plasticity (Step 3).

The frozen CfC core supplies the liquid features z_t; a numpy readout
(``LinearReadout`` / ``RLSReadout``, initialised to the offline J0) is the sole
owner of the adaptable weights. Per the Step 3 plan (option b), the torch head
is NOT written back each step — adaptation lives entirely in numpy:

    z_t, hx = model.features(x_t, hx)        # frozen core
    tau_ff  = readout.predict(z)             # numpy J z
    tau_fb  = Kp(q*-q) + Kd(qdot*-qdot)      # climbing-fiber signal
    tau     = tau_ff + tau_fb
    if adapt: readout.update(error=tau_fb, z=z)   # before the plant step
    plant step

hx is reset to None at the start of each reach; the readout J is carried across
reaches by the caller (the protocol), accumulating adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .arm import TwoLinkArm
from .core import CfCInverseModel
from .tasks import Reach


@dataclass
class AdaptiveResult:
    target_xy: np.ndarray
    t: np.ndarray
    q: np.ndarray  # (T, 2)
    qdot: np.ndarray
    q_des: np.ndarray
    ee: np.ndarray  # (T, 2)
    ee_des: np.ndarray
    tau: np.ndarray
    tau_ff: np.ndarray
    tau_fb: np.ndarray
    j_norm: np.ndarray  # (T,) ||J|| after each step
    dj_norm: np.ndarray  # (T,) ||J - J0||
    adapted: bool
    finite: bool


def run_adaptive_closed_loop(
    arm: TwoLinkArm,
    model: CfCInverseModel,
    readout,
    reach: Reach,
    Kp: np.ndarray,
    Kd: np.ndarray,
    J0: np.ndarray,
    dt: float = 0.005,
    field_b: float = 0.0,
    adapt: bool = True,
    error_signal: str = "fel",
    forward_model=None,
    sensory_sign: float = 1.0,
    sensory_gain: float = 1.0,
    device: str = "cpu",
) -> AdaptiveResult:
    """Simulate one reach under FEL with optional online readout adaptation.

    ``error_signal`` selects the climbing-fiber signal driving the readout:
      * "fel"          -> tau_fb = Kp(q*-q) + Kd(qdot*-qdot) (motor-command space, main)
      * "tracking"     -> Kp(q*-q) (raw joint position error mapped to torque dim)
      * "sensory_pred" -> sensory-prediction error from a forward model, mapped to a
        disturbance torque: cf = sensory_sign*sensory_gain*(-M(q) * qdd_resid), where
        qdd_resid = (qdot_next - qdot_pred_next)/dt. Computed AFTER the plant step
        (it needs the realised next state) and applied to the features z_t of step t.
    The control torque always uses the full PD tau_fb; only the *learning* signal
    differs. ``forward_model`` (a ForwardCfCModel) is required for "sensory_pred".
    """
    if error_signal not in ("fel", "tracking", "sensory_pred"):
        raise ValueError("error_signal must be 'fel', 'tracking' or 'sensory_pred'")
    if error_signal == "sensory_pred" and forward_model is None:
        raise ValueError("sensory_pred requires a forward_model")
    Kp = np.asarray(Kp, dtype=float)
    Kd = np.asarray(Kd, dtype=float)
    J0 = np.asarray(J0, dtype=float)
    n_steps = reach.q_des.shape[0]

    q = reach.q_des[0].copy()
    qdot = reach.qdot_des[0].copy()
    hx = None  # reset core state per reach
    hx_fwd = None  # forward-model carry state (sensory_pred)

    log_q = np.zeros((n_steps, 2))
    log_qdot = np.zeros((n_steps, 2))
    log_ee = np.zeros((n_steps, 2))
    log_tau = np.zeros((n_steps, 2))
    log_ff = np.zeros((n_steps, 2))
    log_fb = np.zeros((n_steps, 2))
    log_jn = np.zeros(n_steps)
    log_djn = np.zeros(n_steps)
    finite = True

    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            qd = reach.q_des[i]
            qdotd = reach.qdot_des[i]
            qddotd = reach.qddot_des[i]
            x = np.concatenate([qd, qdotd, qddotd, q, qdot]).astype(np.float32)
            xt = torch.from_numpy(x).view(1, 1, -1).to(device)
            z_t, hx = model.features(xt, hx)
            z = z_t[0, 0].cpu().numpy().astype(np.float64)

            tau_ff = readout.predict(z)
            tau_fb = Kp @ (qd - q) + Kd @ (qdotd - qdot)
            tau = tau_ff + tau_fb

            if adapt and error_signal in ("fel", "tracking"):
                # motor-command-space signals: update before the plant step
                cf = tau_fb if error_signal == "fel" else Kp @ (qd - q)
                readout.update(cf, z)

            # forward-model one-step velocity prediction (sensory_pred, pre-step)
            qdot_pred_next = None
            if error_signal == "sensory_pred":
                ds_hat, hx_fwd = forward_model.predict_step(q, qdot, tau, hx_fwd, device=device)
                qdot_pred_next = qdot + ds_hat[2:4]

            log_q[i] = q
            log_qdot[i] = qdot
            log_ee[i] = arm.fk(q)
            log_tau[i] = tau
            log_ff[i] = tau_ff
            log_fb[i] = tau_fb
            log_jn[i] = np.linalg.norm(readout.J)
            log_djn[i] = np.linalg.norm(readout.J - J0)

            if not np.all(np.isfinite(tau)):
                finite = False
                break

            q_next, qdot_next = arm.rk4_step(q, qdot, tau, dt, field_b=field_b)
            next_finite = np.all(np.isfinite(q_next)) and np.all(np.isfinite(qdot_next))

            if adapt and error_signal == "sensory_pred" and next_finite:
                # sensory-prediction error -> disturbance-torque estimate (post-step)
                qdd_resid = (qdot_next - qdot_pred_next) / dt
                tau_ext_est = arm.mass_matrix(q) @ qdd_resid
                cf = sensory_sign * sensory_gain * (-tau_ext_est)
                readout.update(cf, z)
            q, qdot = q_next, qdot_next
            if not next_finite:
                finite = False
                if i + 1 < n_steps:
                    log_q[i + 1 :] = log_q[i]
                    log_ee[i + 1 :] = log_ee[i]
                break

    ee_des = np.array([arm.fk(qd) for qd in reach.q_des])
    return AdaptiveResult(
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
        j_norm=log_jn,
        dj_norm=log_djn,
        adapted=adapt,
        finite=finite,
    )
