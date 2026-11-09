"""Actual-input replay for the readout-content probe (exchange 2026-11-06, implementation).

The learned readout difference of a receiver model is evaluated on controller inputs whose
actual-state part comes from a regenerated closed-loop trajectory (A or catch phase of a
supplier model) instead of the nominal desired state. The ideal comparison waveform
``s_nom`` and its energy ``D_nom`` stay fixed at the nominal reach values, so only the input
changes. This is an input-sensitivity description of a fixed learned function; it does not
measure realised closed-loop compensation or feedback sharing.

    x_input   = [qd, qdotd, qddotd, q_source, qdot_source]   (float32, as in the closed loop)
    delta_in  = tau_post(x_input) - tau_pre(x_input)
    g_input   = <delta_in, s_nom> / D_nom
    r_input   = ||delta_in - s_nom||^2 / D_nom
    h_input   = ||delta_in - g_input s_nom||^2 / D_nom
    delta_sensitivity = ||delta_in - delta_nom||^2 / D_nom
    s_shift   = ||s_input - s_nom||^2 / D_nom   (secondary; s_input from the actual q, qdot
                with the hypothetical b = 13 field; the A and catch phases had no field)
"""

from __future__ import annotations

import math

import numpy as np

from . import readout_probe as rp


def build_inputs(reach, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    """(n_steps, 10) float32 inputs [qd, qdotd, qddotd, q, qdot], built per step as the
    closed loop does (concatenate float64, then cast)."""
    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)
    n = reach.q_des.shape[0]
    if q.shape != (n, 2) or qdot.shape != (n, 2):
        raise ValueError(f"state arrays must be ({n}, 2)")
    return np.stack([np.concatenate([reach.q_des[i], reach.qdot_des[i], reach.qddot_des[i],
                                     q[i], qdot[i]]).astype(np.float32) for i in range(n)])


def features_from_inputs(model, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Feature sequence on given float32 inputs; hidden state reset (None) at reach start."""
    import torch

    hx = None
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(X.shape[0]):
            xt = torch.from_numpy(np.ascontiguousarray(X[i])).view(1, 1, -1).to(device)
            z_t, hx = model.features(xt, hx)
            out.append(z_t[0, 0].cpu().numpy().astype(np.float64))
    return np.stack(out)


def field_compensation(arm, q: np.ndarray, qdot: np.ndarray, b: float) -> np.ndarray:
    """s = -J(q)^T F_curl(q, qdot, b) at each step (depends on q and qdot only)."""
    return np.stack([-arm.jacobian(qi).T @ arm.curl_force(qi, qdi, b=b)
                     for qi, qdi in zip(np.asarray(q, float), np.asarray(qdot, float))])


def regenerate(arm, model, readout, reach, Kp, Kd, J0_ref, dt: float):
    """Closed-loop reach with a fixed readout, adapt off, null field (A or catch phase)."""
    from .adaptation import run_adaptive_closed_loop

    return run_adaptive_closed_loop(arm, model, readout, reach, Kp, Kd, J0_ref, dt=dt,
                                    field_b=0.0, adapt=False, error_signal="fel", device="cpu")


def _ip(a, b) -> float:
    return float(np.sum(np.asarray(a, float) * np.asarray(b, float)))


def replay_metrics(delta_in: np.ndarray, delta_nom: np.ndarray, s_nom: np.ndarray,
                   s_input: np.ndarray | None = None) -> dict:
    """Primary metrics against the fixed nominal ideal waveform. NaN if D_nom is not
    finite and positive (no epsilon)."""
    delta_in, delta_nom, s_nom = (np.asarray(x, float) for x in (delta_in, delta_nom, s_nom))
    D = _ip(s_nom, s_nom)
    c_star = s_nom.mean(axis=0, keepdims=True)
    out = {"D_nom": D, "ip_delta_s": _ip(delta_in, s_nom),
           "ss_delta_minus_s": _ip(delta_in - s_nom, delta_in - s_nom),
           "ss_delta": _ip(delta_in, delta_in),
           "ss_delta_minus_delta_nom": _ip(delta_in - delta_nom, delta_in - delta_nom),
           "ss_s_input_minus_s_nom": (math.nan if s_input is None
                                      else _ip(np.asarray(s_input, float) - s_nom,
                                               np.asarray(s_input, float) - s_nom))}
    if not (math.isfinite(D) and D > 0):
        for k in ("g", "r", "h", "delta_sensitivity", "r_const_min", "s_shift"):
            out[k] = math.nan
        return out
    g = out["ip_delta_s"] / D
    out.update({"g": g, "r": out["ss_delta_minus_s"] / D,
                "h": _ip(delta_in - g * s_nom, delta_in - g * s_nom) / D,
                "delta_sensitivity": out["ss_delta_minus_delta_nom"] / D,
                "r_const_min": _ip(s_nom - c_star, s_nom - c_star) / D,
                "s_shift": out["ss_s_input_minus_s_nom"] / D})
    out["identity_r_err"] = out["r"] - ((g - 1.0) ** 2 + out["h"])
    return out


def rms_dev(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column RMS of a - b (e.g. per joint), units of the inputs."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return np.sqrt(np.mean(d ** 2, axis=0))


def feature_rel_dev(Z: np.ndarray, Z_nom: np.ndarray) -> float:
    """sqrt(sum ||z - z_nom||^2 / sum ||z_nom||^2); NaN if the denominator is not positive."""
    den = _ip(Z_nom, Z_nom)
    if not (math.isfinite(den) and den > 0):
        return math.nan
    return math.sqrt(_ip(Z - Z_nom, Z - Z_nom) / den)


__all__ = ["build_inputs", "features_from_inputs", "field_compensation", "regenerate",
           "replay_metrics", "rms_dev", "feature_rel_dev", "rp"]
