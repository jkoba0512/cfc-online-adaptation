"""Frozen-readout closed-loop validation (exchange 2026-11-06, implementation).

Reaches are run with a fixed (pre or post) readout, adapt off, field b in {0, 13}. On the
field-on trajectories, the learned readout change is evaluated on the actor's own sampled
state and compared with the curl-compensation torque at that sampled state:

    delta_actor = tau_post(x_actor) - tau_pre(x_actor)
    s_actor     = -J(q_actor)^T F_curl(q_actor, qdot_actor, b = 13)
    g = <delta, s>/D,  r = ||delta - s||^2/D,  h = ||delta - g s||^2/D,  D = <s, s>

These are controller-sample-time comparisons (torque is held within each RK4 step while the
field is evaluated at RK4 stage states); they are not integrated force or energy balances.
``delta_actor`` may contain baseline corrections and is not a pure field-learning quantity.
"""

from __future__ import annotations

import math

import numpy as np

from .metrics import endpoint_error, max_abs_perpendicular_deviation


def _ip(a, b) -> float:
    return float(np.sum(np.asarray(a, float) * np.asarray(b, float)))


def actor_metrics(delta: np.ndarray, s: np.ndarray) -> dict:
    """Projection metrics of delta on s; NaN (no epsilon) if D is not finite and positive."""
    delta, s = np.asarray(delta, float), np.asarray(s, float)
    out = {"D": _ip(s, s), "ip_delta_s": _ip(delta, s), "ss_delta": _ip(delta, delta),
           "ss_delta_minus_s": _ip(delta - s, delta - s)}
    finite = bool(np.all(np.isfinite(delta)) and np.all(np.isfinite(s)))
    if not (finite and math.isfinite(out["D"]) and out["D"] > 0):
        out.update({"g": math.nan, "r": math.nan, "h": math.nan, "valid": False,
                    "invalid_reason": "non-finite input" if not finite else "D not positive"})
        return out
    g = out["ip_delta_s"] / out["D"]
    out.update({"g": g, "r": out["ss_delta_minus_s"] / out["D"],
                "h": _ip(delta - g * s, delta - g * s) / out["D"], "valid": True,
                "invalid_reason": ""})
    out["identity_r_err"] = out["r"] - ((g - 1.0) ** 2 + out["h"])
    return out


def reach_summary(res) -> dict:
    """Per-reach closed-loop description from an ``AdaptiveResult``. Non-finite reaches are
    flagged and their metrics left NaN (never zero)."""
    out = {"finite": bool(res.finite)}
    keys = ("max_abs_perp_mm", "signed_peak_perp_mm", "endpoint_err_mm",
            "fb_rms_vec", "fb_rms_j1", "fb_rms_j2", "ff_rms_vec", "ff_rms_j1", "ff_rms_j2",
            "tau_rms_vec", "tau_rms_j1", "tau_rms_j2", "ff_absmax_j1", "ff_absmax_j2",
            "tau_absmax_j1", "tau_absmax_j2", "fb_absmax_j1", "fb_absmax_j2")
    if not res.finite:
        return {**out, **{k: math.nan for k in keys}}
    mabs, msig = max_abs_perpendicular_deviation(res.ee, res.ee_des[0], res.target_xy)
    out["max_abs_perp_mm"] = float(mabs) * 1e3
    out["signed_peak_perp_mm"] = float(msig) * 1e3
    out["endpoint_err_mm"] = float(endpoint_error(res.ee[-1], res.target_xy)) * 1e3
    for name, x in (("fb", res.tau_fb), ("ff", res.tau_ff), ("tau", res.tau)):
        x = np.asarray(x, float)
        out[f"{name}_rms_vec"] = float(np.sqrt(np.mean(np.sum(x ** 2, axis=1))))
        out[f"{name}_rms_j1"] = float(np.sqrt(np.mean(x[:, 0] ** 2)))
        out[f"{name}_rms_j2"] = float(np.sqrt(np.mean(x[:, 1] ** 2)))
        out[f"{name}_absmax_j1"] = float(np.max(np.abs(x[:, 0])))
        out[f"{name}_absmax_j2"] = float(np.max(np.abs(x[:, 1])))
    return out
