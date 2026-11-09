"""Fixed-input readout-content probe (pilot spec v1.0, 2026-11-06).

Evaluates what an exposure-trained readout adds to the feedforward torque on a
*fixed nominal input*, against the known curl-field compensation torque. This is a
diagnostic of the learned function on the nominal (teacher-forced form) input, not a
closed-loop measurement: it does not measure realised compensation, feedback sharing,
or anything about human internal models.

Definitions (per reach direction; all controller time steps and both joints weighted
equally; ``<a, b>`` is the sum over steps and joints of the elementwise product):

    s      = -J(q*)^T F_curl(q*, qdot*, b)     ideal field-compensation torque
    n      = inverse_dynamics(q*, qdot*, qddot*)   null-field torque
    delta  = tau_post - tau_pre
    D      = <s, s>
    g      = <delta, s> / D
    r      = ||delta - s||^2 / D
    h      = ||delta - g s||^2 / D
    p      = ||tau_pre - n||^2 / D
    e_pre  = ||tau_pre - n - s||^2 / D
    e_post = ||tau_post - n - s||^2 / D
    k      = 2 <tau_pre - n, delta - s> / D
    r_const_min = ||s - mean_t s||^2 / D

Identities: r = (g - 1)^2 + h and e_post = p + r + k. ``r_const_min`` is the lower bound
of r over all time-constant deltas (a bound on the residual, not on the gain).
``h`` must not be used to judge compensation: delta = 0 gives h = 0 and r = 1.

The module is pure numpy except ``nominal_features`` (torch, mirrors the closed loop in
``adaptation.run_adaptive_closed_loop``: float32 input of shape (1, 1, 10), hidden state
reset to None at the start of each reach, z returned as float64).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

TOL_ABS = 1e-10
TOL_REL = 1e-10


def close(a, b, atol: float = TOL_ABS, rtol: float = TOL_REL) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(np.abs(a - b) <= atol + rtol * np.abs(b)))


# ----------------------------------------------------------------------------------------
# nominal input and ideal torques
# ----------------------------------------------------------------------------------------
def nominal_inputs(reach) -> np.ndarray:
    """(n_steps, 10) float32 controller inputs [qd, qdotd, qddotd, qd, qdotd].

    Built step by step exactly as the closed loop does (concatenate float64, then cast).
    """
    rows = [np.concatenate([reach.q_des[i], reach.qdot_des[i], reach.qddot_des[i],
                            reach.q_des[i], reach.qdot_des[i]]).astype(np.float32)
            for i in range(reach.q_des.shape[0])]
    return np.stack(rows)


def nominal_features(model, reach, device: str = "cpu") -> np.ndarray:
    """Feature sequence z (n_steps, d_z) float64 on the nominal input, hx reset per reach."""
    import torch

    X = nominal_inputs(reach)
    hx = None
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(X.shape[0]):
            xt = torch.from_numpy(X[i].copy()).view(1, 1, -1).to(device)
            z_t, hx = model.features(xt, hx)
            out.append(z_t[0, 0].cpu().numpy().astype(np.float64))
    return np.stack(out)


def ideal_torques(arm, reach, b: float) -> dict:
    """Null-field torque n, ideal field-compensation torque s, and two cross-checks."""
    n_steps = reach.q_des.shape[0]
    n = np.zeros((n_steps, 2))
    s = np.zeros((n_steps, 2))
    s_via_id = np.zeros((n_steps, 2))
    qddot_check = np.zeros((n_steps, 2))
    for i in range(n_steps):
        q, qd, qdd = reach.q_des[i], reach.qdot_des[i], reach.qddot_des[i]
        F = arm.curl_force(q, qd, b=b)
        n[i] = arm.inverse_dynamics(q, qd, qdd)
        s[i] = -arm.jacobian(q).T @ F
        s_via_id[i] = arm.inverse_dynamics(q, qd, qdd, field_force=F) - n[i]
        qddot_check[i] = arm.dynamics(q, qd, n[i] + s[i], field_b=b)
    return {"n": n, "s": s, "s_via_inverse_dynamics": s_via_id,
            "qddot_from_plant": qddot_check, "qddot_des": np.asarray(reach.qddot_des, float)}


def predict_series(predictor, Z: np.ndarray, context: int | None = None) -> np.ndarray:
    """Evaluate ``predictor.predict`` on each row of Z without updating it."""
    if context is not None:
        predictor.set_context(context)
    return np.stack([np.asarray(predictor.predict(z), dtype=float) for z in Z])


# ----------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------
def _ip(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.asarray(a, float) * np.asarray(b, float)))


def probe_metrics(tau_pre: np.ndarray, tau_post: np.ndarray, n: np.ndarray,
                  s: np.ndarray) -> dict:
    """All per-direction quantities of the spec. Normalised values are NaN if D is not
    finite and positive (no epsilon is added)."""
    tau_pre, tau_post, n, s = (np.asarray(x, dtype=float) for x in (tau_pre, tau_post, n, s))
    delta = tau_post - tau_pre
    D = _ip(s, s)
    c_star = s.mean(axis=0, keepdims=True)
    raw = {
        "D": D,
        "ss_delta_minus_s": _ip(delta - s, delta - s),
        "ip_delta_s": _ip(delta, s),
        "ss_pre_minus_n": _ip(tau_pre - n, tau_pre - n),
        "ss_pre_minus_target": _ip(tau_pre - n - s, tau_pre - n - s),
        "ss_post_minus_target": _ip(tau_post - n - s, tau_post - n - s),
        "ip_pre_minus_n_delta_minus_s": _ip(tau_pre - n, delta - s),
        "ss_s_minus_mean": _ip(s - c_star, s - c_star),
        "ss_delta": _ip(delta, delta),
    }
    ok = math.isfinite(D) and D > 0.0
    out = {**raw, "D_valid": ok}
    if not ok:
        for key in ("g", "r", "h", "p", "e_pre", "e_post", "k", "r_const_min"):
            out[key] = math.nan
        out["beats_best_constant"] = None
        return out
    g = raw["ip_delta_s"] / D
    out.update({
        "g": g,
        "r": raw["ss_delta_minus_s"] / D,
        "h": _ip(delta - g * s, delta - g * s) / D,
        "p": raw["ss_pre_minus_n"] / D,
        "e_pre": raw["ss_pre_minus_target"] / D,
        "e_post": raw["ss_post_minus_target"] / D,
        "k": 2.0 * raw["ip_pre_minus_n_delta_minus_s"] / D,
        "r_const_min": raw["ss_s_minus_mean"] / D,
    })
    out["beats_best_constant"] = bool(
        out["r_const_min"] - out["r"] > TOL_ABS + TOL_REL * abs(out["r_const_min"]))
    out["identity_r_err"] = out["r"] - ((g - 1.0) ** 2 + out["h"])
    out["identity_epost_err"] = out["e_post"] - (out["p"] + out["r"] + out["k"])
    return out


def constant_residual(s: np.ndarray, c: np.ndarray) -> float:
    """||c - s||^2 / <s,s> for a time-constant delta c (shape (2,))."""
    s = np.asarray(s, float)
    c = np.broadcast_to(np.asarray(c, float), s.shape)
    return _ip(c - s, c - s) / _ip(s, s)


# ----------------------------------------------------------------------------------------
# snapshots (non-pickle NPZ)
# ----------------------------------------------------------------------------------------
def snapshot_arrays(readout, kind: str) -> dict:
    """Independent copies of the state needed to rebuild a predictor."""
    if kind == "lms":
        return {"kind": np.array("lms"), "J": np.array(readout.J, dtype=float, copy=True),
                "eta": np.array(float(readout.eta)), "leakage": np.array(float(readout.leakage))}
    if kind == "bias":
        return {"kind": np.array("bias"),
                "base_J": np.array(readout.base_J, dtype=float, copy=True),
                "biases": np.array(readout.biases, dtype=float, copy=True),
                "eta": np.array(float(readout.eta)),
                "n_contexts": np.array(int(readout.n_contexts))}
    if kind == "rls":
        return {"kind": np.array("rls"), "J": np.array(readout.J, dtype=float, copy=True),
                "P": np.array(readout.P, dtype=float, copy=True),
                "beta": np.array(float(readout.beta)), "p0": np.array(float(readout.p0))}
    raise ValueError(f"unsupported readout kind {kind!r} (probe covers lms, bias and rls)")


def save_snapshot(path: Path, readout, kind: str) -> None:
    np.savez(Path(path), **snapshot_arrays(readout, kind))


def load_snapshot(path: Path):
    """Rebuild a predictor from an NPZ snapshot (no pickle). Bias state is restored
    explicitly through ``base_J`` and ``biases``; the read-only ``J`` is never assigned."""
    from .readout import DirectionalIntegralBiasReadout, LinearReadout, RLSReadout

    with np.load(Path(path), allow_pickle=False) as f:
        kind = str(f["kind"])
        if kind == "lms":
            return LinearReadout.from_weights(f["J"], eta=float(f["eta"]),
                                              leakage=float(f["leakage"]))
        if kind == "bias":
            ro = DirectionalIntegralBiasReadout(f["base_J"], n_contexts=int(f["n_contexts"]),
                                                eta=float(f["eta"]))
            if ro.biases.shape != f["biases"].shape:
                raise ValueError("bias shape mismatch on restore")
            ro.biases[...] = f["biases"]
            return ro
        if kind == "rls":
            ro = RLSReadout.from_weights(f["J"], beta=float(f["beta"]), p0=float(f["p0"]))
            if ro.P.shape != f["P"].shape:
                raise ValueError("RLS covariance shape mismatch on restore")
            ro.P = np.array(f["P"], dtype=float, copy=True)
            return ro
    raise ValueError(f"unknown snapshot kind {kind!r}")


def state_equal(a, b, kind: str) -> bool:
    sa, sb = snapshot_arrays(a, kind), snapshot_arrays(b, kind)
    return all(np.array_equal(sa[k], sb[k]) for k in sa)


def frozen_copy(readout):
    return copy.deepcopy(readout)
