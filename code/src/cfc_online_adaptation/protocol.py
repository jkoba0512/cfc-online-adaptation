"""A/B/C force-field adaptation mini-protocol (Step 3C).

    A baseline : null field, adapt OFF, 1 cycle      -> reference tracking
    B adaptation: curl field, adapt ON, N cycles     -> learning curve
    C washout  : null field, adapt OFF (catch) then  -> preliminary after-effect
                 optional adapt-ON washout cycles

One readout instance is shared across all cycles, so adaptation accumulates
across the whole protocol; the CfC core hidden state is reset per reach inside
``run_adaptive_closed_loop``. Each cycle iterates the 8 directions in a recorded
order (fixed, or a seeded shuffle).
"""

from __future__ import annotations

import numpy as np

from .adaptation import run_adaptive_closed_loop
from .arm import TwoLinkArm
from .core import CfCInverseModel
from .metrics import endpoint_error, max_abs_perpendicular_deviation
from .tasks import Reach


def _cycle_metrics(phase: str, cycle_idx: int, results: list) -> dict:
    perp_abs, perp_signed, ep_err, fb_sq = [], [], [], []
    finite = True
    for res in results:
        finite = finite and res.finite
        start_xy = res.ee_des[0]
        mabs, msig = max_abs_perpendicular_deviation(res.ee, start_xy, res.target_xy)
        perp_abs.append(mabs)
        perp_signed.append(msig)
        ep_err.append(endpoint_error(res.ee[-1], res.target_xy))
        fb_sq.append(np.mean(np.sum(res.tau_fb**2, axis=1)))
    return {
        "phase": phase,
        "cycle": cycle_idx,
        "mean_max_perp_dev": float(np.mean(perp_abs)),
        "mean_signed_perp_dev": float(np.mean(perp_signed)),
        "signed_perp_dev_per_dir": [float(v) for v in perp_signed],
        "mean_endpoint_error": float(np.mean(ep_err)),
        "taufb_rms": float(np.sqrt(np.mean(fb_sq))),
        "j_norm": float(np.linalg.norm(results[-1].j_norm[-1])),
        "dj_norm": float(results[-1].dj_norm[-1]),
        "all_finite": bool(finite),
    }


def _stack(results: list) -> dict:
    return {
        "ee": np.stack([r.ee for r in results]),
        "ee_des": np.stack([r.ee_des for r in results]),
        "tau_ff": np.stack([r.tau_ff for r in results]),
        "tau_fb": np.stack([r.tau_fb for r in results]),
        "dj_norm": np.stack([r.dj_norm for r in results]),
        "targets": np.stack([r.target_xy for r in results]),
    }


def run_abc_protocol(
    arm: TwoLinkArm,
    model: CfCInverseModel,
    readout,
    reaches: list[Reach],
    Kp: np.ndarray,
    Kd: np.ndarray,
    J0: np.ndarray,
    dt: float = 0.005,
    field_b: float = 13.0,
    b_cycles: int = 10,
    washout_cycles: int = 1,
    washout_adapt: bool = False,
    error_signal: str = "fel",
    forward_model=None,
    sensory_sign: float = 1.0,
    sensory_gain: float = 1.0,
    arm_b: TwoLinkArm | None = None,
    direction_seed: int | None = None,
    device: str = "cpu",
) -> dict:
    """Run the A/B/C protocol with a single persistent readout.

    ``arm_b`` lets the B (exposure) phase use a different plant than the A/C
    phases — e.g. a payload-changed arm — while A and C catch run on the baseline
    ``arm``. With ``arm_b=None`` all phases use ``arm`` (the curl-field case,
    where only ``field_b`` differs between phases).
    """
    n_dir = len(reaches)
    if direction_seed is None:
        order = list(range(n_dir))
    else:
        order = list(np.random.default_rng(direction_seed).permutation(n_dir))

    def run_cycle(phase, cycle_idx, b, adapt, plant):
        results = []
        for d in order:
            if hasattr(readout, "set_context"):
                readout.set_context(d)
            res = run_adaptive_closed_loop(
                plant, model, readout, reaches[d], Kp, Kd, J0,
                dt=dt, field_b=b, adapt=adapt, error_signal=error_signal,
                forward_model=forward_model, sensory_sign=sensory_sign,
                sensory_gain=sensory_gain, device=device,
            )
            results.append(res)
        return _cycle_metrics(phase, cycle_idx, results), results

    arm_exposure = arm_b if arm_b is not None else arm

    curves: list[dict] = []
    saved: dict[str, dict] = {}

    # A baseline: null field, baseline arm, adapt off
    m_a, r_a = run_cycle("A", 0, 0.0, False, arm)
    curves.append(m_a)
    saved["A"] = _stack(r_a)

    # B exposure: field on (and/or payload arm), adapt on
    for c in range(b_cycles):
        m_b, r_b = run_cycle("B", c, field_b, True, arm_exposure)
        curves.append(m_b)
        if c == 0:
            saved["B_first"] = _stack(r_b)
        if c == b_cycles - 1:
            saved["B_last"] = _stack(r_b)

    # C washout: baseline arm, null field, adapt off catch cycle (after-effect)
    m_c, r_c = run_cycle("C_catch", 0, 0.0, False, arm)
    curves.append(m_c)
    saved["C_catch"] = _stack(r_c)
    after_effect = {
        "signed_perp_dev_per_dir": m_c["signed_perp_dev_per_dir"],
        "mean_signed_perp_dev": m_c["mean_signed_perp_dev"],
        "mean_max_perp_dev": m_c["mean_max_perp_dev"],
    }

    # optional adapt-on washout cycles (baseline arm, null field)
    for c in range(washout_cycles):
        m_w, _r_w = run_cycle("C_washout", c, 0.0, washout_adapt, arm)
        curves.append(m_w)

    return {
        "order": [int(d) for d in order],
        "curves": curves,
        "after_effect": after_effect,
        "saved_trajectories": saved,
        "final_J": readout.J.copy(),
    }
