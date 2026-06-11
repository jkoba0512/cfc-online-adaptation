"""After-effect judgment for the force-field protocol (Step 4C).

The internal-model claim rests on a *mirror-signed* after-effect: when the curl
field is removed (C catch), an adapted controller deflects to the OPPOSITE side
of where the field pushed it (B0 curl). A feedback-only controller (B2), having
learned nothing, should show little or no mirror deflection.

Reference sign comes from the B0 curl exposure (per-direction signed
perpendicular deviation); the catch sign comes from each condition's C-catch.
"""

from __future__ import annotations

import numpy as np


def get_phase_cycles(out: dict, phase: str) -> list[dict]:
    """Return the per-cycle metric dicts belonging to a protocol phase."""
    return [c for c in out["curves"] if c["phase"] == phase]


def curl_reference_signed(b0_out: dict) -> np.ndarray:
    """Per-direction signed perpendicular deviation under B0 curl exposure.

    Uses the last B cycle of the B0 run (no adaptation -> every B cycle is
    identical, so any cycle works; last is taken for definiteness).
    """
    b_cycles = get_phase_cycles(b0_out, "B")
    return np.asarray(b_cycles[-1]["signed_perp_dev_per_dir"], dtype=float)


def after_effect_metrics(
    curl_ref_signed_per_dir: np.ndarray, catch_signed_per_dir: np.ndarray
) -> dict:
    """Mirror flags and after-effect ratio vs the B0 curl reference.

        curl_sign  = sign(mean curl reference)
        catch_sign = sign(mean catch)
        mirror     = catch_sign == -curl_sign
        ratio      = |mean catch| / |mean curl reference|
        n_mirror   = # directions whose catch sign == -curl sign
    """
    curl = np.asarray(curl_ref_signed_per_dir, dtype=float)
    catch = np.asarray(catch_signed_per_dir, dtype=float)
    curl_mean = float(np.mean(curl))
    catch_mean = float(np.mean(catch))
    curl_sign = int(np.sign(curl_mean))
    catch_sign = int(np.sign(catch_mean))
    mirror = bool(catch_sign == -curl_sign and curl_sign != 0)
    ratio = float(abs(catch_mean) / abs(curl_mean)) if curl_mean != 0 else float("nan")
    # per-direction mirror count
    per_dir = np.sign(catch) == -np.sign(curl)
    n_mirror = int(np.sum(per_dir & (np.sign(curl) != 0)))
    return {
        "curl_mean_signed_mm": curl_mean * 1e3,
        "catch_mean_signed_mm": catch_mean * 1e3,
        "curl_sign": curl_sign,
        "catch_sign": catch_sign,
        "mirror": mirror,
        "after_effect_ratio": ratio,
        "n_mirror_dirs": n_mirror,
        "n_dirs": int(curl.shape[0]),
    }


def after_effect_decay(catch_signed_mm: float, washout_signed_mm_per_cycle: list[float]) -> dict:
    """Simple after-effect decay summary over the C washout cycles.

        decay_ratio = |last washout signed| / |catch signed|
    A small ratio means the after-effect washed out (de-adapted) toward baseline.
    """
    washout = [float(v) for v in washout_signed_mm_per_cycle]
    catch = float(catch_signed_mm)
    last = washout[-1] if washout else float("nan")
    ratio = float(abs(last) / abs(catch)) if catch != 0 else float("nan")
    return {
        "catch_signed_mm": catch,
        "washout_signed_mm_per_cycle": washout,
        "last_washout_signed_mm": last,
        "decay_ratio": ratio,
    }


def claim_supported(method_ae: dict, b2_ae: dict, ratio_threshold: float = 0.25) -> dict:
    """Phase-1 after-effect claim test (Step 4C recommended basis)."""
    method_ok = bool(method_ae["mirror"] and method_ae["after_effect_ratio"] >= ratio_threshold)
    b2_absent = bool((b2_ae["after_effect_ratio"] < 0.1) or (not b2_ae["mirror"]))
    return {
        "method_mirror_and_strong": method_ok,
        "b2_after_effect_absent": b2_absent,
        "claim_supported": bool(method_ok and b2_absent),
        "ratio_threshold": ratio_threshold,
    }
