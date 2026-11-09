"""Online comparison for the revision baselines (frozen manifests A/B).

Pure helpers used by ``scripts/run_revision_online.py``: manifest verification, model and
readout construction, the eta-selection rule and the per-run metrics. The frozen sources
recorded in manifest A (``protocol.py``, ``readout.py``, ``analysis.py`` metric semantics,
``revision_manifest.validate_ab_curves``) are imported, never modified.

Conventions
-----------
* A run is one ABC protocol (A1 -> B30 -> catch1 -> washout5) for one method, seed, field
  ``b`` and PD multiplier. ``curves`` is ``run_abc_protocol(...)["curves"]``.
* Metrics are taken from cycles whose ``all_finite`` flag is true; a cycle with a non-finite
  reach contributes NaN (never zero) and the run carries the ``diverged`` code.
* ``j_norm`` / ``dj_norm`` are the cycle-end values recorded by ``protocol._cycle_metrics``
  (last reach, final step); their maxima are called *cycle-end maxima*.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .analysis import after_effect_metrics
from .revision_manifest import validate_ab_curves

SELECTION_SEEDS = (0, 1, 2)
EVALUATION_SEEDS = (3, 4, 5, 6, 7)
EVAL_FIELDS = (13.0, -13.0, 25.0)
SELECTION_FIELD = 13.0
PD_MULTIPLIERS = (0.5, 1.0, 2.0)
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
MAIN_THRESHOLD = 0.25
CATCH_SMALL_MM = 1.0
B_CYCLES, WASHOUT_CYCLES = 30, 5

# method -> (model family, readout kind, eta-grid key or None)
METHODS = {
    "B0-CfC": ("cfc", "frozen", None),
    "B0-NCP": ("ncp", "frozen", None),
    "B0-ESN": ("esn", "frozen", None),
    "B0-RBF": ("rbf", "frozen", None),
    "B2": ("cfc", "null", None),
    "CfC-LMS": ("cfc", "lms", "cfc"),
    "CfC-RLS-reg": ("cfc", "rls", None),
    "NCP-LMS": ("ncp", "lms", "ncp"),
    "ESN-LMS": ("esn", "lms", "esn"),
    "RBF-LMS": ("rbf", "lms", "rbf"),
    "CfC-direction-specific-bias": ("cfc", "bias", "bias"),
}
SELECTED_METHODS = ("CfC-LMS", "NCP-LMS", "ESN-LMS", "RBF-LMS", "CfC-direction-specific-bias")
SENSITIVITY_METHODS = ("CfC-LMS", "ESN-LMS", "RBF-LMS", "CfC-direction-specific-bias")
B0_OF_FAMILY = {"cfc": "B0-CfC", "ncp": "B0-NCP", "esn": "B0-ESN", "rbf": "B0-RBF"}
RLS_BETA, RLS_P0 = 1.0, 0.1


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------
# manifest verification
# ----------------------------------------------------------------------------------------
def verify_manifests(a_path: Path, b_path: Path, *, expected_a_sha: str | None,
                     expected_b_sha: str | None, code_root: Path) -> dict:
    """Check hashes, seed split, methods and eta grids of the frozen manifests.

    Raises ``ValueError`` on any mismatch; returns the loaded manifests and a report.
    """
    a_path, b_path = Path(a_path), Path(b_path)
    a_sha, b_sha = sha256(a_path), sha256(b_path)
    problems = []
    if expected_a_sha and a_sha != expected_a_sha:
        problems.append(f"manifest A sha256 {a_sha} != expected {expected_a_sha}")
    if expected_b_sha and b_sha != expected_b_sha:
        problems.append(f"manifest B sha256 {b_sha} != expected {expected_b_sha}")
    A = json.loads(a_path.read_text())
    B = json.loads(b_path.read_text())
    if B.get("manifest_A_sha256") != a_sha:
        problems.append("manifest B does not record manifest A's sha256")
    for key, want in (("selection_seeds", list(SELECTION_SEEDS)),
                      ("evaluation_seeds", list(EVALUATION_SEEDS))):
        if A.get(key) != want or B.get(key) != want:
            problems.append(f"{key} differs from the frozen split {want}")
    if B.get("methods") != list(METHODS):
        problems.append("manifest B method list differs from the runner's METHODS")
    sel = B.get("selection", {})
    if sel.get("selected_methods") != list(SELECTED_METHODS):
        problems.append("selected methods differ")
    if sel.get("field_b") != SELECTION_FIELD:
        problems.append("selection field differs")
    if sel.get("eta_grids_from_manifest_A") != A.get("eta_grid"):
        problems.append("eta grids in B differ from A")
    for key in ("cfc", "ncp", "esn", "rbf", "bias"):
        if len(A["eta_grid"][key]["grid"]) != 7:
            problems.append(f"{key}: eta grid is not 7 points")
    ev = B.get("evaluation", {})
    if [ev.get("field_b_main")] + list(ev.get("field_b_auxiliary", [])) != list(EVAL_FIELDS):
        problems.append("evaluation fields differ")
    sens = B.get("sensitivity", {})
    if (sens.get("methods") != list(SENSITIVITY_METHODS)
            or sens.get("PD_multipliers") != list(PD_MULTIPLIERS)
            or sens.get("seeds") != list(SELECTION_SEEDS)):
        problems.append("sensitivity design differs")
    if B.get("metrics", {}).get("after_effect_threshold_sweep") != list(THRESHOLDS):
        problems.append("threshold sweep differs")
    fp = B.get("fixed_parameters", {})
    if fp.get("RLS_beta") != RLS_BETA or fp.get("RLS_p0") != RLS_P0:
        problems.append("RLS parameters differ")
    # checkpoints
    n_ckpt = 0
    for fam in ("cfc", "ncp", "esn", "rbf"):
        for seed in SELECTION_SEEDS + EVALUATION_SEEDS:
            row = A["families"][fam]["seeds"][str(seed)]
            p = Path(row["checkpoint"])
            if not p.exists():
                problems.append(f"missing checkpoint {p}")
            elif sha256(p) != row["sha256"]:
                problems.append(f"checkpoint sha256 mismatch {p}")
            n_ckpt += 1
    # frozen sources
    changed = [name for name, h in A["source_sha256"].items()
               if sha256(code_root / name) != h]
    if changed:
        problems.append(f"frozen sources changed since manifest A: {changed}")
    if problems:
        raise ValueError("manifest verification failed:\n  " + "\n  ".join(problems))
    return {"A": A, "B": B, "report": {
        "manifest_A": str(a_path.resolve()), "manifest_A_sha256": a_sha,
        "manifest_B": str(b_path.resolve()), "manifest_B_sha256": b_sha,
        "checkpoints_verified": n_ckpt, "frozen_sources_verified": len(A["source_sha256"]),
    }}


# ----------------------------------------------------------------------------------------
# model / readout construction
# ----------------------------------------------------------------------------------------
def load_family_model(A: dict, family: str, seed: int):
    """Rebuild the frozen model of ``family`` / ``seed`` from manifest A."""
    import torch

    from .bases import ESNInverseModel, RBFInverseModel
    from .core import CfCInverseModel, WiredCfCInverseModel

    row = A["families"][family]["seeds"][str(seed)]
    cls, kw = row["model_class"], dict(row["model_kwargs"])
    expected = {"cfc": "CfCInverseModel", "ncp": "WiredCfCInverseModel",
                "esn": "ESNInverseModel", "rbf": "RBFInverseModel"}[family]
    if cls != expected:
        raise ValueError(f"{family} seed {seed}: model_class {cls} != {expected}")
    if cls == "CfCInverseModel":
        model = CfCInverseModel(10, 64, 2)
    elif cls == "WiredCfCInverseModel":
        model = WiredCfCInverseModel(10, 64, 2, **kw)
    elif cls == "ESNInverseModel":
        model = ESNInverseModel(10, 64, 2, **kw)
    else:
        model = RBFInverseModel(10, 64, 2, **kw)
    state = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.freeze_core()
    model.eval()
    return model


def build_readout(kind: str, J0: np.ndarray, eta: float | None):
    """Readout and the J0 reference passed to ``run_abc_protocol``."""
    from .baselines import NullReadout
    from .readout import DirectionalIntegralBiasReadout, LinearReadout, RLSReadout

    J0 = np.asarray(J0, dtype=float)
    if kind == "frozen":
        return LinearReadout.from_weights(J0, eta=0.0, leakage=0.0), J0
    if kind == "null":
        return NullReadout(J0.shape[0], J0.shape[1]), np.zeros_like(J0)
    if kind == "lms":
        return LinearReadout.from_weights(J0, eta=float(eta), leakage=0.0), J0
    if kind == "rls":
        return RLSReadout.from_weights(J0, beta=RLS_BETA, p0=RLS_P0), J0
    if kind == "bias":
        ro = DirectionalIntegralBiasReadout(J0, n_contexts=8, eta=float(eta))
        return ro, ro.J0  # (2, 72): [J0 | zero biases]
    raise ValueError(f"unknown readout kind {kind!r}")


# ----------------------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------------------
def b_last_mm(curves: list[dict]) -> float:
    b = [c for c in curves if c["phase"] == "B"]
    return float(b[-1]["mean_max_perp_dev"] * 1e3)


def select_eta(grid: list[float], runs_by_eta: dict[float, dict[int, list[dict]]]) -> dict:
    """Frozen selection rule (manifest B ``selection``).

    ``runs_by_eta[eta][seed]`` are the curves of selection seed ``seed`` at ``eta``. A grid
    point is valid only if ``validate_ab_curves`` returns no reason for every selection seed.
    Among valid points the smallest selection-seed mean B_last wins; ties go to the smaller
    eta. Every grid point is reported with its reasons.
    """
    rows = []
    for i, eta in enumerate(grid):
        per_seed = {}
        reasons = set()
        for seed in SELECTION_SEEDS:
            curves = runs_by_eta.get(eta, {}).get(seed)
            if curves is None:
                per_seed[seed] = {"reasons": ["missing_run"], "B_last_mm": math.nan}
                reasons.add("missing_run")
                continue
            r = validate_ab_curves(curves)
            reasons.update(r)
            bl = b_last_mm(curves) if not r else math.nan
            if not r and not math.isfinite(bl):
                r = ["ab_nonfinite_metric"]
                reasons.update(r)
            per_seed[seed] = {"reasons": r, "B_last_mm": bl}
        valid = not reasons
        mean_bl = float(np.mean([per_seed[s]["B_last_mm"] for s in SELECTION_SEEDS])) \
            if valid else math.nan
        rows.append({"grid_index": i, "eta": eta, "valid": valid,
                     "reasons": sorted(reasons), "mean_B_last_mm": mean_bl,
                     "per_seed": {str(s): per_seed[s] for s in SELECTION_SEEDS}})
    valid_rows = [r for r in rows if r["valid"]]
    if not valid_rows:
        return {"selected_eta": None, "selected_index": None, "codes": ["no_valid_eta"],
                "rows": rows}
    best = min(valid_rows, key=lambda r: (r["mean_B_last_mm"], r["eta"]))
    codes = ["grid_edge"] if best["grid_index"] in (0, len(grid) - 1) else []
    return {"selected_eta": best["eta"], "selected_index": best["grid_index"],
            "codes": codes, "rows": rows}


# ----------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------
def _phase(curves, phase):
    return [c for c in curves if c["phase"] == phase]


def _mm(cycle: dict, key: str) -> float:
    if not cycle.get("all_finite", False):
        return math.nan
    v = float(cycle[key])
    return v * 1e3 if math.isfinite(v) else math.nan


def final_b_reference(curves: list[dict]) -> np.ndarray | None:
    """Per-direction signed B-final deviation of a B0 run (the after-effect reference)."""
    b = _phase(curves, "B")
    if not b or not b[-1].get("all_finite", False):
        return None
    ref = np.asarray(b[-1]["signed_perp_dev_per_dir"], dtype=float)
    return ref if np.all(np.isfinite(ref)) else None


def adaptation_cycle(b_mm: list[float]) -> tuple[float, str | None]:
    """1-based first B cycle with B_c <= B_first - 0.5 (B_first - B_last)."""
    if len(b_mm) != B_CYCLES or not all(math.isfinite(v) for v in b_mm):
        return math.nan, "diverged"
    first, last = b_mm[0], b_mm[-1]
    if not last < first:
        return math.nan, "not_reached"
    target = first - 0.5 * (first - last)
    for c, v in enumerate(b_mm):
        if v <= target:
            return float(c + 1), None
    return math.nan, "not_reached"  # unreachable: the last cycle always satisfies it


def ae_block(ref: np.ndarray | None, catch: np.ndarray | None) -> dict:
    """``after_effect_metrics`` plus the threshold sweep; NaN when an input is missing."""
    nan = {"ratio": math.nan, "mirror": None, "n_mirror": None,
           "reference_mean_signed_mm": math.nan}
    out = dict(nan)
    if ref is not None and catch is not None:
        m = after_effect_metrics(ref, catch)
        out = {"ratio": m["after_effect_ratio"], "mirror": m["mirror"],
               "n_mirror": m["n_mirror_dirs"], "reference_mean_signed_mm": m["curl_mean_signed_mm"]}
    for th in THRESHOLDS:
        ok = (out["mirror"] is not None and math.isfinite(out["ratio"]))
        out[f"ae_{th:.2f}"] = (bool(out["mirror"] and out["ratio"] >= th) if ok else None)
    return out


def run_metrics(curves: list[dict], b0_family_curves: list[dict] | None,
                b0_cfc_curves: list[dict] | None) -> dict:
    """All per-run metrics of manifest B (``metrics``) with missing codes."""
    codes = set()
    all_finite = all(c.get("all_finite", False) for c in curves)
    if not all_finite:
        codes.add("diverged")
    A = _phase(curves, "A")
    B = _phase(curves, "B")
    C = _phase(curves, "C_catch")
    W = _phase(curves, "C_washout")
    b_mm = [_mm(c, "mean_max_perp_dev") for c in B]
    b_first = b_mm[0] if b_mm else math.nan
    b_last = b_mm[-1] if len(b_mm) == B_CYCLES else math.nan
    ad_cycle, ad_code = adaptation_cycle(b_mm)
    if ad_code:
        codes.add(ad_code)
    catch_ok = bool(C) and C[0].get("all_finite", False)
    catch_dirs = np.asarray(C[0]["signed_perp_dev_per_dir"], float) if catch_ok else None
    if catch_dirs is not None and not np.all(np.isfinite(catch_dirs)):
        catch_dirs = None
    catch_mm = _mm(C[0], "mean_signed_perp_dev") if C else math.nan
    w5_mm = _mm(W[-1], "mean_signed_perp_dev") if len(W) == WASHOUT_CYCLES else math.nan
    if math.isfinite(catch_mm) and abs(catch_mm) < CATCH_SMALL_MM:
        decay = math.nan
        codes.add("catch_small")
    elif math.isfinite(catch_mm) and math.isfinite(w5_mm):
        decay = abs(w5_mm) / abs(catch_mm)
    else:
        decay = math.nan

    def ref_last(ref_curves):
        if ref_curves is None:
            return math.nan
        rb = _phase(ref_curves, "B")
        return _mm(rb[-1], "mean_max_perp_dev") if len(rb) == B_CYCLES else math.nan

    b0_fam_last, b0_cfc_last = ref_last(b0_family_curves), ref_last(b0_cfc_curves)
    fin = [c for c in curves if c.get("all_finite", False)]
    out = {
        "all_phases_finite": all_finite,
        "n_nonfinite_cycles": sum(not c.get("all_finite", False) for c in curves),
        "A_max_perp_mm": _mm(A[0], "mean_max_perp_dev") if A else math.nan,
        "A_endpoint_err_mm": _mm(A[0], "mean_endpoint_error") if A else math.nan,
        "B_first_mm": b_first, "B_last_mm": b_last,
        "B_curve_mm": b_mm,
        "improvement_vs_B0_family": (1.0 - b_last / b0_fam_last
                                     if math.isfinite(b_last) and math.isfinite(b0_fam_last)
                                     else math.nan),
        "improvement_vs_B0_CfC": (1.0 - b_last / b0_cfc_last
                                  if math.isfinite(b_last) and math.isfinite(b0_cfc_last)
                                  else math.nan),
        "adaptation_cycle": ad_cycle,
        "catch_signed_mm": catch_mm,
        "washout5_signed_mm": w5_mm,
        "washout_decay": decay,
        "max_cycle_end_j_norm": (max(float(c["j_norm"]) for c in fin) if fin else math.nan),
        "max_cycle_end_dj_norm": (max(float(c["dj_norm"]) for c in fin) if fin else math.nan),
        "max_cycle_taufb_rms": (max(float(c["taufb_rms"]) for c in fin) if fin else math.nan),
        "ae_family": ae_block(final_b_reference(b0_family_curves) if b0_family_curves else None,
                              catch_dirs),
        "ae_cfc_b0": ae_block(final_b_reference(b0_cfc_curves) if b0_cfc_curves else None,
                              catch_dirs),
    }
    if not all_finite:
        out["max_cycle_end_dj_norm_note"] = "diverged: maxima over finite cycles only"
    out["codes"] = sorted(codes)
    return out


def json_safe(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.generic):
        return json_safe(o.item())
    return o
