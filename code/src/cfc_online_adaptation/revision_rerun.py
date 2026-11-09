"""400-epoch online rerun (plan ``cfc-online-adaptation_400epoch-online-rerun-plan_2026-11-07``, v0.3).

Pure helpers for ``scripts/run_revision_rerun.py``: design constants, the seed-parametrised
ESN/RBF and eta selection rules, the manifest-A3 model loader (pilot checkpoint format and
ridge state dicts), job enumeration, the append-only run store, curve comparison that
distinguishes bit-identical finite output from a reproduced non-finite failure, the stage-D
diagnostics of one condition, and the paired log-ratio bootstrap.

The frozen sources recorded in manifest A3 (``protocol.py``, ``adaptation.py``, ``readout.py``,
``revision_manifest.validate_ab_curves`` ...) are imported, never modified. Nothing here reads
manifest A/B of the 2026-11-02 experiment or changes its outputs.

Seeds: selection 100-102 (pilot ``epoch0400.pt`` reused), evaluation 110-119 (new 400-epoch
series). NCP wiring seed = 22200 + model seed (the 2026-11-02 experiment used 22222 for every
seed). All three CfC configurations are compared at their final 400th epoch; this is a
comparison of implementation configurations, not a single-factor test.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import budget_pilot as bp
from . import frozen_closed_loop as fc
from . import input_replay as ir
from . import readout_probe as rp
from . import revision_online as ro
from .revision_manifest import validate_ab_curves

# ----------------------------------------------------------------------------------------
# design constants (plan v0.3)
# ----------------------------------------------------------------------------------------
SELECTION_SEEDS = (100, 101, 102)
EVALUATION_SEEDS = tuple(range(110, 120))
ALL_SEEDS = SELECTION_SEEDS + EVALUATION_SEEDS
WIRING_SEEDS = {s: 22200 + s for s in ALL_SEEDS}  # 100 -> 22300, 110 -> 22310, ...
OLD_NCP_WIRING_SEED = 22222  # 2026-11-02 experiment: one wiring for every seed
ETA_MULTIPLIERS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
EVAL_FIELDS = (13.0, -13.0, 25.0)
SELECTION_FIELD = 13.0
PD_MULTIPLIERS = (0.5, 1.0, 2.0)
SENSITIVITY_EXTRA_PD = (0.5, 2.0)
B_CYCLES, WASHOUT_CYCLES = 30, 5
EPOCHS, CKPT_EVERY = 400, 10
BOOTSTRAP_N, BOOTSTRAP_SEED = 10000, 20261107

CENTER_XY, RADIUS, T_MOVE, DT = (0.0, 0.45), 0.10, 0.5, 0.005
KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])

CFC_FAMILIES = {"dense": "dense_backbone", "ncp": "ncp_wired", "fc": "fc_wired"}
RIDGE_FAMILIES = ("esn", "rbf")
FAMILIES = ("dense", "ncp", "fc", "esn", "rbf")

# method -> (model family, readout kind, eta-grid key or None)
METHODS = {
    "B0-dense-CfC": ("dense", "frozen", None),
    "B0-NCP-CfC": ("ncp", "frozen", None),
    "B0-FC-CfC": ("fc", "frozen", None),
    "B0-ESN": ("esn", "frozen", None),
    "B0-RBF": ("rbf", "frozen", None),
    "PD-only": ("dense", "null", None),
    "dense-CfC-LMS": ("dense", "lms", "dense"),
    "NCP-CfC-LMS": ("ncp", "lms", "ncp"),
    "FC-CfC-LMS": ("fc", "lms", "fc"),
    "ESN-LMS": ("esn", "lms", "esn"),
    "RBF-LMS": ("rbf", "lms", "rbf"),
    "dense-CfC-bias": ("dense", "bias", "bias"),
    "dense-CfC-RLS": ("dense", "rls", None),
}
SELECTED_METHODS = ("dense-CfC-LMS", "NCP-CfC-LMS", "FC-CfC-LMS", "ESN-LMS", "RBF-LMS",
                    "dense-CfC-bias")
ADAPTIVE_METHODS = SELECTED_METHODS + ("dense-CfC-RLS",)
B0_METHODS = ("B0-dense-CfC", "B0-NCP-CfC", "B0-FC-CfC", "B0-ESN", "B0-RBF")
B0_OF_FAMILY = {"dense": "B0-dense-CfC", "ncp": "B0-NCP-CfC", "fc": "B0-FC-CfC",
                "esn": "B0-ESN", "rbf": "B0-RBF"}
PRIMARY_PAIR = ("NCP-CfC-LMS", "dense-CfC-LMS")  # numerator, denominator of the log ratio
PD_ONLY_CHECK_FAMILIES = ("dense", "esn")
DIAGNOSTIC_FIELD = 13.0
CURVE_FLOATS = ("mean_max_perp_dev", "mean_signed_perp_dev", "mean_endpoint_error",
                "taufb_rms", "j_norm", "dj_norm")
RECORD_ARRAYS = ("q", "qdot", "tau", "tau_ff", "tau_fb", "ee", "ee_des")

EXPECTED_JOB_COUNTS = {"select": 162, "evaluate_adaptive": 210, "evaluate_B0": 150,
                       "evaluate_PD_only": 3, "diagnostics_rerun": 70,
                       "sensitivity_adaptive_new": 324, "sensitivity_B0": 45, "total": 964,
                       "frozen_cycles_not_counted": 140}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------
# task
# ----------------------------------------------------------------------------------------
def make_task():
    """Arm, centre and the eight joint-space reaches shared by every run."""
    from .arm import ArmParams, TwoLinkArm
    from .tasks import joint_space_reaches

    arm = TwoLinkArm(ArmParams())
    cq = arm.inverse_kinematics(np.array(CENTER_XY), elbow="down")
    reaches = joint_space_reaches(arm, cq, radius=RADIUS, T=T_MOVE, dt=DT, n_targets=8)
    return arm, cq, reaches


_DATA_CACHE: dict = {}


def split_data(seed: int):
    """(train, val, audit) for ``seed`` from the regenerated datasets (cached per process)."""
    if "ds" not in _DATA_CACHE:
        arm, cq = bp.make_arm_and_center()
        _DATA_CACHE["ds"] = bp.training_dataset(arm, cq)
        _DATA_CACHE["audit"] = bp.audit_dataset(arm, cq)
    tr, va = bp.split_for_seed(_DATA_CACHE["ds"], seed)
    return tr, va, _DATA_CACHE["audit"]


def data_hashes(seeds) -> dict:
    ds_tr, _, audit = split_data(seeds[0])
    ds = _DATA_CACHE["ds"]
    splits = {}
    for s in seeds:
        tr, va, _ = split_data(s)
        splits[str(s)] = {"n_train": len(tr), "n_val": len(va),
                          "train_sha256_x": sha256_array(tr.x), "val_sha256_x": sha256_array(va.x),
                          "val_radii": sorted({m["radius"] for m in va.meta}),
                          "val_directions": bp.direction_index(va).tolist()}
    return {"center_xy": list(bp.CENTER_XY), "train_radii": list(bp.TRAIN_RADII),
            "directions": bp.N_TARGETS, "duration": bp.DURATION, "dt": bp.DT,
            "n_sequences": len(ds), "val_frac": bp.VAL_FRAC,
            "split_rule": "dataset.train_val_split(seed=model seed); shared by all families",
            "x_sha256": sha256_array(ds.x), "tau_sha256": sha256_array(ds.tau), "splits": splits,
            "audit_set": {"radii": list(bp.AUDIT_RADII), "n_sequences": len(audit),
                          "x_sha256": sha256_array(audit.x), "tau_sha256": sha256_array(audit.tau),
                          "use": "offline quality record only; never used for selection"}}


def verify_data_hashes(spec_data: dict, seed: int) -> None:
    tr, va, audit = split_data(seed)
    ds = _DATA_CACHE["ds"]
    want = {"x": spec_data["x_sha256"], "tau": spec_data["tau_sha256"],
            "audit_x": spec_data["audit_set"]["x_sha256"],
            "train_x": spec_data["splits"][str(seed)]["train_sha256_x"],
            "val_x": spec_data["splits"][str(seed)]["val_sha256_x"]}
    got = {"x": sha256_array(ds.x), "tau": sha256_array(ds.tau), "audit_x": sha256_array(audit.x),
           "train_x": sha256_array(tr.x), "val_x": sha256_array(va.x)}
    bad = [k for k in want if want[k] != got[k]]
    if bad:
        raise ValueError(f"seed {seed}: regenerated data differ from the spec hashes: {bad}")


# ----------------------------------------------------------------------------------------
# ESN / RBF with explicit seeds (M3)
# ----------------------------------------------------------------------------------------
def build_ridge_model(family: str, setting, train_x: np.ndarray, seed: int):
    from .bases import build_esn, build_rbf

    if family == "esn":
        return build_esn(train_x, *setting, seed=seed)
    if family == "rbf":
        return build_rbf(train_x, setting, seed=seed)
    raise ValueError(family)


def choose_family_for_seeds(family: str, split_data_by_seed: dict, seeds, *, settings,
                            lam_grid) -> tuple[dict, list[dict]]:
    """Same rule as ``run_revision_offline.choose_family`` (mean validation NMSE over the
    selection seeds after the per-seed ridge choice), with the seeds passed explicitly.
    The grid is never extended: an edge selection is flagged, not repaired."""
    from .bases import fit_and_validate

    seeds = tuple(int(s) for s in seeds)
    candidates = []
    for setting in settings:
        rows = []
        for seed in seeds:
            train, val = split_data_by_seed[seed]
            model = build_ridge_model(family, setting, train.x, seed)
            fit = fit_and_validate(model, train.x, train.tau, val.x, val.tau, lam_grid)
            rows.append({"seed": seed, "lambda": fit["lam"], "val_nmse": fit["val_nmse"]})
        candidates.append({"setting": list(setting) if family == "esn" else setting,
                           "selection_rows": rows,
                           "mean_val_nmse": float(np.mean([r["val_nmse"] for r in rows]))})
    best = min(candidates, key=lambda r: r["mean_val_nmse"])
    idx = candidates.index(best)
    best = dict(best, grid_index=idx, n_candidates=len(candidates),
                at_grid_edge=(idx in (0, len(candidates) - 1)), selection_seeds=list(seeds))
    return best, candidates


def _model_val_pred(model, x: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(x, dtype=np.float32)))[0].cpu().numpy().astype(np.float32)


def ridge_row(model, family: str, train, val, audit, path: Path, *, fit: dict, setting) -> dict:
    """Save the ridge-fitted model's state dict and describe it (A3 checkpoint-index row)."""
    import torch

    from .bases import normalised_mse

    model.freeze_core()
    model.eval()
    tmp = path.with_name(path.name + ".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)
    ptr, pva = _model_val_pred(model, train.x), _model_val_pred(model, val.x)
    ev_audit = bp.evaluate(model, audit)
    return {"family": family, "seed": int(model.seed), "format": "state_dict",
            "checkpoint": str(path.resolve()), "sha256": sha256_file(path),
            "model_class": type(model).__name__, "model_kwargs": model.config(),
            "selected_setting": list(setting) if family == "esn" else setting,
            "lambda": fit["lam"], "lambda_grid": fit["grid"],
            "train_nmse": normalised_mse(ptr.astype(np.float64), train.tau),
            "val_nmse": normalised_mse(pva.astype(np.float64), val.tau),
            "audit_nmse": ev_audit["nmse"], "audit_nmse_by_direction": ev_audit["nmse_by_direction"],
            "feature_energy_train": bp.feature_energy(model, train.x),
            "readout_norm": float(np.linalg.norm(model.export_readout_weights())),
            "verification": {"val_input_sha256": sha256_array(val.x.astype(np.float32)),
                             "val_pred_sha256": sha256_array(pva)}}


# ----------------------------------------------------------------------------------------
# CfC checkpoint index rows and the A3 loader (M2)
# ----------------------------------------------------------------------------------------
def cfc_row_from_checkpoint(family: str, seed: int, ckpt_path: Path, *, origin: str,
                            series_dir: Path | None = None) -> dict:
    """Describe a pilot-format checkpoint (``budget_pilot`` format) as an A3 index row and
    verify it: fresh instance + saved state reproduces the saved model's forward bit for bit,
    masks match the stored digests and a fresh instance, epoch is the final epoch."""
    config = CFC_FAMILIES[family]
    wiring_seed = None if family == "dense" else WIRING_SEEDS[seed]
    tr, va, audit = split_data(seed)
    builder = lambda: bp.build_model(config, seed, wiring_seed)  # noqa: E731
    ck = bp.load_checkpoint(ckpt_path)
    rc = bp.reload_check(builder, ckpt_path, va.x)
    fresh_masks = [m["sha256"] for m in bp.audit_model(builder())["masks"]]
    rc["mask_matches_fresh_instance"] = rc["mask_sha256"] == fresh_masks
    model = builder()
    model.load_state_dict(ck["model_state"])
    model.freeze_core()
    model.eval()
    summ = ck.get("summary") or {}
    ev_audit = bp.evaluate(model, audit)
    row = {"family": family, "config": config, "seed": int(seed), "wiring_seed": wiring_seed,
           "format": "budget_pilot_checkpoint_v%d" % ck["format_version"],
           "checkpoint": str(Path(ckpt_path).resolve()), "sha256": sha256_file(ckpt_path),
           "epoch": int(ck["epoch"]), "epoch_is_final": int(ck["epoch"]) == EPOCHS,
           "origin": origin, "series_dir": None if series_dir is None else str(series_dir.resolve()),
           "model_class": type(model).__name__,
           "checkpoint_config": {k: v for k, v in ck["config"].items()},
           "checkpoint_provenance": ck.get("provenance", {}),
           "train_nmse": (summ.get("train") or {}).get("nmse", bp.evaluate(model, tr)["nmse"]),
           "val_nmse": (summ.get("val") or {}).get("nmse", bp.evaluate(model, va)["nmse"]),
           "audit_nmse": ev_audit["nmse"], "audit_nmse_by_direction": ev_audit["nmse_by_direction"],
           "feature_energy_train": summ.get("feature_energy_train", bp.feature_energy(model, tr.x)),
           "readout_norm": float(np.linalg.norm(model.export_readout_weights())),
           "reload_check": rc, "audit": bp.audit_model(model),
           "verification": {"val_input_sha256": ck["verification"]["input_sha256"],
                            "val_pred_sha256": ck["verification"]["val_pred_sha256"]}}
    row["reload_ok"] = bool(rc["forward_bit_identical_to_saved_model"] and rc["input_matches_saved_input"]
                            and rc["masks_equal_to_saved_digests"] and rc["mask_matches_fresh_instance"]
                            and row["epoch_is_final"])
    return row


def index_row(A3: dict, family: str, seed: int) -> dict:
    try:
        return A3["checkpoint_index"][family][str(seed)]
    except KeyError as e:
        raise KeyError(f"manifest A3 has no checkpoint for {family} seed {seed}") from e


def load_a3_model(A3: dict, family: str, seed: int, *, verify: bool = True):
    """Rebuild the frozen model of ``family`` / ``seed`` from manifest A3.

    CfC configurations come from pilot-format checkpoints (``budget_pilot.build_model`` +
    ``model_state``); ESN/RBF from plain state dicts. With ``verify`` the file hash and the
    forward on the seed's validation input are checked against the index row (the row's
    digests were computed from the model that was saved, so this is not a self-comparison).
    """
    import torch

    from .bases import ESNInverseModel, RBFInverseModel

    row = index_row(A3, family, seed)
    path = Path(row["checkpoint"])
    if verify and sha256_file(path) != row["sha256"]:
        raise ValueError(f"{family} seed {seed}: checkpoint sha256 mismatch {path}")
    if family in CFC_FAMILIES:
        if row["config"] != CFC_FAMILIES[family] or int(row["seed"]) != int(seed):
            raise ValueError(f"{family} seed {seed}: index row config/seed mismatch")
        if family != "dense" and row["wiring_seed"] != WIRING_SEEDS[seed]:
            raise ValueError(f"{family} seed {seed}: wiring seed {row['wiring_seed']} != "
                             f"{WIRING_SEEDS[seed]}")
        ck = bp.load_checkpoint(path)
        if int(ck["epoch"]) != EPOCHS:
            raise ValueError(f"{family} seed {seed}: checkpoint epoch {ck['epoch']} != {EPOCHS}")
        model = bp.build_model(row["config"], int(seed), row["wiring_seed"])
        model.load_state_dict(ck["model_state"])
    elif family in RIDGE_FAMILIES:
        expected = {"esn": "ESNInverseModel", "rbf": "RBFInverseModel"}[family]
        if row["model_class"] != expected:
            raise ValueError(f"{family} seed {seed}: model_class {row['model_class']} != {expected}")
        kw = dict(row["model_kwargs"])
        model = (ESNInverseModel(10, 64, 2, **kw) if family == "esn"
                 else RBFInverseModel(10, 64, 2, **kw))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    else:
        raise ValueError(family)
    model.freeze_core()
    model.eval()
    if verify:
        _, va, _ = split_data(seed)
        xv = va.x.astype(np.float32)
        v = row["verification"]
        if sha256_array(xv) != v["val_input_sha256"]:
            raise ValueError(f"{family} seed {seed}: validation input differs from the index row")
        if sha256_array(_model_val_pred(model, xv)) != v["val_pred_sha256"]:
            raise ValueError(f"{family} seed {seed}: forward differs from the saved model")
    return model


# ----------------------------------------------------------------------------------------
# eta grid and selection with explicit seeds
# ----------------------------------------------------------------------------------------
def eta_grid_from_index(index: dict, selection_seeds=SELECTION_SEEDS) -> dict:
    """center = 1e-3 * g_dense / g_family with g = mean selection-seed training feature energy
    of the final-epoch models; bias uses g = 1 (center = 1e-3 * g_dense)."""
    def g(fam):
        return float(np.mean([index[fam][str(s)]["feature_energy_train"] for s in selection_seeds]))

    g_dense = g("dense")
    out = {}
    for fam in FAMILIES:
        gf = g(fam)
        center = 1e-3 * g_dense / gf
        out[fam] = {"selection_feature_energy": gf, "reference_feature_energy_dense": g_dense,
                    "center": center, "grid": [center * m for m in ETA_MULTIPLIERS]}
    out["bias"] = {"selection_feature_energy": 1.0, "reference_feature_energy_dense": g_dense,
                   "center": 1e-3 * g_dense, "grid": [1e-3 * g_dense * m for m in ETA_MULTIPLIERS]}
    return out


def select_eta_for_seeds(grid: list[float], runs_by_eta: dict, seeds) -> dict:
    """``revision_online.select_eta`` with the selection seeds passed explicitly.

    Validity: ``validate_ab_curves`` returns no reason for every seed (A0 and B0..29 present,
    finite, B29 < B0, no B above 2 x B0). Among valid grid points the smallest mean B_last wins;
    ties go to the smaller eta. Edge selections carry the ``grid_edge`` code; the grid is not
    extended.
    """
    seeds = tuple(int(s) for s in seeds)
    rows = []
    for i, eta in enumerate(grid):
        per_seed, reasons = {}, set()
        for seed in seeds:
            curves = runs_by_eta.get(eta, {}).get(seed)
            if curves is None:
                per_seed[seed] = {"reasons": ["missing_run"], "B_last_mm": math.nan}
                reasons.add("missing_run")
                continue
            r = validate_ab_curves(curves)
            reasons.update(r)
            bl = ro.b_last_mm(curves) if not r else math.nan
            if not r and not math.isfinite(bl):
                r = ["ab_nonfinite_metric"]
                reasons.update(r)
            per_seed[seed] = {"reasons": r, "B_last_mm": bl}
        valid = not reasons
        mean_bl = float(np.mean([per_seed[s]["B_last_mm"] for s in seeds])) if valid else math.nan
        rows.append({"grid_index": i, "eta": eta, "valid": valid, "reasons": sorted(reasons),
                     "mean_B_last_mm": mean_bl, "per_seed": {str(s): per_seed[s] for s in seeds}})
    valid_rows = [r for r in rows if r["valid"]]
    if not valid_rows:
        return {"selected_eta": None, "selected_index": None, "codes": ["no_valid_eta"],
                "rows": rows, "seeds": list(seeds)}
    best = min(valid_rows, key=lambda r: (r["mean_B_last_mm"], r["eta"]))
    codes = ["grid_edge"] if best["grid_index"] in (0, len(grid) - 1) else []
    return {"selected_eta": best["eta"], "selected_index": best["grid_index"], "codes": codes,
            "rows": rows, "seeds": list(seeds)}


# ----------------------------------------------------------------------------------------
# jobs and the run store
# ----------------------------------------------------------------------------------------
def make_job(method, seed, b, pd=1.0, eta=None, *, stage, b_cycles=B_CYCLES,
             washout_cycles=WASHOUT_CYCLES, family_override=None):
    j = {"stage": stage, "method": method, "seed": int(seed), "b": float(b), "pd": float(pd),
         "eta": None if eta is None else float(eta), "b_cycles": int(b_cycles),
         "washout_cycles": int(washout_cycles)}
    if family_override is not None:
        j["family_override"] = family_override
    return j


def job_key(j: dict) -> str:
    eta = "none" if j.get("eta") is None else repr(float(j["eta"]))
    fam = f"|fam{j['family_override']}" if j.get("family_override") else ""
    return (f"{j['method']}|s{j['seed']}|b{j['b']:+g}|pd{j['pd']:g}|eta{eta}"
            f"|B{j['b_cycles']}|W{j['washout_cycles']}{fam}")


def job_family(j: dict) -> str:
    return j.get("family_override") or METHODS[j["method"]][0]


def restore_curves(r: dict) -> dict:
    """JSON null -> NaN (strict JSON stores non-finite values as null)."""
    for c in r["curves"]:
        for k in CURVE_FLOATS:
            if c.get(k) is None:
                c[k] = math.nan
        c["signed_perp_dev_per_dir"] = [math.nan if v is None else v
                                        for v in c["signed_perp_dev_per_dir"]]
    return r


class RunStore:
    """Append-only runs; preserve and repair an interrupted final line, reject conflicts."""

    @staticmethod
    def scientific_record(r):
        return ro.json_safe({k: v for k, v in r.items() if k not in ("pid", "wall_s", "stage")})

    def __init__(self, path: Path):
        self.path = Path(path)
        self.runs: dict[str, dict] = {}
        self.recovery = None
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        kept = []
        repair = False
        for i, line in enumerate(lines):
            if not line.strip():
                kept.append(line)
                continue
            try:
                r = restore_curves(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if any(x.strip() for x in lines[i + 1:]):
                    raise ValueError("run store contains corruption before its final record") from None
                repair = True
                break
            key = job_key(r)
            if key in self.runs and self.scientific_record(self.runs[key]) != self.scientific_record(r):
                raise ValueError(f"conflicting duplicate job: {key}")
            self.runs.setdefault(key, r)
            kept.append(line)
        if raw and not raw.endswith(b"\n"):
            repair = True
        if repair:
            backup = self.path.with_name(self.path.name + f".before-recovery-{time.time_ns()}")
            backup.write_bytes(raw)
            recovered = b"".join(kept).rstrip(b"\r\n")
            atomic_write_text(self.path, (recovered.decode() + "\n") if recovered else "")
            self.recovery = {"backup": str(backup), "kept_jobs": len(self.runs),
                             "bytes_before": len(raw), "bytes_after": self.path.stat().st_size}
            atomic_write_text(self.path.with_name(backup.name + ".json"), json.dumps(self.recovery))

    def add(self, r: dict) -> None:
        key = job_key(r)
        if key in self.runs:
            if self.scientific_record(self.runs[key]) != self.scientific_record(r):
                raise ValueError(f"conflicting duplicate job: {key}")
            return
        with self.path.open("a") as f:
            f.write(json.dumps(ro.json_safe(r), allow_nan=False) + "\n")
            f.flush()
        self.runs[key] = r

    def get(self, j: dict) -> dict | None:
        return self.runs.get(job_key(j))

    def missing(self, jobs: list[dict]) -> list[dict]:
        todo, seen = [], set()
        for j in jobs:
            k = job_key(j)
            if k not in self.runs and k not in seen:
                todo.append(j)
                seen.add(k)
        return todo


def run_protocol_job(arm, reaches, model, job: dict) -> dict:
    """One ABC protocol for a loaded model (fresh readout every call; no shared state)."""
    from .protocol import run_abc_protocol

    kind = METHODS[job["method"]][1]
    J0 = model.export_readout_weights().astype(float)
    readout, J0_ref = ro.build_readout(kind, J0, job.get("eta"))
    pd = job["pd"]
    t0 = time.perf_counter()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = run_abc_protocol(arm, model, readout, reaches, KP * pd, KD * pd, J0_ref, dt=DT,
                               field_b=job["b"], b_cycles=job["b_cycles"],
                               washout_cycles=job["washout_cycles"], washout_adapt=True,
                               error_signal="fel", device="cpu")
    return {**job, "wall_s": time.perf_counter() - t0, "pid": os.getpid(), "order": out["order"],
            "curves": out["curves"]}


def jobs_select(A3: dict) -> list[dict]:
    return [make_job(m, s, SELECTION_FIELD, 1.0, eta, stage="select")
            for m in SELECTED_METHODS
            for eta in A3["eta_grid"][METHODS[m][2]]["grid"]
            for s in SELECTION_SEEDS]


def jobs_evaluate(selection: dict) -> list[dict]:
    jobs = []
    for m in ADAPTIVE_METHODS:
        gkey = METHODS[m][2]
        eta = selection[m]["selected_eta"] if gkey else None
        if gkey and eta is None:
            continue  # no_valid_eta: reported in selection, not evaluated
        jobs += [make_job(m, s, b, 1.0, eta, stage="evaluate")
                 for s in EVALUATION_SEEDS for b in EVAL_FIELDS]
    jobs += [make_job(m, s, b, 1.0, None, stage="evaluate")
             for m in B0_METHODS for s in EVALUATION_SEEDS for b in EVAL_FIELDS]
    jobs += [make_job("PD-only", EVALUATION_SEEDS[0], b, 1.0, None, stage="evaluate")
             for b in EVAL_FIELDS]
    return jobs


def jobs_sensitivity(A3: dict) -> list[dict]:
    jobs = []
    for pd in PD_MULTIPLIERS:
        for s in SELECTION_SEEDS:
            jobs += [make_job(m, s, SELECTION_FIELD, pd, None, stage="sensitivity")
                     for m in B0_METHODS]
            if pd == 1.0:
                continue  # PD x1 adaptive runs are the selection runs (same key)
            for m in SELECTED_METHODS:
                jobs += [make_job(m, s, SELECTION_FIELD, pd, eta, stage="sensitivity")
                         for eta in A3["eta_grid"][METHODS[m][2]]["grid"]]
    return jobs


def jobs_sensitivity_pd1_reused(A3: dict) -> list[dict]:
    return [make_job(m, s, SELECTION_FIELD, 1.0, eta, stage="select")
            for s in SELECTION_SEEDS for m in SELECTED_METHODS
            for eta in A3["eta_grid"][METHODS[m][2]]["grid"]]


def diagnostic_conditions(selection: dict) -> list[dict]:
    conds = []
    for m in ADAPTIVE_METHODS:
        gkey = METHODS[m][2]
        eta = selection[m]["selected_eta"] if gkey else None
        if gkey and eta is None:
            continue
        for s in EVALUATION_SEEDS:
            conds.append({"method": m, "seed": s, "b": DIAGNOSTIC_FIELD, "pd": 1.0, "eta": eta,
                          "b_cycles": B_CYCLES})
    return conds


def job_counts(A3: dict, selection: dict | None = None) -> dict:
    """Protocol-run counts of the design (distinct keys; PD x1 sensitivity reuses selection)."""
    sel = jobs_select(A3)
    sens = jobs_sensitivity(A3)
    out = {"select": len(sel),
           "sensitivity_adaptive_new": sum(not m.startswith("B0-") for m in (j["method"] for j in sens)),
           "sensitivity_B0": sum(j["method"].startswith("B0-") for j in sens)}
    if selection is not None:
        ev = jobs_evaluate(selection)
        out["evaluate_adaptive"] = sum(j["method"] in ADAPTIVE_METHODS for j in ev)
        out["evaluate_B0"] = sum(j["method"] in B0_METHODS for j in ev)
        out["evaluate_PD_only"] = sum(j["method"] == "PD-only" for j in ev)
        out["diagnostics_rerun"] = len(diagnostic_conditions(selection))
        out["total"] = sum(v for k, v in out.items())
    return out


# ----------------------------------------------------------------------------------------
# curve comparison: bit-identical finite output vs reproduced non-finite failure
# ----------------------------------------------------------------------------------------
def _same_value(x, y) -> tuple[bool, bool]:
    """(equal, nonfinite_involved). Non-finite values are equal when their pattern is
    (NaN with NaN; +inf with +inf; -inf with -inf)."""
    x, y = float(x), float(y)
    fx, fy = math.isfinite(x), math.isfinite(y)
    if fx and fy:
        return x == y, False
    if math.isnan(x) and math.isnan(y):
        return True, True
    return (x == y), True  # inf with matching sign, or a finite/non-finite mismatch


def compare_cycle(a: dict, b: dict) -> dict:
    eq, nonfin = True, False
    for k in CURVE_FLOATS:
        e, n = _same_value(a[k], b[k])
        eq, nonfin = eq and e, nonfin or n
    pa, pb = a["signed_perp_dev_per_dir"], b["signed_perp_dev_per_dir"]
    if len(pa) != len(pb):
        eq = False
    else:
        for x, y in zip(pa, pb):
            e, n = _same_value(x, y)
            eq, nonfin = eq and e, nonfin or n
    if bool(a.get("all_finite")) != bool(b.get("all_finite")):
        eq = False
    return {"equal": eq, "nonfinite_involved": nonfin or not bool(a.get("all_finite"))}


def compare_curves(new: list[dict], old: list[dict], phases=("A", "B", "C_catch")) -> dict:
    """status: ``bit_identical`` (all compared values finite and equal), ``nonfinite_reproduced``
    (the same non-finite pattern and failure flags; a scientific failure, not a determinism
    failure) or ``mismatch``."""
    n = [c for c in new if c["phase"] in phases]
    o = [c for c in old if c["phase"] in phases]
    if [(c["phase"], c["cycle"]) for c in n] != [(c["phase"], c["cycle"]) for c in o]:
        return {"status": "mismatch", "reason": "cycle_structure", "n_cycles": (len(n), len(o))}
    eq, nonfin, first_bad = True, False, None
    for a, b in zip(n, o):
        c = compare_cycle(a, b)
        if not c["equal"] and first_bad is None:
            first_bad = (a["phase"], a["cycle"])
        eq, nonfin = eq and c["equal"], nonfin or c["nonfinite_involved"]
    status = "mismatch" if not eq else ("nonfinite_reproduced" if nonfin else "bit_identical")
    return {"status": status, "n_cycles": len(n), "nonfinite_involved": nonfin,
            "first_mismatch": first_bad}


# ----------------------------------------------------------------------------------------
# paired log-ratio bootstrap (descriptive; not a superiority test)
# ----------------------------------------------------------------------------------------
def paired_log_ratio_bootstrap(values_num: dict, values_den: dict, *, n_boot: int = BOOTSTRAP_N,
                               seed: int = BOOTSTRAP_SEED) -> dict:
    """``values_*[seed] = metric``; pairs with both values finite and positive enter
    log(num/den). Percentile bootstrap over pairs with a fixed RNG seed. Excluded pairs and
    their reasons are listed; the interval is not a hypothesis test."""
    pairs, used, excluded = [], [], []
    for s in sorted(set(values_num) | set(values_den)):
        x, y = values_num.get(s, math.nan), values_den.get(s, math.nan)
        ok = all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (x, y))
        row = {"seed": s, "num": x, "den": y, "used": ok,
               "log_ratio": math.log(x / y) if ok else math.nan,
               "reason": "" if ok else ("missing" if s not in values_num or s not in values_den
                                        else "non-finite or non-positive")}
        pairs.append(row)
        (used if ok else excluded).append(row)
    logs = np.array([r["log_ratio"] for r in used], dtype=float)
    out = {"n_pairs_total": len(pairs), "n_pairs_used": len(used), "n_excluded": len(excluded),
           "excluded": [{k: r[k] for k in ("seed", "reason")} for r in excluded],
           "pairs": pairs, "n_boot": n_boot, "bootstrap_seed": seed,
           "n_num_smaller": int(np.sum(logs < 0)), "n_den_smaller": int(np.sum(logs > 0)),
           "n_equal": int(np.sum(logs == 0)),
           "note": "descriptive paired interval over shared data-split pairs; not a superiority test; "
                   "pair differences include configuration, init and wiring-seed effects"}
    if len(used) == 0:
        out.update({"mean_log_ratio": math.nan, "ci95": [math.nan, math.nan], "ratio_of_geomeans": math.nan})
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(logs), size=(n_boot, len(logs)))
    means = logs[idx].mean(axis=1)
    out.update({"mean_log_ratio": float(logs.mean()),
                "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                "ratio_of_geomeans": float(math.exp(logs.mean()))})
    return out


# ----------------------------------------------------------------------------------------
# stage D: one condition
# ----------------------------------------------------------------------------------------
def array_state_equal(a, b) -> bool:
    """Finite bytes must match; non-finite positions and infinity signs must match."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind not in "fc":
        return bool(np.array_equal(a, b))
    if a.dtype.kind == "c":
        return array_state_equal(a.real, b.real) and array_state_equal(a.imag, b.imag)
    finite = np.isfinite(a)
    return (np.array_equal(finite, np.isfinite(b))
            and np.array_equal(np.isnan(a), np.isnan(b))
            and np.array_equal(np.isposinf(a), np.isposinf(b))
            and np.array_equal(np.isneginf(a), np.isneginf(b))
            and a[finite].tobytes() == b[finite].tobytes())


def readout_state_equal(a, b, kind) -> bool:
    sa, sb = rp.snapshot_arrays(a, kind), rp.snapshot_arrays(b, kind)
    return sa.keys() == sb.keys() and all(array_state_equal(sa[k], sb[k]) for k in sa)


def _frozen_cycle(arm, reaches, model, predictor, J0_ref, b: float, pd: float, kind: str):
    """Eight reaches with a fixed readout (adapt off); context set per direction as the protocol
    does. Returns the AdaptiveResults and whether the readout state stayed unchanged."""
    from .adaptation import run_adaptive_closed_loop

    actor = copy.deepcopy(predictor)
    state0 = rp.snapshot_arrays(actor, kind)
    results = []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for d, reach in enumerate(reaches):
            if hasattr(actor, "set_context"):
                actor.set_context(d)
            results.append(run_adaptive_closed_loop(arm, model, actor, reach, KP * pd, KD * pd, J0_ref,
                                                    dt=DT, field_b=b, adapt=False, error_signal="fel",
                                                    device="cpu"))
    unchanged = all(array_state_equal(state0[k], rp.snapshot_arrays(actor, kind)[k]) for k in state0)
    return results, unchanged


def _records(results, reaches) -> dict:
    arr = {k: np.stack([getattr(r, k) for r in results]) for k in RECORD_ARRAYS}
    arr["t"] = np.stack([r.t for r in results])
    arr["q_des"] = np.stack([r.q_des for r in results])
    arr["qdot_des"] = np.stack([r.qdot_des for r in reaches])
    arr["qddot_des"] = np.stack([r.qddot_des for r in reaches])
    arr["targets"] = np.stack([r.target_xy for r in results])
    arr["finite"] = np.array([r.finite for r in results])
    return arr


def _ctx(kind, d):
    return d if kind == "bias" else None


def diagnose_condition(arm, reaches, model, cond: dict, reference_curves: list[dict] | None,
                       out_dir: Path) -> dict:
    """Stage-D acquisition for one b = 13 condition (plan v0.3 M1):

    1. rerun A1 -> B(b_cycles) -> catch1 with washout 0 (fresh readout) -> J_post = final readout;
    2. compare A/B/catch curves with the stage-C run of the same condition (determinism);
    3. snapshot J_pre / J_post (bias: base_J + biases; RLS: J + P) and check restore;
    4. nominal-input probe per direction;
    5. frozen pre/post cycles at b = 0 (A and catch reproduction checks) and at b = 13
       (diagnostic), saving q/qdot/tau/ee, the float32 inputs and recomputed features.
    Non-finite outcomes are recorded with reasons; they never raise.
    """
    from .protocol import _cycle_metrics, run_abc_protocol

    method, seed, b, pd, eta = cond["method"], cond["seed"], cond["b"], cond["pd"], cond["eta"]
    family, kind, _ = METHODS[method]
    if kind not in ("lms", "bias", "rls"):
        raise ValueError(f"diagnostics cover adaptive readouts only, not {kind!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    core0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    J0 = model.export_readout_weights().astype(float)
    readout, J0_ref = ro.build_readout(kind, J0, eta)
    pre = copy.deepcopy(readout)
    checks = {"pre_equals_fresh_readout": readout_state_equal(pre, ro.build_readout(kind, J0, eta)[0], kind)}
    t0 = time.perf_counter()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        res = run_abc_protocol(arm, model, readout, reaches, KP * pd, KD * pd, J0_ref, dt=DT,
                               field_b=b, b_cycles=cond["b_cycles"], washout_cycles=0,
                               washout_adapt=True, error_signal="fel", device="cpu")
    wall_rerun = time.perf_counter() - t0
    post = copy.deepcopy(readout)
    checks["final_J_equals_live_readout"] = array_state_equal(res["final_J"], readout.J)
    checks["pre_unchanged_by_protocol"] = readout_state_equal(pre, ro.build_readout(kind, J0, eta)[0], kind)
    checks["post_state_finite"] = all(bool(np.all(np.isfinite(v))) for k, v in
                                      rp.snapshot_arrays(post, kind).items() if v.dtype.kind == "f")
    checks["determinism_vs_stage_c"] = (compare_curves(res["curves"], reference_curves)
                                        if reference_curves is not None else None)
    (out_dir / "curves_A1_B_catch1.json").write_text(json.dumps(ro.json_safe(res["curves"]), indent=1))
    np.savez(out_dir / "saved_trajectories.npz",
             **{f"{ph}__{k}": v for ph, d in res["saved_trajectories"].items() for k, v in d.items()})
    rp.save_snapshot(out_dir / "readout_pre.npz", pre, kind)
    rp.save_snapshot(out_dir / "readout_post.npz", post, kind)
    pre_back, post_back = rp.load_snapshot(out_dir / "readout_pre.npz"), rp.load_snapshot(out_dir / "readout_post.npz")
    checks["snapshot_restore_equal"] = readout_state_equal(pre_back, pre, kind) and readout_state_equal(post_back, post, kind)

    # A reproduced divergent adaptation is a scientific failure, not a valid probe.
    if not checks["post_state_finite"] or not all(c["all_finite"] for c in res["curves"]):
        import torch
        checks["core_unchanged"] = all(torch.equal(core0[k], v) for k, v in model.state_dict().items())
        Bc = [c for c in res["curves"] if c["phase"] == "B"]
        summary = {"method": method, "seed": seed, "b": b, "pd": pd, "eta": eta,
                   "readout_kind": kind, "family": family,
                   "adaptive_B_first_mm": Bc[0]["mean_max_perp_dev"] * 1e3,
                   "adaptive_B_last_mm": Bc[-1]["mean_max_perp_dev"] * 1e3,
                   **{k: math.nan for k in ("frozen_pre_b13_mm", "frozen_post_b13_mm",
                       "frozen_post_over_pre", "frozen_post_over_adaptive_last", "probe_r_median", "probe_g_median")},
                   "all_phases_finite": all(c["all_finite"] for c in res["curves"]),
                   "frozen_all_finite": False, "wall_rerun_s": wall_rerun,
                   "diagnostic_status": "unavailable", "missing_reason": "nonfinite_adaptation_or_readout"}
        return {"cond": cond, "checks": checks, "summary": summary, "probe_rows": [],
                "reach_rows": [], "cycle_rows": [], "actor_rows": [],
                "bytes": sum(p.stat().st_size for p in out_dir.iterdir()),
                "files": sorted(p.name for p in out_dir.iterdir())}

    # nominal probe
    probe_rows, waves, Zs = [], {k: [] for k in ("tau_pre", "tau_post", "n", "s")}, []
    for d, reach in enumerate(reaches):
        Z = rp.nominal_features(model, reach)
        tau_pre = rp.predict_series(pre, Z, _ctx(kind, d))
        tau_post = rp.predict_series(post, Z, _ctx(kind, d))
        it = rp.ideal_torques(arm, reach, b)
        m = rp.probe_metrics(tau_pre, tau_post, it["n"], it["s"])
        probe_rows.append({"method": method, "seed": seed, "b": b, "pd": pd, "eta": eta,
                           "direction_index": d, "input": "nominal", **m})
        for k, v in (("tau_pre", tau_pre), ("tau_post", tau_post), ("n", it["n"]), ("s", it["s"])):
            waves[k].append(v)
        Zs.append(Z)
    np.savez(out_dir / "probe_waveforms.npz", Z=np.stack(Zs), **{k: np.stack(v) for k, v in waves.items()},
             delta=np.stack(waves["tau_post"]) - np.stack(waves["tau_pre"]),
             time=np.asarray(reaches[0].times, float))
    checks["weights_unchanged_by_probe"] = (readout_state_equal(pre_back, pre, kind)
                                            and readout_state_equal(post_back, post, kind))

    # frozen cycles
    reach_rows, cycle_rows, actor_rows = [], [], []
    cycles = {}
    for actor, predictor in (("pre", pre), ("post", post)):
        for field in (0.0, b):
            cid = f"{actor}_b{int(field)}"
            results, unchanged = _frozen_cycle(arm, reaches, model, predictor, J0_ref, field, pd, kind)
            checks[f"{cid}:readout_unchanged"] = unchanged
            cm = _cycle_metrics(f"frozen_{cid}", 0, results)
            cycles[cid] = cm
            arr = _records(results, reaches)
            if field != 0.0:
                X = np.stack([ir.build_inputs(reach, arr["q"][d], arr["qdot"][d]) for d, reach in enumerate(reaches)])
                Z = np.stack([ir.features_from_inputs(model, X[d]) for d in range(len(reaches))])
                tau_pred = np.stack([rp.predict_series(predictor, Z[d], _ctx(kind, d)) for d in range(len(reaches))])
                fin = arr["finite"]
                diff = np.abs(tau_pred[fin] - arr["tau_ff"][fin]) if fin.any() else np.zeros(1)
                checks[f"{cid}:tau_ff_reproduced_from_inputs"] = {
                    "max_abs_diff_finite_reaches": float(diff.max()) if fin.any() and diff.size else math.nan,
                    "bit_identical": bool(np.array_equal(tau_pred[fin], arr["tau_ff"][fin])) if fin.any() else None,
                    "status": "checked" if fin.any() else "unavailable",
                    "within_1e-12": bool(diff.max() <= 1e-12) if fin.any() and diff.size else None,
                    "n_finite_reaches": int(fin.sum())}
                arr["X"], arr["Z"] = X, Z
                tau_pre_a = np.stack([rp.predict_series(pre, Z[d], _ctx(kind, d)) for d in range(len(reaches))])
                tau_post_a = np.stack([rp.predict_series(post, Z[d], _ctx(kind, d)) for d in range(len(reaches))])
                arr["tau_pre_on_actor"], arr["tau_post_on_actor"] = tau_pre_a, tau_post_a
                for d in range(len(reaches)):
                    s_act = ir.field_compensation(arm, arr["q"][d], arr["qdot"][d], b)
                    am = fc.actor_metrics(tau_post_a[d] - tau_pre_a[d], s_act)
                    actor_rows.append({"method": method, "seed": seed, "actor_trajectory": cid,
                                       "direction_index": d, "finite": bool(fin[d]), **am})
            np.savez(out_dir / f"frozen_{cid}.npz", **arr)
            for d, r in enumerate(results):
                reach_rows.append({"method": method, "seed": seed, "condition": cid, "actor_readout": actor,
                                   "field_b": field, "direction_index": d, **fc.reach_summary(r)})
            cycle_rows.append({"method": method, "seed": seed, "source": f"frozen {cid}",
                               "mean_max_perp_mm": cm["mean_max_perp_dev"] * 1e3,
                               "mean_signed_perp_mm": cm["mean_signed_perp_dev"] * 1e3,
                               "mean_endpoint_mm": cm["mean_endpoint_error"] * 1e3,
                               "taufb_rms": cm["taufb_rms"], "all_finite": cm["all_finite"],
                               "signed_perp_dev_per_dir_mm": [v * 1e3 for v in cm["signed_perp_dev_per_dir"]]})
    A = [c for c in res["curves"] if c["phase"] == "A"][0]
    Bc = [c for c in res["curves"] if c["phase"] == "B"]
    C = [c for c in res["curves"] if c["phase"] == "C_catch"][0]
    checks["A_cycle_reproduced_by_frozen_pre_b0"] = compare_cycle(cycles["pre_b0"], A)
    checks["catch_cycle_reproduced_by_frozen_post_b0"] = compare_cycle(cycles["post_b0"], C)
    for name, cm in (("adaptive A", A), ("adaptive B first", Bc[0]), ("adaptive B last", Bc[-1]),
                     ("adaptive catch", C)):
        cycle_rows.append({"method": method, "seed": seed, "source": name,
                           "mean_max_perp_mm": cm["mean_max_perp_dev"] * 1e3,
                           "mean_signed_perp_mm": cm["mean_signed_perp_dev"] * 1e3,
                           "mean_endpoint_mm": cm["mean_endpoint_error"] * 1e3,
                           "taufb_rms": cm["taufb_rms"], "all_finite": cm["all_finite"],
                           "signed_perp_dev_per_dir_mm": [v * 1e3 for v in cm["signed_perp_dev_per_dir"]]})
    import torch

    checks["core_unchanged"] = all(torch.equal(core0[k], v) for k, v in model.state_dict().items())
    checks["pre_post_unchanged_after_frozen_runs"] = (readout_state_equal(pre_back, pre, kind)
                                                     and readout_state_equal(post_back, post, kind))
    summary = {
        "method": method, "seed": seed, "b": b, "pd": pd, "eta": eta, "readout_kind": kind, "family": family,
        "adaptive_B_first_mm": Bc[0]["mean_max_perp_dev"] * 1e3, "adaptive_B_last_mm": Bc[-1]["mean_max_perp_dev"] * 1e3,
        "frozen_pre_b13_mm": cycles[f"pre_b{int(b)}"]["mean_max_perp_dev"] * 1e3,
        "frozen_post_b13_mm": cycles[f"post_b{int(b)}"]["mean_max_perp_dev"] * 1e3,
        "frozen_post_over_pre": (cycles[f"post_b{int(b)}"]["mean_max_perp_dev"]
                                 / cycles[f"pre_b{int(b)}"]["mean_max_perp_dev"]),
        "frozen_post_over_adaptive_last": (cycles[f"post_b{int(b)}"]["mean_max_perp_dev"]
                                           / Bc[-1]["mean_max_perp_dev"]),
        "probe_r_median": float(np.nanmedian([r["r"] for r in probe_rows])),
        "probe_g_median": float(np.nanmedian([r["g"] for r in probe_rows])),
        "all_phases_finite": all(c["all_finite"] for c in res["curves"]),
        "frozen_all_finite": all(cm["all_finite"] for cm in cycles.values()),
        "diagnostic_status": "available" if all(cm["all_finite"] for cm in cycles.values()) else "partial",
        "missing_reason": "" if all(cm["all_finite"] for cm in cycles.values()) else "nonfinite_frozen_reaches",
        "wall_rerun_s": wall_rerun,
    }
    bytes_total = sum(p.stat().st_size for p in out_dir.iterdir())
    return {"cond": cond, "checks": checks, "summary": summary, "probe_rows": probe_rows,
            "reach_rows": reach_rows, "cycle_rows": cycle_rows, "actor_rows": actor_rows,
            "bytes": bytes_total, "files": sorted(p.name for p in out_dir.iterdir())}


def diagnostics_passed(checks: dict) -> tuple[bool, list[str]]:
    """Technical pass/fail of one condition's checks. A reproduced non-finite failure is a
    scientific outcome (reported), not a technical failure; a mismatch is."""
    bad = []
    for k, v in checks.items():
        if v is None:
            continue
        if k == "determinism_vs_stage_c":
            if v["status"] == "mismatch":
                bad.append(k)
        elif isinstance(v, dict) and "within_1e-12" in v:
            if v["within_1e-12"] is False:
                bad.append(k)
        elif isinstance(v, dict) and "status" in v:
            if v["status"] == "mismatch":
                bad.append(k)
        elif isinstance(v, dict) and "equal" in v:
            if not v["equal"]:
                bad.append(k)
        elif isinstance(v, dict) and "within_1e-12" in v:
            if v["within_1e-12"] is False:
                bad.append(k)
        elif k == "post_state_finite":
            continue  # scientific outcome, reported in the summary
        elif v is not True:
            bad.append(k)
    return not bad, bad


# ----------------------------------------------------------------------------------------
# manifest verification
# ----------------------------------------------------------------------------------------
def verify_sources(source_sha256: dict, code_root: Path) -> list[str]:
    return [n for n, h in source_sha256.items() if sha256_file(Path(code_root) / n) != h]


def verify_a3(A3: dict, code_root: Path, *, require_evaluation_seeds: bool) -> dict:
    """Hashes of sources and every indexed checkpoint, seed roles, grids. Raises on mismatch."""
    problems = []
    changed = verify_sources(A3["spec"]["source_sha256"], code_root)
    if changed:
        problems.append(f"sources changed since the A3 spec: {changed}")
    seeds = list(SELECTION_SEEDS) + (list(EVALUATION_SEEDS) if require_evaluation_seeds else [])
    n_ckpt = 0
    for fam in FAMILIES:
        rows = A3["checkpoint_index"].get(fam, {})
        for s in seeds:
            row = rows.get(str(s))
            if row is None:
                problems.append(f"missing index row {fam} seed {s}")
                continue
            p = Path(row["checkpoint"])
            if not p.exists():
                problems.append(f"missing checkpoint {p}")
            elif sha256_file(p) != row["sha256"]:
                problems.append(f"checkpoint sha256 mismatch {p}")
            if fam in CFC_FAMILIES and not row.get("reload_ok"):
                problems.append(f"{fam} seed {s}: reload check not passed in the index")
            n_ckpt += 1
    for key in FAMILIES + ("bias",):
        if len(A3["eta_grid"][key]["grid"]) != len(ETA_MULTIPLIERS):
            problems.append(f"{key}: eta grid is not {len(ETA_MULTIPLIERS)} points")
    if A3["spec"]["selection_seeds"] != list(SELECTION_SEEDS) or A3["spec"]["evaluation_seeds"] != list(EVALUATION_SEEDS):
        problems.append("seed roles differ from the design constants")
    if require_evaluation_seeds and A3.get("smoke"):
        problems.append("manifest A3 is a smoke manifest (selection seeds only); production stages refuse it")
    if problems:
        raise ValueError("manifest A3 verification failed:\n  " + "\n  ".join(problems))
    return {"checkpoints_verified": n_ckpt, "sources_verified": len(A3["spec"]["source_sha256"])}


def verify_b3(A3: dict, a3_sha: str, B3: dict) -> None:
    problems = []
    if B3.get("manifest_A3_sha256") != a3_sha:
        problems.append("manifest B3 does not record the final manifest A3 sha256")
    if B3.get("methods") != list(METHODS):
        problems.append("B3 method list differs from METHODS")
    if B3["selection"].get("eta_grids_from_manifest_A3") != A3["eta_grid"]:
        problems.append("eta grids in B3 differ from A3")
    if B3["selection"].get("multipliers") != list(ETA_MULTIPLIERS):
        problems.append("eta multipliers differ")
    if B3["evaluation"].get("fields") != list(EVAL_FIELDS) or B3["evaluation"].get("seeds") != list(EVALUATION_SEEDS):
        problems.append("evaluation design differs")
    if B3["sensitivity"].get("PD_multipliers") != list(PD_MULTIPLIERS):
        problems.append("sensitivity design differs")
    if problems:
        raise ValueError("manifest B3 verification failed:\n  " + "\n  ".join(problems))


__all__ = [n for n in dir() if not n.startswith("_")]
