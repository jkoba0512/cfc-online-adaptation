"""400-epoch online rerun driver (plan v0.3, 2026-11-07): manifests A3/B3, smoke, selection,
evaluation, sensitivity, stage-D diagnostics and tables.

Stages (from code/; UV_PROJECT_ENVIRONMENT=$HOME/.venvs/cfc-online-adaptation, uv run --no-sync):

  spec             --out DIR            write manifest_A3_spec.json (design, data and source hashes,
                                        pilot checkpoint references); nothing is trained
  prepare-offline  --out DIR [--smoke]  train the 30 evaluation series (skipped with --smoke), index
                   [--resume]           every final-epoch checkpoint, select and fit ESN/RBF with the
                                        explicit selection seeds, write manifest_A3.json
  design-b3        --out DIR            write manifest_B3.json bound to the final A3 digest
  smoke            --out DIR            shortened-protocol technical checks on the selection seeds
                                        (requires a --smoke A3); no comparison result
  select | evaluate | sensitivity | diagnostics | report   --out DIR [--workers N]
                                        production stages; refuse a smoke A3

Every online stage re-verifies the source hashes, every indexed checkpoint hash and the A3/B3
binding before running. Runs are appended to DIR/runs.jsonl (never rewritten); cached runs are
reused by key. Old outputs, manifests A/B/A2 and frozen sources are read only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_revision_offline as offline_frozen  # noqa: E402  (frozen script: grids only)
from cfc_online_adaptation import budget_pilot as bp  # noqa: E402
from cfc_online_adaptation import revision_online as ro  # noqa: E402
from cfc_online_adaptation import revision_rerun as rr  # noqa: E402

CODE_ROOT = SCRIPTS.parent
REPO = CODE_ROOT.parent
PILOT_DIR = REPO / "outputs/budget_pilot_20261107_182839"
PILOT_A2_SHA = "0f1db04cad9b31fab49b860d8d4092a656e72cea6c003db8a749dd5d33d72d57"
OLD_A = REPO / "outputs/revision_offline_20261102_174429/manifest_A.json"
OLD_A_SHA = "1d72f763e4c3d61f6b5ec3e4e1b5ee318446b5b10d76ce796f5f29ca4c30c56e"
OLD_SMOKE_RUNS = REPO / "outputs/revision_online_smoke_20261102_w1/runs.jsonl"
OLD_MAIN_RUNS = REPO / "outputs/revision_online_20261102_202742/runs.jsonl"
OLD_TO_NEW_METHOD = {"B0-CfC": "B0-dense-CfC", "B0-NCP": "B0-NCP-CfC", "B0-ESN": "B0-ESN",
                     "B0-RBF": "B0-RBF", "B2": "PD-only", "CfC-LMS": "dense-CfC-LMS",
                     "CfC-RLS-reg": "dense-CfC-RLS", "NCP-LMS": "NCP-CfC-LMS", "ESN-LMS": "ESN-LMS",
                     "RBF-LMS": "RBF-LMS", "CfC-direction-specific-bias": "dense-CfC-bias"}
OLD_FAMILY = {"cfc": "dense", "ncp": "ncp", "esn": "esn", "rbf": "rbf"}
SOURCE_FILES = list(offline_frozen.SOURCE_FILES) + [
    "src/cfc_online_adaptation/budget_pilot.py", "src/cfc_online_adaptation/revision_rerun.py",
    "scripts/run_revision_rerun.py", "scripts/run_budget_pilot.py",
    "src/cfc_online_adaptation/revision_online.py", "src/cfc_online_adaptation/readout_probe.py",
    "src/cfc_online_adaptation/input_replay.py", "src/cfc_online_adaptation/frozen_closed_loop.py",
    "src/cfc_online_adaptation/metrics.py", "src/cfc_online_adaptation/baselines.py",
    "src/cfc_online_adaptation/analysis.py", "tests/test_revision_rerun.py", "uv.lock"]
ESN_GRID, RBF_GRID, LAM_GRID = offline_frozen.ESN_GRID, offline_frozen.RBF_GRID, offline_frozen.LAM_GRID
assert len(ESN_GRID) == 27 and len(RBF_GRID) == 5 and len(LAM_GRID) == 5
CAPACITY_MARGIN = 1.5
SMOKE_B, SMOKE_W = 2, 1


# ---------------------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------------------
def git(*a) -> str:
    try:
        return subprocess.check_output(["git", *a], text=True, cwd=CODE_ROOT).strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def environment() -> dict:
    import ncps
    from importlib import metadata as md
    return {"python": sys.version.split()[0], "platform": platform.platform(), "hostname": platform.node(),
            "cpu_count": os.cpu_count(), "torch": torch.__version__, "numpy": np.__version__,
            "ncps_module_version": getattr(ncps, "__version__", None), "ncps_dist_version": md.version("ncps"),
            "torch_num_threads": torch.get_num_threads(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "git_commit": git("rev-parse", "HEAD"), "git_status_code": git("status", "--porcelain", "--", "."),
            "venv": os.environ.get("VIRTUAL_ENV") or sys.prefix}


def jdump(path: Path, obj) -> None:
    rr.atomic_write_text(path, json.dumps(ro.json_safe(obj), indent=2, allow_nan=False, default=float) + "\n")


def jload(path: Path):
    return json.loads(Path(path).read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and not math.isfinite(v)) or v is None else v)
                        for k, v in r.items()})


def refuse_if_exists(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")


def maxrss_mb() -> dict:
    return {"self_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "children_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024}


def series_id(config: str, seed: int) -> str:
    return f"{config}_s{seed}"


# ---------------------------------------------------------------------------------------
# spec (design fixed before any offline production run)
# ---------------------------------------------------------------------------------------
def pilot_references() -> dict:
    a2_path = PILOT_DIR / "manifest_A2.json"
    a2_sha = rr.sha256_file(a2_path)
    if a2_sha != PILOT_A2_SHA:
        raise SystemExit(f"pilot manifest A2 sha256 {a2_sha} != expected {PILOT_A2_SHA}")
    A2 = jload(a2_path)
    ck = {}
    for fam, config in rr.CFC_FAMILIES.items():
        for s in rr.SELECTION_SEEDS:
            p = PILOT_DIR / "series" / series_id(config, s) / f"epoch{rr.EPOCHS:04d}.pt"
            if not p.exists():
                raise SystemExit(f"pilot checkpoint missing: {p}")
            ck[f"{fam}:{s}"] = {"checkpoint": str(p), "sha256": rr.sha256_file(p)}
    pv = jload(PILOT_DIR / "post_run_verification.json")
    return {"dir": str(PILOT_DIR), "manifest_A2": str(a2_path), "manifest_A2_sha256": a2_sha,
            "A2_data_x_sha256": A2["data"]["x_sha256"], "A2_data_tau_sha256": A2["data"]["tau_sha256"],
            "A2_wiring_seeds": {str(sr["seed"]): sr["wiring_seed"] for sr in A2["series"] if sr["config"] == "ncp_wired"},
            "post_run_verification_all_ok": pv["all_ok"],
            "series_disk_bytes": {k: v["disk_bytes"] for k, v in pv["series"].items()},
            "checkpoints": ck}


def stage_spec(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest_A3_spec.json"
    refuse_if_exists(path)
    data = rr.data_hashes(rr.ALL_SEEDS)
    pilot = pilot_references()
    if (pilot["A2_data_x_sha256"], pilot["A2_data_tau_sha256"]) != (data["x_sha256"], data["tau_sha256"]):
        raise SystemExit("regenerated data differ from the pilot's A2 data hashes")
    if any(pilot["A2_wiring_seeds"][str(s)] != rr.WIRING_SEEDS[s] for s in rr.SELECTION_SEEDS):
        raise SystemExit("pilot NCP wiring seeds differ from WIRING_SEEDS")
    spec = {
        "schema": "rerun400-A3-spec-v1",
        "plan": "cfc-online-adaptation_400epoch-online-rerun-plan_2026-11-07 (v0.3)",
        "purpose": "all CfC configurations at their final 400th epoch; implementation-configuration "
                   "comparison, not a single-factor test; manuscript numbers unchanged until all results exist",
        "selection_seeds": list(rr.SELECTION_SEEDS), "evaluation_seeds": list(rr.EVALUATION_SEEDS),
        "wiring_seeds": {str(s): rr.WIRING_SEEDS[s] for s in rr.ALL_SEEDS},
        "wiring_note": f"NCP wiring seed = 22200 + model seed; the 2026-11-02 experiment used "
                       f"{rr.OLD_NCP_WIRING_SEED} for every seed, so NCP seed-to-seed variance now "
                       f"includes wiring variance",
        "configs": dict(rr.CFC_FAMILIES), "ridge_families": list(rr.RIDGE_FAMILIES),
        "training": {"epochs": rr.EPOCHS, "ckpt_every": rr.CKPT_EVERY, "lr": bp.LR, "optimizer": "Adam",
                     "batch": "full", "loss": "MSE", "selection": "none: final epoch 400 of every series",
                     "loop": "budget_pilot.train_series (identical to the pilot)",
                     "selection_seed_checkpoints": "pilot epoch0400.pt reused (no retraining)",
                     "evaluation_seed_checkpoints": "new 400-epoch series"},
        "ridge": {"esn_grid": [list(g) for g in ESN_GRID], "rbf_grid": list(RBF_GRID), "lam_grid": list(LAM_GRID),
                  "rule": "family setting = min mean validation NMSE over the selection seeds after the "
                          "per-seed ridge choice; per-seed lambda by validation; an edge selection is "
                          "recorded, never extended; online results never adjust these settings",
                  "selection_seeds": list(rr.SELECTION_SEEDS)},
        "eta": {"multipliers": list(rr.ETA_MULTIPLIERS),
                "center": "1e-3 * g_dense / g_family, g = mean selection-seed training feature energy of "
                          "the final-epoch models; bias g = 1",
                "edge_rule": "an edge selection is reported with code grid_edge; the grid is not extended"},
        "online": {"selection_field": rr.SELECTION_FIELD, "evaluation_fields": list(rr.EVAL_FIELDS),
                   "pd_multipliers": list(rr.PD_MULTIPLIERS),
                   "protocol": f"A1 -> B{rr.B_CYCLES} -> catch1 (adapt off) -> washout{rr.WASHOUT_CYCLES} (adapt on)",
                   "task": {"center_xy": list(rr.CENTER_XY), "radius": rr.RADIUS, "T": rr.T_MOVE, "dt": rr.DT,
                            "Kp": rr.KP.diagonal().tolist(), "Kd": rr.KD.diagonal().tolist(),
                            "direction_order": "fixed", "error_signal": "fel"},
                   "methods": rr.METHODS, "selected_methods": list(rr.SELECTED_METHODS),
                   "adaptive_methods": list(rr.ADAPTIVE_METHODS), "B0_methods": list(rr.B0_METHODS),
                   "RLS": {"beta": ro.RLS_BETA, "p0": ro.RLS_P0, "selection": "none (fixed)"},
                   "expected_job_counts": rr.EXPECTED_JOB_COUNTS},
        "data": data, "pilot": pilot,
        "source_sha256": {n: rr.sha256_file(CODE_ROOT / n) for n in SOURCE_FILES},
        "environment": environment(), "created_at": datetime.now().astimezone().isoformat(),
    }
    jdump(path, spec)
    print(f"wrote {path}\n  sha256 {rr.sha256_file(path)}")


def load_spec(out: Path) -> dict:
    path = out / "manifest_A3_spec.json"
    spec = jload(path)
    changed = rr.verify_sources(spec["source_sha256"], CODE_ROOT)
    if changed:
        raise SystemExit(f"sources changed since the A3 spec: {changed}")
    spec["_sha256"] = rr.sha256_file(path)
    return spec


# ---------------------------------------------------------------------------------------
# prepare-offline
# ---------------------------------------------------------------------------------------
def capacity_check(spec: dict, out: Path, n_series_per_config: int) -> dict:
    sizes = spec["pilot"]["series_disk_bytes"]
    per_config = {c: max(v for k, v in sizes.items() if k.startswith(c)) for c in rr.CFC_FAMILIES.values()}
    est = sum(per_config.values()) * n_series_per_config
    free = shutil.disk_usage(out).free
    rep = {"per_series_bytes_from_pilot": per_config, "n_series": n_series_per_config * len(per_config),
           "estimate_bytes": est, "margin": CAPACITY_MARGIN, "required_bytes": int(est * CAPACITY_MARGIN),
           "free_bytes": free, "sufficient": free >= est * CAPACITY_MARGIN}
    if not rep["sufficient"]:
        raise SystemExit(f"insufficient free space: {rep}")
    return rep


def train_evaluation_series(spec: dict, out: Path, resume: bool) -> dict:
    record_path = out / "offline_run_record.json"
    record = jload(record_path) if record_path.exists() else {"series": {}}
    record["capacity"] = capacity_check(spec, out, len(rr.EVALUATION_SEEDS))
    for fam, config in rr.CFC_FAMILIES.items():
        for s in rr.EVALUATION_SEEDS:
            sid = series_id(config, s)
            d = out / "series" / sid
            if (d / f"epoch{rr.EPOCHS:04d}.pt").exists():
                print(f"[{sid}] complete; skipping", flush=True)
                continue
            rr.verify_data_hashes(spec["data"], s)
            tr, va, audit = rr.split_data(s)
            wiring = None if fam == "dense" else rr.WIRING_SEEDS[s]
            model = bp.build_model(config, s, wiring)
            cfg = {"config": config, "seed": s, "wiring_seed": wiring, "epochs": rr.EPOCHS,
                   "ckpt_every": rr.CKPT_EVERY, "lr": bp.LR,
                   "optimizer": "Adam(lr=1e-3), full batch, one update per epoch",
                   "loss": "MSELoss on teacher torque", "selection": "none: final epoch 400"}
            prov = {"manifest_A3_spec_sha256": spec["_sha256"], "series_id": sid,
                    "data_x_sha256": spec["data"]["x_sha256"], "schema": spec["schema"]}
            t0 = datetime.now().astimezone()
            res = bp.train_series(model, tr, va, audit, epochs=rr.EPOCHS, seed=s, out_dir=d,
                                  ckpt_every=rr.CKPT_EVERY, config=cfg, resume=resume, provenance=prov)
            res["started_at"], res["finished_at"] = t0.isoformat(), datetime.now().astimezone().isoformat()
            res["checkpoint_sha256"] = {p.name: rr.sha256_file(p) for p in sorted(d.glob("epoch*.pt"))}
            res["tmp_files_left"] = [p.name for p in d.glob("*.tmp")]
            record["series"][sid] = res
            jdump(record_path, record)
            print(f"[{sid}] done: {res['timing']['epoch_time_s_mean_excl_warmup']:.3f} s/epoch", flush=True)
    record["environment"] = environment()
    jdump(record_path, record)
    return record


def stage_prepare_offline(out: Path, smoke: bool, resume: bool) -> None:
    spec = load_spec(out)
    a3_path = out / "manifest_A3.json"
    refuse_if_exists(a3_path)
    seeds = list(rr.SELECTION_SEEDS) + ([] if smoke else list(rr.EVALUATION_SEEDS))
    for s in seeds:
        rr.verify_data_hashes(spec["data"], s)
    record = None if smoke else train_evaluation_series(spec, out, resume)

    index = {fam: {} for fam in rr.FAMILIES}
    for fam, config in rr.CFC_FAMILIES.items():
        for s in seeds:
            if s in rr.SELECTION_SEEDS:
                ref = spec["pilot"]["checkpoints"][f"{fam}:{s}"]
                p = Path(ref["checkpoint"])
                if rr.sha256_file(p) != ref["sha256"]:
                    raise SystemExit(f"pilot checkpoint changed since spec: {p}")
                row = rr.cfc_row_from_checkpoint(fam, s, p, origin="pilot budget_pilot_20261107_182839 epoch0400",
                                                 series_dir=p.parent)
                row["pilot_manifest_A2_sha256"] = spec["pilot"]["manifest_A2_sha256"]
            else:
                p = out / "series" / series_id(config, s) / f"epoch{rr.EPOCHS:04d}.pt"
                row = rr.cfc_row_from_checkpoint(fam, s, p, origin="new series (this manifest)", series_dir=p.parent)
                if row["checkpoint_provenance"].get("manifest_A3_spec_sha256") != spec["_sha256"]:
                    raise SystemExit(f"{p}: checkpoint provenance does not reference this spec")
            if not row["reload_ok"]:
                raise SystemExit(f"{fam} seed {s}: reload check failed: {row['reload_check']}")
            index[fam][str(s)] = row
            print(f"  indexed {fam:5s} seed {s}: val {row['val_nmse']:.5g} audit {row['audit_nmse']:.5g} "
                  f"g {row['feature_energy_train']:.3f} reload_ok={row['reload_ok']}", flush=True)

    split_by_seed = {s: rr.split_data(s)[:2] for s in seeds}
    audit = rr.split_data(seeds[0])[2]
    (out / "ridge").mkdir(exist_ok=True)
    offline_selection = {}
    from cfc_online_adaptation.bases import fit_and_validate
    for fam in rr.RIDGE_FAMILIES:
        settings = ESN_GRID if fam == "esn" else RBF_GRID
        best, candidates = rr.choose_family_for_seeds(fam, split_by_seed, rr.SELECTION_SEEDS,
                                                      settings=settings, lam_grid=LAM_GRID)
        offline_selection[fam] = {"best": best, "candidates": candidates, "n_candidate_fits": len(candidates) * len(rr.SELECTION_SEEDS),
                                  "grid_edge_rule": "recorded only; grid not extended"}
        setting = tuple(best["setting"]) if fam == "esn" else best["setting"]
        for s in seeds:
            tr, va = split_by_seed[s]
            model = rr.build_ridge_model(fam, setting, tr.x, s)
            fit = fit_and_validate(model, tr.x, tr.tau, va.x, va.tau, LAM_GRID)
            row = rr.ridge_row(model, fam, tr, va, audit, out / "ridge" / f"{fam}_seed{s}.pt", fit=fit, setting=setting)
            index[fam][str(s)] = row
        print(f"  {fam}: setting {best['setting']} (grid index {best['grid_index']}/{best['n_candidates'] - 1}, "
              f"edge={best['at_grid_edge']}) fitted for {len(seeds)} seeds", flush=True)

    eta_grid = rr.eta_grid_from_index(index)
    A3 = {"schema": "rerun400-A3-v1", "smoke": smoke,
          "status": ("SMOKE: selection seeds only; not for production" if smoke
                     else "production: all 13 seeds indexed"),
          "spec_sha256": spec["_sha256"], "spec": {k: v for k, v in spec.items() if k != "_sha256"},
          "seeds_indexed": seeds, "checkpoint_index": index, "offline_selection": offline_selection,
          "eta_grid": eta_grid,
          "offline_training_record": None if record is None else {
              k: {"timing": v["timing"], "checkpoint_sha256": v["checkpoint_sha256"], "recovery": v["recovery"],
                  "started_at": v["started_at"], "finished_at": v["finished_at"], "tmp_files_left": v["tmp_files_left"]}
              for k, v in record["series"].items()} | {"capacity": record["capacity"]},
          "n_models": sum(len(v) for v in index.values()),
          "environment": environment(), "created_at": datetime.now().astimezone().isoformat()}
    jdump(a3_path, A3)
    print(f"wrote {a3_path}\n  sha256 {rr.sha256_file(a3_path)}; models {A3['n_models']}; smoke={smoke}")
    for k, v in eta_grid.items():
        print(f"  eta grid {k:5s}: center {v['center']:.4g}  [{v['grid'][0]:.3g} .. {v['grid'][-1]:.3g}]")


def load_a3(out: Path, *, require_evaluation_seeds: bool) -> tuple[dict, str]:
    path = out / "manifest_A3.json"
    A3 = jload(path)
    rr.verify_a3(A3, CODE_ROOT, require_evaluation_seeds=require_evaluation_seeds)
    return A3, rr.sha256_file(path)


# ---------------------------------------------------------------------------------------
# design-b3
# ---------------------------------------------------------------------------------------
def stage_design_b3(out: Path, smoke_report: Path | None = None) -> None:
    path = out / "manifest_B3.json"
    refuse_if_exists(path)
    A3, a3_sha = load_a3(out, require_evaluation_seeds=False)
    counts = rr.job_counts(A3)
    counts.update({"evaluate_adaptive": len(rr.ADAPTIVE_METHODS) * len(rr.EVALUATION_SEEDS) * len(rr.EVAL_FIELDS),
                   "evaluate_B0": len(rr.B0_METHODS) * len(rr.EVALUATION_SEEDS) * len(rr.EVAL_FIELDS),
                   "evaluate_PD_only": len(rr.EVAL_FIELDS),
                   "diagnostics_rerun": len(rr.ADAPTIVE_METHODS) * len(rr.EVALUATION_SEEDS)})
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    counts_match = all(counts.get(k) == v for k, v in rr.EXPECTED_JOB_COUNTS.items()
                       if k != "frozen_cycles_not_counted")
    smoke_ref = None
    sr_path = smoke_report or (out / "smoke_report.json")
    if sr_path.exists():
        sr = jload(sr_path)
        smoke_ref = {"path": str(sr_path.resolve()), "sha256": rr.sha256_file(sr_path), "passed": sr.get("passed"),
                     "pd_only_two_family_identical": sr["checks"].get("pd_only_two_family_identical"),
                     "diagnostics_determinism_all_bit_identical": sr["checks"].get("diagnostics_determinism_all_bit_identical")}
    B3 = {
        "schema": "rerun400-B3-v1", "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": git("rev-parse", "HEAD"), "manifest_A3": str((out / "manifest_A3.json").resolve()),
        "manifest_A3_sha256": a3_sha, "manifest_A3_is_smoke": bool(A3.get("smoke")),
        "selection_seeds": list(rr.SELECTION_SEEDS), "evaluation_seeds": list(rr.EVALUATION_SEEDS),
        "methods": list(rr.METHODS), "method_definitions": rr.METHODS,
        "shared_task": A3["spec"]["online"]["task"] | {
            "protocol": {"A_null_cycles": 1, "B_curl_cycles": rr.B_CYCLES, "C_catch_null_cycles": 1,
                         "C_washout_null_cycles": rr.WASHOUT_CYCLES, "catch_adapt": False, "washout_adapt": True},
            "readout": "kept across reaches; recurrent state reset per reach; PD torque always applied, "
                       "used as the learning signal only when updates are on"},
        "selection": {"validity_function": "cfc_online_adaptation.revision_manifest.validate_ab_curves",
                      "selection_function": "cfc_online_adaptation.revision_rerun.select_eta_for_seeds",
                      "field_b": rr.SELECTION_FIELD, "pd": 1.0, "seeds": list(rr.SELECTION_SEEDS),
                      "selected_methods": list(rr.SELECTED_METHODS), "multipliers": list(rr.ETA_MULTIPLIERS),
                      "eta_grids_from_manifest_A3": A3["eta_grid"],
                      "rule": "valid = every selection seed passes validate_ab_curves; among valid etas the "
                              "smallest mean B_last; ties to the smaller eta; edge selection recorded, grid not "
                              "extended; no rescue on evaluation seeds; after-effect, probe and frozen "
                              "results never enter selection",
                      "RLS": "fixed beta=1, P0=0.1I on the dense 400-epoch checkpoint; no selection runs"},
        "evaluation": {"fields": list(rr.EVAL_FIELDS), "field_main": rr.SELECTION_FIELD, "seeds": list(rr.EVALUATION_SEEDS),
                       "pd": 1.0, "same_selected_eta_across_fields": True,
                       "B0_reference": "same family, seed, field and PD (readout frozen at J0); bias and RLS "
                                       "reference the dense B0; PD-only is one deterministic run per field",
                       "primary": {"metric": "B_last_mm (direction mean of max |perpendicular deviation|, "
                                             "final B cycle, b = 13)",
                                   "pair": "evaluation seed s: NCP-CfC-LMS seed s vs dense-CfC-LMS seed s "
                                           "(same data split and init seed; NCP wiring seed 22200+s)",
                                   "statistic": "mean log(error_NCP/error_dense) over finite positive pairs, "
                                                f"paired percentile bootstrap 95% ({rr.BOOTSTRAP_N} resamples, "
                                                f"seed {rr.BOOTSTRAP_SEED}); all pairs and sign counts published; "
                                                "excluded pairs listed; not a superiority test"},
                       "summary": "median per method and field; no superiority significance test"},
        "sensitivity": {"methods": list(rr.SELECTED_METHODS), "eta_points": len(rr.ETA_MULTIPLIERS),
                        "PD_multipliers": list(rr.PD_MULTIPLIERS), "seeds": list(rr.SELECTION_SEEDS),
                        "field_b": rr.SELECTION_FIELD, "outcomes": ["B_last", "after_effect_at_threshold_0.25"],
                        "pd1_adaptive_runs": "reused from selection (identical key)",
                        "B0_runs": "5 families x 3 selection seeds x 3 PD = 45 (evaluation B0 are other seeds)"},
        "diagnostics": {"field_b": rr.DIAGNOSTIC_FIELD, "conditions": "7 adaptive methods x 10 evaluation seeds",
                        "acquisition": ["rerun A1 -> B30 -> catch1 (washout 0) with a fresh readout; J_post = final readout",
                                        "determinism: A/B/catch curves compared with the stage-C run (bit-identical "
                                        "finite values, or reproduced non-finite pattern; a mismatch stops the stage)",
                                        "snapshots J_pre/J_post: lms J; bias base_J + biases; RLS J + P",
                                        "nominal-input probe per direction (readout_probe.probe_metrics)",
                                        "frozen pre/post cycles at b = 0 (A and catch reproduction checks) and "
                                        "b = 13 (diagnostic): adapt off, PD on, context per direction"],
                        "saved_arrays": ["q", "qdot", "tau", "tau_ff", "tau_fb", "ee", "ee_des", "t", "q_des",
                                         "qdot_des", "qddot_des", "targets", "finite",
                                         "X (float32 inputs, b=13 cycles)", "Z (features recomputed, float64, b=13 cycles)",
                                         "tau_pre_on_actor", "tau_post_on_actor",
                                         "saved_trajectories A/B_first/B_last/C_catch", "probe waveforms"],
                        "not_saved": "actual inputs inside adaptive B cycles (replay excluded from this plan)",
                        "nonfinite_policy": "non-finite adaptive outcomes are recorded with reasons and excluded "
                                            "from valid probe/frozen statistics; denominators are published"},
        "fixed_parameters": {"RLS_beta": ro.RLS_BETA, "RLS_p0": ro.RLS_P0, "LMS_leakage": 0.0,
                             "direction_bias_contexts": 8},
        "metrics": {"functions": "cfc_online_adaptation.revision_online.run_metrics",
                    "after_effect_threshold_sweep": list(ro.THRESHOLDS), "main_threshold": ro.MAIN_THRESHOLD},
        "expected_job_counts": counts, "design_counts_match_plan": counts_match,
        "frozen_cycles_not_counted": rr.EXPECTED_JOB_COUNTS["frozen_cycles_not_counted"],
        "smoke_reference": smoke_ref,
        "analysis_scope": [
            "primary: b = 13, final-B absolute error, dense 400 vs NCP 400 (paired by evaluation seed)",
            "FC, ESN, RBF, bias and RLS are auxiliary; parameter counts, offline fit, learning rule and "
            "context information differ and are reported alongside",
            "equal epochs do not equalise preparation across families; no single-factor causal claim",
            "b = -13 / 25 and the other methods are secondary analyses",
            "the 2026-11-02 results (seeds 0-7, dense 200 / NCP 400, wiring 22222) are not mixed into "
            "this comparison; old replay results, if kept, are labelled as exploratory on old checkpoints",
            "no equivalence claim when the difference is unclear",
        ],
    }
    jdump(path, B3)
    print(f"wrote {path}\n  sha256 {rr.sha256_file(path)}\n  expected job counts {counts}")


def load_a3_b3(out: Path, *, smoke: bool) -> tuple[dict, str, dict, str]:
    A3, a3_sha = load_a3(out, require_evaluation_seeds=not smoke)
    if smoke and not A3.get("smoke"):
        raise SystemExit("smoke stage requires a --smoke A3 (selection seeds only)")
    b3_path = out / "manifest_B3.json"
    B3 = jload(b3_path)
    rr.verify_b3(A3, a3_sha, B3)
    return A3, a3_sha, B3, rr.sha256_file(b3_path)


# ---------------------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------------------
_W: dict = {}


def _init_worker(a3_path: str) -> None:
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W["A3"] = jload(a3_path)
    _W["arm"], _W["cq"], _W["reaches"] = rr.make_task()
    _W["models"] = {}


def _model(family: str, seed: int):
    key = (family, int(seed))
    if key not in _W["models"]:
        _W["models"][key] = rr.load_a3_model(_W["A3"], family, int(seed), verify=True)
    return _W["models"][key]


def run_job(job: dict) -> dict:
    model = _model(rr.job_family(job), job["seed"])
    return rr.run_protocol_job(_W["arm"], _W["reaches"], model, job)


def run_diag(job: dict) -> dict:
    cond = job["cond"]
    model = _model(rr.METHODS[cond["method"]][0], cond["seed"])
    res = rr.diagnose_condition(_W["arm"], _W["reaches"], model, cond, job["reference_curves"], Path(job["out_dir"]))
    res["pid"] = os.getpid()
    return res


def execute(jobs: list[dict], store: rr.RunStore, workers: int, a3_path: Path, label: str) -> dict:
    todo = store.missing(jobs)
    print(f"[{label}] {len(jobs)} jobs, {len(jobs) - len(todo)} cached, {len(todo)} to run with {workers} workers",
          flush=True)
    t0 = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(a3_path),)) as ex:
            for i, r in enumerate(ex.map(run_job, todo, chunksize=1), 1):
                store.add(r)
                if i % max(1, len(todo) // 10) == 0 or i == len(todo):
                    print(f"  [{label}] {i}/{len(todo)} done, {time.perf_counter() - t0:.0f} s", flush=True)
    return {"label": label, "n_jobs": len(jobs), "n_run": len(todo), "wall_s": time.perf_counter() - t0,
            "workers": workers, "finished_at": datetime.now().astimezone().isoformat()}


def execute_diag(jobs: list[dict], workers: int, a3_path: Path) -> list[dict]:
    out = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(a3_path),)) as ex:
        for i, r in enumerate(ex.map(run_diag, jobs, chunksize=1), 1):
            out.append(r)
            if i % max(1, len(jobs) // 10) == 0 or i == len(jobs):
                print(f"  [diagnostics] {i}/{len(jobs)} done, {time.perf_counter() - t0:.0f} s", flush=True)
    return out


def run_identity(out: Path, a3_sha: str, b3_sha: str) -> dict:
    identity = {"manifest_A3_sha256": a3_sha, "manifest_B3_sha256": b3_sha, "code_commit": git("rev-parse", "HEAD"),
                "runner_sha256": rr.sha256_file(Path(__file__)), "revision_rerun_sha256": rr.sha256_file(Path(rr.__file__))}
    path = out / "run_identity.json"
    if path.exists():
        if jload(path) != identity:
            raise SystemExit("existing output uses a different code or manifest identity; refusing to reuse runs")
    elif (out / "runs.jsonl").exists():
        raise SystemExit("existing runs.jsonl has no identity; refusing to reuse its runs")
    else:
        jdump(path, identity)
    return identity


def append_stage_record(out: Path, rec: dict) -> None:
    path = out / "run_manifest.json"
    rm = jload(path) if path.exists() else {"schema": "rerun400-run-v1", "stages": [],
                                            "created_at": datetime.now().astimezone().isoformat()}
    rm["stages"].append(rec | {"host": platform.node(), "maxrss": maxrss_mb()})
    jdump(path, rm)


# ---------------------------------------------------------------------------------------
# production stages
# ---------------------------------------------------------------------------------------
def stage_select(out: Path, workers: int) -> None:
    A3, a3_sha, B3, b3_sha = load_a3_b3(out, smoke=False)
    run_identity(out, a3_sha, b3_sha)
    sel_path = out / "selection.json"
    refuse_if_exists(sel_path)
    store = rr.RunStore(out / "runs.jsonl")
    jobs = rr.jobs_select(A3)
    timing = execute(jobs, store, workers, out / "manifest_A3.json", "select")
    selection, rows_csv = {}, []
    for m in rr.SELECTED_METHODS:
        grid = A3["eta_grid"][rr.METHODS[m][2]]["grid"]
        by_eta = {eta: {s: store.get(rr.make_job(m, s, rr.SELECTION_FIELD, 1.0, eta, stage="select"))["curves"]
                        for s in rr.SELECTION_SEEDS} for eta in grid}
        res = rr.select_eta_for_seeds(grid, by_eta, rr.SELECTION_SEEDS)
        for row in res["rows"]:
            rows_csv.append({"method": m, "grid_index": row["grid_index"], "eta": row["eta"], "valid": row["valid"],
                             "reasons": ";".join(row["reasons"]), "mean_B_last_mm": row["mean_B_last_mm"],
                             **{f"B_last_mm_seed{s}": row["per_seed"][str(s)]["B_last_mm"] for s in rr.SELECTION_SEEDS},
                             "selected": row["grid_index"] == res["selected_index"]})
        selection[m] = res
        print(f"  {m}: eta {res['selected_eta']} (index {res['selected_index']}) codes {res['codes']}", flush=True)
    jdump(sel_path, selection)
    write_csv(out / "selection_table.csv", rows_csv)
    append_stage_record(out, timing)


def load_selection(out: Path) -> dict:
    p = out / "selection.json"
    if not p.exists():
        raise SystemExit("selection.json missing: run select first")
    return jload(p)


def flat_row(j: dict, r: dict, m: dict, extra: dict | None = None) -> dict:
    kind = rr.METHODS[j["method"]][1]
    row = {"method": j["method"], "family": rr.job_family(j), "seed": j["seed"], "b": j["b"], "pd": j["pd"],
           "eta": j["eta"], "readout_kind": kind, "run_stage": r["stage"], "run_key": rr.job_key(r),
           **(extra or {}), "wall_s": r["wall_s"]}
    for k, v in m.items():
        if k in ("ae_family", "ae_cfc_b0"):
            for kk, vv in v.items():
                row[f"{k.replace('cfc_b0', 'dense_b0')}_{kk}"] = vv
        elif k == "B_curve_mm":
            continue
        elif k == "codes":
            row["codes"] = ";".join(v)
        else:
            row[k] = v
    return row


def _median(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
    return float(np.median(v)) if v else math.nan


def summarise(rows: list[dict], methods, fields) -> list[dict]:
    keys = ["B_first_mm", "B_last_mm", "improvement_vs_B0_family", "improvement_vs_B0_CfC", "adaptation_cycle",
            "catch_signed_mm", "ae_family_ratio", "ae_family_n_mirror", "washout_decay", "A_max_perp_mm",
            "max_cycle_end_dj_norm", "max_cycle_taufb_rms", "offline_val_nmse", "wall_s"]
    out = []
    for m in methods:
        for b in fields:
            sel = [r for r in rows if r["method"] == m and r["b"] == b]
            if not sel:
                continue
            row = {"method": m, "b": b, "eta": sel[0]["eta"], "n_seeds": len(sel),
                   "n_diverged": sum(not r["all_phases_finite"] for r in sel),
                   "n_ae_family_0.25": sum(r.get("ae_family_ae_0.25") is True for r in sel)}
            for k in keys:
                row[f"median_{k}"] = _median([r.get(k, math.nan) for r in sel])
            out.append(row)
    return out


def paired_tables(rows: list[dict]) -> dict:
    """Primary pair and, as secondary descriptions, every adaptive method vs dense LMS at b = 13."""
    def vals(method, b):
        return {r["seed"]: r["B_last_mm"] for r in rows if r["method"] == method and r["b"] == b}
    out = {"primary": {"pair": list(rr.PRIMARY_PAIR), "field_b": rr.SELECTION_FIELD, "metric": "B_last_mm",
                       **rr.paired_log_ratio_bootstrap(vals(rr.PRIMARY_PAIR[0], rr.SELECTION_FIELD),
                                                       vals(rr.PRIMARY_PAIR[1], rr.SELECTION_FIELD))},
           "secondary": {}}
    for b in rr.EVAL_FIELDS:
        for m in rr.ADAPTIVE_METHODS:
            if m == rr.PRIMARY_PAIR[1]:
                continue
            res = rr.paired_log_ratio_bootstrap(vals(m, b), vals(rr.PRIMARY_PAIR[1], b))
            out["secondary"][f"{m} vs {rr.PRIMARY_PAIR[1]} b={b:+g}"] = {k: v for k, v in res.items() if k != "pairs"}
    return out


def stage_evaluate(out: Path, workers: int) -> None:
    A3, a3_sha, B3, b3_sha = load_a3_b3(out, smoke=False)
    run_identity(out, a3_sha, b3_sha)
    selection = load_selection(out)
    store = rr.RunStore(out / "runs.jsonl")
    jobs = rr.jobs_evaluate(selection)
    timing = execute(jobs, store, workers, out / "manifest_A3.json", "evaluate")
    rows = []
    for j in jobs:
        if j["method"] == "PD-only":
            fam = "dense"
        else:
            fam = rr.job_family(j)
        r = store.get(j)
        ref_f = store.get(rr.make_job(rr.B0_OF_FAMILY[fam], j["seed"], j["b"], 1.0, None, stage="evaluate"))
        ref_c = store.get(rr.make_job("B0-dense-CfC", j["seed"], j["b"], 1.0, None, stage="evaluate"))
        m = ro.run_metrics(r["curves"], ref_f and ref_f["curves"], ref_c and ref_c["curves"])
        rows.append(flat_row(j, r, m, extra={"offline_val_nmse": rr.index_row(A3, fam, j["seed"])["val_nmse"],
                                             "B0_reference_method": rr.B0_OF_FAMILY[fam]}))
    write_csv(out / "evaluation_per_seed.csv", rows)
    jdump(out / "evaluation_per_seed.json", rows)
    write_csv(out / "evaluation_summary.csv", summarise(rows, list(rr.METHODS), rr.EVAL_FIELDS))
    jdump(out / "paired_bootstrap.json", paired_tables(rows))
    append_stage_record(out, timing)


def stage_sensitivity(out: Path, workers: int) -> None:
    A3, a3_sha, B3, b3_sha = load_a3_b3(out, smoke=False)
    run_identity(out, a3_sha, b3_sha)
    store = rr.RunStore(out / "runs.jsonl")
    new_jobs = rr.jobs_sensitivity(A3)
    reused = rr.jobs_sensitivity_pd1_reused(A3)
    if store.missing(reused):
        raise SystemExit("PD x1 selection runs missing: run select first")
    timing = execute(new_jobs, store, workers, out / "manifest_A3.json", "sensitivity")
    rows = []
    for j in new_jobs + reused:
        if j["method"].startswith("B0-"):
            continue
        fam = rr.job_family(j)
        r = store.get(j)
        ref_f = store.get(rr.make_job(rr.B0_OF_FAMILY[fam], j["seed"], j["b"], j["pd"], None, stage="sensitivity"))
        ref_c = store.get(rr.make_job("B0-dense-CfC", j["seed"], j["b"], j["pd"], None, stage="sensitivity"))
        m = ro.run_metrics(r["curves"], ref_f and ref_f["curves"], ref_c and ref_c["curves"])
        rows.append(flat_row(j, r, m, extra={"valid_by_selection_rule": not ro.validate_ab_curves(r["curves"])}))
    write_csv(out / "sensitivity_eta_pd_per_seed.csv", rows)
    grid = []
    for m in rr.SELECTED_METHODS:
        for pd in rr.PD_MULTIPLIERS:
            for eta in A3["eta_grid"][rr.METHODS[m][2]]["grid"]:
                sel = [r for r in rows if r["method"] == m and r["pd"] == pd and r["eta"] == eta]
                grid.append({"method": m, "pd": pd, "eta": eta,
                             **{f"B_last_mm_seed{r['seed']}": r["B_last_mm"] for r in sel},
                             **{f"ae_0.25_seed{r['seed']}": r.get("ae_family_ae_0.25") for r in sel},
                             **{f"valid_seed{r['seed']}": r["valid_by_selection_rule"] for r in sel},
                             "n_diverged": sum(not r["all_phases_finite"] for r in sel)})
    write_csv(out / "sensitivity_eta_pd_grid.csv", grid)
    append_stage_record(out, timing)


def stage_diagnostics(out: Path, workers: int, archive_existing: bool = False) -> None:
    A3, a3_sha, B3, b3_sha = load_a3_b3(out, smoke=False)
    run_identity(out, a3_sha, b3_sha)
    selection = load_selection(out)
    store = rr.RunStore(out / "runs.jsonl")
    ddir = out / "diagnostics"
    if ddir.exists() and any(ddir.iterdir()):
        if not archive_existing:
            raise SystemExit("diagnostics already contains output; use --archive-diagnostics to preserve it and rerun")
        ddir.rename(ddir.with_name("diagnostics_archived_" + str(time.time_ns())))
    ddir.mkdir(parents=True, exist_ok=True)
    conds = rr.diagnostic_conditions(selection)
    jobs = []
    for c in conds:
        ref = store.get(rr.make_job(c["method"], c["seed"], c["b"], c["pd"], c["eta"], stage="evaluate"))
        if ref is None:
            raise SystemExit(f"stage-C run missing for {c}; run evaluate first")
        jobs.append({"cond": c, "reference_curves": ref["curves"],
                     "out_dir": str(ddir / f"{c['method']}_s{c['seed']}")})
    t0 = time.perf_counter()
    results = execute_diag(jobs, workers, out / "manifest_A3.json")
    write_diagnostics(ddir, results, {"n_conditions": len(conds), "wall_s": time.perf_counter() - t0,
                                      "workers": workers})
    append_stage_record(out, {"label": "diagnostics", "n_jobs": len(jobs), "wall_s": time.perf_counter() - t0,
                              "workers": workers, "finished_at": datetime.now().astimezone().isoformat()})


def write_diagnostics(ddir: Path, results: list[dict], info: dict) -> dict:
    checks, summaries, probe, reach, cycle, actor = {}, [], [], [], [], []
    for r in results:
        cid = f"{r['cond']['method']}_s{r['cond']['seed']}"
        ok, bad = rr.diagnostics_passed(r["checks"])
        checks[cid] = {"passed": ok, "failed_checks": bad, "bytes": r["bytes"], **r["checks"]}
        summaries.append({**r["summary"], "technical_checks_passed": ok, "bytes": r["bytes"],
                          "determinism": (r["checks"]["determinism_vs_stage_c"] or {}).get("status")})
        probe += r["probe_rows"]
        reach += r["reach_rows"]
        actor += r["actor_rows"]
        for row in r["cycle_rows"]:
            cycle.append({k: (";".join(f"{x:.6g}" for x in v) if isinstance(v, list) else v) for k, v in row.items()})
    write_csv(ddir / "diagnostics_summary.csv", summaries)
    write_csv(ddir / "probe_per_direction.csv", probe)
    write_csv(ddir / "frozen_per_reach.csv", reach)
    write_csv(ddir / "frozen_vs_adaptive_cycles.csv", cycle)
    write_csv(ddir / "actor_probe_per_direction.csv", actor)
    jdump(ddir / "checks.json", checks)
    n_mismatch = sum(1 for c in checks.values() if (c.get("determinism_vs_stage_c") or {}).get("status") == "mismatch")
    status = {"technical_checks_passed": all(c["passed"] for c in checks.values()), "n_conditions": len(checks),
              "n_determinism_mismatch": n_mismatch,
              "n_nonfinite_reproduced": sum(1 for c in checks.values()
                                            if (c.get("determinism_vs_stage_c") or {}).get("status") == "nonfinite_reproduced"),
              "n_adaptive_nonfinite": sum(not s["all_phases_finite"] for s in summaries),
              "total_bytes": sum(c["bytes"] for c in checks.values()), **info,
              "finished_at": datetime.now().astimezone().isoformat(), "environment": environment()}
    jdump(ddir / "status.json", status)
    print(f"diagnostics: technical checks passed {status['technical_checks_passed']}; "
          f"mismatch {n_mismatch}; bytes {status['total_bytes'] / 1e6:.1f} MB")
    if not status["technical_checks_passed"]:
        raise SystemExit("diagnostic technical checks failed; see checks.json and status.json")
    return status


def stage_report(out: Path) -> None:
    A3, a3_sha, B3, b3_sha = load_a3_b3(out, smoke=False)
    store = rr.RunStore(out / "runs.jsonl")
    rep = {"manifest_A3_sha256": a3_sha, "manifest_B3_sha256": b3_sha, "n_runs_stored": len(store.runs),
           "runs_by_stage": {}, "expected_job_counts": B3["expected_job_counts"]}
    for r in store.runs.values():
        rep["runs_by_stage"][r["stage"]] = rep["runs_by_stage"].get(r["stage"], 0) + 1
    if (out / "selection.json").exists():
        sel = load_selection(out)
        rep["selection"] = {m: {"eta": v["selected_eta"], "index": v["selected_index"], "codes": v["codes"]}
                            for m, v in sel.items()}
        rep["job_counts_with_selection"] = rr.job_counts(A3, sel)
    if (out / "evaluation_per_seed.json").exists():
        rows = jload(out / "evaluation_per_seed.json")
        rep["paired_bootstrap"] = paired_tables(rows)
        write_csv(out / "evaluation_summary.csv", summarise(rows, list(rr.METHODS), rr.EVAL_FIELDS))
    if (out / "diagnostics" / "status.json").exists():
        rep["diagnostics_status"] = jload(out / "diagnostics" / "status.json")
        rep["technical_checks_passed"] = rep["diagnostics_status"]["technical_checks_passed"]
    rep["runs_jsonl_bytes"] = (out / "runs.jsonl").stat().st_size if (out / "runs.jsonl").exists() else 0
    jdump(out / "report.json", rep)
    if rep.get("technical_checks_passed") is False:
        raise SystemExit("diagnostics failed technical checks; report saved, not a successful run")
    print(json.dumps({k: rep[k] for k in ("n_runs_stored", "runs_by_stage", "runs_jsonl_bytes")}, indent=1))


# ---------------------------------------------------------------------------------------
# smoke (selection seeds only; shortened protocol; technical checks)
# ---------------------------------------------------------------------------------------
def old_driver_equivalence(arm, reaches) -> dict:
    """New ``run_protocol_job`` on the 2026-11-02 models reproduces the old smoke runs (B2, W1)."""
    if not (OLD_A.exists() and OLD_SMOKE_RUNS.exists()):
        return {"available": False}
    if rr.sha256_file(OLD_A) != OLD_A_SHA:
        return {"available": False, "reason": "old manifest A hash differs"}
    A = jload(OLD_A)
    out = {"available": True, "runs": {}}
    for line in OLD_SMOKE_RUNS.read_text().splitlines():
        old = rr.restore_curves(json.loads(line))
        fam_old = ro.METHODS[old["method"]][0]
        model = ro.load_family_model(A, fam_old, old["seed"])
        job = rr.make_job(OLD_TO_NEW_METHOD[old["method"]], old["seed"], old["b"], old["pd"], old["eta"], stage="smoke",
                          b_cycles=old["b_cycles"], washout_cycles=old["washout_cycles"])
        new = rr.run_protocol_job(arm, reaches, model, job)
        out["runs"][old["method"]] = rr.compare_curves(new["curves"], old["curves"], phases=("A", "B", "C_catch", "C_washout"))
    out["all_identical"] = all(v["status"] == "bit_identical" for v in out["runs"].values())
    return out


def select_eta_equivalence() -> dict:
    """``select_eta_for_seeds`` reproduces ``revision_online.select_eta`` on the old selection runs."""
    if not (OLD_A.exists() and OLD_MAIN_RUNS.exists()):
        return {"available": False}
    A = jload(OLD_A)
    runs = {}
    for line in OLD_MAIN_RUNS.read_text().splitlines():
        r = rr.restore_curves(json.loads(line))
        runs[(r["method"], r["seed"], r["b"], r["pd"], r["eta"], r["b_cycles"], r["washout_cycles"])] = r["curves"]
    out = {"available": True, "methods": {}}
    for m in ro.SELECTED_METHODS:
        grid = A["eta_grid"][ro.METHODS[m][2]]["grid"]
        by_eta = {eta: {s: runs.get((m, s, 13.0, 1.0, eta, 30, 5)) for s in ro.SELECTION_SEEDS} for eta in grid}
        a = ro.select_eta(grid, by_eta)
        b = rr.select_eta_for_seeds(grid, by_eta, ro.SELECTION_SEEDS)
        out["methods"][m] = {"identical": json.dumps(ro.json_safe(a), sort_keys=True) == json.dumps(
            ro.json_safe({k: v for k, v in b.items() if k != "seeds"}), sort_keys=True),
            "selected_eta": a["selected_eta"], "codes": a["codes"]}
    out["all_identical"] = all(v["identical"] for v in out["methods"].values())
    return out


def stage_smoke(out: Path, workers: int) -> None:
    A3, a3_sha, B3, b3_sha = load_a3_b3(out, smoke=True)
    refuse_if_exists(out / "smoke_report.json")
    run_identity(out, a3_sha, b3_sha)
    store = rr.RunStore(out / "runs.jsonl")
    seed = rr.SELECTION_SEEDS[0]
    jobs = []
    for m, (fam, kind, gkey) in rr.METHODS.items():
        eta = A3["eta_grid"][gkey]["center"] if gkey else None
        jobs.append(rr.make_job(m, seed, rr.SELECTION_FIELD, 1.0, eta, stage="smoke", b_cycles=SMOKE_B, washout_cycles=SMOKE_W))
    jobs.append(rr.make_job("PD-only", seed, rr.SELECTION_FIELD, 1.0, None, stage="smoke", b_cycles=SMOKE_B,
                            washout_cycles=SMOKE_W, family_override="esn"))
    jobs += [rr.make_job("B0-dense-CfC", seed, rr.SELECTION_FIELD, pd, None, stage="smoke", b_cycles=SMOKE_B,
                         washout_cycles=SMOKE_W) for pd in rr.SENSITIVITY_EXTRA_PD]
    t_all = time.perf_counter()
    timing = execute(jobs, store, workers, out / "manifest_A3.json", "smoke")
    expected_phases = [("A", 0)] + [("B", i) for i in range(SMOKE_B)] + [("C_catch", 0)] + [("C_washout", i) for i in range(SMOKE_W)]
    rows, checks = [], {}
    for j in jobs:
        r = store.get(j)
        n_reach = 8 * len(r["curves"])
        rows.append({"method": j["method"], "family": rr.job_family(j), "pd": j["pd"], "eta": j["eta"],
                     "phases_ok": [(c["phase"], c["cycle"]) for c in r["curves"]] == expected_phases,
                     "all_finite": all(c["all_finite"] for c in r["curves"]), "wall_s": r["wall_s"],
                     "s_per_reach": r["wall_s"] / n_reach, "est_full_abc_s": r["wall_s"] / n_reach * 8 * (2 + rr.B_CYCLES + rr.WASHOUT_CYCLES),
                     "line_bytes": len(json.dumps(ro.json_safe(r)))})
    checks["all_phases_ok"] = all(r["phases_ok"] for r in rows)
    checks["all_finite_shortened"] = all(r["all_finite"] for r in rows)
    pd_dense = store.get(rr.make_job("PD-only", seed, rr.SELECTION_FIELD, 1.0, None, stage="smoke", b_cycles=SMOKE_B, washout_cycles=SMOKE_W))
    pd_esn = store.get(jobs[len(rr.METHODS)])
    checks["pd_only_two_family"] = rr.compare_curves(pd_dense["curves"], pd_esn["curves"], phases=("A", "B", "C_catch", "C_washout"))
    checks["pd_only_two_family_identical"] = checks["pd_only_two_family"]["status"] == "bit_identical"
    b0 = store.get(rr.make_job("B0-dense-CfC", seed, rr.SELECTION_FIELD, 1.0, None, stage="smoke", b_cycles=SMOKE_B, washout_cycles=SMOKE_W))
    b0B = [c for c in b0["curves"] if c["phase"] == "B"]
    checks["B0_B_cycles_identical"] = rr.compare_cycle(b0B[0], b0B[1])["equal"]
    checks["B0_pd_runs_differ_from_pd1"] = all(
        rr.compare_curves(store.get(j)["curves"], b0["curves"])["status"] == "mismatch" for j in jobs[-2:])

    arm, cq, reaches = rr.make_task()
    checks["old_driver_equivalence"] = old_driver_equivalence(arm, reaches)
    checks["select_eta_equivalence"] = select_eta_equivalence()

    # diagnostics on the shortened runs (7 adaptive methods, seed 100)
    ddir = out / "smoke_diagnostics"
    diag_jobs = []
    for m in rr.ADAPTIVE_METHODS:
        gkey = rr.METHODS[m][2]
        eta = A3["eta_grid"][gkey]["center"] if gkey else None
        ref = store.get(rr.make_job(m, seed, rr.SELECTION_FIELD, 1.0, eta, stage="smoke", b_cycles=SMOKE_B, washout_cycles=SMOKE_W))
        diag_jobs.append({"cond": {"method": m, "seed": seed, "b": rr.SELECTION_FIELD, "pd": 1.0, "eta": eta, "b_cycles": SMOKE_B},
                          "reference_curves": ref["curves"], "out_dir": str(ddir / f"{m}_s{seed}")})
    t0 = time.perf_counter()
    results = execute_diag(diag_jobs, workers, out / "manifest_A3.json")
    diag_status = write_diagnostics(ddir, results, {"n_conditions": len(diag_jobs), "wall_s": time.perf_counter() - t0,
                                                    "workers": workers, "protocol": f"A1 B{SMOKE_B} catch1 (smoke)"})
    checks["diagnostics_technical_checks_passed"] = diag_status["technical_checks_passed"]
    checks["diagnostics_determinism_all_bit_identical"] = all(
        (r["checks"]["determinism_vs_stage_c"] or {}).get("status") == "bit_identical" for r in results)
    checks["ridge_selection"] = {fam: {"setting": A3["offline_selection"][fam]["best"]["setting"],
                                       "grid_index": A3["offline_selection"][fam]["best"]["grid_index"],
                                       "at_grid_edge": A3["offline_selection"][fam]["best"]["at_grid_edge"],
                                       "n_candidate_fits": A3["offline_selection"][fam]["n_candidate_fits"]}
                                 for fam in rr.RIDGE_FAMILIES}
    checks["models_indexed"] = A3["n_models"]
    checks["all_cfc_reload_ok"] = all(row["reload_ok"] for fam in rr.CFC_FAMILIES for row in A3["checkpoint_index"][fam].values())

    # capacity and time estimates (smoke measurements; not guarantees)
    full_cycles, smoke_cycles = 2 + rr.B_CYCLES + rr.WASHOUT_CYCLES, 2 + SMOKE_B + SMOKE_W
    mean_line = float(np.mean([r["line_bytes"] for r in rows])) * full_cycles / smoke_cycles
    per_method_full_s = {r["method"]: r["est_full_abc_s"] for r in rows if r["pd"] == 1.0}
    diag_bytes = [r["bytes"] for r in results]
    counts = B3["expected_job_counts"]
    sizes = A3["spec"]["pilot"]["series_disk_bytes"]
    offline_bytes = sum(max(v for k, v in sizes.items() if k.startswith(c)) for c in rr.CFC_FAMILIES.values()) * len(rr.EVALUATION_SEEDS)
    est = {"runs_jsonl_bytes_per_full_run_est": mean_line, "runs_jsonl_total_est": mean_line * counts["total"],
           "diagnostics_bytes_per_condition_smoke": diag_bytes,
           "diagnostics_total_est_70": float(np.mean(diag_bytes)) * 70 if diag_bytes else math.nan,
           "offline_30_series_bytes_est": offline_bytes,
           "total_online_bytes_est": mean_line * counts["total"] + (float(np.mean(diag_bytes)) * 70 if diag_bytes else 0),
           "free_bytes": shutil.disk_usage(out).free,
           "full_run_s_by_method_est": per_method_full_s,
           "sequential_online_s_est": sum(per_method_full_s.get(m, 0) * n for m, n in _method_run_counts(A3).items()),
           "workers": workers, "smoke_wall_s": time.perf_counter() - t_all,
           "pilot_epoch_s": {k: v["timing"]["epoch_time_s_mean_excl_warmup"] for k, v in
                             jload(PILOT_DIR / "run_record.json")["series"].items()},
           "note": "smoke measurements on shortened protocols; wall-clock estimates are indicative, not guarantees"}
    est["offline_30_series_s_est"] = sum(est["pilot_epoch_s"][series_id(c, 100)] * rr.EPOCHS for c in rr.CFC_FAMILIES.values()) * len(rr.EVALUATION_SEEDS)
    report = {"purpose": "technical smoke on selection seed 100 with shortened protocols; no comparison result",
              "manifest_A3_sha256": a3_sha, "manifest_B3_sha256": b3_sha, "protocol": {"B": SMOKE_B, "washout": SMOKE_W},
              "runs": rows, "checks": checks, "estimates": est, "timing": timing, "maxrss": maxrss_mb(),
              "environment": environment(), "finished_at": datetime.now().astimezone().isoformat()}
    hard = ["all_phases_ok", "pd_only_two_family_identical", "B0_B_cycles_identical", "B0_pd_runs_differ_from_pd1",
            "diagnostics_technical_checks_passed", "diagnostics_determinism_all_bit_identical", "all_cfc_reload_ok"]
    passed = all(checks[k] is True for k in hard)
    for k in ("old_driver_equivalence", "select_eta_equivalence"):
        if checks[k].get("available"):
            passed = passed and checks[k]["all_identical"]
    report["passed"] = passed
    jdump(out / "smoke_report.json", report)
    write_csv(out / "smoke_timing.csv", rows)
    append_stage_record(out, timing | {"label": "smoke", "passed": passed})
    print(f"smoke passed: {passed}")
    for k in hard + ["old_driver_equivalence", "select_eta_equivalence"]:
        v = checks[k]
        print(f"  {k}: {v if not isinstance(v, dict) else v.get('all_identical', v.get('status'))}")


def _method_run_counts(A3: dict) -> dict:
    """Protocol runs per method in the full design (for the sequential-time estimate)."""
    n = {m: 0 for m in rr.METHODS}
    for m in rr.SELECTED_METHODS:
        n[m] += len(rr.ETA_MULTIPLIERS) * len(rr.SELECTION_SEEDS) * len(rr.PD_MULTIPLIERS)  # select + sensitivity
    for m in rr.ADAPTIVE_METHODS:
        n[m] += len(rr.EVALUATION_SEEDS) * len(rr.EVAL_FIELDS) + len(rr.EVALUATION_SEEDS)  # evaluate + diag rerun
    for m in rr.B0_METHODS:
        n[m] += len(rr.EVALUATION_SEEDS) * len(rr.EVAL_FIELDS) + len(rr.SELECTION_SEEDS) * len(rr.PD_MULTIPLIERS)
    n["PD-only"] += len(rr.EVAL_FIELDS)
    return n


# ---------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["spec", "prepare-offline", "design-b3", "smoke", "select", "evaluate",
                                      "sensitivity", "diagnostics", "report"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="prepare-offline: selection seeds only, no training")
    ap.add_argument("--resume", action="store_true", help="prepare-offline: resume interrupted series")
    ap.add_argument("--archive-diagnostics", action="store_true", help="diagnostics: archive existing outputs before explicitly rerunning")
    ap.add_argument("--smoke-report", type=Path, default=None,
                    help="design-b3: smoke_report.json of a separate smoke directory to reference")
    a = ap.parse_args()
    out = a.out.resolve()
    if a.stage == "spec":
        stage_spec(out)
    elif a.stage == "prepare-offline":
        stage_prepare_offline(out, a.smoke, a.resume)
    elif a.stage == "design-b3":
        stage_design_b3(out, a.smoke_report)
    elif a.stage == "smoke":
        stage_smoke(out, a.workers)
    elif a.stage == "select":
        stage_select(out, a.workers)
    elif a.stage == "evaluate":
        stage_evaluate(out, a.workers)
    elif a.stage == "sensitivity":
        stage_sensitivity(out, a.workers)
    elif a.stage == "diagnostics":
        stage_diagnostics(out, a.workers, a.archive_diagnostics)
    else:
        stage_report(out)


if __name__ == "__main__":
    main()
