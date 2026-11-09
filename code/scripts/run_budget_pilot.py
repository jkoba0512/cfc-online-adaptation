"""Training-budget pilot, stage 1 (offline only): manifest A2, smoke, capacity check, 9-series run.

Three implementation configurations (dense_backbone, fc_wired, ncp_wired) x selection seeds
100-102, each trained for 400 full-batch Adam updates with model+optimizer checkpoints every
10 epochs. Epochs 200 and 400 are the recorded comparison points; no validation-based
selection. This is an exploratory comparison of implementation configurations, not a
single-factor (mask-only / backbone-only) causal test, and it does not change the current
manuscript.

Stages (from code/):
  export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/cfc-online-adaptation
  uv run --no-sync python scripts/run_budget_pilot.py prepare --out ../outputs/budget_pilot_<stamp>
  uv run --no-sync python scripts/run_budget_pilot.py smoke   --out ../outputs/budget_pilot_<stamp>   # 3 configs x seed 100 x 20 epochs
  uv run --no-sync python scripts/run_budget_pilot.py dry-run --out ../outputs/budget_pilot_<stamp>   # capacity from smoke sizes
  uv run --no-sync python scripts/run_budget_pilot.py run     --out ../outputs/budget_pilot_<stamp>   # 9 series x 400 epochs (production)
  uv run --no-sync python scripts/run_budget_pilot.py report  --out ../outputs/budget_pilot_<stamp>

``run`` refuses to start unless ``capacity_report.json`` from ``dry-run`` says the free space
is sufficient. ``run --resume`` continues interrupted series from their ``latest.pt``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata as _md
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from cfc_online_adaptation import budget_pilot as bp  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO = CODE_ROOT.parent
SEEDS = (100, 101, 102)
WIRING_SEEDS = {100: 22300, 101: 22301, 102: 22302}
EPOCHS, CKPT_EVERY, RECORD_EPOCHS = 400, 10, (200, 400)
SMOKE_EPOCHS, SMOKE_SEED = 20, 100
CAPACITY_MARGIN = 1.5
SOURCE_FILES = ["scripts/run_budget_pilot.py", "src/cfc_online_adaptation/budget_pilot.py",
                "tests/test_budget_pilot.py", "src/cfc_online_adaptation/core.py",
                "src/cfc_online_adaptation/dataset.py", "src/cfc_online_adaptation/bases.py",
                "src/cfc_online_adaptation/experiment.py", "src/cfc_online_adaptation/tasks.py",
                "src/cfc_online_adaptation/arm.py", "uv.lock"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def git(*a) -> str:
    try:
        return subprocess.check_output(["git", *a], text=True, cwd=CODE_ROOT).strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def environment() -> dict:
    import ncps
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "hostname": platform.node(), "cpu_count": os.cpu_count(),
            "torch": torch.__version__, "numpy": np.__version__,
            "ncps_module_version": getattr(ncps, "__version__", None),
            "ncps_dist_version": _md.version("ncps"), "ncps_lock_version": "1.0.1",
            "torch_num_threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "git_commit": git("rev-parse", "HEAD"), "git_status_code": git("status", "--porcelain", "--", "."),
            "venv": os.environ.get("VIRTUAL_ENV") or sys.prefix}


def series_list() -> list[dict]:
    return [{"series_id": f"{c}_s{s}", "config": c, "seed": s,
             "wiring_seed": WIRING_SEEDS[s] if c != "dense_backbone" else None}
            for c in bp.CONFIGS for s in SEEDS]


def series_config(sr: dict) -> dict:
    return {"config": sr["config"], "seed": sr["seed"], "wiring_seed": sr["wiring_seed"],
            "epochs": EPOCHS, "ckpt_every": CKPT_EVERY, "lr": bp.LR,
            "optimizer": "Adam(lr=1e-3), full batch, one update per epoch, no reset at 200",
            "loss": "MSELoss on teacher torque", "record_epochs": list(RECORD_EPOCHS),
            "selection": "none: epochs 200 and 400 are fixed; no validation-best replacement"}


def seed_collision_scan() -> dict:
    pat = re.compile(r'"(?:seed|wiring_seed|model_seed)"\s*:\s*(\d+)')
    want = set(SEEDS) | set(WIRING_SEEDS.values())
    hits = {}
    for p in (REPO / "outputs").rglob("*.json"):
        if "budget_pilot" in str(p):
            continue
        try:
            found = {int(v) for v in pat.findall(p.read_text(errors="ignore"))} & want
        except OSError:
            continue
        if found:
            hits[str(p.relative_to(REPO))] = sorted(found)
    return {"seeds_checked": sorted(want), "files_with_matching_seed_values": hits,
            "collision": bool(hits)}


# ----------------------------------------------------------------------------------------
def prepare(out: Path) -> None:
    mpath = out / "manifest_A2.json"
    if mpath.exists():
        raise SystemExit(f"{mpath} exists; refusing to overwrite")
    arm, cq = bp.make_arm_and_center()
    ds, audit = bp.training_dataset(arm, cq), bp.audit_dataset(arm, cq)
    splits = {}
    for s in SEEDS:
        tr, va = bp.split_for_seed(ds, s)
        splits[str(s)] = {"n_train": len(tr), "n_val": len(va),
                          "train_sha256_x": sha256_array(tr.x), "val_sha256_x": sha256_array(va.x),
                          "val_radii": sorted({m["radius"] for m in va.meta}),
                          "val_directions": bp.direction_index(va).tolist()}
    audits, reload_checks = {}, {}
    for sr in series_list():
        m = bp.build_model(sr["config"], sr["seed"], sr["wiring_seed"])
        audits[sr["series_id"]] = bp.audit_model(m)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "init.pt"
            xv = torch.from_numpy(ds.x[:2].astype(np.float32))
            bp.atomic_torch_save({"format_version": bp.CHECKPOINT_FORMAT_VERSION, "epoch": 0,
                                  "model_state": m.state_dict(), "optimizer_state": {},
                                  **bp._rng_states(), "config": series_config(sr), "provenance": {},
                                  "verification": bp._verification(m, xv), "complete": True}, p)
            reload_checks[sr["series_id"]] = bp.reload_check(
                lambda sr=sr: bp.build_model(sr["config"], sr["seed"], sr["wiring_seed"]), p, ds.x[:2])
            reload_checks[sr["series_id"]]["meaning"] = ("initial (epoch 0) state: fresh instance loading the "
                                                         "saved state reproduces the saved model's forward")
    collisions = seed_collision_scan()
    manifest = {
        "schema": "budget-pilot-A2-v1",
        "plan": "cfc-online-adaptation_training-budget-and-fit-matched-plan_2026-11-07 (v0.3), stage 1",
        "status": "offline pilot of three implementation configurations; exploratory; "
                  "not a single-factor causal comparison; does not alter the current manuscript",
        "n_series": len(series_list()), "series": series_list(),
        "configs": {"dense_backbone": "CfCInverseModel: CfCCell with default 128-unit backbone",
                    "fc_wired": "FullyConnectedWiredCfCInverseModel: WiredCfCCell, one layer, all-ones mask",
                    "ncp_wired": "WiredCfCInverseModel: WiredCfCCell, AutoNCP(64,2,0.5), three layers"},
        "data": {"center_xy": list(bp.CENTER_XY), "train_radii": list(bp.TRAIN_RADII), "directions": bp.N_TARGETS,
                 "duration": bp.DURATION, "dt": bp.DT, "n_sequences": len(ds), "val_frac": bp.VAL_FRAC,
                 "split_rule": "dataset.train_val_split(seed=model seed); shared by the three configurations",
                 "x_sha256": sha256_array(ds.x), "tau_sha256": sha256_array(ds.tau), "splits": splits,
                 "audit_set": {"radii": list(bp.AUDIT_RADII), "n_sequences": len(audit),
                               "x_sha256": sha256_array(audit.x), "tau_sha256": sha256_array(audit.tau),
                               "use": "evaluation only at checkpoints; never used for selection; "
                                      "radii are interpolated between training radii"}},
        "training": {"epochs": EPOCHS, "ckpt_every": CKPT_EVERY, "record_epochs": list(RECORD_EPOCHS),
                     "lr": bp.LR, "optimizer": "Adam", "batch": "full", "loss": "MSE",
                     "continuous": "200 to 400 without optimizer reset or scheduler",
                     "selection": "none (fixed epochs; no validation-best replacement)",
                     "equal_optimizer_updates_note": "equal update counts are not equal wall time or FLOPs; "
                                                     "parameter counts and measured times are reported separately"},
        "checkpoint": {"format_version": bp.CHECKPOINT_FORMAT_VERSION, "write": "tmp file then atomic rename",
                       "contents": ["model_state", "optimizer_state", "torch/numpy RNG", "epoch", "config", "summary"],
                       "latest_copy": "latest.pt rewritten at every save for resumption"},
        "timing": {"epoch_time": "per-epoch train step + validation forward, checkpoint IO excluded, "
                                 "first 2 epochs excluded from the mean", "forward": "median of 5 no-grad "
                   "forwards over the validation set after a warm-up", "threads": torch.get_num_threads()},
        "seed_collision_scan": collisions,
        "audits": audits, "reload_checks": reload_checks,
        "source_sha256": {n: sha256_file(CODE_ROOT / n) for n in SOURCE_FILES},
        "environment": environment(),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    out.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    print(f"wrote {mpath}: {manifest['n_series']} series, collision={collisions['collision']}")
    for sid, a in audits.items():
        print(f"  {sid:22s} core total={a['totals']['core']['total']:6d} trainable={a['totals']['core']['trainable']:6d}"
              f" layers={a['structure']['layer_sizes']} masks={[m['nonzero'] for m in a['masks']]}")


def load_manifest(out: Path) -> dict:
    m = json.loads((out / "manifest_A2.json").read_text())
    changed = [n for n, h in m["source_sha256"].items() if sha256_file(CODE_ROOT / n) != h]
    if changed:
        raise SystemExit(f"sources changed since prepare: {changed}")
    m["_sha256"] = sha256_file(out / "manifest_A2.json")
    return m


def data_for(m: dict, seed: int):
    """Regenerate the datasets and verify them against the hashes fixed in the manifest."""
    arm, cq = bp.make_arm_and_center()
    ds, audit = bp.training_dataset(arm, cq), bp.audit_dataset(arm, cq)
    tr, va = bp.split_for_seed(ds, seed)
    want = {"x": m["data"]["x_sha256"], "tau": m["data"]["tau_sha256"],
            "audit_x": m["data"]["audit_set"]["x_sha256"], "audit_tau": m["data"]["audit_set"]["tau_sha256"],
            "train_x": m["data"]["splits"][str(seed)]["train_sha256_x"],
            "val_x": m["data"]["splits"][str(seed)]["val_sha256_x"]}
    got = {"x": sha256_array(ds.x), "tau": sha256_array(ds.tau), "audit_x": sha256_array(audit.x),
           "audit_tau": sha256_array(audit.tau), "train_x": sha256_array(tr.x), "val_x": sha256_array(va.x)}
    bad = [k for k in want if want[k] != got[k]]
    if bad:
        raise SystemExit(f"regenerated data differ from manifest hashes: {bad}")
    return tr, va, audit


def run_one(m: dict, sr: dict, out_dir: Path, epochs: int, resume: bool, restart_unsaved: bool = False) -> dict:
    tr, va, audit = data_for(m, sr["seed"])
    model = bp.build_model(sr["config"], sr["seed"], sr["wiring_seed"])
    cfg = series_config(sr) | {"epochs": epochs}
    prov = {"manifest_A2_sha256": m["_sha256"], "series_id": sr["series_id"],
            "data_x_sha256": m["data"]["x_sha256"], "schema": m["schema"]}
    res = bp.train_series(model, tr, va, audit, epochs=epochs, seed=sr["seed"], out_dir=out_dir,
                          ckpt_every=CKPT_EVERY, config=cfg, resume=resume, provenance=prov,
                          allow_restart_unsaved=restart_unsaved)
    last = out_dir / f"epoch{epochs:04d}.pt"
    res["reload_check"] = bp.reload_check(
        lambda: bp.build_model(sr["config"], sr["seed"], sr["wiring_seed"]), last, va.x)
    fresh_masks = [m_["sha256"] for m_ in bp.audit_model(bp.build_model(sr["config"], sr["seed"], sr["wiring_seed"]))["masks"]]
    res["reload_check"]["mask_sha256_matches_fresh_instance"] = res["reload_check"]["mask_sha256"] == fresh_masks
    res["data_hashes_verified_against_manifest"] = True
    res["checkpoint_sha256"] = {p.name: sha256_file(p) for p in sorted(out_dir.glob("epoch*.pt"))}
    res["tmp_files_left"] = [p.name for p in out_dir.glob("*.tmp")]
    return res


# ----------------------------------------------------------------------------------------
def smoke(out: Path) -> None:
    m = load_manifest(out)
    sdir = out / "smoke"
    if sdir.exists():
        raise SystemExit("smoke output exists; refusing to overwrite")
    report = {"purpose": "timing, checkpoint size, reload and resume checks only; not a scientific result",
              "epochs": SMOKE_EPOCHS, "seed": SMOKE_SEED, "series": {}}
    for sr in [s for s in m["series"] if s["seed"] == SMOKE_SEED]:
        d = sdir / sr["series_id"]
        res = run_one(m, sr, d, SMOKE_EPOCHS, resume=False)
        # resume check: 10 epochs, then resume to 20, compare curve with the continuous run
        d2 = sdir / (sr["series_id"] + "_resume")
        run_one(m, sr, d2, SMOKE_EPOCHS // 2, resume=False)
        run_one(m, sr, d2, SMOKE_EPOCHS, resume=True)
        c1 = [json.loads(line) for line in (d / "curve.jsonl").read_text().splitlines()]
        c2 = [json.loads(line) for line in (d2 / "curve.jsonl").read_text().splitlines()]
        keys = ("train_mse", "val_mse")
        maxdiff = max(abs(a[k] - b[k]) for a, b in zip(c1, c2) for k in keys)
        res["resume_check"] = {"epochs_continuous": len(c1), "epochs_resumed": len(c2),
                               "bit_identical": all(a[k] == b[k] for a, b in zip(c1, c2) for k in keys),
                               "max_abs_diff": maxdiff, "within_1e-12": maxdiff <= 1e-12,
                               "final_state_bit_identical": bool(all(
                                   torch.equal(x, y) for x, y in zip(
                                       bp.load_checkpoint(d / f"epoch{SMOKE_EPOCHS:04d}.pt")["model_state"].values(),
                                       bp.load_checkpoint(d2 / f"epoch{SMOKE_EPOCHS:04d}.pt")["model_state"].values())))}
        res["checkpoint_rule_ok"] = res["saved_epochs"] == list(range(CKPT_EVERY, SMOKE_EPOCHS + 1, CKPT_EVERY))
        report["series"][sr["series_id"]] = res
        print(f"[{sr['series_id']}] epoch {res['timing']['epoch_time_s_mean_excl_warmup']:.3f}s "
              f"forward {res['timing']['forward_val_s_median']:.4f}s ckpt {res['checkpoint_bytes']['latest.pt']} B "
              f"resume bit-identical={res['resume_check']['bit_identical']}", flush=True)
    report["environment"] = environment()
    (out / "smoke_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    ok = all(r["reload_check"]["masks_equal_after_reload"] and r["reload_check"]["forward_bit_identical_to_saved_model"]
             and r["reload_check"]["masks_equal_to_saved_digests"] and r["reload_check"]["input_matches_saved_input"]
             and r["resume_check"]["within_1e-12"] and r["checkpoint_rule_ok"] and not r["tmp_files_left"]
             for r in report["series"].values())
    print(f"smoke checks passed: {ok}")


def dry_run(out: Path) -> None:
    m = load_manifest(out)
    s = json.loads((out / "smoke_report.json").read_text())
    n_ckpt = EPOCHS // CKPT_EVERY
    per_series, total = {}, 0
    for sr in m["series"]:
        sm = s["series"][f"{sr['config']}_s{SMOKE_SEED}"]
        ck = max(v for k, v in sm["checkpoint_bytes"].items() if k != "latest.pt")
        curve = 400 * 160 + n_ckpt * 2500  # per-epoch jsonl line + per-checkpoint summary (upper bound)
        est = n_ckpt * ck + ck + ck + curve  # checkpoints + latest + one tmp during write
        per_series[sr["series_id"]] = {"checkpoint_bytes": ck, "n_checkpoints": n_ckpt, "estimate_bytes": est,
                                       "epoch_time_s": sm["timing"]["epoch_time_s_mean_excl_warmup"],
                                       "estimated_series_time_s": sm["timing"]["epoch_time_s_mean_excl_warmup"] * EPOCHS}
        total += est
    free = shutil.disk_usage(out).free
    required = int(total * CAPACITY_MARGIN)
    future = {"series": 39, "checkpoints_per_series": 80,
              "estimate_bytes": int(sum(per_series[f"{c}_s{SMOKE_SEED}"]["checkpoint_bytes"] * 82 for c in bp.CONFIGS) / 3 * 39)}
    rep = {"n_series": m["n_series"], "epochs": EPOCHS, "ckpt_every": CKPT_EVERY,
           "per_series": per_series, "total_estimate_bytes": total, "margin": CAPACITY_MARGIN,
           "required_bytes": required, "free_bytes_at": str(out), "free_bytes": free,
           "sufficient": free >= required,
           "estimated_sequential_time_s": sum(v["estimated_series_time_s"] for v in per_series.values()),
           "future_39_series_80_checkpoints": future, "note": "future estimate reported only; not reserved"}
    (out / "capacity_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: rep[k] for k in ("n_series", "total_estimate_bytes", "required_bytes", "free_bytes",
                                           "sufficient", "estimated_sequential_time_s")}, indent=1))


def run(out: Path, only: str | None, resume: bool, restart_unsaved: bool = False) -> None:
    m = load_manifest(out)
    cap = out / "capacity_report.json"
    if not cap.exists():
        raise SystemExit("run dry-run first")
    rep = json.loads(cap.read_text())
    if not rep["sufficient"] or shutil.disk_usage(out).free < rep["required_bytes"]:
        raise SystemExit("insufficient free space for the production run")
    rdir = out / "series"
    record_path = out / "run_record.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {"series": {}}
    todo = [s for s in m["series"] if only is None or s["series_id"] == only]
    print(f"production: {len(todo)} of {m['n_series']} series, {EPOCHS} epochs each", flush=True)
    for sr in todo:
        d = rdir / sr["series_id"]
        if (d / f"epoch{EPOCHS:04d}.pt").exists() and not resume:
            print(f"[{sr['series_id']}] complete; skipping", flush=True)
            continue
        t0 = datetime.now().astimezone()
        res = run_one(m, sr, d, EPOCHS, resume=resume, restart_unsaved=restart_unsaved)
        res["started_at"], res["finished_at"] = t0.isoformat(), datetime.now().astimezone().isoformat()
        record["series"][sr["series_id"]] = res
        record_path.write_text(json.dumps(record, indent=2, default=float) + "\n")
        print(f"[{sr['series_id']}] done: {res['timing']['epoch_time_s_mean_excl_warmup']:.3f} s/epoch", flush=True)
    record["environment"] = environment()
    record_path.write_text(json.dumps(record, indent=2, default=float) + "\n")
    report(out)


def report(out: Path) -> None:
    """Summary table from curve and checkpoint files only (no training)."""
    m = load_manifest(out)
    rows = []
    for sr in m["series"]:
        d = out / "series" / sr["series_id"]
        if not (d / "checkpoint_summaries.jsonl").exists():
            continue
        summ = {s["epoch"]: s for s in (json.loads(line) for line in (d / "checkpoint_summaries.jsonl").read_text().splitlines())}
        row = {"series_id": sr["series_id"], "config": sr["config"], "seed": sr["seed"]}
        for e in RECORD_EPOCHS:
            s = summ.get(e)
            for k in ("train", "val", "audit"):
                row[f"{k}_nmse_e{e}"] = s[k]["nmse"] if s else math.nan
            row[f"feature_energy_e{e}"] = s["feature_energy_train"] if s else math.nan
            row[f"readout_norm_e{e}"] = s["readout_norm"] if s else math.nan
        if all(e in summ for e in RECORD_EPOCHS):
            row["val_nmse_change_200_to_400"] = summ[400]["val"]["nmse"] - summ[200]["val"]["nmse"]
            row["val_nmse_ratio_400_over_200"] = summ[400]["val"]["nmse"] / summ[200]["val"]["nmse"]
        rows.append(row)
    if rows:
        with (out / "summary_200_400.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote summary for {len(rows)} series")
    else:
        print("no completed series yet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("prepare", "smoke", "dry-run", "run", "report"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", help="run: a single series_id")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--restart-unsaved", action="store_true",
                    help="with --resume: a series interrupted before its first checkpoint is restarted")
    a = ap.parse_args()
    out = a.out.resolve()
    if a.stage == "prepare":
        prepare(out)
    elif a.stage == "smoke":
        smoke(out)
    elif a.stage == "dry-run":
        dry_run(out)
    elif a.stage == "run":
        run(out, a.only, a.resume, a.restart_unsaved)
    else:
        report(out)


if __name__ == "__main__":
    main()
