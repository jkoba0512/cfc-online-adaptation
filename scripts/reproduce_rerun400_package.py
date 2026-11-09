"""Portable 400-update reproduction. Run from the extracted supplement root."""

from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code/scripts"))
from cfc_online_adaptation import revision_rerun as rr  # noqa: E402
from cfc_online_adaptation import budget_pilot as bp  # noqa: E402
from cfc_online_adaptation import revision_online as ro  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_package():
    count = 0
    for line in (ROOT / "PACKAGE_SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if sha(ROOT / name) != digest:
            raise ValueError("Package checksum mismatch: " + name)
        count += 1
    return count


ADAPTIVE_METHODS = (
    "dense-CfC-LMS",
    "NCP-CfC-LMS",
    "FC-CfC-LMS",
    "ESN-LMS",
    "RBF-LMS",
    "dense-CfC-RLS",
)
TUNED_METHODS = ADAPTIVE_METHODS[:-1]
REFERENCE_METHODS = (*rr.B0_METHODS, "PD-only")


def jobs_select(a):
    return [
        rr.make_job(m, seed, 13.0, 1.0, eta, stage="select")
        for m in TUNED_METHODS
        for eta in a["eta_grid"][rr.METHODS[m][2]]["grid"]
        for seed in rr.SELECTION_SEEDS
    ]


def jobs_evaluate(selection):
    jobs = [
        rr.make_job(
            m,
            seed,
            b,
            1.0,
            selection[m]["selected_eta"] if m in TUNED_METHODS else None,
            stage="evaluate",
        )
        for m in ADAPTIVE_METHODS
        for seed in rr.EVALUATION_SEEDS
        for b in rr.EVAL_FIELDS
    ]
    jobs += [
        rr.make_job(m, seed, b, 1.0, None, stage="evaluate")
        for m in rr.B0_METHODS
        for seed in rr.EVALUATION_SEEDS
        for b in rr.EVAL_FIELDS
    ]
    jobs += [
        rr.make_job("PD-only", rr.EVALUATION_SEEDS[0], b, 1.0, None, stage="evaluate")
        for b in rr.EVAL_FIELDS
    ]
    return jobs


def jobs_sensitivity(a):
    jobs = [
        rr.make_job(m, seed, 13.0, pd, None, stage="sensitivity")
        for pd in rr.PD_MULTIPLIERS
        for seed in rr.SELECTION_SEEDS
        for m in rr.B0_METHODS
    ]
    jobs += [
        rr.make_job(m, seed, 13.0, pd, eta, stage="sensitivity")
        for pd in rr.SENSITIVITY_EXTRA_PD
        for seed in rr.SELECTION_SEEDS
        for m in TUNED_METHODS
        for eta in a["eta_grid"][rr.METHODS[m][2]]["grid"]
    ]
    return jobs


def diagnostic_conditions(selection):
    return [
        {
            "method": m,
            "seed": seed,
            "b": 13.0,
            "pd": 1.0,
            "eta": selection[m]["selected_eta"] if m in TUNED_METHODS else None,
            "b_cycles": rr.B_CYCLES,
        }
        for m in ADAPTIVE_METHODS
        for seed in rr.EVALUATION_SEEDS
    ]


def runtime_manifest():
    a = json.loads((ROOT / "manifests/manifest_A3.json").read_text())
    design = json.loads((ROOT / "manifests/dataset_scope.json").read_text())
    if a.get("smoke") or a["n_models"] != 65:
        raise ValueError("Expected all 65 final models")
    if design["counts"]["stored_online"] != 783 or design["counts"]["diagnostics"] != 60:
        raise ValueError("Dataset counts differ from the six-method experiment")
    if design["included_adaptive_methods"] != list(ADAPTIVE_METHODS):
        raise ValueError("Dataset method list differs from the six-method reproducer")
    if a["spec_sha256"] != sha(ROOT / "manifests/manifest_A3_spec.json"):
        raise ValueError("Model specification checksum differs")
    if rr.verify_sources(a["spec"]["source_sha256"], ROOT / "code"):
        raise ValueError("Numerical source checksum differs")
    if a["spec"]["selection_seeds"] != list(rr.SELECTION_SEEDS) or a["spec"][
        "evaluation_seeds"
    ] != list(rr.EVALUATION_SEEDS):
        raise ValueError("Seed roles differ")
    if set(a["eta_grid"]) != set(rr.FAMILIES):
        raise ValueError("Expected learning-rate grids for five feature families")
    for family in rr.FAMILIES:
        if len(a["eta_grid"][family]["grid"]) != 9:
            raise ValueError("Expected nine candidate rates")
        for seed in rr.ALL_SEEDS:
            row = a["checkpoint_index"][family][str(seed)]
            row["checkpoint"] = str(ROOT / row["checkpoint"])
            if sha(row["checkpoint"]) != row["sha256"]:
                raise ValueError("Checkpoint checksum differs")
            if family in rr.CFC_FAMILIES and not row["reload_ok"]:
                raise ValueError("Checkpoint reload validation missing")
    selection = json.loads((ROOT / "results/selection.json").read_text())
    if set(selection) != set(TUNED_METHODS):
        raise ValueError("Selection must contain exactly five tuned methods")
    expected = jobs_select(a) + jobs_evaluate(selection) + jobs_sensitivity(a)
    expected_keys = {rr.job_key(j) for j in expected}
    supplied = rr.RunStore(ROOT / "results/runs.jsonl")
    if len(expected) != 783 or len(expected_keys) != 783 or set(supplied.runs) != expected_keys:
        raise ValueError("Supplied protocols do not match the six-method experiment")
    if design["figure_input_sha256"] != {
        name: sha(ROOT / "results" / name) for name in design["figure_input_sha256"]
    }:
        raise ValueError("Dataset source checksum differs")
    return a


WORK = {}


def init_worker(a):
    torch.set_num_threads(1)
    WORK.update(a=a, models={}, task=rr.make_task())


def model(family, seed):
    key = (family, seed)
    if key not in WORK["models"]:
        WORK["models"][key] = rr.load_a3_model(WORK["a"], family, seed, verify=True)
    return WORK["models"][key]


def online_job(j):
    arm, _, reaches = WORK["task"]
    return rr.run_protocol_job(arm, reaches, model(rr.job_family(j), j["seed"]), j)


def diag_job(job):
    c, reference, out = job
    arm, _, reaches = WORK["task"]
    result = rr.diagnose_condition(
        arm, reaches, model(rr.METHODS[c["method"]][0], c["seed"]), c, reference, Path(out)
    )
    ok, bad = rr.diagnostics_passed(result["checks"])
    if not ok:
        raise ValueError(f"Diagnostic checks failed: {bad}")
    reference_dir = ROOT / "results/diagnostics" / Path(out).name
    n_arrays = 0
    for p in reference_dir.glob("*.npz"):
        with (
            np.load(p, allow_pickle=False) as expected,
            np.load(Path(out) / p.name, allow_pickle=False) as got,
        ):
            if not set(expected.files).issubset(got.files):
                raise ValueError("Missing array keys: " + str(p))
            derived = json.loads((ROOT / "DERIVED_FEATURE_HASHES.json").read_text()).get(
                str(p.relative_to(ROOT))
            )
            if (
                derived
                and hashlib.sha256(np.ascontiguousarray(got["Z"]).tobytes()).hexdigest()
                != derived["sha256"]
            ):
                raise ValueError("Derived features differ")
            for key in expected.files:
                if not rr.array_state_equal(expected[key], got[key]):
                    raise ValueError(f"Array mismatch {p.name}:{key}")
                n_arrays += 1
    return {
        "method": c["method"],
        "seed": c["seed"],
        "arrays_bit_identical": n_arrays,
        "technical_checks_passed": ok,
    }


def run_jobs(jobs, store, a, workers):
    todo = store.missing(jobs)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(a,)) as ex:
        for index, r in enumerate(ex.map(online_job, todo, chunksize=1), 1):
            store.add(r)
            if index % 25 == 0 or index == len(todo):
                print(f"{r['stage']}: {index}/{len(todo)}", flush=True)
    return len(todo)


def online(a, out, workers):
    reference = rr.RunStore(ROOT / "results/runs.jsonl")
    store = rr.RunStore(out / "runs.jsonl")
    n = run_jobs(jobs_select(a), store, a, workers)
    selected = {}
    supplied = json.loads((ROOT / "results/selection.json").read_text())
    for m in TUNED_METHODS:
        grid = a["eta_grid"][rr.METHODS[m][2]]["grid"]
        runs = {
            eta: {
                s: store.get(rr.make_job(m, s, 13.0, 1.0, eta, stage="select"))["curves"]
                for s in rr.SELECTION_SEEDS
            }
            for eta in grid
        }
        selected[m] = rr.select_eta_for_seeds(grid, runs, rr.SELECTION_SEEDS)
        if selected[m]["selected_eta"] != supplied[m]["selected_eta"]:
            raise ValueError("Selected rate differs: " + m)
    n += run_jobs(jobs_evaluate(selected), store, a, workers)
    n += run_jobs(jobs_sensitivity(a), store, a, workers)
    # Compare persisted records consistently: JSON represents NaN/Inf as null.
    store = rr.RunStore(out / "runs.jsonl")
    comparisons = {}
    for key, r in store.runs.items():
        ref = reference.runs[key]
        status = rr.compare_curves(
            r["curves"], ref["curves"], phases=("A", "B", "C_catch", "C_washout")
        )
        if status["status"] == "mismatch":
            raise ValueError("Protocol differs: " + key)
        comparisons[key] = status["status"]
    (out / "selection.json").write_text(json.dumps(ro.json_safe(selected), indent=2) + "\n")
    return {
        "protocol_runs": len(store.runs),
        "computed_runs": n,
        "previously_computed_runs": len(store.runs) - n,
        "comparison_representation": "Persisted JSON: NaN and infinities share the stored null representation",
        "selected_rates_match": True,
        "all_protocol_curves_match": True,
        "statuses": comparisons,
    }


def diagnostics(a, out, workers, seeds):
    selection = json.loads((ROOT / "results/selection.json").read_text())
    ref = rr.RunStore(ROOT / "results/runs.jsonl")
    jobs = []
    for c in diagnostic_conditions(selection):
        if seeds and c["seed"] not in seeds:
            continue
        curves = ref.get(
            rr.make_job(c["method"], c["seed"], c["b"], c["pd"], c["eta"], stage="evaluate")
        )["curves"]
        jobs.append((c, curves, str(out / f"{c['method']}_s{c['seed']}")))
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(a,)) as ex:
        checks = list(ex.map(diag_job, jobs, chunksize=1))
    return {"conditions": len(checks), "checks": checks, "all_saved_arrays_bit_identical": True}


def snapshots(a, out, seeds):
    from cfc_online_adaptation import readout_probe as rp

    torch.set_num_threads(1)
    arm, _, reaches = rr.make_task()
    checks = []
    for method in ADAPTIVE_METHODS:
        for seed in rr.EVALUATION_SEEDS:
            if seeds and seed not in seeds:
                continue
            family, kind, _ = rr.METHODS[method]
            m = rr.load_a3_model(a, family, seed)
            rdir = ROOT / "results/diagnostics" / f"{method}_s{seed}"
            pre, post = (
                rp.load_snapshot(rdir / "readout_pre.npz"),
                rp.load_snapshot(rdir / "readout_post.npz"),
            )
            for actor, predictor in [("pre", pre), ("post", post)]:
                res, unchanged = rr._frozen_cycle(
                    arm, reaches, m, predictor, pre.J.copy(), 13.0, 1.0, kind
                )
                if not unchanged:
                    raise ValueError("Readout changed")
                got = rr._records(res, reaches)
                with np.load(rdir / f"frozen_{actor}_b13.npz") as expected:
                    for k in ["q", "qdot", "tau_ff", "tau_fb", "tau", "ee", "ee_des"]:
                        if not rr.array_state_equal(got[k], expected[k]):
                            raise ValueError(f"Frozen mismatch {method}:{seed}:{k}")
            with np.load(rdir / "probe_waveforms.npz") as expected:
                for d, reach in enumerate(reaches):
                    z = rp.nominal_features(m, reach)
                    record = json.loads((ROOT / "DERIVED_FEATURE_HASHES.json").read_text())[
                        str((rdir / "probe_waveforms.npz").relative_to(ROOT))
                    ]
                    # Full-array feature digest is checked once per condition below.
                    for name, predictor in [("tau_pre", pre), ("tau_post", post)]:
                        y = rp.predict_series(predictor, z)
                        if not rr.array_state_equal(y, expected[name][d]):
                            raise ValueError("Nominal torque differs")
            features = np.stack([rp.nominal_features(m, reach) for reach in reaches])
            if (
                hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
                != record["sha256"]
            ):
                raise ValueError("Nominal feature digest differs")
            checks.append({"method": method, "seed": seed, "frozen_and_nominal_arrays_match": True})
    return {"conditions": len(checks), "checks": checks}


def offline(a, out, seeds):
    torch.set_num_threads(1)
    checks = []
    for family, config in rr.CFC_FAMILIES.items():
        for seed in rr.ALL_SEEDS:
            if seeds and seed not in seeds:
                continue
            tr, va, audit = rr.split_data(seed)
            wiring = None if family == "dense" else rr.WIRING_SEEDS[seed]
            m = bp.build_model(config, seed, wiring)
            bp.train_series(
                m,
                tr,
                va,
                audit,
                epochs=400,
                seed=seed,
                out_dir=out / f"{family}_s{seed}",
                ckpt_every=10,
                config={"family": family, "seed": seed, "epoch": 400},
            )
            expected = bp.load_checkpoint(Path(rr.index_row(a, family, seed)["checkpoint"]))[
                "model_state"
            ]
            if not all(torch.equal(v, expected[k]) for k, v in m.state_dict().items()):
                raise ValueError("Offline state differs")
            checks.append({"family": family, "seed": seed, "all_model_tensors_bit_identical": True})
    # Refit ridge models with supplied settings/penalties; full candidate grids are supplied in A3.
    for family in rr.RIDGE_FAMILIES:
        for seed in rr.ALL_SEEDS:
            if seeds and seed not in seeds:
                continue
            from cfc_online_adaptation.bases import fit_and_validate

            tr, va, _ = rr.split_data(seed)
            row = rr.index_row(a, family, seed)
            setting = a["offline_selection"][family]["best"]["setting"]
            m = rr.build_ridge_model(family, setting, tr.x, seed)
            fit_and_validate(m, tr.x, tr.tau, va.x, va.tau, [1e-4, 1e-3, 1e-2, 1e-1, 1])
            expected = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
            if not all(torch.equal(v, expected[k]) for k, v in m.state_dict().items()):
                raise ValueError("Ridge state differs")
            checks.append({"family": family, "seed": seed, "all_model_tensors_bit_identical": True})
    return {"models": len(checks), "checks": checks}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=["verify", "figures", "snapshots", "diagnostics", "online", "offline"],
        required=True,
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--resume", action="store_true", help="Resume an existing online output directory"
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="*")
    a = p.parse_args()
    out = a.out.resolve()
    if a.resume and a.stage != "online":
        raise SystemExit("--resume is supported only for online")
    if out.exists() and not a.resume:
        raise SystemExit("Output directory must be new, or use --resume for online")
    verified = verify_package()
    manifest = runtime_manifest()
    out.mkdir(parents=True, exist_ok=a.resume)
    if a.stage == "verify":
        result = {"models": 65, "source_files": len(manifest["spec"]["source_sha256"])}
    elif a.stage == "figures":
        from plot_rerun400_submission import generate

        scope = ROOT / "manifests/dataset_scope.json"
        generate(ROOT / "results", out, scope=scope if scope.exists() else None)
        result = {"figures_and_tables_regenerated": True}
        if scope.exists():
            result["publication_methods"] = json.loads(scope.read_text())[
                "included_adaptive_methods"
            ]
    elif a.stage == "snapshots":
        result = snapshots(manifest, out, set(a.seeds or []))
    elif a.stage == "diagnostics":
        result = diagnostics(manifest, out, a.workers, set(a.seeds or []))
    elif a.stage == "online":
        result = online(manifest, out, a.workers)
    else:
        result = offline(manifest, out, set(a.seeds or []))
    report = {
        "stage": a.stage,
        "package_files_verified": verified,
        "seeds_filter": a.seeds,
        "checks": result,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    (out / "REPRODUCTION_REPORT.json").write_text(json.dumps(ro.json_safe(report), indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}))
    print("Reproduction checks passed")


if __name__ == "__main__":
    main()
