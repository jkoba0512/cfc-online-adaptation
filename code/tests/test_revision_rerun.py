"""400-epoch rerun helpers: seed-explicit selection rules, loader checks, job design, curve
comparison with non-finite patterns, bootstrap and the stage-D diagnostics on a tiny protocol."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from cfc_online_adaptation import budget_pilot as bp
from cfc_online_adaptation import revision_online as ro
from cfc_online_adaptation import revision_rerun as rr

torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------
# design
# ----------------------------------------------------------------------------------------
def test_design_constants_and_job_counts():
    assert rr.WIRING_SEEDS[100] == 22300 and rr.WIRING_SEEDS[119] == 22319
    assert set(rr.WIRING_SEEDS.values()).isdisjoint({rr.OLD_NCP_WIRING_SEED})
    assert len(rr.ETA_MULTIPLIERS) == 9 and rr.ETA_MULTIPLIERS[0] == 0.03 and rr.ETA_MULTIPLIERS[-1] == 300.0
    index = {fam: {str(s): {"feature_energy_train": float(1 + i + 0.1 * (s - 100))}
                   for s in rr.ALL_SEEDS} for i, fam in enumerate(rr.FAMILIES)}
    A3 = {"eta_grid": rr.eta_grid_from_index(index)}
    assert A3["eta_grid"]["dense"]["center"] == pytest.approx(1e-3)
    assert A3["eta_grid"]["bias"]["grid"][3] == pytest.approx(1e-3 * A3["eta_grid"]["dense"]["selection_feature_energy"])
    selection = {m: {"selected_eta": 0.5} for m in rr.SELECTED_METHODS}
    counts = rr.job_counts(A3, selection)
    assert counts["select"] == 162 and counts["evaluate_adaptive"] == 210 and counts["evaluate_B0"] == 150
    assert counts["evaluate_PD_only"] == 3 and counts["diagnostics_rerun"] == 70
    assert counts["sensitivity_adaptive_new"] == 324 and counts["sensitivity_B0"] == 45
    assert counts["total"] == 964 == rr.EXPECTED_JOB_COUNTS["total"]
    # PD x1 sensitivity reuses the selection keys exactly
    keys_sel = {rr.job_key(j) for j in rr.jobs_select(A3)}
    assert {rr.job_key(j) for j in rr.jobs_sensitivity_pd1_reused(A3)} == keys_sel
    assert keys_sel.isdisjoint({rr.job_key(j) for j in rr.jobs_sensitivity(A3)})
    # a method without a valid eta is dropped from evaluation and diagnostics, not rescued
    selection["RBF-LMS"] = {"selected_eta": None}
    assert rr.job_counts(A3, selection)["evaluate_adaptive"] == 180
    assert len(rr.diagnostic_conditions(selection)) == 60


def test_job_key_distinguishes_protocol_length_and_family_override():
    short = rr.make_job("dense-CfC-LMS", 100, 13, eta=0.001, stage="smoke", b_cycles=2, washout_cycles=1)
    full = rr.make_job("dense-CfC-LMS", 100, 13, eta=0.001, stage="select")
    sens = rr.make_job("dense-CfC-LMS", 100, 13, eta=0.001, stage="sensitivity")
    assert rr.job_key(short) != rr.job_key(full) and rr.job_key(full) == rr.job_key(sens)
    pd = rr.make_job("PD-only", 100, 13, stage="smoke")
    pd_esn = rr.make_job("PD-only", 100, 13, stage="smoke", family_override="esn")
    assert rr.job_key(pd) != rr.job_key(pd_esn) and rr.job_family(pd_esn) == "esn"


# ----------------------------------------------------------------------------------------
# selection rules with explicit seeds
# ----------------------------------------------------------------------------------------
def _curves(b_values, a=0.02, finite=True):
    cyc = [{"phase": "A", "cycle": 0, "mean_max_perp_dev": a, "all_finite": True}]
    for i, v in enumerate(b_values):
        cyc.append({"phase": "B", "cycle": i, "mean_max_perp_dev": v, "all_finite": finite})
    return cyc


def test_select_eta_for_seeds_matches_frozen_rule_and_accepts_new_seeds():
    grid = [0.1, 0.2, 0.3]
    def runs(seeds):
        out = {}
        for eta in grid:
            out[eta] = {}
            for s in seeds:
                if eta == 0.3:
                    b = [0.05] + [0.2] * 29  # transient above 2x
                elif eta == 0.2:
                    b = list(np.linspace(0.05, 0.01, 30))
                else:
                    b = list(np.linspace(0.05, 0.02, 30))
                out[eta][s] = _curves(b)
        return out
    old = ro.select_eta(grid, runs((0, 1, 2)))
    new = rr.select_eta_for_seeds(grid, runs((0, 1, 2)), (0, 1, 2))
    assert {k: v for k, v in new.items() if k != "seeds"} == old
    new100 = rr.select_eta_for_seeds(grid, runs((100, 101, 102)), (100, 101, 102))
    assert new100["selected_eta"] == 0.2 and new100["codes"] == [] and new100["seeds"] == [100, 101, 102]
    assert set(new100["rows"][0]["per_seed"]) == {"100", "101", "102"}
    # the frozen rule cannot see the new seeds (this is why the explicit-seed version exists)
    assert ro.select_eta(grid, runs((100, 101, 102)))["codes"] == ["no_valid_eta"]
    # edge selection is flagged, missing runs invalidate
    r = rr.select_eta_for_seeds([0.2, 0.3], runs((100, 101, 102)), (100, 101, 102))
    assert r["codes"] == ["grid_edge"]
    partial = runs((100, 101, 102))
    del partial[0.2][101]
    assert rr.select_eta_for_seeds(grid, partial, (100, 101, 102))["selected_eta"] == 0.1


def test_choose_family_for_seeds_uses_given_seeds(data_splits):
    best, cands = rr.choose_family_for_seeds("rbf", data_splits, (100, 101), settings=(1.0, 2.0), lam_grid=(1e-3, 1e-1))
    assert best["selection_seeds"] == [100, 101] and best["n_candidates"] == 2
    assert all({r["seed"] for r in c["selection_rows"]} == {100, 101} for c in cands)
    assert best["at_grid_edge"] is True  # two candidates: every choice is an edge; recorded, not repaired
    assert best["setting"] in (1.0, 2.0)


@pytest.fixture(scope="module")
def data_splits():
    return {s: rr.split_data(s)[:2] for s in (100, 101)}


# ----------------------------------------------------------------------------------------
# loader
# ----------------------------------------------------------------------------------------
def test_cfc_row_and_loader_detect_tampering(tmp_path, data_splits):
    tr, va = data_splits[100]
    audit = rr.split_data(100)[2]
    model = bp.build_model("ncp_wired", 100, rr.WIRING_SEEDS[100])
    res = bp.train_series(model, tr, va, audit, epochs=2, seed=100, out_dir=tmp_path / "s", ckpt_every=2,
                          config={"c": 1})
    assert res["saved_epochs"] == [2]
    ck = tmp_path / "s" / "epoch0002.pt"
    with pytest.raises(ValueError):  # not the final epoch -> loader refuses
        A3 = {"checkpoint_index": {"ncp": {"100": {"checkpoint": str(ck), "sha256": rr.sha256_file(ck),
                                                    "config": "ncp_wired", "seed": 100, "wiring_seed": 22300,
                                                    "verification": {}}}}}
        rr.load_a3_model(A3, "ncp", 100)
    row = rr.cfc_row_from_checkpoint("ncp", 100, ck, origin="test")
    assert row["reload_check"]["forward_bit_identical_to_saved_model"] and not row["epoch_is_final"]
    assert row["reload_ok"] is False and row["audit"]["structure"]["layer_sizes"] == [38, 24, 2]
    # sha mismatch and wiring-seed mismatch are rejected before any forward
    bad = {"checkpoint_index": {"ncp": {"100": dict(row, sha256="0" * 64)}}}
    with pytest.raises(ValueError, match="sha256"):
        rr.load_a3_model(bad, "ncp", 100)
    bad = {"checkpoint_index": {"ncp": {"100": dict(row, wiring_seed=22222)}}}
    with pytest.raises(ValueError, match="wiring seed"):
        rr.load_a3_model(bad, "ncp", 100)
    with pytest.raises(KeyError):
        rr.index_row({"checkpoint_index": {"ncp": {}}}, "ncp", 110)


def test_ridge_row_roundtrip_and_forward_verification(tmp_path, data_splits):
    from cfc_online_adaptation.bases import fit_and_validate
    tr, va = data_splits[101]
    audit = rr.split_data(101)[2]
    model = rr.build_ridge_model("esn", (0.5, 0.1, 1.0), tr.x, 101)
    fit = fit_and_validate(model, tr.x, tr.tau, va.x, va.tau, (1e-3, 1e-1))
    row = rr.ridge_row(model, "esn", tr, va, audit, tmp_path / "esn_seed101.pt", fit=fit, setting=(0.5, 0.1, 1.0))
    A3 = {"checkpoint_index": {"esn": {"101": row}}}
    m = rr.load_a3_model(A3, "esn", 101, verify=True)
    assert np.array_equal(m.export_readout_weights(), model.export_readout_weights())
    # a different readout changes the forward digest -> verification fails
    model.load_readout_weights(model.export_readout_weights() * 1.01)
    torch.save(model.state_dict(), tmp_path / "esn_seed101.pt")
    row2 = dict(row, sha256=rr.sha256_file(tmp_path / "esn_seed101.pt"))
    with pytest.raises(ValueError, match="forward differs"):
        rr.load_a3_model({"checkpoint_index": {"esn": {"101": row2}}}, "esn", 101, verify=True)


# ----------------------------------------------------------------------------------------
# curve comparison and bootstrap
# ----------------------------------------------------------------------------------------
def _cycle(phase, i, v, finite=True):
    return {"phase": phase, "cycle": i, "mean_max_perp_dev": v, "mean_signed_perp_dev": v, "mean_endpoint_error": v,
            "taufb_rms": 1.0, "j_norm": 1.0, "dj_norm": 0.0, "signed_perp_dev_per_dir": [v] * 8, "all_finite": finite}


def test_compare_curves_distinguishes_nonfinite_reproduction_from_mismatch():
    a = [_cycle("A", 0, 0.01), _cycle("B", 0, 0.02), _cycle("C_catch", 0, -0.01), _cycle("C_washout", 0, 0.0)]
    assert rr.compare_curves(a, copy.deepcopy(a))["status"] == "bit_identical"
    b = copy.deepcopy(a)
    b[1]["signed_perp_dev_per_dir"][3] += 1e-17
    c = rr.compare_curves(a, b)
    assert c["status"] == "mismatch" and c["first_mismatch"] == ("B", 0)
    nan_a = copy.deepcopy(a)
    nan_a[1] = _cycle("B", 0, math.nan, finite=False)
    assert rr.compare_curves(nan_a, copy.deepcopy(nan_a))["status"] == "nonfinite_reproduced"
    inf_b = copy.deepcopy(nan_a)
    inf_b[1]["mean_max_perp_dev"] = math.inf
    assert rr.compare_curves(nan_a, inf_b)["status"] == "mismatch"
    flag = copy.deepcopy(a)
    flag[2]["all_finite"] = False
    assert rr.compare_curves(a, flag)["status"] == "mismatch"
    assert rr.compare_curves(a, a[:3])["status"] == "bit_identical"  # washout ignored by default phases
    assert rr.compare_curves(a, a[:3], phases=("A", "B", "C_catch", "C_washout"))["status"] == "mismatch"
    assert rr.compare_curves(a, a[:2])["status"] == "mismatch"  # cycle structure
    # JSON round trip (null <-> NaN) keeps the reproduced non-finite status
    rt = rr.restore_curves(json.loads(json.dumps(ro.json_safe({"curves": nan_a}))))["curves"]
    assert rr.compare_curves(nan_a, rt)["status"] == "nonfinite_reproduced"


def test_paired_log_ratio_bootstrap_reports_exclusions_and_is_reproducible():
    num = {s: 2.0 for s in range(110, 120)}
    den = {s: 1.0 for s in range(110, 120)}
    num[115], den[116] = math.nan, 0.0
    r1 = rr.paired_log_ratio_bootstrap(num, den, n_boot=200, seed=1)
    r2 = rr.paired_log_ratio_bootstrap(num, den, n_boot=200, seed=1)
    assert r1["n_pairs_used"] == 8 and r1["n_excluded"] == 2 and {e["seed"] for e in r1["excluded"]} == {115, 116}
    assert r1["mean_log_ratio"] == pytest.approx(math.log(2)) and r1["ci95"] == pytest.approx([math.log(2)] * 2)
    assert r1["n_den_smaller"] == 8 and r1["n_num_smaller"] == 0
    assert json.dumps(ro.json_safe(r1)) == json.dumps(ro.json_safe(r2))
    empty = rr.paired_log_ratio_bootstrap({1: math.nan}, {1: 1.0})
    assert empty["n_pairs_used"] == 0 and math.isnan(empty["mean_log_ratio"])


# ----------------------------------------------------------------------------------------
# protocol job and stage-D diagnostics on a tiny protocol (untrained model, B 1 cycle)
# ----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def task_and_model():
    arm, cq, reaches = rr.make_task()
    model = bp.build_model("dense_backbone", 100, None)
    model.freeze_core()
    model.eval()
    return arm, reaches, model


@pytest.mark.parametrize("method", ["dense-CfC-LMS", "dense-CfC-bias", "dense-CfC-RLS"])
def test_diagnose_condition_checks_and_files(tmp_path, task_and_model, method):
    arm, reaches, model = task_and_model
    eta = None if method == "dense-CfC-RLS" else 1e-3
    job = rr.make_job(method, 100, 13.0, 1.0, eta, stage="test", b_cycles=1, washout_cycles=1)
    ref = rr.run_protocol_job(arm, reaches, model, job)
    assert [c["phase"] for c in ref["curves"]] == ["A", "B", "C_catch", "C_washout"]
    cond = {"method": method, "seed": 100, "b": 13.0, "pd": 1.0, "eta": eta, "b_cycles": 1}
    res = rr.diagnose_condition(arm, reaches, model, cond, ref["curves"], tmp_path / method)
    ok, bad = rr.diagnostics_passed(res["checks"])
    assert ok, bad
    assert res["checks"]["determinism_vs_stage_c"]["status"] == "bit_identical"
    assert res["checks"]["A_cycle_reproduced_by_frozen_pre_b0"]["equal"]
    assert res["checks"]["catch_cycle_reproduced_by_frozen_post_b0"]["equal"]
    assert res["checks"]["post_b13:tau_ff_reproduced_from_inputs"]["within_1e-12"]
    assert len(res["probe_rows"]) == 8 and len(res["reach_rows"]) == 32 and len(res["actor_rows"]) == 16
    files = set(res["files"])
    assert {"readout_pre.npz", "readout_post.npz", "probe_waveforms.npz", "saved_trajectories.npz",
            "frozen_pre_b0.npz", "frozen_post_b0.npz", "frozen_pre_b13.npz", "frozen_post_b13.npz",
            "curves_A1_B_catch1.json"} <= files
    with np.load(tmp_path / method / "frozen_post_b13.npz") as f:
        assert f["X"].shape == (8, 101, 10) and f["X"].dtype == np.float32
        assert f["Z"].shape == (8, 101, 64) and f["Z"].dtype == np.float64
        assert f["q"].shape == (8, 101, 2)
    # a tampered reference is reported as a mismatch (and fails the technical check)
    bad_ref = copy.deepcopy(ref["curves"])
    bad_ref[1]["mean_max_perp_dev"] *= 1.0000001
    res2 = rr.diagnose_condition(arm, reaches, model, cond, bad_ref, tmp_path / (method + "_bad"))
    assert res2["checks"]["determinism_vs_stage_c"]["status"] == "mismatch"
    assert not rr.diagnostics_passed(res2["checks"])[0]


def test_pd_only_is_model_independent(task_and_model):
    arm, reaches, model = task_and_model
    other = bp.build_model("fc_wired", 100, 22300)
    other.freeze_core()
    other.eval()
    job = rr.make_job("PD-only", 100, 13.0, 1.0, None, stage="test", b_cycles=1, washout_cycles=0)
    a = rr.run_protocol_job(arm, reaches, model, job)
    b = rr.run_protocol_job(arm, reaches, other, dict(job, family_override="fc"))
    assert rr.compare_curves(a["curves"], b["curves"])["status"] == "bit_identical"


def test_run_store_roundtrip(tmp_path):
    store = rr.RunStore(tmp_path / "runs.jsonl")
    r = rr.make_job("ESN-LMS", 110, -13.0, 1.0, 0.01, stage="evaluate") | {
        "wall_s": 1.0, "pid": 1, "order": list(range(8)),
        "curves": [_cycle("A", 0, 0.01), _cycle("B", 0, math.nan, finite=False)]}
    store.add(r)
    again = rr.RunStore(tmp_path / "runs.jsonl")
    got = again.get(rr.make_job("ESN-LMS", 110, -13.0, 1.0, 0.01, stage="sensitivity"))
    assert got is not None and math.isnan(got["curves"][1]["mean_max_perp_dev"])
    assert again.missing([r, rr.make_job("ESN-LMS", 111, -13.0, 1.0, 0.01, stage="evaluate")]) == [
        rr.make_job("ESN-LMS", 111, -13.0, 1.0, 0.01, stage="evaluate")]


def test_run_store_recovers_tail_and_rejects_conflicts(tmp_path):
    path = tmp_path / "runs.jsonl"
    r = rr.make_job("ESN-LMS", 110, 13.0, eta=0.01, stage="evaluate") | {
        "wall_s": 1.0, "pid": 1, "curves": [_cycle("A", 0, 0.01)]}
    rr.RunStore(path).add(r)
    original = path.read_bytes()
    path.write_bytes(original + b'{"method":')
    store = rr.RunStore(path)
    assert store.get(r) is not None and store.recovery is not None
    assert Path(store.recovery["backup"]).read_bytes() == original + b'{"method":'
    r2 = dict(r, seed=111)
    store.add(r2)
    assert len(rr.RunStore(path).runs) == 2
    store.add(dict(r, wall_s=9.0, pid=22, stage="sensitivity"))
    assert len(path.read_text().splitlines()) == 2
    bad = copy.deepcopy(r)
    bad["curves"][0]["mean_max_perp_dev"] = 0.9
    with pytest.raises(ValueError, match="conflicting duplicate"):
        store.add(bad)
    path.write_text(json.dumps(ro.json_safe(r)) + '\n{"broken":\n' + json.dumps(ro.json_safe(r2)) + '\n')
    with pytest.raises(ValueError, match="corruption"):
        rr.RunStore(path)
    path.write_text(json.dumps(ro.json_safe(r)) + '\n' + json.dumps(ro.json_safe(bad)) + '\n')
    with pytest.raises(ValueError, match="conflicting duplicate"):
        rr.RunStore(path)


def test_nonfinite_readout_state_roundtrip(tmp_path):
    from cfc_online_adaptation import readout_probe as rp
    a, _ = ro.build_readout("rls", np.zeros((2, 64)), None)
    a.J[0, 0] = np.nan
    a.J[0, 1] = np.inf
    a.P[1, 1] = -np.inf
    rp.save_snapshot(tmp_path / "state.npz", a, "rls")
    b = rp.load_snapshot(tmp_path / "state.npz")
    assert rr.readout_state_equal(a, b, "rls")
    b.P[1, 1] = np.inf
    assert not rr.readout_state_equal(a, b, "rls")
    b = rp.load_snapshot(tmp_path / "state.npz")
    b.J[0, 0], b.J[1, 0] = 0, np.nan
    assert not rr.readout_state_equal(a, b, "rls")
    assert not rr.array_state_equal(np.array([0.]), np.array([-0.]))


def test_nonfinite_adaptation_is_missing_diagnostic_not_technical_failure(task_and_model, tmp_path, monkeypatch):
    from cfc_online_adaptation import protocol
    arm, reaches, model = task_and_model
    curves = [_cycle("A", 0, .01), _cycle("B", 0, math.nan, finite=False),
              _cycle("C_catch", 0, math.nan, finite=False)]

    def fail_scientifically(arm, model, readout, *args, **kwargs):
        readout.J[:] = np.nan
        return {"curves": curves, "final_J": readout.J.copy(), "saved_trajectories": {}}

    monkeypatch.setattr(protocol, "run_abc_protocol", fail_scientifically)
    cond = {"method": "dense-CfC-LMS", "seed": 100, "b": 13., "pd": 1., "eta": .001, "b_cycles": 1}
    res = rr.diagnose_condition(arm, reaches, model, cond, curves, tmp_path)
    assert rr.diagnostics_passed(res["checks"])[0]
    assert res["checks"]["determinism_vs_stage_c"]["status"] == "nonfinite_reproduced"
    assert res["summary"]["diagnostic_status"] == "unavailable"
    assert res["summary"]["missing_reason"] and not res["probe_rows"]
    assert not list(tmp_path.glob("frozen_*.npz"))


@pytest.fixture(scope="module")
def rerun_driver():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts/run_revision_rerun.py"
    spec = importlib.util.spec_from_file_location("rerun_review_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostics_stage_rejects_other_technical_failure(tmp_path, rerun_driver):
    def result(checks):
        return {"cond": {"method": "dense-CfC-LMS", "seed": 110}, "checks": checks,
                "bytes": 0, "summary": {"all_phases_finite": False}, "probe_rows": [],
                "reach_rows": [], "cycle_rows": [], "actor_rows": []}
    bad = {"determinism_vs_stage_c": {"status": "bit_identical"}, "snapshot_restore_equal": False}
    with pytest.raises(SystemExit, match="technical checks failed"):
        rerun_driver.write_diagnostics(tmp_path, [result(bad)], {})
    assert json.loads((tmp_path / "status.json").read_text())["technical_checks_passed"] is False
    good = {"determinism_vs_stage_c": {"status": "nonfinite_reproduced"},
            "post_state_finite": False, "snapshot_restore_equal": True}
    status = rerun_driver.write_diagnostics(tmp_path, [result(good)], {})
    assert status["technical_checks_passed"] and status["n_adaptive_nonfinite"] == 1


def test_diagnostics_existing_outputs_require_explicit_archive(tmp_path, rerun_driver, monkeypatch):
    d = rerun_driver
    monkeypatch.setattr(d, "load_a3_b3", lambda *a, **k: ({}, "a", {}, "b"))
    monkeypatch.setattr(d, "run_identity", lambda *a: {})
    monkeypatch.setattr(d, "load_selection", lambda *a: {})
    monkeypatch.setattr(d.rr, "diagnostic_conditions", lambda *a: [])
    monkeypatch.setattr(d, "execute_diag", lambda *a: [])
    monkeypatch.setattr(d, "append_stage_record", lambda *a: None)
    old = tmp_path / "diagnostics"
    old.mkdir()
    (old / "partial.npz").write_bytes(b"unfinished data")
    with pytest.raises(SystemExit, match="archive-diagnostics"):
        d.stage_diagnostics(tmp_path, 1)
    d.stage_diagnostics(tmp_path, 1, archive_existing=True)
    archives = list(tmp_path.glob("diagnostics_archived_*"))
    assert len(archives) == 1 and (archives[0] / "partial.npz").read_bytes() == b"unfinished data"


def test_report_does_not_mark_failed_diagnostics_success(tmp_path, rerun_driver, monkeypatch):
    d = rerun_driver
    monkeypatch.setattr(d, "load_a3_b3", lambda *a, **k: ({}, "a", {"expected_job_counts": {}}, "b"))
    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "diagnostics/status.json").write_text(json.dumps({"technical_checks_passed": False}))
    with pytest.raises(SystemExit, match="not a successful run"):
        d.stage_report(tmp_path)
    assert json.loads((tmp_path / "report.json").read_text())["technical_checks_passed"] is False
