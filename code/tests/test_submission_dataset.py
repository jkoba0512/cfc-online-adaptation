"""Regression checks for the submission's six-method scheduling boundary."""

from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/reproduce_rerun400_package.py"
if not SCRIPT.exists():
    SCRIPT = ROOT / "code/scripts/reproduce_rerun400_package.py"
spec = importlib.util.spec_from_file_location("submission_reproducer", SCRIPT)
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)


def test_complete_protocol_inventory_needs_only_five_tuned_methods():
    a = {
        "eta_grid": {
            family: {"grid": [0.001 * (i + 1) for i in range(9)]}
            for family in submission.rr.FAMILIES
        }
    }
    selection = {m: {"selected_eta": 0.001} for m in submission.TUNED_METHODS}
    select = submission.jobs_select(a)
    evaluate = submission.jobs_evaluate(selection)
    sensitivity = submission.jobs_sensitivity(a)
    assert (len(select), len(evaluate), len(sensitivity)) == (135, 333, 315)
    jobs = select + evaluate + sensitivity
    assert len({submission.rr.job_key(j) for j in jobs}) == 783
    assert {j["method"] for j in jobs} == set(
        submission.ADAPTIVE_METHODS + submission.REFERENCE_METHODS
    )
    assert len(submission.ADAPTIVE_METHODS) == 6 and len(submission.TUNED_METHODS) == 5


def test_diagnostics_cover_each_supplied_readout_and_seed_once():
    selection = {m: {"selected_eta": 0.001} for m in submission.TUNED_METHODS}
    conditions = submission.diagnostic_conditions(selection)
    assert len(conditions) == 60
    pairs = {(c["method"], c["seed"]) for c in conditions}
    assert pairs == {
        (m, seed) for m in submission.ADAPTIVE_METHODS for seed in submission.rr.EVALUATION_SEEDS
    }
    assert all(c["b"] == 13 and c["pd"] == 1 and c["b_cycles"] == 30 for c in conditions)
