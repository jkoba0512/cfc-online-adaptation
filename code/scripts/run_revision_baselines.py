"""Write the frozen online-comparison design (manifest B); no trials are run here."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from cfc_online_adaptation.revision_manifest import validate_ab_curves

SELECTION_SEEDS = [0, 1, 2]
EVALUATION_SEEDS = [3, 4, 5, 6, 7]
METHODS = [
    "B0-CfC", "B0-NCP", "B0-ESN", "B0-RBF", "B2", "CfC-LMS",
    "CfC-RLS-reg", "NCP-LMS", "ESN-LMS", "RBF-LMS",
    "CfC-direction-specific-bias",
]


def build_manifest(a: dict, a_path: Path) -> dict:
    if a["selection_seeds"] != SELECTION_SEEDS or a["evaluation_seeds"] != EVALUATION_SEEDS:
        raise ValueError("manifest A seed split does not match frozen design")
    required = {"cfc", "ncp", "esn", "rbf", "bias"}
    if set(a["eta_grid"]) != required:
        raise ValueError("manifest A eta grids incomplete")
    for family in required:
        if len(a["eta_grid"][family]["grid"]) != 7:
            raise ValueError(f"{family}: eta grid must have seven points")
    return {
        "schema": "revision-baselines-B-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "manifest_A": str(a_path.resolve()),
        "manifest_A_sha256": hashlib.sha256(a_path.read_bytes()).hexdigest(),
        "selection_seeds": SELECTION_SEEDS,
        "evaluation_seeds": EVALUATION_SEEDS,
        "methods": METHODS,
        "shared_task": {
            "plant": "TwoLinkArm(ArmParams defaults)",
            "center_xy_m": [0.0, 0.45], "radius_m": 0.10, "directions": 8,
            "minimum_jerk_duration_s": 0.5, "dt_s": 0.005,
            "Kp_diagonal": [35.0, 25.0], "Kd_diagonal": [3.5, 2.5],
            "direction_order": "fixed, direction_seed=None",
            "input": "[q_des,qdot_des,qddot_des,q,qdot]",
            "error_signal": "FEL tau_fb, update before plant step",
            "protocol": {"A_null_cycles": 1, "B_curl_cycles": 30,
                         "C_catch_null_cycles": 1, "C_washout_null_cycles": 5,
                         "catch_adapt": False, "washout_adapt": True},
        },
        "selection": {
            "validity_function": f"{validate_ab_curves.__module__}.{validate_ab_curves.__name__}",
            "field_b": 13.0,
            "eta_grids_from_manifest_A": a["eta_grid"],
            "selected_methods": ["CfC-LMS", "NCP-LMS", "ESN-LMS", "RBF-LMS",
                                 "CfC-direction-specific-bias"],
            "required_A_cycle_indices": [0],
            "required_B_cycle_indices": list(range(30)),
            "finite_rule": "all 31 A/B cycle all_finite flags true and metrics finite",
            "catch_and_washout_excluded": True,
            "other_validity": ["B_last < B_first", "each B_cycle <= 2 * B_first"],
            "all_three_selection_seeds_required": True,
            "objective": "smallest mean B_last across selection seeds; tie -> smaller eta",
            "invalid_grid_points": "report all with reason codes",
            "no_valid_eta": "exclude method from evaluation and report grid",
            "edge_eta": "flag; never expand grid",
        },
        "evaluation": {
            "field_b_main": 13.0, "field_b_auxiliary": [-13.0, 25.0],
            "seeds": EVALUATION_SEEDS, "cycles": 30,
            "same_selected_eta_across_fields": True,
            "whole_protocol_finite_reported": True,
            "per_seed_points_and_failures": True,
            "summary": "median; no superiority significance test",
        },
        "B0_reference": {
            "primary": "same seed, field sign/strength, and family; B final cycle per direction",
            "families": ["CfC", "NCP", "ESN", "RBF"],
            "bias_family": "CfC",
            "CfC_B0_also_reported": True,
        },
        "fixed_parameters": {"RLS_beta": 1.0, "RLS_p0": 0.1,
                             "LMS_leakage": 0.0, "direction_bias_contexts": 8},
        "metrics": {
            "B_first_B_last": "cycle 0/29 mean_max_perp_dev in mm",
            "improvement": "1 - B_last/B0_family_last",
            "adaptation_cycle": "first B_c <= B_first - 0.5*(B_first-B_last)",
            "catch_signed_mm": "C_catch mean_signed_perp_dev",
            "ratio": "abs(mean_d catch_d)/abs(mean_d B0_final_d)",
            "mirror": "sign(mean catch) == -sign(mean B0_final)",
            "n_mirror": "number of matching opposite signs over 8 directions",
            "after_effect_threshold_main": 0.25,
            "after_effect_threshold_sweep": [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            "washout_decay": "abs(washout5 signed)/abs(catch signed); missing if abs(catch)<1mm",
            "finite": "all phases; selection uses only A/B",
            "j_norm": "maximum of cycle-end last-reach final-step values",
            "dj_norm": "maximum of cycle-end last-reach final-step values",
            "taufb_rms": "maximum over cycle RMS values",
            "initial_quality": ["offline validation NMSE", "A tracking error", "B_first"],
        },
        "missing_codes": ["diverged", "not_reached", "catch_small", "no_valid_eta"],
        "flags": ["grid_edge"],
        "sensitivity": {
            "methods": ["CfC-LMS", "ESN-LMS", "RBF-LMS", "CfC-direction-specific-bias"],
            "eta_points": 7, "PD_multipliers": [0.5, 1.0, 2.0],
            "seeds": SELECTION_SEEDS, "field_b": 13.0,
            "outcomes": ["B_last", "after_effect_at_threshold_0.25"],
        },
        "analysis_scope": "planar torque-controlled arm; b in {13,-13,25}; no online recurrent baseline",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true", required=True)
    parser.add_argument("--manifest-a", type=Path, required=True)
    parser.add_argument("--outputs", type=Path)
    args = parser.parse_args()
    a = json.loads(args.manifest_a.read_text())
    out = args.outputs or Path("../outputs") / f"revision_baselines_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    path = out / "manifest_B.json"
    path.write_text(json.dumps(build_manifest(a, args.manifest_a), indent=2, allow_nan=False) + "\n")
    print(path.resolve())


if __name__ == "__main__":
    main()
