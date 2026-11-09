"""Summarize paired direction errors from saved six-method experiment results."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


METHODS = (
    "dense-CfC-LMS", "NCP-CfC-LMS", "FC-CfC-LMS",
    "ESN-LMS", "RBF-LMS", "dense-CfC-RLS",
)


def summarize(package_root, out):
    source = package_root / "results"
    inputs = [source / "runs.jsonl", source / "diagnostics/frozen_per_reach.csv"]
    adaptive = {}
    for line in inputs[0].read_text().splitlines():
        run = json.loads(line)
        if run["stage"] != "evaluate" or run["b"] != 13 or run["method"] not in METHODS:
            continue
        last = [cycle for cycle in run["curves"] if cycle["phase"] == "B"][-1]
        for direction, value in zip(run["order"], last["signed_perp_dev_per_dir"], strict=True):
            adaptive[run["method"], run["seed"], direction] = abs(value) * 1000
    fixed = {}
    with inputs[1].open() as stream:
        for row in csv.DictReader(stream):
            if row["condition"] == "post_b13":
                fixed[row["method"], int(row["seed"]), int(row["direction_index"])] = float(row["max_abs_perp_mm"])
    expected = {(m, s, d) for m in METHODS for s in range(110, 120) for d in range(8)}
    if set(adaptive) != expected or set(fixed) != expected:
        raise ValueError("Expected six methods, ten seeds, and eight directions")
    rows = []
    for method in METHODS:
        for seed in range(110, 120):
            for direction in range(8):
                key = method, seed, direction
                if adaptive[key] <= 0:
                    raise ValueError("Non-positive adaptive error")
                rows.append({"method": method, "seed": seed, "direction_index": direction,
                             "adaptive_final_error_mm": adaptive[key],
                             "fixed_post_error_mm": fixed[key],
                             "fixed_post_over_adaptive": fixed[key] / adaptive[key]})
    summaries = []
    for method in METHODS:
        for direction in range(8):
            group = [r for r in rows if r["method"] == method and r["direction_index"] == direction]
            summaries.append({"method": method, "direction_index": direction, "n_seeds": len(group),
                              "median_paired_ratio": median(r["fixed_post_over_adaptive"] for r in group),
                              "n_paired_ratios_below_one": sum(r["fixed_post_over_adaptive"] < 1 for r in group)})
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("fixed_direction_pairs.csv", rows), ("fixed_direction_summary.csv", summaries)]:
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    provenance = {
        "analysis": "Post-hoc direction-level reaggregation; no new experiments",
        "source_sha256": {str(p.relative_to(package_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "metric": "maximum absolute perpendicular deviation in millimeters",
        "summary": "median across ten within-seed fixed-post/adaptive-final error ratios, separately by method and direction",
        "last_trained_direction_index": 7,
        "pairs": len(rows), "summary_rows": len(summaries),
    }
    (out / "fixed_direction_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.package_root, args.out)
