"""Phase 2 Step 2: Exp 2A formal completion (WiredCfC vs dense Phase 1).

Closes Exp 2A by running the WiredCfC (NCP wiring, state feature) through the
same comparison structure as Phase 1 Step 4/5, and tabling it next to the dense
Phase 1 final result so "the central claim reproduces on a wired CfC" is shown
in one place.

WiredCfC-state conditions (B=30 cycles, A/B/C + washout):
  B0, B2 feedback-only, LMS-main (eta=1e-3,leak=0),
  RLS-reg (beta=1.0,p0=0.1), RLS-forgetting (beta=0.999) negative check.
Plus a small feature_mode="output" ablation (d_z=2, narrow motor basis).
Dense rows are reused from the Phase 1 final summary JSON.

Run:
  export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/cfc-online-adaptation
  cd code
  uv run python scripts/run_phase2_step2_wiredcfc_formal_2a.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cfc_online_adaptation.analysis import (
    after_effect_metrics,
    claim_supported,
    curl_reference_signed,
    get_phase_cycles,
)
from cfc_online_adaptation.arm import ArmParams, TwoLinkArm
from cfc_online_adaptation.baselines import NullReadout
from cfc_online_adaptation.core import WiredCfCInverseModel
from cfc_online_adaptation.dataset import build_reach_sequences, train_val_split
from cfc_online_adaptation.experiment import train_inverse_model
from cfc_online_adaptation.plots import (
    plot_after_effect_bars,
    plot_j_norm,
    plot_learning_curves,
)
from cfc_online_adaptation.protocol import run_abc_protocol
from cfc_online_adaptation.readout import LinearReadout, RLSReadout
from cfc_online_adaptation.tasks import joint_space_reaches

CENTER_XY = (0.0, 0.45)
DT, T_MOVE, FIELD_B = 0.005, 0.5, 13.0
KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])
DENSE_SUMMARY = "../outputs/phase1_final_20260610_131754/phase1_summary.json"


def json_safe(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    return o


def b_curve_mm(out):
    return [c["mean_max_perp_dev"] * 1e3 for c in get_phase_cycles(out, "B")]


def b_last_mm(out):
    return float(get_phase_cycles(out, "B")[-1]["mean_max_perp_dev"] * 1e3)


def max_jnorm(out):
    return max(c["j_norm"] for c in out["curves"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--b-cycles", type=int, default=30)
    ap.add_argument("--washout-cycles", type=int, default=5)
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outputs", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(args.outputs) if args.outputs
               else Path("../outputs") / f"phase2_step2_wiredcfc_formal2a_{stamp}").resolve()
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    device = "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    arm = TwoLinkArm(ArmParams())
    center_q = arm.inverse_kinematics(np.array(CENTER_XY), elbow="down")
    ds = build_reach_sequences(
        arm, center_q, radii=(0.08, 0.10, 0.12), durations=(T_MOVE,), dt=DT, n_targets=8
    )
    train_ds, val_ds = train_val_split(ds, val_frac=0.25, seed=args.seed)
    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=T_MOVE, dt=DT, n_targets=8)

    def train_wired(feature_mode):
        torch.manual_seed(args.seed)
        m = WiredCfCInverseModel(10, 64, 2, feature_mode=feature_mode,
                                 sparsity_level=args.sparsity, seed=22222)
        h = train_inverse_model(m, train_ds, val_ds, epochs=args.epochs, lr=1e-3,
                                device=device, seed=args.seed)
        m.freeze_core()
        return m, h, m.export_readout_weights()

    # ---- wired-state model ----
    model, hist, J0 = train_wired("state")
    j0_norm = float(np.linalg.norm(J0))
    wp = model.wiring_params()
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"wired-state: {wp} | val MSE {hist['val_mse'][0]:.4f}->{hist['val_mse'][-1]:.4f} "
          f"||J0||={j0_norm:.4f}")

    def protocol(mdl, readout, J0_ref, field_b, b_cycles, washout):
        with np.errstate(over="ignore", invalid="ignore"):
            return run_abc_protocol(
                arm, mdl, readout, reaches, KP, KD, J0_ref, dt=DT, field_b=field_b,
                b_cycles=b_cycles, washout_cycles=washout, washout_adapt=True, device="cpu",
            )

    bc, wc = args.b_cycles, args.washout_cycles
    print("running wired-state conditions ...")
    conds = {
        "B0": protocol(model, LinearReadout.from_weights(J0, eta=0.0), J0, FIELD_B, bc, wc),
        "B2": protocol(model, NullReadout(2, 64), np.zeros((2, 64)), FIELD_B, bc, wc),
        "LMS-main": protocol(model, LinearReadout.from_weights(J0, eta=1e-3), J0, FIELD_B, bc, wc),
        "RLS-reg": protocol(model, RLSReadout.from_weights(J0, beta=1.0, p0=0.1), J0, FIELD_B, bc, wc),
        "RLS-forgetting": protocol(model, RLSReadout.from_weights(J0, beta=0.999, p0=1.0), J0,
                                   FIELD_B, bc, wc),
    }
    curl_ref = curl_reference_signed(conds["B0"])

    # ---- wired-output ablation (small) ----
    model_o, hist_o, J0_o = train_wired("output")
    print(f"wired-output ablation: d_z={model_o.d_z} val MSE "
          f"{hist_o['val_mse'][0]:.4f}->{hist_o['val_mse'][-1]:.4f}")
    b0_o = protocol(model_o, LinearReadout.from_weights(J0_o, eta=0.0), J0_o, FIELD_B, 1, 0)
    lms_o = protocol(model_o, LinearReadout.from_weights(J0_o, eta=1e-3), J0_o, FIELD_B, 15, 1)
    curl_ref_o = curl_reference_signed(b0_o)

    # ---- dense rows from Phase 1 final ----
    dense_rows = {}
    dp = Path(DENSE_SUMMARY)
    if dp.exists():
        dense = json.loads(dp.read_text())
        for r in dense["table"]:
            if r["condition"] in ("B0", "LMS-main", "RLS-reg"):
                dense_rows[r["condition"]] = r

    # ---- build unified summary table ----
    def row(condition, model_tag, out, b0_curl_mm, ref):
        ae = after_effect_metrics(ref, np.asarray(out["after_effect"]["signed_perp_dev_per_dir"]))
        finite = all(c["all_finite"] for c in out["curves"])
        return {
            "condition": condition, "model": model_tag,
            "B_last_mm": round(b_last_mm(out), 3),
            "improvement_vs_B0_percent": round(100.0 * (1.0 - b_last_mm(out) / b0_curl_mm), 1),
            "catch_signed_mm": round(ae["catch_mean_signed_mm"], 2),
            "after_effect_ratio": round(ae["after_effect_ratio"], 3),
            "mirror_dirs": ae["n_mirror_dirs"], "max_J_norm": round(max_jnorm(out), 3),
            "finite": finite,
        }

    b0_state_mm = b_last_mm(conds["B0"])
    rows = []
    # dense rows (reused)
    for c in ("B0", "LMS-main", "RLS-reg"):
        if c in dense_rows:
            d = dense_rows[c]
            rows.append({
                "condition": c, "model": "dense",
                "B_last_mm": d["B_last_mm"],
                "improvement_vs_B0_percent": d["improvement_vs_B0_percent"],
                "catch_signed_mm": d["C_catch_signed_mm"],
                "after_effect_ratio": d["after_effect_ratio"],
                "mirror_dirs": d["n_mirror_dirs"], "max_J_norm": d["max_J_norm"],
                "finite": d["finite"],
            })
    # wired-state rows
    for c in ("B0", "B2", "LMS-main", "RLS-reg", "RLS-forgetting"):
        rows.append(row(c, "wired-state", conds[c], b0_state_mm, curl_ref))
    # wired-output ablation
    rows.append(row("LMS-output-ablation", "wired-output", lms_o, b_last_mm(b0_o), curl_ref_o))

    claim = claim_supported(
        after_effect_metrics(curl_ref, np.asarray(conds["LMS-main"]["after_effect"]["signed_perp_dev_per_dir"])),
        after_effect_metrics(curl_ref, np.asarray(conds["B2"]["after_effect"]["signed_perp_dev_per_dir"])),
    )

    # ---- dump ----
    config = {
        "stamp": stamp, "seed": args.seed, "device": device, "deterministic": "cpu-seeded",
        "wiring_params": wp, "sparsity": args.sparsity, "field_b": FIELD_B,
        "b_cycles": bc, "washout_cycles": wc, "epochs": args.epochs,
        "Kp": KP.diagonal().tolist(), "Kd": KD.diagonal().tolist(),
        "dense_source": DENSE_SUMMARY,
    }
    metrics = {
        "wired_state_val_mse_last": hist["val_mse"][-1],
        "wired_output_val_mse_last": hist_o["val_mse"][-1],
        "j0_norm": j0_norm, "b0_curl_state_mm": b0_state_mm,
        "table": rows, "after_effect_claim": claim,
    }
    (out_dir / "config.json").write_text(json.dumps(json_safe(config), indent=2, allow_nan=False))
    (out_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, allow_nan=False))

    cols = ["condition", "model", "B_last_mm", "improvement_vs_B0_percent", "catch_signed_mm",
            "after_effect_ratio", "mirror_dirs", "max_J_norm", "finite"]
    with open(out_dir / "summary_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    head = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
    body = "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)
    md = [
        f"# Phase 2 Step 2: Exp 2A formal completion ({stamp})", "",
        f"- WiredCfC AutoNCP units={wp['units']} sparsity={args.sparsity} "
        f"core params={wp['n_core_params']} (dense 42624); feature=state d_z={wp['d_z']}",
        f"- wired-state val MSE -> {hist['val_mse'][-1]:.4f}, ||J0||={j0_norm:.4f}",
        f"- **after-effect claim supported (wired-state): {claim['claim_supported']}**", "",
        head + body, "",
        "dense rows reused from Phase 1 final; wired RLS-forgetting (beta=0.999) is the "
        "negative-result check (expected to diverge under long exposure).",
    ]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    def curve_arr(out):
        return {k: np.array([c[k] for c in out["curves"]])
                for k in ["mean_max_perp_dev", "mean_signed_perp_dev", "taufb_rms",
                          "j_norm", "dj_norm"]} | {"phase": np.array([c["phase"] for c in out["curves"]])}

    cdump = {}
    for name, out in conds.items():
        cdump.update({f"{name.replace('-', '_')}_{k}": v for k, v in curve_arr(out).items()})
    np.savez_compressed(out_dir / "curves.npz", J0=J0, **cdump)

    # ---- figures ----
    styles = {
        "B2": {"color": "tab:orange"}, "LMS-main": {"color": "tab:blue"},
        "RLS-reg": {"color": "tab:green"},
        "RLS-forgetting": {"color": "tab:red", "ls": "--", "alpha": 0.5},
    }
    lc = {n: b_curve_mm(conds[n]) for n in ["B2", "LMS-main", "RLS-reg"]}
    plot_learning_curves(lc, b0_ref_mm=b0_state_mm, path=out_dir / "figures" / "learning_curve.png",
                         styles=styles)
    aeb = {n: after_effect_metrics(
        curl_ref, np.asarray(conds[n]["after_effect"]["signed_perp_dev_per_dir"])
    )["catch_mean_signed_mm"] for n in ["B0", "B2", "LMS-main", "RLS-reg"]}
    plot_after_effect_bars(aeb, path=out_dir / "figures" / "after_effect_bars.png")
    jn = {n: np.array([c["j_norm"] for c in conds[n]["curves"]])
          for n in ["LMS-main", "RLS-reg", "RLS-forgetting"]}
    plot_j_norm(jn, path=out_dir / "figures" / "j_norm_curve.png", logy=True, styles=styles)

    # ---- console ----
    print("\ncondition          model         Blast  impr%  catch  ratio mirror  maxJ  finite")
    for r in rows:
        print(f"{r['condition']:18s} {r['model']:12s} {r['B_last_mm']:6.2f} "
              f"{r['improvement_vs_B0_percent']:6.1f} {r['catch_signed_mm']:6.1f} "
              f"{r['after_effect_ratio']:5.2f} {r['mirror_dirs']}/8  {r['max_J_norm']:7.1f}  {r['finite']}")
    print(f"\nclaim supported (wired-state): {claim['claim_supported']}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
