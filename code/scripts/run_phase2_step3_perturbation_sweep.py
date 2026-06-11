"""Phase 2 Step 3 (Exp 2B initial): perturbation generalization sweep.

Maps where readout-only LMS adaptation generalizes across perturbations. The
inverse model + J0 are trained ONCE on the baseline null-field arm and frozen;
each perturbation then tests whether the same frozen core + online readout can
re-calibrate.

Main axis: dense CfC + LMS-main (eta=1e-3, leakage=0, FEL), A/B/C with B=30.
Perturbations:
  curl strength : b = 8, 13, 20, 30
  curl reversal : b = -13
  payload       : m2 x {1.25, 1.5}  (heavier forearm during B; field off)
Representative wired-state re-check at b=20.

Each perturbation reports B0, LMS B_last, improvement, after-effect (vs that
perturbation's own B0 reference), and a success flag. Failures are kept
(success=false) as candidate readout-only limits (-> Exp 2E).

Run:
  export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/cfc-online-adaptation
  cd code
  uv run python scripts/run_phase2_step3_perturbation_sweep.py
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
    curl_reference_signed,
    get_phase_cycles,
)
from cfc_online_adaptation.arm import ArmParams, TwoLinkArm
from cfc_online_adaptation.core import CfCInverseModel, WiredCfCInverseModel
from cfc_online_adaptation.dataset import build_reach_sequences, train_val_split
from cfc_online_adaptation.experiment import train_inverse_model
from cfc_online_adaptation.plots import plot_learning_curves
from cfc_online_adaptation.protocol import run_abc_protocol
from cfc_online_adaptation.readout import LinearReadout
from cfc_online_adaptation.tasks import joint_space_reaches

CENTER_XY = (0.0, 0.45)
DT, T_MOVE = 0.005, 0.5
KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])
M2_BASE = 1.1


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
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--wired-epochs", type=int, default=400)
    ap.add_argument("--b-cycles", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outputs", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(args.outputs) if args.outputs
               else Path("../outputs") / f"phase2_step3_perturbation_sweep_{stamp}").resolve()
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

    # dense model trained once on baseline null arm
    model = CfCInverseModel(10, 64, 2)
    hist = train_inverse_model(model, train_ds, val_ds, epochs=args.epochs, lr=1e-3,
                               device=device, seed=args.seed)
    model.freeze_core()
    J0 = model.export_readout_weights()
    j0_norm = float(np.linalg.norm(J0))
    print(f"dense val MSE {hist['val_mse'][0]:.4f}->{hist['val_mse'][-1]:.4f} ||J0||={j0_norm:.4f}")

    def protocol(mdl, readout, J0_ref, field_b, b_cycles, arm_b=None):
        with np.errstate(over="ignore", invalid="ignore"):
            return run_abc_protocol(
                arm, mdl, readout, reaches, KP, KD, J0_ref, dt=DT, field_b=field_b,
                b_cycles=b_cycles, washout_cycles=0, arm_b=arm_b, device="cpu",
            )

    # null reference (no perturbation) for "is the perturbation effective" check
    null_b0 = protocol(model, LinearReadout.from_weights(J0, eta=0.0), J0, 0.0, 1)
    null_mm = b_last_mm(null_b0)
    print(f"null B0 (baseline tracking) = {null_mm:.3f}mm")

    rows = []
    lc_repr = {}  # representative learning curves

    def evaluate(label, kind, model_use, J0_use, field_b, arm_b, b_cycles, keep_curve=False):
        b0 = protocol(model_use, LinearReadout.from_weights(J0_use, eta=0.0), J0_use,
                      field_b, 1, arm_b=arm_b)
        lms = protocol(model_use, LinearReadout.from_weights(J0_use, eta=1e-3), J0_use,
                       field_b, b_cycles, arm_b=arm_b)
        b0_mm = b_last_mm(b0)
        lms_mm = b_last_mm(lms)
        improve = 100.0 * (1.0 - lms_mm / b0_mm) if b0_mm > 0 else 0.0
        ref = curl_reference_signed(b0)
        ae = after_effect_metrics(ref, np.asarray(lms["after_effect"]["signed_perp_dev_per_dir"]))
        finite = all(c["all_finite"] for c in lms["curves"])
        success = bool(
            (b0_mm > max(2.0, 1.5 * null_mm))      # perturbation actually distorts
            and (improve >= 50.0)                   # LMS recovers
            and (ae["after_effect_ratio"] >= 0.25)  # internal-model signature
            and (ae["n_mirror_dirs"] >= 6)
            and finite
        )
        if keep_curve:
            lc_repr[label] = b_curve_mm(lms)
        row = {
            "perturbation": label, "kind": kind, "model": ("wired-state"
                if isinstance(model_use, WiredCfCInverseModel) else "dense"),
            "B0_mm": round(b0_mm, 3), "LMS_B_last_mm": round(lms_mm, 3),
            "improvement_percent": round(improve, 1),
            "catch_signed_mm": round(ae["catch_mean_signed_mm"], 2),
            "after_effect_ratio": round(ae["after_effect_ratio"], 3),
            "mirror_dirs": ae["n_mirror_dirs"], "max_J_norm": round(max_jnorm(lms), 3),
            "finite": finite, "success": success,
        }
        rows.append(row)
        print(f"{label:16s} B0 {b0_mm:6.2f} LMS {lms_mm:6.2f} ({improve:5.1f}%) "
              f"ratio {ae['after_effect_ratio']:.2f} mirror {ae['n_mirror_dirs']}/8 "
              f"maxJ {max_jnorm(lms):.1f} finite={finite} success={success}")
        return row

    # ---- curl strength sweep ----
    print("== curl strength ==")
    for b in (8, 13, 20, 30):
        evaluate(f"curl_b{b}", "curl_strength", model, J0, float(b), None,
                 args.b_cycles, keep_curve=(b in (8, 13, 30)))
    # ---- curl reversal ----
    print("== curl reversal ==")
    evaluate("curl_b-13", "curl_reversal", model, J0, -13.0, None, args.b_cycles)

    # ---- payload (heavier forearm during B) ----
    print("== payload ==")
    for scale in (1.25, 1.5):
        payload_arm = TwoLinkArm(ArmParams(m2=M2_BASE * scale))
        evaluate(f"payload_m2x{scale}", "payload", model, J0, 0.0, payload_arm,
                 args.b_cycles, keep_curve=(scale == 1.5))

    # ---- wired representative at b=20 ----
    print("== wired representative (b=20) ==")
    torch.manual_seed(args.seed)
    wmodel = WiredCfCInverseModel(10, 64, 2, feature_mode="state", sparsity_level=0.5, seed=22222)
    whist = train_inverse_model(wmodel, train_ds, val_ds, epochs=args.wired_epochs, lr=1e-3,
                                device=device, seed=args.seed)
    wmodel.freeze_core()
    wJ0 = wmodel.export_readout_weights()
    evaluate("curl_b20_wired", "wired_check", wmodel, wJ0, 20.0, None, args.b_cycles)

    n_success = sum(r["success"] for r in rows)
    print(f"\nsuccess {n_success}/{len(rows)}")

    # ---- dump ----
    config = {
        "stamp": stamp, "seed": args.seed, "device": device, "deterministic": "cpu-seeded",
        "main_model": "dense", "method": "LMS-main eta=1e-3 leak=0 FEL",
        "b_cycles": args.b_cycles, "epochs": args.epochs, "wired_epochs": args.wired_epochs,
        "Kp": KP.diagonal().tolist(), "Kd": KD.diagonal().tolist(),
        "m2_base": M2_BASE, "null_b0_mm": null_mm, "j0_norm": j0_norm,
    }
    metrics = {
        "dense_val_mse_last": hist["val_mse"][-1],
        "wired_val_mse_last": whist["val_mse"][-1],
        "null_b0_mm": null_mm, "j0_norm": j0_norm,
        "table": rows, "n_success": n_success, "n_conditions": len(rows),
    }
    (out_dir / "config.json").write_text(json.dumps(json_safe(config), indent=2, allow_nan=False))
    (out_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, allow_nan=False))

    cols = ["perturbation", "kind", "model", "B0_mm", "LMS_B_last_mm", "improvement_percent",
            "catch_signed_mm", "after_effect_ratio", "mirror_dirs", "max_J_norm",
            "finite", "success"]
    with open(out_dir / "summary_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    head = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
    body = "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)
    md = [
        f"# Phase 2 Step 3: perturbation generalization sweep ({stamp})", "",
        f"- dense CfC + LMS-main (eta=1e-3, leak=0, FEL), B={args.b_cycles} cycles, "
        f"||J0||={j0_norm:.4f}, null B0={null_mm:.3f}mm",
        f"- success {n_success}/{len(rows)} conditions", "",
        head + body, "",
        "success = B0 distorts (>max(2mm,1.5*null)) AND LMS improvement>=50% AND "
        "after_effect_ratio>=0.25 AND mirror_dirs>=6 AND finite. "
        "Failures kept as readout-only-limit candidates (Exp 2E).",
    ]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    np.savez_compressed(
        out_dir / "curves.npz", J0=J0,
        perturbations=np.array([r["perturbation"] for r in rows]),
        B0_mm=np.array([r["B0_mm"] for r in rows]),
        LMS_B_last_mm=np.array([r["LMS_B_last_mm"] for r in rows]),
        improvement=np.array([r["improvement_percent"] for r in rows]),
        after_effect_ratio=np.array([r["after_effect_ratio"] for r in rows]),
        mirror_dirs=np.array([r["mirror_dirs"] for r in rows]),
        **{f"lc_{k}": np.array(v) for k, v in lc_repr.items()},
    )

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # representative learning curves
    if lc_repr:
        plot_learning_curves(lc_repr, b0_ref_mm=null_mm,
                             path=out_dir / "figures" / "representative_learning_curves.png")

    # perturbation vs B0/LMS and improvement
    labels = [r["perturbation"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [r["B0_mm"] for r in rows], 0.4, label="B0", color="tab:gray")
    ax.bar(x + 0.2, [r["LMS_B_last_mm"] for r in rows], 0.4, label="LMS B_last", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("mean max perp dev (mm)")
    ax.set_title("Perturbation: B0 vs LMS-adapted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "perturbation_B0_vs_LMS.png", dpi=140)
    plt.close(fig)

    # perturbation vs after-effect ratio / mirror dirs
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["tab:green" if r["success"] else "tab:red" for r in rows]
    ax.bar(x, [r["after_effect_ratio"] for r in rows], color=colors)
    ax.axhline(0.25, ls="--", c="k", lw=1, label="ratio threshold 0.25")
    for i, r in enumerate(rows):
        ax.text(i, r["after_effect_ratio"] + 0.02, f"{r['mirror_dirs']}/8",
                ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("after-effect ratio")
    ax.set_title("After-effect by perturbation (green=success, red=fail; n/8=mirror dirs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "perturbation_after_effect.png", dpi=140)
    plt.close(fig)

    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
