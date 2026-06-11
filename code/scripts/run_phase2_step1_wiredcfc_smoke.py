"""Phase 2 Step 1: WiredCfC (NCP wiring) readout-only adaptation smoke.

Question (Exp 2A): does replacing the dense CfC core with a WiredCfCCell / NCP
wiring keep readout-only online adaptation working?

Step 1 is deliberately small (curl b=13 only; no perturbation sweep, no B2/RLS,
no decay analysis):
  1. WiredCfC API/shape already inspected -> feature_mode="state" (z = 64-dim
     recurrent state; the wiring's 2-dim motor output is NOT used as the feature)
  2. null-field teacher-forced offline training (CPU seeded) -> freeze, export J0
  3. B0 no-adaptation: null vs curl (curl distortion must persist)
  4. LMS online adaptation smoke over A/B/C (must beat B0 + show mirror after-effect)

Run:
  export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/cfc-online-adaptation
  cd code
  uv run python scripts/run_phase2_step1_wiredcfc_smoke.py
"""

from __future__ import annotations

import argparse
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
from cfc_online_adaptation.core import WiredCfCInverseModel
from cfc_online_adaptation.dataset import build_reach_sequences, train_val_split
from cfc_online_adaptation.experiment import train_inverse_model
from cfc_online_adaptation.protocol import run_abc_protocol
from cfc_online_adaptation.readout import LinearReadout
from cfc_online_adaptation.tasks import joint_space_reaches

CENTER_XY = (0.0, 0.45)
DT, T_MOVE, FIELD_B = 0.005, 0.5, 13.0
KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])


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
    ap.add_argument("--b-cycles", type=int, default=15)
    ap.add_argument("--feature-mode", default="state", choices=["state", "output"])
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outputs", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(args.outputs) if args.outputs
               else Path("../outputs") / f"phase2_step1_wiredcfc_{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    arm = TwoLinkArm(ArmParams())
    center_q = arm.inverse_kinematics(np.array(CENTER_XY), elbow="down")

    ds = build_reach_sequences(
        arm, center_q, radii=(0.08, 0.10, 0.12), durations=(T_MOVE,), dt=DT, n_targets=8
    )
    train_ds, val_ds = train_val_split(ds, val_frac=0.25, seed=args.seed)

    model = WiredCfCInverseModel(
        10, 64, 2, feature_mode=args.feature_mode, sparsity_level=args.sparsity, seed=22222
    )
    wp = model.wiring_params()
    print(f"WiredCfC: {wp}")
    history = train_inverse_model(
        model, train_ds, val_ds, epochs=args.epochs, lr=1e-3, device=device, seed=args.seed
    )
    model.freeze_core()
    J0 = model.export_readout_weights()
    j0_norm = float(np.linalg.norm(J0))
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"val MSE {history['val_mse'][0]:.4f} -> {history['val_mse'][-1]:.4f}; "
          f"||J0||={j0_norm:.4f}, d_z={model.d_z}")

    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=T_MOVE, dt=DT, n_targets=8)

    def protocol(readout, J0_ref, field_b, b_cycles, washout=0):
        with np.errstate(over="ignore", invalid="ignore"):
            return run_abc_protocol(
                arm, model, readout, reaches, KP, KD, J0_ref, dt=DT, field_b=field_b,
                b_cycles=b_cycles, washout_cycles=washout, washout_adapt=True, device="cpu",
            )

    # B0 no-adaptation: null vs curl
    b0_null = protocol(LinearReadout.from_weights(J0, eta=0.0), J0, 0.0, 1)
    b0_curl = protocol(LinearReadout.from_weights(J0, eta=0.0), J0, FIELD_B, 1)
    null_mm, curl_mm = b_last_mm(b0_null), b_last_mm(b0_curl)
    print(f"B0 null {null_mm:.2f}mm | B0 curl {curl_mm:.2f}mm | curl>null: {curl_mm > null_mm}")

    # LMS online adaptation smoke
    lms = protocol(LinearReadout.from_weights(J0, eta=1e-3, leakage=0.0), J0,
                   FIELD_B, args.b_cycles, washout=1)
    lms_first, lms_last = b_curve_mm(lms)[0], b_last_mm(lms)
    curl_ref = curl_reference_signed(b0_curl)
    ae = after_effect_metrics(curl_ref, np.asarray(lms["after_effect"]["signed_perp_dev_per_dir"]))
    improve = 100.0 * (1.0 - lms_last / curl_mm)
    finite = all(c["all_finite"] for c in lms["curves"])
    print(f"LMS B {lms_first:.2f}->{lms_last:.2f}mm ({improve:.1f}% vs B0 curl) finite={finite} "
          f"maxJ={max_jnorm(lms):.2f}")
    print(f"after-effect: catch {ae['catch_mean_signed_mm']:.2f}mm ratio {ae['after_effect_ratio']:.2f} "
          f"mirror={ae['mirror']} n={ae['n_mirror_dirs']}/8")

    # acceptance check (Step 1 criteria)
    accept = {
        "curl_persists": bool(curl_mm > 1.5 * null_mm),
        "lms_beats_b0_50pct": bool(improve >= 50.0),
        "after_effect_mirror": bool(ae["mirror"] and ae["after_effect_ratio"] >= 0.25
                                    and ae["n_mirror_dirs"] >= 6),
        "finite": bool(finite),
    }
    accept["all_pass"] = bool(all(accept.values()))
    print(f"acceptance: {accept}")

    # dump
    config = {
        "stamp": stamp, "seed": args.seed, "device": device, "deterministic": "cpu-seeded",
        "wiring_params": wp, "feature_mode": args.feature_mode, "d_z": model.d_z,
        "center_xy": list(CENTER_XY), "dt": DT, "field_b": FIELD_B,
        "Kp": KP.diagonal().tolist(), "Kd": KD.diagonal().tolist(),
        "epochs": args.epochs, "b_cycles": args.b_cycles,
    }
    metrics = {
        "offline_val_mse_first": history["val_mse"][0],
        "offline_val_mse_last": history["val_mse"][-1],
        "j0_norm": j0_norm, "d_z": model.d_z,
        "b0_null_mm": null_mm, "b0_curl_mm": curl_mm,
        "lms_b_first_mm": lms_first, "lms_b_last_mm": lms_last,
        "lms_improvement_vs_b0_percent": improve,
        "lms_max_j_norm": max_jnorm(lms), "lms_finite": finite,
        "after_effect": ae, "acceptance": accept,
    }
    (out_dir / "config.json").write_text(json.dumps(json_safe(config), indent=2, allow_nan=False))
    (out_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, allow_nan=False))

    md = [
        f"# Phase 2 Step 1: WiredCfC readout-only adaptation smoke ({stamp})", "",
        f"- wiring: AutoNCP units={wp['units']} output={wp['output_dim']} "
        f"sparsity={wp['sparsity_level']} | feature_mode={model.feature_mode} d_z={model.d_z} "
        f"| core params={wp['n_core_params']}",
        f"- offline val MSE {history['val_mse'][0]:.4f} -> {history['val_mse'][-1]:.4f}, "
        f"||J0||={j0_norm:.4f}", "",
        "| metric | value |", "| --- | --- |",
        f"| B0 null | {null_mm:.2f} mm |",
        f"| B0 curl | {curl_mm:.2f} mm |",
        f"| LMS B first | {lms_first:.2f} mm |",
        f"| LMS B last | {lms_last:.2f} mm |",
        f"| improvement vs B0 curl | {improve:.1f}% |",
        f"| after-effect catch | {ae['catch_mean_signed_mm']:.2f} mm |",
        f"| after-effect ratio | {ae['after_effect_ratio']:.2f} |",
        f"| mirror dirs | {ae['n_mirror_dirs']}/8 |",
        f"| max ||J|| | {max_jnorm(lms):.2f} |",
        f"| finite | {finite} |", "",
        f"**acceptance: {accept}**",
    ]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    def curve_arr(out):
        return {k: np.array([c[k] for c in out["curves"]])
                for k in ["mean_max_perp_dev", "mean_signed_perp_dev", "taufb_rms",
                          "j_norm", "dj_norm"]} | {
            "phase": np.array([c["phase"] for c in out["curves"]])}

    cdump = {}
    for name, out in [("b0_null", b0_null), ("b0_curl", b0_curl), ("lms", lms)]:
        cdump.update({f"{name}_{k}": v for k, v in curve_arr(out).items()})
    cdump["J0"] = J0
    cdump["train_mse"] = np.array(history["train_mse"])
    cdump["val_mse"] = np.array(history["val_mse"])
    np.savez_compressed(out_dir / "curves.npz", **cdump)

    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
