"""Step 2 smoke: offline CfC inverse-model training + B0 no-adaptation closed loop.

Pipeline:
  1. build teacher-forced dataset from center-out reaches (null-field teacher torque)
  2. train CfC core + linear readout by supervised BPTT
  3. freeze core, export J0
  4. run B0 (no online adaptation) closed loop under null field and curl field
  5. dump config / metrics / trajectories / model to outputs/phase1_step2_<stamp>/

Goal is a smoke, not a result: "inverse model learns" + "B0 stays finite and the
curl field distorts the trajectory more than null field".

Run:
  export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/cfc-online-adaptation
  cd code
  uv run python scripts/run_step2_b0_smoke.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cfc_online_adaptation.arm import ArmParams, TwoLinkArm
from cfc_online_adaptation.dataset import build_reach_sequences, train_val_split
from cfc_online_adaptation.core import CfCInverseModel
from cfc_online_adaptation.experiment import run_b0, train_inverse_model
from cfc_online_adaptation.tasks import joint_space_reaches

CENTER_XY = (0.0, 0.45)
DT = 0.005
T_MOVE = 0.5
FIELD_B = 13.0
KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outputs", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.outputs) if args.outputs else Path("../outputs") / f"phase1_step2_{stamp}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    arm = TwoLinkArm(ArmParams())
    center_q = arm.inverse_kinematics(np.array(CENTER_XY), elbow="down")

    # 1. dataset (augment over radii for a meaningful val split)
    ds = build_reach_sequences(
        arm, center_q, radii=(0.08, 0.10, 0.12), durations=(T_MOVE,), dt=DT, n_targets=8
    )
    train_ds, val_ds = train_val_split(ds, val_frac=0.25, seed=args.seed)
    print(f"dataset: {len(ds)} seqs (train {len(train_ds)} / val {len(val_ds)}), "
          f"x {ds.x.shape}, tau {ds.tau.shape}, |tau| max {np.abs(ds.tau).max():.3f}")

    # 2. train inverse model
    model = CfCInverseModel(input_dim=10, units=64, output_dim=2)
    history = train_inverse_model(
        model, train_ds, val_ds, epochs=args.epochs, lr=args.lr,
        device=args.device, seed=args.seed,
    )
    print(f"offline train MSE {history['train_mse'][0]:.4f} -> {history['train_mse'][-1]:.4f}; "
          f"val MSE {history['val_mse'][0]:.4f} -> {history['val_mse'][-1]:.4f}")

    # 3. freeze core, export J0
    model.freeze_core()
    J0 = model.export_readout_weights()

    # 4. B0 closed loop, null and curl
    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=T_MOVE, dt=DT, n_targets=8)
    b0_null = run_b0(arm, model, reaches, KP, KD, dt=DT, field_b=0.0, device=args.device)
    b0_curl = run_b0(arm, model, reaches, KP, KD, dt=DT, field_b=FIELD_B, device=args.device)

    null_s, curl_s = b0_null["summary"], b0_curl["summary"]
    print(f"B0 null  : finite={null_s['all_finite']} "
          f"perp={null_s['mean_max_perp_dev']*1e3:.2f}mm ep={null_s['mean_endpoint_error']*1e3:.2f}mm")
    print(f"B0 curl  : finite={curl_s['all_finite']} "
          f"perp={curl_s['mean_max_perp_dev']*1e3:.2f}mm ep={curl_s['mean_endpoint_error']*1e3:.2f}mm")

    distorts = curl_s["mean_max_perp_dev"] > null_s["mean_max_perp_dev"]
    print(f"curl distorts more than null: {distorts}")

    # 5. dump artifacts
    config = {
        "stamp": stamp, "center_xy": list(CENTER_XY), "center_q": center_q.tolist(),
        "dt": DT, "T_move": T_MOVE, "field_b": FIELD_B,
        "Kp": KP.diagonal().tolist(), "Kd": KD.diagonal().tolist(),
        "epochs": args.epochs, "lr": args.lr, "seed": args.seed, "device": args.device,
        "units": 64, "input_dim": 10, "output_dim": 2,
        "n_seq": len(ds), "n_train": len(train_ds), "n_val": len(val_ds),
    }
    metrics = {
        "offline_train_mse_first": history["train_mse"][0],
        "offline_train_mse_last": history["train_mse"][-1],
        "offline_val_mse_first": history["val_mse"][0],
        "offline_val_mse_last": history["val_mse"][-1],
        "teacher_tau_abs_max": float(np.abs(ds.tau).max()),
        "b0_null": null_s,
        "b0_curl": curl_s,
        "curl_distorts_more_than_null": bool(distorts),
        "all_finite": bool(null_s["all_finite"] and curl_s["all_finite"]),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def stack(field_run):
        return {
            "ee": np.stack([r.ee for r in field_run["results"]]),
            "ee_des": np.stack([r.ee_des for r in field_run["results"]]),
            "q": np.stack([r.q for r in field_run["results"]]),
            "q_des": np.stack([r.q_des for r in field_run["results"]]),
            "tau": np.stack([r.tau for r in field_run["results"]]),
            "tau_ff": np.stack([r.tau_ff for r in field_run["results"]]),
            "tau_fb": np.stack([r.tau_fb for r in field_run["results"]]),
            "targets": np.stack([r.target_xy for r in field_run["results"]]),
        }

    null_arr, curl_arr = stack(b0_null), stack(b0_curl)
    np.savez_compressed(
        out_dir / "trajectories.npz",
        J0=J0,
        train_mse=np.array(history["train_mse"]),
        val_mse=np.array(history["val_mse"]),
        **{f"null_{k}": v for k, v in null_arr.items()},
        **{f"curl_{k}": v for k, v in curl_arr.items()},
    )
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
