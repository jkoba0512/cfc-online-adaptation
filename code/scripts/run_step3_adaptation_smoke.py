"""Step 3 smoke: FEL + LMS/RLS online readout adaptation over the A/B/C protocol.

Pipeline:
  1. train CfC inverse model (null-field teacher), freeze core, export J0
  2. B0 reference: curl field, no adaptation
  3. LMS smoke (eta=1e-4, leakage=1e-4) over A/B/C; optional LMS grid
  4. RLS smoke (beta=0.999, p0=1.0); RLS beta compare
  5. dump config / metrics / trajectories / adaptation_curves / final_readouts

Goal (Step 3): show that adapting J online with the FEL signal tau_fb reduces
the curl-field distortion below B0; after-effect is logged but not a gate here.

Run:
  export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/cfc-online-adaptation
  cd code
  uv run python scripts/run_step3_adaptation_smoke.py            # smoke + grid
  uv run python scripts/run_step3_adaptation_smoke.py --no-grid  # smoke only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cfc_online_adaptation.arm import ArmParams, TwoLinkArm
from cfc_online_adaptation.core import CfCInverseModel
from cfc_online_adaptation.dataset import build_reach_sequences, train_val_split
from cfc_online_adaptation.experiment import train_inverse_model
from cfc_online_adaptation.protocol import run_abc_protocol
from cfc_online_adaptation.readout import LinearReadout, RLSReadout
from cfc_online_adaptation.tasks import joint_space_reaches

CENTER_XY = (0.0, 0.45)
DT = 0.005
T_MOVE = 0.5
FIELD_B = 13.0
KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])
ETA_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
LEAK_GRID = [0.0, 1e-4, 1e-3]


def _b_curve(curves):
    """Per-cycle B-phase mean_max_perp_dev as an array (learning curve)."""
    return np.array([c["mean_max_perp_dev"] for c in curves if c["phase"] == "B"])


def _summarise(out, j0_norm, label):
    b = [c for c in curves_of(out) if c["phase"] == "B"]
    finite = all(c["all_finite"] for c in out["curves"])
    b_first = b[0]["mean_max_perp_dev"] if b else float("nan")
    b_last = b[-1]["mean_max_perp_dev"] if b else float("nan")
    max_jn = max(c["j_norm"] for c in out["curves"])
    max_djn = max(c["dj_norm"] for c in out["curves"])
    taufb_first = b[0]["taufb_rms"] if b else float("nan")
    taufb_last = b[-1]["taufb_rms"] if b else float("nan")
    return {
        "label": label,
        "finite": bool(finite),
        "b_first_perp_mm": b_first * 1e3,
        "b_last_perp_mm": b_last * 1e3,
        "max_j_norm": max_jn,
        "max_dj_norm": max_djn,
        "j0_norm": j0_norm,
        "taufb_rms_first": taufb_first,
        "taufb_rms_last": taufb_last,
        "after_effect_mean_signed_mm": out["after_effect"]["mean_signed_perp_dev"] * 1e3,
        "after_effect_per_dir_mm": [v * 1e3 for v in out["after_effect"]["signed_perp_dev_per_dir"]],
    }


def curves_of(out):
    return out["curves"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--b-cycles", type=int, default=15)
    ap.add_argument("--grid", dest="grid", action="store_true", default=True)
    ap.add_argument("--no-grid", dest="grid", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outputs", default=None)
    args = ap.parse_args()

    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.outputs) if args.outputs else Path("../outputs") / f"phase1_step3_{stamp}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    arm = TwoLinkArm(ArmParams())
    center_q = arm.inverse_kinematics(np.array(CENTER_XY), elbow="down")

    # 1. train + freeze + export J0
    ds = build_reach_sequences(
        arm, center_q, radii=(0.08, 0.10, 0.12), durations=(T_MOVE,), dt=DT, n_targets=8
    )
    train_ds, val_ds = train_val_split(ds, val_frac=0.25, seed=args.seed)
    torch.manual_seed(args.seed)  # seed BEFORE model init for reproducible J0
    model = CfCInverseModel(input_dim=10, units=64, output_dim=2)
    history = train_inverse_model(
        model, train_ds, val_ds, epochs=args.epochs, lr=1e-3, device=train_device, seed=args.seed
    )
    model.freeze_core()
    model.to("cpu")  # online adaptation loop runs on CPU (cheap per-step, no sync)
    J0 = model.export_readout_weights()
    j0_norm = float(np.linalg.norm(J0))
    print(f"offline val MSE {history['val_mse'][0]:.4f} -> {history['val_mse'][-1]:.4f}; "
          f"||J0||={j0_norm:.3f}")

    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=T_MOVE, dt=DT, n_targets=8)

    # 2. B0 reference (curl, no adapt) via the same numpy path
    ro_b0 = LinearReadout.from_weights(J0, eta=0.0, leakage=0.0)
    b0 = run_abc_protocol(
        arm, model, ro_b0, reaches, KP, KD, J0, dt=DT, field_b=FIELD_B,
        b_cycles=1, washout_cycles=0, device="cpu",
    )
    b0_ref_perp = [c for c in b0["curves"] if c["phase"] == "B"][0]["mean_max_perp_dev"]
    print(f"B0 reference curl perp = {b0_ref_perp*1e3:.2f} mm")

    # 3. LMS smoke
    ro_lms = LinearReadout.from_weights(J0, eta=1e-4, leakage=1e-4)
    lms_smoke = run_abc_protocol(
        arm, model, ro_lms, reaches, KP, KD, J0, dt=DT, field_b=FIELD_B,
        b_cycles=args.b_cycles, washout_cycles=1, washout_adapt=False, device="cpu",
    )
    lms_smoke_sum = _summarise(lms_smoke, j0_norm, "LMS eta=1e-4 leak=1e-4")
    print(f"LMS smoke: B {lms_smoke_sum['b_first_perp_mm']:.2f} -> "
          f"{lms_smoke_sum['b_last_perp_mm']:.2f} mm, finite={lms_smoke_sum['finite']}, "
          f"after-effect {lms_smoke_sum['after_effect_mean_signed_mm']:.2f} mm")

    # 4. optional LMS grid
    grid_results = []
    if args.grid:
        for eta in ETA_GRID:
            for leak in LEAK_GRID:
                ro = LinearReadout.from_weights(J0, eta=eta, leakage=leak)
                out = run_abc_protocol(
                    arm, model, ro, reaches, KP, KD, J0, dt=DT, field_b=FIELD_B,
                    b_cycles=args.b_cycles, washout_cycles=0, device="cpu",
                )
                b = [c for c in out["curves"] if c["phase"] == "B"]
                finite = all(c["all_finite"] for c in out["curves"])
                last = b[-1]["mean_max_perp_dev"]
                max_jn = max(c["j_norm"] for c in out["curves"])
                unstable = (not finite) or (last > b0_ref_perp) or (max_jn > 10 * j0_norm)
                grid_results.append({
                    "eta": eta, "leakage": leak,
                    "b_last_perp_mm": last * 1e3,
                    "improvement_ratio": float(last / b0_ref_perp),
                    "finite": bool(finite), "max_j_norm": max_jn,
                    "unstable": bool(unstable),
                })
                print(f"  grid eta={eta:.0e} leak={leak:.0e}: "
                      f"B_last={last*1e3:.2f}mm finite={finite} unstable={unstable}")
        stable = [g for g in grid_results if not g["unstable"]]
        best = min(stable, key=lambda g: g["b_last_perp_mm"]) if stable else None
    else:
        best = None

    # 5. RLS smoke + beta compare
    ro_rls = RLSReadout.from_weights(J0, beta=0.999, p0=1.0)
    rls_smoke = run_abc_protocol(
        arm, model, ro_rls, reaches, KP, KD, J0, dt=DT, field_b=FIELD_B,
        b_cycles=args.b_cycles, washout_cycles=1, washout_adapt=False, device="cpu",
    )
    rls_smoke_sum = _summarise(rls_smoke, j0_norm, "RLS beta=0.999 p0=1.0")
    print(f"RLS smoke: B {rls_smoke_sum['b_first_perp_mm']:.2f} -> "
          f"{rls_smoke_sum['b_last_perp_mm']:.2f} mm, finite={rls_smoke_sum['finite']}")

    rls_compare = []
    for beta in (1.0, 0.999):
        ro = RLSReadout.from_weights(J0, beta=beta, p0=1.0)
        out = run_abc_protocol(
            arm, model, ro, reaches, KP, KD, J0, dt=DT, field_b=FIELD_B,
            b_cycles=args.b_cycles, washout_cycles=0, device="cpu",
        )
        b = [c for c in out["curves"] if c["phase"] == "B"]
        finite = all(c["all_finite"] for c in out["curves"])
        rls_compare.append({
            "beta": beta, "p0": 1.0,
            "b_last_perp_mm": b[-1]["mean_max_perp_dev"] * 1e3,
            "improvement_ratio": float(b[-1]["mean_max_perp_dev"] / b0_ref_perp),
            "finite": bool(finite),
        })

    # ----- dump -----
    config = {
        "stamp": stamp, "center_xy": list(CENTER_XY), "center_q": center_q.tolist(),
        "dt": DT, "T_move": T_MOVE, "field_b": FIELD_B,
        "Kp": KP.diagonal().tolist(), "Kd": KD.diagonal().tolist(),
        "epochs": args.epochs, "b_cycles": args.b_cycles, "seed": args.seed,
        "train_device": train_device, "adapt_device": "cpu",
        "eta_grid": ETA_GRID, "leak_grid": LEAK_GRID, "grid": args.grid,
        "direction_order": lms_smoke["order"],
    }
    metrics = {
        "offline_val_mse_first": history["val_mse"][0],
        "offline_val_mse_last": history["val_mse"][-1],
        "j0_norm": j0_norm,
        "b0_reference_perp_mm": b0_ref_perp * 1e3,
        "lms_smoke": lms_smoke_sum,
        "lms_grid": grid_results,
        "lms_grid_best": best,
        "rls_smoke": rls_smoke_sum,
        "rls_compare": rls_compare,
        "lms_improvement_ratio": float(
            lms_smoke_sum["b_last_perp_mm"] / (b0_ref_perp * 1e3)
        ),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # learning curves (per cycle) for LMS smoke and RLS smoke
    def curve_arrays(out):
        return {
            "phase": np.array([c["phase"] for c in out["curves"]]),
            "mean_max_perp_dev": np.array([c["mean_max_perp_dev"] for c in out["curves"]]),
            "mean_endpoint_error": np.array([c["mean_endpoint_error"] for c in out["curves"]]),
            "taufb_rms": np.array([c["taufb_rms"] for c in out["curves"]]),
            "j_norm": np.array([c["j_norm"] for c in out["curves"]]),
            "dj_norm": np.array([c["dj_norm"] for c in out["curves"]]),
        }

    lms_c = curve_arrays(lms_smoke)
    rls_c = curve_arrays(rls_smoke)
    np.savez_compressed(
        out_dir / "adaptation_curves.npz",
        b0_ref_perp=np.array(b0_ref_perp),
        **{f"lms_{k}": v for k, v in lms_c.items()},
        **{f"rls_{k}": v for k, v in rls_c.items()},
    )

    # saved trajectories (A/B_first/B_last/C_catch) for LMS smoke
    traj = {}
    for tag, arrs in lms_smoke["saved_trajectories"].items():
        for k, v in arrs.items():
            traj[f"lms_{tag}_{k}"] = v
    np.savez_compressed(out_dir / "trajectories.npz", J0=J0, **traj)

    np.savez_compressed(
        out_dir / "final_readouts.npz",
        J0=J0, lms_final_J=lms_smoke["final_J"], rls_final_J=rls_smoke["final_J"],
    )
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
