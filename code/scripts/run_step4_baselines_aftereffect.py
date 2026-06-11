"""Step 4: B0 / B2 / LMS / RLS formal comparison + after-effect adjudication.

Lifts the Step 3 preliminary result to a near-Phase-1 evaluation:
  - reproducible CPU-seeded offline training (J0 fixed, checkpoint saved)
  - conditions: B0 (no adapt), B2 (feedback-only), LMS-main (eta=1e-3,leak=0),
    LMS leak-check, RLS-unconstrained, RLS-regularized p0 grid
  - A/B(30)/C-catch/C-washout protocol
  - after-effect judged against the B0 curl reference (mirror sign, ratio,
    n_mirror_dirs), B2 used as the no-internal-model control
  - quick diagnostic plots + dumped arrays

Run:
  export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/cfc-online-adaptation
  cd code
  uv run python scripts/run_step4_baselines_aftereffect.py
  uv run python scripts/run_step4_baselines_aftereffect.py --b-cycles 10 --no-rls-grid  # fast
"""

from __future__ import annotations

import argparse
import json
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
from cfc_online_adaptation.core import CfCInverseModel
from cfc_online_adaptation.dataset import build_reach_sequences, train_val_split
from cfc_online_adaptation.experiment import train_inverse_model
from cfc_online_adaptation.plots import (
    plot_after_effect_bars,
    plot_j_norm,
    plot_learning_curves,
    plot_trajectories_example,
)
from cfc_online_adaptation.protocol import run_abc_protocol
from cfc_online_adaptation.readout import LinearReadout, RLSReadout
from cfc_online_adaptation.tasks import joint_space_reaches

CENTER_XY = (0.0, 0.45)
DT = 0.005
T_MOVE = 0.5
FIELD_B = 13.0
KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])
RLS_P0_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
RLS_BETA_GRID = [1.0, 0.999]
LEAK_CHECK = [1e-3, 1e-2, 1e-1]


def b_curve_mm(out):
    return np.array([c["mean_max_perp_dev"] for c in get_phase_cycles(out, "B")]) * 1e3


def b_last_mm(out):
    return float(get_phase_cycles(out, "B")[-1]["mean_max_perp_dev"] * 1e3)


def cycle_jnorm(out):
    return np.array([c["j_norm"] for c in out["curves"]])


def taufb_first_last(out):
    b = get_phase_cycles(out, "B")
    return float(b[0]["taufb_rms"]), float(b[-1]["taufb_rms"])


def max_norms(out):
    return (max(c["j_norm"] for c in out["curves"]),
            max(c["dj_norm"] for c in out["curves"]))


def catch_signed_per_dir(out):
    return np.asarray(out["after_effect"]["signed_perp_dev_per_dir"], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--b-cycles", type=int, default=30)
    ap.add_argument("--washout-cycles", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rls-grid", dest="rls_grid", action="store_true", default=True)
    ap.add_argument("--no-rls-grid", dest="rls_grid", action="store_false")
    ap.add_argument("--outputs", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.outputs) if args.outputs else Path("../outputs") / f"phase1_step4_{stamp}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- reproducible CPU-seeded offline training ----
    device = "cpu"  # CPU training for reproducible J0 (Step 4 §3)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    arm = TwoLinkArm(ArmParams())
    center_q = arm.inverse_kinematics(np.array(CENTER_XY), elbow="down")
    ds = build_reach_sequences(
        arm, center_q, radii=(0.08, 0.10, 0.12), durations=(T_MOVE,), dt=DT, n_targets=8
    )
    train_ds, val_ds = train_val_split(ds, val_frac=0.25, seed=args.seed)
    model = CfCInverseModel(input_dim=10, units=64, output_dim=2)
    history = train_inverse_model(
        model, train_ds, val_ds, epochs=args.epochs, lr=1e-3, device=device, seed=args.seed
    )
    model.freeze_core()
    J0 = model.export_readout_weights()
    j0_norm = float(np.linalg.norm(J0))
    ckpt = out_dir / "model.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"offline val MSE {history['val_mse'][0]:.4f} -> {history['val_mse'][-1]:.4f}; "
          f"||J0||={j0_norm:.4f} (CPU seeded, reproducible)")

    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=T_MOVE, dt=DT, n_targets=8)

    def protocol(readout, J0_ref, b_cycles=None, washout=None):
        return run_abc_protocol(
            arm, model, readout, reaches, KP, KD, J0_ref, dt=DT, field_b=FIELD_B,
            b_cycles=b_cycles or args.b_cycles,
            washout_cycles=args.washout_cycles if washout is None else washout,
            washout_adapt=True, device="cpu",
        )

    # ---- main conditions ----
    print("running B0 ...")
    b0 = protocol(LinearReadout.from_weights(J0, eta=0.0), J0)
    print("running B2 ...")
    b2 = protocol(NullReadout(2, 64), np.zeros((2, 64)))
    print("running LMS-main (eta=1e-3) ...")
    lms = protocol(LinearReadout.from_weights(J0, eta=1e-3, leakage=0.0), J0)
    print("running RLS-unconstrained (beta=0.999,p0=1.0) ...")
    rls_unc = protocol(RLSReadout.from_weights(J0, beta=0.999, p0=1.0), J0)

    curl_ref = curl_reference_signed(b0)  # per-direction B0 curl sign reference

    def ae(out):
        return after_effect_metrics(curl_ref, catch_signed_per_dir(out))

    ae_b0, ae_b2, ae_lms, ae_rls = ae(b0), ae(b2), ae(lms), ae(rls_unc)
    claim = claim_supported(ae_lms, ae_b2)

    print(f"B0  B_last {b_last_mm(b0):.2f}mm | AE ratio {ae_b0['after_effect_ratio']:.2f} "
          f"mirror={ae_b0['mirror']} n={ae_b0['n_mirror_dirs']}/8")
    print(f"B2  B_last {b_last_mm(b2):.2f}mm | AE ratio {ae_b2['after_effect_ratio']:.2f} "
          f"mirror={ae_b2['mirror']} n={ae_b2['n_mirror_dirs']}/8")
    print(f"LMS B_last {b_last_mm(lms):.2f}mm | AE ratio {ae_lms['after_effect_ratio']:.2f} "
          f"mirror={ae_lms['mirror']} n={ae_lms['n_mirror_dirs']}/8")
    print(f"RLS B_last {b_last_mm(rls_unc):.2f}mm | AE ratio {ae_rls['after_effect_ratio']:.2f} "
          f"mirror={ae_rls['mirror']} n={ae_rls['n_mirror_dirs']}/8 | maxJ {max_norms(rls_unc)[0]:.1f}")
    print(f"claim supported: {claim['claim_supported']}")

    # ---- LMS leak check ----
    leak_check = []
    for leak in LEAK_CHECK:
        out = protocol(LinearReadout.from_weights(J0, eta=1e-3, leakage=leak), J0, washout=0)
        leak_check.append({"eta": 1e-3, "leakage": leak, "b_last_mm": b_last_mm(out),
                           "finite": all(c["all_finite"] for c in out["curves"])})
        print(f"  LMS leak={leak:.0e}: B_last {leak_check[-1]['b_last_mm']:.2f}mm")

    # ---- RLS regularized grid ----
    rls_grid = []
    rls_selected = None
    if args.rls_grid:
        lms_best = b_last_mm(lms)
        for beta in RLS_BETA_GRID:
            for p0 in RLS_P0_GRID:
                out = protocol(RLSReadout.from_weights(J0, beta=beta, p0=p0), J0, washout=0)
                maxj, _ = max_norms(out)
                a = after_effect_metrics(curl_ref, catch_signed_per_dir(out))
                finite = all(c["all_finite"] for c in out["curves"])
                ok = (finite and (b_last_mm(out) <= 1.5 * lms_best)
                      and (maxj <= 10 * j0_norm) and a["mirror"])
                rls_grid.append({
                    "beta": beta, "p0": p0, "b_last_mm": b_last_mm(out),
                    "max_j_norm": maxj, "mirror": a["mirror"],
                    "after_effect_ratio": a["after_effect_ratio"],
                    "finite": finite, "accept": bool(ok),
                })
                print(f"  RLS beta={beta} p0={p0}: B_last {b_last_mm(out):.2f}mm "
                      f"maxJ {maxj:.2f} mirror={a['mirror']} accept={ok}")
        accepted = [g for g in rls_grid if g["accept"]]
        # prefer lowest B_last among accepted, tie-break smaller J norm
        if accepted:
            rls_selected = min(accepted, key=lambda g: (g["b_last_mm"], g["max_j_norm"]))

    # full protocol for the selected stable RLS-regularized condition (for plots)
    rls_reg = None
    ae_rls_reg = None
    if rls_selected is not None:
        rls_reg = protocol(
            RLSReadout.from_weights(J0, beta=rls_selected["beta"], p0=rls_selected["p0"]), J0
        )
        ae_rls_reg = after_effect_metrics(curl_ref, catch_signed_per_dir(rls_reg))

    # ---- dump metrics ----
    config = {
        "stamp": stamp, "seed": args.seed, "device": device,
        "deterministic": "cpu-seeded", "model_checkpoint": str(ckpt),
        "center_xy": list(CENTER_XY), "center_q": center_q.tolist(),
        "dt": DT, "T_move": T_MOVE, "field_b": FIELD_B,
        "Kp": KP.diagonal().tolist(), "Kd": KD.diagonal().tolist(),
        "b_cycles": args.b_cycles, "washout_cycles": args.washout_cycles,
        "epochs": args.epochs, "rls_p0_grid": RLS_P0_GRID, "rls_beta_grid": RLS_BETA_GRID,
        "leak_check": LEAK_CHECK, "direction_order": b0["order"],
    }

    def cond_summary(out, ae_dict):
        mj, mdj = max_norms(out)
        tf, tl = taufb_first_last(out)
        return {
            "b_first_mm": float(b_curve_mm(out)[0]),
            "b_last_mm": b_last_mm(out),
            "max_j_norm": mj, "max_dj_norm": mdj,
            "taufb_rms_first": tf, "taufb_rms_last": tl,
            "after_effect": ae_dict,
            "finite": all(c["all_finite"] for c in out["curves"]),
        }

    metrics = {
        "offline_train_mse_last": history["train_mse"][-1],
        "offline_val_mse_first": history["val_mse"][0],
        "offline_val_mse_last": history["val_mse"][-1],
        "j0_norm": j0_norm,
        "b0_reference_perp_mm": b_last_mm(b0),
        "B0": cond_summary(b0, ae_b0),
        "B2": cond_summary(b2, ae_b2),
        "LMS_main": cond_summary(lms, ae_lms),
        "RLS_unconstrained": cond_summary(rls_unc, ae_rls),
        "LMS_leak_check": leak_check,
        "RLS_regularized_grid": rls_grid,
        "RLS_regularized_selected": rls_selected,
        "RLS_regularized_selected_full": (
            cond_summary(rls_reg, ae_rls_reg) if rls_reg is not None else None
        ),
        "after_effect_claim": claim,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- dump arrays ----
    def curve_arr(out):
        return {
            "phase": np.array([c["phase"] for c in out["curves"]]),
            "mean_max_perp_dev": np.array([c["mean_max_perp_dev"] for c in out["curves"]]),
            "mean_signed_perp_dev": np.array([c["mean_signed_perp_dev"] for c in out["curves"]]),
            "mean_endpoint_error": np.array([c["mean_endpoint_error"] for c in out["curves"]]),
            "taufb_rms": np.array([c["taufb_rms"] for c in out["curves"]]),
            "j_norm": np.array([c["j_norm"] for c in out["curves"]]),
            "dj_norm": np.array([c["dj_norm"] for c in out["curves"]]),
        }

    curves_dump = {}
    for name, out in [("B0", b0), ("B2", b2), ("LMS", lms), ("RLS_unc", rls_unc)]:
        curves_dump.update({f"{name}_{k}": v for k, v in curve_arr(out).items()})
    if rls_reg is not None:
        curves_dump.update({f"RLS_reg_{k}": v for k, v in curve_arr(rls_reg).items()})
    np.savez_compressed(out_dir / "curves.npz", **curves_dump)

    traj = {"J0": J0}
    for tag, arrs in lms["saved_trajectories"].items():
        for k, v in arrs.items():
            traj[f"lms_{tag}_{k}"] = v
    np.savez_compressed(out_dir / "trajectories.npz", **traj)
    np.savez_compressed(
        out_dir / "final_readouts.npz",
        J0=J0, lms_final_J=lms["final_J"], rls_final_J=rls_unc["final_J"],
    )

    # ---- plots (stable methods front-and-centre; RLS-unconstrained shown only
    #      in the J-norm panel to expose its divergence) ----
    lc = {"LMS (eta=1e-3)": b_curve_mm(lms)}
    aeb = {
        "B0": ae_b0["catch_mean_signed_mm"],
        "B2": ae_b2["catch_mean_signed_mm"],
        "LMS": ae_lms["catch_mean_signed_mm"],
    }
    jn = {"LMS": cycle_jnorm(lms), "RLS-unconstrained (b=.999,p0=1)": cycle_jnorm(rls_unc)}
    if rls_reg is not None:
        tag = f"RLS-reg (b={rls_selected['beta']},p0={rls_selected['p0']})"
        lc[tag] = b_curve_mm(rls_reg)
        aeb[tag] = ae_rls_reg["catch_mean_signed_mm"]
        jn[tag] = cycle_jnorm(rls_reg)
    plot_learning_curves(lc, b0_ref_mm=b_last_mm(b0), path=out_dir / "learning_curve.png")
    plot_after_effect_bars(aeb, path=out_dir / "after_effect_bars.png")
    plot_trajectories_example(traj, dir_idx=2, path=out_dir / "trajectories_example.png")
    plot_j_norm(jn, path=out_dir / "j_norm_curve.png", logy=True)

    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
