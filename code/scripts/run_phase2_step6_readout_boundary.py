"""Phase 2 Step 6 (Exp 2E): readout-only boundary search.

Question: is linear readout-only adaptation "broadly sufficient", or does some
perturbation make it clearly break? This is a boundary search only — no core
adaptation (RFLO/e-prop) is implemented here.

Probes (dense CfC, frozen core, LMS-main = eta=1e-3/leak=0/FEL; B=30 A/B/C):
  A. extreme curl       b = {40, 50, 70, 100}
  B. nonstationary curl alternating +/-13 (period 5) and abrupt +13->-13 mid-B,
                        LMS-main vs RLS-cov-reset-p0.1 (reset P at each sign switch)
  C. structural         high joint viscosity (3x, 5x) and large payload (m2 x3),
                        applied to the B-phase plant (arm_b)

A condition is a *true readout-only-limit candidate* iff: B0 distorts (>=10mm or
>=10x null) AND LMS improvement < 50% (or B_last > 10mm) AND after-effect weak
(ratio<0.25 / mirror<6 / not finite) AND RLS-cov-reset cannot rescue it. If LMS
fails but RLS-cov-reset rescues, it is a plasticity-control issue, not a limit.

Run:
  export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/cfc-online-adaptation
  cd code
  uv run python scripts/run_phase2_step6_readout_boundary.py
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

from cfc_online_adaptation.adaptation import run_adaptive_closed_loop
from cfc_online_adaptation.analysis import after_effect_metrics, curl_reference_signed, get_phase_cycles
from cfc_online_adaptation.arm import ArmParams, TwoLinkArm
from cfc_online_adaptation.core import CfCInverseModel
from cfc_online_adaptation.dataset import build_reach_sequences, train_val_split
from cfc_online_adaptation.experiment import train_inverse_model
from cfc_online_adaptation.metrics import max_abs_perpendicular_deviation
from cfc_online_adaptation.protocol import run_abc_protocol
from cfc_online_adaptation.readout import LinearReadout, RLSReadout
from cfc_online_adaptation.tasks import joint_space_reaches

CENTER_XY = (0.0, 0.45)
DT, T_MOVE = 0.005, 0.5
KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])
B_BASE = 0.05  # baseline joint viscous damping
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
    ap.add_argument("--b-cycles", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outputs", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(args.outputs) if args.outputs
               else Path("../outputs") / f"phase2_step6_readout_boundary_{stamp}").resolve()
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    arm = TwoLinkArm(ArmParams())
    center_q = arm.inverse_kinematics(np.array(CENTER_XY), elbow="down")
    ds = build_reach_sequences(arm, center_q, radii=(0.08, 0.10, 0.12), durations=(T_MOVE,),
                               dt=DT, n_targets=8)
    train_ds, val_ds = train_val_split(ds, val_frac=0.25, seed=args.seed)
    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=T_MOVE, dt=DT, n_targets=8)
    order = list(range(8))

    model = CfCInverseModel(10, 64, 2)
    hist = train_inverse_model(model, train_ds, val_ds, epochs=args.epochs, lr=1e-3,
                               device="cpu", seed=args.seed)
    model.freeze_core()
    J0 = model.export_readout_weights()
    j0_norm = float(np.linalg.norm(J0))
    print(f"||J0||={j0_norm:.4f}; val MSE -> {hist['val_mse'][-1]:.4f}")

    def protocol(readout, field_b, arm_b=None):
        with np.errstate(over="ignore", invalid="ignore"):
            return run_abc_protocol(
                arm, model, readout, reaches, KP, KD, J0, dt=DT, field_b=field_b,
                b_cycles=args.b_cycles, washout_cycles=0, arm_b=arm_b, device="cpu",
            )

    # null reference
    null_mm = b_last_mm(protocol(LinearReadout.from_weights(J0, eta=0.0), 0.0))

    rows = []

    def evaluate(label, kind, field_b, arm_b, method="LMS", keep=False):
        if method == "LMS":
            ro_b0 = LinearReadout.from_weights(J0, eta=0.0)
            ro = LinearReadout.from_weights(J0, eta=1e-3, leakage=0.0)
        else:  # RLS-reg confirmation
            ro_b0 = LinearReadout.from_weights(J0, eta=0.0)
            ro = RLSReadout.from_weights(J0, beta=1.0, p0=0.1)
        b0 = protocol(ro_b0, field_b, arm_b)
        out = protocol(ro, field_b, arm_b)
        b0_mm, last_mm = b_last_mm(b0), b_last_mm(out)
        improve = 100.0 * (1.0 - last_mm / b0_mm) if b0_mm > 0 else 0.0
        ref = curl_reference_signed(b0)
        ae = after_effect_metrics(ref, np.asarray(out["after_effect"]["signed_perp_dev_per_dir"]))
        finite = all(c["all_finite"] for c in out["curves"])
        b0_effective = b0_mm >= max(10.0, 10.0 * null_mm)
        limit_candidate = bool(
            b0_effective and ((improve < 50.0) or (last_mm > 10.0))
            and ((ae["after_effect_ratio"] < 0.25) or (ae["n_mirror_dirs"] < 6) or (not finite))
        )
        row = {
            "label": label, "kind": kind, "method": method,
            "B0_mm": round(b0_mm, 3), "LMS_B_last_mm": round(last_mm, 3),
            "improvement_percent": round(improve, 1),
            "after_effect_ratio": round(ae["after_effect_ratio"], 3),
            "mirror_dirs": ae["n_mirror_dirs"], "max_J_norm": round(max_jnorm(out), 3),
            "finite": finite, "b0_effective": bool(b0_effective),
            "limit_candidate": limit_candidate,
        }
        rows.append(row)
        print(f"{label:22s}[{method:7s}] B0 {b0_mm:7.2f} -> {last_mm:7.2f} ({improve:6.1f}%) "
              f"ratio {ae['after_effect_ratio']:.2f} mir {ae['n_mirror_dirs']}/8 "
              f"maxJ {max_jnorm(out):7.1f} fin {finite} limit={limit_candidate}")
        return row

    # ---- A. extreme curl ----
    print("== A. extreme curl ==")
    for b in (40, 50, 70, 100):
        evaluate(f"curl_b{b}", "extreme_curl", float(b), None, "LMS")
    evaluate("curl_b70_rls", "extreme_curl", 70.0, None, "RLS-reg")

    # ---- C. structural ----
    print("== C. structural ==")
    for scale in (3.0, 5.0):
        visc_arm = TwoLinkArm(ArmParams(b=B_BASE * scale))
        evaluate(f"viscosity_{scale:.0f}x", "viscosity", 0.0, visc_arm, "LMS")
    payload_arm = TwoLinkArm(ArmParams(m2=M2_BASE * 3.0))
    evaluate("payload_m2x3", "payload", 0.0, payload_arm, "LMS")
    # RLS confirmation on the strongest structural (viscosity 5x)
    evaluate("viscosity_5x_rls", "viscosity", 0.0, TwoLinkArm(ArmParams(b=B_BASE * 5.0)), "RLS-reg")

    # ---- B. nonstationary curl (manual per-cycle loop) ----
    print("== B. nonstationary curl ==")
    n = args.b_cycles
    alt = [13.0 if (c // 5) % 2 == 0 else -13.0 for c in range(n)]      # +/-13 period 5
    abrupt = [13.0 if c < n // 2 else -13.0 for c in range(n)]          # +13 then -13

    def run_nonstationary(schedule, method, reset_on_switch):
        if method == "LMS":
            ro = LinearReadout.from_weights(J0, eta=1e-3, leakage=0.0)
        elif method == "RLS-reg":
            ro = RLSReadout.from_weights(J0, beta=1.0, p0=0.1)
        else:  # RLS-cov-reset
            ro = RLSReadout.from_weights(J0, beta=1.0, p0=0.1)
        perp_cycle, jnorm_cycle = [], []
        finite = True
        prev_sign = None
        with np.errstate(over="ignore", invalid="ignore"):
            for fb in schedule:
                sign = np.sign(fb)
                if (reset_on_switch and prev_sign is not None and sign != prev_sign
                        and isinstance(ro, RLSReadout)):
                    ro.reset_covariance(0.1)
                prev_sign = sign
                perp = []
                for d in order:
                    res = run_adaptive_closed_loop(arm, model, ro, reaches[d], KP, KD, J0,
                                                   dt=DT, field_b=fb, adapt=True, device="cpu")
                    finite = finite and res.finite
                    mabs, _ = max_abs_perpendicular_deviation(res.ee, res.ee_des[0], res.target_xy)
                    perp.append(mabs)
                perp_cycle.append(float(np.mean(perp)) * 1e3)
                jnorm_cycle.append(float(np.linalg.norm(ro.J)))
        return {"perp_mm": perp_cycle, "jnorm": jnorm_cycle, "finite": finite,
                "mean_mm": float(np.mean(perp_cycle)), "last5_mm": float(np.mean(perp_cycle[-5:])),
                "max_jnorm": float(max(jnorm_cycle))}

    nonstat = {}
    for sched_name, sched in [("alternating", alt), ("abrupt_switch", abrupt)]:
        for method, reset in [("LMS", False), ("RLS-reg", False), ("RLS-cov-reset", True)]:
            key = f"{sched_name}_{method}"
            r = run_nonstationary(sched, method, reset)
            nonstat[key] = r
            print(f"{key:28s} mean {r['mean_mm']:6.2f}mm last5 {r['last5_mm']:6.2f}mm "
                  f"maxJ {r['max_jnorm']:7.1f} fin {r['finite']}")

    n_limit = sum(r["limit_candidate"] for r in rows)
    print(f"\nstationary limit candidates: {n_limit}/{len(rows)}")

    # ---- dump ----
    config = {
        "stamp": stamp, "seed": args.seed, "device": "cpu", "deterministic": "cpu-seeded",
        "model": "dense", "method_main": "LMS-main", "b_cycles": args.b_cycles,
        "epochs": args.epochs, "Kp": KP.diagonal().tolist(), "Kd": KD.diagonal().tolist(),
        "null_b0_mm": null_mm, "j0_norm": j0_norm, "b_base": B_BASE, "m2_base": M2_BASE,
    }
    metrics = {"j0_norm": j0_norm, "null_b0_mm": null_mm, "stationary_table": rows,
               "nonstationary": nonstat, "n_stationary_limit_candidates": n_limit}
    (out_dir / "config.json").write_text(json.dumps(json_safe(config), indent=2, allow_nan=False))
    (out_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, allow_nan=False))

    cols = ["label", "kind", "method", "B0_mm", "LMS_B_last_mm", "improvement_percent",
            "after_effect_ratio", "mirror_dirs", "max_J_norm", "finite", "b0_effective",
            "limit_candidate"]
    with open(out_dir / "summary_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    head = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
    body = "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)
    ns_lines = ["", "## Nonstationary curl", "",
                "| condition | mean mm | last5 mm | max||J|| | finite |",
                "| --- | ---: | ---: | ---: | --- |"]
    for k, r in nonstat.items():
        ns_lines.append(f"| {k} | {r['mean_mm']:.2f} | {r['last5_mm']:.2f} | "
                        f"{r['max_jnorm']:.1f} | {r['finite']} |")
    md = [
        f"# Phase 2 Step 6: readout-only boundary search ({stamp})", "",
        f"- dense CfC + LMS-main, B={args.b_cycles}, ||J0||={j0_norm:.4f}, null B0={null_mm:.3f}mm",
        f"- stationary readout-only-limit candidates: {n_limit}/{len(rows)}", "",
        "## Stationary perturbations", "", head + body,
        "\n".join(ns_lines), "",
        "limit_candidate = B0 effective (>=10mm or 10x null) AND LMS improvement<50% (or "
        "B_last>10mm) AND weak after-effect (ratio<0.25/mirror<6/not finite). If LMS fails "
        "but RLS-cov-reset rescues (nonstationary), it is a plasticity-control issue, not a limit.",
    ]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    cdump = {"J0": J0,
             "stat_labels": np.array([r["label"] for r in rows]),
             "stat_B0_mm": np.array([r["B0_mm"] for r in rows]),
             "stat_LMS_last_mm": np.array([r["LMS_B_last_mm"] for r in rows]),
             "stat_improvement": np.array([r["improvement_percent"] for r in rows])}
    for k, r in nonstat.items():
        cdump[f"ns_{k}_perp_mm"] = np.array(r["perp_mm"])
    np.savez_compressed(out_dir / "curves.npz", **cdump)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.2, [r["B0_mm"] for r in rows], 0.4, label="B0", color="tab:gray")
    ax.bar(x + 0.2, [r["LMS_B_last_mm"] for r in rows], 0.4, label="adapted", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("mean max perp dev (mm)")
    ax.set_title("Boundary probes: B0 vs adapted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "boundary_B0_vs_adapted.png", dpi=140)
    plt.close(fig)

    # nonstationary curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, sched_name in zip(axes, ("alternating", "abrupt_switch")):
        for method in ("LMS", "RLS-reg", "RLS-cov-reset"):
            r = nonstat[f"{sched_name}_{method}"]
            ax.plot(range(1, len(r["perp_mm"]) + 1), r["perp_mm"], marker=".", label=method)
        ax.set_title(sched_name)
        ax.set_xlabel("cycle")
        ax.set_ylabel("mean max perp dev (mm)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "nonstationary_curves.png", dpi=140)
    plt.close(fig)

    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
