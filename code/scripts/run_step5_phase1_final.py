"""Step 5: Phase 1 completion package.

No new headline experiment -- this fixes the Phase 1 result into a portable
form: a single summary table, paper-grade-adjacent figures, a tracking-error
signal ablation (one condition), and an after-effect decay summary.

Conditions (all on the same reproducible CPU-seeded J0):
  B0                 no adaptation (frozen J0 inverse model)
  B2                 feedback-only (NullReadout)
  LMS-main           eta=1e-3, leakage=0, FEL signal           [main method]
  RLS-reg            beta=1.0, p0=0.1, FEL signal              [fast comparison]
  RLS-unstable       beta=0.999, p0=1.0  (negative result, diverges)
  LMS-tracking       eta=1e-3, FEL replaced by Kp(q*-q)        [ablation]

Protocol: A null (1) -> B curl b=13 (30) -> C catch null/adapt-off (1)
          -> C washout null/adapt-on (5).

Run:
  export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/cfc-online-adaptation
  cd code
  uv run python scripts/run_step5_phase1_final.py
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
    after_effect_decay,
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
DT, T_MOVE, FIELD_B = 0.005, 0.5, 13.0
KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])


def json_safe(o):
    """Recursively replace non-finite floats (inf/nan) with None for strict JSON."""
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


def b_first_mm(out):
    return float(get_phase_cycles(out, "B")[0]["mean_max_perp_dev"] * 1e3)


def cycle_jnorm(out):
    return np.array([c["j_norm"] for c in out["curves"]])


def max_norms(out):
    return (max(c["j_norm"] for c in out["curves"]),
            max(c["dj_norm"] for c in out["curves"]),
            max(c["taufb_rms"] for c in out["curves"]))


def catch_signed(out):
    return np.asarray(out["after_effect"]["signed_perp_dev_per_dir"], dtype=float)


def washout_signed_mm(out):
    return [c["mean_signed_perp_dev"] * 1e3 for c in get_phase_cycles(out, "C_washout")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--b-cycles", type=int, default=30)
    ap.add_argument("--washout-cycles", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outputs", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.outputs) if args.outputs else Path("../outputs") / f"phase1_final_{stamp}"
    out_dir = out_dir.resolve()
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
    model = CfCInverseModel(10, 64, 2)
    history = train_inverse_model(
        model, train_ds, val_ds, epochs=args.epochs, lr=1e-3, device=device, seed=args.seed
    )
    model.freeze_core()
    J0 = model.export_readout_weights()
    j0_norm = float(np.linalg.norm(J0))
    ckpt = out_dir / "model.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"||J0||={j0_norm:.4f} (CPU seeded); val MSE "
          f"{history['val_mse'][0]:.4f}->{history['val_mse'][-1]:.4f}")

    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=T_MOVE, dt=DT, n_targets=8)

    def protocol(readout, J0_ref, error_signal="fel"):
        # suppress expected overflow from intentionally-divergent conditions;
        # the finite/diverged flags remain the source of truth.
        with np.errstate(over="ignore", invalid="ignore"):
            return run_abc_protocol(
                arm, model, readout, reaches, KP, KD, J0_ref, dt=DT, field_b=FIELD_B,
                b_cycles=args.b_cycles, washout_cycles=args.washout_cycles,
                washout_adapt=True, error_signal=error_signal, device="cpu",
            )

    conds = {}
    print("running conditions ...")
    conds["B0"] = protocol(LinearReadout.from_weights(J0, eta=0.0), J0)
    conds["B2"] = protocol(NullReadout(2, 64), np.zeros((2, 64)))
    conds["LMS-main"] = protocol(LinearReadout.from_weights(J0, eta=1e-3, leakage=0.0), J0)
    conds["RLS-reg"] = protocol(RLSReadout.from_weights(J0, beta=1.0, p0=0.1), J0)
    conds["RLS-unstable"] = protocol(RLSReadout.from_weights(J0, beta=0.999, p0=1.0), J0)
    conds["LMS-tracking"] = protocol(
        LinearReadout.from_weights(J0, eta=1e-3, leakage=0.0), J0, error_signal="tracking"
    )

    curl_ref = curl_reference_signed(conds["B0"])

    # ---- summary table ----
    roles = {
        "B0": "lower bound (no adaptation)",
        "B2": "feedback-only control (no internal model)",
        "LMS-main": "main method (cerebellum-faithful)",
        "RLS-reg": "fast comparison (regularized, no forgetting)",
        "RLS-unstable": "negative result (forgetting diverges)",
        "LMS-tracking": "ablation (tracking-error climbing-fiber signal)",
    }
    b0_last = b_last_mm(conds["B0"])
    rows = []
    for name, out in conds.items():
        ae = after_effect_metrics(curl_ref, catch_signed(out))
        mj, mdj, mtf = max_norms(out)
        finite = all(c["all_finite"] for c in out["curves"])
        rows.append({
            "condition": name,
            "B_first_mm": round(b_first_mm(out), 3),
            "B_last_mm": round(b_last_mm(out), 3),
            "improvement_vs_B0_percent": round(100.0 * (1.0 - b_last_mm(out) / b0_last), 1),
            "C_catch_signed_mm": round(ae["catch_mean_signed_mm"], 2),
            "after_effect_ratio": round(ae["after_effect_ratio"], 3),
            "n_mirror_dirs": ae["n_mirror_dirs"],
            "max_J_norm": round(mj, 3),
            "max_taufb_rms": round(mtf, 4),
            "finite": finite,
            "diverged": (not finite),
            "claim_role": roles[name],
        })

    claim = claim_supported(
        after_effect_metrics(curl_ref, catch_signed(conds["LMS-main"])),
        after_effect_metrics(curl_ref, catch_signed(conds["B2"])),
    )

    # ---- after-effect decay ----
    decay = {}
    for name in ("LMS-main", "RLS-reg", "B2"):
        out = conds[name]
        ae = after_effect_metrics(curl_ref, catch_signed(out))
        decay[name] = after_effect_decay(ae["catch_mean_signed_mm"], washout_signed_mm(out))

    # ---- dump json/md ----
    summary = {
        "stamp": stamp, "seed": args.seed, "device": device, "deterministic": "cpu-seeded",
        "model_checkpoint": str(ckpt), "j0_norm": j0_norm,
        "offline_val_mse_first": history["val_mse"][0],
        "offline_val_mse_last": history["val_mse"][-1],
        "field_b": FIELD_B, "b_cycles": args.b_cycles, "Kp": KP.diagonal().tolist(),
        "Kd": KD.diagonal().tolist(), "b0_curl_mm": b0_last,
        "table": rows, "after_effect_claim": claim, "after_effect_decay": decay,
        "direction_order": conds["B0"]["order"],
    }
    # strict JSON: non-finite floats (e.g. diverged max_taufb_rms=inf) -> null
    (out_dir / "phase1_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )

    cols = ["condition", "B_first_mm", "B_last_mm", "improvement_vs_B0_percent",
            "C_catch_signed_mm", "after_effect_ratio", "n_mirror_dirs",
            "max_J_norm", "max_taufb_rms", "finite", "diverged", "claim_role"]
    with open(out_dir / "summary_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def md_table(rows):
        head = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
        body = ""
        for r in rows:
            body += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
        return head + body

    md = [
        f"# Phase 1 final summary ({stamp})", "",
        f"- seed={args.seed}, device={device} (cpu-seeded reproducible), ||J0||={j0_norm:.4f}",
        f"- curl b={FIELD_B}, B exposure cycles={args.b_cycles}, "
        f"Kp={KP.diagonal().tolist()}, Kd={KD.diagonal().tolist()}",
        f"- B0 curl baseline = {b0_last:.2f} mm",
        f"- **after-effect claim supported: {claim['claim_supported']}**", "",
        "## Conditions", "", md_table(rows), "",
        "## After-effect decay (C washout, adapt on)", "",
    ]
    for name, d in decay.items():
        md.append(f"- **{name}**: catch {d['catch_signed_mm']:.2f} mm -> "
                  f"washout {[round(v, 2) for v in d['washout_signed_mm_per_cycle']]} mm, "
                  f"decay_ratio {d['decay_ratio']:.3f}")
    (out_dir / "phase1_summary.md").write_text("\n".join(md) + "\n")

    # ---- arrays ----
    def curve_arr(out):
        return {k: np.array([c[k] for c in out["curves"]])
                for k in ["mean_max_perp_dev", "mean_signed_perp_dev",
                          "mean_endpoint_error", "taufb_rms", "j_norm", "dj_norm"]} | {
            "phase": np.array([c["phase"] for c in out["curves"]])}

    cdump = {}
    for name, out in conds.items():
        cdump.update({f"{name.replace('-', '_')}_{k}": v for k, v in curve_arr(out).items()})
    np.savez_compressed(out_dir / "curves.npz", **cdump)

    traj = {"J0": J0}
    for tag, arrs in conds["LMS-main"]["saved_trajectories"].items():
        for k, v in arrs.items():
            traj[f"lms_{tag}_{k}"] = v
    np.savez_compressed(out_dir / "trajectories.npz", **traj)
    np.savez_compressed(
        out_dir / "final_readouts.npz", J0=J0,
        lms_final_J=conds["LMS-main"]["final_J"], rls_reg_final_J=conds["RLS-reg"]["final_J"],
    )

    # ---- figures ----
    styles = {
        "B2": {"color": "tab:orange", "ls": "-"},
        "LMS-main": {"color": "tab:blue", "ls": "-"},
        "RLS-reg": {"color": "tab:green", "ls": "-"},
        "LMS-tracking": {"color": "tab:purple", "ls": "-."},
        "RLS-unstable": {"color": "tab:red", "ls": "--", "alpha": 0.5},
    }
    lc = {n: b_curve_mm(conds[n]) for n in ["B2", "LMS-main", "RLS-reg", "LMS-tracking"]}
    plot_learning_curves(lc, b0_ref_mm=b0_last, path=out_dir / "learning_curve.png", styles=styles)

    aeb = {n: after_effect_metrics(curl_ref, catch_signed(conds[n]))["catch_mean_signed_mm"]
           for n in ["B0", "B2", "LMS-main", "RLS-reg", "LMS-tracking"]}
    plot_after_effect_bars(aeb, path=out_dir / "after_effect_bars.png")

    plot_trajectories_example(traj, dir_idx=2, path=out_dir / "trajectories_example.png")

    jn = {n: cycle_jnorm(conds[n]) for n in ["LMS-main", "RLS-reg", "RLS-unstable"]}
    plot_j_norm(jn, path=out_dir / "j_norm_curve.png", logy=True, styles=styles)

    # ---- console ----
    print("\ncondition       Bfirst  Blast  impr%  catch(mm) ratio mirror  maxJ  finite")
    for r in rows:
        print(f"{r['condition']:14s} {r['B_first_mm']:6.1f} {r['B_last_mm']:6.2f} "
              f"{r['improvement_vs_B0_percent']:5.1f} {r['C_catch_signed_mm']:8.1f} "
              f"{r['after_effect_ratio']:5.2f} {r['n_mirror_dirs']}/8  "
              f"{r['max_J_norm']:6.1f}  {r['finite']}")
    print(f"\nclaim supported: {claim['claim_supported']}")
    for n, d in decay.items():
        print(f"decay {n}: catch {d['catch_signed_mm']:.1f}mm -> "
              f"last washout {d['last_washout_signed_mm']:.1f}mm (ratio {d['decay_ratio']:.2f})")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
