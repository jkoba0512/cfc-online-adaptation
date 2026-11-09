"""Freeze offline baselines and the selection-seed-only LMS grids (manifest A)."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cfc_online_adaptation.arm import ArmParams, TwoLinkArm
from cfc_online_adaptation.bases import (
    build_esn,
    build_rbf,
    fit_and_validate,
    normalised_mse,
    teacher_forced_z,
)
from cfc_online_adaptation.core import CfCInverseModel, WiredCfCInverseModel
from cfc_online_adaptation.dataset import build_reach_sequences, train_val_split
from cfc_online_adaptation.experiment import train_inverse_model

SELECTION_SEEDS = (0, 1, 2)
EVALUATION_SEEDS = (3, 4, 5, 6, 7)
ALL_SEEDS = SELECTION_SEEDS + EVALUATION_SEEDS
LAM_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
ETA_MULTIPLIERS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
ESN_GRID = tuple(itertools.product((0.5, 0.9, 1.2), (0.1, 0.3, 1.0), (0.1, 0.3, 1.0)))
RBF_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
SOURCE_FILES = (
    "scripts/run_revision_offline.py", "scripts/run_revision_baselines.py",
    "src/cfc_online_adaptation/arm.py", "src/cfc_online_adaptation/tasks.py",
    "src/cfc_online_adaptation/dataset.py", "src/cfc_online_adaptation/core.py",
    "src/cfc_online_adaptation/experiment.py", "src/cfc_online_adaptation/bases.py",
    "src/cfc_online_adaptation/readout.py", "src/cfc_online_adaptation/adaptation.py",
    "src/cfc_online_adaptation/protocol.py", "src/cfc_online_adaptation/revision_manifest.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_for_seed(ds, seed):
    return train_val_split(ds, val_frac=0.25, seed=seed)


def feature_energy(model, train_x: np.ndarray) -> float:
    z = teacher_forced_z(model, train_x)
    return float(np.mean(np.sum(z * z, axis=-1)))


def offline_row(model, train, val, path: Path, *, history=None, fit=None, config=None):
    model.freeze_core()
    model.eval()
    with torch.no_grad():
        xtr = torch.from_numpy(train.x.astype(np.float32))
        xva = torch.from_numpy(val.x.astype(np.float32))
        ptr = model(xtr)[0].cpu().numpy()
        pva = model(xva)[0].cpu().numpy()
    torch.save(model.state_dict(), path)
    row = {
        "checkpoint": str(path.resolve()),
        "sha256": sha256(path),
        "model_class": type(model).__name__,
        "model_kwargs": config or {},
        "train_nmse": normalised_mse(ptr, train.tau),
        "val_nmse": normalised_mse(pva, val.tau),
        "train_mse": float(np.mean((ptr - train.tau) ** 2)),
        "val_mse": float(np.mean((pva - val.tau) ** 2)),
        "feature_energy_train": feature_energy(model, train.x),
        "readout_norm": float(np.linalg.norm(model.export_readout_weights())),
    }
    if history is not None:
        row["epochs"] = len(history["train_mse"])
    if fit is not None:
        row["lambda"] = fit["lam"]
        row["lambda_grid"] = fit["grid"]
    return row


def choose_family(family: str, split_data: dict[int, tuple]):
    candidates = []
    settings = ESN_GRID if family == "esn" else RBF_GRID
    for setting in settings:
        rows = []
        for seed in SELECTION_SEEDS:
            train, val = split_data[seed]
            model = (
                build_esn(train.x, *setting, seed=seed)
                if family == "esn"
                else build_rbf(train.x, setting, seed=seed)
            )
            fit = fit_and_validate(model, train.x, train.tau, val.x, val.tau, LAM_GRID)
            rows.append({"seed": seed, "lambda": fit["lam"], "val_nmse": fit["val_nmse"]})
        candidates.append({
            "setting": list(setting) if family == "esn" else setting,
            "selection_rows": rows,
            "mean_val_nmse": float(np.mean([r["val_nmse"] for r in rows])),
        })
    best = min(candidates, key=lambda r: r["mean_val_nmse"])
    return best, candidates


def compare_old_checkpoint(new_model, old_path: Path, val_x: np.ndarray):
    old = torch.load(old_path, map_location="cpu", weights_only=True)
    new = new_model.state_dict()
    matching_keys = set(old) == set(new)
    matching_shapes = matching_keys and all(old[k].shape == new[k].shape for k in old)
    exact = matching_shapes and all(torch.equal(old[k], new[k]) for k in old)
    result = {
        "old_checkpoint": str(old_path.resolve()),
        "old_sha256": sha256(old_path),
        "matching_keys": matching_keys,
        "matching_shapes": matching_shapes,
        "all_tensors_exact": exact,
    }
    if matching_shapes:
        result["max_tensor_abs_diff"] = max(
            float((old[k] - new[k]).abs().max()) for k in old
        )
        old_model = CfCInverseModel(10, 64, 2)
        old_model.load_state_dict(old)
        old_model.eval()
        new_model.eval()
        x = torch.from_numpy(val_x.astype(np.float32))
        with torch.no_grad():
            old_tau = old_model(x)[0].numpy()
            new_tau = new_model(x)[0].numpy()
        result["max_val_torque_abs_diff"] = float(np.max(np.abs(old_tau - new_tau)))
        result["validation_input"] = "seed-0 held-out null-field sequences, reset state per sequence"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path)
    args = parser.parse_args()
    out = args.outputs or Path("../outputs") / f"revision_offline_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    arm = TwoLinkArm(ArmParams())
    center_q = arm.inverse_kinematics(np.array([0.0, 0.45]), elbow="down")
    ds = build_reach_sequences(
        arm, center_q, radii=(0.08, 0.10, 0.12), durations=(0.5,), dt=0.005, n_targets=8
    )
    split_data = {seed: data_for_seed(ds, seed) for seed in ALL_SEEDS}
    manifest = {
        "schema": "revision-offline-A-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {name: sha256(Path(name)) for name in SOURCE_FILES},
        "selection_seeds": list(SELECTION_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "data": {"radii": [0.08, 0.10, 0.12], "directions": 8, "dt": 0.005,
                 "duration": 0.5, "val_frac": 0.25, "split_seed": "model seed"},
        "families": {},
        "offline_selection": {},
        "eta_multipliers": list(ETA_MULTIPLIERS),
    }

    for family in ("cfc", "ncp"):
        rows = {}
        for seed in ALL_SEEDS:
            train, val = split_data[seed]
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = (
                CfCInverseModel(10, 64, 2)
                if family == "cfc"
                else WiredCfCInverseModel(10, 64, 2, feature_mode="state",
                                          sparsity_level=0.5, seed=22222)
            )
            history = train_inverse_model(model, train, val, epochs=200 if family == "cfc" else 400,
                                          lr=1e-3, seed=seed)
            path = out / f"{family}_seed{seed}.pt"
            config = {} if family == "cfc" else {
                "feature_mode": "state", "sparsity_level": 0.5, "seed": 22222
            }
            rows[str(seed)] = offline_row(model, train, val, path, history=history, config=config)
            if family == "cfc" and seed == 0:
                old_path = Path("../outputs/phase1_final_20260610_131754/model.pt")
                manifest["cfc_seed0_old_comparison"] = compare_old_checkpoint(
                    model, old_path, val.x
                )
            print(f"{family} seed {seed}: val NMSE {rows[str(seed)]['val_nmse']:.5g}", flush=True)
        manifest["families"][family] = {"seeds": rows}

    for family in ("esn", "rbf"):
        best, candidates = choose_family(family, split_data)
        manifest["offline_selection"][family] = {
            "best": best,
            "best_at_upper_grid_edge": None if family == "esn" else best["setting"] == RBF_GRID[-1],
            "candidates": candidates,
        }
        setting = best["setting"]
        rows = {}
        for seed in ALL_SEEDS:
            train, val = split_data[seed]
            model = (
                build_esn(train.x, *setting, seed=seed)
                if family == "esn"
                else build_rbf(train.x, setting, seed=seed)
            )
            fit = fit_and_validate(model, train.x, train.tau, val.x, val.tau, LAM_GRID)
            path = out / f"{family}_seed{seed}.pt"
            rows[str(seed)] = offline_row(model, train, val, path, fit=fit, config=model.config())
            print(f"{family} seed {seed}: val NMSE {rows[str(seed)]['val_nmse']:.5g}", flush=True)
        manifest["families"][family] = {"seeds": rows, "selected_setting": setting}

    cfc_energy = float(np.mean([
        manifest["families"]["cfc"]["seeds"][str(s)]["feature_energy_train"]
        for s in SELECTION_SEEDS
    ]))
    eta_grid = {}
    for family in ("cfc", "ncp", "esn", "rbf"):
        rows = manifest["families"][family]["seeds"]
        selection_energy = float(np.mean([rows[str(s)]["feature_energy_train"]
                                          for s in SELECTION_SEEDS]))
        center = 1e-3 * cfc_energy / selection_energy
        eta_grid[family] = {
            "selection_feature_energy": selection_energy,
            "evaluation_feature_energy_descriptive": {
                str(s): rows[str(s)]["feature_energy_train"] for s in EVALUATION_SEEDS
            },
            "center": center,
            "grid": [center * m for m in ETA_MULTIPLIERS],
        }
    eta_grid["bias"] = {"selection_feature_energy": 1.0,
                        "center": 1e-3 * cfc_energy,
                        "grid": [1e-3 * cfc_energy * m for m in ETA_MULTIPLIERS]}
    manifest["eta_grid"] = eta_grid
    path = out / "manifest_A.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(path.resolve())


if __name__ == "__main__":
    main()
