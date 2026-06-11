"""Offline training and B0 (no-adaptation) experiment orchestration (Step 2).

* ``train_inverse_model`` runs supervised BPTT over the teacher-forced dataset
  (CfC core + linear readout jointly), returning the train/val MSE history.
* ``run_b0`` simulates the frozen model in closed loop over all reaches, under
  null field and curl field, aggregating the trajectory-error metrics.
"""

from __future__ import annotations

import numpy as np
import torch

from .arm import TwoLinkArm
from .control import run_closed_loop
from .core import CfCInverseModel
from .dataset import InverseDataset
from .metrics import (
    endpoint_error,
    endpoint_tracking_rmse,
    joint_tracking_rmse,
    max_abs_perpendicular_deviation,
)
from .tasks import Reach


def _to_tensor(ds: InverseDataset, device: str):
    x = torch.from_numpy(ds.x.astype(np.float32)).to(device)
    tau = torch.from_numpy(ds.tau.astype(np.float32)).to(device)
    return x, tau


def train_inverse_model(
    model: CfCInverseModel,
    train_ds: InverseDataset,
    val_ds: InverseDataset,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Supervised BPTT training of the CfC inverse model. Returns history dict."""
    torch.manual_seed(seed)
    model.to(device)
    x_tr, tau_tr = _to_tensor(train_ds, device)
    x_va, tau_va = _to_tensor(val_ds, device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = {"train_mse": [], "val_mse": []}

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred, _z, _hx = model(x_tr)
        loss = loss_fn(pred, tau_tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred_va, _z, _hx = model(x_va)
            val = loss_fn(pred_va, tau_va)
        history["train_mse"].append(float(loss.item()))
        history["val_mse"].append(float(val.item()))

    return history


def run_b0(
    arm: TwoLinkArm,
    model: CfCInverseModel,
    reaches: list[Reach],
    Kp: np.ndarray,
    Kd: np.ndarray,
    dt: float = 0.005,
    field_b: float = 0.0,
    device: str = "cpu",
) -> dict:
    """Run the frozen model (B0, no adaptation) over all reaches in one field.

    Returns per-direction results and aggregate metrics.
    """
    results = []
    perp_max = []
    perp_signed = []
    ep_err = []
    joint_rmse = []
    ee_rmse = []
    all_finite = True

    for reach in reaches:
        res = run_closed_loop(
            arm, model, reach, Kp, Kd, dt=dt, field_b=field_b, device=device
        )
        results.append(res)
        all_finite = all_finite and res.finite
        start_xy = res.ee_des[0]
        mabs, msig = max_abs_perpendicular_deviation(res.ee, start_xy, res.target_xy)
        perp_max.append(mabs)
        perp_signed.append(msig)
        ep_err.append(endpoint_error(res.ee[-1], res.target_xy))
        joint_rmse.append(joint_tracking_rmse(res.q, res.q_des))
        ee_rmse.append(endpoint_tracking_rmse(res.ee, res.ee_des))

    summary = {
        "field_b": float(field_b),
        "all_finite": bool(all_finite),
        "mean_max_perp_dev": float(np.mean(perp_max)),
        "mean_signed_perp_dev": float(np.mean(perp_signed)),
        "mean_endpoint_error": float(np.mean(ep_err)),
        "mean_joint_tracking_rmse": float(np.mean(joint_rmse)),
        "mean_endpoint_tracking_rmse": float(np.mean(ee_rmse)),
        "per_direction": {
            "max_perp_dev": [float(v) for v in perp_max],
            "signed_perp_dev": [float(v) for v in perp_signed],
            "endpoint_error": [float(v) for v in ep_err],
            "joint_tracking_rmse": [float(v) for v in joint_rmse],
        },
    }
    return {"results": results, "summary": summary}
