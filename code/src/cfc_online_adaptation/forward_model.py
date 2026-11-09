"""Forward (predictive) CfC model for the sensory-prediction climbing-fiber route
(Phase 2D, formulation v0.3 §6 / v0.2 §4.3 signal (c)).

A light dense CfC predicts the one-step state change of the plant from the
current state and applied torque:

    x_fwd  = [q_t, qdot_t, tau_t]            (dim 6)
    delta_s = [q_{t+1}-q_t, qdot_{t+1}-qdot_t]   (dim 4, predicted)

Trained on null-field on-policy data (the frozen inverse model + PD controller).
Under a force field the actual next state departs from the null-field prediction;
that residual, mapped through the mass matrix, estimates the disturbance torque
and serves as a sensory-prediction climbing-fiber signal.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from ncps.torch import CfC

from .arm import TwoLinkArm
from .control import run_closed_loop
from .core import CfCInverseModel
from .tasks import Reach

FWD_INPUT_DIM = 6
FWD_OUTPUT_DIM = 4


class ForwardCfCModel(nn.Module):
    """Dense CfC one-step forward model: [q, qdot, tau] -> [dq, dqdot]."""

    def __init__(self, input_dim: int = FWD_INPUT_DIM, units: int = 64,
                 output_dim: int = FWD_OUTPUT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.units = units
        self.output_dim = output_dim
        self.core = CfC(input_dim, units, batch_first=True, return_sequences=True)
        self.head = nn.Linear(units, output_dim)

    def forward(self, x: torch.Tensor, hx: torch.Tensor | None = None):
        z, hx = self.core(x, hx)
        return self.head(z), hx

    def predict_step(self, q, qdot, tau, hx, device: str = "cpu"):
        """One-step delta_s prediction for the online loop. Returns (delta_s, hx)."""
        x = np.concatenate([q, qdot, tau]).astype(np.float32)
        xt = torch.from_numpy(x).view(1, 1, -1).to(device)
        with torch.no_grad():
            delta, hx = self.forward(xt, hx)
        return delta.view(-1).cpu().numpy().astype(np.float64), hx


def collect_forward_dataset(
    arm: TwoLinkArm,
    inverse_model: CfCInverseModel,
    reaches: list[Reach],
    Kp: np.ndarray,
    Kd: np.ndarray,
    dt: float = 0.005,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Collect on-policy null-field (x_fwd, delta_s) sequences via the B0 controller.

    Runs the frozen inverse model + PD in the null field over each reach and logs
    per-step (q, qdot, tau) and the realised one-step state change. Returns
    X (n_seq, T-1, 6), Y (n_seq, T-1, 4).
    """
    xs, ys = [], []
    for reach in reaches:
        res = run_closed_loop(arm, inverse_model, reach, Kp, Kd, dt=dt, field_b=0.0, device=device)
        q, qdot, tau = res.q, res.qdot, res.tau
        x = np.concatenate([q[:-1], qdot[:-1], tau[:-1]], axis=1)  # (T-1, 6)
        dq = q[1:] - q[:-1]
        dqd = qdot[1:] - qdot[:-1]
        y = np.concatenate([dq, dqd], axis=1)  # (T-1, 4)
        xs.append(x)
        ys.append(y)
    return np.stack(xs).astype(np.float64), np.stack(ys).astype(np.float64)


def train_forward_model(
    fwd: ForwardCfCModel,
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    val_frac: float = 0.25,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Supervised BPTT training of the forward model. Returns history + one-step MSE."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])

    fwd.to(device)
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    Yt = torch.from_numpy(Y.astype(np.float32)).to(device)
    x_tr, y_tr = Xt[tr_idx], Yt[tr_idx]
    x_va, y_va = Xt[val_idx], Yt[val_idx]

    opt = torch.optim.Adam(fwd.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"train_mse": [], "val_mse": []}
    for _ in range(epochs):
        fwd.train()
        opt.zero_grad()
        pred, _ = fwd(x_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        opt.step()
        fwd.eval()
        with torch.no_grad():
            pred_va, _ = fwd(x_va)
            val = loss_fn(pred_va, y_va)
        history["train_mse"].append(float(loss.item()))
        history["val_mse"].append(float(val.item()))

    # one-step prediction MSE on the velocity channel (used by the sensory signal)
    with torch.no_grad():
        pred_va, _ = fwd(x_va)
        vel_mse = float(((pred_va[..., 2:] - y_va[..., 2:]) ** 2).mean().item())
    history["onestep_val_mse"] = history["val_mse"][-1]
    history["onestep_vel_mse"] = vel_mse
    return history
