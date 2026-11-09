"""Untrained feature bases for the model-family comparison (spec C v0.1, 2026-10-30):
an echo-state network (ESN) and a Gaussian radial-basis-function (RBF) layer, each with
the same ``features`` / ``forward`` / readout interface as ``CfCInverseModel`` so that the
closed-loop, clamp and stage-2 code runs unchanged, plus the ridge readout fit and the
offline validation used to select candidates on null-field data only.

ESN: h_{t+1} = (1 - a) h_t + a tanh(W_in x~_t + W_rec h_t + b), h_0 = 0 at every reach,
W_in ~ U(-s_in, s_in), W_rec dense Gaussian scaled to spectral radius rho, b ~ U(-0.1, 0.1),
x~ = standardised input (train-split mean / SD). RBF: z_k = exp(-||x~ - c_k||^2 / (2 sigma^2))
with centres c_k from k-means on the standardised train inputs and sigma = multiplier x
median nearest-centre distance; no recurrence (the carried state is unused).

Ridge readout: J = T^T Z (Z^T Z + lam n s2 I)^-1 with s2 the mean feature variance, so lam
is a normalised regulariser comparable across feature scales. Pure numpy for the fits;
torch modules for the drop-in interface.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# fits
# ----------------------------------------------------------------------
def standardiser(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD (floored at 1e-8) over all samples of X (..., dim)."""
    flat = X.reshape(-1, X.shape[-1])
    return flat.mean(axis=0), np.maximum(flat.std(axis=0), 1e-8)


def kmeans_centres(X: np.ndarray, k: int, seed: int, n_iter: int = 50) -> np.ndarray:
    """Lloyd's k-means on rows of X (n, dim); seeded initial centres from the data."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    C = X[rng.choice(X.shape[0], size=k, replace=False)].copy()
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lab = d2.argmin(axis=1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(axis=0)
            else:  # re-seed an empty cluster from a random point
                C[j] = X[rng.integers(X.shape[0])]
    return C


def median_nn_distance(C: np.ndarray) -> float:
    d2 = ((C[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(axis=1))))


def ridge_readout(Z: np.ndarray, T: np.ndarray, lam: float) -> np.ndarray:
    """J (out, units) minimising ||Z J^T - T||^2 + lam n s2 ||J||^2 on flattened sequences."""
    Zf = Z.reshape(-1, Z.shape[-1]).astype(float)
    Tf = T.reshape(-1, T.shape[-1]).astype(float)
    n = Zf.shape[0]
    s2 = float(Zf.var(axis=0).mean())
    A = Zf.T @ Zf + lam * n * max(s2, 1e-12) * np.eye(Zf.shape[1])
    return np.linalg.solve(A, Zf.T @ Tf).T


def normalised_mse(pred: np.ndarray, T: np.ndarray) -> float:
    return float(((pred - T) ** 2).mean() / T.var())


# ----------------------------------------------------------------------
# modules
# ----------------------------------------------------------------------
class _BaseInverseModel(nn.Module):
    """Shared readout interface (mirrors ``CfCInverseModel``)."""

    def __init__(self, input_dim: int, units: int, output_dim: int):
        super().__init__()
        self.input_dim, self.units, self.output_dim = input_dim, units, output_dim
        self.readout = nn.Linear(units, output_dim, bias=False)
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))

    def set_standardiser(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.x_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.x_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def freeze_core(self) -> None:
        for name, p in self.named_parameters():
            if not name.startswith("readout"):
                p.requires_grad_(False)
        self.eval()

    def export_readout_weights(self) -> np.ndarray:
        return self.readout.weight.detach().cpu().numpy().copy()

    def load_readout_weights(self, J: np.ndarray) -> None:
        J = np.asarray(J, dtype=np.float32)
        if J.shape != (self.output_dim, self.units):
            raise ValueError(f"J shape {J.shape} != expected {(self.output_dim, self.units)}")
        with torch.no_grad():
            self.readout.weight.copy_(torch.from_numpy(J))

    def forward(self, x: torch.Tensor, hx=None):
        z, hx = self.features(x, hx)
        return self.readout(z), z, hx


class ESNInverseModel(_BaseInverseModel):
    """Leaky echo-state reservoir with a linear readout (state reset per reach)."""

    def __init__(self, input_dim: int = 10, units: int = 64, output_dim: int = 2, *, rho: float = 0.9,
                 s_in: float = 0.3, leak: float = 0.3, seed: int = 0):
        super().__init__(input_dim, units, output_dim)
        self.rho, self.s_in, self.leak, self.seed = float(rho), float(s_in), float(leak), int(seed)
        rng = np.random.default_rng(seed)
        W_in = rng.uniform(-s_in, s_in, size=(units, input_dim))
        W = rng.normal(size=(units, units))
        W_rec = W * (rho / max(np.abs(np.linalg.eigvals(W)).max(), 1e-12))
        b = rng.uniform(-0.1, 0.1, size=units)
        self.register_buffer("W_in", torch.as_tensor(W_in, dtype=torch.float32))
        self.register_buffer("W_rec", torch.as_tensor(W_rec, dtype=torch.float32))
        self.register_buffer("b", torch.as_tensor(b, dtype=torch.float32))

    def features(self, x: torch.Tensor, hx: torch.Tensor | None = None):
        xs = (x - self.x_mean) / self.x_std
        B, T, _ = xs.shape
        h = torch.zeros(B, self.units, dtype=xs.dtype, device=xs.device) if hx is None else hx
        out = []
        a = self.leak
        for t in range(T):
            h = (1.0 - a) * h + a * torch.tanh(xs[:, t] @ self.W_in.T + h @ self.W_rec.T + self.b)
            out.append(h)
        return torch.stack(out, dim=1), h

    def config(self) -> dict:
        return {"rho": self.rho, "s_in": self.s_in, "leak": self.leak, "seed": self.seed}


class RBFInverseModel(_BaseInverseModel):
    """Gaussian RBF layer on the standardised input with a linear readout (no recurrence)."""

    def __init__(self, input_dim: int = 10, units: int = 64, output_dim: int = 2, *, centres=None,
                 sigma: float = 1.0, width_mult: float = 1.0, seed: int = 0):
        super().__init__(input_dim, units, output_dim)
        self.sigma, self.width_mult, self.seed = float(sigma), float(width_mult), int(seed)
        C = np.zeros((units, input_dim)) if centres is None else np.asarray(centres, dtype=float)
        self.register_buffer("centres", torch.as_tensor(C, dtype=torch.float32))

    def set_centres(self, centres: np.ndarray, sigma: float) -> None:
        self.centres.copy_(torch.as_tensor(centres, dtype=torch.float32))
        self.sigma = float(sigma)

    def features(self, x: torch.Tensor, hx=None):
        xs = (x - self.x_mean) / self.x_std
        d2 = ((xs[..., None, :] - self.centres) ** 2).sum(dim=-1)
        return torch.exp(-d2 / (2.0 * self.sigma ** 2)), hx

    def config(self) -> dict:
        return {"sigma": self.sigma, "width_mult": self.width_mult, "seed": self.seed}


def build_rbf(train_x: np.ndarray, width_mult: float, seed: int, units: int = 64, input_dim: int = 10,
              output_dim: int = 2) -> RBFInverseModel:
    """RBF model with k-means centres on the standardised train inputs and sigma =
    width_mult x median nearest-centre distance."""
    mean, std = standardiser(train_x)
    Xs = (train_x.reshape(-1, input_dim) - mean) / std
    C = kmeans_centres(Xs, units, seed)
    sigma = width_mult * median_nn_distance(C)
    m = RBFInverseModel(input_dim, units, output_dim, centres=C, sigma=sigma, width_mult=width_mult, seed=seed)
    m.set_standardiser(mean, std)
    return m


def build_esn(train_x: np.ndarray, rho: float, s_in: float, leak: float, seed: int, units: int = 64,
              input_dim: int = 10, output_dim: int = 2) -> ESNInverseModel:
    m = ESNInverseModel(input_dim, units, output_dim, rho=rho, s_in=s_in, leak=leak, seed=seed)
    m.set_standardiser(*standardiser(train_x))
    return m


def teacher_forced_z(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Features (n_seq, T, units) of an input array (n_seq, T, input_dim), state reset per sequence."""
    model.eval()
    with torch.no_grad():
        z, _ = model.features(torch.from_numpy(np.asarray(X, dtype=np.float32)), None)
    return z.cpu().numpy().astype(np.float64)


def fit_and_validate(model: nn.Module, train_x, train_tau, val_x, val_tau, lam_grid) -> dict:
    """Ridge readout per lambda on the train split; normalised validation MSE; the best lambda
    (validation only) is loaded into the model's readout."""
    Ztr, Zva = teacher_forced_z(model, train_x), teacher_forced_z(model, val_x)
    rows = []
    for lam in lam_grid:
        J = ridge_readout(Ztr, train_tau, lam)
        rows.append({"lam": float(lam), "val_nmse": normalised_mse(Zva @ J.T, val_tau),
                     "train_nmse": normalised_mse(Ztr @ J.T, train_tau), "J": J})
    best = min(rows, key=lambda r: r["val_nmse"])
    model.load_readout_weights(best["J"])
    return {"lam": best["lam"], "val_nmse": best["val_nmse"], "train_nmse": best["train_nmse"],
            "grid": [{k: v for k, v in r.items() if k != "J"} for r in rows],
            "feature_norm_mean": float(np.linalg.norm(Ztr, axis=2).mean()),
            "feature_norm_min": float(np.linalg.norm(Ztr, axis=2).min()),
            "feature_norm_max": float(np.linalg.norm(Ztr, axis=2).max())}


# ----------------------------------------------------------------------
# checkpoint loading with metadata verification
# ----------------------------------------------------------------------
FAMILY_CLASSES = {"base": "CfCInverseModel", "centered": "CfCInverseModel", "cfc_ridge": "CfCInverseModel",
                  "esn": "ESNInverseModel", "rbf": "RBFInverseModel"}


def load_checkpoint_model(ckpt: str, expected_family: str | None = None):
    """Model rebuilt from ``<dir>/model.pt`` and ``<dir>/summary.json`` (``model_class`` and
    ``model_kwargs``; absent for the stage-2 pilot CfC checkpoints). With ``expected_family``
    the summary's family (or, for pilot checkpoints, the absence of one) must match, so a
    family never runs silently on another family's checkpoints."""
    import json
    from pathlib import Path

    from .core import CfCInverseModel
    path = Path(ckpt)
    summ_path = path.parent / "summary.json"
    summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    cls = summ.get("model_class", "CfCInverseModel")
    kw = summ.get("model_kwargs", {})
    if expected_family is not None:
        want = FAMILY_CLASSES[expected_family]
        if cls != want:
            raise ValueError(f"{ckpt}: model_class {cls!r} does not match family {expected_family!r} ({want})")
        fam = summ.get("family")
        if expected_family in ("cfc_ridge", "esn", "rbf") and fam != expected_family:
            raise ValueError(f"{ckpt}: checkpoint family {fam!r} != requested {expected_family!r}")
        if expected_family in ("base", "centered") and fam is not None:
            raise ValueError(f"{ckpt}: a spec-C checkpoint ({fam!r}) cannot be used for family {expected_family!r}")
    if cls == "ESNInverseModel":
        model = ESNInverseModel(10, 64, 2, **kw)
    elif cls == "RBFInverseModel":
        model = RBFInverseModel(10, 64, 2, **kw)
    elif cls == "CfCInverseModel":
        model = CfCInverseModel(10, 64, 2)
    else:
        raise ValueError(f"unknown model_class {cls!r}")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.freeze_core()
    model.eval()
    return model
