"""CfC inverse-dynamics model: frozen liquid core + linear readout (§2).

The CfC core produces the liquid feature sequence ``z_t`` (dim = units); a
bias-free linear head maps it to joint torque, ``tau_ff = J z_t``. That head IS
the adaptable readout ``J`` of the formulation: after offline BPTT training the
core is frozen (``freeze_core``) and ``J0`` is exported (``export_readout_weights``)
for the online LMS/RLS adaptation of Step 3.

ncps API (verified with ncps 1.0.1):
    CfC(input_size, units, batch_first=True, return_sequences=True)
    forward(x, hx=None) -> (output, hx)
      output: (batch, time, units)   # = z, the liquid features
      hx:     (batch, units)         # carry state for online stepping

timespans are left at the default (uniform unit step): Phase 1 uses a fixed
dt, so the CfC behaves as a fixed-rate continuous-time RNN. Irregular Δt is a
Phase 2 concern.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class CfCInverseModel(nn.Module):
    """Frozen-core CfC inverse model with an adaptable linear readout."""

    def __init__(self, input_dim: int = 10, units: int = 64, output_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.units = units
        self.output_dim = output_dim
        self.core = CfC(input_dim, units, batch_first=True, return_sequences=True)
        # Linear readout J (no bias): tau_ff = J z.
        self.readout = nn.Linear(units, output_dim, bias=False)

    def forward(
        self, x: torch.Tensor, hx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map an input sequence to a torque sequence.

        x: (batch, time, input_dim) -> tau: (batch, time, output_dim),
        also returns liquid features z (batch, time, units) and the carry
        state hx (batch, units) for online single-step continuation.
        """
        z, hx = self.core(x, hx)
        tau = self.readout(z)
        return tau, z, hx

    def features(self, x: torch.Tensor, hx: torch.Tensor | None = None):
        """Return only the liquid features z (and carry state)."""
        z, hx = self.core(x, hx)
        return z, hx

    # ------------------------------------------------------------------
    # Step-3-facing interface (freeze + readout export/inject)
    # ------------------------------------------------------------------
    def freeze_core(self) -> None:
        """Freeze all CfC core parameters (readout stays trainable)."""
        for p in self.core.parameters():
            p.requires_grad_(False)
        self.core.eval()

    def export_readout_weights(self) -> np.ndarray:
        """Return J0 as a numpy array of shape (output_dim, units)."""
        return self.readout.weight.detach().cpu().numpy().copy()

    def load_readout_weights(self, J: np.ndarray) -> None:
        """Inject a readout matrix J (output_dim, units) into the linear head."""
        J = np.asarray(J, dtype=np.float32)
        if J.shape != (self.output_dim, self.units):
            raise ValueError(
                f"J shape {J.shape} != expected {(self.output_dim, self.units)}"
            )
        with torch.no_grad():
            self.readout.weight.copy_(torch.from_numpy(J))


class WiredCfCInverseModel(nn.Module):
    """Frozen WiredCfC (NCP wiring) core + adaptable linear readout (Phase 2 2A).

    Unlike the dense CfC (whose ``output`` is the full ``units``-dim liquid
    feature), a wired CfC built from an ``AutoNCP`` wiring projects its output to
    the ``output_dim`` motor neurons. To keep the Phase 1 architecture — a wide
    liquid feature ``z`` with an adaptable readout ``J`` on top — we read the
    full recurrent **state** (``units``-dim) via the public forward's hidden
    return, not the 2-dim motor output. ``feature_mode`` makes this explicit:

        "state"  -> z = recurrent state h  (dim = units; default, matches dense)
        "output" -> z = wiring motor output (dim = output_dim; narrow)

    In "state" mode ``features`` steps the core one timestep at a time to collect
    the per-timestep state sequence (CfC returns only the final hidden for a
    multi-step call). The wiring's own motor-output projection is left unused.
    """

    def __init__(
        self,
        input_dim: int = 10,
        units: int = 64,
        output_dim: int = 2,
        *,
        feature_mode: str = "state",
        sparsity_level: float = 0.5,
        seed: int = 22222,
    ):
        super().__init__()
        if feature_mode not in ("state", "output"):
            raise ValueError("feature_mode must be 'state' or 'output'")
        self.input_dim = input_dim
        self.units = units
        self.output_dim = output_dim
        self.feature_mode = feature_mode
        self.sparsity_level = sparsity_level
        self.seed = seed
        self.wiring = AutoNCP(units, output_dim, sparsity_level=sparsity_level, seed=seed)
        self.core = CfC(input_dim, self.wiring, batch_first=True, return_sequences=True)
        # readout input dim depends on the feature mode
        self.d_z = units if feature_mode == "state" else output_dim
        self.readout = nn.Linear(self.d_z, output_dim, bias=False)

    def features(
        self, x: torch.Tensor, hx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the liquid feature sequence z (batch, time, d_z) and carry hx."""
        if self.feature_mode == "output":
            z, hx = self.core(x, hx)  # z = (batch, time, output_dim)
            return z, hx
        # state mode: collect per-timestep recurrent state (batch, units)
        steps = []
        T = x.shape[1]
        for t in range(T):
            _out, hx = self.core(x[:, t : t + 1, :], hx)
            steps.append(hx)
        z = torch.stack(steps, dim=1)  # (batch, time, units)
        return z, hx

    def forward(
        self, x: torch.Tensor, hx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, hx = self.features(x, hx)
        tau = self.readout(z)
        return tau, z, hx

    def freeze_core(self) -> None:
        """Freeze all wired-CfC core parameters (readout stays trainable)."""
        for p in self.core.parameters():
            p.requires_grad_(False)
        self.core.eval()

    def export_readout_weights(self) -> np.ndarray:
        """Return J0 as a numpy array of shape (output_dim, d_z)."""
        return self.readout.weight.detach().cpu().numpy().copy()

    def load_readout_weights(self, J: np.ndarray) -> None:
        """Inject a readout matrix J (output_dim, d_z)."""
        J = np.asarray(J, dtype=np.float32)
        if J.shape != (self.output_dim, self.d_z):
            raise ValueError(
                f"J shape {J.shape} != expected {(self.output_dim, self.d_z)}"
            )
        with torch.no_grad():
            self.readout.weight.copy_(torch.from_numpy(J))

    def wiring_params(self) -> dict:
        """Wiring metadata to persist in config/result."""
        return {
            "wiring": "AutoNCP",
            "units": self.units,
            "output_dim": self.output_dim,
            "feature_mode": self.feature_mode,
            "d_z": self.d_z,
            "sparsity_level": self.sparsity_level,
            "seed": self.seed,
            "n_core_params": int(sum(p.numel() for p in self.core.parameters())),
        }
