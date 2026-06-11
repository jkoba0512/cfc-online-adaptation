"""Phase 1 baselines (Step 4).

B2 = feedback-only control (no internal model): tau_ff = 0, tau = tau_fb. It is
implemented as a ``NullReadout`` that always outputs zero and never adapts, so
it slots straight into ``run_abc_protocol`` / ``run_adaptive_closed_loop``:

    * no J state, nothing carries across trials,
    * the CfC core may still be stepped, but its features are ignored,
    * therefore no learned (mirror) after-effect can appear in the C catch.

B0 (no-adaptation, frozen J0 inverse model) is not here — it is just a
``LinearReadout.from_weights(J0, eta=0)`` run with the field on and adapt off.
"""

from __future__ import annotations

import numpy as np


class NullReadout:
    """Feedback-only readout: predicts zero torque and never updates (B2)."""

    def __init__(self, output_dim: int = 2, feature_dim: int = 64):
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        self.J = np.zeros((output_dim, feature_dim), dtype=float)

    def predict(self, z: np.ndarray) -> np.ndarray:
        return np.zeros(self.output_dim, dtype=float)

    def update(self, error: np.ndarray, z: np.ndarray) -> None:
        # B2 has no internal model: nothing is learned.
        return None

    def copy(self) -> "NullReadout":
        return NullReadout(self.output_dim, self.feature_dim)
