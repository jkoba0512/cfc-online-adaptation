"""Linear readout and its online update rules (formulation §5).

The CfC core is frozen; only the linear readout ``J`` (shape output_dim x
feature_dim) adapts online. Two rules are provided:

    * LMS / Widrow-Hoff (§5.1, cerebellum-faithful, main):
        J <- (1 - eta*leakage) J + eta * outer(error, z)
    * RLS / FORCE (§5.2, fast & stable, comparison):
        standard recursive-least-squares with forgetting factor beta.

The sign convention is the feedback-error-learning one: ``error`` is the
desired-minus-actual residual (e.g. tau_fb), so a positive error grows the
feedforward output in that direction. In the synthetic regression test
``error = target - predict(z)`` and both rules drive the MSE down.

Pure numpy.
"""

from __future__ import annotations

import copy as _copy

import numpy as np


class LinearReadout:
    """Linear readout J with LMS / Widrow-Hoff online update.

    ``eta`` and ``leakage`` are stored on the instance so the online loop can
    call the unified ``update(error, z)`` interface shared with ``RLSReadout``.
    ``lms_update(error, z, eta, leakage)`` keeps an explicit-arg form too.
    """

    def __init__(
        self,
        output_dim: int,
        feature_dim: int,
        *,
        eta: float = 1e-3,
        leakage: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        self.eta = eta
        self.leakage = leakage
        # Start from zero: the offline-trained J0 is injected via from_weights
        # when used in the control loop; for the unit test a zero start is clean.
        self.J = np.zeros((output_dim, feature_dim), dtype=float)
        self._rng = rng
        # plasticity diagnostics (Phase 2C)
        self.last_dJ_norm = 0.0
        self.last_gain_norm = 0.0
        self.cumulative_update_norm = 0.0

    @classmethod
    def from_weights(
        cls, J: np.ndarray, *, eta: float = 1e-3, leakage: float = 0.0
    ) -> "LinearReadout":
        """Build a readout initialised to J0 (output_dim, feature_dim)."""
        J = np.asarray(J, dtype=float)
        ro = cls(J.shape[0], J.shape[1], eta=eta, leakage=leakage)
        ro.J = J.copy()
        return ro

    def copy(self) -> "LinearReadout":
        return _copy.deepcopy(self)

    def predict(self, z: np.ndarray) -> np.ndarray:
        """Readout output J @ z, shape (output_dim,)."""
        return self.J @ np.asarray(z, dtype=float)

    def lms_update(
        self,
        error: np.ndarray,
        z: np.ndarray,
        eta: float = 1e-3,
        leakage: float = 0.0,
    ) -> None:
        """In-place LMS update: J <- (1-eta*leakage)J + eta * outer(error, z)."""
        error = np.asarray(error, dtype=float)
        z = np.asarray(z, dtype=float)
        if leakage:
            self.J *= 1.0 - eta * leakage
        dJ = eta * np.outer(error, z)
        self.J += dJ
        self.last_dJ_norm = float(np.linalg.norm(dJ))
        self.cumulative_update_norm += self.last_dJ_norm

    def update(self, error: np.ndarray, z: np.ndarray) -> None:
        """Unified online update using the instance eta/leakage."""
        self.lms_update(error, z, self.eta, self.leakage)

    def pop_cumulative_update_norm(self) -> float:
        """Return the accumulated ||ΔJ|| since the last call, then reset it."""
        v = self.cumulative_update_norm
        self.cumulative_update_norm = 0.0
        return v


class DirectionalIntegralBiasReadout:
    """Frozen readout plus an LMS bias indexed by the known reach direction."""

    def __init__(self, base_J: np.ndarray, *, n_contexts: int = 8, eta: float = 1e-3):
        self.base_J = np.asarray(base_J, dtype=float).copy()
        if self.base_J.ndim != 2 or n_contexts <= 0 or eta < 0:
            raise ValueError("invalid base_J, n_contexts or eta")
        self.n_contexts = int(n_contexts)
        self.eta = float(eta)
        self.biases = np.zeros((n_contexts, self.base_J.shape[0]))
        self.current_context = 0

    @property
    def J(self) -> np.ndarray:
        return np.concatenate([self.base_J, self.biases.T], axis=1)

    @property
    def J0(self) -> np.ndarray:
        return np.concatenate([self.base_J, np.zeros_like(self.biases.T)], axis=1)

    def set_context(self, context: int) -> None:
        if not 0 <= context < self.n_contexts:
            raise ValueError("context outside range")
        self.current_context = int(context)

    def predict(self, z: np.ndarray) -> np.ndarray:
        return self.base_J @ np.asarray(z, dtype=float) + self.biases[self.current_context]

    def update(self, error: np.ndarray, z: np.ndarray) -> None:
        self.biases[self.current_context] += self.eta * np.asarray(error, dtype=float)


class RLSReadout:
    """Linear readout with recursive-least-squares (FORCE) update (§5.2)."""

    def __init__(
        self,
        output_dim: int,
        feature_dim: int,
        *,
        beta: float = 0.999,
        p0: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        self.beta = beta
        self.p0 = p0
        self.J = np.zeros((output_dim, feature_dim), dtype=float)
        # P0 = p0 * I  (formulation P_0 = alpha^{-1} I; p0 is that diagonal).
        self.P = p0 * np.eye(feature_dim, dtype=float)
        # plasticity diagnostics (Phase 2C)
        self.last_dJ_norm = 0.0
        self.last_gain_norm = 0.0
        self.cumulative_update_norm = 0.0

    @classmethod
    def from_weights(
        cls, J: np.ndarray, *, beta: float = 0.999, p0: float = 1.0
    ) -> "RLSReadout":
        """Build an RLS readout initialised to J0 (output_dim, feature_dim)."""
        J = np.asarray(J, dtype=float)
        ro = cls(J.shape[0], J.shape[1], beta=beta, p0=p0)
        ro.J = J.copy()
        return ro

    def copy(self) -> "RLSReadout":
        return _copy.deepcopy(self)

    def predict(self, z: np.ndarray) -> np.ndarray:
        """Readout output J @ z, shape (output_dim,)."""
        return self.J @ np.asarray(z, dtype=float)

    def update(self, error: np.ndarray, z: np.ndarray) -> None:
        """In-place RLS/FORCE update with forgetting factor beta.

        Standard explicit-gain form equivalent to formulation §5.2:
            Pz    = P z
            denom = beta + z^T P z
            k     = Pz / denom
            J    += outer(error, k)
            P     = (P - outer(Pz, Pz) / denom) / beta
        ``error`` follows the desired-minus-actual convention.
        """
        error = np.asarray(error, dtype=float)
        z = np.asarray(z, dtype=float)
        Pz = self.P @ z
        denom = self.beta + z @ Pz
        k = Pz / denom
        dJ = np.outer(error, k)
        self.J += dJ
        self.P = (self.P - np.outer(Pz, Pz) / denom) / self.beta
        self.last_dJ_norm = float(np.linalg.norm(dJ))
        self.last_gain_norm = float(np.linalg.norm(k))
        self.cumulative_update_norm += self.last_dJ_norm

    # ------------------------------------------------------------------
    # Covariance diagnostics and safe-forgetting controls (Phase 2C)
    # ------------------------------------------------------------------
    def pop_cumulative_update_norm(self) -> float:
        """Return the accumulated ||ΔJ|| since the last call, then reset it."""
        v = self.cumulative_update_norm
        self.cumulative_update_norm = 0.0
        return v

    def trace_P(self) -> float:
        return float(np.trace(self.P))

    def pnorm_fro(self) -> float:
        return float(np.linalg.norm(self.P))

    def eig_extremes(self) -> tuple[float, float]:
        """(min, max) eigenvalue of the symmetric covariance P."""
        w = np.linalg.eigvalsh(0.5 * (self.P + self.P.T))
        return float(w[0]), float(w[-1])

    def reset_covariance(self, p0_reset: float) -> None:
        """Reset P to p0_reset*I (J unchanged): keep the learned model, reopen
        plasticity. The covariance-reset / safe-forgetting control."""
        self.P = p0_reset * np.eye(self.feature_dim, dtype=float)

    def inflate_covariance(self, alpha: float) -> None:
        """Add alpha*I to P: a softer, continuous re-opening of plasticity."""
        self.P = self.P + alpha * np.eye(self.feature_dim, dtype=float)
