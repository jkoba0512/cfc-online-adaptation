"""2-link planar arm in the horizontal plane (no gravity).

Implements the rigid-body dynamics fixed in the project formulation
(``cfc-online-adaptation_formulation_v0`` §6.1):

    M(q) qddot + C(q, qdot) qdot + B qdot = tau + tau_field
    tau_field = J(q)^T F

where ``B qdot`` is linear joint viscous damping and ``F`` is an optional
external end-effector force (the velocity-dependent curl field, §6.2).

Sign convention:
    * All torques on the RHS are external/actuation torques that *add* to the
      acceleration:  qddot = M^{-1} (tau + tau_field - C qdot - B qdot).
    * ``inverse_dynamics`` is the exact algebraic inverse of ``dynamics``:
      given (q, qdot, qddot) it returns the tau that produces that qddot,
      subtracting any field torque so that the *commanded* actuation is
      returned (teacher torque for the null-field inverse model is obtained
      with ``field_force=None``).

Pure numpy; torch is intentionally not used at this layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayLike = np.ndarray


@dataclass(frozen=True)
class ArmParams:
    """Inertial / geometric parameters (formulation §6.1).

    Defaults are the fixed Phase 1 values. Lengths in metres, masses in kg,
    inertias in kg*m^2 (about each link COM), damping in N*m*s/rad.
    """

    l1: float = 0.30
    l2: float = 0.33
    m1: float = 1.6
    m2: float = 1.1
    lc1: float = 0.11
    lc2: float = 0.16
    I1: float = 0.025
    I2: float = 0.045
    b: float = 0.05  # joint viscous damping (same for both joints)


class TwoLinkArm:
    """Planar 2-link arm dynamics (horizontal plane, gravity-free)."""

    def __init__(self, params: ArmParams | None = None):
        self.p = params if params is not None else ArmParams()
        # Precompute the constant inertial lumped parameters.
        p = self.p
        # a1 = I1 + I2 + m1 lc1^2 + m2 (l1^2 + lc2^2)
        self._a1 = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2)
        # a2 = m2 l1 lc2  (coupling term)
        self._a2 = p.m2 * p.l1 * p.lc2
        # a3 = I2 + m2 lc2^2
        self._a3 = p.I2 + p.m2 * p.lc2**2

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    def fk(self, q: ArrayLike) -> np.ndarray:
        """Forward kinematics: joint angles -> end-effector (x, y)."""
        q = np.asarray(q, dtype=float)
        q1, q2 = q[0], q[1]
        l1, l2 = self.p.l1, self.p.l2
        x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        return np.array([x, y])

    def jacobian(self, q: ArrayLike) -> np.ndarray:
        """Geometric Jacobian J(q): end-effector velocity = J(q) @ qdot."""
        q = np.asarray(q, dtype=float)
        q1, q2 = q[0], q[1]
        l1, l2 = self.p.l1, self.p.l2
        s1, s12 = np.sin(q1), np.sin(q1 + q2)
        c1, c12 = np.cos(q1), np.cos(q1 + q2)
        return np.array(
            [
                [-l1 * s1 - l2 * s12, -l2 * s12],
                [l1 * c1 + l2 * c12, l2 * c12],
            ]
        )

    def inverse_kinematics(self, xy: ArrayLike, elbow: str = "down") -> np.ndarray:
        """Inverse kinematics for a reachable (x, y).

        ``elbow="down"`` selects q2 <= 0, ``elbow="up"`` selects q2 >= 0.
        Raises ValueError if the point is outside the reachable annulus.
        """
        xy = np.asarray(xy, dtype=float)
        x, y = xy[0], xy[1]
        l1, l2 = self.p.l1, self.p.l2
        r2 = x * x + y * y
        c2 = (r2 - l1**2 - l2**2) / (2.0 * l1 * l2)
        if c2 < -1.0 - 1e-9 or c2 > 1.0 + 1e-9:
            raise ValueError(f"target {xy} is outside the reachable workspace")
        c2 = float(np.clip(c2, -1.0, 1.0))
        if elbow == "down":
            sign = -1.0
        elif elbow == "up":
            sign = 1.0
        else:
            raise ValueError("elbow must be 'down' or 'up'")
        q2 = sign * np.arccos(c2)
        s2 = np.sin(q2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * s2, l1 + l2 * c2)
        return np.array([q1, q2])

    # ------------------------------------------------------------------
    # Dynamics terms
    # ------------------------------------------------------------------
    def mass_matrix(self, q: ArrayLike) -> np.ndarray:
        """Symmetric positive-definite mass matrix M(q)."""
        q = np.asarray(q, dtype=float)
        c2 = np.cos(q[1])
        m11 = self._a1 + 2.0 * self._a2 * c2
        m12 = self._a3 + self._a2 * c2
        m22 = self._a3
        return np.array([[m11, m12], [m12, m22]])

    def coriolis_vector(self, q: ArrayLike, qdot: ArrayLike) -> np.ndarray:
        """Coriolis/centrifugal torque vector C(q, qdot) @ qdot.

        h = a2 sin(q2);  returns [-h(2 q1d q2d + q2d^2), h q1d^2].
        """
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        h = self._a2 * np.sin(q[1])
        q1d, q2d = qdot[0], qdot[1]
        c1 = -h * (2.0 * q1d * q2d + q2d**2)
        c2 = h * q1d**2
        return np.array([c1, c2])

    def damping_vector(self, qdot: ArrayLike) -> np.ndarray:
        """Linear joint viscous damping B @ qdot."""
        qdot = np.asarray(qdot, dtype=float)
        return self.p.b * qdot

    # ------------------------------------------------------------------
    # Force field
    # ------------------------------------------------------------------
    def curl_force(self, q: ArrayLike, qdot: ArrayLike, b: float = 13.0) -> np.ndarray:
        """Velocity-dependent curl force at the end-effector (§6.2).

        F = [[0, b], [-b, 0]] @ v, with v = J(q) @ qdot the end-effector
        velocity. Returns the 2-vector F = [b*vy, -b*vx].
        """
        v = self.jacobian(q) @ np.asarray(qdot, dtype=float)
        return np.array([b * v[1], -b * v[0]])

    def field_torque(self, q: ArrayLike, field_force: ArrayLike) -> np.ndarray:
        """Map an end-effector force to joint torque: tau_field = J(q)^T F."""
        return self.jacobian(q).T @ np.asarray(field_force, dtype=float)

    # ------------------------------------------------------------------
    # Inverse / forward dynamics
    # ------------------------------------------------------------------
    def inverse_dynamics(
        self,
        q: ArrayLike,
        qdot: ArrayLike,
        qddot: ArrayLike,
        field_force: ArrayLike | None = None,
    ) -> np.ndarray:
        """Actuation torque that realises qddot at (q, qdot).

        tau = M qddot + C qdot + B qdot - tau_field.
        With ``field_force=None`` this is the null-field inverse dynamics used
        as the teacher torque for offline CfC inverse-model training (§3).
        """
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        qddot = np.asarray(qddot, dtype=float)
        tau = (
            self.mass_matrix(q) @ qddot
            + self.coriolis_vector(q, qdot)
            + self.damping_vector(qdot)
        )
        if field_force is not None:
            tau = tau - self.field_torque(q, field_force)
        return tau

    def dynamics(
        self,
        q: ArrayLike,
        qdot: ArrayLike,
        tau: ArrayLike,
        field_b: float = 0.0,
    ) -> np.ndarray:
        """Forward dynamics: joint acceleration qddot given actuation tau.

        qddot = M^{-1} (tau + tau_field - C qdot - B qdot).
        ``field_b`` > 0 enables the curl field with that strength.
        """
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        tau = np.asarray(tau, dtype=float)
        rhs = tau - self.coriolis_vector(q, qdot) - self.damping_vector(qdot)
        if field_b != 0.0:
            rhs = rhs + self.field_torque(q, self.curl_force(q, qdot, b=field_b))
        qddot = np.linalg.solve(self.mass_matrix(q), rhs)
        return qddot

    def rk4_step(
        self,
        q: ArrayLike,
        qdot: ArrayLike,
        tau: ArrayLike,
        dt: float,
        field_b: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One RK4 integration step of the arm under constant tau over dt.

        Returns the next (q, qdot). ``tau`` is held constant across the step.
        """
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        tau = np.asarray(tau, dtype=float)

        def deriv(qs, vs):
            return vs, self.dynamics(qs, vs, tau, field_b=field_b)

        k1q, k1v = deriv(q, qdot)
        k2q, k2v = deriv(q + 0.5 * dt * k1q, qdot + 0.5 * dt * k1v)
        k3q, k3v = deriv(q + 0.5 * dt * k2q, qdot + 0.5 * dt * k2v)
        k4q, k4v = deriv(q + dt * k3q, qdot + dt * k3v)
        q_next = q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
        qdot_next = qdot + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return q_next, qdot_next
