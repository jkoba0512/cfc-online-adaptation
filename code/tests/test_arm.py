"""Unit tests for the 2-link arm dynamics (Step 1A acceptance criteria)."""

import numpy as np
import pytest

from cfc_online_adaptation.arm import ArmParams, TwoLinkArm

POSES = [
    np.array([0.3, 0.5]),
    np.array([-0.4, 1.2]),
    np.array([1.0, -0.7]),
    np.array([0.0, 0.1]),
    np.array([1.57, 0.9]),
]


@pytest.fixture
def arm():
    return TwoLinkArm(ArmParams())


def test_mass_matrix_symmetric(arm):
    for q in POSES:
        M = arm.mass_matrix(q)
        assert np.allclose(M, M.T)


def test_mass_matrix_positive_definite(arm):
    for q in POSES:
        eig = np.linalg.eigvalsh(arm.mass_matrix(q))
        assert np.all(eig > 0.0)


def test_inverse_forward_dynamics_roundtrip(arm):
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-1.5, 1.5, size=2)
        qdot = rng.uniform(-2.0, 2.0, size=2)
        qddot = rng.uniform(-5.0, 5.0, size=2)
        tau = arm.inverse_dynamics(q, qdot, qddot)
        qddot_back = arm.dynamics(q, qdot, tau)
        assert np.allclose(qddot, qddot_back, atol=1e-9)


def test_inverse_forward_dynamics_roundtrip_with_field(arm):
    """Roundtrip must also hold when a curl field acts on the plant."""
    rng = np.random.default_rng(1)
    b = 13.0
    for _ in range(20):
        q = rng.uniform(-1.5, 1.5, size=2)
        qdot = rng.uniform(-2.0, 2.0, size=2)
        qddot = rng.uniform(-5.0, 5.0, size=2)
        # actuation that, together with the field, produces qddot
        F = arm.curl_force(q, qdot, b=b)
        tau = arm.inverse_dynamics(q, qdot, qddot, field_force=F)
        qddot_back = arm.dynamics(q, qdot, tau, field_b=b)
        assert np.allclose(qddot, qddot_back, atol=1e-9)


def test_jacobian_finite(arm):
    rng = np.random.default_rng(2)
    for _ in range(10):
        q = rng.uniform(-2.0, 2.0, size=2)
        qdot = rng.uniform(-2.0, 2.0, size=2)
        v = arm.jacobian(q) @ qdot
        assert np.all(np.isfinite(v))


def test_fk_ik_roundtrip(arm):
    rng = np.random.default_rng(3)
    for _ in range(30):
        q = np.array([rng.uniform(-1.0, 1.5), rng.uniform(-2.0, -0.2)])  # elbow down
        xy = arm.fk(q)
        q_back = arm.inverse_kinematics(xy, elbow="down")
        assert np.allclose(arm.fk(q_back), xy, atol=1e-9)


def test_ik_elbow_conventions(arm):
    xy = np.array([0.35, 0.15])
    q_down = arm.inverse_kinematics(xy, elbow="down")
    q_up = arm.inverse_kinematics(xy, elbow="up")
    assert q_down[1] <= 0.0
    assert q_up[1] >= 0.0
    assert np.allclose(arm.fk(q_down), xy, atol=1e-9)
    assert np.allclose(arm.fk(q_up), xy, atol=1e-9)


def test_ik_unreachable_raises(arm):
    with pytest.raises(ValueError):
        arm.inverse_kinematics([10.0, 10.0])


def test_rk4_step_finite(arm):
    q = np.array([0.5, 0.8])
    qdot = np.array([0.1, -0.2])
    tau = np.array([0.3, 0.1])
    q1, v1 = arm.rk4_step(q, qdot, tau, dt=0.005)
    assert np.all(np.isfinite(q1)) and np.all(np.isfinite(v1))


def test_rk4_tiny_dt_small_change(arm):
    q = np.array([0.5, 0.8])
    qdot = np.array([0.1, -0.2])
    tau = np.array([0.3, 0.1])
    q1, v1 = arm.rk4_step(q, qdot, tau, dt=1e-9)
    assert np.allclose(q1, q, atol=1e-7)
    assert np.allclose(v1, qdot, atol=1e-7)


def test_damping_vector(arm):
    qdot = np.array([1.0, -2.0])
    assert np.allclose(arm.damping_vector(qdot), arm.p.b * qdot)
