"""Unit tests for the linear readout and online update rules (Step 1C)."""

import numpy as np

from cfc_online_adaptation.readout import LinearReadout, RLSReadout


def test_predict_shape():
    ro = LinearReadout(output_dim=2, feature_dim=64)
    z = np.ones(64)
    assert ro.predict(z).shape == (2,)


def test_lms_update_formula():
    ro = LinearReadout(output_dim=2, feature_dim=3)
    error = np.array([1.0, -2.0])
    z = np.array([0.5, 0.0, -1.0])
    J0 = ro.J.copy()
    ro.lms_update(error, z, eta=0.1, leakage=0.0)
    assert np.allclose(ro.J, J0 + 0.1 * np.outer(error, z))


def test_lms_update_leakage_formula():
    ro = LinearReadout(output_dim=2, feature_dim=3)
    ro.J = np.ones((2, 3))
    error = np.array([1.0, -2.0])
    z = np.array([0.5, 0.0, -1.0])
    eta, leak = 0.1, 0.5
    expected = (1 - eta * leak) * np.ones((2, 3)) + eta * np.outer(error, z)
    ro.lms_update(error, z, eta=eta, leakage=leak)
    assert np.allclose(ro.J, expected)


def _synthetic_regression(rng, feature_dim=8, output_dim=2):
    J_true = rng.standard_normal((output_dim, feature_dim))
    Z = rng.standard_normal((400, feature_dim))
    Y = Z @ J_true.T
    return J_true, Z, Y


def test_lms_reduces_mse():
    rng = np.random.default_rng(0)
    _, Z, Y = _synthetic_regression(rng)
    ro = LinearReadout(output_dim=2, feature_dim=8)
    mse0 = np.mean((Y[:50] - np.array([ro.predict(z) for z in Z[:50]])) ** 2)
    for z, y in zip(Z, Y):
        err = y - ro.predict(z)
        ro.lms_update(err, z, eta=0.05)
    mse1 = np.mean((Y[-50:] - np.array([ro.predict(z) for z in Z[-50:]])) ** 2)
    assert mse1 < 0.1 * mse0


def test_rls_reduces_mse():
    rng = np.random.default_rng(1)
    _, Z, Y = _synthetic_regression(rng)
    ro = RLSReadout(output_dim=2, feature_dim=8, beta=0.999, p0=1.0)
    mse0 = np.mean((Y[:50] - np.array([ro.predict(z) for z in Z[:50]])) ** 2)
    # MSE drops sharply within a few tens of steps
    for z, y in zip(Z[:40], Y[:40]):
        ro.update(y - ro.predict(z), z)
    mse = np.mean((Y[100:150] - np.array([ro.predict(z) for z in Z[100:150]])) ** 2)
    assert mse < 0.02 * mse0


def test_rls_converges_faster_than_lms():
    """At equal step count RLS should reach a lower MSE than LMS (§5.2)."""
    rng = np.random.default_rng(7)
    _, Z, Y = _synthetic_regression(rng)
    n = 40
    lms = LinearReadout(output_dim=2, feature_dim=8)
    rls = RLSReadout(output_dim=2, feature_dim=8, beta=0.999, p0=1.0)
    for z, y in zip(Z[:n], Y[:n]):
        lms.lms_update(y - lms.predict(z), z, eta=0.05)
        rls.update(y - rls.predict(z), z)
    idx = slice(100, 150)
    mse_lms = np.mean((Y[idx] - np.array([lms.predict(z) for z in Z[idx]])) ** 2)
    mse_rls = np.mean((Y[idx] - np.array([rls.predict(z) for z in Z[idx]])) ** 2)
    assert mse_rls < mse_lms


def test_rls_P_finite_and_shape():
    rng = np.random.default_rng(2)
    _, Z, Y = _synthetic_regression(rng)
    ro = RLSReadout(output_dim=2, feature_dim=8)
    for z, y in zip(Z[:50], Y[:50]):
        ro.update(y - ro.predict(z), z)
    assert ro.P.shape == (8, 8)
    assert np.all(np.isfinite(ro.P))


def test_linear_from_weights_predict():
    J0 = np.random.default_rng(0).standard_normal((2, 8))
    ro = LinearReadout.from_weights(J0, eta=1e-4, leakage=1e-4)
    z = np.random.default_rng(1).standard_normal(8)
    assert np.allclose(ro.predict(z), J0 @ z)
    assert ro.eta == 1e-4 and ro.leakage == 1e-4


def test_rls_from_weights_predict():
    J0 = np.random.default_rng(2).standard_normal((2, 8))
    ro = RLSReadout.from_weights(J0, beta=0.999, p0=1.0)
    z = np.random.default_rng(3).standard_normal(8)
    assert np.allclose(ro.predict(z), J0 @ z)
    assert ro.beta == 0.999 and ro.P.shape == (8, 8)


def test_unified_update_changes_J():
    J0 = np.zeros((2, 8))
    lin = LinearReadout.from_weights(J0, eta=0.1)
    rls = RLSReadout.from_weights(J0)
    z = np.ones(8)
    err = np.array([1.0, -1.0])
    lin.update(err, z)
    rls.update(err, z)
    assert not np.allclose(lin.J, J0)
    assert not np.allclose(rls.J, J0)


def test_copy_is_independent():
    ro = LinearReadout.from_weights(np.ones((2, 8)), eta=0.1)
    clone = ro.copy()
    ro.update(np.array([1.0, 1.0]), np.ones(8))
    assert not np.allclose(ro.J, clone.J)  # clone untouched


def test_rls_covariance_diagnostics():
    ro = RLSReadout.from_weights(np.zeros((2, 8)), beta=1.0, p0=0.5)
    assert abs(ro.trace_P() - 8 * 0.5) < 1e-9
    assert ro.pnorm_fro() > 0
    lo, hi = ro.eig_extremes()
    assert lo > 0 and hi > 0
    ro.update(np.array([1.0, -1.0]), np.ones(8))
    assert ro.last_gain_norm > 0
    assert ro.last_dJ_norm > 0


def test_rls_reset_and_inflate_covariance():
    ro = RLSReadout.from_weights(np.zeros((2, 8)), beta=1.0, p0=1.0)
    for _ in range(30):
        ro.update(np.array([1.0, -1.0]), np.random.default_rng(0).standard_normal(8))
    collapsed = ro.trace_P()
    J_before = ro.J.copy()
    ro.reset_covariance(1.0)
    assert ro.trace_P() > collapsed  # plasticity reopened
    assert np.allclose(ro.J, J_before)  # learned model kept
    t0 = ro.trace_P()
    ro.inflate_covariance(0.5)
    assert ro.trace_P() > t0


def test_cumulative_update_norm_pop():
    ro = LinearReadout.from_weights(np.zeros((2, 8)), eta=0.1)
    ro.update(np.array([1.0, 1.0]), np.ones(8))
    ro.update(np.array([1.0, 1.0]), np.ones(8))
    v = ro.pop_cumulative_update_norm()
    assert v > 0
    assert ro.pop_cumulative_update_norm() == 0.0  # reset after pop
