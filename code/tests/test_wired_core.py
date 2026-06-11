"""Tests for the WiredCfC (NCP wiring) inverse model (Phase 2 Step 1B). CPU."""

import numpy as np
import torch

from cfc_online_adaptation.core import CfCInverseModel, WiredCfCInverseModel


def test_wired_forward_shapes_state_mode():
    m = WiredCfCInverseModel(10, 64, 2, feature_mode="state")
    x = torch.randn(3, 6, 10)
    tau, z, hx = m(x)
    assert tau.shape == (3, 6, 2)
    assert z.shape == (3, 6, 64)  # full recurrent state as liquid features
    assert hx.shape == (3, 64)
    assert m.d_z == 64


def test_wired_features_dz_matches_readout():
    m = WiredCfCInverseModel(10, 64, 2, feature_mode="state")
    x = torch.randn(2, 4, 10)
    z, _ = m.features(x)
    assert z.shape[-1] == m.d_z == m.readout.in_features == 64


def test_wired_output_mode_dz():
    m = WiredCfCInverseModel(10, 64, 2, feature_mode="output")
    x = torch.randn(2, 4, 10)
    tau, z, _ = m(x)
    assert m.d_z == 2
    assert z.shape == (2, 4, 2)  # wiring motor output
    assert tau.shape == (2, 4, 2)


def test_wired_online_step_continuation():
    m = WiredCfCInverseModel()
    x_seq = torch.randn(1, 5, 10)
    _, _, hx = m(x_seq)
    x1 = torch.randn(1, 1, 10)
    tau, z, hx2 = m(x1, hx)
    assert tau.shape == (1, 1, 2)
    assert z.shape == (1, 1, 64)
    assert hx2.shape == (1, 64)


def test_wired_backward_runs():
    m = WiredCfCInverseModel()
    x = torch.randn(2, 5, 10)
    target = torch.randn(2, 5, 2)
    tau, _, _ = m(x)
    loss = torch.nn.functional.mse_loss(tau, target)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_wired_freeze_export_inject_roundtrip():
    m = WiredCfCInverseModel()
    m.freeze_core()
    assert all(not p.requires_grad for p in m.core.parameters())
    assert m.readout.weight.requires_grad
    J0 = m.export_readout_weights()
    assert J0.shape == (2, 64)
    J_new = np.arange(2 * 64, dtype=np.float32).reshape(2, 64)
    m.load_readout_weights(J_new)
    assert np.allclose(m.export_readout_weights(), J_new)


def test_wired_load_wrong_shape_raises():
    m = WiredCfCInverseModel()
    try:
        m.load_readout_weights(np.zeros((2, 32)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on wrong J shape")


def test_wired_invalid_feature_mode_raises():
    try:
        WiredCfCInverseModel(feature_mode="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad feature_mode")


def test_dense_path_unaffected():
    """Regression: dense CfC model still has the Phase 1 shapes."""
    m = CfCInverseModel(10, 64, 2)
    x = torch.randn(2, 4, 10)
    tau, z, hx = m(x)
    assert tau.shape == (2, 4, 2)
    assert z.shape == (2, 4, 64)
    assert hx.shape == (2, 64)


def test_wired_reproducible_state():
    """Same seed -> identical wiring and forward output."""
    x = torch.randn(1, 3, 10)
    torch.manual_seed(0)
    m1 = WiredCfCInverseModel(seed=123)
    torch.manual_seed(0)
    m2 = WiredCfCInverseModel(seed=123)
    with torch.no_grad():
        t1, _, _ = m1(x)
        t2, _, _ = m2(x)
    assert torch.allclose(t1, t2)
