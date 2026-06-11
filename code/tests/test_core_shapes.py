"""Shape / gradient / freeze tests for the CfC inverse model (Step 2B). CPU."""

import numpy as np
import torch

from cfc_online_adaptation.core import CfCInverseModel


def test_forward_shape():
    model = CfCInverseModel(input_dim=10, units=64, output_dim=2)
    x = torch.randn(4, 7, 10)
    tau, z, hx = model(x)
    assert tau.shape == (4, 7, 2)
    assert z.shape == (4, 7, 64)
    assert hx.shape == (4, 64)


def test_online_step_continuation():
    model = CfCInverseModel()
    x_seq = torch.randn(1, 5, 10)
    _, _, hx = model(x_seq)
    x1 = torch.randn(1, 1, 10)
    tau, _, hx2 = model(x1, hx)
    assert tau.shape == (1, 1, 2)
    assert hx2.shape == (1, 64)


def test_backward_runs():
    model = CfCInverseModel()
    x = torch.randn(3, 6, 10)
    target = torch.randn(3, 6, 2)
    tau, _, _ = model(x)
    loss = torch.nn.functional.mse_loss(tau, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_freeze_core_and_export_inject():
    model = CfCInverseModel()
    model.freeze_core()
    assert all(not p.requires_grad for p in model.core.parameters())
    # readout stays trainable
    assert model.readout.weight.requires_grad
    J0 = model.export_readout_weights()
    assert J0.shape == (2, 64)
    # inject a new J and read it back
    J_new = np.arange(2 * 64, dtype=np.float32).reshape(2, 64)
    model.load_readout_weights(J_new)
    assert np.allclose(model.export_readout_weights(), J_new)


def test_load_readout_wrong_shape_raises():
    model = CfCInverseModel()
    try:
        model.load_readout_weights(np.zeros((2, 32)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on wrong J shape")
