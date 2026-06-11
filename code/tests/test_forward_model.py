"""Tests for the forward model and sensory-prediction route (Phase 2D). CPU."""

import numpy as np
import torch

from cfc_online_adaptation.adaptation import run_adaptive_closed_loop
from cfc_online_adaptation.arm import TwoLinkArm
from cfc_online_adaptation.core import CfCInverseModel
from cfc_online_adaptation.forward_model import (
    FWD_INPUT_DIM,
    FWD_OUTPUT_DIM,
    ForwardCfCModel,
    collect_forward_dataset,
    train_forward_model,
)
from cfc_online_adaptation.readout import LinearReadout
from cfc_online_adaptation.tasks import joint_space_reaches

KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])


def _setup():
    arm = TwoLinkArm()
    center_q = arm.inverse_kinematics(np.array([0.0, 0.45]), elbow="down")
    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=0.5, dt=0.005, n_targets=8)
    model = CfCInverseModel()
    model.freeze_core()
    J0 = model.export_readout_weights()
    return arm, model, reaches, J0


def test_forward_model_shapes():
    fwd = ForwardCfCModel()
    x = torch.randn(3, 7, FWD_INPUT_DIM)
    out, hx = fwd(x)
    assert out.shape == (3, 7, FWD_OUTPUT_DIM)
    assert hx.shape == (3, 64)


def test_forward_predict_step():
    fwd = ForwardCfCModel()
    ds, hx = fwd.predict_step(np.zeros(2), np.zeros(2), np.zeros(2), None)
    assert ds.shape == (4,)
    assert np.all(np.isfinite(ds))


def test_collect_and_train_forward_model():
    arm, model, reaches, _ = _setup()
    X, Y = collect_forward_dataset(arm, model, reaches, KP, KD, dt=0.005)
    assert X.shape[2] == FWD_INPUT_DIM == 6
    assert Y.shape[2] == FWD_OUTPUT_DIM == 4
    assert X.shape[1] == Y.shape[1]  # T-1 aligned
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(Y))
    fwd = ForwardCfCModel()
    hist = train_forward_model(fwd, X, Y, epochs=20, device="cpu", seed=0)
    assert hist["val_mse"][-1] < hist["val_mse"][0]  # learns
    assert np.isfinite(hist["onestep_vel_mse"])


def test_sensory_pred_signal_runs_and_changes_J():
    arm, model, reaches, J0 = _setup()
    X, Y = collect_forward_dataset(arm, model, reaches, KP, KD, dt=0.005)
    fwd = ForwardCfCModel()
    train_forward_model(fwd, X, Y, epochs=20, device="cpu", seed=0)
    ro = LinearReadout.from_weights(J0, eta=1e-3)
    res = run_adaptive_closed_loop(
        arm, model, ro, reaches[0], KP, KD, J0, field_b=13.0, adapt=True,
        error_signal="sensory_pred", forward_model=fwd, sensory_sign=1.0, sensory_gain=1.0,
    )
    assert res.finite
    assert res.dj_norm[-1] > 0.0  # J moved


def test_sensory_pred_requires_forward_model():
    arm, model, reaches, J0 = _setup()
    ro = LinearReadout.from_weights(J0, eta=1e-3)
    try:
        run_adaptive_closed_loop(arm, model, ro, reaches[0], KP, KD, J0,
                                 error_signal="sensory_pred")
    except ValueError:
        return
    raise AssertionError("expected ValueError when forward_model is missing")
