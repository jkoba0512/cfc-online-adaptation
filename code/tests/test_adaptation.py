"""Smoke tests for the adaptive closed loop (Step 3B). CPU, short."""

import numpy as np

from cfc_online_adaptation.adaptation import run_adaptive_closed_loop
from cfc_online_adaptation.arm import TwoLinkArm
from cfc_online_adaptation.core import CfCInverseModel
from cfc_online_adaptation.readout import LinearReadout, RLSReadout
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


def test_adaptive_loop_finite_no_adapt():
    arm, model, reaches, J0 = _setup()
    ro = LinearReadout.from_weights(J0, eta=1e-4, leakage=1e-4)
    res = run_adaptive_closed_loop(
        arm, model, ro, reaches[0], KP, KD, J0, dt=0.005, field_b=13.0, adapt=False
    )
    assert res.finite
    # no adaptation -> J unchanged -> dj_norm stays zero
    assert np.allclose(res.dj_norm, 0.0)


def test_adaptive_loop_changes_J_when_adapting():
    arm, model, reaches, J0 = _setup()
    ro = LinearReadout.from_weights(J0, eta=1e-4, leakage=1e-4)
    res = run_adaptive_closed_loop(
        arm, model, ro, reaches[0], KP, KD, J0, dt=0.005, field_b=13.0, adapt=True
    )
    assert res.finite
    assert res.dj_norm[-1] > 0.0  # J moved away from J0
    assert np.all(np.isfinite(res.j_norm))


def test_adaptive_loop_rls_finite():
    arm, model, reaches, J0 = _setup()
    ro = RLSReadout.from_weights(J0, beta=0.999, p0=1.0)
    res = run_adaptive_closed_loop(
        arm, model, ro, reaches[0], KP, KD, J0, dt=0.005, field_b=13.0, adapt=True
    )
    assert res.finite
    assert np.all(np.isfinite(res.tau))


def test_tracking_error_signal_runs_and_differs():
    arm, model, reaches, J0 = _setup()
    fel = LinearReadout.from_weights(J0, eta=1e-3)
    trk = LinearReadout.from_weights(J0, eta=1e-3)
    r_fel = run_adaptive_closed_loop(
        arm, model, fel, reaches[1], KP, KD, J0, field_b=13.0, adapt=True, error_signal="fel"
    )
    r_trk = run_adaptive_closed_loop(
        arm, model, trk, reaches[1], KP, KD, J0, field_b=13.0, adapt=True, error_signal="tracking"
    )
    assert r_fel.finite and r_trk.finite
    # different climbing-fiber signals -> different learned J
    assert not np.allclose(fel.J, trk.J)


def test_invalid_error_signal_raises():
    arm, model, reaches, J0 = _setup()
    ro = LinearReadout.from_weights(J0, eta=1e-3)
    try:
        run_adaptive_closed_loop(arm, model, ro, reaches[0], KP, KD, J0, error_signal="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad error_signal")
