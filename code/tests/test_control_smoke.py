"""Fast closed-loop smoke tests (Step 2C). CPU, untrained model, short reach."""

import numpy as np

from cfc_online_adaptation.arm import TwoLinkArm
from cfc_online_adaptation.control import run_closed_loop
from cfc_online_adaptation.core import CfCInverseModel
from cfc_online_adaptation.tasks import joint_space_reaches

KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])


def _setup():
    arm = TwoLinkArm()
    center_q = arm.inverse_kinematics(np.array([0.0, 0.45]), elbow="down")
    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=0.5, dt=0.005, n_targets=8)
    model = CfCInverseModel()
    model.freeze_core()
    return arm, model, reaches


def test_closed_loop_finite_null():
    arm, model, reaches = _setup()
    res = run_closed_loop(arm, model, reaches[0], KP, KD, dt=0.005, field_b=0.0)
    assert res.finite
    assert res.q.shape == res.q_des.shape
    assert np.all(np.isfinite(res.ee))


def test_closed_loop_finite_curl():
    arm, model, reaches = _setup()
    res = run_closed_loop(arm, model, reaches[0], KP, KD, dt=0.005, field_b=13.0)
    assert res.finite
    assert np.all(np.isfinite(res.tau))


def test_closed_loop_tracks_under_pd():
    """Even with an untrained readout, PD feedback should keep the endpoint
    near the desired path (bounded error), confirming the loop is stable."""
    arm, model, reaches = _setup()
    res = run_closed_loop(arm, model, reaches[0], KP, KD, dt=0.005, field_b=0.0)
    err = np.linalg.norm(res.ee - res.ee_des, axis=1)
    assert np.max(err) < 0.10  # within 10 cm of the 10 cm reach path
