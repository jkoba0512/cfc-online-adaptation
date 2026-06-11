"""Tests for the B2 feedback-only baseline (Step 4A)."""

import numpy as np

from cfc_online_adaptation.arm import TwoLinkArm
from cfc_online_adaptation.baselines import NullReadout
from cfc_online_adaptation.core import CfCInverseModel
from cfc_online_adaptation.protocol import run_abc_protocol
from cfc_online_adaptation.tasks import joint_space_reaches

KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])


def test_null_readout_predicts_zero_and_no_update():
    ro = NullReadout(2, 64)
    z = np.random.default_rng(0).standard_normal(64)
    assert np.allclose(ro.predict(z), np.zeros(2))
    ro.update(np.array([1.0, -1.0]), z)
    assert np.allclose(ro.J, 0.0)  # never changes


def test_b2_protocol_finite_and_no_learned_after_effect():
    arm = TwoLinkArm()
    center_q = arm.inverse_kinematics(np.array([0.0, 0.45]), elbow="down")
    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=0.5, dt=0.005, n_targets=8)
    model = CfCInverseModel()
    model.freeze_core()
    ro = NullReadout(2, 64)
    J0 = np.zeros((2, 64))
    out = run_abc_protocol(
        arm, model, ro, reaches, KP, KD, J0,
        dt=0.005, field_b=13.0, b_cycles=2, washout_cycles=0,
    )
    assert all(c["all_finite"] for c in out["curves"])
    # B2 carries no state -> J norm stays zero across the protocol
    assert all(c["j_norm"] == 0.0 for c in out["curves"])
    assert out["final_J"].shape == (2, 64)
