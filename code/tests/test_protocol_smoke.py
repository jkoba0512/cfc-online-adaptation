"""Smoke test for the A/B/C protocol (Step 3C). CPU, tiny cycle count."""

import numpy as np

from cfc_online_adaptation.arm import ArmParams, TwoLinkArm
from cfc_online_adaptation.core import CfCInverseModel
from cfc_online_adaptation.protocol import run_abc_protocol
from cfc_online_adaptation.readout import LinearReadout
from cfc_online_adaptation.tasks import joint_space_reaches

KP = np.diag([35.0, 25.0])
KD = np.diag([3.5, 2.5])


def test_abc_protocol_structure_and_finite():
    arm = TwoLinkArm()
    center_q = arm.inverse_kinematics(np.array([0.0, 0.45]), elbow="down")
    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=0.5, dt=0.005, n_targets=8)
    model = CfCInverseModel()
    model.freeze_core()
    J0 = model.export_readout_weights()
    ro = LinearReadout.from_weights(J0, eta=1e-4, leakage=1e-4)

    out = run_abc_protocol(
        arm, model, ro, reaches, KP, KD, J0,
        dt=0.005, field_b=13.0, b_cycles=2, washout_cycles=1,
    )
    phases = [c["phase"] for c in out["curves"]]
    assert phases == ["A", "B", "B", "C_catch", "C_washout"]
    assert all(c["all_finite"] for c in out["curves"])
    assert len(out["order"]) == 8
    assert out["final_J"].shape == (2, 64)
    # adaptation moved J away from J0 by the end of B
    assert out["curves"][2]["dj_norm"] > 0.0
    # after-effect block present with 8 per-direction signed deviations
    assert len(out["after_effect"]["signed_perp_dev_per_dir"]) == 8
    for key in ("A", "B_first", "B_last", "C_catch"):
        assert key in out["saved_trajectories"]


def test_abc_protocol_payload_arm_b():
    """B phase can run on a payload (heavier m2) arm via arm_b; finite + structured."""
    arm = TwoLinkArm()
    payload = TwoLinkArm(ArmParams(m2=1.1 * 1.5))  # +50% forearm mass during B
    center_q = arm.inverse_kinematics(np.array([0.0, 0.45]), elbow="down")
    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=0.5, dt=0.005, n_targets=8)
    model = CfCInverseModel()
    model.freeze_core()
    J0 = model.export_readout_weights()
    ro = LinearReadout.from_weights(J0, eta=1e-3)
    out = run_abc_protocol(
        arm, model, ro, reaches, KP, KD, J0,
        dt=0.005, field_b=0.0, b_cycles=2, washout_cycles=0, arm_b=payload,
    )
    assert all(c["all_finite"] for c in out["curves"])
    assert [c["phase"] for c in out["curves"]] == ["A", "B", "B", "C_catch"]
