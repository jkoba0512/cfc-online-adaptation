"""Unit tests for the center-out task generation (Step 1B acceptance)."""

import numpy as np

from cfc_online_adaptation.arm import TwoLinkArm
from cfc_online_adaptation.tasks import (
    center_out_targets,
    joint_space_reaches,
    minimum_jerk_scalar,
    minimum_jerk_trajectory,
)


def test_minimum_jerk_endpoints():
    assert np.isclose(minimum_jerk_scalar(0.0, 0.5), 0.0)
    assert np.isclose(minimum_jerk_scalar(0.5, 0.5), 1.0)
    # zero velocity/accel at endpoints -> symmetric, monotone, passes 0.5 at mid
    assert np.isclose(minimum_jerk_scalar(0.25, 0.5), 0.5)


def test_minimum_jerk_clamped_outside():
    assert np.isclose(minimum_jerk_scalar(-1.0, 0.5), 0.0)
    assert np.isclose(minimum_jerk_scalar(1.0, 0.5), 1.0)


def test_minimum_jerk_trajectory_endpoints():
    p0 = np.array([0.0, 0.3])
    p1 = np.array([0.1, 0.4])
    times, pos = minimum_jerk_trajectory(p0, p1, T=0.5, dt=0.005)
    assert np.allclose(pos[0], p0)
    assert np.allclose(pos[-1], p1)
    assert np.all(np.isfinite(pos))
    assert times[0] == 0.0 and np.isclose(times[-1], 0.5)


def test_center_out_targets_on_circle():
    center = np.array([0.0, 0.4])
    radius = 0.10
    targets = center_out_targets(center, radius=radius, n_targets=8)
    assert targets.shape == (8, 2)
    dists = np.linalg.norm(targets - center, axis=1)
    assert np.allclose(dists, radius)


def test_joint_space_reaches_shapes_and_finite():
    arm = TwoLinkArm()
    # a comfortable center pose (elbow down), well inside the workspace
    center_q = arm.inverse_kinematics(np.array([0.0, 0.45]), elbow="down")
    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=0.5, dt=0.005)
    assert len(reaches) == 8
    for r in reaches:
        n = r.q_des.shape[0]
        assert r.qdot_des.shape == (n, 2)
        assert r.qddot_des.shape == (n, 2)
        assert np.all(np.isfinite(r.q_des))
        assert np.all(np.isfinite(r.qdot_des))
        assert np.all(np.isfinite(r.qddot_des))
        # the desired joint trajectory starts at the center pose
        assert np.allclose(r.q_des[0], center_q, atol=1e-9)


def test_joint_space_reaches_endpoint_matches_target():
    arm = TwoLinkArm()
    center_q = arm.inverse_kinematics(np.array([0.0, 0.45]), elbow="down")
    reaches = joint_space_reaches(arm, center_q, radius=0.10, T=0.5, dt=0.005)
    for r in reaches:
        assert np.allclose(arm.fk(r.q_des[-1]), r.target_xy, atol=1e-9)
