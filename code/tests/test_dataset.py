"""Unit tests for the supervised inverse-model dataset (Step 2A)."""

import numpy as np

from cfc_online_adaptation.arm import TwoLinkArm
from cfc_online_adaptation.dataset import (
    INPUT_DIM,
    TARGET_DIM,
    build_reach_sequences,
    train_val_split,
)


def _ds():
    arm = TwoLinkArm()
    center_q = arm.inverse_kinematics(np.array([0.0, 0.45]), elbow="down")
    return build_reach_sequences(
        arm, center_q, radii=(0.08, 0.10, 0.12), durations=(0.5,), dt=0.005, n_targets=8
    )


def test_dataset_shapes():
    ds = _ds()
    assert ds.x.shape[0] == 24  # 8 targets x 3 radii
    assert ds.x.shape[2] == INPUT_DIM == 10
    assert ds.tau.shape[2] == TARGET_DIM == 2
    assert ds.x.shape[1] == ds.tau.shape[1]


def test_dataset_finite():
    ds = _ds()
    assert np.all(np.isfinite(ds.x))
    assert np.all(np.isfinite(ds.tau))


def test_teacher_forced_input_structure():
    """x_t = [q*, qdot*, qddot*, q*, qdot*]: cols 0:2 == cols 6:8, 2:4 == 8:10."""
    ds = _ds()
    assert np.allclose(ds.x[..., 0:2], ds.x[..., 6:8])
    assert np.allclose(ds.x[..., 2:4], ds.x[..., 8:10])


def test_train_val_split_reproducible():
    ds = _ds()
    tr1, va1 = train_val_split(ds, val_frac=0.25, seed=42)
    tr2, va2 = train_val_split(ds, val_frac=0.25, seed=42)
    assert len(tr1) + len(va1) == len(ds)
    assert np.allclose(tr1.x, tr2.x)
    assert np.allclose(va1.x, va2.x)
    # different seed -> generally different val set membership
    _, va3 = train_val_split(ds, val_frac=0.25, seed=7)
    assert va3.x.shape == va1.x.shape


def test_train_val_disjoint():
    ds = _ds()
    tr, va = train_val_split(ds, val_frac=0.25, seed=1)
    # no validation sequence should appear in train (compare on tau signatures)
    tr_keys = {r["target_xy"].tobytes() + bytes(str(r["radius"]), "utf8") for r in tr.meta}
    va_keys = {r["target_xy"].tobytes() + bytes(str(r["radius"]), "utf8") for r in va.meta}
    assert tr_keys.isdisjoint(va_keys)
