"""Tests for the after-effect judgment (Step 4C)."""

import numpy as np

from cfc_online_adaptation.analysis import (
    after_effect_decay,
    after_effect_metrics,
    claim_supported,
)


def test_mirror_after_effect_detected():
    # curl pushed +side; catch deflects -side in all 8 dirs -> mirror, ratio 0.5
    curl = np.full(8, 0.024)
    catch = np.full(8, -0.012)
    ae = after_effect_metrics(curl, catch)
    assert ae["mirror"] is True
    assert ae["n_mirror_dirs"] == 8
    assert abs(ae["after_effect_ratio"] - 0.5) < 1e-9


def test_no_mirror_when_same_sign():
    curl = np.full(8, 0.024)
    catch = np.full(8, 0.002)  # same side, tiny
    ae = after_effect_metrics(curl, catch)
    assert ae["mirror"] is False


def test_claim_supported_logic():
    method_ae = {"mirror": True, "after_effect_ratio": 0.45}
    b2_ae = {"mirror": False, "after_effect_ratio": 0.03}
    res = claim_supported(method_ae, b2_ae)
    assert res["claim_supported"] is True

    # weak method -> not supported
    weak = {"mirror": True, "after_effect_ratio": 0.1}
    res2 = claim_supported(weak, b2_ae)
    assert res2["claim_supported"] is False

    # strong B2 mirror -> not supported (method not distinguished)
    strong_b2 = {"mirror": True, "after_effect_ratio": 0.5}
    res3 = claim_supported(method_ae, strong_b2)
    assert res3["claim_supported"] is False


def test_per_direction_mirror_count():
    curl = np.array([0.02, 0.02, 0.02, 0.02])
    catch = np.array([-0.01, -0.01, 0.005, -0.01])  # 3 mirror, 1 same-side
    ae = after_effect_metrics(curl, catch)
    assert ae["n_mirror_dirs"] == 3


def test_after_effect_decay():
    d = after_effect_decay(-22.0, [-18.0, -12.0, -6.0, -3.0, -1.0])
    assert abs(d["decay_ratio"] - (1.0 / 22.0)) < 1e-9
    assert d["last_washout_signed_mm"] == -1.0
    assert len(d["washout_signed_mm_per_cycle"]) == 5
