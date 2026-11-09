"""Predeclared validity rule for the revision's online eta selection."""

from __future__ import annotations

import math


def validate_ab_curves(curves: list[dict]) -> list[str]:
    """Selection validity uses only A0 and B0..29, never catch or washout."""
    ab = [c for c in curves if c.get("phase") in {"A", "B"}]
    expected = {("A", 0)} | {("B", i) for i in range(30)}
    actual = [(c.get("phase"), c.get("cycle")) for c in ab]
    reasons = []
    if len(actual) != 31 or set(actual) != expected or len(set(actual)) != 31:
        reasons.append("ab_cycle_structure")
    if not all(c.get("all_finite") is True for c in ab):
        reasons.append("ab_nonfinite_reach")
    if not all(math.isfinite(c.get("mean_max_perp_dev", math.nan)) for c in ab):
        reasons.append("ab_nonfinite_metric")
    if reasons:
        return reasons
    b = {c["cycle"]: c["mean_max_perp_dev"] for c in ab if c["phase"] == "B"}
    if not b[29] < b[0]:
        reasons.append("no_b_improvement")
    if any(v > 2.0 * b[0] for v in b.values()):
        reasons.append("b_transient_above_2x_first")
    return reasons
