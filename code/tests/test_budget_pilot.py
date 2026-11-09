"""Tests for the training-budget pilot (three CfC implementation configurations)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cfc_online_adaptation import budget_pilot as bp
from cfc_online_adaptation.experiment import train_inverse_model

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def data():
    arm, cq = bp.make_arm_and_center()
    ds, audit = bp.training_dataset(arm, cq), bp.audit_dataset(arm, cq)
    tr, va = bp.split_for_seed(ds, 100)
    return ds, audit, tr, va


def test_three_configurations_structure():
    d = bp.build_model("dense_backbone", 100, None)
    f = bp.build_model("fc_wired", 100, 22300)
    n = bp.build_model("ncp_wired", 100, 22300)
    x = torch.randn(2, 5, 10)
    for m in (d, f, n):
        tau, z, hx = m(x)
        assert z.shape == (2, 5, 64) and tau.shape == (2, 5, 2) and m.readout.bias is None
    ad, af, an = (bp.audit_model(m) for m in (d, f, n))
    assert ad["structure"]["backbone"] == {"units": 128, "layers": 1} and ad["masks"] == []
    assert af["structure"]["layer_sizes"] == [64] and af["masks"][0]["nonzero"] == af["masks"][0]["elements"] == 64 * 74
    assert an["structure"]["layer_sizes"] == [38, 24, 2] and len(an["masks"]) == 3
    assert all(m["recurrent_block_all_ones"] for m in an["masks"] + af["masks"])
    assert all(not m["requires_grad"] for m in an["masks"] + af["masks"])
    assert (ad["totals"]["core"]["total"], ad["totals"]["core"]["trainable"]) == (42624, 42624)
    assert (af["totals"]["core"]["total"], af["totals"]["core"]["trainable"]) == (23936, 19200)
    assert (an["totals"]["core"]["total"], an["totals"]["core"]["trainable"]) == (17076, 13712)
    assert [m["nonzero"] for m in an["masks"]] == [1634, 1032, 33]
    assert all(t["totals"]["readout"]["total"] == 128 for t in (ad, af, an))


def test_fc_polarity_seed_does_not_change_effective_mask():
    a = bp.build_model("fc_wired", 100, 1111)
    b = bp.build_model("fc_wired", 100, 22300)
    assert bp.masks_equal(a, b)


def test_split_shared_and_audit_disjoint(data):
    ds, audit, tr, va = data
    tr2, va2 = bp.split_for_seed(ds, 100)
    assert np.array_equal(tr.x, tr2.x) and np.array_equal(va.x, va2.x)
    assert len(tr) == 18 and len(va) == 6 and len(audit) == 16
    assert {m["radius"] for m in audit.meta} == {0.09, 0.11}
    assert not ({m["radius"] for m in audit.meta} & {m["radius"] for m in ds.meta})
    assert sorted(set(bp.direction_index(audit).tolist())) == list(range(8))
    assert sorted(set(bp.direction_index(ds).tolist())) == list(range(8))


def test_loop_matches_train_inverse_model(data, tmp_path):
    ds, audit, tr, va = data
    ref = bp.build_model("ncp_wired", 100, 22300)
    hist = train_inverse_model(ref, tr, va, epochs=4, lr=bp.LR, seed=100)
    model = bp.build_model("ncp_wired", 100, 22300)
    bp.train_series(model, tr, va, audit, epochs=4, seed=100, out_dir=tmp_path / "s",
                    ckpt_every=2, config={"c": 1})
    curve = [json.loads(line) for line in (tmp_path / "s" / "curve.jsonl").read_text().splitlines()]
    assert [c["train_mse"] for c in curve] == hist["train_mse"]
    assert [c["val_mse"] for c in curve] == hist["val_mse"]
    assert all(torch.equal(p, q) for p, q in zip(ref.state_dict().values(), model.state_dict().values()))


def test_checkpoint_rule_resume_and_atomicity(data, tmp_path):
    ds, audit, tr, va = data
    cfg = {"c": 2}
    m1 = bp.build_model("fc_wired", 101, 22301)
    r1 = bp.train_series(m1, tr, va, audit, epochs=6, seed=101, out_dir=tmp_path / "a", ckpt_every=2, config=cfg)
    assert r1["saved_epochs"] == [2, 4, 6]
    assert not list((tmp_path / "a").glob("*.tmp"))
    m2 = bp.build_model("fc_wired", 101, 22301)
    bp.train_series(m2, tr, va, audit, epochs=3, seed=101, out_dir=tmp_path / "b", ckpt_every=2, config=cfg)
    m3 = bp.build_model("fc_wired", 101, 22301)
    bp.train_series(m3, tr, va, audit, epochs=6, seed=101, out_dir=tmp_path / "b", ckpt_every=2, config=cfg, resume=True)
    ca = [json.loads(line) for line in (tmp_path / "a" / "curve.jsonl").read_text().splitlines()]
    cb = [json.loads(line) for line in (tmp_path / "b" / "curve.jsonl").read_text().splitlines()]
    assert len(ca) == len(cb) == 6
    assert [c["train_mse"] for c in ca] == [c["train_mse"] for c in cb]
    assert all(torch.equal(p, q) for p, q in zip(m1.state_dict().values(), m3.state_dict().values()))
    # incomplete / temporary files are rejected
    bad = tmp_path / "a" / "broken.pt"
    torch.save({"epoch": 1}, bad)
    with pytest.raises(ValueError):
        bp.load_checkpoint(bad)
    with pytest.raises(ValueError):
        bp.load_checkpoint(tmp_path / "a" / "x.pt.tmp")
    # refuses to overwrite an existing series without resume
    with pytest.raises(FileExistsError):
        bp.train_series(bp.build_model("fc_wired", 101, 22301), tr, va, audit, epochs=2, seed=101,
                        out_dir=tmp_path / "a", ckpt_every=2, config=cfg)


def test_reload_check_against_saved_model_and_negative_case(data, tmp_path):
    ds, audit, tr, va = data
    m = bp.build_model("ncp_wired", 102, 22302)
    bp.train_series(m, tr, va, audit, epochs=4, seed=102, out_dir=tmp_path / "r", ckpt_every=2, config={"c": 3})
    build = lambda: bp.build_model("ncp_wired", 102, 22302)  # noqa: E731
    rc = bp.reload_check(build, tmp_path / "r" / "epoch0004.pt", va.x)
    assert rc["input_matches_saved_input"] and rc["forward_bit_identical_to_saved_model"]
    assert rc["masks_equal_to_saved_digests"] and rc["masks_equal_after_reload"]
    # the live model that was saved must agree with the freshly loaded one
    with torch.no_grad():
        live = m(torch.from_numpy(va.x.astype(np.float32)))[0].numpy().astype(np.float32)
    assert np.array_equal(live, bp.load_checkpoint(tmp_path / "r" / "epoch0004.pt")["verification"]["val_pred"])
    # a different checkpoint must be detected
    bad = bp.reload_check(build, tmp_path / "r" / "epoch0002.pt", va.x, reference_ckpt_path=tmp_path / "r" / "epoch0004.pt")
    assert not bad["forward_bit_identical_to_saved_model"] and bad["forward_max_abs_diff_to_saved_model"] > 0
    # a different input is detected
    assert not bp.reload_check(build, tmp_path / "r" / "epoch0004.pt", audit.x)["input_matches_saved_input"]
    ev = bp.evaluate(m, audit)
    assert ev["n_sequences"] == 16 and len(ev["nmse_by_direction"]) == 8 and ev["finite"]
    assert ev["nmse"] == pytest.approx(ev["mse"] / float(audit.tau.var()))


def _series(tmp, name, epochs, resume=False, **kw):
    ds, audit, tr, va = kw["data"]
    m = bp.build_model("fc_wired", 100, 22300)
    res = bp.train_series(m, tr, va, audit, epochs=epochs, seed=100, out_dir=tmp / name, ckpt_every=2,
                          config={"c": 9}, resume=resume, allow_restart_unsaved=kw.get("restart", False))
    return m, res


def _curve(d):
    return [json.loads(line) for line in (d / "curve.jsonl").read_text().splitlines() if line.strip()]


def test_interruption_between_checkpoint_and_latest(data, tmp_path):
    """Crash after curve row 6 and epoch0006.pt were written but before latest.pt (= epoch 4)."""
    ref, _ = _series(tmp_path, "ref", 8, data=data)
    m, _ = _series(tmp_path, "x", 6, data=data)
    d = tmp_path / "x"
    import shutil
    shutil.copy(d / "epoch0004.pt", d / "latest.pt")  # latest rolled back: rows 1-6, summaries 2,4,6, epoch0006 orphan
    m2, res = _series(tmp_path, "x", 8, resume=True, data=data)
    assert res["start_epoch"] == 4 and res["recovery"]["curve_rows_dropped"] == 2
    assert res["recovery"]["summary_rows_dropped"] == 1 and res["recovery"]["orphan_checkpoints_deleted"] == ["epoch0006.pt"]
    assert [r["train_mse"] for r in _curve(d)] == [r["train_mse"] for r in _curve(tmp_path / "ref")]
    summ = [json.loads(line)["epoch"] for line in (d / "checkpoint_summaries.jsonl").read_text().splitlines()]
    assert summ == [2, 4, 6, 8]
    assert all(torch.equal(p, q) for p, q in zip(ref.state_dict().values(), m2.state_dict().values()))


def test_interruption_with_incomplete_trailing_lines(data, tmp_path):
    ref, _ = _series(tmp_path, "ref", 6, data=data)
    _series(tmp_path, "y", 4, data=data)
    d = tmp_path / "y"
    with (d / "curve.jsonl").open("a") as f:
        f.write('{"epoch": 5, "train_mse": 0.1')  # partial row after latest (epoch 4)
    with (d / "checkpoint_summaries.jsonl").open("a") as f:
        f.write('{"epoch": 6, "tr')
    m2, res = _series(tmp_path, "y", 6, resume=True, data=data)
    assert res["recovery"]["curve_trailing_incomplete_line_dropped"]
    assert res["recovery"]["summary_trailing_incomplete_line_dropped"]
    assert [r["train_mse"] for r in _curve(d)] == [r["train_mse"] for r in _curve(tmp_path / "ref")]
    assert all(torch.equal(p, q) for p, q in zip(ref.state_dict().values(), m2.state_dict().values()))


def test_interruption_before_first_checkpoint_and_inconsistent_records(data, tmp_path):
    ref, _ = _series(tmp_path, "ref", 4, data=data)
    _series(tmp_path, "z", 1, data=data)  # one epoch, no checkpoint (ckpt_every=2 ... epoch==epochs saves)
    d = tmp_path / "z"
    (d / "latest.pt").unlink()
    (d / "epoch0001.pt").unlink()
    with pytest.raises(FileExistsError):
        _series(tmp_path, "z", 4, resume=True, data=data)
    m2, res = _series(tmp_path, "z", 4, resume=True, restart=True, data=data)
    assert res["recovery"]["restarted_unsaved"]
    assert [r["train_mse"] for r in _curve(d)] == [r["train_mse"] for r in _curve(tmp_path / "ref")]
    assert all(torch.equal(p, q) for p, q in zip(ref.state_dict().values(), m2.state_dict().values()))
    # latest ahead of the curve rows is unrecoverable
    rows = _curve(tmp_path / "ref")
    (tmp_path / "ref" / "curve.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:2]))
    with pytest.raises(ValueError):
        _series(tmp_path, "ref", 6, resume=True, data=data)
