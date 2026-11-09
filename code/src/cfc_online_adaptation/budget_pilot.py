"""Training-budget pilot: three CfC implementation configurations trained for the same
number of full-batch optimizer updates (plan v0.3, 2026-11-07, stage 1: offline only).

Configurations
    dense_backbone : ``CfCInverseModel``            (ncps ``CfCCell`` with its default 128-unit backbone)
    fc_wired       : ``FullyConnectedWiredCfCInverseModel`` (``WiredCfCCell``, one layer, all-ones mask)
    ncp_wired      : ``WiredCfCInverseModel``       (``WiredCfCCell``, AutoNCP wiring, three layers)

These are *implementation configurations*; the differences are not a single factor
(backbone block, layer splitting, sparse inter-layer masks and parameter counts all
differ). Nothing here modifies the frozen sources ``core.py`` / ``experiment.py``; the
training loop below reproduces ``experiment.train_inverse_model`` step for step and adds
checkpoints, timing and audit-set evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import FullyConnected

from .arm import ArmParams, TwoLinkArm
from .bases import normalised_mse, teacher_forced_z
from .core import CfCInverseModel, WiredCfCInverseModel
from .dataset import InverseDataset, build_reach_sequences, train_val_split

CONFIGS = ("dense_backbone", "fc_wired", "ncp_wired")
CENTER_XY = (0.0, 0.45)
TRAIN_RADII = (0.08, 0.10, 0.12)
AUDIT_RADII = (0.09, 0.11)
DURATION, DT, N_TARGETS, VAL_FRAC = 0.5, 0.005, 8, 0.25
INPUT_DIM, UNITS, OUTPUT_DIM = 10, 64, 2
LR = 1e-3
CHECKPOINT_FORMAT_VERSION = 1


# ----------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------
class FullyConnectedWiredCfCInverseModel(WiredCfCInverseModel):
    """``WiredCfCCell`` with a fully connected wiring (all-ones effective mask, one layer).

    Same cell parametrisation as the NCP-wired model (no backbone); the full 64-dim
    recurrent state is read by the external bias-free linear readout. The wiring's
    polarity seed is recorded but has no effect on the cell, which takes ``abs`` of the mask.
    """

    def __init__(self, input_dim: int = INPUT_DIM, units: int = UNITS,
                 output_dim: int = OUTPUT_DIM, *, erev_init_seed: int = 1111):
        nn.Module.__init__(self)
        self.input_dim, self.units, self.output_dim = input_dim, units, output_dim
        self.feature_mode = "state"
        self.sparsity_level = 0.0
        self.seed = erev_init_seed
        self.wiring = FullyConnected(units, output_dim, erev_init_seed=erev_init_seed)
        self.core = CfC(input_dim, self.wiring, batch_first=True, return_sequences=True)
        self.d_z = units
        self.readout = nn.Linear(self.d_z, output_dim, bias=False)

    def wiring_params(self) -> dict:
        return {"wiring": "FullyConnected", "units": self.units, "output_dim": self.output_dim,
                "feature_mode": self.feature_mode, "d_z": self.d_z,
                "erev_init_seed": self.seed,
                "n_core_params": int(sum(p.numel() for p in self.core.parameters()))}


def build_model(config: str, seed: int, wiring_seed: int):
    """Instantiate a configuration under the same seeding as ``run_revision_offline``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if config == "dense_backbone":
        return CfCInverseModel(INPUT_DIM, UNITS, OUTPUT_DIM)
    if config == "fc_wired":
        return FullyConnectedWiredCfCInverseModel(INPUT_DIM, UNITS, OUTPUT_DIM,
                                                  erev_init_seed=wiring_seed)
    if config == "ncp_wired":
        return WiredCfCInverseModel(INPUT_DIM, UNITS, OUTPUT_DIM, feature_mode="state",
                                    sparsity_level=0.5, seed=wiring_seed)
    raise ValueError(config)


# ----------------------------------------------------------------------------------------
# audit
# ----------------------------------------------------------------------------------------
def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def audit_model(model: nn.Module) -> dict:
    """Machine-readable structure audit: tensors, masks, layers, wiring counts."""
    tensors = []
    for name, p in model.named_parameters():
        tensors.append({"name": name, "shape": list(p.shape), "numel": int(p.numel()),
                        "requires_grad": bool(p.requires_grad),
                        "part": "readout" if name.startswith("readout") else "core",
                        "is_mask": name.endswith("sparsity_mask")})
    totals = {}
    for part in ("core", "readout"):
        rows = [t for t in tensors if t["part"] == part]
        totals[part] = {"total": sum(t["numel"] for t in rows),
                        "trainable": sum(t["numel"] for t in rows if t["requires_grad"]),
                        "mask_elements": sum(t["numel"] for t in rows if t["is_mask"])}
    masks = []
    for name, p in model.named_parameters():
        if name.endswith("sparsity_mask"):
            m = p.detach().cpu().numpy()
            cell = name.rsplit(".", 1)[0]
            hidden = m.shape[0]
            masks.append({"name": name, "shape": list(m.shape), "nonzero": int(np.count_nonzero(m)),
                          "elements": int(m.size), "requires_grad": bool(p.requires_grad),
                          "sha256": _sha(m.astype(np.float32)),
                          "recurrent_block_all_ones": bool(np.all(m[:, -hidden:] == 1.0)),
                          "applied_to": [f"{cell}.ff1.weight", f"{cell}.ff2.weight"],
                          "not_applied_to": [f"{cell}.time_a.weight", f"{cell}.time_b.weight",
                                             f"{cell}.ff1.bias", f"{cell}.ff2.bias",
                                             f"{cell}.time_a.bias", f"{cell}.time_b.bias"],
                          "effective_ff_weights": int(2 * np.count_nonzero(m)),
                          "nominal_ff_weights": int(2 * m.size)})
    cell = model.core.rnn_cell
    wired = type(cell).__name__ == "WiredCfCCell"
    info = {"cell_class": type(cell).__name__, "wired": wired,
            "layer_sizes": list(cell.layer_sizes) if wired else [model.units],
            "update_order": "sequential layers within each time step" if wired else "single cell",
            "backbone": (None if wired else {"units": int(cell.backbone[0].out_features),
                                             "layers": int(cell.backbone_layers)}),
            "mode": cell.mode if not wired else cell._layers[0].mode,
            "state_dim": int(model.units), "readout_in": int(model.readout.in_features),
            "readout_out": int(model.readout.out_features), "readout_bias": model.readout.bias is not None}
    if wired:
        w = model.wiring
        info["wiring"] = {"class": type(w).__name__, "synapse_count": int(w.synapse_count),
                          "sensory_synapse_count": int(w.sensory_synapse_count),
                          "note": "wiring-object synapse counts are not the effective connection "
                                  "counts; the cell appends an all-ones recurrent block per layer "
                                  "and masks only ff1/ff2"}
        if hasattr(model, "wiring_params"):
            info["wiring"].update(model.wiring_params())
    return {"tensors": tensors, "totals": totals, "masks": masks, "structure": info}


def masks_equal(model_a: nn.Module, model_b: nn.Module) -> bool:
    sa = {n: p for n, p in model_a.named_parameters() if n.endswith("sparsity_mask")}
    sb = {n: p for n, p in model_b.named_parameters() if n.endswith("sparsity_mask")}
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


# ----------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------
def make_arm_and_center():
    arm = TwoLinkArm(ArmParams())
    return arm, arm.inverse_kinematics(np.array(CENTER_XY), elbow="down")


def training_dataset(arm, center_q) -> InverseDataset:
    return build_reach_sequences(arm, center_q, radii=TRAIN_RADII, durations=(DURATION,),
                                 dt=DT, n_targets=N_TARGETS)


def audit_dataset(arm, center_q) -> InverseDataset:
    """Unused-for-selection set at interpolated radii (same directions, same duration)."""
    return build_reach_sequences(arm, center_q, radii=AUDIT_RADII, durations=(DURATION,),
                                 dt=DT, n_targets=N_TARGETS)


def split_for_seed(ds: InverseDataset, seed: int):
    return train_val_split(ds, val_frac=VAL_FRAC, seed=seed)


def direction_index(ds: InverseDataset) -> np.ndarray:
    """Direction index 0..7 of each sequence (targets are generated in angle order per radius)."""
    arm, center_q = make_arm_and_center()
    center_xy = arm.fk(center_q)
    ang = [math.atan2(m["target_xy"][1] - center_xy[1], m["target_xy"][0] - center_xy[0]) for m in ds.meta]
    return np.array([int(round((a % (2 * math.pi)) / (2 * math.pi / N_TARGETS))) % N_TARGETS for a in ang])


def evaluate(model: nn.Module, ds: InverseDataset) -> dict:
    """Teacher-forced fit: NMSE and MSE (existing definitions) plus per-direction NMSE."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(ds.x.astype(np.float32)))[0].cpu().numpy().astype(np.float64)
    out = {"nmse": normalised_mse(pred, ds.tau), "mse": float(np.mean((pred - ds.tau) ** 2)),
           "finite": bool(np.all(np.isfinite(pred))), "n_sequences": int(len(ds))}
    d = direction_index(ds)
    out["nmse_by_direction"] = [normalised_mse(pred[d == k], ds.tau[d == k]) if np.any(d == k) else math.nan
                                for k in range(N_TARGETS)]
    out["n_by_direction"] = [int(np.sum(d == k)) for k in range(N_TARGETS)]
    return out


def feature_energy(model: nn.Module, x: np.ndarray) -> float:
    z = teacher_forced_z(model, x)
    return float(np.mean(np.sum(z * z, axis=-1)))


# ----------------------------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------------------------
def atomic_torch_save(obj, path: Path) -> int:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path.stat().st_size


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint written by ``atomic_torch_save``; reject incomplete files."""
    if path.name.endswith(".tmp"):
        raise ValueError(f"refusing temporary file {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    required = {"format_version", "epoch", "model_state", "optimizer_state", "torch_rng_state",
                "numpy_rng_state", "config", "complete"}
    if not isinstance(ck, dict) or not required.issubset(ck) or not ck["complete"]:
        raise ValueError(f"incomplete or foreign checkpoint {path}")
    return ck


def _rng_states() -> dict:
    return {"torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state()}


def _restore_rng(ck: dict) -> None:
    torch.set_rng_state(ck["torch_rng_state"])
    np.random.set_state(ck["numpy_rng_state"])


# ----------------------------------------------------------------------------------------
# training loop (reproduces experiment.train_inverse_model, plus bookkeeping)
# ----------------------------------------------------------------------------------------
def _read_jsonl(path: Path) -> tuple[list[dict], bool]:
    """Parse a JSONL file; a trailing incomplete line is dropped and reported."""
    if not path.exists():
        return [], False
    rows, truncated = [], False
    lines = path.read_text().split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1 or all(not x.strip() for x in lines[i + 1:]):
                truncated = True
                break
            raise
    return rows, truncated


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, path)


def _verification(model: nn.Module, x_va: torch.Tensor) -> dict:
    """Predictions of the live model on the validation input, stored in the checkpoint so that
    a freshly built model that loads the checkpoint can be compared with the model that was
    saved (not with itself)."""
    model.eval()
    with torch.no_grad():
        pred = model(x_va)[0].cpu().numpy().astype(np.float32)
    return {"input_sha256": _sha(x_va.cpu().numpy().astype(np.float32)), "val_pred": pred,
            "val_pred_sha256": _sha(pred), "mask_sha256": [m["sha256"] for m in audit_model(model)["masks"]]}


def recover_records(out_dir: Path, latest_epoch: int) -> dict:
    """Make curve/summary/checkpoint files consistent with ``latest.pt`` (epoch N):
    rows and summaries after N are dropped, duplicates removed, trailing incomplete lines
    dropped, and checkpoint files beyond N deleted. Rows missing before N are an error."""
    rep = {"latest_epoch": latest_epoch}
    curve, trunc = _read_jsonl(out_dir / "curve.jsonl")
    epochs = [r["epoch"] for r in curve]
    if epochs[:latest_epoch] != list(range(1, latest_epoch + 1)):
        raise ValueError("resume: curve.jsonl does not contain epochs 1..N for latest epoch N")
    rep["curve_rows_dropped"] = len(curve) - latest_epoch
    rep["curve_trailing_incomplete_line_dropped"] = trunc
    if rep["curve_rows_dropped"] or trunc:
        _write_jsonl(out_dir / "curve.jsonl", curve[:latest_epoch])
    summ, strunc = _read_jsonl(out_dir / "checkpoint_summaries.jsonl")
    kept, seen = [], set()
    for s in summ:
        if s["epoch"] <= latest_epoch and s["epoch"] not in seen:
            kept.append(s)
            seen.add(s["epoch"])
    rep["summary_rows_dropped"] = len(summ) - len(kept)
    rep["summary_trailing_incomplete_line_dropped"] = strunc
    if rep["summary_rows_dropped"] or strunc:
        _write_jsonl(out_dir / "checkpoint_summaries.jsonl", kept)
    orphans = [p for p in out_dir.glob("epoch*.pt") if int(p.stem[5:]) > latest_epoch]
    for p in orphans:
        p.unlink()
    rep["orphan_checkpoints_deleted"] = [p.name for p in orphans]
    for p in out_dir.glob("*.tmp"):
        p.unlink()
    return rep


def train_series(model: nn.Module, train: InverseDataset, val: InverseDataset,
                 audit: InverseDataset, *, epochs: int, seed: int, out_dir: Path,
                 ckpt_every: int, config: dict, resume: bool = False, lr: float = LR,
                 forward_timing_repeats: int = 5, allow_restart_unsaved: bool = False,
                 provenance: dict | None = None) -> dict:
    """Full-batch Adam training with periodic model+optimizer checkpoints.

    Step semantics are identical to ``experiment.train_inverse_model``: seed, Adam(lr),
    MSELoss, one update per epoch, validation after the update. No validation-based model
    selection is performed.

    Save order at a checkpoint epoch N (process interruption at any point is recoverable):
    1. curve row N is appended and flushed; 2. ``epochNNNN.pt`` is written (tmp + rename);
    3. the summary line is appended; 4. ``latest.pt`` is written last. On ``resume`` the
    records are reconciled to the epoch of ``latest.pt`` (``recover_records``). Interruption
    before the first checkpoint leaves no ``latest.pt``: resuming then requires
    ``allow_restart_unsaved=True`` (the partial records are discarded), otherwise it raises.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "curve.jsonl"
    x_tr = torch.from_numpy(train.x.astype(np.float32))
    tau_tr = torch.from_numpy(train.tau.astype(np.float32))
    x_va = torch.from_numpy(val.x.astype(np.float32))
    tau_va = torch.from_numpy(val.tau.astype(np.float32))
    loss_fn = torch.nn.MSELoss()
    provenance = provenance or {}

    start_epoch, recovery = 0, None
    latest = out_dir / "latest.pt"
    if resume and latest.exists():
        ck = load_checkpoint(latest)
        static = lambda c: {k: v for k, v in c.items() if k != "epochs"}  # noqa: E731
        if static(ck["config"]) != static(config):
            raise ValueError("resume: checkpoint config differs from requested config")
        if ck.get("provenance", {}) != provenance:
            raise ValueError("resume: checkpoint provenance differs from requested provenance")
        if int(ck["epoch"]) > epochs:
            raise ValueError("resume: checkpoint epoch exceeds requested epochs")
        recovery = recover_records(out_dir, int(ck["epoch"]))
        model.load_state_dict(ck["model_state"])
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        opt.load_state_dict(ck["optimizer_state"])
        _restore_rng(ck)
        start_epoch = int(ck["epoch"])
    elif resume and curve_path.exists():
        if not allow_restart_unsaved:
            raise FileExistsError(f"{out_dir}: interrupted before the first checkpoint; pass "
                                  "allow_restart_unsaved=True to discard the partial records")
        recovery = {"restarted_unsaved": True, "discarded_rows": len(_read_jsonl(curve_path)[0])}
        for p in list(out_dir.glob("*.jsonl")) + list(out_dir.glob("*.tmp")):
            p.unlink()
        torch.manual_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        if curve_path.exists() or latest.exists():
            raise FileExistsError(f"{out_dir} already has training output; use resume or a new dir")
        torch.manual_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    saved, ckpt_bytes, ckpt_io_s = [], {}, 0.0
    epoch_times = []
    with curve_path.open("a") as curve:
        for epoch in range(start_epoch + 1, epochs + 1):
            t0 = time.perf_counter()
            model.train()
            opt.zero_grad()
            pred, _z, _hx = model(x_tr)
            loss = loss_fn(pred, tau_tr)
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(x_va)[0], tau_va)
            t_epoch = time.perf_counter() - t0
            epoch_times.append(t_epoch)
            tr_mse, va_mse = float(loss.item()), float(val_loss.item())
            row = {"epoch": epoch, "train_mse": tr_mse, "val_mse": va_mse,
                   "train_nmse": tr_mse / float(train.tau.var()), "val_nmse": va_mse / float(val.tau.var()),
                   "finite": bool(math.isfinite(tr_mse) and math.isfinite(va_mse)),
                   "epoch_time_s": t_epoch}
            is_ckpt = epoch % ckpt_every == 0 or epoch == epochs
            if is_ckpt:
                row["checkpoint"] = f"epoch{epoch:04d}.pt"
            curve.write(json.dumps(row) + "\n")  # 1. curve row first
            curve.flush()
            if is_ckpt:
                t1 = time.perf_counter()
                summary = {"epoch": epoch, "train": evaluate(model, train), "val": evaluate(model, val),
                           "audit": evaluate(model, audit),
                           "feature_energy_train": feature_energy(model, train.x),
                           "readout_norm": float(np.linalg.norm(model.export_readout_weights()))}
                ck = {"format_version": CHECKPOINT_FORMAT_VERSION, "epoch": epoch,
                      "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                      **_rng_states(), "config": config, "provenance": provenance,
                      "summary": summary, "verification": _verification(model, x_va), "complete": True}
                path = out_dir / f"epoch{epoch:04d}.pt"
                ckpt_bytes[path.name] = atomic_torch_save(ck, path)  # 2. checkpoint
                with (out_dir / "checkpoint_summaries.jsonl").open("a") as f:  # 3. summary
                    f.write(json.dumps(summary) + "\n")
                ckpt_bytes["latest.pt"] = atomic_torch_save(ck, latest)  # 4. latest last
                ckpt_io_s += time.perf_counter() - t1
                saved.append(epoch)

    # forward timing (val set, no grad), after training; warm-up excluded
    model.eval()
    with torch.no_grad():
        model(x_va)
        fw = []
        for _ in range(forward_timing_repeats):
            t0 = time.perf_counter()
            model(x_va)
            fw.append(time.perf_counter() - t0)
    n_warm = min(2, max(0, len(epoch_times) - 1))
    timing = {"epochs_timed": len(epoch_times), "warmup_excluded": n_warm,
              "epoch_time_s_mean_excl_warmup": float(np.mean(epoch_times[n_warm:])) if epoch_times[n_warm:] else math.nan,
              "epoch_time_s_median": float(np.median(epoch_times)) if epoch_times else math.nan,
              "epoch_time_definition": "train step (zero_grad, forward, backward, Adam step) plus "
                                       "validation forward; checkpoint evaluation and IO excluded",
              "checkpoint_io_s_total": ckpt_io_s,
              "forward_val_s_median": float(np.median(fw)) if fw else math.nan,
              "forward_definition": f"no-grad forward over the validation set ({len(val)} sequences), "
                                    f"median of {forward_timing_repeats} after one warm-up call",
              "torch_num_threads": torch.get_num_threads()}
    return {"saved_epochs": saved, "checkpoint_bytes": ckpt_bytes, "timing": timing,
            "final_epoch": epochs, "start_epoch": start_epoch, "recovery": recovery}


def _loaded(model_builder, ck: dict) -> nn.Module:
    m = model_builder()
    m.load_state_dict(ck["model_state"])
    return m


def reload_check(model_builder, ckpt_path: Path, x: np.ndarray,
                 reference_ckpt_path: Path | None = None) -> dict:
    """Build a fresh model, load ``ckpt_path`` and compare its forward on ``x`` with the
    predictions the *saved* model produced before saving (stored in the checkpoint, or in
    ``reference_ckpt_path`` when given). Masks are compared with the stored digests and with
    a second fresh instance."""
    ck = load_checkpoint(ckpt_path)
    ref = load_checkpoint(reference_ckpt_path) if reference_ckpt_path else ck
    fresh = model_builder()
    fresh.load_state_dict(ck["model_state"])
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    fresh.eval()
    with torch.no_grad():
        pred = fresh(xt)[0].cpu().numpy().astype(np.float32)
    fresh_masks = [m["sha256"] for m in audit_model(fresh)["masks"]]
    v = ref["verification"]
    return {"input_matches_saved_input": _sha(xt.numpy()) == v["input_sha256"],
            "forward_bit_identical_to_saved_model": bool(np.array_equal(pred, v["val_pred"])),
            "forward_max_abs_diff_to_saved_model": float(np.max(np.abs(pred - v["val_pred"]))) if pred.shape == v["val_pred"].shape else math.nan,
            "masks_equal_to_saved_digests": fresh_masks == v["mask_sha256"],
            "masks_equal_after_reload": masks_equal(fresh, _loaded(model_builder, ck)),
            "mask_sha256": fresh_masks, "checkpoint_epoch": int(ck["epoch"]),
            "reference_epoch": int(ref["epoch"])}
