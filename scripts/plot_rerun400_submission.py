"""Regenerate all numerical manuscript figures/tables from the 400-update results."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

METHODS = [
    "dense-CfC-LMS",
    "NCP-CfC-LMS",
    "FC-CfC-LMS",
    "ESN-LMS",
    "RBF-LMS",
    "dense-CfC-RLS",
]
LABELS = [
    "Dense CfC LMS",
    "NCP-wired CfC LMS",
    "FC-wired CfC LMS",
    "ESN LMS",
    "RBF LMS",
    "Dense CfC RLS",
]
SHORT = ["Dense", "NCP", "FC", "ESN", "RBF", "RLS"]
COLORS = ["#0072B2", "#56B4E9", "#9467bd", "#009E73", "#E69F00", "#CC79A7"]


def read(p):
    return list(csv.DictReader(Path(p).open()))


def med(q, k):
    return float(np.nanmedian([float(r[k]) if r[k] else np.nan for r in q]))


def table(path, caption, label, header, rows, align, resize=True):
    text = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        (r"\resizebox{\linewidth}{!}{" if resize else "") + r"\begin{tabular}{" + align + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    text += [r" & ".join(row) + r" \\" for row in rows]
    text += [r"\bottomrule\end{tabular}" + ("}" if resize else ""), r"\end{table}"]
    path.write_text("\n".join(text) + "\n")


def generate(root, out, methods=None, scope=None):
    root, out = Path(root), Path(out)
    if scope is not None:
        scope = json.loads(Path(scope).read_text()) if not isinstance(scope, dict) else scope
        chosen = scope["included_adaptive_methods"]
        if methods is not None and list(methods) != chosen:
            raise ValueError("Method allowlist differs from publication scope")
        methods = chosen
        for name, digest in scope["figure_input_sha256"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Publication input checksum differs: " + name)
    methods = list(METHODS if methods is None else methods)
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(METHODS):
        raise ValueError("Expected a nonempty unique allowlist of known methods")
    indices = [METHODS.index(m) for m in methods]
    labels = [LABELS[i] for i in indices]
    short = [SHORT[i] for i in indices]
    colors = [COLORS[i] for i in indices]
    sensitivity_methods = [
        m
        for m in methods
        if m in ["dense-CfC-LMS", "NCP-CfC-LMS", "FC-CfC-LMS", "ESN-LMS", "RBF-LMS"]
    ]
    if out.exists() and any(out.iterdir()):
        raise ValueError("Figure output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    e = read(root / "evaluation_per_seed.csv")
    d = read(root / "diagnostics/diagnostics_summary.csv")
    sens = read(root / "sensitivity_eta_pd_per_seed.csv")
    counts = (scope or {}).get(
        "source_dataset_counts",
        {"evaluation": 333, "diagnostics": 60, "adaptive_sensitivity_outcomes": 405},
    )
    if (len(e), len(d), len(sens)) != (
        counts["evaluation"],
        counts["diagnostics"],
        counts["adaptive_sensitivity_outcomes"],
    ):
        raise ValueError("Dataset row counts differ from the supplied scope")
    allruns = [
        json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines() if line.strip()
    ]

    def evalrows(m, b):
        return [r for r in e if r["method"] == m and float(r["b"]) == b]

    rows = []
    for m, label in zip(methods, labels):
        q = evalrows(m, 13)
        assert len(q) == 10
        mirrors = [int(r["ae_family_n_mirror"]) for r in q]
        span = (
            str(min(mirrors)) if min(mirrors) == max(mirrors) else f"{min(mirrors)}--{max(mirrors)}"
        )
        rows.append(
            [
                label,
                f"{med(q, 'B_first_mm'):.2f}",
                f"{med(q, 'B_last_mm'):.3f}",
                f"{100 * med(q, 'improvement_vs_B0_family'):.1f}",
                f"{med(q, 'catch_signed_mm'):+.2f}",
                str(sum(r["ae_family_ae_0.25"] == "True" for r in q)) + "/10",
                span,
            ]
        )
    table(
        out / "revision_main_table.tex",
        r"Online adaptation at $b=13$ on evaluation seeds 110--119. First B, Final B, Reduction, and Catch are medians across ten seeds. AE counts passing seeds; Reversed /8 gives the minimum--maximum direction count across seeds. Reduction uses each model\textquotesingle s own frozen-readout reference. The after-effect decision uses the sign and direction-mean ratio threshold 0.25. Direction counts describe target directions within a run, not independent samples.",
        "tab:revision-main",
        r"Method & First B (mm) & Final B (mm) & Reduction (\%) & Catch (mm) & AE & Reversed /8",
        rows,
        "lrrrrrr",
    )
    rows = []
    for m, label in zip(methods, labels):
        q = [r for r in d if r["method"] == m]
        rows.append(
            [
                label,
                f"{med(q, 'probe_r_median'):.4f}",
                f"{med(q, 'frozen_pre_b13_mm'):.2f}",
                f"{med(q, 'adaptive_B_last_mm'):.3f}",
                f"{med(q, 'frozen_post_b13_mm'):.2f}",
                f"{100 * (1 - med(q, 'frozen_post_over_pre')):.1f}",
                f"{med(q, 'frozen_post_over_adaptive_last'):.3f}",
            ]
        )
    table(
        out / "diagnostic_table.tex",
        r"Compensation and fixed-readout diagnostics on the same ten evaluation seeds, $b=13$. Each entry is a median over ten runs; $r_{\rm nom}$ first takes a median over the eight directions within each run. Tracking errors are in millimeters. Pre/post refer to readouts before/after exposure, held fixed with the same active PD feedback. Reduction is $100(1-E_{\rm post}/E_{\rm pre})$; post/adaptive is computed within each seed before taking its median. All conditions pass the coarse catch decision, but RBF reverses seven of eight directions.",
        "tab:diagnostics",
        r"Method & $r_{\rm nom}$ & Fixed pre & Adaptive & Fixed post & Reduction (\%) & Post/adaptive",
        rows,
        "lrrrrrr",
    )
    rows = []
    for m, label in zip(methods, labels):
        for b in [-13.0, 25.0]:
            q = evalrows(m, b)
            mirrors = [int(r["ae_family_n_mirror"]) for r in q]
            rows.append(
                [
                    label,
                    f"{b:+.0f}",
                    f"{med(q, 'B_last_mm'):.3f}",
                    f"{med(q, 'ae_family_ratio'):.3f}",
                    str(sum(r["ae_family_ae_0.25"] == "True" for r in q)) + "/10",
                    f"{min(mirrors)}--{max(mirrors)}",
                ]
            )
    table(
        out / "revision_aux_table.tex",
        r"Reversed and stronger fields on evaluation seeds 110--119, without rate retuning. Final B and AE ratio are medians across ten seeds; AE counts seeds passing threshold 0.25; Reversed /8 gives the minimum--maximum direction count across seeds.",
        "tab:revision-aux",
        r"Method & $b$ & Final B (mm) & AE ratio & AE & Reversed /8",
        rows,
        "llrrrr",
        resize=False,
    )
    lines = [
        r"\begin{longtable}{llrrrrr}",
        r"\caption{Main-field individual adaptive outcomes, evaluation seeds 110--119. Tracking errors and catch deviations are in millimeters. Offline fit and both reference baselines are included in the machine-readable supplement.}\label{tab:revision-seeds}\\",
        r"\toprule Method & Seed & First B & Final B & Catch & AE ratio & Reversed /8 \\ \midrule\endfirsthead",
        r"\toprule Method & Seed & First B & Final B & Catch & AE ratio & Reversed /8 \\ \midrule\endhead",
    ]
    for m, label in zip(methods, short):
        for r in sorted(evalrows(m, 13), key=lambda r: int(r["seed"])):
            lines.append(
                f"{label} & {r['seed']} & {float(r['B_first_mm']):.2f} & {float(r['B_last_mm']):.3f} & {float(r['catch_signed_mm']):+.2f} & {float(r['ae_family_ratio']):.3f} & {r['ae_family_n_mirror']} "
                + r"\\"
            )
    lines += [r"\bottomrule\end{longtable}"]
    (out / "revision_seed_table.tex").write_text("\n".join(lines) + "\n")
    # Online curves and individual catch deviations.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, m in enumerate(methods):
        runs = [
            r for r in allruns if r["method"] == m and r["stage"] == "evaluate" and r["b"] == 13
        ]
        curves = np.array(
            [
                [c["mean_max_perp_dev"] * 1000 for c in r["curves"] if c["phase"] == "B"]
                for r in runs
            ]
        )
        assert curves.shape == (10, 30)
        axes[0].plot(range(1, 31), np.median(curves, axis=0), label=short[i], color=colors[i])
        vals = [float(r["catch_signed_mm"]) for r in evalrows(m, 13)]
        axes[1].scatter(i + np.linspace(-0.18, 0.18, 10), vals, color=colors[i], s=14)
    axes[0].set(xlabel="Exposure cycle (updates on)", ylabel="Tracking error (mm)", yscale="log")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set(xticks=range(len(methods)), xticklabels=short, ylabel="Signed catch deviation (mm)")
    fig.tight_layout()
    fig.savefig(out / "revision_main.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, m in enumerate(methods):
        q = sorted([r for r in d if r["method"] == m], key=lambda r: int(r["seed"]))
        x = i + np.linspace(-0.22, 0.22, 10)
        axes[0].scatter(x, [float(r["probe_r_median"]) for r in q], color=colors[i], s=17)
        axes[1].scatter(
            x - 0.025,
            [float(r["adaptive_B_last_mm"]) for r in q],
            color=colors[i],
            marker="x",
            s=22,
        )
        axes[1].scatter(
            x + 0.025,
            [float(r["frozen_post_b13_mm"]) for r in q],
            edgecolors=colors[i],
            facecolors="none",
            s=23,
        )
    axes[0].axhline(0.311, color="grey", ls="--", lw=0.9, label="Median best-constant residual")
    axes[0].legend(fontsize=8)
    axes[0].set(ylabel="Nominal squared residual", yscale="log")
    axes[1].scatter([], [], color="black", marker="x", label="Updates on")
    axes[1].scatter([], [], facecolors="none", edgecolors="black", label="Fixed post")
    axes[1].legend(fontsize=8)
    axes[1].set(ylabel="Tracking error (mm)", yscale="log")
    for ax in axes:
        ax.set(xticks=range(len(methods)), xticklabels=short)
    fig.tight_layout()
    fig.savefig(out / "diagnostic_comparison.pdf")
    plt.close(fig)
    columns = 3 if len(methods) <= 6 else 4
    nrows = (len(methods) + columns - 1) // columns
    fig, axes = plt.subplots(
        nrows, columns, figsize=(12, 2.5 * nrows), sharex=True, sharey=True, squeeze=False
    )
    for i, (m, label) in enumerate(zip(methods, labels)):
        w = np.load(root / "diagnostics" / f"{m}_s110" / "probe_waveforms.npz")
        ax = axes.flat[i]
        ax.plot(w["time"], w["s"][0, :, 0], color="black", ls="--", label="Ideal compensation")
        ax.plot(w["time"], w["delta"][0, :, 0], color=colors[i], label="Learned output change")
        ax.set_title(label, fontsize=9)
    for ax in list(axes.flat)[len(methods) :]:
        ax.axis("off")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=9)
    fig.supxlabel("Time (s)")
    fig.supylabel("Joint-1 torque (N m)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out / "diagnostic_waveforms.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    matrices = []
    for m in sensitivity_methods:
        q = [r for r in sens if r["method"] == m]
        etas = sorted({float(r["eta"]) for r in q})
        z = np.full((3, 9), np.nan)
        for iy, pd in enumerate([0.5, 1.0, 2.0]):
            for ix, eta in enumerate(etas):
                a = [
                    float(r["B_last_mm"])
                    for r in q
                    if float(r["pd"]) == pd and float(r["eta"]) == eta and r["B_last_mm"]
                ]
                if a:
                    z[iy, ix] = np.log10(np.median(a))
        matrices.append(z)
    lo = min(np.nanmin(z) for z in matrices)
    hi = max(np.nanmax(z) for z in matrices)
    for i, m in enumerate(sensitivity_methods):
        ax = axes.flat[i]
        q = [r for r in sens if r["method"] == m]
        etas = sorted({float(r["eta"]) for r in q})
        im = ax.imshow(matrices[i], origin="lower", vmin=lo, vmax=hi, cmap="viridis", aspect="auto")
        for iy, pd in enumerate([0.5, 1.0, 2.0]):
            for ix, eta in enumerate(etas):
                a = [r for r in q if float(r["pd"]) == pd and float(r["eta"]) == eta]
                v = [r for r in a if r["valid_by_selection_rule"] == "True"]
                n = sum(r["ae_family_ae_0.25"] == "True" for r in v)
                if len(v) < 3:
                    ax.add_patch(
                        Rectangle(
                            (ix - 0.5, iy - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="//",
                            edgecolor="grey",
                            lw=0,
                        )
                    )
                ax.text(
                    ix, iy, f"{n}/{len(v)}", ha="center", va="center", fontsize=6, color="white"
                )
        ax.set(
            title=labels[methods.index(m)],
            yticks=range(3),
            yticklabels=["0.5", "1", "2"],
            xticks=range(9),
            xticklabels=[".03", ".1", ".3", "1", "3", "10", "30", "100", "300"],
        )
    for ax in list(axes.flat)[len(sensitivity_methods) :]:
        ax.axis("off")
    fig.supxlabel("Learning-rate multiplier")
    fig.supylabel("PD-gain multiplier")
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    cax = fig.add_axes([0.90, 0.16, 0.015, 0.68])
    fig.colorbar(im, cax=cax, label="log10 median finite error (mm)")
    fig.savefig(out / "revision_sensitivity.pdf")
    plt.close(fig)
    files = [
        "evaluation_per_seed.csv",
        "diagnostics/diagnostics_summary.csv",
        "runs.jsonl",
        "sensitivity_eta_pd_per_seed.csv",
    ]
    (out / "figure_manifest.json").write_text(
        json.dumps(
            {
                "schema": "rerun400-figures-v1",
                "source_files": {
                    f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files
                },
                "methods": methods,
                "sensitivity_methods": sensitivity_methods,
                "source_scope": scope,
                "evaluation_seeds": list(range(110, 120)),
                "waveform_seed": 110,
                "waveform_direction": 0,
                "waveform_joint": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, help="Ordered method allowlist")
    p.add_argument(
        "--scope", type=Path, help="Publication scope manifest (also checks input hashes)"
    )
    a = p.parse_args()
    generate(a.results, a.out, methods=a.methods, scope=a.scope)
