#!/usr/bin/env python3
"""Publication figures from completed stored evidence; no fitting or selection.

python work/scripts/plot_rebuttal_evidence.py
PNG, editable SVG and vector PDF outputs, captions, provenance manifest and plot-data CSVs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np


BLUE = "#286A95"
TEAL = "#23857B"
ORANGE = "#B77A2B"
RED = "#AF4C48"
GRAY = "#636A70"
FOLD_COLORS = [BLUE, TEAL, ORANGE]
FOLDS = ["fold_1", "fold_2", "fold_3"]
PERIODS = ["2020–2021", "2022–2023", "2024–Mar 2025"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label(name, factor_names):
    groups = {
        "momentum": "Momentum", "up_semivar": "Upside log-semivariance",
        "down_semivar": "Downside log-semivariance", "jump_share": "Jump share",
        "log_illiquidity": "Log illiquidity",
    }
    if name == "order_flow":
        return "Signed-volume flow"
    if name.startswith("factor_beta_"):
        factor = factor_names[int(name.rsplit("_", 1)[1])].replace("MKT_RF", "MKT−RF")
        return f"Factor beta: {factor}"
    prefix, window = name.rsplit("_", 1)
    return f"{groups.get(prefix, prefix)} ({window})"


def style_axis(ax, *, ygrid=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C8CED2")
    ax.tick_params(length=3, color="#A6AFB5")
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", color="#E7EBEE", linewidth=0.65)


def save(fig, name, output):
    paths = []
    for extension in ("png", "svg", "pdf"):
        p = output / f"{name}.{extension}"
        fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.12)
        paths.append(p)
    plt.close(fig)
    return paths


def concept_figure(evidence, metadata, output):
    names = metadata["concept_names"]
    rows = evidence["folds"]["fold_1"]["concepts"]
    lookup = {(r["coordinate"], r["lead"]): r for r in rows}
    skill = np.array([[100 * lookup[name, h]["relative_mae_improvement"] for h in range(1, 6)] for name in names])
    if not np.isfinite(skill).all():
        raise ValueError("Undefined persistence skill cannot be silently plotted")
    # A genuinely symmetric linear range displays every observed cell without
    # clipping. Positive and negative changes use equal color magnitude.
    limit = max(25.0, np.ceil(np.abs(skill).max() / 25.0) * 25.0)
    fig = plt.figure(figsize=(11.8, 9.8))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.13, 1.0], wspace=0.43, hspace=0.42,
                           left=0.23, right=0.97, bottom=0.16, top=0.88)
    ax = fig.add_subplot(grid[:, 0])
    image = ax.pcolormesh(np.arange(6)-0.5, np.arange(len(names)+1)-0.5, skill,
                         cmap="RdBu", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                         shading="flat", antialiased=False, rasterized=False)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(len(names)-0.5, -0.5)
    ax.set_xticks(range(5), [str(h) for h in range(1, 6)])
    ax.set_yticks(range(len(names)), [label(name, metadata["factor_names"]) for name in names])
    ax.tick_params(axis="y", labelsize=8.1, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Forecast lead (trading observations)")
    ax.set_title("a  Coordinate-level skill · fold 1", loc="left", pad=11, fontsize=11, weight="bold")
    for edge in [3.5, 7.5, 11.5, 15.5, 19.5, 20.5]:
        ax.axhline(edge, color="white", linewidth=1.8)
    for k in range(len(names)):
        for h in range(5):
            ax.text(h, k, f"{skill[k,h]:+.1f}", ha="center", va="center", fontsize=7.3,
                    color="white" if abs(skill[k,h]) > limit * 0.62 else "#1E292F")
    for spine in ax.spines.values():
        spine.set_visible(False)
    position = ax.get_position()
    cax = fig.add_axes([position.x0, 0.088, position.width, 0.016])
    cb = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[-limit, -limit/2, 0, limit/2, limit])
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.ax.tick_params(labelsize=8, length=2)
    cb.set_label("MAE skill (%) · positive favors learned", fontsize=8.5)
    lead_rows = []
    for fidx, fold in enumerate(FOLDS):
        panel = fig.add_subplot(grid[fidx, 1])
        values = evidence["folds"][fold]["concept_mean_mae_per_lead"]
        leads = np.arange(1, 6)
        panel.plot(leads, values["learned"], color=BLUE, marker="o", lw=1.9, ms=4.3, label="Learned rollout")
        panel.plot(leads, values["persistence"], color=GRAY, marker="s", lw=1.6, ms=4, linestyle="--", label="Persistence")
        style_axis(panel)
        panel.set_xlim(0.8, 5.2)
        panel.set_xticks(leads)
        panel.set_ylim(0.25, 0.70)
        panel.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
        panel.set_ylabel("Concept MAE\n(standardized units)", fontsize=9)
        panel.set_title(f"{'bcd'[fidx]}  Fold {fidx+1} · {PERIODS[fidx]}", loc="left", fontsize=10.5, weight="bold", pad=9)
        if fidx == 2:
            panel.set_xlabel("Forecast lead (trading observations)")
        if fidx == 0:
            panel.legend(loc="upper left", frameon=False, fontsize=8.3)
        for h in range(5):
            lead_rows.append({"fold": fold, "lead": h+1, "learned_mae": values["learned"][h],
                              "persistence_mae": values["persistence"][h]})
    fig.suptitle("Concept forecasts against matched persistence", x=0.23, y=0.96, ha="left", fontsize=15, weight="bold")
    fig.text(0.23, 0.922, "Five training seeds; future leads only. All 26 measured coordinates are retained.",
             fontsize=9.5, ha="left", color=GRAY)
    fig.text(0.23, 0.023, "Skill = 100 × (1 − learned MAE / persistence MAE). The full symmetric color range is shown.\n"
             "Line panels show means averaged equally over concepts, assets, origins and seeds.",
             fontsize=8.3, color=GRAY)
    write_csv(output / "concept_leadwise_plot_data.csv", lead_rows)
    paths = save(fig, "R1_concept_persistence", output)
    caption = (
        "Revised concept-path diagnostic. (a) Fold-1 coordinate-level MAE skill relative to persistence, "
        "100(1 − MAE_learned/MAE_persistence), for all 26 concepts and leads 1–5. Each MAE is averaged over "
        "the same 500 held-out origins, 32 assets and five fitted seeds. Positive values indicate lower learned MAE; "
        "negative values favor persistence. Colors use a symmetric linear scale centered at zero without clipping; "
        "cell labels are rounded percentages, and exact values are in the source table. (b–d) Absolute mean concept "
        "MAE across the three chronological folds, in train-fitted robust-standardized projected coordinates. "
        "Persistence carries the measured input C_0 to every future lead. Curves are five-seed means."
    )
    return paths, caption


def graph_figure(evidence, output):
    expected = "strict abs(edge)>0.05; off diagonal; per-origin IoU; empty union assigned zero"
    if evidence["protocol"]["graph_support"] != expected:
        raise ValueError("Graph figure must match the specified 0.05/empty-union-zero convention")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.15), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.25, top=0.75, wspace=0.15)
    for fidx, (fold, ax) in enumerate(zip(FOLDS, axes)):
        rows = sorted([r for r in evidence["folds"][fold]["graphs"] if r["metric"] == "iou"], key=lambda r:r["lead"])
        x = np.array([r["lead"] for r in rows])
        y = np.array([r["learned"] for r in rows])
        p = np.array([r["persistence"] for r in rows])
        std = np.array([r["learned_seed_std"] for r in rows])
        ax.fill_between(x, y-std, y+std, color=BLUE, alpha=0.22, linewidth=0)
        ax.plot(x, y, color=BLUE, marker="o", ms=4.7, lw=1.8)
        ax.plot(x, p, color=GRAY, marker="s", ms=4.3, lw=1.6, linestyle="--")
        style_axis(ax)
        ax.set_ylim(0, 1.02)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlim(0.8, 5.55)
        ax.set_xticks(x)
        ax.set_xlabel("Forecast lead")
        ax.set_title(f"{'abc'[fidx]}  Fold {fidx+1} · {PERIODS[fidx]}", loc="left", fontsize=10.4, weight="bold")
        ax.annotate(f"{y[-1]:.3f}", (5, y[-1]), xytext=(4, -10), textcoords="offset points", fontsize=8, color=BLUE)
        ax.annotate(f"{p[-1]:.3f}", (5, p[-1]), xytext=(4, 5), textcoords="offset points", fontsize=8, color=GRAY)
    axes[0].set_ylabel("Graph-support IoU (higher is better)")
    handles = [Line2D([], [], color=BLUE, marker="o", lw=1.8, label="Learned rollout"),
               Line2D([], [], color=GRAY, marker="s", linestyle="--", lw=1.6, label="Persistence")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.90), frameon=False, ncol=2, fontsize=9)
    fig.suptitle("Graph forecasts against matched persistence", x=0.07, y=0.98, ha="left", fontsize=15, weight="bold")
    fig.text(0.07, 0.10, "Support: |edge| > 0.05; empty union → 0. Bands show ±1 seed SD and may be narrower than the line.\n"
             "Both forecasts use the same measured targets; persistence retains the observed graph at the forecast origin.", fontsize=8.5, color=GRAY)
    caption = (
        "Revised graph-path diagnostic. Mean per-origin support intersection-over-union for learned graph forecasts "
        "and matched persistence A_hat_h=A_0, using the same targets and origins in each fold. Support is defined by "
        "strict |A_ij|>0.05 on off-diagonal entries, and empty unions receive zero, exactly matching the stored-evidence "
        "evaluator. Lines average over five seeds; blue bands show ±1 sample standard deviation across seeds. "
        "Persistence is deterministic for the fixed "
        "prepared data. Measured graphs are carried forward between five-observation refits. Persistence has higher "
        "IoU at every displayed lead in all three folds. The full IoU range is shown."
    )
    return save(fig, "R2_graph_persistence", output), caption


def sensitivity_arrays(sensitivity, source):
    result = {}
    for fold in FOLDS:
        array = source[f"{fold}__mean_derivative_calibrated"].astype(np.float64) * 1e4
        if array.ndim != 5 or array.shape[0] != 5 or array.shape[1] != 32 or array.shape[2] != 5 or array.shape[3] != 32 or array.shape[4] != 26:
            raise ValueError(f"Unexpected sensitivity axes for {fold}: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite derivative")
        result[fold] = array
    return result


def signed_sensitivity_figure(values, metadata, output):
    names = metadata["concept_names"]
    labels = [label(name, metadata["factor_names"]) for name in names]
    summaries = {}
    rows = []
    for fold in FOLDS:
        v = values[fold].reshape(-1, len(names))
        q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        summaries[fold] = q
        for k, name in enumerate(names):
            rows.append({"fold":fold,"coordinate":name,"q05_bp":q[0,k],"q25_bp":q[1,k],
                         "median_bp":q[2,k],"q75_bp":q[3,k],"q95_bp":q[4,k],
                         "fraction_positive":float((v[:,k]>0).mean()),"derivatives_per_coordinate":len(v)})
    lim = max(float(np.abs(q).max()) for q in summaries.values()) * 1.06
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 9.1), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.235, right=0.975, top=0.86, bottom=0.14, wspace=0.12)
    y = np.arange(len(names))
    for fidx, (fold, ax) in enumerate(zip(FOLDS, axes)):
        q = summaries[fold]
        for k in y:
            color = BLUE if q[2,k] >= 0 else RED
            ax.hlines(k, q[0,k], q[4,k], color=color, linewidth=1.15, alpha=0.62)
            ax.hlines(k, q[1,k], q[3,k], color=color, linewidth=4, alpha=0.80)
            ax.plot(q[2,k], k, "o", ms=3.5, color=color, markeredgecolor="white", markeredgewidth=0.35)
        ax.axvline(0, color="#78858C", linewidth=0.9, linestyle=(0,(3,3)))
        ax.set_yticks(y, labels, fontsize=8.3)
        ax.set_ylim(len(names)-0.35, -0.7)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("Mean derivative\n(bp / standardized concept unit)", fontsize=9)
        ax.set_title(f"{'abc'[fidx]}  Fold {fidx+1} · {PERIODS[fidx]}", loc="left", fontsize=10.3, weight="bold", pad=10)
        style_axis(ax, ygrid=False)
        ax.grid(axis="x", color="#E7EBEE", linewidth=0.65)
        ax.tick_params(axis="y", length=0)
        for edge in [3.5,7.5,11.5,15.5,19.5,20.5]:
            ax.axhline(edge, color="#E7EBEE", lw=0.65)
    fig.suptitle("Direct mean responses through the measured state", x=0.235, y=0.965, ha="left", fontsize=15, weight="bold")
    fig.text(0.235, 0.922, "Own-asset derivatives of the calibrated mean; graph and all other decoder inputs held fixed.", fontsize=9.3, color=GRAY)
    fig.text(0.235, 0.055, "Point: median  •  Thick line: 25th–75th percentiles  •  Thin line: 5th–95th percentiles\n"
             "Conditional fitted responses across 5 seeds × 32 origins × 5 leads × 32 assets.", fontsize=8.7, color=GRAY)
    write_csv(output / "mean_signed_distribution_plot_data.csv", rows)
    caption = (
        "Direct own-asset mean sensitivities at evolved predicted states. For each fold, derivatives "
        "∂μ_(t,h,i)/∂C_hat_(t,h,i,k) are computed by autograd for all 26 concept coordinates, 32 evenly spaced "
        "held-out origins, five forecast leads, 32 assets and all five training seeds (25,600 derivatives per coordinate "
        "and fold). These are conditional responses of the fitted decoder, holding the evolved graph, other concept "
        "coordinates, observed market state, lead embedding and fitted parameters fixed. The plotted means include the "
        "original validation-fitted positive scalar. Values are decimal-return derivatives multiplied by 10,000, hence "
        "basis points per robust-standardized concept unit. Dots show medians, thick segments interquartile ranges, "
        "and thin segments 5th–95th percentiles of the sampled derivatives. "
        "All panels use the same signed horizontal scale."
    )
    return save(fig, "R3_direct_mean_signed", output), caption


def absolute_sensitivity_figure(values, metadata, output):
    names = metadata["concept_names"]
    labels = [label(name, metadata["factor_names"]) for name in names]
    seed_means = {fold:np.abs(values[fold]).mean(axis=(1,2,3)) for fold in FOLDS}
    mean_abs = {fold:v.mean(axis=0) for fold,v in seed_means.items()}
    order = np.argsort(-mean_abs["fold_1"])
    lim = max(float(v.max()) for v in seed_means.values()) * 1.08
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 9.1), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.235, right=0.975, top=0.86, bottom=0.14, wspace=0.12)
    y = np.arange(len(names))
    rows = []
    for fidx, (fold, ax) in enumerate(zip(FOLDS, axes)):
        ax.barh(y, mean_abs[fold][order], height=0.61, color=FOLD_COLORS[fidx], alpha=0.77)
        for seed_index in range(5):
            ax.scatter(seed_means[fold][seed_index, order], y+(seed_index-2)*0.07,
                       s=9, facecolors="white", edgecolors="#3A444B", linewidths=0.5, zorder=3)
        ax.set_yticks(y, [labels[i] for i in order], fontsize=8.3)
        ax.set_ylim(len(names)-0.35, -0.7)
        ax.set_xlim(0, lim)
        ax.set_xlabel("Mean absolute derivative\n(bp / standardized unit)", fontsize=9)
        ax.set_title(f"{'abc'[fidx]}  Fold {fidx+1} · {PERIODS[fidx]}", loc="left", fontsize=10.3, weight="bold", pad=10)
        style_axis(ax, ygrid=False)
        ax.grid(axis="x", color="#E7EBEE", linewidth=0.65)
        ax.tick_params(axis="y", length=0)
        for rank, k in enumerate(order):
            row = {"fold":fold,"fold1_display_rank":rank+1,"coordinate":names[k],
                   "mean_absolute_derivative_bp":float(mean_abs[fold][k])}
            row.update({f"seed_{seed}_mean_absolute_bp":float(seed_means[fold][j,k]) for j,seed in enumerate([1,7,19,37,73])})
            rows.append(row)
    fig.suptitle("Magnitude of direct mean responses", x=0.235, y=0.965, ha="left", fontsize=15, weight="bold")
    fig.text(0.235, 0.922, "All 26 coordinates, ordered by fold-1 mean absolute sensitivity; the same order and scale apply to every fold.", fontsize=9.1, color=GRAY)
    fig.text(0.235, 0.055, "Bars: mean absolute own-asset derivative  •  Open points: each seed’s mean absolute derivative\n"
             "Calibrated means; conditional fitted responses in basis points per robust-standardized concept unit.", fontsize=8.7, color=GRAY)
    write_csv(output / "mean_absolute_rank_plot_data.csv", rows)
    caption = (
        "Magnitude of direct mean responses using the same derivative sample as R3. Bars show the mean absolute "
        "own-asset derivative, averaged equally over sampled origins, leads, assets and seeds; five open points show "
        "each seed's corresponding mean absolute value. Coordinates are sorted once by the fold-1 pooled magnitude, "
        "and that ordering is retained in folds 2 and 3. All panels share an axis beginning at zero and the same units "
        "(basis points per robust-standardized concept unit). The derivative includes each checkpoint's original "
        "validation-fitted positive mean scalar, so differences across folds include differences in mean shrinkage."
    )
    return save(fig, "R4_direct_mean_magnitude", output), caption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path("work/archived_evidence"))
    parser.add_argument("--sensitivity-dir", type=Path, default=Path("work/mean_sensitivity"))
    parser.add_argument("--metadata", type=Path, default=Path("work/compact_prepared/fold_1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
                         "axes.labelsize":9, "xtick.labelsize":8.5, "ytick.labelsize":8.5,
                         "svg.fonttype":"none", "pdf.fonttype":42, "ps.fonttype":42,
                         "axes.titleweight":"normal", "savefig.transparent":False})
    evidence_path = args.evidence_dir / "evidence.json"
    sensitivity_path = args.sensitivity_dir / "mean_sensitivity.json"
    array_path = args.sensitivity_dir / "mean_sensitivity_arrays.npz"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if digest(array_path) != sensitivity["arrays_sha256"]:
        raise ValueError("Sensitivity array digest does not match the completed audit")
    for fold in FOLDS:
        if sensitivity["folds"][fold]["concept_names"] != metadata["concept_names"]:
            raise ValueError("Concept-name alignment mismatch")
    with np.load(array_path, allow_pickle=False) as source:
        values = sensitivity_arrays(sensitivity, source)
    entries = [concept_figure(evidence, metadata, args.output_dir), graph_figure(evidence, args.output_dir),
               signed_sensitivity_figure(values, metadata, args.output_dir), absolute_sensitivity_figure(values, metadata, args.output_dir)]
    sources = [evidence_path, args.evidence_dir/"concept_persistence.csv", args.evidence_dir/"graph_persistence.csv",
               sensitivity_path, array_path, args.metadata]
    manifest = {"schema":"trefin-rebuttal-figures-v1", "script":str(Path(__file__).resolve()),
                "script_sha256":digest(Path(__file__)), "sources":[{"path":str(p.resolve()),"sha256":digest(p)} for p in sources],
                "figures":[], "generated_data_tables":[]}
    captions = []
    for paths, caption in entries:
        figure = paths[0].stem
        captions.extend([f"## {figure}", "", caption, ""])
        manifest["figures"].append({"id":figure,"caption":caption,
                                    "artifacts":[{"path":str(p.resolve()),"sha256":digest(p)} for p in paths]})
    for p in sorted(args.output_dir.glob("*_plot_data.csv")):
        manifest["generated_data_tables"].append({"path":str(p.resolve()),"sha256":digest(p)})
    (args.output_dir/"captions.md").write_text("\n".join(captions), encoding="utf-8")
    (args.output_dir/"figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output":str(args.output_dir.resolve()),"figure_count":len(entries),"formats":["png","svg","pdf"]}))


if __name__ == "__main__":
    main()
