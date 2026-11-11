from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fconcord.utils import json_dump


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(figure: plt.Figure, path: Path, title: str, records: list[dict[str, str]]) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    records.append(
        {
            "index": str(len(records) + 1),
            "title": title,
            "file": path.name,
            "sha256": _sha256(path),
        }
    )


def generate_comparison_figures(
    comparison_directory: str | Path, output_directory: str | Path | None = None
) -> Path:
    """Generate paper comparison figures only from the frozen statistical report."""
    comparison = Path(comparison_directory).resolve()
    output = (
        comparison / "figures"
        if output_directory is None
        else Path(output_directory).resolve()
    )
    output.mkdir(parents=True, exist_ok=False)
    report = _load_json(comparison / "comparison_report.json")
    sns.set_theme(style="whitegrid", context="paper")
    records: list[dict[str, str]] = []
    loss_labels = {
        "squared_error": "Origin-level mean squared error",
        "absolute_error": "Origin-level mean absolute error",
        "energy_score": "Origin-level Energy Score",
    }
    for loss_index, (loss_name, loss_label) in enumerate(loss_labels.items(), start=1):
        loss = report["loss_reports"][loss_name]
        paired = loss["paired_baselines_vs_fconcord"]
        models = sorted(paired)
        effects = np.asarray(
            [
                paired[model]["diebold_mariano_hac"]
                ["mean_loss_difference_candidate_minus_reference"]
                for model in models
            ]
        )
        colors = ["#2a9d8f" if value < 0.0 else "#e76f51" for value in effects]
        figure, axis = plt.subplots(figsize=(9, max(4, 0.42 * len(models))))
        axis.barh(models, effects, color=colors)
        axis.axvline(0.0, color="black", linewidth=1.0)
        axis.set_xlabel("Candidate loss minus F-CONCORD loss")
        axis.set_title(f"Effect sizes: {loss_label}")
        _save(
            figure,
            output / f"{loss_index:02d}_{loss_name}_effect_sizes.png",
            f"{loss_label} effect sizes",
            records,
        )

        lower = np.asarray(
            [
                paired[model]["circular_block_bootstrap"]["confidence_interval"][0]
                for model in models
            ]
        )
        upper = np.asarray(
            [
                paired[model]["circular_block_bootstrap"]["confidence_interval"][1]
                for model in models
            ]
        )
        figure, axis = plt.subplots(figsize=(9, max(4, 0.42 * len(models))))
        positions = np.arange(len(models))
        axis.errorbar(
            effects,
            positions,
            xerr=np.vstack([effects - lower, upper - effects]),
            fmt="o",
            color="#264653",
            ecolor="#457b9d",
            capsize=3,
        )
        axis.axvline(0.0, color="black", linewidth=1.0)
        axis.set_yticks(positions, models)
        axis.set_xlabel("Paired mean loss difference with 95% block-bootstrap CI")
        axis.set_title(f"Uncertainty in effects: {loss_label}")
        _save(
            figure,
            output / f"{loss_index + 3:02d}_{loss_name}_effect_intervals.png",
            f"{loss_label} block-bootstrap intervals",
            records,
        )

        p_values = np.asarray(
            [
                paired[model]["diebold_mariano_hac"]["holm_adjusted_p_value"]
                for model in models
            ]
        )
        figure, axis = plt.subplots(figsize=(9, max(4, 0.42 * len(models))))
        axis.barh(models, -np.log10(np.maximum(p_values, 1e-300)), color="#6d597a")
        axis.axvline(-np.log10(0.05), color="#e63946", linestyle="--")
        axis.set_xlabel(r"$-\log_{10}$(Holm-adjusted DM p-value)")
        axis.set_title(f"Multiplicity-controlled evidence: {loss_label}")
        _save(
            figure,
            output / f"{loss_index + 6:02d}_{loss_name}_holm_significance.png",
            f"{loss_label} Holm-adjusted significance",
            records,
        )

        mcs = loss["model_confidence_set"]
        mean_losses = mcs["mean_losses"]
        ordered = sorted(mean_losses, key=mean_losses.get)
        included = set(mcs["included_models"])
        figure, axis = plt.subplots(figsize=(9, max(4, 0.42 * len(ordered))))
        axis.barh(
            ordered,
            [mean_losses[model] for model in ordered],
            color=["#2a9d8f" if model in included else "#adb5bd" for model in ordered],
        )
        axis.invert_yaxis()
        axis.set_xlabel("Seed-averaged pooled origin loss")
        axis.set_title(f"95% Model Confidence Set: {loss_label}")
        _save(
            figure,
            output / f"{loss_index + 9:02d}_{loss_name}_mcs.png",
            f"{loss_label} Model Confidence Set",
            records,
        )

        ablations = loss.get("paired_retrained_ablations_vs_full", {})
        if ablations:
            names = sorted(ablations)
            values = [
                ablations[name]["diebold_mariano_hac"]
                ["mean_loss_difference_candidate_minus_reference"]
                for name in names
            ]
            figure, axis = plt.subplots(figsize=(9, max(4, 0.55 * len(names))))
            axis.barh(
                names,
                values,
                color=["#2a9d8f" if value < 0.0 else "#f4a261" for value in values],
            )
            axis.axvline(0.0, color="black", linewidth=1.0)
            axis.set_xlabel("Retrained ablation loss minus full-model loss")
            axis.set_title(f"Structural ablation effects: {loss_label}")
            _save(
                figure,
                output / f"{loss_index + 12:02d}_{loss_name}_retrained_ablations.png",
                f"{loss_label} retrained-ablation effects",
                records,
            )

    spa_rows = []
    for loss_name, loss_label in loss_labels.items():
        spa = report["loss_reports"][loss_name]["spa"]
        spa_rows.append(
            [loss_label, float(spa["spa_statistic"]), float(spa["p_value"])]
        )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].barh([row[0] for row in spa_rows], [row[1] for row in spa_rows], color="#457b9d")
    axes[0].set_title("SPA maximum studentized improvement")
    axes[1].barh([row[0] for row in spa_rows], [row[2] for row in spa_rows], color="#e9c46a")
    axes[1].axvline(0.05, color="#e63946", linestyle="--")
    axes[1].set_title("SPA p-value")
    _save(figure, output / "16_spa_summary.png", "SPA omnibus summary", records)

    models = report["baseline_models"]
    matrix = np.zeros((len(loss_labels), len(models)), dtype=float)
    for row, loss_name in enumerate(loss_labels):
        paired = report["loss_reports"][loss_name]["paired_baselines_vs_fconcord"]
        for column, model in enumerate(models):
            matrix[row, column] = paired[model]["diebold_mariano_hac"][
                "mean_loss_difference_candidate_minus_reference"
            ]
    standardized = matrix / np.maximum(np.std(matrix, axis=1, keepdims=True), 1e-12)
    figure, axis = plt.subplots(figsize=(max(9, 0.7 * len(models)), 4))
    sns.heatmap(
        standardized,
        xticklabels=models,
        yticklabels=list(loss_labels.values()),
        center=0.0,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        axis=axis,
    )
    axis.set_title("Standardized loss-effect map; negative favors candidate")
    _save(figure, output / "17_effect_heatmap.png", "Cross-metric effect heatmap", records)

    json_dump(
        {
            "schema": "fconcord-comparison-figure-manifest-v1",
            "comparison_report": str((comparison / "comparison_report.json").resolve()),
            "figure_count": len(records),
            "figures": records,
        },
        output / "manifest.json",
    )
    return output
