from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def generate_runtime_figure(
    run_dir: Path, audit: dict[str, Any], config: dict[str, Any]
) -> None:
    """Overwrite figure 44 after the complete figure runtime is known."""
    labels = ["data", "training", "prediction", "figures"]
    values = [float(audit["runtime_seconds"].get(label, 0.0)) for label in labels]
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Seconds")
    plt.title("End-to-end runtime profile")
    plt.tight_layout()
    for extension in config["formats"]:
        plt.savefig(
            run_dir / "figures" / f"44_runtime_profile.{extension}",
            dpi=int(config["dpi"]),
            bbox_inches="tight",
        )
    plt.close()


def generate_all_figures(
    run_dir: Path,
    predictions: dict[str, np.ndarray],
    prepared: dict[str, Any],
    history: list[dict[str, float]],
    metrics: dict[str, Any],
    baselines: dict[str, Any],
    ablations: dict[str, Any],
    audit: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, str]]:
    """Generate a broad, data-backed figure suite for prediction and theory audits."""
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config["dpi"])
    formats = list(config["formats"])
    sns.set_theme(style="whitegrid", context="paper")
    manifest: list[dict[str, str]] = []

    def save(number: int, slug: str, title: str, category: str) -> None:
        stem = f"{number:02d}_{slug}"
        plt.title(title)
        plt.tight_layout()
        for extension in formats:
            plt.savefig(figure_dir / f"{stem}.{extension}", dpi=dpi, bbox_inches="tight")
        plt.close()
        manifest.append({"id": str(number), "file": f"{stem}.{formats[0]}", "title": title, "category": category})

    frame = pd.DataFrame(history)
    epochs = frame["epoch"]
    plt.figure()
    plt.plot(epochs, frame["train_total"], label="train")
    plt.plot(epochs, frame["val_total"], label="validation")
    plt.xlabel("Epoch"); plt.ylabel("Total loss"); plt.legend()
    save(1, "total_loss", "Total objective convergence", "optimization")

    plt.figure()
    plt.plot(epochs, frame["train_distribution"], label="train energy score")
    plt.plot(epochs, frame["val_distribution"], label="validation energy score")
    plt.xlabel("Epoch"); plt.ylabel("Energy score"); plt.legend()
    save(2, "distribution_loss", "Strict distribution-score trajectory", "optimization")

    plt.figure()
    for column in ["train_concept", "train_graph", "train_monotonicity", "train_gain"]:
        plt.plot(epochs, frame[column], label=column.replace("train_", ""))
    plt.yscale("symlog", linthresh=1e-6); plt.xlabel("Epoch"); plt.ylabel("Auxiliary loss"); plt.legend()
    save(3, "auxiliary_losses", "Concept, graph, monotonicity and gain losses", "optimization")

    plt.figure(); plt.plot(epochs, frame["learning_rate"]); plt.xlabel("Epoch"); plt.ylabel("Learning rate")
    save(4, "learning_rate", "Learning-rate schedule", "optimization")

    plt.figure(); plt.plot(epochs, frame["gradient_norm"]); plt.axhline(1.0, ls="--", c="r", label="default clip"); plt.xlabel("Epoch"); plt.ylabel("Gradient norm"); plt.legend()
    save(5, "gradient_norm", "Gradient-norm and clipping audit", "optimization")

    plt.figure(); plt.plot(epochs, frame["train_gain_radius"], label="train"); plt.plot(epochs, frame["val_gain_radius"], label="validation"); plt.axhline(1.0, ls="--", c="r", label="small-gain threshold"); plt.xlabel("Epoch"); plt.ylabel("Optimization proxy radius"); plt.legend()
    save(6, "gain_radius", "Small-gain optimization proxy", "theory")

    y = predictions["y"]
    mu = predictions["mu"]
    residual = y - mu
    samples = predictions["samples"]
    plt.figure(); plt.scatter(y.ravel(), mu.ravel(), s=8, alpha=0.45); limits = np.quantile(np.abs(np.r_[y.ravel(), mu.ravel()]), 0.99); plt.plot([-limits, limits], [-limits, limits], "k--"); plt.xlabel("Realized return"); plt.ylabel("Predicted mean")
    save(7, "prediction_scatter", "Predicted versus realized returns", "forecast")

    plt.figure(); sns.histplot(residual.ravel(), bins=35, stat="density", kde=True); plt.xlabel("Realized minus predicted mean")
    save(8, "residual_histogram", "Forecast residual distribution", "forecast")

    plt.figure(); stats.probplot(residual.ravel(), dist="norm", plot=plt)
    save(9, "residual_qq", "Residual normal Q-Q diagnostic", "forecast")

    lower = np.quantile(samples[:, :, 0, 0], 0.05, axis=0)
    upper = np.quantile(samples[:, :, 0, 0], 0.95, axis=0)
    center = mu[:, 0, 0]
    x = np.arange(len(center))
    plt.figure(figsize=(8, 3.5)); plt.fill_between(x, lower, upper, alpha=0.25, label="90% interval"); plt.plot(x, center, label="mean"); plt.plot(x, y[:, 0, 0], lw=1, label="realized"); plt.xlabel("Test forecast origin"); plt.ylabel("Return"); plt.legend(ncol=3)
    save(10, "forecast_interval_timeline", "Forecast interval timeline for one asset", "forecast")

    horizon_rows = metrics["horizon"]
    h = [row["h"] for row in horizon_rows]
    plt.figure(); plt.bar(h, [row["rmse"] for row in horizon_rows]); plt.xlabel("Horizon"); plt.ylabel("RMSE")
    save(11, "horizon_rmse", "RMSE by forecast horizon", "forecast")

    plt.figure(); plt.bar(h, [row["mae"] for row in horizon_rows]); plt.xlabel("Horizon"); plt.ylabel("MAE")
    save(12, "horizon_mae", "MAE by forecast horizon", "forecast")

    plt.figure(); plt.bar(h, [row["directional_accuracy"] for row in horizon_rows]); plt.axhline(0.5, ls="--", c="k"); plt.xlabel("Horizon"); plt.ylabel("Directional accuracy")
    save(13, "horizon_direction", "Directional accuracy by horizon", "forecast")

    alpha_names = list(metrics["coverage"])
    plt.figure(); plt.bar(alpha_names, [metrics["coverage"][key] for key in alpha_names], alpha=0.75); plt.plot(alpha_names, [float(key) for key in alpha_names], "ko--", label="nominal"); plt.xlabel("Nominal interval"); plt.ylabel("Empirical coverage"); plt.legend()
    save(14, "interval_coverage", "Prediction-interval coverage calibration", "calibration")

    plt.figure(); plt.bar(alpha_names, [metrics["interval_width"][key] for key in alpha_names]); plt.xlabel("Central interval probability"); plt.ylabel("Average width")
    save(15, "interval_width", "Sharpness of prediction intervals", "calibration")

    pit = np.mean(samples <= y[None], axis=0).ravel()
    plt.figure(); plt.hist(pit, bins=10, range=(0, 1), density=True, alpha=0.8); plt.axhline(1.0, ls="--", c="k"); plt.xlabel("Randomized empirical PIT"); plt.ylabel("Density")
    save(16, "pit_histogram", "Probability integral transform diagnostic", "calibration")

    quantiles = np.linspace(0.05, 0.95, 19)
    empirical = [np.mean(y <= np.quantile(samples, q, axis=0)) for q in quantiles]
    plt.figure(); plt.plot(quantiles, empirical, "o-", label="empirical"); plt.plot([0, 1], [0, 1], "k--", label="ideal"); plt.xlabel("Nominal quantile"); plt.ylabel("Empirical frequency"); plt.legend()
    save(17, "quantile_calibration", "Full quantile calibration curve", "calibration")

    concept_prediction = predictions["concepts"]
    concept_target = predictions["c_target"]
    state_h = list(range(concept_prediction.shape[1]))
    plt.figure(); plt.scatter(concept_target.ravel(), concept_prediction.ravel(), s=4, alpha=0.25); plt.xlabel("Measured future concept"); plt.ylabel("Predicted future concept")
    save(18, "concept_fidelity_scatter", "Concept prediction fidelity", "interpretability")

    concept_mae = np.mean(np.abs(concept_prediction - concept_target), axis=(0, 2))
    plt.figure(figsize=(10, 3.5)); sns.heatmap(concept_mae, cmap="mako", xticklabels=prepared["metadata"]["concept_names"], yticklabels=[f"h={value}" for value in state_h]); plt.xlabel("Concept coordinate"); plt.ylabel("Horizon")
    save(19, "concept_error_heatmap", "Concept MAE by coordinate and horizon", "interpretability")

    plt.figure(); plt.plot(concept_target[:, 1, 0, 0], label="target"); plt.plot(concept_prediction[:, 1, 0, 0], label="prediction"); plt.xlabel("Test origin"); plt.ylabel(prepared["metadata"]["concept_names"][0]); plt.legend()
    save(20, "concept_trajectory", "One named concept through the test period", "interpretability")

    concept_matrix = np.corrcoef(concept_target[:, 0].reshape(-1, concept_target.shape[-1]), rowvar=False)
    plt.figure(figsize=(7, 6)); sns.heatmap(concept_matrix, center=0, cmap="vlag", xticklabels=False, yticklabels=prepared["metadata"]["concept_names"])
    save(21, "concept_correlation", "Measured concept correlation structure", "interpretability")

    graph_prediction = predictions["graphs"]
    graph_target = predictions["a_target"]
    graph_pred_mean = graph_prediction[:, 1].mean(axis=0)
    graph_target_mean = graph_target[:, 1].mean(axis=0)
    plt.figure(); sns.heatmap(graph_pred_mean, center=0, cmap="vlag", square=True)
    save(22, "predicted_graph_heatmap", "Mean predicted signed graph", "graph")

    plt.figure(); sns.heatmap(graph_target_mean, center=0, cmap="vlag", square=True)
    save(23, "target_graph_heatmap", "Mean measured future signed graph", "graph")

    plt.figure(); sns.heatmap(graph_pred_mean - graph_target_mean, center=0, cmap="vlag", square=True)
    save(24, "graph_error_heatmap", "Signed graph prediction error", "graph")

    off_diagonal = ~np.eye(graph_pred_mean.shape[0], dtype=bool)
    plt.figure(); sns.histplot(graph_prediction[:, 1][:, off_diagonal].ravel(), bins=35, label="predicted", stat="density", alpha=0.5); sns.histplot(graph_target[:, 1][:, off_diagonal].ravel(), bins=35, label="target", stat="density", alpha=0.5); plt.xlabel("Signed edge weight"); plt.legend()
    save(25, "edge_distribution", "Predicted and measured edge distributions", "graph")

    degree_pred = np.abs(graph_pred_mean).sum(axis=1)
    degree_target = np.abs(graph_target_mean).sum(axis=1)
    positions = np.arange(len(degree_pred)); width = 0.38
    plt.figure(); plt.bar(positions - width / 2, degree_pred, width, label="predicted"); plt.bar(positions + width / 2, degree_target, width, label="target"); plt.xlabel("Asset"); plt.ylabel("Absolute weighted degree"); plt.legend()
    save(26, "weighted_degree", "Asset-level weighted graph degree", "graph")

    laplacian = np.diag(np.abs(graph_pred_mean).sum(axis=1)) - graph_pred_mean
    eigenvalues = np.linalg.eigvalsh(laplacian)
    plt.figure(); plt.stem(np.arange(len(eigenvalues)), eigenvalues); plt.xlabel("Eigenvalue index"); plt.ylabel("Signed-Laplacian eigenvalue")
    save(27, "graph_spectrum", "Spectrum of the predicted signed Laplacian", "theory")

    network = nx.from_numpy_array(graph_pred_mean)
    positions_graph = nx.spring_layout(network, seed=7, weight="weight")
    edges = list(network.edges(data=True)); colors = ["#D1495B" if edge[2]["weight"] < 0 else "#087E8B" for edge in edges]; widths = [0.5 + 5 * abs(edge[2]["weight"]) for edge in edges]
    plt.figure(figsize=(6, 5)); nx.draw_networkx(network, positions_graph, node_size=500, font_size=7, edge_color=colors, width=widths); plt.axis("off")
    save(28, "network_topology", "Signed residual-dependence network topology", "graph")

    jaccard = []
    for index in range(graph_prediction.shape[1]):
        pred_edges = np.abs(graph_prediction[:, index]) > 0.05
        true_edges = np.abs(graph_target[:, index]) > 0.05
        intersection = np.logical_and(pred_edges, true_edges).sum(axis=(-2, -1))
        union = np.logical_or(pred_edges, true_edges).sum(axis=(-2, -1))
        jaccard.append(np.mean(intersection / np.maximum(union, 1)))
    plt.figure(); plt.plot(state_h, jaccard, "o-"); plt.xlabel("State lead h"); plt.ylabel("Edge Jaccard")
    save(29, "graph_jaccard", "Graph edge recovery across horizons", "graph")

    test_indices = predictions["indices"]
    beta = prepared["arrays"]["beta0"][test_indices].mean(axis=0)
    plt.figure(figsize=(6, 4)); sns.heatmap(beta, center=0, cmap="vlag", xticklabels=prepared["metadata"]["factor_names"], yticklabels=prepared["metadata"]["asset_names"])
    save(30, "factor_beta_heatmap", "Rolling systematic factor exposures", "concept")

    raw = prepared["arrays"]["raw_corr"][test_indices]
    partial = prepared["arrays"]["partial_corr"][test_indices]
    upper_index = np.triu_indices(raw.shape[-1], k=1)
    raw_strength = np.mean(np.abs(raw[:, upper_index[0], upper_index[1]]), axis=1)
    partial_strength = np.mean(np.abs(partial[:, upper_index[0], upper_index[1]]), axis=1)
    plt.figure(); plt.plot(raw_strength, label="raw return correlation"); plt.plot(partial_strength, label="factor-residual partial correlation"); plt.xlabel("Test origin"); plt.ylabel("Mean absolute dependence"); plt.legend()
    save(31, "raw_vs_residual_dependence", "Common-factor removal changes network dependence", "concept")

    market = prepared["arrays"]["s0"][test_indices]
    plt.figure(figsize=(8, 3.5));
    for index, name in enumerate(prepared["metadata"]["market_names"]): plt.plot(market[:, index], label=name)
    plt.xlabel("Test origin"); plt.ylabel("Robust standardized state"); plt.legend(fontsize=7, ncol=3)
    save(32, "market_state_timeline", "Three interpretable market-level concepts", "concept")

    plt.figure(); scatter = plt.scatter(market[:, 0], market[:, 1], c=market[:, 2], cmap="viridis", s=25); plt.xlabel("Market log realized variance"); plt.ylabel("Mean residual dependence"); plt.colorbar(scatter, label="Market illiquidity")
    save(33, "market_state_phase", "Market-state phase portrait", "concept")

    grid = np.asarray(audit["intervention_grid"])
    plt.figure(); plt.plot(grid, audit["intervention_mean"], "o-", label="conditional mean"); plt.plot(grid, audit["intervention_down_scale"], "s-", label="downside scale"); plt.xlabel("Risk-concept intervention"); plt.ylabel("Average decoder response"); plt.legend()
    save(34, "concept_intervention", "Fixed-graph conditional risk-concept response", "theory")

    plt.figure(); plt.plot(grid, audit["intervention_down_scale"], "o-"); plt.xlabel("Risk concept"); plt.ylabel("Downside scale"); plt.text(0.02, 0.95, f"min finite slope={audit['minimum_intervention_slope']:.3e}", transform=plt.gca().transAxes, va="top")
    save(35, "monotonic_response", "Certified conditional downside monotonicity", "theory")

    plt.figure(); plt.loglog(audit["perturbation_size"], audit["perturbation_amplification"], "o-"); plt.xlabel("Input perturbation norm"); plt.ylabel("Output/Input amplification")
    save(36, "perturbation_amplification", "Empirical perturbation stability", "theory")

    baseline_names = ["F-CONCORD"] + list(baselines)
    baseline_rmse = [metrics["rmse"]] + [baselines[name]["rmse"] for name in baselines]
    plt.figure(figsize=(7, 3.5)); plt.bar(baseline_names, baseline_rmse); plt.xticks(rotation=25, ha="right"); plt.ylabel("RMSE")
    save(37, "baseline_comparison", "Lightweight baseline comparison", "comparison")

    ablation_names = ["Full"] + list(ablations)
    ablation_rmse = [metrics["rmse"]] + [ablations[name]["rmse"] for name in ablations]
    plt.figure(); plt.bar(ablation_names, ablation_rmse); plt.xticks(rotation=20, ha="right"); plt.ylabel("RMSE")
    save(38, "ablation_comparison", "Model-component ablation audit", "comparison")

    fraction = float(audit["strategy_top_fraction"])
    count_assets = max(1, int(mu.shape[-1] * fraction))
    scores = mu[:, 0]
    realized = y[:, 0]
    order = np.argsort(scores, axis=1)
    weights = np.zeros_like(scores)
    for row in range(len(scores)):
        weights[row, order[row, -count_assets:]] = 1.0 / count_assets
        weights[row, order[row, :count_assets]] = -1.0 / count_assets
    gross = np.sum(weights * realized, axis=1)
    turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1) / 2.0
    costs = np.r_[0.0, turnover] * float(audit["transaction_cost_bps"]) * 1e-4
    net = gross - costs
    equity = np.cumprod(1.0 + np.clip(net, -0.95, None))
    plt.figure(); plt.plot(equity); plt.xlabel("Test origin"); plt.ylabel("Net wealth")
    save(39, "strategy_equity", "Illustrative cost-adjusted long-short wealth", "decision")

    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    plt.figure(); plt.fill_between(np.arange(len(drawdown)), drawdown, 0, color="#8C2F39", alpha=0.7); plt.xlabel("Test origin"); plt.ylabel("Drawdown")
    save(40, "strategy_drawdown", "Illustrative strategy drawdown", "decision")

    plt.figure(); plt.plot(np.r_[0.0, turnover], label="turnover"); plt.plot(costs, label="transaction cost"); plt.xlabel("Test origin"); plt.legend()
    save(41, "turnover_cost", "Turnover and transaction-cost path", "decision")

    regime_bin = pd.qcut(market[:, 0], q=min(3, len(np.unique(market[:, 0]))), labels=False, duplicates="drop")
    regime_error = [np.sqrt(np.mean((residual[np.asarray(regime_bin) == value]) ** 2)) for value in np.unique(regime_bin)]
    plt.figure(); plt.bar([f"vol regime {value}" for value in np.unique(regime_bin)], regime_error); plt.ylabel("RMSE")
    save(42, "error_by_regime", "Forecast error across volatility regimes", "robustness")

    plt.figure(); plt.plot(h, [row["energy_score"] for row in horizon_rows], "o-"); plt.xlabel("Horizon"); plt.ylabel("Energy score")
    save(43, "energy_by_horizon", "Joint energy score by horizon", "forecast")

    runtime_labels = ["data", "training", "prediction", "figures"]
    runtime_values = [float(audit["runtime_seconds"].get(label, 0.0)) for label in runtime_labels]
    plt.figure(); plt.bar(runtime_labels, runtime_values); plt.ylabel("Seconds")
    save(44, "runtime_profile", "End-to-end runtime profile", "systems")

    gain_matrix = np.asarray(audit["gain_matrix"])
    plt.figure(figsize=(4, 3.5)); sns.heatmap(gain_matrix, annot=True, fmt=".3f", cmap="mako", xticklabels=["concept", "graph"], yticklabels=["concept", "graph"])
    save(45, "small_gain_matrix", "Two-block small-gain proxy matrix", "theory")

    plt.figure(); sns.heatmap(predictions["correlation"][:, 0].mean(axis=0), center=0, cmap="vlag", square=True)
    save(46, "forecast_correlation", "Decoder-implied latent shock correlation", "distribution")

    plt.figure(); sns.heatmap(predictions["precision"][:, 0].mean(axis=0), cmap="mako", square=True)
    save(47, "forecast_precision", "Decoder precision structure", "distribution")

    plt.figure(); plt.scatter(predictions["sigma_plus"].ravel(), predictions["sigma_minus"].ravel(), s=6, alpha=0.3); maximum = max(predictions["sigma_plus"].max(), predictions["sigma_minus"].max()); plt.plot([0, maximum], [0, maximum], "k--"); plt.xlabel("Upside scale"); plt.ylabel("Downside scale")
    save(48, "tail_asymmetry", "Asymmetric upside and downside scales", "distribution")

    plt.figure(); sns.histplot(predictions["nu"].ravel(), bins=25); plt.axvline(30, ls="--", c="k", label="near-Gaussian reference"); plt.xlabel("Student-t degrees of freedom"); plt.legend()
    save(49, "degrees_of_freedom", "Learned heavy-tail degrees of freedom", "distribution")

    graph_energy = np.linalg.norm(graph_prediction, axis=(-2, -1)).mean(axis=1)
    concept_energy = np.linalg.norm(concept_prediction, axis=(-2, -1)).mean(axis=1)
    plt.figure(); plt.scatter(concept_energy, graph_energy, c=market[:, 0], cmap="coolwarm", s=25); plt.xlabel("Concept-state norm"); plt.ylabel("Graph-state norm"); plt.colorbar(label="Market volatility state")
    save(50, "joint_state_phase", "Concept-graph joint-state phase portrait", "theory")

    with (figure_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
