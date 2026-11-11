from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from fconcord.benchmark_training import residual_bootstrap_scenarios
from fconcord.metrics import evaluate_predictions
from fconcord.statistical_tests import (
    per_origin_absolute_error,
    per_origin_energy_score,
    per_origin_squared_error,
)


def _indices(benchmark: dict[str, Any], split: str) -> np.ndarray:
    return np.asarray(benchmark["metadata"]["split_indices"][split], dtype=np.int64)


def _var_features(
    benchmark: dict[str, Any], split: str, lags: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = _indices(benchmark, split)
    history = benchmark["arrays"]["return_history"][positions]
    if lags < 1 or lags > history.shape[1]:
        raise ValueError("VAR lags must lie within the saved history")
    features = history[:, -lags:].reshape(len(positions), -1)
    target = benchmark["arrays"]["target_scaled"][positions]
    return features, target.reshape(len(positions), -1), positions


def _long_tree_features(
    benchmark: dict[str, Any], split: str, lags: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = _indices(benchmark, split)
    arrays = benchmark["arrays"]
    history = arrays["return_history"][positions]
    if lags < 1 or lags > history.shape[1]:
        raise ValueError("tree lags must lie within the saved history")
    batch, _, assets = history.shape
    horizon = arrays["target_scaled"].shape[1]
    history_by_asset = history[:, -lags:].transpose(0, 2, 1)
    concepts = arrays["concept_state"][positions]
    graph = arrays["graph_state"][positions]
    market = arrays["market_state"][positions]
    signed_degree = graph.sum(axis=-1, keepdims=True)
    absolute_degree = np.abs(graph).sum(axis=-1, keepdims=True)
    asset_base = np.concatenate(
        [
            history_by_asset,
            concepts,
            signed_degree,
            absolute_degree,
            np.broadcast_to(market[:, None, :], (batch, assets, market.shape[-1])),
            np.broadcast_to(np.eye(assets, dtype=np.float32)[None], (batch, assets, assets)),
        ],
        axis=-1,
    )
    base_expanded = np.broadcast_to(
        asset_base[:, :, None, :], (batch, assets, horizon, asset_base.shape[-1])
    )
    lead_identity = np.eye(horizon, dtype=np.float32)
    lead_expanded = np.broadcast_to(
        lead_identity[None, None], (batch, assets, horizon, horizon)
    )
    features = np.concatenate([base_expanded, lead_expanded], axis=-1).reshape(
        batch * assets * horizon, -1
    )
    target = arrays["target_scaled"][positions].transpose(0, 2, 1).reshape(-1)
    return features.astype(np.float32), target.astype(np.float32), positions


def _predict_classical(
    name: str,
    model: Any,
    benchmark: dict[str, Any],
    split: str,
    parameters: dict[str, Any],
) -> dict[str, np.ndarray]:
    if name == "ShrunkVAR":
        features, _, positions = _var_features(
            benchmark, split, int(parameters["lags"])
        )
        scaled = model.predict(features).reshape(
            len(positions),
            int(benchmark["metadata"]["dimensions"]["horizon"]),
            int(benchmark["metadata"]["dimensions"]["num_assets"]),
        )
    elif name == "HistGradientBoosting":
        features, _, positions = _long_tree_features(
            benchmark, split, int(parameters["lags"])
        )
        dimensions = benchmark["metadata"]["dimensions"]
        scaled = model.predict(features).reshape(
            len(positions),
            int(dimensions["num_assets"]),
            int(dimensions["horizon"]),
        ).transpose(0, 2, 1)
    else:
        raise ValueError(f"Unknown classical benchmark: {name}")
    scale = benchmark["arrays"]["return_scale"]
    return {
        "mean": np.asarray(scaled, dtype=np.float32) * scale[None, None],
        "target": benchmark["arrays"]["target"][positions],
        "indices": positions,
    }


def fit_classical_candidate(
    name: str,
    parameters: dict[str, Any],
    benchmark: dict[str, Any],
    evaluation: dict[str, Any],
    candidate_directory: Path,
    seed: int,
) -> dict[str, Any]:
    candidate_directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    if name == "ShrunkVAR":
        features, target, _ = _var_features(
            benchmark, "train", int(parameters["lags"])
        )
        model = Ridge(alpha=float(parameters["alpha"]), solver="lsqr")
        model.fit(features, target)
    elif name == "HistGradientBoosting":
        features, target, _ = _long_tree_features(
            benchmark, "train", int(parameters["lags"])
        )
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(parameters["learning_rate"]),
            max_iter=int(parameters["max_iter"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            l2_regularization=float(parameters["l2_regularization"]),
            early_stopping=False,
            random_state=seed,
        )
        model.fit(features, target)
    else:
        raise ValueError(f"Unknown classical benchmark: {name}")
    joblib.dump(model, candidate_directory / "model.joblib")
    train_prediction = _predict_classical(name, model, benchmark, "train", parameters)
    val_prediction = _predict_classical(name, model, benchmark, "val", parameters)
    residual_pool = train_prediction["target"] - train_prediction["mean"]
    scenarios, _ = residual_bootstrap_scenarios(
        val_prediction["mean"],
        residual_pool,
        int(evaluation["validation_samples"]),
        int(evaluation["evaluation_seed"]),
    )
    metrics = evaluate_predictions(
        val_prediction["target"],
        val_prediction["mean"],
        scenarios,
        [float(value) for value in evaluation["alpha_levels"]],
    )
    reloaded = joblib.load(candidate_directory / "model.joblib")
    restored = _predict_classical(name, reloaded, benchmark, "val", parameters)["mean"]
    reload_difference = float(np.max(np.abs(restored - val_prediction["mean"])))
    if reload_difference >= 1e-10:
        raise RuntimeError(f"Classical model reload mismatch: {reload_difference}")
    return {
        "model_name": name,
        "parameters": parameters,
        "best_val_energy_score": float(metrics["energy_score"]),
        "validation_metrics": metrics,
        "training_seconds": float(time.perf_counter() - start),
        "checkpoint_reload_max_abs_difference": reload_difference,
        "checkpoint": str((candidate_directory / "model.joblib").resolve()),
    }


def evaluate_selected_classical_candidate(
    candidate_summary: dict[str, Any],
    benchmark: dict[str, Any],
    evaluation: dict[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=False)
    name = str(candidate_summary["model_name"])
    parameters = dict(candidate_summary["parameters"])
    model = joblib.load(candidate_summary["checkpoint"])
    train_prediction = _predict_classical(name, model, benchmark, "train", parameters)
    # This is the sole selected-model test access.
    test_prediction = _predict_classical(name, model, benchmark, "test", parameters)
    residual_pool = train_prediction["target"] - train_prediction["mean"]
    scenarios, residual_indices = residual_bootstrap_scenarios(
        test_prediction["mean"],
        residual_pool,
        int(evaluation["test_samples"]),
        int(evaluation["evaluation_seed"]) + 1,
    )
    metrics = evaluate_predictions(
        test_prediction["target"],
        test_prediction["mean"],
        scenarios,
        [float(value) for value in evaluation["alpha_levels"]],
    )
    per_origin = {
        "squared_error": per_origin_squared_error(
            test_prediction["target"], test_prediction["mean"]
        ),
        "absolute_error": per_origin_absolute_error(
            test_prediction["target"], test_prediction["mean"]
        ),
        "energy_score": per_origin_energy_score(scenarios, test_prediction["target"]),
    }
    np.savez_compressed(
        output_directory / "reproducible_predictions.npz",
        mean=test_prediction["mean"],
        target=test_prediction["target"],
        test_indices=test_prediction["indices"],
        train_residual_pool=residual_pool,
        scenario_residual_indices=residual_indices,
        **per_origin,
    )
    return {
        "model_name": name,
        "selected_parameters": parameters,
        "validation_energy_score": candidate_summary["best_val_energy_score"],
        "test_metrics": metrics,
        "test_access_count": 1,
        "checkpoint_reload_before_test": True,
        "test_samples": int(evaluation["test_samples"]),
        "scenario_semantics": (
            "point forecast plus centered empirical joint [H,N] training residual path; "
            "saved residual pool and draw indices exactly reconstruct scenarios"
        ),
    }
