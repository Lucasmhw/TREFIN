from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fconcord.benchmark_data import BenchmarkTensorDataset
from fconcord.benchmark_models import BenchmarkForecaster, make_benchmark_model
from fconcord.metrics import evaluate_predictions, point_metrics
from fconcord.statistical_tests import (
    per_origin_absolute_error,
    per_origin_energy_score,
    per_origin_squared_error,
)
from fconcord.utils import count_parameters, seed_everything


def residual_bootstrap_scenarios(
    mean: np.ndarray,
    residual_pool: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Add centered empirical joint path residuals to every point forecast."""
    if mean.ndim != 3 or residual_pool.ndim != 3 or mean.shape[1:] != residual_pool.shape[1:]:
        raise ValueError("mean and residual_pool must agree on [H,N]")
    if sample_count < 2 or len(residual_pool) < 2:
        raise ValueError("residual bootstrap requires at least two samples and residuals")
    centered = residual_pool - residual_pool.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(centered), size=(sample_count, len(mean)), dtype=np.int64
    )
    scenarios = mean[None] + centered[indices]
    return scenarios.astype(np.float32), indices


def _loader(
    benchmark: dict[str, Any], split: str, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        BenchmarkTensorDataset(benchmark, split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _model_output(model: BenchmarkForecaster, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["return_history"],
        state=batch["state"],
        asset_features=batch["asset_features"],
        market_history=batch["market_history"],
    )


@torch.no_grad()
def predict_neural_split(
    model: BenchmarkForecaster,
    benchmark: dict[str, Any],
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    scale = np.asarray(benchmark["arrays"]["return_scale"], dtype=np.float32)
    predictions, targets, indices = [], [], []
    for raw_batch in _loader(benchmark, split, batch_size, shuffle=False):
        batch = _move(raw_batch, device)
        scaled = _model_output(model, batch)
        predictions.append(scaled.detach().cpu().numpy() * scale[None, None])
        targets.append(batch["target"].detach().cpu().numpy())
        indices.append(batch["index"].detach().cpu().numpy())
    return {
        "mean": np.concatenate(predictions),
        "target": np.concatenate(targets),
        "indices": np.concatenate(indices),
    }


def _evaluation(
    train_prediction: dict[str, np.ndarray],
    evaluation_prediction: dict[str, np.ndarray],
    sample_count: int,
    seed: int,
    alpha_levels: list[float],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    residual_pool = train_prediction["target"] - train_prediction["mean"]
    scenarios, residual_indices = residual_bootstrap_scenarios(
        evaluation_prediction["mean"], residual_pool, sample_count, seed
    )
    metrics = evaluate_predictions(
        evaluation_prediction["target"],
        evaluation_prediction["mean"],
        scenarios,
        alpha_levels,
    )
    return metrics, scenarios, residual_indices


def train_neural_candidate(
    name: str,
    parameters: dict[str, Any],
    benchmark: dict[str, Any],
    training: dict[str, Any],
    candidate_directory: Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    candidate_directory.mkdir(parents=True, exist_ok=False)
    (candidate_directory / "checkpoints").mkdir()
    (candidate_directory / "logs").mkdir()
    logger = logging.getLogger(f"fconcord.benchmark.{candidate_directory}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(candidate_directory / "logs" / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

    seed_everything(seed, bool(training.get("deterministic", True)))
    dimensions = benchmark["metadata"]["dimensions"]
    factory = lambda: make_benchmark_model(name, dimensions, parameters)
    model = factory().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(training["epochs"]))
    )
    batch_size = int(training["batch_size"])
    alpha_levels = [float(value) for value in training["alpha_levels"]]
    history: list[dict[str, float]] = []
    best_value = float("inf")
    best_epoch = -1
    stale = 0
    start = time.perf_counter()
    logger.info(
        "candidate_started name=%s seed=%d parameters=%s trainable_parameters=%d",
        name,
        seed,
        parameters,
        count_parameters(model),
    )
    for epoch in range(1, int(training["epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimization_losses = []
        gradient_norms = []
        for raw_batch in _loader(benchmark, "train", batch_size, shuffle=True):
            batch = _move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _model_output(model, batch)
            loss = torch.mean((prediction - batch["target_scaled"]) ** 2)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["grad_clip"])
            )
            optimizer.step()
            optimization_losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))

        train_prediction = predict_neural_split(
            model, benchmark, "train", batch_size, device
        )
        val_prediction = predict_neural_split(model, benchmark, "val", batch_size, device)
        train_point = point_metrics(train_prediction["target"], train_prediction["mean"])
        val_metrics, _, _ = _evaluation(
            train_prediction,
            val_prediction,
            int(training["validation_samples"]),
            int(training["evaluation_seed"]),
            alpha_levels,
        )
        scheduler.step()
        record = {
            "epoch": float(epoch),
            "optimization_train_mse_scaled": float(np.mean(optimization_losses)),
            "train_rmse": float(train_point["rmse"]),
            "train_mae": float(train_point["mae"]),
            "train_directional_accuracy": float(train_point["directional_accuracy"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_mae": float(val_metrics["mae"]),
            "val_directional_accuracy": float(val_metrics["directional_accuracy"]),
            "val_energy_score": float(val_metrics["energy_score"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm": float(np.mean(gradient_norms)),
            "epoch_seconds": float(time.perf_counter() - epoch_start),
        }
        history.append(record)
        pd.DataFrame(history).to_csv(
            candidate_directory / "logs" / "training_history.csv", index=False
        )
        logger.info(
            "epoch=%d train_rmse=%.8f train_mae=%.8f val_rmse=%.8f val_mae=%.8f "
            "val_es=%.8f grad=%.6f seconds=%.2f",
            epoch,
            record["train_rmse"],
            record["train_mae"],
            record["val_rmse"],
            record["val_mae"],
            record["val_energy_score"],
            record["gradient_norm"],
            record["epoch_seconds"],
        )
        checkpoint = {
            "model_name": name,
            "parameters": parameters,
            "dimensions": dimensions,
            "epoch": epoch,
            "validation_energy_score": record["val_energy_score"],
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(checkpoint, candidate_directory / "checkpoints" / "last.pt")
        if record["val_energy_score"] < best_value - float(training["selection_min_delta"]):
            best_value = record["val_energy_score"]
            best_epoch = epoch
            stale = 0
            torch.save(checkpoint, candidate_directory / "checkpoints" / "best.pt")
        else:
            stale += 1
        if stale >= int(training["patience"]):
            logger.info("early_stop epoch=%d best_epoch=%d", epoch, best_epoch)
            break

    checkpoint = torch.load(
        candidate_directory / "checkpoints" / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    reference = predict_neural_split(model, benchmark, "val", batch_size, device)["mean"]
    reloaded = factory().to(device)
    reloaded.load_state_dict(checkpoint["model_state"])
    restored = predict_neural_split(reloaded, benchmark, "val", batch_size, device)["mean"]
    reload_difference = float(np.max(np.abs(reference - restored)))
    if reload_difference >= 1e-7:
        raise RuntimeError(f"Benchmark checkpoint reload mismatch: {reload_difference}")
    logger.info(
        "training_complete best_epoch=%d best_val_energy_score=%.8f reload_difference=%.3e",
        best_epoch,
        best_value,
        reload_difference,
    )
    return {
        "model_name": name,
        "parameters": parameters,
        "trainable_parameters": count_parameters(reloaded),
        "best_epoch": best_epoch,
        "best_val_energy_score": best_value,
        "epochs_completed": len(history),
        "training_seconds": float(time.perf_counter() - start),
        "checkpoint_reload_max_abs_difference": reload_difference,
        "checkpoint": str((candidate_directory / "checkpoints" / "best.pt").resolve()),
    }


def evaluate_selected_neural_candidate(
    candidate_summary: dict[str, Any],
    benchmark: dict[str, Any],
    training: dict[str, Any],
    output_directory: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Fresh-load the validation-selected candidate and touch the test split once."""
    output_directory.mkdir(parents=True, exist_ok=False)
    dimensions = benchmark["metadata"]["dimensions"]
    model = make_benchmark_model(
        str(candidate_summary["model_name"]),
        dimensions,
        dict(candidate_summary["parameters"]),
    ).to(device)
    checkpoint = torch.load(
        candidate_summary["checkpoint"], map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    batch_size = int(training["batch_size"])
    train_prediction = predict_neural_split(model, benchmark, "train", batch_size, device)
    # This is the sole test access in the selected-candidate path.
    test_prediction = predict_neural_split(model, benchmark, "test", batch_size, device)
    residual_pool = train_prediction["target"] - train_prediction["mean"]
    scenarios, residual_indices = residual_bootstrap_scenarios(
        test_prediction["mean"],
        residual_pool,
        int(training["test_samples"]),
        int(training["evaluation_seed"]) + 1,
    )
    metrics = evaluate_predictions(
        test_prediction["target"],
        test_prediction["mean"],
        scenarios,
        [float(value) for value in training["alpha_levels"]],
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
        "model_name": candidate_summary["model_name"],
        "selected_parameters": candidate_summary["parameters"],
        "best_epoch": candidate_summary["best_epoch"],
        "validation_energy_score": candidate_summary["best_val_energy_score"],
        "test_metrics": metrics,
        "test_access_count": 1,
        "checkpoint_reload_before_test": True,
        "scenario_semantics": (
            "point forecast plus centered empirical joint [H,N] training residual path; "
            "saved residual pool and draw indices exactly reconstruct scenarios"
        ),
        "test_samples": int(training["test_samples"]),
    }
