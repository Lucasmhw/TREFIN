from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import pdist


def energy_score_numpy(samples: np.ndarray, target: np.ndarray) -> float:
    if samples.shape[0] < 2:
        raise ValueError("Energy score requires at least two Monte Carlo samples")
    first = np.linalg.norm(samples - target[None], axis=-1).mean(axis=0)
    sample_count, batch_size, horizon, _ = samples.shape
    second = np.empty((batch_size, horizon), dtype=np.float64)
    denominator = sample_count * (sample_count - 1)
    for batch_index in range(batch_size):
        for lead_index in range(horizon):
            # pdist returns each unordered pair once.  That is exactly the
            # manual's 1/[2S(S-1)] ordered double sum after cancelling the two.
            second[batch_index, lead_index] = (
                pdist(samples[:, batch_index, lead_index], metric="euclidean").sum()
                / denominator
            )
    return float(np.mean(first - second))


def multi_step_information_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """FinTSBridge msIC/msIR using its released-code semantics.

    Both inputs have shape [B,H,N]. Despite the paper's use of the phrase
    "rank correlation", the official implementation computes population
    Pearson correlation along the horizon axis for every origin and asset.
    We reproduce that code-level definition, retain full precision, and make
    constant paths finite by assigning zero correlation and reporting their
    fraction explicitly.
    """
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError("msIC/msIR inputs must share shape [B,H,N]")
    if target.shape[1] < 2:
        raise ValueError("msIC/msIR require at least two forecast leads")
    target_centered = target - target.mean(axis=1, keepdims=True)
    prediction_centered = prediction - prediction.mean(axis=1, keepdims=True)
    covariance = np.mean(target_centered * prediction_centered, axis=1)
    denominator = target.std(axis=1) * prediction.std(axis=1)
    valid = denominator > epsilon
    correlations = np.zeros_like(covariance, dtype=np.float64)
    correlations[valid] = covariance[valid] / denominator[valid]
    per_asset_ic = np.mean(correlations, axis=0)
    per_asset_std = np.std(correlations, axis=0)
    per_asset_ir = per_asset_ic / np.maximum(per_asset_std, epsilon)
    return {
        "msic": float(np.mean(per_asset_ic)),
        "msir": float(np.mean(per_asset_ir)),
        "msic_per_asset": per_asset_ic.tolist(),
        "msir_per_asset": per_asset_ir.tolist(),
        "valid_origin_asset_fraction": float(np.mean(valid)),
        "definition": (
            "FinTSBridge released-code population Pearson correlation along horizon; "
            "msIC averaged by origin/asset and msIR averaged from per-asset IC/std"
        ),
    }


def evaluate_predictions(
    target: np.ndarray,
    mean: np.ndarray,
    samples: np.ndarray,
    alpha_levels: list[float],
) -> dict[str, Any]:
    error = mean - target
    mse = float(np.mean(error**2))
    metrics: dict[str, Any] = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "directional_accuracy": float(np.mean(np.sign(mean) == np.sign(target))),
        "energy_score": energy_score_numpy(samples, target),
        "horizon": [],
        "coverage": {},
        "interval_width": {},
    }
    metrics.update(multi_step_information_metrics(target, mean))
    for h in range(target.shape[1]):
        h_error = error[:, h]
        h_mse = float(np.mean(h_error**2))
        metrics["horizon"].append(
            {
                "h": h + 1,
                "mse": h_mse,
                "rmse": float(np.sqrt(h_mse)),
                "mae": float(np.mean(np.abs(h_error))),
                "directional_accuracy": float(
                    np.mean(np.sign(mean[:, h]) == np.sign(target[:, h]))
                ),
                "energy_score": energy_score_numpy(samples[:, :, h : h + 1], target[:, h : h + 1]),
            }
        )
    for alpha in alpha_levels:
        tail = (1.0 - alpha) / 2.0
        lower = np.quantile(samples, tail, axis=0)
        upper = np.quantile(samples, 1.0 - tail, axis=0)
        metrics["coverage"][str(alpha)] = float(np.mean((target >= lower) & (target <= upper)))
        metrics["interval_width"][str(alpha)] = float(np.mean(upper - lower))
    return metrics


def point_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    mse = float(np.mean(error**2))
    information = multi_step_information_metrics(target, prediction)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "directional_accuracy": float(np.mean(np.sign(prediction) == np.sign(target))),
        "msic": float(information["msic"]),
        "msir": float(information["msir"]),
    }
