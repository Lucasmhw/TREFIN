from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import norm


def per_origin_squared_error(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Mean squared error over horizon and assets for each forecast origin."""
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError("target and prediction must share shape [B,H,N]")
    return np.mean((prediction - target) ** 2, axis=(1, 2), dtype=np.float64)


def per_origin_absolute_error(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Mean absolute error over horizon and assets for each forecast origin."""
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError("target and prediction must share shape [B,H,N]")
    return np.mean(np.abs(prediction - target), axis=(1, 2), dtype=np.float64)


def per_origin_energy_score(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact finite-sample Energy Score averaged over leads, one value per origin."""
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("samples and target must have shape [S,B,H,N] and [B,H,N]")
    if samples.shape[1:] != target.shape:
        raise ValueError("sample and target dimensions do not agree")
    sample_count, batch_size, horizon, _ = samples.shape
    if sample_count < 2:
        raise ValueError("Energy Score requires at least two samples")
    first = np.linalg.norm(samples - target[None], axis=-1).mean(axis=0)
    second = np.empty((batch_size, horizon), dtype=np.float64)
    denominator = sample_count * (sample_count - 1)
    for origin in range(batch_size):
        for lead in range(horizon):
            second[origin, lead] = (
                pdist(samples[:, origin, lead], metric="euclidean").sum()
                / denominator
            )
    return np.mean(first - second, axis=1)


def newey_west_long_run_variance(values: np.ndarray, max_lag: int) -> float:
    """Bartlett-kernel HAC long-run variance of a scalar loss-difference series."""
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(series)) or len(series) < 2:
        raise ValueError("values must contain at least two finite observations")
    lag = min(max(int(max_lag), 0), len(series) - 1)
    centered = series - series.mean()
    variance = float(np.dot(centered, centered) / len(series))
    for offset in range(1, lag + 1):
        covariance = float(
            np.dot(centered[offset:], centered[:-offset]) / len(series)
        )
        weight = 1.0 - offset / (lag + 1.0)
        variance += 2.0 * weight * covariance
    return max(variance, 0.0)


def diebold_mariano_hac(
    candidate_loss: np.ndarray,
    reference_loss: np.ndarray,
    horizon: int,
    max_lag: int | None = None,
) -> dict[str, float | int | str]:
    """Two-sided Diebold–Mariano test with overlap-aware Newey–West correction.

    The loss difference is candidate minus reference, so a negative effect favors
    the candidate. The Harvey finite-sample correction is applied when its radicand
    is positive. A zero HAC variance is handled explicitly rather than producing NaN.
    """
    candidate = np.asarray(candidate_loss, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_loss, dtype=np.float64).reshape(-1)
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference losses must have equal length")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    difference = candidate - reference
    count = len(difference)
    if count < 2:
        raise ValueError("DM test requires at least two origins")
    lag = horizon - 1 if max_lag is None else int(max_lag)
    long_run_variance = newey_west_long_run_variance(difference, lag)
    effect = float(difference.mean())
    standard_error = float(np.sqrt(long_run_variance / count))
    if standard_error <= np.finfo(np.float64).eps:
        statistic = 0.0 if abs(effect) <= np.finfo(np.float64).eps else np.sign(effect) * np.inf
    else:
        statistic = effect / standard_error
    radicand = (count + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / count) / count
    harvey_correction = float(np.sqrt(max(radicand, 0.0)))
    corrected = float(statistic * harvey_correction)
    p_value = float(2.0 * norm.sf(abs(corrected)))
    return {
        "origins": count,
        "horizon": int(horizon),
        "hac_lag": min(max(lag, 0), count - 1),
        "mean_loss_difference_candidate_minus_reference": effect,
        "hac_long_run_variance": long_run_variance,
        "standard_error": standard_error,
        "harvey_correction": harvey_correction,
        "dm_statistic": corrected,
        "two_sided_p_value": p_value,
        "direction": "candidate_better" if effect < 0.0 else "reference_better_or_tied",
    }


def circular_block_bootstrap_mean_difference(
    candidate_loss: np.ndarray,
    reference_loss: np.ndarray,
    block_length: int,
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 271828,
) -> dict[str, Any]:
    """Paired circular-block bootstrap for the mean loss difference."""
    candidate = np.asarray(candidate_loss, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_loss, dtype=np.float64).reshape(-1)
    if candidate.shape != reference.shape or len(candidate) < 2:
        raise ValueError("paired losses must have equal length of at least two")
    if block_length < 1 or repetitions < 100:
        raise ValueError("block_length must be positive and repetitions at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    difference = candidate - reference
    count = len(difference)
    block = min(int(block_length), count)
    blocks_needed = int(np.ceil(count / block))
    offsets = np.arange(block, dtype=np.int64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions, dtype=np.float64)
    for draw in range(repetitions):
        starts = rng.integers(0, count, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % count).reshape(-1)[:count]
        bootstrap[draw] = float(difference[indices].mean())
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap, [tail, 1.0 - tail])
    probability_nonpositive = float(np.mean(bootstrap <= 0.0))
    probability_nonnegative = float(np.mean(bootstrap >= 0.0))
    return {
        "origins": count,
        "block_length": block,
        "repetitions": int(repetitions),
        "confidence": float(confidence),
        "seed": int(seed),
        "mean_loss_difference_candidate_minus_reference": float(difference.mean()),
        "confidence_interval": [float(lower), float(upper)],
        "two_sided_bootstrap_p_value": float(
            min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
        ),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm family-wise-error adjusted p-values, preserving the input keys."""
    if not p_values:
        return {}
    for name, value in p_values.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Invalid p-value for {name}: {value}")
    ordered = sorted(p_values, key=lambda name: (float(p_values[name]), name))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        raw = float(p_values[name])
        running = max(running, min(1.0, (count - rank) * raw))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def _circular_block_indices(
    count: int, block_length: int, repetitions: int, seed: int
) -> np.ndarray:
    if count < 2 or block_length < 1 or repetitions < 100:
        raise ValueError("Invalid circular-block bootstrap dimensions")
    block = min(int(block_length), count)
    blocks_needed = int(np.ceil(count / block))
    offsets = np.arange(block, dtype=np.int64)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, count, size=(repetitions, blocks_needed))
    return ((starts[:, :, None] + offsets[None, None]) % count).reshape(
        repetitions, -1
    )[:, :count]


def block_bootstrap_spa(
    candidate_losses: dict[str, np.ndarray],
    reference_loss: np.ndarray,
    block_length: int,
    repetitions: int = 2000,
    seed: int = 271828,
) -> dict[str, Any]:
    """Studentized block-bootstrap SPA test for any improvement over a reference.

    Positive loss advantage means the reference loss exceeds the candidate loss.
    The null is that no registered candidate has positive expected advantage.
    Bootstrap series are recentered at their observed means to impose the null.
    """
    if not candidate_losses:
        raise ValueError("SPA requires at least one candidate")
    reference = np.asarray(reference_loss, dtype=np.float64).reshape(-1)
    names = sorted(candidate_losses)
    advantages = np.column_stack(
        [reference - np.asarray(candidate_losses[name], dtype=np.float64).reshape(-1) for name in names]
    )
    if advantages.shape[0] != len(reference) or not np.isfinite(advantages).all():
        raise ValueError("SPA losses must be finite and have a shared origin count")
    count = len(reference)
    lag = min(int(block_length) - 1, count - 1)
    scales = np.asarray(
        [
            np.sqrt(max(newey_west_long_run_variance(advantages[:, index], lag), 1e-16))
            for index in range(len(names))
        ]
    )
    observed_by_model = np.sqrt(count) * advantages.mean(axis=0) / scales
    observed = float(max(0.0, observed_by_model.max()))
    indices = _circular_block_indices(count, block_length, repetitions, seed)
    centered = advantages - advantages.mean(axis=0, keepdims=True)
    bootstrap_means = centered[indices].mean(axis=1)
    bootstrap_statistics = np.maximum(
        0.0, np.sqrt(count) * bootstrap_means / scales[None]
    ).max(axis=1)
    p_value = float((1 + np.count_nonzero(bootstrap_statistics >= observed)) / (repetitions + 1))
    best_index = int(np.argmax(advantages.mean(axis=0)))
    return {
        "null": "no candidate has lower expected loss than the reference",
        "origins": count,
        "block_length": min(int(block_length), count),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "spa_statistic": observed,
        "p_value": p_value,
        "best_candidate": names[best_index],
        "mean_advantage_reference_minus_candidate": {
            name: float(advantages[:, index].mean()) for index, name in enumerate(names)
        },
        "studentized_statistics": {
            name: float(observed_by_model[index]) for index, name in enumerate(names)
        },
    }


def model_confidence_set(
    losses: dict[str, np.ndarray],
    block_length: int,
    alpha: float = 0.05,
    repetitions: int = 2000,
    seed: int = 271828,
) -> dict[str, Any]:
    """Sequential range-statistic Model Confidence Set with common block draws."""
    if len(losses) < 2:
        raise ValueError("MCS requires at least two models")
    if not 0.0 < alpha < 1.0:
        raise ValueError("MCS alpha must lie strictly between zero and one")
    names = sorted(losses)
    matrix = np.column_stack(
        [np.asarray(losses[name], dtype=np.float64).reshape(-1) for name in names]
    )
    if matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ValueError("MCS losses must be finite and contain multiple origins")
    if any(len(np.asarray(losses[name]).reshape(-1)) != matrix.shape[0] for name in names):
        raise ValueError("MCS models must share an origin count")
    count = matrix.shape[0]
    indices = _circular_block_indices(count, block_length, repetitions, seed)
    bootstrap_means = matrix[indices].mean(axis=1)
    observed_means = matrix.mean(axis=0)
    centered_bootstrap = bootstrap_means - observed_means[None]
    active = list(range(len(names)))
    elimination_path: list[dict[str, Any]] = []
    while len(active) > 1:
        pair_statistics = []
        bootstrap_pair_statistics = []
        for first_position, first in enumerate(active):
            for second in active[first_position + 1 :]:
                bootstrap_difference = (
                    centered_bootstrap[:, first] - centered_bootstrap[:, second]
                )
                standard_error = max(float(bootstrap_difference.std(ddof=1)), 1e-12)
                pair_statistics.append(
                    abs(float(observed_means[first] - observed_means[second]))
                    / standard_error
                )
                bootstrap_pair_statistics.append(
                    np.abs(bootstrap_difference) / standard_error
                )
        observed_range = float(max(pair_statistics))
        bootstrap_range = np.stack(bootstrap_pair_statistics, axis=1).max(axis=1)
        p_value = float(
            (1 + np.count_nonzero(bootstrap_range >= observed_range))
            / (repetitions + 1)
        )
        record: dict[str, Any] = {
            "models": [names[index] for index in active],
            "range_statistic": observed_range,
            "p_value": p_value,
        }
        if p_value >= alpha:
            record["eliminated"] = None
            elimination_path.append(record)
            break
        worst = max(active, key=lambda index: (observed_means[index], names[index]))
        record["eliminated"] = names[worst]
        elimination_path.append(record)
        active.remove(worst)
    return {
        "confidence_level": 1.0 - alpha,
        "origins": count,
        "block_length": min(int(block_length), count),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "included_models": [names[index] for index in active],
        "excluded_models": [name for name in names if names.index(name) not in active],
        "mean_losses": {name: float(observed_means[index]) for index, name in enumerate(names)},
        "elimination_path": elimination_path,
    }
