from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from fconcord.metrics import energy_score_numpy, point_metrics


def _validate_prediction_arrays(
    mean: np.ndarray, samples: np.ndarray, target: np.ndarray
) -> None:
    if mean.shape != target.shape or mean.ndim != 3:
        raise ValueError("mean and target must share shape [B,H,N]")
    if samples.ndim != 4 or samples.shape[1:] != target.shape:
        raise ValueError("samples must have shape [S,B,H,N]")
    if samples.shape[0] < 2:
        raise ValueError("signal calibration requires at least two scenarios")
    if not all(np.isfinite(value).all() for value in (mean, samples, target)):
        raise ValueError("signal calibration inputs must be finite")


def calibrate_predictable_signal(
    mean: np.ndarray,
    samples: np.ndarray,
    target: np.ndarray,
    lower: float = 0.05,
    upper: float = 1.0,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Fit a positive validation-only gate on the predictable return component.

    F-CONCORD decomposes a scenario into a conditional-mean signal and an
    innovation: ``Y^(s) = mu + epsilon^(s)``.  For a gate ``rho``, calibration
    returns ``rho * mu + epsilon^(s)``.  Every pairwise scenario difference is
    therefore unchanged, so the second (sharpness) term of the energy score is
    constant in ``rho``.  The remaining term is a sum of Euclidean norms of
    affine functions of ``rho`` and is convex.  Bounded scalar minimization thus
    finds the global validation optimum without inspecting the test split.

    A strictly positive lower bound preserves the sign, cross-horizon
    information coefficient, and economic interpretation of every nonzero
    conditional-mean forecast.
    """

    _validate_prediction_arrays(mean, samples, target)
    if not (0.0 < lower <= upper <= 1.0):
        raise ValueError("signal-shrinkage bounds must satisfy 0 < lower <= upper <= 1")
    if tolerance <= 0.0:
        raise ValueError("signal-shrinkage tolerance must be positive")

    mean64 = np.asarray(mean, dtype=np.float64)
    samples64 = np.asarray(samples, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    innovations = samples64 - mean64[None]

    def first_energy_term(rho: float) -> float:
        error = innovations + float(rho) * mean64[None] - target64[None]
        return float(np.linalg.norm(error, axis=-1).mean())

    if lower == upper:
        selected = float(lower)
        optimizer_success = True
        optimizer_evaluations = 1
    else:
        optimization = minimize_scalar(
            first_energy_term,
            bounds=(float(lower), float(upper)),
            method="bounded",
            options={"xatol": float(tolerance)},
        )
        candidates = [float(lower), float(upper), float(optimization.x)]
        selected = min(candidates, key=lambda rho: (first_energy_term(rho), rho))
        optimizer_success = bool(optimization.success)
        optimizer_evaluations = int(optimization.nfev) + 2

    calibrated_mean = selected * mean64
    calibrated_samples = innovations + calibrated_mean[None]
    uncalibrated_metrics = point_metrics(target64, mean64)
    uncalibrated_metrics["energy_score"] = energy_score_numpy(samples64, target64)
    calibrated_metrics = point_metrics(target64, calibrated_mean)
    calibrated_metrics["energy_score"] = energy_score_numpy(
        calibrated_samples, target64
    )
    pairwise_invariance_max_error = float(
        np.max(
            np.abs(
                (calibrated_samples[1] - calibrated_samples[0])
                - (samples64[1] - samples64[0])
            )
        )
    )
    return {
        "selected_signal_shrinkage": selected,
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "tolerance": float(tolerance),
        "selection_split": "validation",
        "selection_metric": "energy_score",
        "convex_objective": True,
        "pairwise_energy_term_invariant": True,
        "pairwise_invariance_max_error": pairwise_invariance_max_error,
        "optimizer_success": optimizer_success,
        "optimizer_evaluations": optimizer_evaluations,
        "uncalibrated_validation_metrics": uncalibrated_metrics,
        "calibrated_validation_metrics": calibrated_metrics,
    }
