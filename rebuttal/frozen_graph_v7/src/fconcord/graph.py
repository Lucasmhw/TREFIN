from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.covariance._graph_lasso import _graphical_lasso
from sklearn.exceptions import ConvergenceWarning


def estimate_partial_correlation(
    standardized_residuals: np.ndarray,
    chronological_weights: np.ndarray,
    alpha: float,
    covariance_init: np.ndarray | None = None,
    return_diagnostics: bool = False,
    max_iter: int = 500,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
]:
    """Return covariance, precision and residual partial correlation.

    Graphical Lasso is retried with stronger regularization when its coordinate
    descent solver reports non-convergence. The final ridge inverse is an
    explicit, finite fallback rather than a silently accepted unconverged fit.
    """
    if max_iter < 1 or tolerance <= 0.0:
        raise ValueError("Graphical-Lasso max_iter and tolerance must be positive")
    covariance = standardized_residuals.T @ (
        chronological_weights[:, None] * standardized_residuals
    )
    covariance = 0.5 * (covariance + covariance.T)
    covariance.flat[:: covariance.shape[0] + 1] += 1e-6
    shrunk_covariance: np.ndarray | None = None
    precision: np.ndarray | None = None
    diagnostics: dict[str, Any] = {
        "converged": False,
        "fallback": False,
        "attempts": 0,
        "alpha_requested": float(alpha),
        "alpha_used": float(alpha),
        "iterations": 0,
        "warm_start_used": covariance_init is not None,
        "solver_max_iter": int(max_iter),
        "solver_tolerance": float(tolerance),
    }
    for multiplier in (1.0, 1.5, 2.25):
        alpha_used = float(alpha * multiplier)
        diagnostics["attempts"] += 1
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                candidate_covariance, candidate_precision, _, iterations = _graphical_lasso(
                    covariance,
                    alpha=alpha_used,
                    cov_init=covariance_init,
                    max_iter=max_iter,
                    tol=tolerance,
                )
            warned = any(issubclass(item.category, ConvergenceWarning) for item in caught)
            finite = np.isfinite(candidate_covariance).all() and np.isfinite(
                candidate_precision
            ).all()
            positive = bool(np.linalg.eigvalsh(candidate_precision).min() > 0.0)
            diagnostics["alpha_used"] = alpha_used
            diagnostics["iterations"] = int(iterations)
            if not warned and finite and positive:
                shrunk_covariance = candidate_covariance
                precision = candidate_precision
                diagnostics["converged"] = True
                break
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            continue
    if shrunk_covariance is None or precision is None:
        ridge = max(float(alpha), 1e-3)
        shrunk_covariance = covariance + ridge * np.eye(covariance.shape[0])
        precision = np.linalg.pinv(shrunk_covariance, hermitian=True)
        diagnostics["fallback"] = True
        diagnostics["alpha_used"] = ridge
        diagnostics["iterations"] = 0
    diagonal = np.sqrt(np.maximum(np.diag(precision), 1e-12))
    partial = -precision / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial, 0.0)
    partial = np.clip(0.5 * (partial + partial.T), -1.0, 1.0)
    result = (shrunk_covariance, precision, partial)
    if return_diagnostics:
        return (*result, diagnostics)
    return result


def smooth_signed_graph(partial: np.ndarray, center: float, temperature: float) -> np.ndarray:
    gate = 1.0 / (1.0 + np.exp(-(np.abs(partial) - center) / temperature))
    graph = partial * gate
    graph = 0.5 * (graph + graph.T)
    np.fill_diagonal(graph, 0.0)
    return graph


def project_graph_numpy(graph: np.ndarray, bound: float) -> np.ndarray:
    graph = 0.5 * (graph + graph.T)
    np.fill_diagonal(graph, 0.0)
    norm = np.linalg.norm(graph)
    if norm > bound:
        graph = graph * (bound / norm)
    return graph
