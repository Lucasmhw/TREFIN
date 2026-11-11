from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fconcord.concepts import (
    ConceptSpec,
    RobustStandardizer,
    compute_asset_concepts,
    exponential_lag_weights,
    fit_weighted_ridge_factor_model,
)
from fconcord.graph import estimate_partial_correlation, project_graph_numpy, smooth_signed_graph


def load_panel_csv(
    input_path: str | Path,
    factor_path: str | Path,
    data_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a balanced point-in-time panel from documented CSV schemas.

    Panel columns: date, asset, adjusted price, dollar_volume, and optionally
    order_flow, market_weight and available_at.  Factor columns: date,
    risk_free and numeric factor-return columns.  A zero risk-free series is
    accepted only when explicitly registered in the data configuration.
    """
    data_config = {} if data_config is None else data_config
    order_flow_mode = str(data_config.get("order_flow_mode", "provided_interval"))
    risk_free_column = str(data_config.get("risk_free_column", "risk_free"))
    risk_free_policy = str(data_config.get("risk_free_policy", "require_column"))
    market_weight_policy = str(data_config.get("market_weight_policy", "equal_weight"))
    panel = pd.read_csv(input_path, parse_dates=["date"])
    factors_frame = pd.read_csv(factor_path, parse_dates=["date"])
    # Normalize all observation keys to timezone-naive UTC.  This keeps daily
    # and intraday sources comparable and avoids platform-dependent casts from
    # timezone-aware pandas timestamps to NumPy datetime64.
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_localize(None)
    factors_frame["date"] = pd.to_datetime(
        factors_frame["date"], utc=True
    ).dt.tz_localize(None)
    required = {"date", "asset", "price", "dollar_volume"}
    if order_flow_mode.startswith("provided"):
        required.add("order_flow")
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")
    if panel.duplicated(["date", "asset"]).any():
        raise ValueError("Panel contains duplicate date-asset observations")
    if factors_frame["date"].duplicated().any():
        raise ValueError("Factor file contains duplicate dates")
    for name, frame in (("panel", panel), ("factor", factors_frame)):
        if "available_at" in frame.columns:
            available = pd.to_datetime(frame["available_at"], utc=True)
            observed = pd.to_datetime(frame["date"], utc=True)
            if (available > observed).any():
                raise ValueError(
                    f"{name} contains an availability timestamp after its observation date"
                )
    assets = sorted(panel["asset"].unique())
    dates = sorted(set(panel["date"]) & set(factors_frame["date"]))
    if not dates:
        raise ValueError("Panel and factor files have no overlapping dates")

    def pivot(column: str) -> np.ndarray:
        frame = panel.pivot(index="date", columns="asset", values=column).reindex(
            index=dates, columns=assets
        )
        if frame.isna().any().any():
            raise ValueError("Primary loader requires a balanced panel; preprocess missing data first")
        return frame.to_numpy(dtype=np.float64)

    prices = pivot("price")
    dollar_volume = pivot("dollar_volume")
    if order_flow_mode.startswith("provided"):
        order_flow = pivot("order_flow")
    else:
        order_flow = np.zeros_like(prices)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Prices must be finite and strictly positive")
    if not np.isfinite(dollar_volume).all() or np.any(dollar_volume <= 0):
        raise ValueError("Dollar volume must be finite and strictly positive")
    if not np.isfinite(order_flow).all():
        raise ValueError("Order flow must be finite")
    total_log_returns = np.zeros_like(prices)
    total_log_returns[1:] = np.log(prices[1:] / prices[:-1])
    indexed_factors = factors_frame.set_index("date").reindex(dates)
    if risk_free_column in indexed_factors.columns:
        risk_free = indexed_factors[risk_free_column].to_numpy(dtype=np.float64)
        risk_free_source = f"column:{risk_free_column}"
    elif risk_free_policy == "explicit_zero":
        risk_free = np.zeros(len(dates), dtype=np.float64)
        risk_free_source = "explicit_zero"
    else:
        raise ValueError(
            f"Factor file is missing {risk_free_column!r}; set risk_free_policy="
            "'explicit_zero' only for a preregistered zero-rate task"
        )
    if not np.isfinite(risk_free).all():
        raise ValueError("Risk-free returns must be finite")
    returns = total_log_returns - risk_free[:, None]
    excluded_factor_columns = {"date", "available_at", risk_free_column}
    factor_columns = [
        column for column in factors_frame.columns if column not in excluded_factor_columns
    ]
    if not factor_columns:
        raise ValueError("Factor file must contain at least one factor-return column")
    factor_values = indexed_factors[factor_columns]
    if factor_values.isna().any().any():
        raise ValueError("Factor data contain missing values after point-in-time alignment")
    try:
        factor_array = factor_values.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Every factor-return column must be numeric") from error
    if not np.isfinite(factor_array).all():
        raise ValueError("Factor returns must be finite")
    if "market_weight" in panel.columns:
        market_weights = pivot("market_weight")
        if np.any(market_weights < 0.0):
            raise ValueError("Market weights must be non-negative")
        row_sums = market_weights.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("Every market-weight row must have positive mass")
        market_weights = market_weights / row_sums
        market_weight_source = "point_in_time_panel_column"
    elif market_weight_policy == "equal_weight":
        market_weights = np.full_like(prices, 1.0 / len(assets))
        market_weight_source = "preregistered_equal_weight"
    else:
        raise ValueError(
            "Panel has no market_weight column and market_weight_policy is not equal_weight"
        )
    return {
        "dates": np.asarray(dates, dtype="datetime64[ns]"),
        "asset_names": np.asarray(assets),
        "factor_names": np.asarray(factor_columns),
        "prices": prices,
        "total_log_returns": total_log_returns,
        "returns": returns,
        "risk_free": risk_free,
        "factors": factor_array,
        "dollar_volume": dollar_volume,
        "order_flow": order_flow,
        "order_flow_mode": order_flow_mode,
        "market_weights": market_weights,
        "risk_free_source": risk_free_source,
        "market_weight_source": market_weight_source,
    }


@dataclass
class _State:
    concepts: np.ndarray
    graph: np.ndarray
    market: np.ndarray
    beta: np.ndarray
    raw_corr: np.ndarray
    partial_corr: np.ndarray
    graph_covariance: np.ndarray
    graph_diagnostics: dict[str, Any]


def _select_graph_alpha(
    panel: dict[str, Any],
    config: dict[str, Any],
    first_origin: int,
    last_train_origin: int,
) -> tuple[float, dict[str, Any]]:
    """Select Graphical-Lasso alpha only inside the training time block.

    Each candidate is fitted at a causal origin and scored on the immediately
    following training observation by Gaussian negative log likelihood.  A
    registered edge-stability term penalizes rapid changes in the partial-
    correlation support.  No validation or test observation participates.
    """
    candidates = [
        float(value)
        for value in config.get(
            "graph_alpha_candidates", [0.01, 0.03, 0.1, 0.3]
        )
    ]
    if not candidates or any(value <= 0.0 for value in candidates):
        raise ValueError("graph_alpha_candidates must contain positive values")
    if str(config.get("graph_alpha_selection", "rolling_train")) == "fixed":
        selected = float(config.get("graph_alpha", candidates[0]))
        return selected, {
            "policy": "fixed",
            "selected": selected,
            "candidates": candidates,
            "uses_validation_or_test": False,
        }

    returns = np.asarray(panel["returns"], dtype=np.float64)
    factors = np.asarray(panel["factors"], dtype=np.float64)
    window = int(config["graph_window"])
    weights = exponential_lag_weights(window)
    chronological_weights = weights[::-1]
    available = np.arange(max(first_origin, window - 1), last_train_origin)
    requested = int(config.get("graph_alpha_selection_origins", 24))
    if len(available) > requested:
        available = available[
            np.unique(np.linspace(0, len(available) - 1, requested, dtype=int))
        ]
    if len(available) < 2:
        raise ValueError("Too few training-only origins for graph-alpha selection")
    stability_weight = float(config.get("graph_edge_stability_weight", 0.05))
    gate_center = float(config["graph_gate_center"])
    candidate_rows: list[dict[str, Any]] = []
    for alpha in candidates:
        nll_values: list[float] = []
        instability_values: list[float] = []
        previous_edges: np.ndarray | None = None
        covariance_init: np.ndarray | None = None
        fallback_count = 0
        for t in available:
            start = int(t) - window + 1
            returns_window = returns[start : int(t) + 1]
            factors_window = factors[start : int(t) + 1]
            intercept, beta = fit_weighted_ridge_factor_model(
                returns_window,
                factors_window,
                weights,
                float(config["ridge_alpha"]),
            )
            residuals = returns_window - (
                intercept[None, :] + factors_window @ beta.T
            )
            mean = np.sum(chronological_weights[:, None] * residuals, axis=0)
            scale = np.sqrt(
                np.sum(
                    chronological_weights[:, None] * (residuals - mean) ** 2,
                    axis=0,
                )
                + 1e-10
            )
            standardized = (residuals - mean) / scale
            covariance, precision, partial, diagnostics = estimate_partial_correlation(
                standardized,
                chronological_weights,
                alpha,
                covariance_init=covariance_init,
                return_diagnostics=True,
                max_iter=int(config.get("graph_solver_max_iter", 500)),
                tolerance=float(config.get("graph_solver_tolerance", 1e-4)),
            )
            covariance_init = covariance
            fallback_count += int(diagnostics["fallback"])
            next_residual = (
                returns[int(t) + 1]
                - intercept
                - beta @ factors[int(t) + 1]
            ) / scale
            sign, logdet = np.linalg.slogdet(precision)
            if sign <= 0.0:
                nll = float("inf")
            else:
                nll = 0.5 * float(next_residual @ precision @ next_residual - logdet)
            nll_values.append(nll)
            edges = np.abs(partial) >= gate_center
            np.fill_diagonal(edges, False)
            if previous_edges is not None:
                union = np.logical_or(edges, previous_edges).sum()
                intersection = np.logical_and(edges, previous_edges).sum()
                instability_values.append(1.0 - intersection / max(union, 1))
            previous_edges = edges
        mean_nll = float(np.mean(nll_values))
        mean_instability = float(np.mean(instability_values))
        candidate_rows.append(
            {
                "alpha": alpha,
                "mean_next_step_gaussian_nll": mean_nll,
                "mean_edge_instability": mean_instability,
                "score": mean_nll + stability_weight * mean_instability,
                "origins": [int(value) for value in available],
                "fallback_count": fallback_count,
            }
        )
    selected_row = min(candidate_rows, key=lambda row: row["score"])
    return float(selected_row["alpha"]), {
        "policy": "rolling_train_next_step_nll_plus_edge_stability",
        "selected": float(selected_row["alpha"]),
        "candidates": candidate_rows,
        "edge_stability_weight": stability_weight,
        "selection_origin_start": int(available[0]),
        "selection_origin_end": int(available[-1]),
        "last_allowed_train_origin": int(last_train_origin),
        "uses_validation_or_test": False,
    }


class _StateBuilder:
    def __init__(self, panel: dict[str, Any], config: dict[str, Any], spec: ConceptSpec):
        self.panel = panel
        self.config = config
        self.spec = spec
        self.returns = np.asarray(panel["returns"], dtype=np.float64)
        self.total_log_returns = np.asarray(
            panel.get("total_log_returns", panel["returns"]), dtype=np.float64
        )
        self.risk_free = np.asarray(
            panel.get("risk_free", np.zeros(self.returns.shape[0])), dtype=np.float64
        )
        self.factors = np.asarray(panel["factors"], dtype=np.float64)
        self.dollar_volume = np.asarray(panel["dollar_volume"], dtype=np.float64)
        self.order_flow = np.asarray(panel["order_flow"], dtype=np.float64)
        self.order_flow_mode = str(
            config.get("order_flow_mode", panel.get("order_flow_mode", "provided_interval"))
        )
        self.market_weights = np.asarray(
            panel.get(
                "market_weights",
                np.full_like(self.returns, 1.0 / self.returns.shape[1]),
            ),
            dtype=np.float64,
        )
        self.graph_window = int(config["graph_window"])
        self.graph_refit_stride = int(config.get("graph_refit_stride", 1))
        if self.graph_refit_stride < 1:
            raise ValueError("graph_refit_stride must be a positive integer")
        self.graph_refit_anchor = self.graph_window - 1
        self.graph_weights = exponential_lag_weights(self.graph_window)
        self.cache: dict[int, _State] = {}

    def build(self, t: int) -> _State:
        if t in self.cache:
            return self.cache[t]
        start = t - self.graph_window + 1
        returns_window = self.returns[start : t + 1]
        factors_window = self.factors[start : t + 1]
        intercept, beta = fit_weighted_ridge_factor_model(
            returns_window,
            factors_window,
            self.graph_weights,
            float(self.config["ridge_alpha"]),
        )
        fitted = intercept[None, :] + factors_window @ beta.T
        residuals = returns_window - fitted
        weights_chronological = self.graph_weights[::-1]
        mean = np.sum(weights_chronological[:, None] * residuals, axis=0)
        scale = np.sqrt(
            np.sum(weights_chronological[:, None] * (residuals - mean) ** 2, axis=0) + 1e-10
        )
        standardized = (residuals - mean) / scale
        previous_state = self.cache.get(t - 1)
        scheduled_refit = (
            (t - self.graph_refit_anchor) % self.graph_refit_stride == 0
        )
        refit = previous_state is None or scheduled_refit
        if refit:
            covariance_init = (
                previous_state.graph_covariance if previous_state is not None else None
            )
            graph_covariance, _, partial, graph_diagnostics = (
                estimate_partial_correlation(
                    standardized,
                    weights_chronological,
                    float(self.config["graph_alpha"]),
                    covariance_init=covariance_init,
                    return_diagnostics=True,
                    max_iter=int(self.config.get("graph_solver_max_iter", 500)),
                    tolerance=float(
                        self.config.get("graph_solver_tolerance", 1e-4)
                    ),
                )
            )
            graph_diagnostics = {
                **graph_diagnostics,
                "refit": True,
                "refit_origin": t,
            }
            graph = smooth_signed_graph(
                partial,
                float(self.config["graph_gate_center"]),
                float(self.config["graph_gate_temperature"]),
            )
        else:
            # A residual-dependence graph is a slower state than daily asset
            # concepts.  Between preregistered refit dates, carry the last
            # available estimate forward; never interpolate with a future fit.
            graph_covariance = previous_state.graph_covariance
            partial = previous_state.partial_corr
            graph = previous_state.graph
            previous_diagnostics = previous_state.graph_diagnostics
            graph_diagnostics = {
                "converged": bool(previous_diagnostics["converged"]),
                "fallback": bool(previous_diagnostics["fallback"]),
                "attempts": 0,
                "alpha_requested": float(self.config["graph_alpha"]),
                "alpha_used": float(previous_diagnostics["alpha_used"]),
                "iterations": 0,
                "warm_start_used": False,
                "solver_max_iter": int(
                    previous_diagnostics.get("solver_max_iter", 500)
                ),
                "solver_tolerance": float(
                    previous_diagnostics.get("solver_tolerance", 1e-4)
                ),
                "refit": False,
                "refit_origin": int(previous_diagnostics["refit_origin"]),
            }
        concepts = compute_asset_concepts(
            t,
            self.returns,
            self.dollar_volume,
            self.order_flow,
            beta,
            self.spec,
            self.order_flow_mode,
        )
        scale_length = int(self.config["market_scale"])
        market_weights = exponential_lag_weights(scale_length)
        simple_total_returns = np.expm1(self.total_log_returns)
        previous_weights = np.vstack([self.market_weights[0], self.market_weights[:-1]])
        market_total_return = np.sum(previous_weights * simple_total_returns, axis=1)
        if np.any(market_total_return <= -1.0):
            raise ValueError("Market portfolio total return must exceed -100%")
        market_log_return = np.log1p(market_total_return) - self.risk_free
        market_rv = np.sum(
            market_weights * market_log_return[t - np.arange(scale_length)] ** 2
        )
        upper = np.abs(partial[np.triu_indices(partial.shape[0], k=1)])
        dependence = float(upper.mean())
        weights_at_origin = previous_weights[t]
        illiquidity = float(
            np.sum(
                weights_at_origin
                * concepts[:, self.spec.illiquidity_index(scale_length)]
            )
        )
        market = np.asarray([np.log(market_rv + 1e-12), dependence, illiquidity])
        raw_corr = np.corrcoef(returns_window, rowvar=False)
        state = _State(
            concepts,
            graph,
            market,
            beta,
            raw_corr,
            partial,
            graph_covariance,
            graph_diagnostics,
        )
        self.cache[t] = state
        return state


def _make_split_indices(
    origins: np.ndarray,
    dates: np.ndarray,
    horizon: int,
    config: dict[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    embargo = int(config.get("embargo", horizon))
    if embargo < horizon:
        raise ValueError("embargo must be at least the maximum forecast horizon")
    split_dates = config.get("split_dates")
    if split_dates is None:
        count = len(origins)
        train_fraction = float(config["train_fraction"])
        val_fraction = float(config["val_fraction"])
        if (
            train_fraction <= 0.0
            or val_fraction <= 0.0
            or train_fraction + val_fraction >= 1.0
        ):
            raise ValueError(
                "train_fraction and val_fraction must be positive and sum to less than one"
            )
        train_boundary = int(count * train_fraction)
        val_boundary = int(count * (train_fraction + val_fraction))
        valid = (
            0
            < train_boundary - embargo
            < train_boundary
            < val_boundary - embargo
            < val_boundary
            < count
        )
        if not valid:
            raise ValueError("Invalid chronological split fractions")
        splits = {
            "train": np.arange(0, train_boundary - embargo),
            "val": np.arange(train_boundary, val_boundary - embargo),
            "test": np.arange(val_boundary, count),
        }
        purged = {
            "train_val": np.arange(train_boundary - embargo, train_boundary),
            "val_test": np.arange(val_boundary - embargo, val_boundary),
        }
        descriptor = {
            "mode": "fractional_chronological_holdout",
            "embargo": embargo,
            "boundaries": {"train_val": train_boundary, "val_test": val_boundary},
            "fractions": {
                "train": train_fraction,
                "validation": val_fraction,
                "test": 1.0 - train_fraction - val_fraction,
            },
        }
    else:
        # Preserve intraday resolution.  Coercing to datetime64[D] would collapse
        # distinct hourly train/validation boundaries onto the same calendar day.
        origin_dates = dates[origins].astype("datetime64[ns]")

        def date_value(name: str) -> np.datetime64:
            if name not in split_dates:
                raise ValueError(f"split_dates is missing {name}")
            return np.datetime64(str(split_dates[name]), "ns")

        train_start = np.datetime64(
            str(split_dates.get("train_start", dates[origins[0]])), "ns"
        )
        train_end = date_value("train_end")
        val_start = date_value("val_start")
        val_end = date_value("val_end")
        test_start = date_value("test_start")
        test_end = date_value("test_end")
        if not train_start <= train_end < val_start <= val_end < test_start <= test_end:
            raise ValueError("Explicit split dates must be strictly chronological")

        def positions(start: np.datetime64, end: np.datetime64) -> np.ndarray:
            return np.flatnonzero((origin_dates >= start) & (origin_dates <= end))

        raw_train = positions(train_start, train_end)
        raw_val = positions(val_start, val_end)
        raw_test = positions(test_start, test_end)
        if min(len(raw_train), len(raw_val), len(raw_test)) == 0:
            raise ValueError("At least one explicit date split is empty")

        def end_position(end: np.datetime64) -> int:
            return int(np.searchsorted(dates, end, side="right") - 1)

        train = raw_train[
            (origins[raw_train] + embargo < origins[raw_val[0]])
            & (origins[raw_train] + horizon <= end_position(train_end))
        ]
        val = raw_val[
            (origins[raw_val] + embargo < origins[raw_test[0]])
            & (origins[raw_val] + horizon <= end_position(val_end))
        ]
        test = raw_test[
            origins[raw_test] + horizon <= end_position(test_end)
        ]
        purged = {
            "train_val": np.setdiff1d(raw_train, train, assume_unique=True),
            "val_test": np.setdiff1d(raw_val, val, assume_unique=True),
            "test_end": np.setdiff1d(raw_test, test, assume_unique=True),
        }
        splits = {"train": train, "val": val, "test": test}
        if min(len(train), len(val), len(test)) == 0:
            raise ValueError("Explicit split became empty after purge/embargo")
        descriptor = {
            "mode": "explicit_date_rolling_fold",
            "embargo": embargo,
            "boundaries": {
                key: str(value)
                for key, value in {
                    "train_start": train_start,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }.items()
            },
            "fractions": None,
            "label_end_enforced": True,
        }
    if origins[splits["train"][-1]] + horizon >= origins[splits["val"]][0]:
        raise RuntimeError("Training labels overlap the validation forecast origin")
    if origins[splits["val"][-1]] + horizon >= origins[splits["test"]][0]:
        raise RuntimeError("Validation labels overlap the test forecast origin")
    return splits, purged, descriptor


def build_research_dataset(
    panel: dict[str, Any],
    config: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    scales = tuple(int(value) for value in config["concept_scales"])
    spec = ConceptSpec(scales=scales, num_factors=int(panel["factors"].shape[1]))
    horizon = int(config["horizon"])
    minimum_t = max(int(config["graph_window"]), max(scales), int(config["market_scale"])) - 1
    final_origin = panel["returns"].shape[0] - horizon - 1
    origins = np.arange(minimum_t, final_origin + 1)
    if len(origins) < 30:
        raise ValueError("Too few forecast origins after applying causal windows")
    dates = np.asarray(panel["dates"], dtype="datetime64[ns]")
    split_indices, purged_indices, split_descriptor = _make_split_indices(
        origins, dates, horizon, config
    )
    last_train_position = int(split_indices["train"][-1])
    graph_alpha, graph_alpha_selection = _select_graph_alpha(
        panel,
        config,
        int(origins[int(split_indices["train"][0])]),
        int(origins[last_train_position]),
    )
    effective_config = {**config, "graph_alpha": graph_alpha}
    builder = _StateBuilder(panel, effective_config, spec)

    c0, a0, s0, c_target, a_target, y = [], [], [], [], [], []
    beta0, raw_corr, partial_corr, last_return, historical_mean = [], [], [], [], []
    graph_converged, graph_fallback, graph_alpha_used, graph_iterations = [], [], [], []
    graph_refit = []
    progress_interval = max(len(origins) // 20, 1)
    for position, t in enumerate(origins, start=1):
        current = builder.build(int(t))
        future = [builder.build(int(t + h)) for h in range(1, horizon + 1)]
        c0.append(current.concepts)
        a0.append(current.graph)
        s0.append(current.market)
        # The manual supervises the complete state path h=0,...,H.  The h=0
        # target is the causally measured initial state and therefore adds an
        # exact anchoring check without becoming a future inference input.
        state_path = [current, *future]
        c_target.append(np.stack([state.concepts for state in state_path]))
        a_target.append(np.stack([state.graph for state in state_path]))
        y.append(np.stack([panel["returns"][t + h] for h in range(1, horizon + 1)]))
        beta0.append(current.beta)
        raw_corr.append(current.raw_corr)
        partial_corr.append(current.partial_corr)
        graph_converged.append(current.graph_diagnostics["converged"])
        graph_fallback.append(current.graph_diagnostics["fallback"])
        graph_alpha_used.append(current.graph_diagnostics["alpha_used"])
        graph_iterations.append(current.graph_diagnostics["iterations"])
        graph_refit.append(current.graph_diagnostics["refit"])
        last_return.append(panel["returns"][t])
        historical_mean.append(panel["returns"][max(0, t - 9) : t + 1].mean(axis=0))
        if progress_callback is not None and (
            position == 1 or position % progress_interval == 0 or position == len(origins)
        ):
            progress_callback(
                f"state_precompute progress={position}/{len(origins)} "
                f"percent={100.0 * position / len(origins):.1f}"
            )

    arrays = {
        "origins": origins,
        "c0_raw": np.stack(c0),
        "a0_raw": np.stack(a0),
        "s0_raw": np.stack(s0),
        "c_target_raw": np.stack(c_target),
        "a_target_raw": np.stack(a_target),
        "y": np.stack(y),
        "beta0": np.stack(beta0),
        "raw_corr": np.stack(raw_corr),
        "partial_corr": np.stack(partial_corr),
        "last_return": np.stack(last_return),
        "historical_mean": np.stack(historical_mean),
        "graph_converged": np.asarray(graph_converged, dtype=np.bool_),
        "graph_fallback": np.asarray(graph_fallback, dtype=np.bool_),
        "graph_alpha_used": np.asarray(graph_alpha_used, dtype=np.float64),
        "graph_iterations": np.asarray(graph_iterations, dtype=np.int64),
        "graph_refit": np.asarray(graph_refit, dtype=np.bool_),
    }
    count = len(origins)

    train_returns = arrays["y"][split_indices["train"]]
    return_median = np.median(train_returns, axis=(0, 1))
    return_mad = 1.4826 * np.median(
        np.abs(train_returns - return_median[None, None, :]), axis=(0, 1)
    )
    return_std = np.std(train_returns, axis=(0, 1))
    return_scale = np.maximum(np.where(return_mad > 1e-6, return_mad, return_std), 1e-4)

    # The manual defines fold statistics over current measured states at the
    # purged training forecast origins.  Future target paths are labels, not
    # extra observations for fitting the scaler; including them would also
    # count the same calendar state repeatedly across overlapping horizons.
    train_concepts = arrays["c0_raw"][split_indices["train"]]
    quantile_values = config.get("projection_quantiles", [0.001, 0.999])
    quantiles = (float(quantile_values[0]), float(quantile_values[1]))
    margin_fraction = float(config.get("projection_margin_fraction", 0.05))
    concept_scaler = RobustStandardizer().fit(
        train_concepts, quantiles=quantiles, margin_fraction=margin_fraction
    )
    market_scaler = RobustStandardizer().fit(
        arrays["s0_raw"][split_indices["train"]],
        quantiles=quantiles,
        margin_fraction=margin_fraction,
    )
    arrays["c0"] = concept_scaler.transform(arrays.pop("c0_raw")).astype(np.float32)
    arrays["c_target"] = concept_scaler.transform(arrays.pop("c_target_raw")).astype(
        np.float32
    )
    arrays["s0"] = market_scaler.transform(arrays.pop("s0_raw")).astype(np.float32)

    train_graphs = arrays["a0_raw"][split_indices["train"]]
    train_graph_norms = np.linalg.norm(train_graphs, axis=(-2, -1)).reshape(-1)
    graph_quantile = float(config.get("graph_bound_quantile", 0.999))
    graph_margin = float(config.get("graph_bound_margin_fraction", 0.05))
    graph_bound = float(np.quantile(train_graph_norms, graph_quantile) * (1.0 + graph_margin))
    graph_bound = max(graph_bound, 1e-6)

    def project_graph_array(values: np.ndarray) -> np.ndarray:
        projected = np.empty_like(values)
        flat_in = values.reshape(-1, values.shape[-2], values.shape[-1])
        flat_out = projected.reshape(flat_in.shape)
        for position, graph in enumerate(flat_in):
            flat_out[position] = project_graph_numpy(graph, graph_bound)
        return projected

    arrays["a0"] = project_graph_array(arrays.pop("a0_raw")).astype(np.float32)
    arrays["a_target"] = project_graph_array(arrays.pop("a_target_raw")).astype(
        np.float32
    )

    train_c_path = arrays["c_target"][split_indices["train"]]
    concept_increments = train_c_path[:, 1:] - train_c_path[:, :-1]
    concept_loss_scale = 1.4826 * np.median(
        np.abs(concept_increments), axis=(0, 1, 2)
    )
    concept_loss_scale = np.maximum(concept_loss_scale, 1e-3)
    train_a_path = arrays["a_target"][split_indices["train"]]
    graph_increments = train_a_path[:, 1:] - train_a_path[:, :-1]
    num_assets = graph_increments.shape[-1]
    off_diagonal = ~np.eye(num_assets, dtype=bool)
    graph_edges = graph_increments[..., off_diagonal]
    graph_loss_scale = float(
        max(1.4826 * np.median(np.abs(graph_edges)), 1e-4)
    )
    for key in [
        "a0",
        "a_target",
        "y",
        "beta0",
        "raw_corr",
        "partial_corr",
        "last_return",
        "historical_mean",
    ]:
        arrays[key] = arrays[key].astype(np.float32)
    arrays["directional_mask"] = spec.directional_mask.astype(np.float32)
    arrays["risk_indices"] = np.asarray(spec.risk_indices, dtype=np.int64)

    def split_date_range(indices: np.ndarray) -> dict[str, str | int]:
        first = int(indices[0])
        last = int(indices[-1])
        first_origin = int(origins[first])
        last_origin = int(origins[last])
        return {
            "samples": len(indices),
            "forecast_origin_start": str(dates[first_origin]),
            "forecast_origin_end": str(dates[last_origin]),
            "last_label_date": str(dates[last_origin + horizon]),
        }

    refit_mask = arrays["graph_refit"]
    if not np.any(refit_mask):
        raise RuntimeError("At least one residual graph refit is required")
    metadata = {
        "concept_names": spec.names,
        "market_names": ["market_log_realized_variance", "mean_residual_dependence", "market_illiquidity"],
        "asset_names": [str(value) for value in panel["asset_names"]],
        "factor_names": [str(value) for value in panel["factor_names"]],
        "num_samples": count,
        "num_assets": int(panel["returns"].shape[1]),
        "num_concepts": spec.num_concepts,
        "horizon": horizon,
        "split_indices": {key: value.tolist() for key, value in split_indices.items()},
        "purged_indices": {key: value.tolist() for key, value in purged_indices.items()},
        "embargo": int(split_descriptor["embargo"]),
        "split_mode": split_descriptor["mode"],
        "split_boundaries": split_descriptor["boundaries"],
        "split_fractions": split_descriptor["fractions"],
        "split_label_end_enforced": bool(
            split_descriptor.get("label_end_enforced", False)
        ),
        "source_date_range": {"start": str(dates[0]), "end": str(dates[-1])},
        "split_date_ranges": {
            key: split_date_range(value) for key, value in split_indices.items()
        },
        "graph_estimation": {
            "refit_stride": int(effective_config.get("graph_refit_stride", 1)),
            "refit_anchor_origin": int(builder.graph_refit_anchor),
            "refit_count": int(np.sum(refit_mask)),
            "refit_rate": float(np.mean(refit_mask)),
            "carry_forward_is_causal": True,
            "convergence_rate": float(np.mean(arrays["graph_converged"][refit_mask])),
            "fallback_rate": float(np.mean(arrays["graph_fallback"][refit_mask])),
            "mean_iterations": float(np.mean(arrays["graph_iterations"][refit_mask])),
            "max_alpha_used": float(np.max(arrays["graph_alpha_used"])),
            "alpha_selection": graph_alpha_selection,
        },
        "state_target_leads": list(range(horizon + 1)),
        "return_target_leads": list(range(1, horizon + 1)),
        "projection_quantiles": list(quantiles),
        "projection_margin_fraction": margin_fraction,
        "normalization_fit_scope": (
            "current measured states at purged training forecast origins only; "
            "future state labels excluded"
        ),
        "concept_lower_bounds": concept_scaler.lower.tolist(),
        "concept_upper_bounds": concept_scaler.upper.tolist(),
        "market_lower_bounds": market_scaler.lower.tolist(),
        "market_upper_bounds": market_scaler.upper.tolist(),
        "graph_bound": graph_bound,
        "graph_bound_estimator": (
            f"train-only Frobenius norm q={graph_quantile} with "
            f"{100.0 * graph_margin:.1f}% outward margin"
        ),
        "concept_loss_scale": concept_loss_scale.tolist(),
        "graph_loss_scale": graph_loss_scale,
        "return_scale": return_scale.tolist(),
        "return_scale_estimator": "train-only per-asset 1.4826*MAD with std fallback",
        "concept_scaler": concept_scaler.state_dict(),
        "market_scaler": market_scaler.state_dict(),
        "minimum_origin": int(origins[0]),
        "maximum_origin": int(origins[-1]),
        "risk_free_source": str(panel.get("risk_free_source", "synthetic_or_unspecified")),
        "market_weight_source": str(
            panel.get("market_weight_source", "synthetic_or_equal_weight")
        ),
        "order_flow_mode": str(
            config.get("order_flow_mode", panel.get("order_flow_mode", "provided_interval"))
        ),
    }
    return {"arrays": arrays, "metadata": metadata}


class FinancialTensorDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, prepared: dict[str, Any], split: str):
        self.arrays = prepared["arrays"]
        self.metadata = prepared["metadata"]
        self.indices = np.asarray(self.metadata["split_indices"][split], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "c0": torch.from_numpy(self.arrays["c0"][index]),
            "a0": torch.from_numpy(self.arrays["a0"][index]),
            "s0": torch.from_numpy(self.arrays["s0"][index]),
            "c_target": torch.from_numpy(self.arrays["c_target"][index]),
            "a_target": torch.from_numpy(self.arrays["a_target"][index]),
            "y": torch.from_numpy(self.arrays["y"][index]),
        }
