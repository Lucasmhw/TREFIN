from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def exponential_lag_weights(length: int, tau: float | None = None) -> np.ndarray:
    """Weights ordered from current lag u=0 to oldest lag u=length-1."""
    if length < 2:
        raise ValueError("length must be at least two")
    tau = float(length / 3 if tau is None else tau)
    lags = np.arange(length, dtype=np.float64)
    weights = np.exp(-lags / tau)
    return weights / weights.sum()


@dataclass(frozen=True)
class ConceptSpec:
    scales: tuple[int, ...]
    num_factors: int

    @property
    def num_concepts(self) -> int:
        return 5 * len(self.scales) + 1 + self.num_factors

    @property
    def names(self) -> list[str]:
        names: list[str] = []
        names += [f"momentum_{scale}" for scale in self.scales]
        names += [f"up_semivar_{scale}" for scale in self.scales]
        names += [f"down_semivar_{scale}" for scale in self.scales]
        names += [f"jump_share_{scale}" for scale in self.scales]
        names += [f"log_illiquidity_{scale}" for scale in self.scales]
        names += ["order_flow"]
        names += [f"factor_beta_{k}" for k in range(self.num_factors)]
        return names

    @property
    def directional_mask(self) -> np.ndarray:
        length = len(self.scales)
        mask = np.zeros(self.num_concepts, dtype=np.float32)
        mask[:length] = 1.0
        mask[5 * length] = 1.0
        mask[5 * length + 1 :] = 1.0
        return mask

    @property
    def risk_indices(self) -> list[int]:
        length = len(self.scales)
        downside = list(range(2 * length, 3 * length))
        jump = list(range(3 * length, 4 * length))
        illiquidity = list(range(4 * length, 5 * length))
        return downside + jump + illiquidity

    def illiquidity_index(self, scale: int) -> int:
        position = list(self.scales).index(scale)
        return 4 * len(self.scales) + position


def fit_weighted_ridge_factor_model(
    returns_window: np.ndarray,
    factors_window: np.ndarray,
    lag_weights: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate intercepts and betas for every asset at one forecast origin.

    Windows are chronological (oldest to newest); lag_weights are current-to-oldest.
    """
    window, num_assets = returns_window.shape
    if factors_window.shape[0] != window or lag_weights.shape != (window,):
        raise ValueError("Incompatible ridge window shapes")
    design = np.column_stack([np.ones(window), factors_window])
    chronological_weights = lag_weights[::-1]
    weighted_design = design * chronological_weights[:, None]
    penalty = np.diag([0.0] + [ridge_alpha] * factors_window.shape[1])
    lhs = design.T @ weighted_design + penalty
    rhs = design.T @ (chronological_weights[:, None] * returns_window)
    coefficients = np.linalg.solve(lhs + 1e-10 * np.eye(lhs.shape[0]), rhs)
    intercept = coefficients[0]
    beta = coefficients[1:].T
    if beta.shape != (num_assets, factors_window.shape[1]):
        raise RuntimeError("Unexpected beta shape")
    return intercept, beta


def compute_asset_concepts(
    t: int,
    returns: np.ndarray,
    dollar_volume: np.ndarray,
    order_flow: np.ndarray,
    beta: np.ndarray,
    spec: ConceptSpec,
    order_flow_mode: str = "provided_interval",
) -> np.ndarray:
    num_assets = returns.shape[1]
    blocks: list[np.ndarray] = []
    momentum: list[np.ndarray] = []
    up_var: list[np.ndarray] = []
    down_var: list[np.ndarray] = []
    jump: list[np.ndarray] = []
    illiquidity: list[np.ndarray] = []
    for scale in spec.scales:
        weights = exponential_lag_weights(scale)
        indices = t - np.arange(scale)
        r = returns[indices]
        dv = dollar_volume[indices]
        momentum.append(np.sum(weights[:, None] * r, axis=0))
        positive = np.maximum(r, 0.0)
        negative = np.maximum(-r, 0.0)
        up = np.sum(weights[:, None] * positive**2, axis=0)
        down = np.sum(weights[:, None] * negative**2, axis=0)
        up_var.append(np.log(up + 1e-10))
        down_var.append(np.log(down + 1e-10))
        rv = np.sum(weights[:, None] * r**2, axis=0)
        normalizer = np.sum(np.sqrt(weights[1:] * weights[:-1]))
        bv = (
            np.pi
            / (2.0 * max(normalizer, 1e-12))
            * np.sum(
                np.sqrt(weights[1:, None] * weights[:-1, None])
                * np.abs(r[1:])
                * np.abs(r[:-1]),
                axis=0,
            )
        )
        jump.append(np.maximum(rv - bv, 0.0) / (rv + 1e-10))
        amihud = np.sum(weights[:, None] * np.abs(r) / (dv + 1e-8), axis=0)
        illiquidity.append(np.log(amihud + 1e-18))
    blocks.extend(
        [
            np.stack(momentum, axis=1),
            np.stack(up_var, axis=1),
            np.stack(down_var, axis=1),
            np.stack(jump, axis=1),
            np.stack(illiquidity, axis=1),
        ]
    )
    of_scale = spec.scales[-1]
    of_weights = exponential_lag_weights(of_scale)
    of_indices = t - np.arange(of_scale)
    if order_flow_mode == "daily_signed_dollar_volume_proxy":
        # Manual Eq. (daily-OFI): the rolling numerator is signed dollar
        # volume and the denominator is dollar volume.  This is intentionally
        # labelled a weak daily proxy, not true limit-order-book OFI.
        signed_dollar_volume = np.sign(returns[of_indices]) * dollar_volume[of_indices]
        numerator = np.sum(of_weights[:, None] * signed_dollar_volume, axis=0)
        denominator = np.sum(
            of_weights[:, None] * dollar_volume[of_indices], axis=0
        )
        of = numerator / (denominator + 1e-12)
    elif order_flow_mode == "provided_interval":
        # A supplied interval OFI has already aggregated the events in that
        # observation interval, so it must not be smoothed for a second time.
        of = order_flow[t]
    elif order_flow_mode == "provided_smoothed":
        # Backward-compatible supplemental mode for already existing datasets.
        of = np.sum(of_weights[:, None] * order_flow[of_indices], axis=0)
    else:
        raise ValueError(f"Unsupported order_flow_mode: {order_flow_mode}")
    blocks.extend([of[:, None], beta])
    concepts = np.concatenate(blocks, axis=1)
    if concepts.shape != (num_assets, spec.num_concepts):
        raise RuntimeError("Concept construction produced inconsistent dimensions")
    return concepts


class RobustStandardizer:
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.median: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.lower: np.ndarray | None = None
        self.upper: np.ndarray | None = None

    def fit(
        self,
        values: np.ndarray,
        quantiles: tuple[float, float] = (0.001, 0.999),
        margin_fraction: float = 0.05,
    ) -> RobustStandardizer:
        flat = values.reshape(-1, values.shape[-1])
        self.median = np.nanmedian(flat, axis=0)
        mad = np.nanmedian(np.abs(flat - self.median), axis=0)
        robust_scale = 1.4826 * mad
        # Sparse concepts such as daily jump share can have median absolute
        # deviation exactly zero.  Dividing rare nonzero observations by only
        # epsilon would create artificial million-scale standardized states.
        # A train-only IQR, then standard-deviation fallback preserves the
        # robust definition whenever it is identified and remains finite when
        # the median plateau is genuine.
        q25, q75 = np.nanquantile(flat, [0.25, 0.75], axis=0)
        iqr_scale = (q75 - q25) / 1.349
        standard_scale = np.nanstd(flat, axis=0)
        self.scale = np.where(
            robust_scale > self.epsilon,
            robust_scale,
            np.where(iqr_scale > self.epsilon, iqr_scale, standard_scale),
        )
        self.scale = np.maximum(self.scale, self.epsilon)
        standardized = (flat - self.median) / self.scale
        lower_q, upper_q = quantiles
        if not 0.0 <= lower_q < upper_q <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
        if margin_fraction < 0.0:
            raise ValueError("margin_fraction must be non-negative")
        lower = np.nanquantile(standardized, lower_q, axis=0)
        upper = np.nanquantile(standardized, upper_q, axis=0)
        span = np.maximum(upper - lower, self.epsilon)
        self.lower = lower - margin_fraction * span
        self.upper = upper + margin_fraction * span
        return self

    def transform(self, values: np.ndarray, clip: bool = True) -> np.ndarray:
        if self.median is None or self.scale is None:
            raise RuntimeError("Standardizer is not fitted")
        transformed = (values - self.median) / self.scale
        if clip:
            if self.lower is None or self.upper is None:
                raise RuntimeError("Coordinate bounds are not fitted")
            transformed = np.clip(transformed, self.lower, self.upper)
        return transformed

    def state_dict(self) -> dict[str, list[float]]:
        if (
            self.median is None
            or self.scale is None
            or self.lower is None
            or self.upper is None
        ):
            raise RuntimeError("Standardizer is not fitted")
        return {
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
        }
