from __future__ import annotations

from typing import Any

import numpy as np


def generate_synthetic_market(
    num_steps: int,
    num_assets: int,
    num_factors: int,
    seed: int,
) -> dict[str, Any]:
    """Generate a heavy-tailed, regime-switching factor-network market.

    The synthetic generator is only a debug fixture. It has known factor exposures,
    asymmetric volatility regimes, sparse cross-asset propagation, liquidity and
    order-flow channels, so every F-CONCORD component receives a meaningful signal.
    """
    if num_steps < 80:
        raise ValueError("num_steps must be at least 80")
    if num_assets < 3 or num_factors < 1:
        raise ValueError("Need at least three assets and one factor")
    rng = np.random.default_rng(seed)

    regime = np.zeros(num_steps, dtype=np.int64)
    for t in range(1, num_steps):
        if rng.random() < (0.05 if regime[t - 1] == 0 else 0.18):
            regime[t] = 1 - regime[t - 1]
        else:
            regime[t] = regime[t - 1]

    factor_scale = np.where(regime[:, None] == 0, 0.006, 0.016)
    factors = rng.standard_t(df=7, size=(num_steps, num_factors)) * factor_scale
    factors[:, 0] += np.where(regime == 0, 0.0003, -0.0005)

    beta = np.empty((num_steps, num_assets, num_factors), dtype=np.float64)
    beta[0] = rng.normal(0.5, 0.25, size=(num_assets, num_factors))
    beta[0, :, 0] += 0.5
    for t in range(1, num_steps):
        target = beta[0] + 0.15 * regime[t]
        beta[t] = 0.985 * beta[t - 1] + 0.015 * target + rng.normal(
            0.0, 0.008, size=(num_assets, num_factors)
        )

    base_graph = np.zeros((num_assets, num_assets), dtype=np.float64)
    for i in range(num_assets):
        j = (i + 1) % num_assets
        sign = 1.0 if i % 3 else -1.0
        base_graph[i, j] = base_graph[j, i] = sign * rng.uniform(0.18, 0.35)
    for _ in range(max(1, num_assets // 2)):
        i, j = rng.choice(num_assets, size=2, replace=False)
        base_graph[i, j] = base_graph[j, i] = rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.22)

    returns = np.zeros((num_steps, num_assets), dtype=np.float64)
    residuals = np.zeros_like(returns)
    order_flow = np.zeros_like(returns)
    dollar_volume = np.zeros_like(returns)
    for t in range(num_steps):
        stress = float(regime[t])
        graph_t = base_graph * (1.0 + 0.55 * stress)
        innovation = rng.standard_t(df=5, size=num_assets) * (0.007 + 0.010 * stress)
        if t > 0:
            innovation += 0.10 * np.tanh(residuals[t - 1] @ graph_t)
        residuals[t] = np.linalg.solve(np.eye(num_assets) - 0.18 * graph_t, innovation)
        downside = np.minimum(residuals[t], 0.0)
        residuals[t] += 0.20 * downside * stress
        returns[t] = beta[t] @ factors[t] + residuals[t]
        order_flow[t] = np.tanh(35.0 * residuals[t] + rng.normal(0, 0.35, num_assets))
        log_dv = 16.0 + rng.normal(0, 0.55, num_assets) - 0.45 * stress - 5.0 * np.abs(residuals[t])
        dollar_volume[t] = np.exp(log_dv)

    prices = np.empty((num_steps, num_assets), dtype=np.float64)
    prices[0] = rng.uniform(40.0, 160.0, size=num_assets)
    for t in range(1, num_steps):
        prices[t] = prices[t - 1] * np.exp(np.clip(returns[t], -0.25, 0.25))

    dates = np.datetime64("2000-01-01") + np.arange(num_steps).astype("timedelta64[D]")
    asset_names = np.asarray([f"asset_{i:02d}" for i in range(num_assets)])
    factor_names = np.asarray([f"factor_{k:02d}" for k in range(num_factors)])
    return {
        "dates": dates,
        "asset_names": asset_names,
        "factor_names": factor_names,
        "prices": prices,
        "returns": returns,
        "factors": factors,
        "dollar_volume": dollar_volume,
        "order_flow": order_flow,
        "true_beta": beta,
        "true_regime": regime,
        "true_graph": base_graph,
    }
