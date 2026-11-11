from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - median), axis=0)
    standard_deviation = np.std(values, axis=0)
    scale = np.maximum(np.where(mad > 1e-8, mad, standard_deviation), 1e-6)
    return median, scale


def _history_windows(values: np.ndarray, origins: np.ndarray, length: int) -> np.ndarray:
    if length < 2:
        raise ValueError("history length must be at least two")
    if int(origins.min()) - length + 1 < 0:
        raise ValueError("an origin does not have the requested causal history")
    return np.stack([values[t - length + 1 : t + 1] for t in origins])


@dataclass(frozen=True)
class BenchmarkDimensions:
    num_assets: int
    horizon: int
    history_length: int
    state_dim: int
    asset_feature_dim: int
    market_feature_dim: int

    def as_dict(self) -> dict[str, int]:
        return {
            "num_assets": self.num_assets,
            "horizon": self.horizon,
            "history_length": self.history_length,
            "state_dim": self.state_dim,
            "asset_feature_dim": self.asset_feature_dim,
            "market_feature_dim": self.market_feature_dim,
        }


def build_benchmark_arrays(
    prepared: dict[str, Any],
    panel: dict[str, Any],
    history_length: int,
) -> dict[str, Any]:
    """Build causal baseline inputs using only fold-training statistics.

    The main F-CONCORD prepared states are reused exactly. Raw-history competitors receive
    only observations through each forecast origin. Volume/order-flow/factor transforms are
    fitted on raw rows available no later than the last purged training origin.
    """
    arrays = prepared["arrays"]
    metadata = prepared["metadata"]
    origins = np.asarray(arrays["origins"], dtype=np.int64)
    split_indices = {
        split: np.asarray(indices, dtype=np.int64)
        for split, indices in metadata["split_indices"].items()
    }
    train_positions = split_indices["train"]
    first_training_origin = int(origins[int(train_positions[0])])
    last_training_origin = int(origins[int(train_positions[-1])])
    fit_start = max(0, first_training_origin - history_length + 1)
    fit_slice = slice(fit_start, last_training_origin + 1)

    returns = np.asarray(panel["returns"], dtype=np.float64)
    dollar_volume = np.asarray(panel["dollar_volume"], dtype=np.float64)
    order_flow = np.asarray(panel["order_flow"], dtype=np.float64)
    factors = np.asarray(panel["factors"], dtype=np.float64)
    num_assets = int(metadata["num_assets"])
    horizon = int(metadata["horizon"])
    return_scale = np.asarray(metadata["return_scale"], dtype=np.float64)
    if returns.shape[1] != num_assets or return_scale.shape != (num_assets,):
        raise ValueError("panel and prepared asset dimensions do not agree")

    normalized_returns = returns / return_scale[None]
    log_volume = np.log1p(np.maximum(dollar_volume, 0.0))
    volume_location, volume_scale = _robust_location_scale(log_volume[fit_slice])
    flow_location, flow_scale = _robust_location_scale(order_flow[fit_slice])
    normalized_volume = (log_volume - volume_location[None]) / volume_scale[None]
    normalized_flow = (order_flow - flow_location[None]) / flow_scale[None]

    factor_location, factor_scale = _robust_location_scale(factors[fit_slice])
    normalized_factors = (factors - factor_location[None]) / factor_scale[None]
    market_mean = normalized_returns.mean(axis=1, keepdims=True)
    market_dispersion = normalized_returns.std(axis=1, keepdims=True)
    market_rows = np.concatenate(
        [market_mean, market_dispersion, normalized_factors], axis=1
    )

    return_history = _history_windows(normalized_returns, origins, history_length)
    asset_feature_rows = np.stack(
        [normalized_returns, normalized_volume, normalized_flow], axis=-1
    )
    asset_features = _history_windows(asset_feature_rows, origins, history_length)
    market_history = _history_windows(market_rows, origins, history_length)

    concept_state = np.asarray(arrays["c0"], dtype=np.float32).reshape(len(origins), -1)
    graph_state = np.asarray(arrays["a0"], dtype=np.float32).reshape(len(origins), -1)
    market_state = np.asarray(arrays["s0"], dtype=np.float32)
    state = np.concatenate([concept_state, graph_state, market_state], axis=1)
    target = np.asarray(arrays["y"], dtype=np.float32)
    target_scaled = target / return_scale[None, None, :]

    output_arrays = {
        "origins": origins,
        "return_history": return_history.astype(np.float32),
        "asset_features": asset_features.astype(np.float32),
        "market_history": market_history.astype(np.float32),
        "state": state.astype(np.float32),
        "concept_state": np.asarray(arrays["c0"], dtype=np.float32),
        "graph_state": np.asarray(arrays["a0"], dtype=np.float32),
        "market_state": market_state,
        "target": target,
        "target_scaled": target_scaled.astype(np.float32),
        "return_scale": return_scale.astype(np.float32),
    }
    dimensions = BenchmarkDimensions(
        num_assets=num_assets,
        horizon=horizon,
        history_length=history_length,
        state_dim=int(state.shape[1]),
        asset_feature_dim=int(asset_features.shape[-1]),
        market_feature_dim=int(market_history.shape[-1]),
    )
    benchmark_metadata = {
        "dimensions": dimensions.as_dict(),
        "split_indices": {key: value.tolist() for key, value in split_indices.items()},
        "input_availability": (
            "Every raw-history window ends at its forecast origin; no future return, "
            "volume, order-flow or factor row is included."
        ),
        "normalization_fit_scope": {
            "first_raw_row": fit_start,
            "last_raw_row": last_training_origin,
            "last_training_origin": last_training_origin,
            "future_state_labels_excluded": True,
        },
        "scalers": {
            "return_scale_source": "reference F-CONCORD training-only return scale",
            "log_volume_location": volume_location.tolist(),
            "log_volume_scale": volume_scale.tolist(),
            "order_flow_location": flow_location.tolist(),
            "order_flow_scale": flow_scale.tolist(),
            "factor_location": factor_location.tolist(),
            "factor_scale": factor_scale.tolist(),
        },
        "state_definition": "flattened c0, full signed a0, then three-coordinate s0",
        "asset_history_features": [
            "scaled_excess_log_return",
            "train_standardized_log1p_dollar_volume",
            "train_standardized_order_flow",
        ],
        "market_history_features": [
            "cross_asset_mean_scaled_return",
            "cross_asset_scaled_return_dispersion",
            *[f"standardized_factor_{index}" for index in range(factors.shape[1])],
        ],
    }
    return {"arrays": output_arrays, "metadata": benchmark_metadata}


class BenchmarkTensorDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, benchmark: dict[str, Any], split: str) -> None:
        self.arrays = benchmark["arrays"]
        self.indices = np.asarray(
            benchmark["metadata"]["split_indices"][split], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "return_history": torch.from_numpy(self.arrays["return_history"][index]),
            "asset_features": torch.from_numpy(self.arrays["asset_features"][index]),
            "market_history": torch.from_numpy(self.arrays["market_history"][index]),
            "state": torch.from_numpy(self.arrays["state"][index]),
            "target": torch.from_numpy(self.arrays["target"][index]),
            "target_scaled": torch.from_numpy(self.arrays["target_scaled"][index]),
        }
