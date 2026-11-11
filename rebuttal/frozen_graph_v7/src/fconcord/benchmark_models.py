from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class BenchmarkForecaster(nn.Module):
    """Common direct multi-horizon interface for trainable benchmark models."""

    input_kind = "return_history"

    def forward(
        self,
        return_history: torch.Tensor,
        state: torch.Tensor | None = None,
        asset_features: torch.Tensor | None = None,
        market_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class StateMLP(BenchmarkForecaster):
    input_kind = "state"

    def __init__(
        self, input_dim: int, num_assets: int, horizon: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.horizon = horizon
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * num_assets),
        )

    def forward(
        self,
        return_history: torch.Tensor,
        state: torch.Tensor | None = None,
        asset_features: torch.Tensor | None = None,
        market_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del return_history, asset_features, market_history
        if state is None:
            raise ValueError("StateMLP requires state features")
        return self.network(state).reshape(-1, self.horizon, self.num_assets)


class RawHistoryMLP(BenchmarkForecaster):
    def __init__(
        self, history_length: int, num_assets: int, horizon: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.horizon = horizon
        self.network = nn.Sequential(
            nn.Linear(history_length * num_assets, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * num_assets),
        )

    def forward(
        self,
        return_history: torch.Tensor,
        state: torch.Tensor | None = None,
        asset_features: torch.Tensor | None = None,
        market_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del state, asset_features, market_history
        return self.network(return_history.flatten(1)).reshape(
            -1, self.horizon, self.num_assets
        )


class DLinearForecaster(BenchmarkForecaster):
    """Channel-independent DLinear trend/seasonal direct forecaster."""

    def __init__(
        self,
        history_length: int,
        num_assets: int,
        horizon: int,
        moving_average: int = 21,
        individual: bool = False,
    ) -> None:
        super().__init__()
        if moving_average < 1 or moving_average % 2 == 0:
            raise ValueError("moving_average must be a positive odd integer")
        self.history_length = history_length
        self.num_assets = num_assets
        self.horizon = horizon
        self.moving_average = moving_average
        self.individual = individual
        if individual:
            self.seasonal = nn.ModuleList(
                nn.Linear(history_length, horizon) for _ in range(num_assets)
            )
            self.trend = nn.ModuleList(
                nn.Linear(history_length, horizon) for _ in range(num_assets)
            )
        else:
            self.seasonal = nn.Linear(history_length, horizon)
            self.trend = nn.Linear(history_length, horizon)

    def _decompose(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # values: [B,N,L]. Replication padding avoids adding artificial zeros at edges.
        padding = (self.moving_average - 1) // 2
        padded = F.pad(values, (padding, padding), mode="replicate")
        trend = F.avg_pool1d(
            padded, kernel_size=self.moving_average, stride=1, padding=0
        )
        return values - trend, trend

    def forward(
        self,
        return_history: torch.Tensor,
        state: torch.Tensor | None = None,
        asset_features: torch.Tensor | None = None,
        market_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del state, asset_features, market_history
        values = return_history.transpose(1, 2)
        seasonal, trend = self._decompose(values)
        if self.individual:
            predictions = [
                self.seasonal[asset](seasonal[:, asset])
                + self.trend[asset](trend[:, asset])
                for asset in range(self.num_assets)
            ]
            output = torch.stack(predictions, dim=1)
        else:
            output = self.seasonal(seasonal) + self.trend(trend)
        return output.transpose(1, 2)


class GRUForecaster(BenchmarkForecaster):
    def __init__(
        self,
        num_assets: int,
        horizon: int,
        hidden_dim: int,
        layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.horizon = horizon
        self.gru = nn.GRU(
            input_size=num_assets,
            hidden_size=hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, horizon * num_assets)

    def forward(
        self,
        return_history: torch.Tensor,
        state: torch.Tensor | None = None,
        asset_features: torch.Tensor | None = None,
        market_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del state, asset_features, market_history
        _, hidden = self.gru(return_history)
        return self.head(hidden[-1]).reshape(-1, self.horizon, self.num_assets)


class TimeMixerForecaster(BenchmarkForecaster):
    """Auditable TimeMixer-style decomposable multiscale forecast mixing."""

    def __init__(
        self,
        history_length: int,
        num_assets: int,
        horizon: int,
        downsampling_factors: list[int],
        moving_average: int = 21,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        factors = [int(value) for value in downsampling_factors]
        if not factors or factors[0] != 1 or any(value < 1 for value in factors):
            raise ValueError("TimeMixer factors must be positive and start at one")
        if factors != sorted(set(factors)):
            raise ValueError("TimeMixer factors must be sorted and unique")
        if moving_average < 1 or moving_average % 2 == 0:
            raise ValueError("TimeMixer moving average must be positive and odd")
        self.num_assets = num_assets
        self.horizon = horizon
        self.factors = factors
        self.moving_average = moving_average
        self.lengths = [history_length // factor for factor in factors]
        if min(self.lengths) < 3:
            raise ValueError("A TimeMixer scale has fewer than three observations")
        self.seasonal_mixers = nn.ModuleList(
            nn.Sequential(nn.Linear(length, length), nn.GELU(), nn.Dropout(dropout))
            for length in self.lengths[1:]
        )
        self.trend_mixers = nn.ModuleList(
            nn.Sequential(nn.Linear(length, length), nn.GELU(), nn.Dropout(dropout))
            for length in self.lengths[:-1]
        )
        self.scale_heads = nn.ModuleList(
            nn.Linear(length, horizon) for length in self.lengths
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.lengths)))

    def _moving_average(self, values: torch.Tensor) -> torch.Tensor:
        kernel = min(self.moving_average, values.shape[-1])
        if kernel % 2 == 0:
            kernel -= 1
        padding = (kernel - 1) // 2
        return F.avg_pool1d(
            F.pad(values, (padding, padding), mode="replicate"),
            kernel_size=kernel,
            stride=1,
        )

    def forward(
        self,
        return_history: torch.Tensor,
        state: torch.Tensor | None = None,
        asset_features: torch.Tensor | None = None,
        market_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del state, asset_features, market_history
        values = return_history.transpose(1, 2)
        scales = [
            values
            if factor == 1
            else F.avg_pool1d(values, kernel_size=factor, stride=factor)
            for factor in self.factors
        ]
        trends = [self._moving_average(scale) for scale in scales]
        seasonal = [scale - trend for scale, trend in zip(scales, trends, strict=True)]

        seasonal_mixed = [seasonal[0]]
        for level in range(1, len(scales)):
            down = F.interpolate(
                seasonal_mixed[-1], size=self.lengths[level], mode="linear"
            )
            mixed = self.seasonal_mixers[level - 1](down)
            seasonal_mixed.append(seasonal[level] + mixed)

        trend_mixed: list[torch.Tensor] = [torch.empty(0)] * len(scales)
        trend_mixed[-1] = trends[-1]
        for level in range(len(scales) - 2, -1, -1):
            up = F.interpolate(
                trend_mixed[level + 1], size=self.lengths[level], mode="linear"
            )
            trend_mixed[level] = trends[level] + self.trend_mixers[level](up)

        forecasts = torch.stack(
            [
                head(season + trend)
                for head, season, trend in zip(
                    self.scale_heads, seasonal_mixed, trend_mixed, strict=True
                )
            ],
            dim=0,
        )
        weights = torch.softmax(self.scale_logits, dim=0).view(-1, 1, 1, 1)
        return (forecasts * weights).sum(dim=0).transpose(1, 2)


class PatchTSTForecaster(BenchmarkForecaster):
    """Channel-independent patched Transformer following PatchTST's core inductive bias."""

    def __init__(
        self,
        history_length: int,
        num_assets: int,
        horizon: int,
        patch_size: int = 16,
        stride: int = 8,
        d_model: int = 32,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_size > history_length or stride < 1:
            raise ValueError("Patch size/stride are incompatible with the history length")
        self.num_assets = num_assets
        self.horizon = horizon
        self.patch_size = patch_size
        self.stride = stride
        self.patch_count = 1 + (history_length - patch_size) // stride
        self.patch_projection = nn.Linear(patch_size, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(self.patch_count * d_model, horizon)

    def forward(
        self,
        return_history: torch.Tensor,
        state: torch.Tensor | None = None,
        asset_features: torch.Tensor | None = None,
        market_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del state, asset_features, market_history
        batch, _, assets = return_history.shape
        patches = return_history.transpose(1, 2).unfold(
            dimension=-1, size=self.patch_size, step=self.stride
        )
        tokens = self.patch_projection(patches).reshape(
            batch * assets, self.patch_count, -1
        )
        encoded = self.encoder(tokens + self.position)
        output = self.head(encoded.flatten(1)).reshape(batch, assets, self.horizon)
        return output.transpose(1, 2)


class ITransformerForecaster(BenchmarkForecaster):
    """Inverted Transformer: assets are tokens and their histories are token features."""

    def __init__(
        self,
        history_length: int,
        num_assets: int,
        horizon: int,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.history_projection = nn.Linear(history_length, d_model)
        self.asset_embedding = nn.Parameter(torch.zeros(1, num_assets, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(
        self,
        return_history: torch.Tensor,
        state: torch.Tensor | None = None,
        asset_features: torch.Tensor | None = None,
        market_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del state, asset_features, market_history
        tokens = self.history_projection(return_history.transpose(1, 2))
        encoded = self.encoder(tokens + self.asset_embedding)
        return self.head(encoded).transpose(1, 2)


class MarketGuidedStockTransformer(BenchmarkForecaster):
    """Auditable MASTER-style market gating and alternating time/stock attention."""

    input_kind = "market_guided_asset_history"

    def __init__(
        self,
        asset_feature_dim: int,
        market_feature_dim: int,
        num_assets: int,
        horizon: int,
        d_model: int = 32,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.horizon = horizon
        self.market_gate = nn.Sequential(
            nn.Linear(market_feature_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, asset_feature_dim),
            nn.Sigmoid(),
        )
        self.asset_projection = nn.Linear(asset_feature_dim, d_model)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        stock_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.intra_stock = nn.TransformerEncoder(temporal_layer, num_layers=1)
        self.inter_stock = nn.TransformerEncoder(stock_layer, num_layers=1)
        self.temporal_query = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.head = nn.Linear(d_model, horizon)

    def forward(
        self,
        return_history: torch.Tensor,
        state: torch.Tensor | None = None,
        asset_features: torch.Tensor | None = None,
        market_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del return_history, state
        if asset_features is None or market_history is None:
            raise ValueError("MASTER-style model requires asset and market histories")
        batch, length, assets, _ = asset_features.shape
        if assets != self.num_assets:
            raise ValueError("Unexpected number of assets")
        gate = self.market_gate(market_history[:, -1])[:, None, None, :]
        gated = asset_features * gate
        temporal = self.asset_projection(gated).permute(0, 2, 1, 3)
        temporal = self.intra_stock(temporal.reshape(batch * assets, length, -1))
        temporal = temporal.reshape(batch, assets, length, -1).permute(0, 2, 1, 3)
        cross_asset = self.inter_stock(temporal.reshape(batch * length, assets, -1))
        cross_asset = cross_asset.reshape(batch, length, assets, -1).permute(0, 2, 1, 3)
        sequence = cross_asset.reshape(batch * assets, length, -1)
        query = sequence[:, -1:]
        pooled, _ = self.temporal_query(query, sequence, sequence, need_weights=False)
        output = self.head(pooled[:, 0]).reshape(batch, assets, self.horizon)
        return output.transpose(1, 2)


def make_benchmark_model(
    name: str,
    dimensions: dict[str, int],
    parameters: dict[str, Any],
) -> BenchmarkForecaster:
    """Construct a benchmark by registered name without silently changing defaults."""
    common = {
        "num_assets": int(dimensions["num_assets"]),
        "horizon": int(dimensions["horizon"]),
    }
    if name == "StateMLP":
        return StateMLP(
            input_dim=int(dimensions["state_dim"]),
            hidden_dim=int(parameters["hidden_dim"]),
            **common,
        )
    if name == "RawHistoryMLP":
        return RawHistoryMLP(
            history_length=int(dimensions["history_length"]),
            hidden_dim=int(parameters["hidden_dim"]),
            **common,
        )
    if name == "DLinear":
        return DLinearForecaster(
            history_length=int(dimensions["history_length"]),
            moving_average=int(parameters.get("moving_average", 21)),
            individual=bool(parameters.get("individual", False)),
            **common,
        )
    if name == "GRU":
        return GRUForecaster(**common, **parameters)
    if name == "TimeMixer":
        return TimeMixerForecaster(
            history_length=int(dimensions["history_length"]),
            **common,
            **parameters,
        )
    if name == "PatchTST":
        return PatchTSTForecaster(
            history_length=int(dimensions["history_length"]), **common, **parameters
        )
    if name == "iTransformer":
        return ITransformerForecaster(
            history_length=int(dimensions["history_length"]), **common, **parameters
        )
    if name == "MASTER":
        return MarketGuidedStockTransformer(
            asset_feature_dim=int(dimensions["asset_feature_dim"]),
            market_feature_dim=int(dimensions["market_feature_dim"]),
            **common,
            **parameters,
        )
    raise ValueError(f"Unknown benchmark model: {name}")
