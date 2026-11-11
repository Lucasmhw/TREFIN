from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

import numpy as np
import pandas as pd

OFFICIAL_REPOSITORY_ID = "FinTSBridge-0D6C"
OFFICIAL_BTCF_REPOSITORY_SHA = "fb187ff4"
OFFICIAL_BTCF_PATH = (
    "FinTSBridge_models/TSLib_baseline/dataset/FBD/BTCF.csv"
)
OFFICIAL_BTCF_URL = (
    "https://anonymous.4open.science/api/repo/"
    f"{OFFICIAL_REPOSITORY_ID}/file/{OFFICIAL_BTCF_PATH}"
)
EXPECTED_BTCF_COLUMNS = (
    "date",
    "open_future",
    "high_future",
    "low_future",
    "close_future",
    "volume_future",
    "taker_buy_volume_future",
    "open_spot",
    "high_spot",
    "low_spot",
    "close_spot",
    "volume_spot",
    "taker_buy_volume_spot",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_btcf(
    destination: str | Path,
    *,
    force: bool = False,
    url: str = OFFICIAL_BTCF_URL,
) -> dict[str, Any]:
    """Download the exact BTCF CSV distributed with the FinTSBridge code.

    The anonymous repository exposes a short repository-file hash rather than
    a full content digest.  We preserve that identifier and compute a full
    SHA-256 digest locally instead of pretending that the short value is a
    cryptographic checksum of the downloaded bytes.
    """
    destination = Path(destination)
    if destination.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {destination}; pass force=True explicitly"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib_request.Request(
        url,
        headers={"User-Agent": "f-concord-research/0.1"},
    )
    try:
        with urllib_request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        preview = pd.read_csv(temporary, nrows=5)
        if tuple(preview.columns) != EXPECTED_BTCF_COLUMNS:
            raise ValueError("Downloaded BTCF file has an unexpected schema")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(destination.resolve()),
        "sha256": _sha256(destination),
        "bytes": destination.stat().st_size,
        "url": url,
        "anonymous_repository_file_sha": OFFICIAL_BTCF_REPOSITORY_SHA,
    }


def prepare_btcf(
    source_csv: str | Path,
    output_directory: str | Path,
    *,
    expected_rows: int | None = 43_014,
) -> dict[str, Any]:
    """Convert FinTSBridge BTCF into the causal F-CONCORD panel schema.

    FinTSBridge stores normalized log-price paths as
    ``Z_t = log(P_t / P_0) + 100`` and log-volumes as ``log(V_t + 1)``.
    We reconstruct a positive relative price whose log differences equal the
    benchmark's original log returns.  Absolute price levels are not needed by
    F-CONCORD; the resulting dollar-volume proxy is therefore identified only
    up to an instrument-specific positive constant, which is recorded in the
    manifest and removed by train-only robust standardization downstream.
    """
    source = Path(source_csv)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source)
    if tuple(frame.columns) != EXPECTED_BTCF_COLUMNS:
        raise ValueError("BTCF source columns do not match the official code artifact")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} BTCF rows, received {len(frame)}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("BTCF timestamps must be unique and strictly increasing")
    numeric_columns = [column for column in frame.columns if column != "date"]
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("BTCF contains missing or non-finite numeric values")

    panel_parts: list[pd.DataFrame] = []
    return_columns: list[np.ndarray] = []
    order_flow_ranges: dict[str, list[float]] = {}
    for market in ("future", "spot"):
        close_path = frame[f"close_{market}"].to_numpy(dtype=np.float64)
        # The arbitrary constant disappears from every log return.
        relative_price = np.exp(close_path - close_path[0])
        base_volume = np.expm1(frame[f"volume_{market}"].to_numpy(dtype=np.float64))
        taker_buy_volume = np.expm1(
            frame[f"taker_buy_volume_{market}"].to_numpy(dtype=np.float64)
        )
        if np.any(relative_price <= 0.0) or np.any(base_volume <= 0.0):
            raise ValueError(f"Reconstructed {market} price/volume is not strictly positive")
        if np.any(taker_buy_volume < 0.0) or np.any(taker_buy_volume > base_volume * 1.000001):
            raise ValueError(f"Reconstructed {market} taker-buy volume is invalid")
        order_flow = (2.0 * taker_buy_volume - base_volume) / base_volume
        if np.any(order_flow < -1.000001) or np.any(order_flow > 1.000001):
            raise ValueError(f"Reconstructed {market} order flow is outside [-1, 1]")
        order_flow = np.clip(order_flow, -1.0, 1.0)
        log_returns = np.zeros_like(close_path)
        log_returns[1:] = np.diff(close_path)
        return_columns.append(log_returns)
        order_flow_ranges[market] = [float(order_flow.min()), float(order_flow.max())]
        panel_parts.append(
            pd.DataFrame(
                {
                    "date": frame["date"],
                    "available_at": frame["date"],
                    "asset": f"BTC_{market.upper()}",
                    "price": relative_price,
                    "dollar_volume": relative_price * base_volume,
                    "order_flow": order_flow,
                }
            )
        )

    panel = pd.concat(panel_parts, ignore_index=True).sort_values(["date", "asset"])
    factor = pd.DataFrame(
        {
            "date": frame["date"],
            "available_at": frame["date"],
            # A causal common-BTC factor separates shared price discovery from
            # the spot/futures basis.  It uses only contemporaneously observed
            # returns and never a future forecast label.
            "BTC_COMMON": 0.5 * (return_columns[0] + return_columns[1]),
        }
    )
    panel_path = output / "panel.csv"
    factor_path = output / "factors.csv"
    panel.to_csv(panel_path, index=False, float_format="%.12g")
    factor.to_csv(factor_path, index=False, float_format="%.12g")

    timestamp_delta = frame["date"].diff().dropna().dt.total_seconds() / 3600.0
    gap_rows = int((timestamp_delta != 1.0).sum())
    manifest = {
        "dataset_layer": "FinTSBridge BTCF multi-task cross-market layer",
        "source": {
            "paper": "https://arxiv.org/abs/2503.06928",
            "code_artifact": (
                "https://anonymous.4open.science/r/FinTSBridge-0D6C/"
                + OFFICIAL_BTCF_PATH
            ),
            "anonymous_repository_id": OFFICIAL_REPOSITORY_ID,
            "anonymous_repository_file_sha": OFFICIAL_BTCF_REPOSITORY_SHA,
            "source_csv": str(source.resolve()),
            "source_sha256": _sha256(source),
        },
        "benchmark_contract": {
            "input_length": 512,
            "forecast_horizons": [5, 21, 63, 126],
            "frequency": "hourly observations",
            "official_rows": 43_014,
            "official_range": ["2020-01-01 01:00:00", "2024-11-30 23:00:00"],
            "official_split_rows": {"train": 30_109, "validation": 4_303, "test": 8_602},
        },
        "output": {
            "rows_per_asset": len(frame),
            "assets": ["BTC_FUTURE", "BTC_SPOT"],
            "first_timestamp": frame["date"].iloc[0].isoformat(),
            "last_timestamp": frame["date"].iloc[-1].isoformat(),
            "non_unit_hour_gaps": gap_rows,
            "panel_sha256": _sha256(panel_path),
            "factors_sha256": _sha256(factor_path),
        },
        "financial_semantics": {
            "target": "future spot and perpetual-futures excess log returns",
            "relative_price": "exp(close_Z - first_close_Z); log differences equal BTCF returns",
            "dollar_volume": (
                "relative_price * expm1(log base volume); identified up to the unavailable "
                "original price normalization constant"
            ),
            "order_flow": "2 * taker-buy base volume / total base volume - 1 per hour",
            "factor": "equal-weight contemporaneous spot/futures log return (BTC_COMMON)",
            "risk_free": "explicit zero per hour, preregistered for the crypto benchmark",
        },
        "limitations": {
            "absolute_original_price_recoverable": False,
            "absolute_dollar_volume_comparable_across_assets": False,
            "treat_gaps_as_observed_sequence_steps": True,
            "gap_count": gap_rows,
            "official_preprocessed_artifact_used": True,
        },
        "validation": {
            "balanced_panel": True,
            "no_missing_values": True,
            "strictly_positive_prices_and_dollar_volume": True,
            "order_flow_ranges": order_flow_ranges,
            "future_values_used_in_measurement": False,
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
