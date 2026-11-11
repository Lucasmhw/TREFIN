"""Download and freeze a balanced real Binance Spot daily panel.

The script uses Binance's public market-data-only REST endpoint.  It needs no API
key.  Each normalized per-symbol file, the model-ready panel/factor files, and their
SHA-256 digests are recorded in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from datetime import time as datetime_time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

API_BASE = "https://data-api.binance.vision"
KLINE_PATH = "/api/v3/klines"
DEFAULT_TARGETS = (
    "ADAUSDT",
    "BCHUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "ETCUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "XLMUSDT",
    "XRPUSDT",
)
DEFAULT_FACTORS = ("BTCUSDT", "ETHUSDT")
KLINE_COLUMNS = (
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "close_time_ms",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)


def _utc_milliseconds(value: date) -> int:
    instant = datetime.combine(value, datetime_time.min, tzinfo=timezone.utc)
    return int(instant.timestamp() * 1000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(parameters: dict[str, Any], retries: int = 6) -> list[list[Any]]:
    url = f"{API_BASE}{KLINE_PATH}?{urllib.parse.urlencode(parameters)}"
    request = urllib.request.Request(url, headers={"User-Agent": "f-concord-research/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list):
                raise TypeError(f"Unexpected Binance response for {parameters['symbol']}: {payload}")
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(
                    f"Binance request failed after {retries} attempts for {parameters['symbol']}"
                ) from error
            time.sleep(min(2**attempt, 20))
    raise AssertionError("unreachable")


def download_symbol(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Download one UTC daily series for the inclusive date interval."""
    start_ms = _utc_milliseconds(start)
    end_exclusive_ms = _utc_milliseconds(end + timedelta(days=1))
    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor < end_exclusive_ms:
        batch = _request_json(
            {
                "symbol": symbol,
                "interval": "1d",
                "startTime": cursor,
                "endTime": end_exclusive_ms - 1,
                "limit": 1000,
            }
        )
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 86_400_000
        if next_cursor <= cursor:
            raise RuntimeError(f"Non-increasing Binance cursor for {symbol}")
        cursor = next_cursor
        time.sleep(0.05)
    if not rows:
        raise RuntimeError(f"No observations returned for {symbol}")
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    frame = frame.drop_duplicates("open_time_ms", keep="last").sort_values("open_time_ms")
    frame = frame.loc[
        (frame["open_time_ms"].astype(np.int64) >= start_ms)
        & (frame["open_time_ms"].astype(np.int64) < end_exclusive_ms)
    ].copy()
    numeric_columns = [column for column in KLINE_COLUMNS if column != "ignore"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    frame["date"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    expected = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").tolist()
    actual = frame["date"].tolist()
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"{symbol} is not a complete UTC daily series; missing={missing[:10]} extra={extra[:10]}"
        )
    for column in ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"):
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Nonfinite {column} values in {symbol}")
    if np.any(frame[["open", "high", "low", "close", "quote_volume"]].to_numpy() <= 0):
        raise RuntimeError(f"Nonpositive price or quote volume in {symbol}")
    return frame.reset_index(drop=True)


def build_model_frames(
    series: dict[str, pd.DataFrame], targets: tuple[str, ...], factors: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert verified klines to the exact F-CONCORD CSV schemas."""
    panel_parts: list[pd.DataFrame] = []
    for symbol in targets:
        frame = series[symbol]
        quote_volume = frame["quote_volume"].to_numpy(dtype=np.float64)
        taker_buy_quote = frame["taker_buy_quote_volume"].to_numpy(dtype=np.float64)
        # A signed daily taker-initiated quote-volume imbalance proxy in [-1, 1].
        order_flow = (2.0 * taker_buy_quote - quote_volume) / quote_volume
        if np.any(order_flow < -1.0000001) or np.any(order_flow > 1.0000001):
            raise RuntimeError(f"Order-flow proxy outside [-1, 1] for {symbol}")
        panel_parts.append(
            pd.DataFrame(
                {
                    "date": frame["date"],
                    "asset": symbol,
                    "price": frame["close"].to_numpy(dtype=np.float64),
                    "dollar_volume": quote_volume,
                    "order_flow": order_flow,
                }
            )
        )
    panel = pd.concat(panel_parts, ignore_index=True).sort_values(["date", "asset"])
    factor_frame = pd.DataFrame({"date": series[factors[0]]["date"]})
    for symbol in factors:
        close = series[symbol]["close"].to_numpy(dtype=np.float64)
        returns = np.zeros_like(close)
        returns[1:] = np.log(close[1:] / close[:-1])
        factor_frame[f"{symbol}_log_return"] = returns
    if panel.duplicated(["date", "asset"]).any() or factor_frame["date"].duplicated().any():
        raise RuntimeError("Duplicate output keys after conversion")
    if panel.isna().any().any() or factor_frame.isna().any().any():
        raise RuntimeError("Missing output values after conversion")
    return panel.reset_index(drop=True), factor_frame.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 6, 30))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/binance_spot_daily")
    )
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--factors", nargs="+", default=list(DEFAULT_FACTORS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start >= args.end:
        raise ValueError("--start must be earlier than --end")
    targets = tuple(dict.fromkeys(str(value).upper() for value in args.targets))
    factors = tuple(dict.fromkeys(str(value).upper() for value in args.factors))
    if set(targets) & set(factors):
        raise ValueError("Target and factor symbol sets must be disjoint")
    if len(factors) < 1:
        raise ValueError("At least one factor symbol is required")
    output_dir = args.output_dir.resolve()
    panel_path = output_dir / "daily_panel.csv"
    factor_path = output_dir / "factors.csv"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (panel_path, factor_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing dataset without --force: {existing}")
    kline_dir = output_dir / "klines"
    kline_dir.mkdir(parents=True, exist_ok=True)

    symbols = tuple(dict.fromkeys(targets + factors))
    series: dict[str, pd.DataFrame] = {}
    symbol_records: dict[str, Any] = {}
    for position, symbol in enumerate(symbols, start=1):
        print(f"[{position}/{len(symbols)}] downloading {symbol}", flush=True)
        frame = download_symbol(symbol, args.start, args.end)
        path = kline_dir / f"{symbol}.csv"
        frame.to_csv(path, index=False, float_format="%.12g")
        series[symbol] = frame
        symbol_records[symbol] = {
            "rows": len(frame),
            "first_date": frame["date"].iloc[0],
            "last_date": frame["date"].iloc[-1],
            "sha256": _sha256(path),
            "normalized_file": str(path),
        }

    panel, factor_frame = build_model_frames(series, targets, factors)
    panel.to_csv(panel_path, index=False, float_format="%.12g")
    factor_frame.to_csv(factor_path, index=False, float_format="%.12g")
    expected_days = (args.end - args.start).days + 1
    if len(panel) != expected_days * len(targets) or len(factor_frame) != expected_days:
        raise RuntimeError("Output row count is inconsistent with the balanced-panel contract")
    manifest = {
        "dataset_id": f"binance_spot_1d_{args.start.isoformat()}_{args.end.isoformat()}",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "Binance public market data",
            "api_base": API_BASE,
            "endpoint": KLINE_PATH,
            "interval": "1d",
            "timezone": "UTC",
            "authentication": "none",
        },
        "requested_range_inclusive": {
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
        },
        "targets": list(targets),
        "factors": list(factors),
        "definitions": {
            "price": "UTC daily close",
            "dollar_volume": "daily quote asset volume in USDT",
            "order_flow": "(2*taker_buy_quote_volume-quote_volume)/quote_volume; daily taker imbalance proxy",
            "factor_return": "one-day log close-to-close return; zero only on snapshot first date",
        },
        "symbols": symbol_records,
        "outputs": {
            "daily_panel.csv": {"rows": len(panel), "sha256": _sha256(panel_path)},
            "factors.csv": {"rows": len(factor_frame), "sha256": _sha256(factor_path)},
        },
        "validation": {
            "expected_calendar_days": expected_days,
            "balanced_panel": True,
            "no_missing_values": True,
            "strictly_positive_prices_and_quote_volume": True,
            "order_flow_proxy_range": [-1.0, 1.0],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": manifest["dataset_id"], "output_dir": str(output_dir), "panel_rows": len(panel), "factor_rows": len(factor_frame)}, indent=2))


if __name__ == "__main__":
    main()
