from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_csv_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in archive.infolist():
        normalized = PurePosixPath(info.filename.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(f"Unsafe ZIP member: {info.filename}")
        if normalized.suffix.lower() == ".csv":
            members.append(info)
    if not members:
        raise ValueError("Time-series ZIP contains no CSV files")
    return members


def parse_ken_french_daily_zip(path: str | Path) -> pd.DataFrame:
    """Parse Ken French's official daily five-factor CSV archive.

    Factor columns are converted from percentage simple returns to decimal
    returns.  RF is converted to a one-period log return so it can be
    subtracted exactly from log total asset returns.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        members = _validated_csv_members(archive)
        if len(members) != 1:
            raise ValueError("Expected one CSV in the Ken French archive")
        text = archive.read(members[0]).decode("latin1")
    rows: list[list[str]] = []
    header: list[str] | None = None
    for raw_line in text.splitlines():
        fields = [value.strip() for value in next(csv.reader([raw_line]))]
        if header is None and fields and fields[0] == "" and "Mkt-RF" in fields:
            header = ["date", *fields[1:]]
            continue
        if header is None or not fields or len(fields[0]) != 8 or not fields[0].isdigit():
            continue
        if len(fields) != len(header):
            continue
        rows.append(fields)
    if header is None or not rows:
        raise ValueError("Could not locate the daily factor table")
    frame = pd.DataFrame(rows, columns=header)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    numeric_columns = [column for column in frame.columns if column != "date"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric) / 100.0
    frame["risk_free"] = np.log1p(frame.pop("RF"))
    frame = frame.rename(columns={"Mkt-RF": "MKT_RF"})
    frame["available_at"] = frame["date"]
    return frame


def _read_stock_csv(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> pd.DataFrame:
    raw = archive.read(info).decode("utf-8-sig")
    frame = pd.read_csv(io.StringIO(raw))
    required = {"Date", "Close", "Volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{info.filename} is missing columns {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["Date"].astype(str).str[:10], errors="raise")
    frame["price"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["Volume"], errors="coerce")
    frame = frame[["date", "price", "volume"]]
    if frame["date"].duplicated().any():
        raise ValueError(f"{info.filename} contains duplicate dates")
    return frame.set_index("date").sort_index()


def prepare_us_finmultitime(
    time_series_zip: str | Path,
    description_csv: str | Path,
    factor_zip: str | Path,
    output_directory: str | Path,
    num_assets: int = 32,
    start_date: str = "2009-01-01",
    end_date: str = "2025-03-28",
    selection_end: str = "2017-12-31",
    minimum_selection_observations: int = 2000,
    minimum_output_dates: int = 1000,
) -> dict[str, Any]:
    """Create a fixed, balanced US equity cohort using training-era liquidity only."""
    if num_assets < 2:
        raise ValueError("num_assets must be at least two")
    source_zip = Path(time_series_zip)
    description_path = Path(description_csv)
    factor_path = Path(factor_zip)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    descriptions = pd.read_csv(description_path)
    descriptions["ticker"] = descriptions["stock_name"].astype(str).str.upper()
    sectors = descriptions.set_index("ticker")["Sector"].to_dict()

    with zipfile.ZipFile(source_zip) as archive:
        members = _validated_csv_members(archive)
        member_by_ticker = {
            PurePosixPath(info.filename).stem.upper(): info for info in members
        }
        candidates: list[dict[str, Any]] = []
        selection_start_timestamp = pd.Timestamp(start_date)
        selection_end_timestamp = pd.Timestamp(selection_end)
        for ticker, info in member_by_ticker.items():
            sector = sectors.get(ticker)
            if not isinstance(sector, str) or not sector or sector == "Client Error":
                continue
            frame = _read_stock_csv(archive, info)
            selection = frame.loc[selection_start_timestamp:selection_end_timestamp]
            valid = selection[
                np.isfinite(selection["price"])
                & (selection["price"] > 0.0)
                & np.isfinite(selection["volume"])
                & (selection["volume"] > 0.0)
            ]
            if len(valid) < minimum_selection_observations:
                continue
            dollar_volume = valid["price"] * valid["volume"]
            candidates.append(
                {
                    "asset": ticker,
                    "sector": sector,
                    "selection_observations": len(valid),
                    "selection_median_dollar_volume": float(
                        statistics.median(dollar_volume)
                    ),
                }
            )
        candidates.sort(
            key=lambda row: row["selection_median_dollar_volume"], reverse=True
        )
        selected = candidates[:num_assets]
        if len(selected) != num_assets:
            raise ValueError("Too few eligible assets for the requested cohort")
        selected_frames: dict[str, pd.DataFrame] = {}
        for row in selected:
            ticker = row["asset"]
            selected_frames[ticker] = _read_stock_csv(
                archive, member_by_ticker[ticker]
            ).loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]

    factors = parse_ken_french_daily_zip(factor_path).set_index("date")
    common_dates = factors.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)].index
    missing_by_asset: dict[str, int] = {}
    for ticker, frame in selected_frames.items():
        usable = frame.index[
            np.isfinite(frame["price"])
            & (frame["price"] > 0.0)
            & np.isfinite(frame["volume"])
            & (frame["volume"] > 0.0)
        ]
        missing_by_asset[ticker] = len(common_dates.difference(usable))
        common_dates = common_dates.intersection(usable)
    common_dates = common_dates.sort_values()
    if len(common_dates) < minimum_output_dates:
        raise ValueError("Balanced cohort has too few common dates")

    panel_rows: list[pd.DataFrame] = []
    for row in selected:
        ticker = row["asset"]
        aligned = selected_frames[ticker].reindex(common_dates)
        panel_rows.append(
            pd.DataFrame(
                {
                    "date": common_dates,
                    "available_at": common_dates,
                    "asset": ticker,
                    "price": aligned["price"].to_numpy(),
                    "dollar_volume": (
                        aligned["price"] * aligned["volume"]
                    ).to_numpy(),
                }
            )
        )
    panel = pd.concat(panel_rows, ignore_index=True).sort_values(["date", "asset"])
    factor_output = factors.reindex(common_dates).reset_index()
    panel_path = output / "panel.csv"
    factors_path = output / "factors.csv"
    panel.to_csv(panel_path, index=False)
    factor_output.to_csv(factors_path, index=False)

    manifest = {
        "dataset_layer": "FinMultiTime daily cross-market main layer / US cohort",
        "source": {
            "finmultitime_zip": {
                "path": str(source_zip.resolve()),
                "sha256": _sha256(source_zip),
            },
            "description_csv": {
                "path": str(description_path.resolve()),
                "sha256": _sha256(description_path),
            },
            "ken_french_factor_zip": {
                "path": str(factor_path.resolve()),
                "sha256": _sha256(factor_path),
            },
        },
        "selection": {
            "rule": (
                "sector-labelled equities ranked by median adjusted-close times volume; "
                "ranking uses only dates through selection_end"
            ),
            "start_date": start_date,
            "selection_end": selection_end,
            "minimum_selection_observations": minimum_selection_observations,
            "minimum_output_dates": minimum_output_dates,
            "requested_assets": num_assets,
            "selected_assets": selected,
            "test_or_validation_values_used_for_ranking": False,
        },
        "output": {
            "start_date": str(common_dates[0].date()),
            "end_date": str(common_dates[-1].date()),
            "dates": len(common_dates),
            "assets": num_assets,
            "panel_rows": len(panel),
            "panel_sha256": _sha256(panel_path),
            "factors_sha256": _sha256(factors_path),
            "dropped_calendar_dates_for_balance": int(
                len(factors.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)])
                - len(common_dates)
            ),
            "preintersection_missing_by_asset": missing_by_asset,
        },
        "financial_semantics": {
            "price": (
                "FinMultiTime Close; historical split-scale inspection is consistent with "
                "Yahoo adjusted OHLC, but the upstream archive has no explicit adjustment flag"
            ),
            "dollar_volume": "FinMultiTime Close times Volume",
            "order_flow": "manual daily signed-dollar-volume proxy computed downstream",
            "factors": "Ken French MKT_RF, SMB, HML, RMW, CMA decimal daily returns",
            "risk_free": "log(1 + Ken French RF decimal simple daily return)",
            "market_weight": "preregistered equal weight because point-in-time market cap is absent",
        },
        "limitations": {
            "historical_index_membership_available": False,
            "delisting_return_field_available": False,
            "factor_vintage_is_first_release": False,
            "survivorship_claim_allowed": False,
            "index_constituent_claim_allowed": False,
            "cohort_claim_allowed": True,
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
