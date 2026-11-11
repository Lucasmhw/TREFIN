from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fconcord.data.dataset import load_panel_csv
from fconcord.data.fintsbridge import EXPECTED_BTCF_COLUMNS, prepare_btcf


def test_btcf_conversion_preserves_log_returns_and_interval_order_flow(tmp_path) -> None:
    dates = pd.date_range("2020-01-01 01:00:00", periods=8, freq="h")
    future_close = 100.0 + np.asarray([0.0, 0.01, 0.03, 0.02, 0.04, 0.05, 0.04, 0.07])
    spot_close = 100.0 + np.asarray([0.0, 0.02, 0.01, 0.03, 0.02, 0.06, 0.05, 0.08])
    values = {"date": dates}
    for market, close in (("future", future_close), ("spot", spot_close)):
        values[f"open_{market}"] = close
        values[f"high_{market}"] = close + 0.01
        values[f"low_{market}"] = close - 0.01
        values[f"close_{market}"] = close
        values[f"volume_{market}"] = np.log1p(np.full(len(dates), 10.0))
        values[f"taker_buy_volume_{market}"] = np.log1p(
            np.full(len(dates), 6.0)
        )
    source = tmp_path / "BTCF.csv"
    pd.DataFrame(values, columns=EXPECTED_BTCF_COLUMNS).to_csv(source, index=False)
    output = tmp_path / "prepared"
    manifest = prepare_btcf(source, output, expected_rows=None)
    loaded = load_panel_csv(
        output / "panel.csv",
        output / "factors.csv",
        {
            "risk_free_policy": "explicit_zero",
            "risk_free_column": "risk_free",
            "order_flow_mode": "provided_interval",
        },
    )
    np.testing.assert_allclose(loaded["total_log_returns"][1:, 0], np.diff(future_close))
    np.testing.assert_allclose(loaded["total_log_returns"][1:, 1], np.diff(spot_close))
    np.testing.assert_allclose(loaded["order_flow"], 0.2, atol=1e-10)
    np.testing.assert_allclose(
        loaded["factors"][:, 0],
        0.5
        * np.column_stack(
            [
                np.r_[0.0, np.diff(future_close)],
                np.r_[0.0, np.diff(spot_close)],
            ]
        ).sum(axis=1),
    )
    assert manifest["benchmark_contract"]["input_length"] == 512
    assert manifest["validation"]["future_values_used_in_measurement"] is False


def test_btcf_config_uses_exact_official_split_boundaries() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "fintsbridge_btcf_debug.yaml"
    )
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))["data"]
    assert "train_fraction" not in data
    assert "val_fraction" not in data
    assert data["split_dates"] == {
        "train_start": "2020-01-22 08:00:00",
        "train_end": "2023-06-11 13:00:00",
        "val_start": "2023-06-11 14:00:00",
        "val_end": "2023-12-08 02:00:00",
        "test_start": "2023-12-08 03:00:00",
        "test_end": "2024-11-30 23:00:00",
    }
