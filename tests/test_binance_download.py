from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "download_binance_daily.py"
SPEC = importlib.util.spec_from_file_location("download_binance_daily", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame(close: list[float], quote: list[float], taker: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "close": close,
            "quote_volume": quote,
            "taker_buy_quote_volume": taker,
        }
    )


def test_binance_frame_conversion_definitions_are_exact() -> None:
    series = {
        "ALTUSDT": _frame([10.0, 11.0, 9.0], [100.0, 200.0, 400.0], [60.0, 50.0, 400.0]),
        "BTCUSDT": _frame([100.0, 105.0, 102.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]),
    }
    panel, factors = MODULE.build_model_frames(series, ("ALTUSDT",), ("BTCUSDT",))
    np.testing.assert_allclose(panel["order_flow"], [0.2, -0.5, 1.0])
    np.testing.assert_allclose(
        factors["BTCUSDT_log_return"], [0.0, np.log(1.05), np.log(102.0 / 105.0)]
    )
    assert list(panel.columns) == ["date", "asset", "price", "dollar_volume", "order_flow"]
