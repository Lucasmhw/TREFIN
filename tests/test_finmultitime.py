import zipfile

import pandas as pd

from fconcord.data.finmultitime import (
    parse_ken_french_daily_zip,
    prepare_us_finmultitime,
)


def _write_fixture_archives(tmp_path):
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    stock_zip = tmp_path / "stocks.zip"
    with zipfile.ZipFile(stock_zip, "w") as archive:
        for ticker, volume in [("aaa", 300.0), ("bbb", 200.0), ("fund", 1000.0)]:
            frame = pd.DataFrame(
                {
                    "Date": dates,
                    "Open": range(10, 16),
                    "High": range(11, 17),
                    "Low": range(9, 15),
                    "Close": range(10, 16),
                    "Volume": [volume] * 6,
                    "Dividends": [0.0] * 6,
                    "Stock Splits": [0.0] * 6,
                }
            )
            archive.writestr(f"S&P500_time_series/{ticker}.csv", frame.to_csv(index=False))
    description = tmp_path / "description.csv"
    pd.DataFrame(
        {
            "stock_name": ["aaa", "bbb", "fund"],
            "Sector": ["Technology", "Industrials", None],
        }
    ).to_csv(description, index=False)
    factor_zip = tmp_path / "factors.zip"
    factor_lines = [
        "fixture",
        ",Mkt-RF,SMB,HML,RMW,CMA,RF",
        *[
            f"{date:%Y%m%d},1.0,0.1,0.2,0.3,0.4,0.01"
            for date in dates
        ],
    ]
    with zipfile.ZipFile(factor_zip, "w") as archive:
        archive.writestr("factors.csv", "\n".join(factor_lines))
    return stock_zip, description, factor_zip


def test_finmultitime_preparation_uses_training_era_liquidity_and_writes_manifest(
    tmp_path,
) -> None:
    stock_zip, description, factor_zip = _write_fixture_archives(tmp_path)
    factors = parse_ken_french_daily_zip(factor_zip)
    assert "risk_free" in factors
    assert "RF" not in factors
    output = tmp_path / "prepared"
    manifest = prepare_us_finmultitime(
        stock_zip,
        description,
        factor_zip,
        output,
        num_assets=2,
        start_date="2020-01-01",
        end_date="2020-01-06",
        selection_end="2020-01-04",
        minimum_selection_observations=3,
        minimum_output_dates=5,
    )
    selected = [row["asset"] for row in manifest["selection"]["selected_assets"]]
    assert selected == ["AAA", "BBB"]
    assert manifest["selection"]["test_or_validation_values_used_for_ranking"] is False
    assert manifest["limitations"]["index_constituent_claim_allowed"] is False
    assert (output / "panel.csv").is_file()
    assert (output / "factors.csv").is_file()
    assert (output / "manifest.json").is_file()
