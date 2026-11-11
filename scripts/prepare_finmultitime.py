from __future__ import annotations

import argparse
import json

from fconcord.data.finmultitime import prepare_us_finmultitime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a leakage-audited FinMultiTime US daily cohort"
    )
    parser.add_argument("--time-series-zip", required=True)
    parser.add_argument("--description-csv", required=True)
    parser.add_argument("--factor-zip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-assets", type=int, default=32)
    parser.add_argument("--start-date", default="2009-01-01")
    parser.add_argument("--end-date", default="2025-03-28")
    parser.add_argument("--selection-end", default="2017-12-31")
    parser.add_argument("--minimum-selection-observations", type=int, default=2000)
    parser.add_argument("--minimum-output-dates", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_us_finmultitime(
        args.time_series_zip,
        args.description_csv,
        args.factor_zip,
        args.output_dir,
        num_assets=args.num_assets,
        start_date=args.start_date,
        end_date=args.end_date,
        selection_end=args.selection_end,
        minimum_selection_observations=args.minimum_selection_observations,
        minimum_output_dates=args.minimum_output_dates,
    )
    print(json.dumps(manifest["output"], indent=2))


if __name__ == "__main__":
    main()
