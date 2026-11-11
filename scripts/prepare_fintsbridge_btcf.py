"""Download the official FinTSBridge BTCF artifact and prepare F-CONCORD CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fconcord.data.fintsbridge import download_official_btcf, prepare_btcf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("data/cache/fintsbridge/BTCF.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/fintsbridge_btcf"),
    )
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.cache_file.exists() or args.force_download:
        record = download_official_btcf(args.cache_file, force=args.force_download)
        print(json.dumps({"download": record}, indent=2), flush=True)
    manifest = prepare_btcf(args.cache_file, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
