#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/f-concord
mkdir -p logs
.venv/bin/python scripts/download_binance_daily.py \
  --start 2021-01-01 \
  --end 2026-06-30 \
  --output-dir data/raw/binance_spot_daily \
  --force 2>&1 | tee "logs/binance_download_$(date -u +%Y%m%dT%H%M%SZ).log"
.venv/bin/python -m pytest
.venv/bin/python -m fconcord run --config configs/binance_daily_real.yaml
