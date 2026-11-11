#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${FCONCORD_REPO_DIR:-/root/autodl-tmp/f-concord}"
cd "$REPO_DIR"
mkdir -p server_logs
LOG_FILE="server_logs/debug_bootstrap_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "repo=$REPO_DIR"
echo "bootstrap_log=$LOG_FILE"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m fconcord run --config configs/debug.yaml
