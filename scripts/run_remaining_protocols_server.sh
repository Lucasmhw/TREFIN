#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/f-concord

wait_pid="${1:?pass the active native benchmark PID}"
while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done

reference_protocol=/root/autodl-tmp/f-concord/runs/protocols/finmultitime_us_rolling_3fold_5seed_v1
native_protocol=/root/autodl-tmp/f-concord/runs/benchmarks/finmultitime_us_native_3fold_5seed_v1
timemixer_protocol=/root/autodl-tmp/f-concord/runs/benchmarks/finmultitime_us_timemixer_3fold_5seed_v1
foundation_protocol=/root/autodl-tmp/f-concord/runs/foundations/finmultitime_us_official_3fold_v1
ablation_protocol=/root/autodl-tmp/f-concord/runs/ablations/finmultitime_us_retrained_3fold_5seed_v1

.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("/root/autodl-tmp/f-concord/runs/benchmarks/finmultitime_us_native_3fold_5seed_v1/protocol_summary.json")
summary = json.loads(path.read_text())
if not summary["all_runs_complete"] or summary["runs_completed"] != 15:
    raise SystemExit("native benchmark protocol did not complete all 15 runs")
PY

.venv/bin/python -m fconcord benchmark-protocol \
  --config configs/finmultitime_us_timemixer_supplement.yaml \
  --reference-protocol "${reference_protocol}" \
  --output-dir "${timemixer_protocol}"

env \
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  HF_HUB_DISABLE_XET=1 \
  HF_HUB_ENABLE_HF_TRANSFER=0 \
  .foundation-venv/bin/python -m fconcord foundation-protocol \
  --config configs/finmultitime_us_foundation.yaml \
  --reference-protocol "${reference_protocol}" \
  --output-dir "${foundation_protocol}"

.venv/bin/python -m fconcord ablation-protocol \
  --reference-protocol "${reference_protocol}" \
  --output-dir "${ablation_protocol}"

.venv/bin/python -m fconcord benchmark-audit \
  --protocol-dir "${native_protocol}" \
  --reference-protocol "${reference_protocol}"

.venv/bin/python -m fconcord ablation-audit \
  --protocol-dir "${ablation_protocol}" \
  --reference-protocol "${reference_protocol}"

.venv/bin/python -m fconcord benchmark-audit \
  --protocol-dir "${timemixer_protocol}" \
  --reference-protocol "${reference_protocol}"

.venv/bin/python -m fconcord comparison \
  --reference-protocol "${reference_protocol}" \
  --native-protocol "${native_protocol}" \
  --timemixer-protocol "${timemixer_protocol}" \
  --foundation-protocol "${foundation_protocol}" \
  --ablation-protocol "${ablation_protocol}" \
  --output-dir /root/autodl-tmp/f-concord/runs/comparisons/finmultitime_us_complete_v1

.venv/bin/python -m fconcord comparison-figures \
  --comparison-dir /root/autodl-tmp/f-concord/runs/comparisons/finmultitime_us_complete_v1

printf '%s\n' "ALL_REMAINING_PROTOCOLS_COMPLETE"
