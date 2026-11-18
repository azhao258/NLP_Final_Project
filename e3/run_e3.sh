#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p e3_runs
mode="${1:-prepare}"
case "$mode" in
  prepare)
    python verify_assets.py
    python e3/prepare_data.py
    ;;
  pilot)
    python -u e3/train.py --output e3_runs/pilot --pilot-updates 10 2>&1 | tee e3_runs/pilot.log
    python e3/check_reload.py --checkpoint e3_runs/pilot/last.pt
    ;;
  train)
    python -u e3/train.py --output e3_runs/control 2>&1 | tee e3_runs/control.log
    python e3/check_reload.py --checkpoint e3_runs/control/last.pt
    ;;
  resume)
    python -u e3/train.py --output e3_runs/control --resume 2>&1 | tee -a e3_runs/control.log
    python e3/check_reload.py --checkpoint e3_runs/control/last.pt
    ;;
  evaluate)
    python e3/evaluate.py --checkpoint e3_runs/control/last.pt --output e3_runs/control_test
    python e3/evaluate.py --checkpoint e3_runs/control/last.pt --split validation --explain --output e3_runs/control_diagnostics
    ;;
  *) echo "Usage: bash e3/run_e3.sh prepare|pilot|train|resume|evaluate"; exit 2 ;;
esac
