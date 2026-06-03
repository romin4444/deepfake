#!/usr/bin/env bash
# run_all.sh — end-to-end: train then evaluate.
# Usage: bash scripts/run_all.sh [config] [extra overrides...]
set -e
CONFIG="${1:-configs/default.yaml}"
shift || true
RUN_DIR="outputs/run_$(date +%Y%m%d_%H%M%S)"

echo "=== Training ==="
python -m src.train --config "$CONFIG" output_dir="$RUN_DIR" "$@"

echo "=== Evaluating ==="
python -m src.evaluate --config "$CONFIG" --ckpt "$RUN_DIR/best.pt" \
  output_dir="$RUN_DIR" "$@"

echo "=== Done. Results in $RUN_DIR ==="
echo "  best.pt           trained checkpoint"
echo "  history.json      per-epoch training log"
echo "  eval_report.json  cross-dataset + robustness metrics"
