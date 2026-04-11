#!/usr/bin/env bash
# Run a single experiment cycle: compress -> (optionally train REN) -> evaluate
# Usage: ./run_experiment.sh [--fast] [--train-ren] [--device mps|cpu|cuda] [--epochs 50]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB_DIR="${HERE}/submissions/autoresearch"

TRAIN_REN=false
DEVICE="mps"
EPOCHS=50
COMPRESS_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) COMPRESS_ARGS="--fast"; shift ;;
    --train-ren) TRAIN_REN=true; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== Step 1: Compress ==="
cd "$HERE"
bash "${SUB_DIR}/compress.sh" $COMPRESS_ARGS

if [ "$TRAIN_REN" = true ]; then
  echo ""
  echo "=== Step 2: Train REN ==="
  cd "$HERE"
  python "${SUB_DIR}/train_ren.py" --epochs "$EPOCHS" --batch-size 1
  python "${SUB_DIR}/package_model.py"
fi

echo ""
echo "=== Step 3: Evaluate ==="
cd "$HERE"
bash evaluate.sh --submission-dir "${SUB_DIR}" --device "$DEVICE"

echo ""
echo "=== Results ==="
cat "${SUB_DIR}/report.txt"

# Extract score for easy parsing
SCORE=$(grep "Final score" "${SUB_DIR}/report.txt" | grep -oE '[0-9]+\.[0-9]+$')
echo ""
echo "SCORE: ${SCORE}"
