#!/bin/bash
# Automated GPU training pipeline
# Usage: bash run_gpu.sh [--epochs 100] [--batch-size 1] [--lr 1e-3]
set -euo pipefail

cd "$(dirname "$0")/../.."
SUB=submissions/autoresearch
EPOCHS="${1:-100}"
BS="${2:-1}"
LR="${3:-1e-3}"

echo "=== Compress (with preprocessing) ==="
uv run bash $SUB/compress.sh

echo ""
echo "=== Train REN (epochs=$EPOCHS bs=$BS lr=$LR) ==="
uv run python -u $SUB/train_ren.py --epochs $EPOCHS --batch-size $BS --lr $LR

echo ""
echo "=== Package model ==="
uv run python $SUB/package_model.py

echo ""
echo "=== Evaluate ==="
uv run bash evaluate.sh --submission-dir $SUB --device cpu

echo ""
echo "=== RESULT ==="
cat $SUB/report.txt
