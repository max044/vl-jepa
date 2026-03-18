#!/bin/bash
# Quick sanity check - Run 1 epoch with 10 samples to verify model loads
# Usage: bash scripts/sanity_check.sh

set -e

cd ~/vl-jepa

echo "=== Pulling latest code ==="
git pull origin main

echo "=== Loading environment ==="
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=== Quick sanity check (10 samples, 1 epoch) ==="
python3 train.py \
    --epochs 1 \
    --batch-size 2 \
    --debug \
    --debug-samples 10 \
    --no-wandb \
    --device cuda \
    2>&1 | tee sanity_check.log

echo ""
echo "=== Sanity check complete ==="
tail -20 sanity_check.log