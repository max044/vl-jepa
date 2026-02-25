#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# VL-JEPA Local Verification Script
# Use this to verify the W&B integration and code before cloud launch.
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ── 1. Determine Device ──────────────────────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
    DEVICE="mps"
else
    DEVICE="cpu"
fi

echo "▸ Detected device: $DEVICE"

# ── 2. Run Minimal Training ──────────────────────────────────────
# --debug uses a tiny subset of data (~100 samples)
# --epochs 1 ensures it runs quickly
# WANDB_MODE=offline prevents data from being uploaded
echo "▸ Starting minimal local training (offline W&B)..."

WANDB_MODE=offline uv run train.py \
    --device "$DEVICE" \
    --debug \
    --epochs 1 \
    --batch-size 2 \
    --wandb-run-name "local-test-$(date +%Y%m%d-%H%M%S)"

echo ""
echo "═══════════════════════════════════════════"
echo "✓ Local verification complete!"
echo "Check the 'wandb/' folder to see the local logs."
echo "If this worked without error, you are ready for the cloud."
echo "═══════════════════════════════════════════"
