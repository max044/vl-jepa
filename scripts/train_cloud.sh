#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# VL-JEPA Cloud Training Launcher
# Optimized defaults for A100/H100 GPU instances
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Load environment ────────────────────────────────────────────
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ── Configuration (override via env vars) ───────────────────────
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-vl-jepa}"
WANDB_ID="${WANDB_ID:-}"
CHECKPOINT="${CHECKPOINT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Construction des arguments de resume si présents
RESUME_ARGS=""
if [ -n "$WANDB_ID" ]; then
    RESUME_ARGS="$RESUME_ARGS --wandb-id $WANDB_ID"
fi
if [ -n "$CHECKPOINT" ]; then
    RESUME_ARGS="$RESUME_ARGS --checkpoint $CHECKPOINT"
fi

# ── Pre-flight checks ──────────────────────────────────────────
echo "╔══════════════════════════════════════════╗"
echo "║       VL-JEPA Cloud Training             ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check GPU
if ! nvidia-smi &>/dev/null; then
    echo "✗ No NVIDIA GPU detected. Aborting."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "▸ GPU: $GPU_NAME ($GPU_MEM)"

# Auto-tune batch size based on GPU memory
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM_MB" -ge 80000 ]; then
    BATCH_SIZE="${BATCH_SIZE:-32}"
    echo "▸ 80GB+ VRAM detected → batch_size=$BATCH_SIZE"
elif [ "$GPU_MEM_MB" -ge 40000 ]; then
    BATCH_SIZE="${BATCH_SIZE:-16}"
    echo "▸ 40GB+ VRAM detected → batch_size=$BATCH_SIZE"
else
    BATCH_SIZE="${BATCH_SIZE:-8}"
    echo "▸ <40GB VRAM detected → batch_size=$BATCH_SIZE"
fi

echo "▸ Epochs: $EPOCHS"
echo "▸ Batch size: $BATCH_SIZE"
echo "▸ Learning rate: $LR"
echo "▸ Workers: $NUM_WORKERS"
echo "▸ W&B project: $WANDB_PROJECT"
echo ""

# ── Launch training ─────────────────────────────────────────────
# Using nohup so training survives SSH disconnect.
# Output goes to both terminal and train_cloud.log.
TRAIN_CMD="uv run python train.py \
    --device cuda \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --num-workers $NUM_WORKERS \
    --wandb-project $WANDB_PROJECT \
    $RESUME_ARGS \
    $EXTRA_ARGS"

echo "▸ Command: $TRAIN_CMD"
echo ""
echo "═══════════════════════════════════════════"
echo "Training started. Logs: train_cloud.log"
echo "To detach: Ctrl+C (training continues in background)"
echo "To monitor: tail -f train_cloud.log"
echo "═══════════════════════════════════════════"
echo ""

nohup $TRAIN_CMD 2>&1 | tee train_cloud.log
