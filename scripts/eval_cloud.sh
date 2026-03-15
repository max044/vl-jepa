#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# VL-JEPA Cloud Evaluation Launcher
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Load environment ────────────────────────────────────────────
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ── Configuration ───────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-checkpoints/best.pth}"
WANDB_PROJECT="${WANDB_PROJECT:-vl-jepa}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "╔══════════════════════════════════════════╗"
echo "║       VL-JEPA Cloud Evaluation           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check if checkpoint exists locally or if it looks like a W&B artifact (contains : or /)
if [[ ! -f "$CHECKPOINT" ]] && [[ "$CHECKPOINT" != *":"* ]] && [[ "$CHECKPOINT" != *"/"* ]]; then
    echo "✗ Checkpoint not found: $CHECKPOINT"
    echo "  Please provide a valid local path or W&B Artifact (e.g., max044/vl-jepa/model-xyz:best) via CHECKPOINT="
    exit 1
fi

echo "▸ Evaluating checkpoint: $CHECKPOINT"
echo "▸ W&B project: $WANDB_PROJECT"
echo ""

# ── Launch evaluation ───────────────────────────────────────────
EVAL_CMD="uv run python eval.py \
    --device cuda \
    --checkpoint $CHECKPOINT \
    --wandb-project $WANDB_PROJECT \
    $EXTRA_ARGS"

echo "▸ Command: $EVAL_CMD"
echo ""
echo "═══════════════════════════════════════════"
echo "Evaluation started. Logs: eval_cloud.log"
echo "═══════════════════════════════════════════"
echo ""

nohup $EVAL_CMD 2>&1 | tee eval_cloud.log
