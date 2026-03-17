#!/bin/bash
# eval_only.sh: Run evaluation on a checkpoint with optional overrides.
# Usage: ./scripts/eval_only.sh checkpoints/best.pth [overrides...]

CHECKPOINT=$1
shift

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 <checkpoint_path> [overrides...]"
    exit 1
fi

python eval.py --checkpoint "$CHECKPOINT" "$@"
