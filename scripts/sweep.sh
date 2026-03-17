#!/bin/bash
# sweep.sh: Automate multiple evaluations or training runs.

# Example: NMS Sweep
for NMS in 0.3 0.4 0.5 0.6 0.7; do
  echo "=== Running Eval with NMS threshold: $NMS ==="
  python eval.py \
    --checkpoint checkpoints/best.pth \
    --wandb-run-name "eval_nms_${NMS}"
done

# Example: Window size comparison
# python train.py --wandb-run-name "train_regression" --use_regression=true
