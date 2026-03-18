#!/bin/bash
# Run Auto-Research experiments on cloud instance
# Usage: bash scripts/run_autoresearch.sh [NUM_EXPERIMENTS]

set -e

NUM_EXP=${1:-15}
WORKDIR=~/vl-jepa

cd $WORKDIR

# Pull latest code
echo "=== Pulling latest code ==="
git pull origin main

# Load environment
if [ -f .env ]; then
    echo "=== Loading environment ==="
    export $(grep -v '^#' .env | xargs)
fi

# Authenticate HF
echo "=== Authenticating Hugging Face ==="
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=True)" 2>/dev/null || echo "HF auth skipped"

# Create results directory
mkdir -p autoresearch/results

# Run experiments
echo "=== Running $NUM_EXP experiments ==="

EXPERIMENTS=$(cat autoresearch/experiments_v2.json | python3 -c "
import json, sys
exps = json.load(sys.stdin)
for i, e in enumerate(exps[:int('$NUM_EXP')]):
    args = []
    for k, v in e.items():
        if k in ['exp_id', 'description']:
            continue
        k_cli = k.replace('_', '-')
        if isinstance(v, bool):
            if v:
                args.append(f'--{k_cli}')
        else:
            args.append(f'--{k_cli} {v}')
    print(f'{i+1}|{\" \".join(args)}')
")

IFS='|'
echo "$EXPERIMENTS" | while read -r idx args; do
    echo ""
    echo "========================================"
    echo "[$idx/$NUM_EXP] Running experiment"
    echo "Args: $args"
    echo "========================================"
    
    # Run training
    python3 train.py $args --epochs 1 --debug --debug-samples 50 --no-wandb 2>&1 | tee autoresearch/results/exp_${idx}.log
    
    echo "[$idx/$NUM_EXP] Done"
done

echo ""
echo "=== All experiments completed ==="
echo "Results saved to: $WORKDIR/autoresearch/results/"

# Parse results
echo ""
echo "=== Summary ==="
python3 << 'PYEOF'
import os
import re
from pathlib import Path

results_dir = Path("autoresearch/results")
logs = sorted(results_dir.glob("exp_*.log"))

print(f"{'Exp':<6} {'Val Loss':<12} {'InfoNCE':<12} {'Status':<10}")
print("-" * 50)

for log in logs:
    exp_id = log.stem.split("_")[1]
    content = log.read_text()
    
    # Find validation loss
    val_loss = "N/A"
    infonce = "N/A"
    
    for line in content.splitlines():
        if "Val loss:" in line:
            m = re.search(r"Val loss:\s*([0-9.]+)", line)
            if m:
                val_loss = m.group(1)
        if "InfoNCE:" in line and "loss/infonce" in line:
            m = re.search(r"loss/infonce.*?([0-9.]+)", line)
            if m:
                infonce = m.group(1)
    
    status = "OK" if val_loss != "N/A" else "FAIL"
    print(f"{exp_id:<6} {val_loss:<12} {infonce:<12} {status:<10}")
PYEOF