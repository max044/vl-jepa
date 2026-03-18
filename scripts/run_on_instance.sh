#!/bin/bash
# Launch auto-research on existing instance

INSTANCE_ID=$1
NUM_EXPERIMENTS=${2:-15}

if [ -z "$INSTANCE_ID" ]; then
    echo "Usage: $0 <instance_id> [num_experiments]"
    echo ""
    echo "Available instances:"
    uv run vastai show instances 2>&1
    exit 1
fi

export VASTAI_API_KEY=$(grep VASTAI_API_KEY .env | cut -d= -f2)

echo "=================================="
echo "VL-JEPA Auto-Research Launcher"
echo "=================================="
echo "Instance: $INSTANCE_ID"
echo "Experiments: $NUM_EXPERIMENTS"
echo ""

# Wait for instance to be ready
echo "Waiting for instance to be ready..."
while true; do
    STATUS=$(uv run vastai show instances 2>&1 | grep "^$INSTANCE_ID" | awk '{print $3}')
    echo "  Status: $STATUS"
    
    if [ "$STATUS" = "running" ]; then
        break
    fi
    
    sleep 10
done

echo ""
echo "✓ Instance is running!"
echo ""

# Generate experiment configs
cat > /tmp/sweep_config.json << 'EOF'
[
  {"exp_id": "baseline", "lr": 0.0003, "batch_size": 4, "lora_r": 64, "lora_alpha": 128, "temperature": 0.07, "sigreg_weight": 0.1},
  {"exp_id": "lr_high", "lr": 0.001, "batch_size": 4, "lora_r": 64, "lora_alpha": 128, "temperature": 0.07, "sigreg_weight": 0.1},
  {"exp_id": "lr_low", "lr": 0.0001, "batch_size": 4, "lora_r": 64, "lora_alpha": 128, "temperature": 0.07, "sigreg_weight": 0.1},
  {"exp_id": "bs_8", "lr": 0.0003, "batch_size": 8, "lora_r": 64, "lora_alpha": 128, "temperature": 0.07, "sigreg_weight": 0.1},
  {"exp_id": "bs_2", "lr": 0.0003, "batch_size": 2, "lora_r": 64, "lora_alpha": 128, "temperature": 0.07, "sigreg_weight": 0.1},
  {"exp_id": "lora_32", "lr": 0.0003, "batch_size": 4, "lora_r": 32, "lora_alpha": 64, "temperature": 0.07, "sigreg_weight": 0.1},
  {"exp_id": "lora_128", "lr": 0.0003, "batch_size": 4, "lora_r": 128, "lora_alpha": 256, "temperature": 0.07, "sigreg_weight": 0.1},
  {"exp_id": "temp_005", "lr": 0.0003, "batch_size": 4, "lora_r": 64, "lora_alpha": 128, "temperature": 0.05, "sigreg_weight": 0.1},
  {"exp_id": "temp_01", "lr": 0.0003, "batch_size": 4, "lora_r": 64, "lora_alpha": 128, "temperature": 0.1, "sigreg_weight": 0.1},
  {"exp_id": "sigreg_0", "lr": 0.0003, "batch_size": 4, "lora_r": 64, "lora_alpha": 128, "temperature": 0.07, "sigreg_weight": 0.0},
  {"exp_id": "sigreg_02", "lr": 0.0003, "batch_size": 4, "lora_r": 64, "lora_alpha": 128, "temperature": 0.07, "sigreg_weight": 0.2},
  {"exp_id": "combo1", "lr": 0.0005, "batch_size": 6, "lora_r": 64, "lora_alpha": 128, "temperature": 0.05, "sigreg_weight": 0.05},
  {"exp_id": "combo2", "lr": 0.0002, "batch_size": 8, "lora_r": 32, "lora_alpha": 64, "temperature": 0.08, "sigreg_weight": 0.15},
  {"exp_id": "combo3", "lr": 0.0004, "batch_size": 4, "lora_r": 96, "lora_alpha": 192, "temperature": 0.06, "sigreg_weight": 0.08},
  {"exp_id": "combo4", "lr": 0.0006, "batch_size": 4, "lora_r": 64, "lora_alpha": 128, "temperature": 0.07, "sigreg_weight": 0.1, "warmup_steps": 300}
]
EOF

echo "Uploading experiment config..."
uv run vastai copy /tmp/sweep_config.json $INSTANCE_ID:/root/sweep_config.json

echo ""
echo "Starting experiments..."
echo ""

# Create and run experiment script
uv run vastai execute $INSTANCE_ID "cd ~/vl-jepa && cat > run_experiments.sh << 'SCRIPT'
#!/bin/bash
set -e

mkdir -p autoresearch/results
echo 'commit\tval_loss\tinfo_nce\trecall_at_1\tstatus\tparams' > autoresearch/results.tsv

CONFIGS=/root/sweep_config.json
NUM_EXP=$1

for i in \$(seq 0 \$((NUM_EXP - 1))); do
    EXP=\$(python3 -c \"import json; print(json.dumps(json.load(open('\$CONFIGS'))[\$i]))\")
    EXP_ID=\$(echo \$EXP | python3 -c \"import sys,json; print(json.load(sys.stdin)['exp_id'])\")
    
    echo \"================================\"
    echo \"Experiment \$((i+1))/\$NUM_EXP: \$EXP_ID\"
    echo \"================================\"
    
    # Build CLI args
    CLI_ARGS=\$(echo \$EXP | python3 -c \"
import sys, json
d = json.load(sys.stdin)
args = []
for k, v in d.items():
    if k != 'exp_id':
        args.append(f'--{k} {v}')
print(' '.join(args))
\")
    
    # Run experiment
    LOG_FILE=\"autoresearch/run_\${EXP_ID}.log\"
    python train.py \$CLI_ARGS --epochs 1 --max-steps 100 --val-every 50 --num-workers 2 2>&1 | tee \$LOG_FILE || true
    
    # Parse results
    VAL_LOSS=\$(grep 'Val loss:' \$LOG_FILE | tail -1 | grep -o 'Val loss: [0-9.]*' | cut -d' ' -f3 || echo '0.0')
    INFO_NCE=\$(grep 'Val InfoNCE:' \$LOG_FILE | tail -1 | grep -o 'Val InfoNCE: [0-9.]*' | cut -d' ' -f3 || echo '0.0')
    
    if [ -n \"\$VAL_LOSS\" ] && [ \"\$VAL_LOST\" != '0.0' ]; then
        STATUS='keep'
        echo \"✓ Results: val_loss=\$VAL_LOSS, info_nce=\$INFO_NCE\"
    else
        STATUS='crash'
        VAL_LOSS='0.0'
        INFO_NCE='0.0'
        echo \"✗ Failed to parse results\"
    fi
    
    # Log results
    echo -e \"\$EXP_ID\t\$VAL_LOSS\t\$INFO_NCE\t0.0\t\$STATUS\t\$EXP\" >> autoresearch/results.tsv
done

echo \"\"
echo \"================================\"
echo \"All experiments completed!\"
echo \"Results: autoresearch/results.tsv\"
cat autoresearch/results.tsv
SCRIPT

chmod +x run_experiments.sh
bash run_experiments.sh $NUM_EXPERIMENTS
"

echo ""
echo "✓ Auto-research launched!"
echo ""
echo "To monitor:"
echo "  uv run vastai logs $INSTANCE_ID"
echo "  uv run vastai show instances"
echo ""
echo "To connect via SSH:"
echo "  uv run vastai ssh $INSTANCE_ID"
echo ""
echo "Results will be in: ~/vl-jepa/autoresearch/"
