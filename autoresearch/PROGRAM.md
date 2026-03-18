# VL-JEPA Auto-Research Program

This is an autonomous experimentation system for VL-JEPA hyperparameter tuning.

## Goal

Find optimal hyperparameters for VL-JEPA training on Charades-STA through rapid experimentation (5-10 min runs), then scale to full training.

## Context

VL-JEPA (Video-Language Joint Embedding Predictive Architecture) is a model for temporal moment retrieval:
- **X-Encoder**: Frozen V-JEPA 2 ViT-L (~300M params) - extracts video features
- **Predictor**: Qwen 2.5 0.5B with LoRA - predicts text embeddings
- **Y-Encoder**: Frozen MiniLM-L6-v2 (~22M params) - encodes target captions

## Key Hyperparameters to Explore

1. **Learning Rate** (default: 3e-4) - [1e-4, 3e-4, 1e-3, 3e-3]
2. **Batch Size** (default: 4) - [2, 4, 8, 16] (constrained by VRAM)
3. **LoRA Rank** (default: 64) - [16, 32, 64, 128]
4. **LoRA Alpha** (default: 128) - [32, 64, 128, 256]
5. **Temperature** (default: 0.07) - [0.03, 0.05, 0.07, 0.1, 0.15]
6. **SIGReg Weight** (default: 0.1) - [0.0, 0.05, 0.1, 0.2, 0.5]
7. **Warmup Steps** (default: 200) - [100, 200, 500]
8. **Weight Decay** (default: 0.01) - [0.0, 0.01, 0.001]

## Experiment Protocol

### Phase 1: Quick Screening (5 min runs)
Each experiment runs for 5 minutes or 100 steps (whichever comes first).

**Metric**: Training loss convergence rate + validation recall@1
- Lower training loss is better
- Higher recall@1 is better
- Track both, optimize for best balance

### Phase 2: Validation (10 min runs)
Take top 3 configurations from Phase 1, run for 10 min or 500 steps.

### Phase 3: Full Training
Best configuration from Phase 2 gets full 20-epoch training.

## Files

- `train.py` — main training script (modify configs/hyperparams)
- `vljepa/config.py` — Config dataclass with all hyperparameters
- `configs/base.yaml` — YAML config overrides
- `results.json` — experiment results log

## Experiment Loop

```
1. Read current results.json to see what's been tried
2. Propose a new hyperparameter configuration
3. Modify configs/base.yaml or pass CLI args to train.py
4. Run: python train.py --epochs 1 --max_steps 100
5. Extract metrics: training loss, val_recall@1, GPU memory
6. Log to results.json
7. If better than baseline, keep config; else discard
8. Repeat
```

## Selection Criteria

**Keep** if:
- Training loss decreases smoothly (no spikes/crashes)
- VRAM usage < 20GB (for RTX 4090 24GB)
- recall@1 improves over baseline

**Discard** if:
- Loss NaN or explosion
- OOM (out of memory)
- Worse metrics than baseline

## Baseline Configuration

```yaml
batch_size: 4
lr: 3e-4
weight_decay: 0.01
lora_r: 64
lora_alpha: 128
temperature: 0.07
sigreg_weight: 0.1
warmup_steps: 200
```

Expected baseline performance after 100 steps:
- Training loss: ~2.0-2.5
- Val recall@1: ~0.15-0.20
- VRAM: ~18-20GB

## Output Format

Log results to `results.json`:

```json
{
  "experiments": [
    {
      "id": "exp001",
      "timestamp": "2026-03-18T20:30:00",
      "config": { ... },
      "train_loss": 2.1,
      "val_recall@1": 0.18,
      "val_mIoU": 0.22,
      "peak_vram_gb": 19.2,
      "steps": 100,
      "status": "keep"
    }
  ]
}
```

## Cloud Execution

Run on Vast.ai RTX 4090 (~$0.30-0.50/hour):
```bash
python scripts/cloud_autoresearch.py --gpu rtx4090 --budget 5
```

This will:
1. Launch instance with RTX 4090
2. Download data from HF Storage bucket
3. Run experiments for specified budget (hours)
4. Collect results
5. Terminate instance

## Never Stop

Once started, continue experimenting autonomously. Try different combinations, explore the hyperparameter space systematically. The goal is to find the Pareto frontier of (speed, accuracy, memory).
