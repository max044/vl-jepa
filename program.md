# VL-JEPA AutoResearch

This is an autonomous experimentation system for VL-JEPA hyperparameter tuning on temporal moment retrieval (Charades-STA).

## Goal

Find optimal hyperparameters for VL-JEPA training to maximize temporal moment retrieval performance (R@1 IoU metrics).

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar19`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context
   - `prepare.py` — fixed constants, data prep, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop, hyperparameters.
   - `vljepa/config.py` — Config dataclass (can read, but modify train.py directly)
4. **Verify data exists**: Check that `data/autoresearch/` contains annotations and videos. If not, run `uv run prepare.py --subset 500`.
5. **Initialize results.tsv**: Create `results.tsv` with header row:
   ```
   commit	val_loss	memory_gb	status	description
   ```
6. **Confirm and go**: Confirm setup looks good.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/model loading).

**Launch**: Simply run `uv run train.py > run.log 2>&1`

**What you CAN do:**
- Modify `train.py` — this is the only file you edit
- Change: model architecture (predictor_layers, bidirectional attention, etc.), optimizer settings, learning rates, batch size, loss weights (temperature, sigreg_weight), warmup steps, etc.

**What you CANNOT do:**
- Modify `prepare.py` — it contains fixed evaluation, data loading constants
- Install new packages beyond what's in `pyproject.toml`
- Modify the evaluation harness — use the validation loss from the training loop

**The goal**: Minimize `val_loss` (validation loss) after 5 minutes of training.

**VRAM constraint**: Stay under 48GB (RTX 6000 Ada). Monitor with `nvidia-smi`.

**Simplicity criterion**: Prefer simple solutions. A small improvement with ugly complexity is not worth it. Removing code and getting equal results is a great outcome.

**The first run**: Always establish baseline by running training as-is.

## Output Format

After training, check the log for:

```
Val loss: 0.4582
```

Or search: `grep "Val loss:" run.log`

The script should print validation metrics at the end of each epoch or validation phase.

## Logging Results

When done, log to `results.tsv` (tab-separated, NOT comma-separated):

```
commit	val_loss	memory_gb	status	description
```

Columns:
1. git commit hash (short, 7 chars)
2. val_loss achieved — use 999.999 for crashes
3. peak memory in GB, round to .1f — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what was tried

Example:
```
commit	val_loss	memory_gb	status	description
a1b2c3d	0.4582	44.0	keep	baseline
b2c3d4e	0.4231	44.2	keep	increase LR to 3e-4
c3d4e5f	0.5123	44.0	discard	switch to ReLU activation
d4e5f6g	999.999	0.0	crash	double batch size (OOM)
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar19`).

LOOP FOREVER:

1. Look at git state: current branch/commit
2. Modify `train.py` with an experimental idea
3. `git commit -am "exp: description"`
4. Run: `uv run train.py > run.log 2>&1` (redirect everything)
5. Read results: `grep "Val loss:" run.log` or check W&B
6. If output is empty/crashed: `tail -n 50 run.log` to diagnose
7. Record in results.tsv (do NOT commit results.tsv)
8. **If val_loss improved (lower)**: `keep` — advance the branch
9. **If val_loss equal or worse**: `discard` — `git reset --hard HEAD~1`

**Timeout**: Each experiment ~5 minutes training + startup overhead. If >10 min total, kill and treat as failure.

**Crashes**: Use judgment. Easy fixes (typos): fix and retry. Fundamental issues: log "crash" and move on.

**NEVER STOP**: Once loop begins, do NOT pause to ask human. Continue autonomously until manually stopped. If out of ideas: re-read files, try combining previous near-misses, try more radical changes. Loop runs until interrupted.

Example: ~12 experiments/hour = ~100 experiments during human sleep.

## Key Hyperparameters to Explore

1. **Learning Rate** (default: 1e-4) — [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
2. **Batch Size** (default: 2) — [2, 4, 8] (limited by VRAM)
3. **Temperature** (default: 0.07) — [0.03, 0.05, 0.07, 0.1, 0.15]
4. **SIGReg Weight** (default: 0.1) — [0.0, 0.05, 0.1, 0.2, 0.5]
5. **Warmup Steps** (default: 100) — [50, 100, 200, 500]
6. **Y-Encoder LR Multiplier** (default: 0.05) — [0.01, 0.05, 0.1]
7. **Predictor Layers** (default: 0=all) — [0, 4, 8, 12]
8. **Weight Decay** (default: 0.05) — [0.0, 0.01, 0.05, 0.1]

## Architecture Notes

- **X-Encoder**: V-JEPA 2 ViT-L (frozen, ~300M params) — DO NOT MODIFY
- **Predictor**: Qwen3.5-0.8B (trainable) — can modify layers, attention type
- **Y-Encoder**: Qwen3-Embedding-0.6B (trainable) — can modify LR multiplier
- **Loss**: InfoNCE + SIGReg — can modify weights

## Tips

- Monitor VRAM: `watch -n 1 nvidia-smi`
- Check W&B for live metrics: https://wandb.ai/maxence-cabiddu-maxence-cabiddu/vl-jepa
- If OOM: reduce batch_size or predictor_layers
- If training unstable: lower LR or increase warmup
- If no improvement: try different temperature or sigreg_weight
