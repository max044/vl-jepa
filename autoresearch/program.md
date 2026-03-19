# VL-JEPA Autoresearch - Agent Instructions

This is an autonomous experimentation system for VL-JEPA hyperparameter tuning.

## Goal

Find optimal hyperparameters for VL-JEPA training to minimize validation loss (`val_loss`). Lower is better.

## In-Scope Files (READ THESE)

1. **README.md** - Repository overview and context
2. **FILE_MANIFEST.md** - What each file does  
3. **prepare.py** - Fixed constants, data prep, evaluation harness. **DO NOT MODIFY.**
4. **train.py** - The ONLY file you edit. Model architecture, optimizer, training loop, hyperparameters.

## Setup (One-time)

Before starting experiments:

1. **Create a branch** from main:
   ```bash
   git checkout -b autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')
   # e.g., autoresearch/mar19
   ```

2. **Verify data exists**:
   ```bash
   ls data/autoresearch/Charades_v1_480/*.mp4 | head -5
   # Should show videos. If not, run: uv run prepare.py --subset 500
   ```

3. **Initialize results.tsv**:
   ```bash
   echo -e "commit\tval_loss\tmemory_gb\tstatus\tdescription" > results.tsv
   ```

## The Experiment Loop

**Each experiment runs for exactly 5 minutes of training time** (wall clock, excluding startup).

### Loop Steps:

1. **Look at git state**: Check current branch and commit

2. **Modify train.py**: Edit the hyperparameters section (lines ~35-55) or architecture
   - Learning rate, batch size, temperature, sigreg weight, etc.
   - See "Hyperparameters to Explore" below

3. **Commit your change**:
   ```bash
   git add train.py
   git commit -m "exp: brief description of what changed"
   ```

4. **Run the experiment**:
   ```bash
   bash run.sh
   # Or: uv run train.py > run.log 2>&1
   ```

5. **Extract the result**:
   ```bash
   grep "^val_loss:" run.log
   # Output: val_loss: 1.023456
   ```

6. **Record in results.tsv** (tab-separated):
   ```bash
   echo -e "$(git rev-parse --short HEAD)\t1.023456\t18.5\tkeep\tdescription" >> results.tsv
   ```

7. **Decide**:
   - **If val_loss improved** (lower than baseline): Keep the commit. Continue from here.
   - **If val_loss worse or equal**: Discard with `git reset --hard HEAD~1`, then go back to step 2.

8. **Repeat**: Go back to step 2 with a new experiment idea.

## Rules

**CAN DO:**
- Edit `train.py` - only file you modify
- Change hyperparameters (LR, batch size, temperature, etc.)
- Change model architecture (predictor layers, attention, etc.)
- Change optimizer settings
- Try any experiment that fits in 5 minutes

**CANNOT DO:**
- Modify `prepare.py` - it's read-only (fixed evaluation)
- Install new packages (use what's in pyproject.toml)
- Modify evaluation harness (use val_loss from training loop)
- Ask the human if you should continue - just keep going

**NEVER STOP**: Continue the loop autonomously. If stuck, re-read files, try combining ideas, try more radical changes. Run until manually stopped.

## Output Format

After each run, train.py prints:
```
---
val_loss:         1.023456
best_val_loss:    1.023456
training_seconds: 300.1
peak_vram_mb:     18330.4
```

Extract with: `grep "^val_loss:" run.log`

## results.tsv Format

Tab-separated values (NOT commas):
```
commit	val_loss	memory_gb	status	description
a1b2c3d	1.023456	18.5	keep	baseline
b2c3d4e	0.987654	18.7	keep	increase LR to 3e-4
c3d4e5f	1.234567	18.5	discard	bad idea
```

Columns:
1. **commit**: Short git hash (7 chars)
2. **val_loss**: Validation loss achieved (lower=better, use 999.999 for crashes)
3. **memory_gb**: Peak VRAM in GB (use 0.0 for crashes)
4. **status**: `keep` (improved), `discard` (worse), or `crash` (failed)
5. **description**: Brief description of what was tried

**Do NOT commit results.tsv** - leave it untracked.

## Hyperparameters to Explore

Located in train.py lines ~35-55. Key ones:

1. **LEARNING_RATE**: 1e-4 → try [1e-5, 3e-5, 3e-4, 1e-3]
2. **BATCH_SIZE**: 2 → try [2, 4, 8] (limited by 48GB VRAM)
3. **TEMPERATURE**: 0.07 → try [0.03, 0.05, 0.1, 0.15]
4. **SIGREG_WEIGHT**: 0.1 → try [0.0, 0.05, 0.2, 0.5]
5. **WARMUP_STEPS**: 100 → try [50, 200, 500]
6. **Y_ENCODER_LR_MULTIPLIER**: 0.05 → try [0.01, 0.1]
7. **PREDICTOR_LAYERS**: 0 (all) → try [4, 8, 12]
8. **WEIGHT_DECAY**: 0.05 → try [0.0, 0.01, 0.1]

## Baseline

Current baseline: **val_loss = 1.101749** (commit 85395c7)

Beat this number to "improve".

## Simplicity Criterion

- Small improvement with ugly complexity? **Discard it.**
- Equal results with simpler code? **Keep it.**
- Deleting code improves results? **Definitely keep.**

Prefer simple, clean solutions.

## Timeout & Crashes

- **Timeout**: Each run should take ~5 minutes training + ~2 min overhead. If >10 min total, kill it (Ctrl+C) and treat as failure.
- **Crashes**: If it crashes:
  - Easy fix (typo, obvious bug)? Fix and retry.
  - Hard to fix or fundamental issue? Log "crash" in results.tsv, reset, and move on.

## Tips

- **First run**: Always run baseline first to confirm setup works
- **Monitor VRAM**: `watch -n 1 nvidia-smi` (stay under 48GB)
- **If OOM**: Reduce BATCH_SIZE or PREDICTOR_LAYERS
- **If unstable**: Lower LEARNING_RATE or increase WARMUP_STEPS
- **Check W&B**: https://wandb.ai/maxence-cabiddu-maxence-cabiddu/vl-jepa-autoresearch
- **Data**: 500 videos (~800MB) in data/autoresearch/ (already prepared on cloud)

## Commands Reference

```bash
# Run experiment
bash run.sh

# Check result
grep "^val_loss:" run.log

# View results
cat results.tsv

# Monitor GPU
watch -n 1 nvidia-smi

# Reset after failed experiment
git reset --hard HEAD~1
```

## Workflow Summary

```
edit train.py → git commit → bash run.sh → check val_loss → record → decide → repeat
```

Keep iterating. ~12 experiments/hour. Run until manually stopped.
