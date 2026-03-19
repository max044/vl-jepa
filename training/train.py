"""
VL-JEPA Training Script - Full Training
Entraînement complet avec tous les hyperparamètres optimisés.

Usage: uv run training/train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from dotenv import load_dotenv
load_dotenv()

from vljepa.config import Config
from vljepa.dataset import CharadesSTADataset, collate_fn
from vljepa.models import VLJepa
from vljepa.losses import vl_jepa_loss, SIGReg

DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Hyperparameters - FULL TRAINING
# Ces paramètres viennent des expériences autoresearch
# ---------------------------------------------------------------------------

# Training
BATCH_SIZE = 2
GRAD_ACCUMULATION = 2  # Effective batch = 4
LEARNING_RATE = 1e-4  # À ajuster selon résultats autoresearch
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.05
MAX_EPOCHS = 20  # Entraînement complet

# Loss
TEMPERATURE = 0.07  # À ajuster selon résultats autoresearch
SIGREG_WEIGHT = 0.1  # À ajuster selon résultats autoresearch

# Model
PREDICTOR_LAYERS = 0  # 0 = use all layers
USE_BIDIRECTIONAL_ATTENTION = True
Y_ENCODER_LR_MULTIPLIER = 0.05

# Data
NUM_WORKERS = 4
MAX_TRAIN_SAMPLES = 0  # 0 = use all data

# W&B
USE_WANDB = True
WANDB_PROJECT = "vl-jepa"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Config
config = Config(
    batch_size=BATCH_SIZE,
    grad_accumulation=GRAD_ACCUMULATION,
    lr=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    temperature=TEMPERATURE,
    sigreg_weight=SIGREG_WEIGHT,
    predictor_layers=PREDICTOR_LAYERS,
    use_bidirectional_attention=USE_BIDIRECTIONAL_ATTENTION,
    y_encoder_lr_multiplier=Y_ENCODER_LR_MULTIPLIER,
    device=str(device),
)

print(f"Config: {asdict(config)}")

# ---------------------------------------------------------------------------
# Data loading (autoresearch subset)
# ---------------------------------------------------------------------------

train_dataset = CharadesSTADataset(
    anno_file=str(DATA_DIR / "charades_sta_train.txt"),
    videos_dir=str(DATA_DIR / "Charades_v1_480"),
    config=config,
    split="train",
)

# Limit to subset for autoresearch speed
if MAX_TRAIN_SAMPLES > 0 and len(train_dataset) > MAX_TRAIN_SAMPLES:
    print(f"Limiting to {MAX_TRAIN_SAMPLES} training samples for autoresearch")
    # Create subset indices
    indices = list(range(min(MAX_TRAIN_SAMPLES, len(train_dataset))))
    train_dataset = torch.utils.data.Subset(train_dataset, indices)

# Split for validation
val_size = min(100, int(0.1 * len(train_dataset)))
train_size = len(train_dataset) - val_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_subset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True,
)

val_loader = DataLoader(
    val_subset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True,
)

print(f"Dataset: {len(train_subset)} train, {len(val_subset)} val")
print(f"Batches: {len(train_loader)}/epoch")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

print("\nLoading models...")
model = VLJepa(config).to(device)
model.eval()  # Frozen encoders in eval mode

# Count parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def get_lr_multiplier(step):
    """Linear warmup then constant."""
    if step < config.warmup_steps:
        return step / max(config.warmup_steps, 1)
    return 1.0

# Parameter groups with different LRs
params_with_lr = []

# Predictor (full LR)
predictor_params = [p for p in model.predictor.parameters() if p.requires_grad]
if predictor_params:
    params_with_lr.append({"params": predictor_params, "lr": config.lr})

# Y-Encoder — modèle complet + projection au même LR réduit
y_encoder_params = [p for p in model.y_encoder.model.parameters() if p.requires_grad]
if y_encoder_params:
    params_with_lr.append({
        "params": y_encoder_params,
        "lr": config.lr * config.y_encoder_lr_multiplier
    })

y_proj_params = [p for p in model.y_encoder.projection.parameters() if p.requires_grad]
if y_proj_params:
    params_with_lr.append({
        "params": y_proj_params,
        "lr": config.lr * config.y_encoder_lr_multiplier
    })

total_trainable = sum(p.numel() for g in params_with_lr for p in g["params"])
print(f"Total trainable params: {total_trainable:,}")

optimizer = torch.optim.AdamW(
    params_with_lr,
    lr=config.lr,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.999),
)

# SIGReg
sigreg = SIGReg().to(device) if config.sigreg_weight > 0 else None
if sigreg:
    print(f"SIGReg: weight={config.sigreg_weight}")

# GradScaler (disabled for fp32)
scaler = None

print(f"\nFull training: {MAX_EPOCHS} epochs")
print(f"Starting training...\n")

# W&B
if USE_WANDB and HAS_WANDB:
    wandb.init(
        project=WANDB_PROJECT,
        config=asdict(config),
        reinit=True,
    )

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
best_val_loss = float('inf')
total_training_time = 0
step = 0
epoch = 0
smooth_train_loss = 0

for epoch in range(MAX_EPOCHS):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    
    # Training epoch
    model.predictor.train()
    model.y_encoder.projection.train()
    
    epoch_loss = 0.0
    epoch_infonce = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue
        
        # Preprocess
        pixel_values = model.x_encoder.preprocess_frames(
            batch["frames"], device=device
        )
        query_tokens = model.query_encoder.tokenize(
            batch["queries"], device=device
        )
        
        # Forward
        outputs = model(
            pixel_values,
            query_tokens["input_ids"],
            query_tokens["attention_mask"],
            batch["captions"],
        )
        
        # Loss - VLJepa returns sy_hat (predicted) and sy (target)
        sy_hat = outputs["sy_hat"]  # Predicted embeddings [B, D]
        sy = outputs["sy"]  # Target embeddings [B, D]

        # For InfoNCE: sy_hat vs sy
        # sy_hat should align with sy (bidirectional)
        loss, loss_dict = vl_jepa_loss(
            sy_hat, sy,
            temperature=config.temperature,
            sigreg_weight=config.sigreg_weight,
            sigreg_module=sigreg,
        )
        
        # Backward
        loss = loss / config.grad_accumulation
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.grad_accumulation == 0:
            # LR schedule
            lr_mult = get_lr_multiplier(step)
            for group in optimizer.param_groups:
                group["lr"] = group.get("initial_lr", config.lr) * lr_mult
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
        
        # Track loss
        train_loss = loss.item() * config.grad_accumulation
        epoch_loss += train_loss
        epoch_infonce += loss_dict["loss/infonce"]
        num_batches += 1
        
        # Smooth loss for display
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss
        debiased_loss = smooth_train_loss / (1 - ema_beta**(batch_idx + 1))
        
        # Progress bar
        pct_done = 100 * (epoch + 1) / MAX_EPOCHS
        
        print(f"\rstep {step:04d} ({pct_done:.1f}%) | "
              f"loss: {debiased_loss:.4f} | "
              f"epoch: {epoch+1}/{MAX_EPOCHS} | "
              f"batch: {batch_idx+1}/{len(train_loader)}    ", end="", flush=True)
    
    # End of epoch timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()
    dt = t1 - t0
    total_training_time += dt
    
    # Validation every epoch
    if epoch % 1 == 0:  # Validate every epoch
        model.eval()
        val_loss = 0.0
        val_infonce = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                
                pixel_values = model.x_encoder.preprocess_frames(
                    batch["frames"], device=device
                )
                query_tokens = model.query_encoder.tokenize(
                    batch["queries"], device=device
                )
                
                outputs = model(
                    pixel_values,
                    query_tokens["input_ids"],
                    query_tokens["attention_mask"],
                    batch["captions"],
                )
                
                sy_hat = outputs["sy_hat"]
                sy = outputs["sy"]
                
                loss, loss_dict = vl_jepa_loss(
                    sy_hat, sy,
                    temperature=config.temperature,
                    sigreg_weight=config.sigreg_weight,
                    sigreg_module=sigreg,
                )
                
                val_loss += loss.item()
                val_infonce += loss_dict["loss/infonce"]
                val_batches += 1
        
        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            best_val_loss = min(best_val_loss, avg_val_loss)
            
            print(f"\n  → Val loss: {avg_val_loss:.4f} (best: {best_val_loss:.4f})")
            
            if USE_WANDB and HAS_WANDB and wandb.run:
                wandb.log({
                    "val/loss": avg_val_loss,
                    "val/best": best_val_loss,
                    "epoch": epoch + 1,
                })
            
            # Save checkpoint if best
            if avg_val_loss <= best_val_loss:
                checkpoint_dir = Path("checkpoints")
                checkpoint_dir.mkdir(exist_ok=True)
                
                checkpoint_path = checkpoint_dir / f"best_e{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': asdict(config),
                }, checkpoint_path)
                
                print(f"  💾 Saved best checkpoint: {checkpoint_path}")
                
                # Also save to W&B
                if USE_WANDB and HAS_WANDB and wandb.run:
                    artifact = wandb.Artifact(f"model-checkpoint-e{epoch+1}", type="model")
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)

print()  # newline

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
final_val_loss = 0.0
final_val_infonce = 0.0
val_batches = 0

with torch.no_grad():
    for batch in val_loader:
        if batch is None:
            continue
        
        pixel_values = model.x_encoder.preprocess_frames(
            batch["frames"], device=device
        )
        query_tokens = model.query_encoder.tokenize(
            batch["queries"], device=device
        )
        
        outputs = model(
            pixel_values,
            query_tokens["input_ids"],
            query_tokens["attention_mask"],
            batch["captions"],
        )
        
        sy_hat = outputs["sy_hat"]
        sy = outputs["sy"]
        
        loss, loss_dict = vl_jepa_loss(
            sy_hat, sy,
            temperature=config.temperature,
            sigreg_weight=config.sigreg_weight,
            sigreg_module=sigreg,
        )

        final_val_loss += loss.item()
        final_val_infonce += loss_dict["loss/infonce"]
        val_batches += 1

if val_batches > 0:
    final_val_loss /= val_batches
    final_val_infonce /= val_batches

best_val_loss = min(best_val_loss, final_val_loss)

# Memory
peak_vram_mb = 0
if torch.cuda.is_available():
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

t_end = time.time()
startup_time = t_start_training - t_start

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("---")
print(f"val_loss:         {final_val_loss:.6f}")
print(f"best_val_loss:    {best_val_loss:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_epochs:       {epoch}")
print(f"batch_size:       {BATCH_SIZE}")
print(f"lr:               {LEARNING_RATE}")

if USE_WANDB and HAS_WANDB and wandb.run:
    wandb.finish()
