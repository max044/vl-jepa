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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--no-sigreg", action="store_true", help="Disable SIGReg regularization")
args, unknown = parser.parse_known_args()

DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Hyperparameters - FULL TRAINING
# Ces paramètres viennent des expériences autoresearch
# ---------------------------------------------------------------------------

# Training - PARAMÈTRES OPTIMISÉS (AutoResearch 2026-03-20)
# Updated 2026-03-21: Use real query and batch size 16
BATCH_SIZE = 16  
GRAD_ACCUMULATION = 1  # Effective batch = 16
LEARNING_RATE = 3e-4  # Optimal trouvé
WARMUP_STEPS = 100  # Optimal trouvé
WEIGHT_DECAY = 0.05  # Optimal trouvé
MAX_EPOCHS = 20  # Entraînement complet
PATIENCE = 5  # Early stopping: arrêt si pas d'amélioration pendant N époques
VAL_FREQUENCY = 0.25  # Valider à 25%, 50%, 75%, 100% de chaque époque

# Loss
TEMPERATURE = 0.025  # Optimal trouvé
SIGREG_WEIGHT = 0.0 if args.no_sigreg else 0.05  # Optimal trouvé

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

# W&B init moved after checkpoint loading to capture resume_wandb_id

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Training state & Resume Logic
# ---------------------------------------------------------------------------

def cleanup_wandb_cache():
    """Vider le cache WandB pour libérer de l'espace disque."""
    import shutil
    import os
    # 1. Manually remove artifacts from local share (DO NOT remove staging as it breaks active wandb-core sync)
    cache_dirs = [
        Path("/root/.cache/wandb/artifacts"),
        Path("/root/.local/share/wandb/artifacts"),
    ]
    for d in cache_dirs:
        if d.exists():
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
    
    # 2. Use W&B CLI cleanup for safe maintenance (prune everything over 1GB)
    if HAS_WANDB:
        try:
            os.system("wandb artifact cache cleanup --bytes 1GB > /dev/null 2>&1")
        except:
            pass

best_val_loss = float('inf')
best_epoch = 0
epochs_no_improve = 0
resume_wandb_id = None
total_training_time = 0
step = 0
start_epoch = 0
smooth_train_loss = 0

RESUME_CHECKPOINT = Path("checkpoints") / "best.pt"
if RESUME_CHECKPOINT.exists():
    print(f"📂 Detected checkpoint: {RESUME_CHECKPOINT}")
    try:
        # Load on CPU first to avoid VRAM spike
        ckpt = torch.load(RESUME_CHECKPOINT, map_location='cpu', weights_only=False)
        
        # Load model weights
        model.load_state_dict(ckpt['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # Restore training state
        start_epoch = ckpt.get('epoch', 0)
        step = ckpt.get('step', 0)
        best_val_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', float('inf')))
        resume_wandb_id = ckpt.get('wandb_id', None)
        
        print(f"  ✓ Resumed from epoch {start_epoch}, step {step} (best val: {best_val_loss:.4f})")
    except Exception as e:
        print(f"  ⚠️  Failed to resume from checkpoint: {e}")
        print("     Starting from scratch.")

# W&B
if USE_WANDB and HAS_WANDB:
    tags = [
        "full_training",
        f"lr_{LEARNING_RATE}",
        f"temp_{TEMPERATURE}",
        f"batch_{BATCH_SIZE}",
        f"epochs_{MAX_EPOCHS}",
    ]
    
    wandb.init(
        project=WANDB_PROJECT,
        entity=os.getenv("WANDB_ENTITY", "maxence-cabiddu-maxence-cabiddu"),
        config=asdict(config),
        tags=tags,
        id=resume_wandb_id,
        resume="allow" if resume_wandb_id else None,
    )
    # Define default step metric
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")

t_start_training = time.time()

print(f"\nTraining with early stopping (patience={PATIENCE})")

for epoch in range(start_epoch, MAX_EPOCHS):
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
        
        # Log to W&B every 10 steps
        if USE_WANDB and HAS_WANDB and wandb.run and (step % 10 == 0 or step < 10):
            wandb.log({
                "global_step": step,
                "train/loss": train_loss,
                "train/infonce": loss_dict["loss/infonce"],
                "train/sigreg": loss_dict.get("loss/sigreg", 0),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch + batch_idx / len(train_loader),
            })
        
        # Smooth loss for display
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss
        debiased_loss = smooth_train_loss / (1 - ema_beta**(batch_idx + 1))
        
        # Progress bar
        pct_done = 100 * (epoch + 1) / MAX_EPOCHS
        batch_pct = 100 * (batch_idx + 1) / len(train_loader)
        
        print(f"\rstep {step:04d} ({pct_done:.1f}%) | "
              f"loss: {debiased_loss:.4f} | "
              f"epoch: {epoch+1}/{MAX_EPOCHS} | "
              f"batch: {batch_idx+1}/{len(train_loader)}    ", end="", flush=True)
        
        # Validation intermédiaire tous les VAL_FREQUENCY % de l'époque
        val_checkpoints = [VAL_FREQUENCY * (i+1) for i in range(int(1/VAL_FREQUENCY))]
        for val_pct in val_checkpoints:
            if batch_idx == int(len(train_loader) * val_pct) - 1:
                # Préserver l'espace disque
                cleanup_wandb_cache()
                
                print(f"\n\n📊 Validation à {int(val_pct*100)}% de l'époque {epoch+1}...")
                
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
                    
                    print(f"  → Val loss: {avg_val_loss:.4f} (best: {best_val_loss:.4f})")
                    
                    if USE_WANDB and HAS_WANDB and wandb.run:
                        wandb.log({
                            "global_step": step,
                            "val/loss": avg_val_loss,
                            "val/best": best_val_loss,
                            "epoch": epoch + 1 + val_pct,
                        })
                    
                    # Check if improved
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_epoch = epoch + 1
                        epochs_no_improve = 0
                        
                        # Save best checkpoint (safely)
                        checkpoint_dir = Path("checkpoints")
                        checkpoint_dir.mkdir(exist_ok=True)
                        checkpoint_path = checkpoint_dir / "best.pt"
                        temp_path = checkpoint_dir / "best.pt.tmp"
                        try:
                            torch.save({
                                'epoch': epoch + 1,
                                'step': step,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_loss': avg_val_loss,
                                'best_val_loss': best_val_loss,
                                'config': asdict(config),
                                'wandb_id': wandb.run.id if (USE_WANDB and HAS_WANDB and wandb.run) else None,
                            }, temp_path)
                            os.replace(temp_path, checkpoint_path)
                            print(f"  💾 Saved best checkpoint (atomic): epoch {epoch+1}@{int(val_pct*100)}% (val_loss: {avg_val_loss:.4f})")
                        except Exception as e:
                            print(f"  ⚠️  Failed to save checkpoint: {e}")
                        
                        if USE_WANDB and HAS_WANDB and wandb.run:
                            # Log as Artifact (recommended, versioned and easier to find)
                            # This will consume space for staging, but cleanup_wandb_cache will clear it at NEXT validation
                            artifact = wandb.Artifact(
                                f"model-{wandb.run.id}", 
                                type="model", 
                                description=f"Best model from epoch {epoch+1}@{int(val_pct*100)}%"
                            )
                            artifact.add_file(str(checkpoint_path), name="best.pt")
                            wandb.log_artifact(artifact)
                            
                            wandb.run.summary["best_epoch"] = epoch + 1
                            wandb.run.summary["best_val_loss"] = best_val_loss
                    else:
                        epochs_no_improve += 1
                        print(f"  ⚠️  No improvement for {epochs_no_improve}/{PATIENCE} epochs")
                        
                        # Early stopping check
                        if epochs_no_improve >= PATIENCE:
                            print(f"\n🛑 Early stopping triggered! Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})")
                            # Set flag to break outer loop
                            early_stop = True
                
                model.train()
                model.predictor.train()
                model.y_encoder.projection.train()
                break
        
        if 'early_stop' in locals() and early_stop:
            break
    
    if 'early_stop' in locals() and early_stop:
        break
    
    # End of epoch timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()
    dt = t1 - t0
    total_training_time += dt
    
    # Validation finale à la fin de l'époque (si pas déjà fait à 100%)
    if VAL_FREQUENCY < 1.0:
        cleanup_wandb_cache()
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
            
            print(f"\n  → Val loss: {avg_val_loss:.4f} (best: {best_val_loss:.4f})")
            
            if USE_WANDB and HAS_WANDB and wandb.run:
                wandb.log({
                    "global_step": step,
                    "val/loss": avg_val_loss,
                    "val/best": best_val_loss,
                    "epoch": epoch + 1,
                })
            
            # Check if improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                epochs_no_improve = 0
                
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'best_val_loss': best_val_loss,
                        'config': asdict(config),
                        'wandb_id': wandb.run.id if (USE_WANDB and HAS_WANDB and wandb.run) else None,
                    }, temp_path)
                    os.replace(temp_path, checkpoint_path)
                    print(f"  💾 Saved best checkpoint (atomic): epoch {epoch+1} (val_loss: {avg_val_loss:.4f})")
                except Exception as e:
                    print(f"  ⚠️  Failed to save best checkpoint: {e}")
                
                if USE_WANDB and HAS_WANDB and wandb.run:
                    # Log as Artifact (final for this epoch)
                    artifact = wandb.Artifact(
                        f"model-{wandb.run.id}", 
                        type="model", 
                        description=f"Best model from epoch {epoch+1} (final for epoch)"
                    )
                    artifact.add_file(str(checkpoint_path), name="best.pt")
                    wandb.log_artifact(artifact)
                    
                    wandb.run.summary["best_epoch"] = epoch + 1
                    wandb.run.summary["best_val_loss"] = best_val_loss
            else:
                epochs_no_improve += 1
                print(f"  ⚠️  No improvement for {epochs_no_improve}/{PATIENCE} epochs")
                
                # Early stopping check
                if epochs_no_improve >= PATIENCE:
                    print(f"\n🛑 Early stopping triggered! Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})")
                    break
            
            try:
                # Always save last checkpoint (safely)
                checkpoint_dir = Path("checkpoints")
                checkpoint_dir.mkdir(exist_ok=True)
                last_checkpoint_path = checkpoint_dir / "last.pt"
                temp_last_path = checkpoint_dir / "last.pt.tmp"
                torch.save({
                    'epoch': epoch + 1,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_val_loss,
                    'config': asdict(config),
                    'wandb_id': wandb.run.id if (USE_WANDB and HAS_WANDB and wandb.run) else None,
                }, temp_last_path)
                os.replace(temp_last_path, last_checkpoint_path)
            except Exception as e:
                print(f"  ⚠️  Failed to save last checkpoint: {e}")

print()  # newline

# ---------------------------------------------------------------------------
# Final evaluation - Validation Set
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
    
    # Log final validation metrics to W&B
    if USE_WANDB and HAS_WANDB and wandb.run:
        wandb.log({
            "global_step": step,
            "final/val_loss": final_val_loss,
            "final/val_infonce": final_val_infonce,
        })

best_val_loss = min(best_val_loss, final_val_loss)

# ---------------------------------------------------------------------------
# Load best checkpoint for final test evaluation
# ---------------------------------------------------------------------------

print(f"\n📂 Loading best checkpoint from epoch {best_epoch} for test evaluation...")
checkpoint_path = Path("checkpoints") / "best.pt"
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Loaded best model (val_loss: {checkpoint['best_val_loss']:.4f})")
else:
    print(f"  ⚠️  Best checkpoint not found, using current model")

# ---------------------------------------------------------------------------
# Final evaluation - Test Set (held-out)
# ---------------------------------------------------------------------------

print("\n🧪 Loading test set for final evaluation...")

from torch.utils.data import Subset
test_dataset = CharadesSTADataset(
    anno_file=str(DATA_DIR / "charades_sta_test.txt"),
    videos_dir=str(DATA_DIR / "Charades_v1_480"),
    config=config,
    split="test",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True,
)

print(f"Test set: {len(test_dataset)} samples")

model.eval()
test_loss = 0.0
test_infonce = 0.0
test_sigreg = 0.0
test_batches = 0

with torch.no_grad():
    for batch in test_loader:
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

        test_loss += loss.item()
        test_infonce += loss_dict["loss/infonce"]
        test_sigreg += loss_dict.get("loss/sigreg", 0)
        test_batches += 1

if test_batches > 0:
    test_loss /= test_batches
    test_infonce /= test_batches
    test_sigreg /= test_batches
    
    print(f"\n📊 FINAL TEST RESULTS:")
    print(f"  test/loss:     {test_loss:.6f}")
    print(f"  test/infonce:  {test_infonce:.6f}")
    print(f"  test/sigreg:   {test_sigreg:.6f}")
    
    # Log final test metrics to W&B
    if USE_WANDB and HAS_WANDB and wandb.run:
        wandb.log({
            "global_step": step,
            "final/test_loss": test_loss,
            "final/test_infonce": test_infonce,
            "final/test_sigreg": test_sigreg,
        })

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
print(f"FINAL VALIDATION:")
print(f"  val_loss:         {final_val_loss:.6f}")
print(f"  best_val_loss:    {best_val_loss:.6f}")
print(f"FINAL TEST:")
print(f"  test_loss:        {test_loss:.6f}")
print(f"  test_infonce:     {test_infonce:.6f}")
print(f"TRAINING INFO:")
print(f"  training_seconds: {total_training_time:.1f}")
print(f"  total_seconds:    {t_end - t_start:.1f}")
print(f"  peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"  num_steps:        {step}")
print(f"  num_epochs:       {MAX_EPOCHS}")
print(f"  batch_size:       {BATCH_SIZE}")
print(f"  lr:               {LEARNING_RATE}")

if USE_WANDB and HAS_WANDB and wandb.run:
    wandb_id = wandb.run.id
    # Final summary
    wandb.run.summary.update({
        "final/val_loss": final_val_loss,
        "final/best_val_loss": best_val_loss,
        "final/test_loss": test_loss,
        "final/test_infonce": test_infonce,
        "final/test_sigreg": test_sigreg,
        "final/training_time": total_training_time,
        "final/total_time": t_end - t_start,
        "final/peak_vram_mb": peak_vram_mb,
        "final/num_steps": step,
        "final/num_epochs": MAX_EPOCHS,
    })
    wandb.finish()
else:
    wandb_id = None

# ---------------------------------------------------------------------------
# Integration: End-to-End Evaluation via eval.py
# ---------------------------------------------------------------------------
best_ckpt_path = os.path.join(config.checkpoint_dir, "best.pt")
if os.path.exists(best_ckpt_path):
    print("\n" + "="*50)
    print("🎉 Training Complete! Initiating Final Sliding-Window Evaluation...")
    print("="*50)
    
    import subprocess
    cmd = [
        "python", "training/eval.py",
        "--checkpoint", best_ckpt_path,
    ]
    if wandb_id:
        cmd.extend(["--wandb-run-path", wandb_id])
    
    # Run eval.py synchronously
    subprocess.run(cmd)

print("\n✅ VL-JEPA Pipeline Complete.")
