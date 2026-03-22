"""
VL-JEPA Training Script
Usage: uv run training/train.py

nohup uv run training/train.py > logs/train.log 2>&1 &
echo $!  # noter le PID
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from dotenv import load_dotenv
load_dotenv()

from vljepa.config import Config
from vljepa.dataset import CharadesSTADataset, collate_fn, make_video_split
from vljepa.models import VLJepa
from vljepa.losses import vl_jepa_loss

CHECKPOINT_DIR = Path("checkpoints")
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
LAST_CHECKPOINT = CHECKPOINT_DIR / "last.pt"

USE_WANDB     = True
WANDB_PROJECT = "vl-jepa"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_validation(model, loader, config):
    model.eval()
    total_loss = total_infonce = batches = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            pixel_values = model.x_encoder.preprocess_frames(batch["frames"], device=config.device)
            query_tokens = model.query_encoder.tokenize(batch["queries"], device=config.device)
            outputs      = model(pixel_values, query_tokens["input_ids"], query_tokens["attention_mask"], batch["captions"])
            loss, loss_dict = vl_jepa_loss(outputs["sy_hat"], outputs["sy"], temperature=config.temperature)
            total_loss    += loss.item()
            total_infonce += loss_dict["loss/infonce"]
            batches += 1

    if batches == 0:
        return float("inf"), float("inf")
    return total_loss / batches, total_infonce / batches


def save_checkpoint(model, optimizer, epoch, step, val_loss, config, path, is_best=False):
    Path(path).parent.mkdir(exist_ok=True)
    tmp = Path(path).with_suffix(".tmp")
    try:
        torch.save({
            "epoch": epoch, "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": val_loss if is_best else None,
            "config": asdict(config),
            "wandb_id": wandb.run.id if (USE_WANDB and HAS_WANDB and wandb.run) else None,
        }, tmp)
        os.replace(tmp, path)
        print(f"  💾 Saved {'best' if is_best else 'last'} checkpoint: epoch {epoch} (val_loss: {val_loss:.4f})")
    except Exception as e:
        print(f"  ⚠️  Failed to save checkpoint: {e}")


def log_best_to_wandb(path, epoch, val_loss, label=""):
    if not (USE_WANDB and HAS_WANDB and wandb.run):
        return
    artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model", description=f"Best model {label}".strip())
    artifact.add_file(str(path), name="best.pt")
    wandb.log_artifact(artifact)
    wandb.run.summary["best_epoch"]    = epoch
    wandb.run.summary["best_val_loss"] = val_loss


def handle_validation_result(avg_val_loss, model, optimizer, epoch, step, config, state):
    """Update state, save checkpoint if improved. Returns (early_stop, improved)."""
    improved = avg_val_loss < state["best_val_loss"]
    if improved:
        state["best_val_loss"]    = avg_val_loss
        state["best_epoch"]       = epoch
        state["epochs_no_improve"] = 0
        save_checkpoint(model, optimizer, epoch, step, avg_val_loss, config, BEST_CHECKPOINT, is_best=True)
        log_best_to_wandb(BEST_CHECKPOINT, epoch, avg_val_loss, label=f"epoch {epoch}")
        return False, True

    state["epochs_no_improve"] += 1
    return state["epochs_no_improve"] >= config.early_stopping_patience, False


def cleanup_wandb_cache():
    import shutil
    for d in [Path("/root/.cache/wandb/artifacts"), Path("/root/.local/share/wandb/artifacts")]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    if HAS_WANDB:
        os.system("wandb artifact cache cleanup --bytes 1GB > /dev/null 2>&1")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

config = Config()
print(f"Config: {asdict(config)}")

# ---------------------------------------------------------------------------
# Data — split by video_id to avoid leakage
# ---------------------------------------------------------------------------

train_dataset = CharadesSTADataset(
    anno_file=config.anno_train,
    videos_dir=config.videos_dir,
    config=config,
    split="train",
)

if config.max_train_samples > 0:
    train_dataset.samples = train_dataset.samples[: config.max_train_samples]
    print(f"Limiting to {config.max_train_samples} samples")

train_indices, val_indices = make_video_split(train_dataset, val_split=config.val_split)
train_subset = Subset(train_dataset, train_indices)
val_subset   = Subset(train_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True,
                          num_workers=config.num_workers, collate_fn=collate_fn,
                          pin_memory=config.pin_memory,
                          prefetch_factor=config.prefetch_factor,
                          persistent_workers=config.persistent_workers)
val_loader   = DataLoader(val_subset,   batch_size=config.batch_size, shuffle=False,
                          num_workers=config.num_workers, collate_fn=collate_fn,
                          pin_memory=config.pin_memory,
                          prefetch_factor=config.prefetch_factor,
                          persistent_workers=config.persistent_workers)

print(f"Dataset: {len(train_subset)} train / {len(val_subset)} val (split by video_id)")
print(f"Batches: {len(train_loader)}/epoch")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

print("\nLoading models...")
model = VLJepa(config).to(config.device)
model.x_encoder.eval()
model.y_encoder.st_modules[0].auto_model.eval()  # backbone Gemma gelé
model.predictor.train()
model.y_encoder.projection.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Parameters: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def get_lr_multiplier(step):
    if step < config.warmup_steps:
        return step / max(config.warmup_steps, 1)
    return 1.0

params_with_lr = []

# Predictor LoRA + projections (full LR)
predictor_params = [p for p in model.predictor.parameters() if p.requires_grad]
if predictor_params:
    params_with_lr.append({"params": predictor_params, "lr": config.lr})

# Y-Encoder st_modules (backbone Gemma + Dense layers) — LR × 0.05
y_backbone_params = [p for p in model.y_encoder.st_modules.parameters() if p.requires_grad]
if y_backbone_params:
    params_with_lr.append({"params": y_backbone_params, "lr": config.lr * config.y_encoder_lr_multiplier})

# Y-Encoder projection — LR × 0.05
y_proj_params = [p for p in model.y_encoder.projection.parameters() if p.requires_grad]
if y_proj_params:
    params_with_lr.append({"params": y_proj_params, "lr": config.lr * config.y_encoder_lr_multiplier})

total_trainable = sum(p.numel() for g in params_with_lr for p in g["params"])
print(f"Total trainable params: {total_trainable:,}")

optimizer = torch.optim.AdamW(params_with_lr, lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999))

for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

print(f"\nFull training: {config.epochs} epochs")

# ---------------------------------------------------------------------------
# Training state & resume
# ---------------------------------------------------------------------------

state = {"best_val_loss": float("inf"), "best_epoch": 0, "epochs_no_improve": 0}
step = start_epoch = 0
smooth_train_loss = 0.0
early_stop = False
resume_wandb_id = None

if BEST_CHECKPOINT.exists():
    print(f"📂 Resuming from {BEST_CHECKPOINT}...")
    try:
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch               = ckpt.get("epoch", 0)
        step                      = ckpt.get("step", 0)
        state["best_val_loss"]    = ckpt.get("best_val_loss") or ckpt.get("val_loss", float("inf"))
        state["best_epoch"]       = start_epoch
        resume_wandb_id           = ckpt.get("wandb_id")
        print(f"  ✓ Resumed epoch {start_epoch}, step {step} (best val: {state['best_val_loss']:.4f})")
    except Exception as e:
        print(f"  ⚠️  Resume failed ({e}), starting from scratch.")

if USE_WANDB and HAS_WANDB:
    wandb.init(
        project=WANDB_PROJECT,
        entity=os.getenv("WANDB_ENTITY", "maxence-cabiddu-maxence-cabiddu"),
        config=asdict(config),
        tags=["full_training", f"lr_{config.lr}", f"batch_{config.batch_size}"],
        id=resume_wandb_id,
        resume="allow" if resume_wandb_id else None,
    )
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")

t_start_training = time.time()
total_training_time = 0.0
print(f"\nTraining (early_stopping_patience={config.early_stopping_patience})\n")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

val_checkpoints = [config.val_frequency * (i + 1) for i in range(int(1 / config.val_frequency))]

for epoch in range(start_epoch, config.epochs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    model.predictor.train()
    model.y_encoder.projection.train()

    epoch_loss = epoch_infonce = num_batches = 0
    epoch_last_val_loss = None

    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue

        pixel_values = model.x_encoder.preprocess_frames(batch["frames"], device=config.device)
        query_tokens = model.query_encoder.tokenize(batch["queries"], device=config.device)
        outputs      = model(pixel_values, query_tokens["input_ids"], query_tokens["attention_mask"], batch["captions"])

        loss, loss_dict = vl_jepa_loss(outputs["sy_hat"], outputs["sy"], temperature=config.temperature)

        (loss / config.grad_accumulation).backward()

        if (batch_idx + 1) % config.grad_accumulation == 0:
            lr_mult = get_lr_multiplier(step)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lr_mult
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

        train_loss     = loss.item()
        epoch_loss    += train_loss
        epoch_infonce += loss_dict["loss/infonce"]
        num_batches   += 1

        if USE_WANDB and HAS_WANDB and wandb.run and (step % 10 == 0 or step < 10):
            wandb.log({
                "global_step": step,
                "train/loss": train_loss,
                "train/infonce": loss_dict["loss/infonce"],
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch + batch_idx / len(train_loader),
            })

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss
        debiased = smooth_train_loss / (1 - ema_beta ** (num_batches))
        print(f"\rstep {step:04d} | loss: {debiased:.4f} | epoch: {epoch+1}/{config.epochs} | batch: {batch_idx+1}/{len(train_loader)}    ", end="", flush=True)

        # Intermediate validation
        for val_pct in val_checkpoints:
            if batch_idx != int(len(train_loader) * val_pct) - 1:
                continue

            cleanup_wandb_cache()
            print(f"\n\n📊 Validation à {int(val_pct*100)}% de l'époque {epoch+1}...")
            avg_val_loss, _ = run_validation(model, val_loader, config)
            epoch_last_val_loss = avg_val_loss
            print(f"  → Val loss: {avg_val_loss:.4f} (best: {state['best_val_loss']:.4f})")

            if USE_WANDB and HAS_WANDB and wandb.run:
                wandb.log({"global_step": step, "val/loss": avg_val_loss, "val/best": state["best_val_loss"], "epoch": epoch + val_pct})

            early_stop, improved = handle_validation_result(avg_val_loss, model, optimizer, epoch + 1, step, config, state)
            if improved:
                print(f"  ✅ Amélioration: nouveau best = {state['best_val_loss']:.4f}")
            else:
                print(f"  ⚠️  No improvement ({state['epochs_no_improve']}/{config.early_stopping_patience})")

            if early_stop:
                print(f"\n🛑 Early stopping! Best epoch: {state['best_epoch']} (val_loss: {state['best_val_loss']:.4f})")

            model.predictor.train()
            model.y_encoder.projection.train()
            break

        if early_stop:
            break

    if early_stop:
        break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_training_time += time.time() - t0

    # End-of-epoch validation (if val_frequency < 1.0, we haven't validated at 100% yet)
    if config.val_frequency < 1.0:
        cleanup_wandb_cache()
        avg_val_loss, _ = run_validation(model, val_loader, config)
        epoch_last_val_loss = avg_val_loss
        print(f"\n  → Val loss: {avg_val_loss:.4f} (best: {state['best_val_loss']:.4f})")

        if USE_WANDB and HAS_WANDB and wandb.run:
            wandb.log({"global_step": step, "val/loss": avg_val_loss, "val/best": state["best_val_loss"], "epoch": epoch + 1})

        early_stop, improved = handle_validation_result(avg_val_loss, model, optimizer, epoch + 1, step, config, state)
        if improved:
            print(f"  ✅ Amélioration: nouveau best = {state['best_val_loss']:.4f}")
        else:
            print(f"  ⚠️  No improvement ({state['epochs_no_improve']}/{config.early_stopping_patience})")

        if early_stop:
            print(f"\n🛑 Early stopping! Best epoch: {state['best_epoch']} (val_loss: {state['best_val_loss']:.4f})")
            break

    if epoch_last_val_loss is not None:
        save_checkpoint(model, optimizer, epoch + 1, step, epoch_last_val_loss, config, LAST_CHECKPOINT)

# ---------------------------------------------------------------------------
# Final validation & summary
# ---------------------------------------------------------------------------

print()
final_val_loss, final_val_infonce = run_validation(model, val_loader, config)

if USE_WANDB and HAS_WANDB and wandb.run:
    wandb.log({"global_step": step, "final/val_loss": final_val_loss, "final/val_infonce": final_val_infonce})

if final_val_loss < state["best_val_loss"]:
    state["best_val_loss"] = final_val_loss

peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
t_end = time.time()

print("---")
print(f"FINAL VALIDATION:  val_loss={final_val_loss:.6f}  best={state['best_val_loss']:.6f}")
print(f"TRAINING INFO:     steps={step}  epochs={config.epochs}  time={total_training_time:.0f}s  vram={peak_vram_mb:.0f}MB")

if USE_WANDB and HAS_WANDB and wandb.run:
    wandb.run.summary.update({
        "final/val_loss": final_val_loss,
        "final/best_val_loss": state["best_val_loss"],
        "final/training_time": total_training_time,
        "final/total_time": t_end - t_start,
        "final/peak_vram_mb": peak_vram_mb,
        "final/num_steps": step,
        "final/num_epochs": config.epochs,
    })
    wandb.finish()