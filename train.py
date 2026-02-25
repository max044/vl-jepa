"""VL-JEPA Training Script.

Train the predictor to align video+query embeddings with target text embeddings
using bidirectional InfoNCE loss on Charades-STA.

Usage:
    python train.py                          # Default settings
    python train.py --debug --epochs 2       # Quick sanity check
    python train.py --device cuda --epochs 20 --batch-size 16  # Full training on GPU
    python train.py --device cuda --wandb-project vl-jepa      # With W&B tracking
"""

import argparse
from dataclasses import asdict
import os
import time

import torch
from torch.utils.data import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from vljepa.config import Config
from vljepa.dataset import CharadesSTADataset, collate_fn
from vljepa.models import VLJepa
from vljepa.losses import vl_jepa_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train VL-JEPA")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num-workers", type=int, default=None)
    # W&B arguments
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="vl-jepa", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B team/entity")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-id", type=str, default=None, help="W&B run ID to resume")
    return parser.parse_args()


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Cosine schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, scheduler, config, epoch, scaler=None):
    """Run one training epoch."""
    model.predictor.train()
    model.y_encoder.projection.train()

    total_loss = 0.0
    total_infonce = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue

        # Preprocess frames using CLIP processor
        pixel_values = model.x_encoder.preprocess_frames(
            batch["frames"], device=config.device
        )

        # Tokenize queries
        query_tokens = model.query_encoder.tokenize(
            batch["queries"], device=config.device
        )

        optimizer.zero_grad()

        # Use autocast for mixed precision (if available)
        use_amp = config.device == "cuda" and scaler is not None
        with torch.autocast(device_type="cuda", enabled=use_amp):
            sy_hat, sy = model(
                pixel_values,
                query_tokens["input_ids"],
                query_tokens["attention_mask"],
                batch["captions"],
            )

            loss, metrics = vl_jepa_loss(
                sy_hat, sy,
                temperature=config.temperature,
                sigreg_weight=config.sigreg_weight,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), config.grad_clip)
            optimizer.step()

        scheduler.step()

        total_loss += metrics["loss/total"]
        total_infonce += metrics["loss/infonce"]
        num_batches += 1

        # ── W&B: log per-step metrics ───────────────────────
        if HAS_WANDB and wandb.run:
            wandb.log({
                "train/loss": metrics["loss/total"],
                "train/infonce": metrics["loss/infonce"],
                "train/lr": optimizer.param_groups[0]["lr"],
            })

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            print(
                f"  [{batch_idx+1}/{len(dataloader)}] "
                f"loss={metrics['loss/total']:.4f} "
                f"infonce={metrics['loss/infonce']:.4f} "
                f"lr={lr:.2e} "
                f"({elapsed:.1f}s)"
            )

    avg_loss = total_loss / max(num_batches, 1)
    avg_infonce = total_infonce / max(num_batches, 1)
    elapsed = time.time() - start_time

    return {
        "avg_loss": avg_loss,
        "avg_infonce": avg_infonce,
        "elapsed": elapsed,
        "num_batches": num_batches,
    }


@torch.no_grad()
def validate_one_epoch(model, dataloader, config):
    """Run one validation epoch."""
    model.predictor.eval()
    model.y_encoder.projection.eval()

    total_loss = 0.0
    total_infonce = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue

        # Preprocess frames using CLIP processor
        pixel_values = model.x_encoder.preprocess_frames(
            batch["frames"], device=config.device
        )

        # Tokenize queries
        query_tokens = model.query_encoder.tokenize(
            batch["queries"], device=config.device
        )

        with torch.autocast(device_type="cuda", enabled=config.device == "cuda"):
            sy_hat, sy = model(
                pixel_values,
                query_tokens["input_ids"],
                query_tokens["attention_mask"],
                batch["captions"],
            )

            loss, metrics = vl_jepa_loss(
                sy_hat, sy,
                temperature=config.temperature,
                sigreg_weight=config.sigreg_weight,
            )

        total_loss += metrics["loss/total"]
        total_infonce += metrics["loss/infonce"]
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_infonce = total_infonce / max(num_batches, 1)
    elapsed = time.time() - start_time

    return {
        "avg_loss": avg_loss,
        "avg_infonce": avg_infonce,
        "elapsed": elapsed,
        "num_batches": num_batches,
    }


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save a training checkpoint."""
    torch.save({
        "epoch": epoch,
        "predictor_state_dict": model.predictor.state_dict(),
        "y_projection_state_dict": model.y_encoder.projection.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def log_artifact(path, name, artifact_type, metadata=None):
    """Upload a checkpoint as a W&B Artifact for model versioning."""
    if not (HAS_WANDB and wandb.run):
        return
    artifact = wandb.Artifact(
        name=name,
        type=artifact_type,
        metadata=metadata or {},
    )
    artifact.add_file(path)
    wandb.log_artifact(artifact)


def load_checkpoint(model, optimizer, path, device):
    """Load checkpoint and return start epoch."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.predictor.load_state_dict(ckpt["predictor_state_dict"])
    model.y_encoder.projection.load_state_dict(ckpt["y_projection_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Resumed from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")
    return ckpt["epoch"] + 1


def main():
    args = parse_args()

    # Config
    config = Config()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.device is not None:
        config.device = args.device
    if args.debug:
        config.debug = True
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    # ── W&B Init ────────────────────────────────────────────
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb_kwargs = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_run_name,
            "config": asdict(config),
            "tags": ["train", config.device] + (["debug"] if config.debug else []),
        }
        
        # Handle Resume Mode
        if args.wandb_id:
            wandb_kwargs["id"] = args.wandb_id
            wandb_kwargs["resume"] = "must"
            print(f"🔄 Attempting to resume W&B run: {args.wandb_id}")
            
        wandb.init(**wandb_kwargs)
        print(f"W&B run: {wandb.run.url}")
    elif not HAS_WANDB and not args.no_wandb:
        print("Warning: wandb not installed. Install with `pip install wandb` for experiment tracking.")

    print(f"Device: {config.device}")
    print(f"Debug: {config.debug}")
    print(f"Epochs: {config.epochs}, Batch size: {config.batch_size}, LR: {config.lr}")
    print()

    # Model
    print("Loading models...")
    model = VLJepa(config)
    model.to(config.device)  # Move entire model (including frozen encoders)

    # Print parameter counts
    param_counts = model.count_parameters()
    total_trainable = sum(v["trainable"] for v in param_counts.values())
    print(f"\nParameter counts:")
    for name, counts in param_counts.items():
        print(f"  {name}: {counts['total']:,} total, {counts['trainable']:,} trainable")
    print(f"  TOTAL trainable: {total_trainable:,}\n")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Dataset
    print("Loading dataset...")
    train_dataset = CharadesSTADataset(
        config.anno_train, config.videos_dir, config, split="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=config.device == "cuda",
    )

    # Validation Dataset
    print("Loading test/val dataset...")
    val_dataset = CharadesSTADataset(
        config.anno_test, config.videos_dir, config, split="test"
    )
    if config.val_samples and not config.debug:
        val_dataset.samples = val_dataset.samples[:config.val_samples]

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=config.device == "cuda",
    )

    # Scheduler
    total_steps = len(train_loader) * config.epochs
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)

    # Optional: resume
    start_epoch = 0
    checkpoint_path = args.checkpoint
    
    if checkpoint_path:
        # Check if it's a W&B artifact path (contains / or :)
        if (":" in checkpoint_path or "/" in checkpoint_path) and not os.path.exists(checkpoint_path):
            if use_wandb:
                print(f"📥 Downloading checkpoint from W&B Artifact: {checkpoint_path}")
                try:
                    artifact = wandb.run.use_artifact(checkpoint_path, type='model')
                    artifact_dir = artifact.download()
                    # Find the .pth file in the artifact
                    checkpoint_path = os.path.join(artifact_dir, "best.pth")
                    if not os.path.exists(checkpoint_path):
                        # Try last.pth or any .pth
                        pths = [f for f in os.listdir(artifact_dir) if f.endswith(".pth")]
                        if pths:
                            checkpoint_path = os.path.join(artifact_dir, pths[0])
                except Exception as e:
                    print(f"❌ Failed to download artifact: {e}")
                    checkpoint_path = None
            else:
                print("⚠ W&B is disabled, cannot download artifact.")
                checkpoint_path = None

        if checkpoint_path and os.path.exists(checkpoint_path):
            start_epoch = load_checkpoint(model, optimizer, checkpoint_path, config.device)
        else:
            print(f"⚠ Could not find checkpoint: {args.checkpoint}. Starting from scratch.")

    # Mixed precision scaler (CUDA only)
    scaler = torch.amp.GradScaler() if config.device == "cuda" else None

    # Training loop
    best_loss = float("inf")
    print(f"Starting training ({len(train_loader)} batches/epoch)...\n")

    global_step = 0

    for epoch in range(start_epoch, config.epochs):
        print(f"═══ Epoch {epoch+1}/{config.epochs} ═══")

        # Ensure frozen parts are in eval mode
        model.eval()
        
        result = train_one_epoch(
            model, train_loader, optimizer, scheduler, config, epoch, scaler
        )

        global_step += result["num_batches"]

        print(
            f"  → Avg loss: {result['avg_loss']:.4f} | "
            f"InfoNCE: {result['avg_infonce']:.4f} | "
            f"Time: {result['elapsed']:.1f}s | "
            f"Batches: {result['num_batches']}"
        )

        # ── W&B: log epoch metrics ──────────────────────────
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "epoch/avg_loss": result["avg_loss"],
                "epoch/avg_infonce": result["avg_infonce"],
                "epoch/time_s": result["elapsed"],
                "epoch/lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

        # ── Validation Phase ────────────────────────────────
        val_result = None
        if (epoch + 1) % config.val_every == 0 or epoch == config.epochs - 1:
            print(f"  🔍 Validating...")
            val_result = validate_one_epoch(model, val_loader, config)
            print(
                f"  → Val loss: {val_result['avg_loss']:.4f} | "
                f"Val InfoNCE: {val_result['avg_infonce']:.4f}"
            )
            
            if use_wandb:
                wandb.log({
                    "val/loss": val_result["avg_loss"],
                    "val/infonce": val_result["avg_infonce"],
                }, step=global_step)

        # Save checkpoints
        if (epoch + 1) % config.save_every == 0 or epoch == config.epochs - 1:
            ckpt_path = os.path.join(config.checkpoint_dir, "last.pth")
            save_checkpoint(model, optimizer, epoch, result["avg_loss"], ckpt_path)
            print(f"  💾 Saved checkpoint: {ckpt_path}")
            log_artifact(ckpt_path, "vl-jepa-last", "model", {
                "epoch": epoch + 1, "loss": result["avg_loss"]
            })

        # Update best based on validation if available
        if val_result:
            current_best_metric = val_result["avg_loss"]
            metric_name = "val/loss"
        else:
            current_best_metric = result["avg_loss"]
            metric_name = "train/loss"

        if current_best_metric < best_loss:
            best_loss = current_best_metric
            best_path = os.path.join(config.checkpoint_dir, "best.pth")
            save_checkpoint(model, optimizer, epoch, current_best_metric, best_path)
            print(f"  ⭐ New best! ({metric_name}) Saved: {best_path}")
            log_artifact(best_path, "vl-jepa-best", "model", {
                "epoch": epoch + 1, metric_name: best_loss
            })

        print()

    print(f"Training complete! Best loss: {best_loss:.4f}")

    # ── W&B: finalize ──────────────────────────────────────
    if use_wandb:
        wandb.summary["best_loss"] = best_loss
        wandb.summary["total_epochs"] = config.epochs
        wandb.finish()


if __name__ == "__main__":
    main()
