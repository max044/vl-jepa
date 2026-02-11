"""VL-JEPA Training Script.

Train the predictor to align video+query embeddings with target text embeddings
using bidirectional InfoNCE loss on Charades-STA.

Usage:
    python train.py                          # Default settings
    python train.py --debug --epochs 2       # Quick sanity check
    python train.py --device cuda --epochs 20 --batch-size 16  # Full training on GPU
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

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


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save a training checkpoint."""
    torch.save({
        "epoch": epoch,
        "predictor_state_dict": model.predictor.state_dict(),
        "y_projection_state_dict": model.y_encoder.projection.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


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

    # Scheduler
    total_steps = len(train_loader) * config.epochs
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)

    # Optional: resume
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint, config.device)

    # Mixed precision scaler (CUDA only)
    scaler = torch.amp.GradScaler() if config.device == "cuda" else None

    # Training loop
    best_loss = float("inf")
    print(f"Starting training ({len(train_loader)} batches/epoch)...\n")

    for epoch in range(start_epoch, config.epochs):
        print(f"═══ Epoch {epoch+1}/{config.epochs} ═══")

        # Ensure frozen parts are in eval mode
        model.eval()
        
        result = train_one_epoch(
            model, train_loader, optimizer, scheduler, config, epoch, scaler
        )

        print(
            f"  → Avg loss: {result['avg_loss']:.4f} | "
            f"InfoNCE: {result['avg_infonce']:.4f} | "
            f"Time: {result['elapsed']:.1f}s | "
            f"Batches: {result['num_batches']}"
        )

        # Save checkpoints
        if (epoch + 1) % config.save_every == 0 or epoch == config.epochs - 1:
            ckpt_path = os.path.join(config.checkpoint_dir, "last.pth")
            save_checkpoint(model, optimizer, epoch, result["avg_loss"], ckpt_path)
            print(f"  💾 Saved checkpoint: {ckpt_path}")

        if result["avg_loss"] < best_loss:
            best_loss = result["avg_loss"]
            best_path = os.path.join(config.checkpoint_dir, "best.pth")
            save_checkpoint(model, optimizer, epoch, result["avg_loss"], best_path)
            print(f"  ⭐ New best! Saved: {best_path}")

        print()

    print(f"Training complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
