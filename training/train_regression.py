"""
VL-JEPA Regression Head Training Script
Trains only the temporal regression head on top of a frozen pre-trained model.

Usage: python training/train_regression.py --checkpoint checkpoints/best.pt
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import asdict

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from vljepa.config import Config
from vljepa.dataset import CharadesSTADataset, collate_fn
from vljepa.models import VLJepa

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Pre-trained checkpoint to load")
    parser.add_argument("--epochs", type=int, default=10, help="Number of regression epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for head")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Config & Model from Checkpoint
    print(f"📂 Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    config = Config()
    if "config" in ckpt:
        for k, v in ckpt["config"].items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    # Force regression enabled
    config.use_regression = True
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.device = str(device)
    
    model = VLJepa(config).to(device)
    
    # Load weights (relaxed strictness because regression_head is new)
    if "model_state_dict" in ckpt:
        msg = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  ✓ Loaded model weights. Missing keys (expected): {len(msg.missing_keys)}")
    
    # 2. Freeze all except Regression Head
    for p in model.parameters():
        p.requires_grad = False
        
    if hasattr(model.predictor, "regression_head"):
        for p in model.predictor.regression_head.parameters():
            p.requires_grad = True
        print("  ✓ Regression Head unfrozen for training.")
    else:
        print("❌ Error: Regression Head not found in model!")
        return

    # 3. Data Loading
    print("\n📦 Loading dataset...")
    train_dataset = CharadesSTADataset(
        anno_file=config.anno_train,
        videos_dir=config.videos_dir,
        config=config,
        split="train"
    )
    
    # Small validation subset
    val_size = min(200, int(0.1 * len(train_dataset)))
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [len(train_dataset)-val_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.predictor.regression_head.parameters(), lr=config.lr)
    l1_criterion = nn.L1Loss()
    from vljepa.models import IntervalIoULoss
    iou_criterion = IntervalIoULoss()
    
    # 5. W&B
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        name = f"regression-2phase-{time.strftime('%m%d-%H%M')}"
        wandb.init(
            project="vl-jepa",
            name=name,
            config={**asdict(config), "task": "regression_head_2phase"}
        )

    print(f"\n🚀 Starting 2-Phase Regression Training for {args.epochs} epochs...")
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        # Phase Switch: Unfreeze Predictor LoRA after 5 epochs
        if epoch == 5:
            print("\n🔥 Phase 2: Unfreezing Predictor LoRA adapters for fine-tuning...")
            if config.use_lora:
                for p in model.predictor.parameters():
                    if p.requires_grad is False: # Check if it's a LoRA param
                        # Actually PEFT marks LoRA params as requires_grad=True automatically
                        # But since we forced ALL frozen at line 64, we need to re-enable
                        pass 
                
                # Correct way for PEFT:
                for name, param in model.predictor.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True
                
                # Refresh optimizer to include new params
                trainable_params = [
                    {'params': model.predictor.regression_head.parameters(), 'lr': config.lr},
                    {'params': [p for n, p in model.predictor.named_parameters() if "lora" in n], 'lr': config.lr * 0.1}
                ]
                optimizer = torch.optim.AdamW(trainable_params)
            else:
                print("  (No LoRA detected, skipping predictor unfreezing)")

        model.train()
        model.x_encoder.eval() 
        model.y_encoder.eval()
        
        train_loss = 0
        pbar = range(len(train_loader))
        for batch_idx, batch in enumerate(train_loader):
            if batch is None: continue
            
            optimizer.zero_grad()
            pixel_values = model.x_encoder.preprocess_frames(batch["frames"], device=device)
            tokens = model.y_encoder.tokenizer(batch["queries"], padding=True, truncation=True, return_tensors="pt").to(device)
            
            with torch.no_grad():
                sv = model.x_encoder(pixel_values)
            
            results = model.predictor(sv, tokens.input_ids, tokens.attention_mask)
            pred_offsets = results.get("offsets")
            gt_offsets = torch.tensor(batch["offset_targets"], dtype=torch.float32, device=device)
            
            loss_l1 = l1_criterion(pred_offsets, gt_offsets)
            loss_iou = iou_criterion(pred_offsets, gt_offsets)
            loss = 0.5 * loss_l1 + 0.5 * loss_iou
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} (L1: {loss_l1.item():.4f}, IoU: {loss_iou.item():.4f})")
                if use_wandb:
                    wandb.log({"reg/batch_loss": loss.item(), "reg/batch_l1": loss_l1.item(), "reg/batch_iou": loss_iou.item()})

        # Validation at end of epoch
        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                pixel_values = model.x_encoder.preprocess_frames(batch["frames"], device=device)
                tokens = model.y_encoder.tokenizer(batch["queries"], padding=True, truncation=True, return_tensors="pt").to(device)
                sv = model.x_encoder(pixel_values)
                results = model.predictor(sv, tokens.input_ids, tokens.attention_mask)
                pred = results["offsets"]
                target = torch.tensor(batch["offset_targets"], dtype=torch.float32, device=device)
                
                v_l1 = l1_criterion(pred, target)
                v_iou = iou_criterion(pred, target)
                val_loss += (0.5 * v_l1 + 0.5 * v_iou).item()
                val_iou += (1.0 - v_iou).item() # Higher is better
        
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        if use_wandb:
            wandb.log({"reg/train_loss": avg_train_loss, "reg/val_loss": val_loss, "reg/val_iou": val_iou, "epoch": epoch})
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config.checkpoint_dir, "best_regression.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "val_loss": val_loss,
                "val_iou": val_iou
            }, save_path)
            print(f"  ⭐ Saved best regression head to {save_path}")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    train()
