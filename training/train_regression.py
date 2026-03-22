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
    criterion = nn.MSELoss()
    
    # 5. W&B
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="vl-jepa",
            name=f"regression-head-only-{time.strftime('%m%d-%H%M')}",
            config={**asdict(config), "task": "regression_head_finetune"}
        )

    print(f"\n🚀 Starting Regression Training for {args.epochs} epochs...")
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        model.train()
        # Ensure encoders stay in eval mode (BN/Dropout)
        model.x_encoder.eval() 
        model.y_encoder.eval()
        
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch is None: continue
            
            optimizer.zero_grad()
            
            # Forward
            pixel_values = torch.stack([torch.from_numpy(np.array(f)).permute(0, 3, 1, 2).float() / 255.0 for f in batch["frames"]]).to(device)
            # Tokenize and move to device
            tokens = model.y_encoder.tokenizer(batch["queries"], padding=True, truncation=True, return_tensors="pt").to(device)
            
            results = model.predictor(model.x_encoder(pixel_values), tokens.input_ids, tokens.attention_mask)
            
            pred_offsets = results.get("offsets") # (B, 2)
            gt_offsets = torch.tensor(batch["offset_targets"], dtype=torch.float32, device=device)
            
            loss = criterion(pred_offsets, gt_offsets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                pixel_values = torch.stack([torch.from_numpy(np.array(batch["frames"][i])).permute(0, 3, 1, 2).float() / 255.0 for i in range(len(batch["frames"]))]).to(device)
                tokens = model.y_encoder.tokenizer(batch["queries"], padding=True, truncation=True, return_tensors="pt").to(device)
                results = model.predictor(model.x_encoder(pixel_values), tokens.input_ids, tokens.attention_mask)
                loss = criterion(results["offsets"], torch.tensor(batch["offset_targets"], dtype=torch.float32, device=device))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if use_wandb:
            wandb.log({"reg/train_loss": avg_train_loss, "reg/val_loss": val_loss, "epoch": epoch})
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config.checkpoint_dir, "best_regression.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "val_loss": val_loss
            }, save_path)
            print(f"  ⭐ Saved best regression head to {save_path}")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    train()
