#!/usr/bin/env python3
"""
Autoresearch GPU-Optimisé
3 essais × ~60 steps avec validation toutes les 25 steps
"""

import json
import time
import torch
from pathlib import Path
from vljepa.models import VLJepa
from vljepa.config import Config
from vljepa.dataset import CharadesSTADataset, collate_fn
from vljepa.losses import vl_jepa_loss, SIGReg
from torch.utils.data import DataLoader, Subset

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Essais: (batch, grad_acc, lr, dtype, workers, name)
EXPERIMENTS = [
    (8, 1, 3e-4, "fp32", 4, "fp32_b8_w4"),
    (16, 1, 4e-4, "bf16", 8, "bf16_b16_w8"),
    (16, 1, 5e-4, "bf16", 16, "bf16_b16_w16"),
]

VAL_EVERY = 25  # Validation toutes les 25 steps

def validate(model, val_loader, sigreg, device, dtype):
    """Validation sur 100 samples"""
    model.eval()
    val_loss = 0
    val_infonce = 0
    val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            
            pixel_values = model.x_encoder.preprocess_frames(batch["frames"], device=device)
            query_tokens = model.query_encoder.tokenize(batch["queries"], device=device)
            
            if dtype == "bf16" and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        pixel_values, query_tokens["input_ids"],
                        query_tokens["attention_mask"], batch["captions"]
                    )
                    loss, loss_dict = vl_jepa_loss(
                        outputs["sy_hat"], outputs["sy"],
                        temperature=0.025, sigreg_weight=0.05, sigreg_module=sigreg
                    )
            else:
                outputs = model(
                    pixel_values, query_tokens["input_ids"],
                    query_tokens["attention_mask"], batch["captions"]
                )
                loss, loss_dict = vl_jepa_loss(
                    outputs["sy_hat"], outputs["sy"],
                    temperature=0.025, sigreg_weight=0.05, sigreg_module=sigreg
                )
            
            val_loss += loss.item()
            val_infonce += loss_dict["loss/infonce"]
            val_batches += 1
    
    model.train()
    return val_loss / val_batches if val_batches > 0 else 999

def test_config(batch, grad_acc, lr, dtype, workers, name, timeout_sec=300):
    """Test une config avec timeout et validation régulière"""
    print(f"\n{'='*60}")
    print(f"Essai: {name}")
    print(f"  batch={batch}, lr={lr}, dtype={dtype}, workers={workers}")
    print(f"  Timeout: {timeout_sec}s, Validation: toutes les {VAL_EVERY} steps")
    print(f"{'='*60}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # W&B run
    if HAS_WANDB:
        run = wandb.init(
            project="vl-jepa-autoresearch",
            name=f"gpu_{name}",
            config={
                "batch_size": batch,
                "grad_accumulation": grad_acc,
                "lr": lr,
                "dtype": dtype,
                "num_workers": workers,
                "val_every": VAL_EVERY,
            },
            reinit=True,
        )
    
    try:
        # Config
        config = Config(batch_size=batch, lr=lr, num_workers=workers, dtype=dtype)
        config.device = "cuda"
        
        print("Chargement modèle...")
        t0 = time.time()
        model = VLJepa(config).to(device)
        load_time = time.time() - t0
        print(f"✓ Modèle en {load_time:.1f}s")
        
        # Dataset train
        print("Chargement train...")
        train_dataset = CharadesSTADataset(
            anno_file="data/charades_sta_train.txt",
            videos_dir="data/Charades_v1_480",
            config=config,
            split="train",
        )
        train_subset = Subset(train_dataset, list(range(min(500, len(train_dataset)))))
        train_loader = DataLoader(
            train_subset, batch_size=batch, shuffle=True,
            num_workers=workers, collate_fn=collate_fn, pin_memory=True
        )
        print(f"✓ Train: {len(train_subset)} samples")
        
        # Dataset validation
        print("Chargement val...")
        val_dataset = CharadesSTADataset(
            anno_file="data/charades_sta_train.txt",
            videos_dir="data/Charades_v1_480",
            config=config,
            split="train",
        )
        val_subset = Subset(val_dataset, list(range(500, min(600, len(val_dataset)))))
        val_loader = DataLoader(
            val_subset, batch_size=batch, shuffle=False,
            num_workers=workers, collate_fn=collate_fn, pin_memory=True
        )
        print(f"✓ Val: {len(val_subset)} samples")
        
        # Optimizer & Loss
        sigreg = SIGReg(knots=17).to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.predictor.parameters(), "lr": lr},
            {"params": model.y_encoder.projection.parameters(), "lr": lr * 0.05},
        ], weight_decay=0.05)
        
        # Training loop avec timeout
        print(f"\n📊 Entraînement (timeout {timeout_sec}s)...")
        model.train()
        torch.cuda.synchronize()
        t0_train = time.time()
        
        steps_done = 0
        train_losses = []
        best_val_loss = float('inf')
        
        for batch_idx, batch in enumerate(train_loader):
            # Timeout check
            if time.time() - t0_train > timeout_sec:
                print(f"\n⏱️ Timeout atteint ({timeout_sec}s)")
                break
            
            if batch is None:
                continue
            
            # Preprocess
            pixel_values = model.x_encoder.preprocess_frames(batch["frames"], device=device)
            query_tokens = model.query_encoder.tokenize(batch["queries"], device=device)
            
            # Forward
            if dtype == "bf16" and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        pixel_values, query_tokens["input_ids"],
                        query_tokens["attention_mask"], batch["captions"]
                    )
                    loss, loss_dict = vl_jepa_loss(
                        outputs["sy_hat"], outputs["sy"],
                        temperature=0.025, sigreg_weight=0.05, sigreg_module=sigreg
                    )
            else:
                outputs = model(
                    pixel_values, query_tokens["input_ids"],
                    query_tokens["attention_mask"], batch["captions"]
                )
                loss, loss_dict = vl_jepa_loss(
                    outputs["sy_hat"], outputs["sy"],
                    temperature=0.025, sigreg_weight=0.05, sigreg_module=sigreg
                )
            
            # Backward
            (loss / grad_acc).backward()
            
            if (batch_idx + 1) % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                steps_done += 1
                train_losses.append(loss.item())
                
                # W&B log train
                if HAS_WANDB:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/infonce": loss_dict["loss/infonce"],
                        "train/sigreg": loss_dict.get("loss/sigreg", 0),
                        "train/step": steps_done,
                    })
                
                # Validation toutes les VAL_EVERY steps
                if steps_done % VAL_EVERY == 0:
                    val_loss = validate(model, val_loader, sigreg, device, dtype)
                    best_val_loss = min(best_val_loss, val_loss)
                    
                    print(f"  step {steps_done} | train_loss: {loss.item():.4f} | val_loss: {val_loss:.4f}")
                    
                    if HAS_WANDB:
                        wandb.log({
                            "val/loss": val_loss,
                            "val/best": best_val_loss,
                            "val/step": steps_done,
                        })
                elif steps_done % 10 == 0:
                    print(f"  step {steps_done} | train_loss: {loss.item():.4f}")
        
        torch.cuda.synchronize()
        train_time = time.time() - t0_train
        train_avg_loss = sum(train_losses) / len(train_losses) if train_losses else 999
        vram = torch.cuda.max_memory_allocated() / 1e9
        throughput = steps_done * batch / train_time if steps_done > 0 else 0
        
        # Validation finale si pas faite
        if steps_done > 0 and (steps_done % VAL_EVERY != 0 or steps_done < VAL_EVERY):
            val_loss = validate(model, val_loader, sigreg, device, dtype)
            best_val_loss = min(best_val_loss, val_loss)
            if HAS_WANDB:
                wandb.log({"val/loss": val_loss, "val/step": steps_done})
        
        print(f"\n✓ Terminé!")
        print(f"  Steps: {steps_done}")
        print(f"  Temps: {train_time:.1f}s")
        print(f"  Throughput: {throughput:.0f} samples/sec")
        print(f"  Train loss: {train_avg_loss:.4f}")
        print(f"  Val loss: {best_val_loss:.4f}")
        print(f"  VRAM: {vram:.1f}GB")
        
        if HAS_WANDB:
            wandb.log({
                "final/throughput": throughput,
                "final/train_loss": train_avg_loss,
                "final/val_loss": best_val_loss,
                "final/vram_gb": vram,
                "final/steps": steps_done,
            })
            run.finish()
        
        return {
            "name": name, "batch": batch, "lr": lr, "dtype": dtype,
            "workers": workers, "throughput": throughput,
            "train_loss": train_avg_loss, "val_loss": best_val_loss,
            "time": train_time, "vram_gb": vram, "steps": steps_done,
            "success": True
        }
            
    except Exception as e:
        print(f"\n✗ ÉCHEC: {e}")
        import traceback
        traceback.print_exc()
        if HAS_WANDB:
            run.finish()
        return {"name": name, "success": False, "reason": str(e)[:100]}
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    print("🚀 Autoresearch GPU-Optimisé (3 essais × 5min, val toutes les 25 steps)")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"W&B: {'✓ activé' if HAS_WANDB else '✗ désactivé'}\n")
    
    results = []
    for batch, grad_acc, lr, dtype, workers, name in EXPERIMENTS:
        result = test_config(batch, grad_acc, lr, dtype, workers, name, timeout_sec=300)
        results.append(result)
        if len(results) < len(EXPERIMENTS):
            print("\n⏳ Pause 5s entre essais...")
            time.sleep(5)
    
    # Résumé
    print(f"\n{'='*60}")
    print("RÉSULTATS FINAUX")
    print(f"{'='*60}\n")
    
    successful = [r for r in results if r.get("success")]
    
    if successful:
        print(f"{'Config':<20} | {'Val Loss':>8} | {'Train Loss':>10} | {'Speed':>8} | {'VRAM':>6}")
        print("-" * 70)
        for r in successful:
            print(f"{r['name']:<20} | {r['val_loss']:8.4f} | {r['train_loss']:10.4f} | {r['throughput']:6.0f} | {r['vram_gb']:4.1f}GB")
        
        # Meilleur par val_loss
        best = min(successful, key=lambda x: x["val_loss"])
        print(f"\n🏆 MEILLEUR CONFIG (val_loss): {best['name']}")
        print(f"   Val loss: {best['val_loss']:.4f}")
        print(f"   Config: batch={best['batch']}, lr={best['lr']}, dtype={best['dtype']}, workers={best['workers']}")
    else:
        print("✗ Tous les essais ont échoué:")
        for r in results:
            print(f"  {r['name']}: {r.get('reason', 'unknown')}")
    
    # Sauvegarder
    Path("autoresearch").mkdir(exist_ok=True)
    with open("autoresearch/gpu_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Résultats: autoresearch/gpu_results.json")

if __name__ == "__main__":
    main()