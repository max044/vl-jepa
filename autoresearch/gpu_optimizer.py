#!/usr/bin/env python3
"""
Autoresearch GPU-Optimisé Minimal
3 essais × ~3 min
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
    (8, 1, 3e-4, "fp32", 4, "baseline_fp32_batch8"),
    (16, 1, 4e-4, "bf16", 8, "bf16_batch16"),
    (16, 1, 5e-4, "bf16", 16, "bf16_batch16_fast"),
]

def test_config(batch, grad_acc, lr, dtype, workers, name, max_steps=75):
    """Test une config pour 75 steps"""
    print(f"\n{'='*60}")
    print(f"Essai: {name}")
    print(f"  batch={batch}, lr={lr}, dtype={dtype}, workers={workers}")
    print(f"{'='*60}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # W&B run
    if HAS_WANDB:
        run = wandb.init(
            project="vl-jepa-autoresearch",
            name=f"gpu_{name}",
            config={"batch": batch, "lr": lr, "dtype": dtype, "workers": workers},
            reinit=True,
        )
    
    try:
        # Config
        config = Config(
            batch_size=batch,
            lr=lr,
            num_workers=workers,
            dtype=dtype,
        )
        config.device = "cuda"
        
        # Modèle
        t0 = time.time()
        model = VLJepa(config)
        model = model.to(device)
        load_time = time.time() - t0
        print(f"✓ Model loaded in {load_time:.1f}s")
        
        # Dataset (500 samples max)
        dataset = CharadesSTADataset(
            anno_file="data/charades_sta_train.txt",
            videos_dir="data/Charades_v1_480",
            config=config,
            split="train",
        )
        subset = Subset(dataset, list(range(min(500, len(dataset)))))
        loader = DataLoader(
            subset, batch_size=batch, shuffle=True, 
            num_workers=workers, collate_fn=collate_fn,
            pin_memory=True,
        )
        print(f"✓ Dataset: {len(subset)} samples") 
        
        # Optimizer
        sigreg = SIGReg(knots=17).to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.predictor.parameters(), "lr": lr},
            {"params": model.y_encoder.projection.parameters(), "lr": lr * 0.05},
        ], weight_decay=0.05)
        
        # Benchmark
        torch.cuda.synchronize()
        model.train()
        t0 = time.time()
        
        steps_done = 0
        total_loss = 0
        errors = []
        
        for batch_idx, batch in enumerate(loader):
            if steps_done >= max_steps:
                break
            
            if batch is None:
                continue
            
            try:
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
                            temperature=0.025, sigreg_weight=0.05,
                            sigreg_module=sigreg
                        )
                else:
                    outputs = model(
                        pixel_values, query_tokens["input_ids"],
                        query_tokens["attention_mask"], batch["captions"]
                    )
                    loss, loss_dict = vl_jepa_loss(
                        outputs["sy_hat"], outputs["sy"],
                        temperature=0.025, sigreg_weight=0.05,
                        sigreg_module=sigreg
                    )
                
                # Backward
                (loss / grad_acc).backward()
                
                if (batch_idx + 1) % grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    steps_done += 1
                    total_loss += loss.item()
                    
                    # W&B log
                    if HAS_WANDB and steps_done % 10 == 0:
                        wandb.log({"train/loss": loss.item(), "step": steps_done})
                    
                    if steps_done % 15 == 0:
                        print(f"  Step {steps_done}/{max_steps} | loss: {loss.item():.4f}")
                
            except Exception as e:
                errors.append(f"step{steps_done}: {str(e)[:50]}")
                if len(errors) > 5:
                    print(f"  ✗ Trop d'erreurs, arrêt")
                    break
                continue
        
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        
        if steps_done >= 30:  # Seuil minimum 30 steps
            throughput = steps_done * batch / elapsed
            avg_loss = total_loss / steps_done if steps_done > 0 else 999
            vram = torch.cuda.max_memory_allocated() / 1e9
            
            print(f"\n✓ SUCCÈS!")
            print(f"  Steps: {steps_done}/{max_steps}")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Throughput: {throughput:.1f} samples/sec")
            print(f"  Avg loss: {avg_loss:.4f}")
            print(f"  VRAM max: {vram:.1f}GB")
            if errors:
                print(f"  Erreurs: {len(errors)}")
            
            result = {
                "name": name,
                "batch": batch,
                "lr": lr,
                "dtype": dtype,
                "workers": workers,
                "throughput": throughput,
                "loss": avg_loss,
                "time": elapsed,
                "vram_gb": vram,
                "steps": steps_done,
                "success": True
            }
            
            if HAS_WANDB:
                wandb.log({
                    "throughput": throughput,
                    "avg_loss": avg_loss,
                    "vram_gb": vram,
                })
                run.finish()
            
            return result
        else:
            print(f"\n✗ ÉCHEC: seulement {steps_done} steps effectués")
            if errors:
                print(f"  Erreurs: {errors[:3]}")
            
            if HAS_WANDB:
                run.finish()
            
            return {"name": name, "success": False, "reason": f"only_{steps_done}_steps", "errors": errors[:3]}
            
    except Exception as e:
        print(f"\n✗ ÉCHEC CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        
        if HAS_WANDB:
            run.finish()
        
        return {"name": name, "success": False, "reason": str(e)[:100]}
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    print("🚀 Autoresearch GPU-Optimisé (3 essais)")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"W&B: {'activé' if HAS_WANDB else 'désactivé'}\n")
    
    results = []
    
    for batch, grad_acc, lr, dtype, workers, name in EXPERIMENTS:
        result = test_config(batch, grad_acc, lr, dtype, workers, name)
        results.append(result)
        
        if len(results) < len(EXPERIMENTS):
            print("\n⏳ Pause 5s...")
            time.sleep(5)
    
    # Résumé
    print(f"\n\n{'='*60}")
    print("RÉSULTATS")
    print(f"{'='*60}\n")
    
    successful = [r for r in results if r.get("success")]
    
    if successful:
        # Trier par throughput
        for r in successful:
            print(f"✓ {r['name']}: {r['throughput']:.1f} samp/s, loss={r['loss']:.4f}, "
                  f"VRAM={r['vram_gb']:.1f}GB ({r['dtype']}, batch={r['batch']})")
        
        best = max(successful, key=lambda x: x["throughput"])
        print(f"\n🏆 MEILLEURE CONFIG: {best['name']}")
        print(f"   batch={best['batch']}, lr={best['lr']}, dtype={best['dtype']}, workers={best['workers']}")
        print(f"   throughput: {best['throughput']:.1f} samples/sec")
    else:
        print("✗ Tous les essais ont échoué")
        for r in results:
            print(f"  {r['name']}: {r.get('reason', 'unknown')}")
    
    # Sauver
    Path("autoresearch").mkdir(exist_ok=True)
    with open("autoresearch/gpu_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Résultats dans autoresearch/gpu_results.json")

if __name__ == "__main__":
    main()