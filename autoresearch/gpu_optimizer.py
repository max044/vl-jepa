#!/usr/bin/env python3
"""
Autoresearch GPU-Optimisé Minimal
3 essais × 5 min max
"""

import json
import time
import torch
from pathlib import Path
from vljepa.models import create_vl_jepa_model
from vljepa.config import VLJEPAConfig
from vljepa.dataset import CharadesSTADataset, collate_fn
from vljepa.losses import vl_jepa_loss, SIGReg
from torch.utils.data import DataLoader, Subset

# Essais: (batch, grad_acc, lr, dtype, workers, name)
EXPERIMENTS = [
    (8, 1, 3e-4, "fp32", 4, "baseline_adapted"),
    (16, 1, 4e-4, "bf16", 8, "bf16_safe"),
    (16, 1, 5e-4, "bf16", 16, "bf16_aggressive"),
]

def test_config(batch, grad_acc, lr, dtype, workers, name, max_steps=100):
    """Test une config pour 100 steps"""
    print(f"\n{'='*60}")
    print(f"Essai: {name}")
    print(f"  batch={batch}, lr={lr}, dtype={dtype}, workers={workers}")
    print(f"{'='*60}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Config
        config = VLJEPAConfig(
            batch_size=batch,
            lr=lr,
            num_workers=workers,
            dtype=dtype,
            device="cuda",
            grad_accumulation=grad_acc,
        )
        
        # Modèle
        t0 = time.time()
        model = create_vl_jepa_model(config, device)
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
            num_workers=workers, collate_fn=collate_fn
        )
        print(f"✓ Dataset: {len(subset)} samples, {len(loader)} batches")
        
        # Warm-up
        sigreg = SIGReg(embed_dim=config.embed_dim).to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.predictor.parameters(), "lr": lr},
            {"params": model.y_encoder.projection.parameters(), "lr": lr * 0.05},
        ], weight_decay=0.05)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        model.train()
        t0 = time.time()
        
        steps_done = 0
        total_loss = 0
        
        for i, batch in enumerate(loader):
            if batch is None or steps_done >= max_steps:
                break
            
            try:
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
                
                (loss / grad_acc).backward()
                
                if (i + 1) % grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    steps_done += 1
                
                total_loss += loss.item()
                
                if steps_done % 25 == 0:
                    print(f"  Step {steps_done}/{max_steps} | loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"  ✗ Error at step {steps_done}: {e}")
                break
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - t0
        
        if steps_done > 50:  # Minimum 50 steps pour valider
            throughput = steps_done * batch / elapsed
            avg_loss = total_loss / steps_done
            vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            
            print(f"\n✓ SUCCÈS!")
            print(f"  Steps: {steps_done}")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Throughput: {throughput:.1f} samples/sec")
            print(f"  Avg loss: {avg_loss:.4f}")
            print(f"  VRAM max: {vram:.1f}GB")
            
            return {
                "name": name,
                "batch": batch,
                "lr": lr,
                "dtype": dtype,
                "workers": workers,
                "throughput": throughput,
                "loss": avg_loss,
                "time": elapsed,
                "vram_gb": vram,
                "success": True
            }
        else:
            print(f"\n✗ ÉCHEC (trop peu de steps)")
            return {"name": name, "success": False, "reason": "too_few_steps"}
            
    except Exception as e:
        print(f"\n✗ ÉCHEC: {e}")
        return {"name": name, "success": False, "reason": str(e)}
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    print("🚀 Autoresearch GPU-Optimisé (3 essais × 5 min max)")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
    
    results = []
    
    for batch, grad_acc, lr, dtype, workers, name in EXPERIMENTS:
        result = test_config(batch, grad_acc, lr, dtype, workers, name)
        results.append(result)
        
        # Pause entre essais
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
        best_speed = max(successful, key=lambda x: x["throughput"])
        best_quality = min(successful, key=lambda x: x["loss"])
        
        print("Essais réussis:")
        for r in successful:
            print(f"  ✓ {r['name']}: {r['throughput']:.1f} samp/s, loss={r['loss']:.4f}, "
                  f"VRAM={r['vram_gb']:.1f}GB ({r['dtype']}, batch={r['batch']})")
        
        print(f"\n🏆 Meilleur throughput: {best_speed['name']}")
        print(f"   {best_speed['throughput']:.1f} samples/sec")
        
        print(f"\n🏆 Meilleure qualité: {best_quality['name']}")
        print(f"   Loss: {best_quality['loss']:.4f}")
        
        # Recommandation
        print(f"\n💡 RECOMMANDATION:")
        if best_speed['name'] == best_quality['name']:
            print(f"   → {best_speed['name']} est optimal!")
            print(f"     batch={best_speed['batch']}, lr={best_speed['lr']}, "
                  f"dtype={best_speed['dtype']}, workers={best_speed['workers']}")
        else:
            # Trade-off
            print(f"   → {best_speed['name']} pour la vitesse")
            print(f"     {best_quality['name']} pour la qualité")
    else:
        print("✗ Tous les essais ont échoué")
    
    # Sauver
    with open("autoresearch/gpu_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Résultats sauvés dans autoresearch/gpu_results.json")

if __name__ == "__main__":
    main()
