#!/usr/bin/env python3
"""
Autoresearch GPU-Optimisé
3 essais × 100 steps avec logging complet sur W&B
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

def test_config(batch, grad_acc, lr, dtype, workers, name, max_steps=100):
    """Test une config - 100 steps avec logging complet"""
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
            config={
                "batch_size": batch,
                "grad_accumulation": grad_acc,
                "lr": lr,
                "dtype": dtype,
                "num_workers": workers,
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
        print(f"✓ Modèle chargé en {time.time()-t0:.1f}s")
        
        # Dataset
        print("Chargement dataset...")
        dataset = CharadesSTADataset(
            anno_file="data/charades_sta_train.txt",
            videos_dir="data/Charades_v1_480",
            config=config,
            split="train",
        )
        subset = Subset(dataset, list(range(min(500, len(dataset)))))
        loader = DataLoader(
            subset, batch_size=batch, shuffle=True,
            num_workers=workers, collate_fn=collate_fn, pin_memory=True
        )
        print(f"✓ Dataset: {len(subset)} échantillons")
        
        # Optimizer & Loss
        sigreg = SIGReg(knots=17).to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.predictor.parameters(), "lr": lr},
            {"params": model.y_encoder.projection.parameters(), "lr": lr * 0.05},
        ], weight_decay=0.05)
        
        # Training loop
        print("\nDébut entraînement...")
        model.train()
        torch.cuda.synchronize()
        t0 = time.time()
        
        steps_done = 0
        losses = []
        
        for batch_idx, batch in enumerate(loader):
            if steps_done >= max_steps:
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
                losses.append(loss.item())
                
                # W&B log - TOUTES les métriques
                if HAS_WANDB:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/infonce": loss_dict["loss/infonce"],
                        "train/sigreg": loss_dict.get("loss/sigreg", 0),
                        "train/step": steps_done,
                        "train/epoch": steps_done // len(loader),
                    })
                
                if steps_done % 20 == 0:
                    print(f"  step {steps_done:3d}/{max_steps} | loss: {loss.item():.4f} | infonce: {loss_dict['loss/infonce']:.4f}")
        
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1e9
        
        if steps_done >= max_steps * 0.7:  # Au moins 70% des steps
            avg_loss = sum(losses) / len(losses) if losses else 999
            throughput = steps_done * batch / elapsed
            
            print(f"\n✓ SUCCÈS!")
            print(f"  Steps: {steps_done}/{max_steps}")
            print(f"  Temps: {elapsed:.1f}s")
            print(f"  Throughput: {throughput:.1f} samples/sec")
            print(f"  Loss moyenne: {avg_loss:.4f}")
            print(f"  VRAM: {vram:.1f}GB")
            
            result = {
                "name": name, "batch": batch, "lr": lr, "dtype": dtype,
                "workers": workers, "throughput": throughput, "loss": avg_loss,
                "time": elapsed, "vram_gb": vram, "steps": steps_done, "success": True
            }
            
            if HAS_WANDB:
                wandb.log({
                    "final/throughput": throughput,
                    "final/avg_loss": avg_loss,
                    "final/vram_gb": vram,
                    "final/steps": steps_done,
                })
                run.finish()
            
            return result
        else:
            print(f"\n✗ ÉCHEC: seulement {steps_done} steps sur {max_steps}")
            if HAS_WANDB:
                run.finish()
            return {"name": name, "success": False, "reason": f"only_{steps_done}_steps"}
            
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
    print("🚀 Autoresearch GPU-Optimisé (3 essais × 100 steps)")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"W&B: {'✓ activé' if HAS_WANDB else '✗ désactivé'}\n")
    
    results = []
    for batch, grad_acc, lr, dtype, workers, name in EXPERIMENTS:
        result = test_config(batch, grad_acc, lr, dtype, workers, name)
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
        for r in sorted(successful, key=lambda x: -x["throughput"]):
            print(f"✓ {r['name']}: {r['throughput']:.0f} samp/s, loss={r['loss']:.4f}, VRAM={r['vram_gb']:.1f}GB")
        
        best = max(successful, key=lambda x: x["throughput"])
        print(f"\n🏆 MEILLEUR: {best['name']}")
        print(f"   batch={best['batch']}, lr={best['lr']}, dtype={best['dtype']}, workers={best['workers']}")
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