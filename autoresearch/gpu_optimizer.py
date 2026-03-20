"""
Autoresearch GPU-Optimisé - 5 min max par essai
Objectif: Trouver config maximisant throughput sans perdre en qualité
Budget: 3-4 essais max
"""

import subprocess
import time
import json
from pathlib import Path

# Essais à tester (batch, grad_acc, lr, dtype, workers)
EXPERIMENTS = [
    {"name": "baseline", "batch": 8, "grad_acc": 1, "lr": 3e-4, "dtype": "fp32", "workers": 4},
    {"name": "bf16_fast", "batch": 16, "grad_acc": 1, "lr": 4e-4, "dtype": "bf16", "workers": 8},
    {"name": "bf16_aggressive", "batch": 16, "grad_acc": 1, "lr": 5e-4, "dtype": "bf16", "workers": 16},
]

RESULTS_FILE = Path("autoresearch/results_gpu.json")

def run_experiment(exp, duration_minutes=5):
    """Lance un essai pour X minutes"""
    print(f"\n{'='*60}")
    print(f"Essai: {exp['name']}")
    print(f"Config: batch={exp['batch']}, lr={exp['lr']}, dtype={exp['dtype']}, workers={exp['workers']}")
    print(f"{'='*60}\n")
    
    # Modifier le script training/train.py temporairement
    # Pour l'autoresearch, on utilise un sous-ensemble de données (500 échantillons)
    
    cmd = f"""
cd ~/vl-jepa && timeout {duration_minutes * 60} uv run python3 << 'PYEOF'
import torch
import time
import sys
from pathlib import Path

# Config expérimentale
BATCH_SIZE = {exp['batch']}
GRAD_ACCUMULATION = {exp['grad_acc']}
LEARNING_RATE = {exp['lr']}
DTYPE = "{exp['dtype']}"
NUM_WORKERS = {exp['workers']}
MAX_SAMPLES = 500  # Sous-ensemble pour aller vite
MAX_STEPS = 100    # Nombre de steps limité

print(f"Testing: batch={{BATCH_SIZE}}, dtype={{DTYPE}}, lr={{LEARNING_RATE}}, workers={{NUM_WORKERS}}")

# Test de chargement
from vljepa.models import create_vl_jepa_model
from vljepa.config import VLJEPAConfig
from vljepa.dataset import CharadesSTADataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {{device}}")

# Config
config = VLJEPAConfig(
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    num_workers=NUM_WORKERS,
    dtype=DTYPE,
)

# Test chargement modèle
t0 = time.time()
model = create_vl_jepa_model(config, device)
load_time = time.time() - t0
print(f"Model load: {{load_time:.1f}}s")

# Test dataset (petit)
dataset = CharadesSTADataset(
    anno_file="data/charades_sta_train.txt",
    videos_dir="data/Charades_v1_480",
    config=config,
    split="train",
)
# Prendre sous-ensemble
indices = list(range(min(MAX_SAMPLES, len(dataset))))
from torch.utils.data import Subset
dataset = Subset(dataset, indices)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=lambda x: x[0] if len(x) == 1 else None,
)

print(f"Dataset: {{len(dataset)}} samples, {{len(loader)}} batches")

# Benchmark forward
torch.cuda.synchronize() if torch.cuda.is_available() else None
t0 = time.time()
steps_done = 0
for i, batch in enumerate(loader):
    if batch is None or steps_done >= MAX_STEPS:
        break
    
    try:
        pixel_values = model.x_encoder.preprocess_frames(batch["frames"], device=device)
        query_tokens = model.query_encoder.tokenize(batch["queries"], device=device)
        
        if DTYPE == "bf16":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(pixel_values, query_tokens["input_ids"], 
                              query_tokens["attention_mask"], batch["captions"])
        else:
            outputs = model(pixel_values, query_tokens["input_ids"], 
                          query_tokens["attention_mask"], batch["captions"])
        
        steps_done += 1
        if steps_done % 20 == 0:
            print(f"  Step {{steps_done}}/{{MAX_STEPS}}")
            
    except Exception as e:
        print(f"Error at step {{steps_done}}: {{e}}")
        break

torch.cuda.synchronize() if torch.cuda.is_available() else None
elapsed = time.time() - t0

if steps_done > 0:
    throughput = steps_done * BATCH_SIZE / elapsed
    print(f"\\n✓ Essai réussi!")
    print(f"  Steps: {{steps_done}}")
    print(f"  Time: {{elapsed:.1f}}s")
    print(f"  Throughput: {{throughput:.1f}} samples/sec")
    print(f"  VRAM used: {{torch.cuda.memory_allocated() / 1e9:.1f}}GB")
    
    # Sauver résultat
    result = {{
        "name": "{exp['name']}",
        "config": exp,
        "throughput": throughput,
        "steps": steps_done,
        "time": elapsed,
        "vram_gb": torch.cuda.memory_allocated() / 1e9,
        "success": True
    }}
else:
    print(f"\\n✗ Essai échoué")
    result = {{
        "name": "{exp['name']}",
        "config": exp,
        "success": False
    }}

import json
with open("autoresearch/results_gpu.json", "a") as f:
    f.write(json.dumps(result) + "\\n")

PYEOF
"""
    
    subprocess.run(cmd, shell=True, executable='/bin/bash')

def main():
    print("🚀 Autoresearch GPU-Optimisé")
    print(f"Budget: {len(EXPERIMENTS)} essais × 5 min = {len(EXPERIMENTS) * 5} min max\n")
    
    # Clean anciens résultats
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
    
    results = []
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n\n{'#'*60}")
        print(f"# ESSAI {i}/{len(EXPERIMENTS)}: {exp['name']}")
        print(f"{'#'*60}")
        
        run_experiment(exp, duration_minutes=5)
        
        # Petit délai entre essais
        if i < len(EXPERIMENTS):
            print(f"\n⏳ Pause 10s...")
            time.sleep(10)
    
    # Résumé
    print(f"\n\n{'='*60}")
    print("RÉSULTATS FINaux")
    print(f"{'='*60}\n")
    
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get('success'):
                        print(f"✓ {r['name']}: {r['throughput']:.1f} samples/sec "
                              f"({r['config']['dtype']}, batch={r['config']['batch']})")
                    else:
                        print(f"✗ {r['name']}: ÉCHEC")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
