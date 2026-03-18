# VL-JEPA Cloud Training Guide

Guide complet pour l'entraînement de VL-JEPA sur le cloud avec optimisation des hyperparamètres via Auto-Research.

## 🎯 Objectif

Créer un modèle de **Temporal Moment Retrieval** performant capable de localiser un moment spécifique dans une vidéo à partir d'une description textuelle (comme un Ctrl+F pour vidéos).

Architecture :
- **X-Encoder**: V-JEPA 2 ViT-L (frozen, ~300M params)
- **Predictor**: Qwen 2.5 0.5B avec LoRA
- **Y-Encoder**: MiniLM-L6-v2 (frozen, ~22M params)

## 📊 Résultats Actuels

Derniers résultats sur Charades-STA test set:
```
R@1 IoU=0.3: 65.24%
R@1 IoU=0.5: 42.88%
R@1 IoU=0.7: 20.32%
mIoU: 41.82%
```

**Objectifs à atteindre (SOTA)**:
- R@1 IoU=0.5: ~50-60%
- R@1 IoU=0.7: ~30-40%

## 🚀 Quick Start

### 1. Configuration environnement

```bash
# Copier et remplir le fichier .env
cp .env.example .env

# Variables nécessaires:
# - WANDB_API_KEY: pour le tracking
# - HF_TOKEN: pour accéder au bucket
# - VASTAI_API_KEY: pour le cloud
# - HF_BUCKET_ID: max044/charades-sta-storage
```

### 2. Test local rapide

```bash
# Vérifier que tout fonctionne
python download_annotations.py

# Test rapide (10 steps)
python train.py --epochs 1 --max-steps 10 --debug

# Évaluation rapide
python eval.py --checkpoint checkpoints/latest.pt --limit 100
```

### 3. Lancer Auto-Research sur le Cloud

```bash
# Lancer 20 expériences sur RTX 4090 (budget: ~$2.50)
python scripts/cloud_autoresearch.py \
    --gpu rtx4090 \
    --budget 5 \
    --experiments 20 \
    --terminate
```

## 📁 Structure du Projet

```
vl-jepa/
├── vljepa/                    # Code source
│   ├── config.py              # Configuration
│   ├── dataset.py             # Dataset Charades-STA
│   ├── models.py              # Architecture VL-JEPA
│   ├── losses.py              # Fonctions de perte
│   └── utils.py               # Utilitaires
├── configs/
│   └── base.yaml              # Config de base
├── scripts/
│   ├── cloud_autoresearch.py  # Lanceur cloud
│   ├── setup_cloud_data.sh    # Setup données cloud
│   └── download_test_data.py  # Téléchargement test
├── autoresearch/
│   └── PROGRAM.md             # Instructions Auto-Research
├── data/
│   ├── charades_sta_train.txt
│   ├── charades_sta_test.txt
│   └── Charades_v1_480/       # Vidéos
├── train.py                   # Entraînement
├── eval.py                    # Évaluation
└── CLOUD.md                   # Ce fichier
```

## ☁️ Cloud Training

### Stockage HF (XET)

Les données sont stockées sur **HF Storage** pour un accès rapide:
- **Bucket**: `max044/charades-sta-storage`
- **Format**: XET (déduplication, transferts rapides)
- **Contenu**: 9,848 vidéos + annotations

### Types de GPU disponibles

| GPU | VRAM | Prix/h | Recommandation |
|-----|------|--------|----------------|
| RTX 3090 | 24GB | ~$0.30-0.40 | ⭐ Meilleur rapport qualité/prix |
| RTX 4090 | 24GB | ~$0.40-0.60 | ⭐ Rapide, un peu plus cher |
| A5000 | 24GB | ~$0.60-0.80 | Stable |
| A6000 | 48GB | ~$0.80-1.00 | Si besoin de plus de VRAM |

### Workflow Auto-Research

1. **Phase 1: Screening** (5-10 min par expérience)
   - Grid search sur hyperparamètres clés
   - 20-30 configurations testées
   - Métrique: convergence loss + R@1

2. **Phase 2: Validation** (10-15 min)
   - Top 3 configurations
   - Run plus long pour validation

3. **Phase 3: Full Training**
   - Meilleure config
   - 20 epochs complets
   - Évaluation finale

### Hyperparamètres à explorer

```python
# Priorité haute
learning_rate: [1e-4, 3e-4, 1e-3, 3e-3]
batch_size: [2, 4, 8]  # Selon VRAM disponible
lora_r: [32, 64, 128]
lora_alpha: [64, 128, 256]

# Priorité moyenne  
temperature: [0.03, 0.05, 0.07, 0.1, 0.15]
sigreg_weight: [0.0, 0.05, 0.1, 0.2, 0.5]
warmup_steps: [100, 200, 500]
weight_decay: [0.0, 0.01, 0.001]
```

## 🔧 Commandes utiles

### Entraînement

```bash
# Entraînement standard
python train.py --epochs 20 --batch-size 4

# Avec W&B
python train.py --epochs 20 --batch-size 4 --wandb

# Reprendre depuis checkpoint
python train.py --resume checkpoints/checkpoint_epoch_10.pt

# Config personnalisée
python train.py --config configs/custom.yaml
```

### Évaluation

```bash
# Évaluation complète
python eval.py --checkpoint checkpoints/best.pt

# Limite le nombre d'échantillons (rapide)
python eval.py --checkpoint checkpoints/best.pt --limit 100

# Avec W&B
python eval.py --checkpoint checkpoints/best.pt --wandb
```

### Cloud

```bash
# Lancer nouvelle instance
python scripts/cloud_autoresearch.py --gpu rtx4090 --budget 5 --experiments 20

# Utiliser instance existante
python scripts/cloud_autoresearch.py --instance-id 12345 --config scripts/sweep_config.json

# Dry run (voir ce qui serait fait)
python scripts/cloud_autoresearch.py --gpu rtx4090 --budget 3 --dry-run

# Terminer instance après
python scripts/cloud_autoresearch.py --gpu rtx4090 --budget 5 --experiments 20 --terminate
```

## 📈 Monitoring

### Weights & Biases

- **Projet**: `vl-jepa`
- **URL**: https://wandb.ai/maxence-cabiddu/vl-jepa
- **Metrics trackés**:
  - Train/Val Loss
  - InfoNCE Loss
  - R@1, R@5 (IoU=0.3, 0.5, 0.7)
  - mIoU
  - VRAM usage

### Logs Cloud

```bash
# Voir instances actives
vastai show instances

# SSH dans une instance
vastai ssh <instance_id>

# Voir logs
vastai logs <instance_id>
```

## 💡 Astuces

### Optimisation mémoire

```python
# Si OOM sur RTX 4090 (24GB)
batch_size: 2
gradient_accumulation: 2  # Effective batch = 4
mixed_precision: True
```

### Speed run

```python
# Pour tests rapides
debug: True
debug_samples: 100
num_workers: 4
```

### Checkpoints

- Sauvegardés automatiquement tous les 2 epochs
- Meilleur modèle selon val_loss
- Reprise possible avec `--resume`

## 🐛 Dépannage

### "No such file or directory: data/charades_sta_train.txt"

```bash
python download_annotations.py
```

### "Video file not found"

```bash
# Télécharger depuis HF Storage
hf sync hf://buckets/max044/charades-sta-storage/Charades_v1_480 data/Charades_v1_480

# Ou utiliser lazy loading (automatique si use_hf_storage=True)
```

### "CUDA out of memory"

```bash
# Réduire batch size
python train.py --batch-size 2

# Ou utiliser accumulation de gradients
python train.py --batch-size 2 --grad-accum 2
```

### "VastAI instance not connecting"

```bash
# Vérifier crédits
vastai show account

# Voir offres disponibles
vastai search offers --limit 10
```

## 📚 Ressources

- **Paper V-JEPA**: https://arxiv.org/abs/2403.XXXXX
- **Dataset Charades**: https://prior.allenai.org/projects/charades
- **HF Storage Docs**: https://huggingface.co/docs/hub/xet
- **Vast.ai**: https://vast.ai/

## 🎯 Prochaines Étapes

1. ✅ Configurer environnement (.env)
2. ✅ Lancer test local
3. 🔄 Lancer Auto-Research Phase 1 (20 expériences)
4. ⏳ Analyser résultats Phase 1
5. ⏳ Lancer Phase 2 (validation top configs)
6. ⏳ Full training avec meilleure config
7. ⏳ Évaluation finale + publication

## 🧠 Leçons Apprises & Bonnes Pratiques

### 🔐 Sécurité - Tokens et Secrets

**NE JAMAIS exposer les tokens dans les logs ou commandes !**

```bash
# ✅ CORRECT - Utiliser un fichier .env
scp .env root@instance:~/vl-jepa/.env

# ❌ INCORRECT - Exposer le token dans la commande
ssh root@instance "export HF_TOKEN=hf_xxx..."
```

**Procédure sécurisée pour le cloud:**
1. Créer `.env` localement avec tous les secrets
2. Copier via SCP : `scp -P [PORT] .env root@[IP]:~/vl-jepa/.env`
3. Sur l'instance, charger avec : `export $(grep -v '^#' .env | xargs)`
4. Authentifier HF : `python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"`

### 📦 Hugging Face Storage (XET) vs Dataset

**Migration de Dataset vers Storage (XET)**

- **Ancien système** : Dataset classique HF (`max044/Charades_v1_480`)
  - Lent, téléchargement vidéo par vidéo
  - Nécessite `datasets` library
  
- **Nouveau système** : HF Storage Bucket (`max044/charades-sta-storage`)
  - Déduplication XET (60%+ d'économie)
  - Transferts rapides avec `hf sync`
  - Commande : `hf sync hf://buckets/[BUCKET]/Charades_v1_480 ./data/`

**Configuration requise:**
```python
# vljepa/config.py
use_hf_storage: bool = True
hf_bucket_id: str = "max044/charades-sta-storage"
```

### ☁️ Problèmes Cloud Vast.ai

**Template recommandé:**
- ✅ `vastai/pytorch:latest` ou `vastai/pytorch:cuda-12.8.1-auto`
- ❌ Éviter `pytorch/pytorch:...` (trop long à charger)

**Variables d'environnement sur l'instance:**
```bash
# Le fichier .env DOIT être présent sur l'instance
ls -la ~/vl-jepa/.env

# Sinon le chargement des modèles HF échoue
# Symptôme: "Loading models..." qui dure indéfiniment
```

**Authentification HF sur l'instance:**
```bash
# Lire le token depuis .env
HF_TOKEN=$(grep '^HF_TOKEN=' .env | cut -d'=' -f2)

# Authentifier Python
python3 << EOF
from huggingface_hub import login
login(token='$HF_TOKEN', add_to_git_credential=True)
print("Authenticated")
EOF
```

### 🐛 Erreurs Fréquentes

**"Loading models..." bloqué:**
- Cause : Pas de HF_TOKEN configuré
- Solution : Voir section authentification ci-dessus

**"No such file or directory: data/charades_sta_train.txt":**
- Cause : Annotations non présentes
- Solution : `python download_annotations.py` ou télécharger depuis GitHub MESM

**"CUDA out of memory":**
- Réduire batch_size : `--batch-size 2`
- Utiliser gradient accumulation
- Désactuer mixed precision si nécessaire

**"401 Client Error" sur HF Storage:**
- Cause : Token invalide ou bucket privé sans auth
- Solution : Vérifier HF_TOKEN et faire `huggingface-cli login`

### 📊 Métriques et Résultats

**Objectifs SOTA Charades-STA:**
- R@1 IoU=0.5: ~50-60% (actuel: 42.88%)
- R@1 IoU=0.7: ~30-40% (actuel: 20.32%)
- mIoU: ~45-50% (actuel: 41.82%)

**Hyperparamètres clés à tuner:**
1. `learning_rate` : [1e-4, 3e-4, 1e-3] - Impact majeur
2. `batch_size` : [2, 4, 8] - Selon VRAM
3. `lora_r` : [32, 64, 128] - Capacité du predictor
4. `temperature` : [0.05, 0.07, 0.1] - InfoNCE
5. `sigreg_weight` : [0.0, 0.05, 0.1, 0.2] - Régularisation

### 🚀 Workflow Auto-Research Efficace

**Phase 1 - Screening (5 min/exp):**
```bash
# Lancer sur RTX 4090
python scripts/cloud_autoresearch.py --gpu rtx4090 --budget 3 --experiments 15
```

**Phase 2 - Validation (15 min/exp):**
- Prendre top 3 configs
- Run 5 epochs complets
- Comparer R@1 et mIoU

**Phase 3 - Full Training:**
- Meilleure config
- 20 epochs
- Évaluation sur test set complet

### 💡 Astuces pour l'Auto-Research

**Scripts à utiliser sur l'instance:**
```bash
# 1. Envoyer .env
scp -P [PORT] .env root@[IP]:~/vl-jepa/.env

# 2. Se connecter
ssh -p [PORT] root@[IP]

# 3. Lancer les expériences
cd ~/vl-jepa
bash scripts/run_autoresearch.sh
```

**Monitoring:**
```bash
# Voir l'avancement
tail -f ~/vl-jepa/autoresearch/run.log

# Voir les résultats
cat ~/vl-jepa/autoresearch/results.csv

# Voir l'utilisation GPU
watch -n 1 nvidia-smi
```

---

**Dernière mise à jour**: 2026-03-18
**Version**: 1.1
**Crédits restants Vast.ai**: $11.00
