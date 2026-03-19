# VL-JEPA - Guide pour Agents IA

## Vue d'ensemble

VL-JEPA est un modèle de **Temporal Moment Retrieval** : trouver un moment dans une vidéo à partir d'une description textuelle.

**Architecture actuelle:**
- **X-Encoder**: V-JEPA 2 ViT-L (frozen, ~300M params)
- **Predictor**: Qwen3.5-0.8B (trainable, full fine-tune, pas de LoRA)
- **Y-Encoder**: Qwen3-Embedding-0.6B (trainable)
- **Embed dim**: 1024
- **Loss**: InfoNCE bidirectionnel + SIGReg

**Baseline actuelle:**
- val_loss: 1.101749 (commit 85395c7)
- Entraînement: 5 min sur 500 échantillons

---

## Structure du Repository

```
vl-jepa/
├── autoresearch/         # Expérimentations rapides (5 min)
│   ├── train.py         # Script principal - MODIFIER UNIQUEMENT CELUI-CI
│   ├── prepare.py       # Prépare sous-ensemble données (500 vidéos)
│   └── run.sh           # Lance expérience: bash autoresearch/run.sh
├── training/            # Entraînement complet
│   ├── train.py         # Training 20 epochs, tous les checkpoints
│   ├── eval.py          # Évaluation
│   └── download_data.py # Télécharge TOUTES les vidéos (15GB)
├── scripts/             # Scripts cloud
│   ├── cloud_train.sh   # Lancer training
│   ├── cloud_eval.sh    # Lancer évaluation
│   └── setup_instance.sh # Setup instance
├── vljepa/              # Code modèle (NE PAS MODIFIER)
│   ├── config.py
│   ├── models.py
│   ├── dataset.py
│   └── losses.py
├── FILE_MANIFEST.md     # Documentation complète fichiers
└── README.md            # Vue d'ensemble
```

---

## Gestion des Données

### Autoresearch (Expérimentations)

**Fichier:** `autoresearch/prepare.py`

Télécharge UN SOUS-ENSEMBLE des données pour aller vite:
- 500 vidéos maximum (vs 9,848 totales)
- ~800MB au lieu de 15GB
- Annotations train/test
- Stocké dans `data/autoresearch/`

**Usage:**
```bash
cd ~/vl-jepa
uv run autoresearch/prepare.py --subset 500
```

**Sur le cloud:** Les données sont déjà préparées dans `data/autoresearch/` (lien symbolique vers les vidéos complètes).

### Training Complet

**Fichier:** `training/download_data.py`

Télécharge TOUTES les données:
- 9,848 vidéos Charades-STA
- ~15GB total
- Annotations train/test
- Stocké dans `data/`

**Usage:**
```bash
uv run training/download_data.py
```

**Important:** Sur l'instance cloud Vast.ai, les vidéos sont DÉJÀ présentes dans `data/Charades_v1_480/` (16GB). Ne pas retélécharger.

### Dataset.py (Streaming/Lazy Loading)

Le dataset `vljepa/dataset.py` fonctionne en **streaming**:
- Ne charge pas toutes les vidéos en mémoire
- Charge les frames à la volée pendant l'entraînement
- Nécessite que les fichiers vidéo soient présents localement
- Pas de téléchargement automatique pendant l'entraînement

**Donc:**
1. **Avant entraînement:** Télécharger les vidéos (prepare.py ou download_data.py)
2. **Pendant entraînement:** Streaming depuis le disque

---

## Workflow Auto-Research (Sur Cloud)

### 1. Se connecter à l'instance

```bash
ssh -p 15212 root@118.163.199.123
cd ~/vl-jepa
git pull origin main  # Synchroniser
```

### 2. Modifier les hyperparamètres

```bash
nano autoresearch/train.py
```

Modifier uniquement la section "Hyperparameters" (lignes 35-55):
```python
LEARNING_RATE = 3e-4    # Tester: 1e-5, 3e-5, 1e-4, 3e-4, 1e-3
BATCH_SIZE = 2          # Tester: 2, 4, 8
TEMPERATURE = 0.1       # Tester: 0.03, 0.05, 0.07, 0.1, 0.15
SIGREG_WEIGHT = 0.2     # Tester: 0.0, 0.05, 0.1, 0.2, 0.5
```

### 3. Commit et lancer

```bash
git add autoresearch/train.py
git commit -m "exp: lr=3e-4, temp=0.1, sigreg=0.2"

bash autoresearch/run.sh
```

### 4. Analyser résultat

```bash
grep "^val_loss:" autoresearch/run.log
# Sortie: val_loss: 1.023456
```

### 5. Décider

**Si val_loss < baseline (1.101749):**
```bash
# Garder le commit (expérience réussie)
echo "abc1234	1.023456	18.5	keep	lr=3e-4,temp=0.1" >> autoresearch/results.tsv
```

**Si val_loss >= baseline:**
```bash
# Abandonner (pas d'amélioration)
git reset --hard HEAD~1
```

### 6. Itérer

Recommencer depuis l'étape 2 avec d'autres hyperparamètres.

---

## Workflow Training Complet

Une fois les meilleurs hyperparamètres trouvés:

```bash
# 1. Copier les meilleurs params
cp autoresearch/train.py training/train.py
nano training/train.py  # Ajuster pour training complet
# - MAX_EPOCHS = 20
# - MAX_TRAIN_SAMPLES = 0 (toutes les données)
# - WARMUP_STEPS = 500

# 2. Lancer
git add training/train.py
git commit -m "feat: full training with best params"
bash scripts/cloud_train.sh

# 3. Checkpoints sauvegardés dans checkpoints/ et W&B
```

---

## Commandes Essentielles

```bash
# Voir GPU
nvidia-smi

# Voir logs en temps réel
tail -f autoresearch/run.log
tail -f training/run.log

# Voir résultats
cat autoresearch/results.tsv

# Lancer autoresearch
bash autoresearch/run.sh

# Lancer training
bash scripts/cloud_train.sh

# Lancer évaluation
bash scripts/cloud_eval.sh checkpoints/best_e20.pt

# Setup instance (si nouvelle instance)
bash scripts/setup_instance.sh
```

---

## Hyperparamètres à Explorer

| Paramètre | Baseline | Valeurs à tester |
|-----------|----------|------------------|
| LEARNING_RATE | 1e-4 | 1e-5, 3e-5, 3e-4, 1e-3 |
| BATCH_SIZE | 2 | 2, 4, 8 (selon VRAM) |
| TEMPERATURE | 0.07 | 0.03, 0.05, 0.1, 0.15 |
| SIGREG_WEIGHT | 0.1 | 0.0, 0.05, 0.2, 0.5 |
| WARMUP_STEPS | 100 | 50, 200, 500 |
| WEIGHT_DECAY | 0.05 | 0.0, 0.01, 0.1 |

---

## Règles Importantes

1. **Modifier UNIQUEMENT** `autoresearch/train.py` pour expérimenter
2. **JAMAIS** modifier `vljepa/` (code modèle)
3. **Un commit = une expérience** (pour rollback facile)
4. **NE PAS** commiter: `data/`, `checkpoints/`, `.env`, `results.tsv`
5. **TOUJOURS** utiliser `uv run`, jamais `python` directement
6. **5 minutes max** par expérience autoresearch

---

## Monitoring

- **Autoresearch W&B**: https://wandb.ai/maxence-cabiddu-maxence-cabiddu/vl-jepa-autoresearch
- **Training W&B**: https://wandb.ai/maxence-cabiddu-maxence-cabiddu/vl-jepa
- **Baseline**: val_loss = 1.101749 (à battre)

---

## Fichiers Importants à Connaître

| Fichier | Rôle | Modifier? |
|---------|------|-----------|
| `autoresearch/train.py` | Expérimentations | ✅ Oui |
| `training/train.py` | Training complet | ✅ Oui (rarement) |
| `vljepa/*.py` | Code modèle | ❌ Non |
| `FILE_MANIFEST.md` | Doc fichiers | Non (lecture) |
| `program.md` | Instructions agents | Non (lecture) |

---

## Instance Cloud Actuelle

- **ID**: 33130021
- **IP**: 118.163.199.123:15212
- **GPU**: RTX 6000 Ada 48GB
- **Status**: Active
- **Données**: Déjà présentes (16GB vidéos + annotations)

---

**Dernière mise à jour**: 2026-03-19
**Version**: 2.0 (restructure)
