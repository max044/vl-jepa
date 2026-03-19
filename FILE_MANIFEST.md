# VL-JEPA File Manifest

Ce document décrit l'organisation du repository et l'usage de chaque fichier.

## Structure du Repository

```
vl-jepa/
├── autoresearch/          # Expérimentations rapides (5 min) pour trouver les meilleurs hyperparamètres
├── training/              # Entraînement complet et évaluation
├── vljepa/               # Code source du modèle (shared)
├── scripts/              # Scripts utilitaires simples
├── configs/              # Configurations
├── data/                 # Données (non versionnées)
├── checkpoints/          # Checkpoints (non versionnés)
├── program.md            # Instructions pour les agents autonomes
├── FILE_MANIFEST.md      # Ce fichier
└── README.md             # Vue d'ensemble du projet
```

---

## Autoresearch (`autoresearch/`)

Objectif : Trouver rapidement les meilleurs hyperparamètres via des expériences de 5 minutes.

| Fichier | Description | Usage |
|---------|-------------|-------|
| `train.py` | Script d'entraînement time-budgeted (5 min) | **Modifier uniquement celui-ci** pour expérimenter |
| `prepare.py` | Prépare UN SOUS-ENSEMBLE de données (500 vidéos max) pour expériences rapides | Exécuter une fois : `uv run prepare.py --subset 500` |
| `run.sh` | Lance une expérience simple | `bash autoresearch/run.sh` |
| `program.md` | **Instructions pour les agents IA** - Comment faire les expériences | Lire par l'agent au démarrage |

**Note sur les données:**
- `prepare.py` télécharge seulement 500 vidéos (~800MB) pour aller vite
- Les données sont stockées dans `data/autoresearch/` (séparé de training)
- Sur le cloud: les données autoresearch sont déjà prêtes (lien symbolique vers les vidéos complètes)
| `results.tsv` | Track les résultats (non versionné) | Gitignore |

**Différence clé avec training:**
- **Autoresearch**: 500 vidéos max, 5 min d'entraînement, pour trouver les hyperparamètres
- **Training**: 9,848 vidéos, 20+ epochs, entraînement complet avec les meilleurs params

**Workflow:**
1. Modifier `train.py` (hyperparams en haut du fichier)
2. Commit : `git add autoresearch/train.py && git commit -m "exp: description"`
3. Lancer : `bash autoresearch/run.sh`
4. Noter résultat dans `results.tsv`
5. Si meilleur : garder le commit, sinon : `git reset --hard HEAD~1`

---

## Training Complet (`training/`)

Objectif : Entraînement complet sur toutes les données avec les meilleurs hyperparamètres trouvés.

| Fichier | Description | Usage |
|---------|-------------|-------|
| `train.py` | Entraînement complet (20+ epochs) | `uv run training/train.py` |
| `eval.py` | Évaluation sur test set | `uv run training/eval.py --checkpoint checkpoints/best.pt` |
| `download_data.py` | Télécharge TOUTES les vidéos (9,848 vidéos, ~15GB) | `uv run training/download_data.py` |

**Important sur les données:**
- `download_data.py` télécharge l'intégralité des 9,848 vidéos (~15GB) depuis Hugging Face Storage
- Stocké dans `data/` (séparé de `data/autoresearch/`)
- **Sur le cloud Vast.ai**: Les vidéos sont DÉJÀ présentes dans `data/Charades_v1_480/` (16GB) - NE PAS retélécharger
- Le dataset charge les frames à la volée (streaming) pendant l'entraînement

**Workflow:**
1. Configurer les hyperparamètres dans `training/train.py` (ceux trouvés par autoresearch)
2. Lancer l'entraînement : `uv run training/train.py`
3. Checkpoints sauvegardés automatiquement sur W&B et local

---

## Code Source Partagé (`vljepa/`)

**NE PAS MODIFIER** sauf si changement d'architecture majeur.

| Fichier | Description |
|---------|-------------|
| `config.py` | Configuration dataclass (Config) |
| `models.py` | Architecture VL-JEPA (X-Encoder, Predictor, Y-Encoder) |
| `dataset.py` | Dataset Charades-STA |
| `losses.py` | Losses InfoNCE et SIGReg |
| `utils.py` | Fonctions utilitaires |

---

## Scripts Utilitaires (`scripts/`)

Scripts simples pour le cloud.

| Fichier | Description |
|---------|-------------|
| `cloud_train.sh` | Lance l'entraînement sur l'instance cloud |
| `cloud_eval.sh` | Lance l'évaluation sur l'instance cloud |
| `setup_instance.sh` | Setup initial d'une nouvelle instance VastAI |

---

## Fichiers Racine

| Fichier | Description | À modifier ? |
|---------|-------------|--------------|
| `README.md` | Vue d'ensemble du projet | Oui (mise à jour) |
| `program.md` | Instructions pour agents autonomes | Rarement |
| `FILE_MANIFEST.md` | Ce fichier | Quand ajout/suppression de fichiers |
| `pyproject.toml` | Dépendances Python | Si nouvelle librairie |
| `uv.lock` | Lock des dépendances | Auto-généré |
| `.env.example` | Template variables d'environnement | Si nouvelles clés nécessaires |
| `.gitignore` | Fichiers ignorés par git | Si besoin |

---

## Fichiers obsolètes (à supprimer)

Ces fichiers sont obsolètes et seront supprimés :

- `train.py` (racine) → remplacé par `autoresearch/train.py` et `training/train.py`
- `train_full.py` (racine) → obsolète
- `eval.py` (racine) → déplacé vers `training/eval.py`
- `infer.py` (racine) → obsolète (pas utilisé)
- `download_annotations.py` (racine) → intégré dans `prepare.py`
- `autoresearch-macos/` → repo externe, à supprimer
- `scripts/cloud_autoresearch.py` → trop complexe
- `scripts/sweep*.sh` → obsolète
- `scripts/*_cloud.sh` multiples → à simplifier
- `configs/` → configuration maintenant dans les scripts

---

## Règles d'or

1. **Autoresearch** : Modifier UNIQUEMENT `autoresearch/train.py` pour expérimenter
2. **Training** : Utiliser les hyperparams trouvés par autoresearch
3. **Model code** : Ne pas toucher sauf changement architecture
4. **Commits** : Un commit = une expérience (pour pouvoir revenir en arrière)
5. **Documentation** : Mettre à jour ce fichier si ajout/suppression de fichiers

---

## Workflow Typique

### Phase 1: Autoresearch (sur cloud)
```bash
# Sur l'instance cloud
ssh -p <port> root@<ip>
cd ~/vl-jepa

# Modifier autoresearch/train.py
nano autoresearch/train.py  # Changer LR, temp, etc.

# Commit
git add autoresearch/train.py
git commit -m "exp: lr=3e-4, temp=0.1"

# Lancer
bash autoresearch/run.sh

# Noter résultat
echo "abc1234	1.05	18.2	keep	lr=3e-4" >> autoresearch/results.tsv
```

### Phase 2: Training complet (sur cloud)
```bash
# Copier les meilleurs hyperparams
cp autoresearch/train.py training/train.py

# Modifier pour training complet (epochs=20, all data)
nano training/train.py

# Lancer
git add training/train.py
git commit -m "feat: full training with best params"
bash scripts/cloud_train.sh
```

### Phase 3: Évaluation
```bash
bash scripts/cloud_eval.sh checkpoints/best.pt
```

---

## Checkpoints et W&B

- Les checkpoints sont sauvegardés dans `checkpoints/` (local) et sur W&B
- W&B project: `vl-jepa` (training) et `vl-jepa-autoresearch` (autoresearch)
- Lien W&B training: https://wandb.ai/maxence-cabiddu-maxence-cabiddu/vl-jepa
- Lien W&B autoresearch: https://wandb.ai/maxence-cabiddu-maxence-cabiddu/vl-jepa-autoresearch

---

## Notes

- **NE JAMAIS** commiter `data/`, `checkpoints/`, `.env`, `results.tsv`
- **TOUJOURS** utiliser `uv run` et non `python` directement
- **SUR LE CLOUD** : Les données sont dans `~/vl-jepa/data/` (déjà téléchargées)
- **METTRE À JOUR** ce fichier si ajout de nouveaux fichiers importants
