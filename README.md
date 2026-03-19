# VL-JEPA: Video-Language Joint Embedding

Implémentation de VL-JEPA pour **Temporal Moment Retrieval** - trouver un moment précis dans une vidéo à partir d'une description textuelle.

## Architecture

- **X-Encoder** (frozen): V-JEPA 2 ViT-L - extraction de features vidéo
- **Predictor** (trainable): Qwen3.5-0.8B - prédit les embeddings texte
- **Y-Encoder** (trainable): Qwen3-Embedding-0.6B - encode les captions

## Structure du Repository

```
vl-jepa/
├── autoresearch/         # Expérimentations rapides (5 min)
│   ├── train.py         # Script d'entraînement time-budgeted
│   ├── prepare.py       # Préparation données
│   └── run.sh           # Lancer une expérience
├── training/            # Entraînement complet
│   ├── train.py         # Entraînement complet (20 epochs)
│   ├── eval.py          # Évaluation
│   └── download_data.py # Téléchargement données
├── scripts/             # Scripts cloud
│   ├── cloud_train.sh   # Lancer entraînement
│   ├── cloud_eval.sh    # Lancer évaluation
│   └── setup_instance.sh # Setup instance
├── vljepa/              # Code source modèle
│   ├── config.py
│   ├── models.py
│   ├── dataset.py
│   └── losses.py
├── FILE_MANIFEST.md     # Documentation fichiers
└── program.md           # Instructions agents
```

## Workflow

### 1. Autoresearch (Trouver les hyperparamètres)

Sur l'instance cloud:
```bash
ssh -p <port> root@<ip>
cd ~/vl-jepa

# Modifier les hyperparams
nano autoresearch/train.py  # Lignes 35-55

# Commit
git add autoresearch/train.py
git commit -m "exp: lr=3e-4, temp=0.1"

# Lancer
bash autoresearch/run.sh

# Noter résultat
echo "abc1234	1.05	18.2	keep	lr=3e-4" >> autoresearch/results.tsv
```

**Si val_loss s'améliore**: garder le commit  
**Si pas d'amélioration**: `git reset --hard HEAD~1`

### 2. Training Complet (Avec les meilleurs hyperparams)

```bash
# Copier les meilleurs params dans training/
nano training/train.py  # Modifier lignes 35-55

# Lancer
bash scripts/cloud_train.sh
```

Les checkpoints sont sauvegardés dans `checkpoints/` et sur W&B.

### 3. Évaluation

```bash
bash scripts/cloud_eval.sh checkpoints/best_e20.pt
```

## Hyperparamètres clés

Dans `train.py`, modifier:
- `LEARNING_RATE`: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
- `BATCH_SIZE`: [2, 4, 8]
- `TEMPERATURE`: [0.03, 0.05, 0.07, 0.1, 0.15]
- `SIGREG_WEIGHT`: [0.0, 0.05, 0.1, 0.2, 0.5]
- `WARMUP_STEPS`: [50, 100, 200, 500]

## Commandes essentielles

```bash
# Setup instance (une fois)
bash scripts/setup_instance.sh

# Autoresearch
bash autoresearch/run.sh

# Training
bash scripts/cloud_train.sh

# Évaluation
bash scripts/cloud_eval.sh checkpoints/best_e20.pt

# Voir résultats
grep "val_loss:" autoresearch/run.log
grep "val_loss:" training/run.log
```

## Monitoring

- **W&B Training**: https://wandb.ai/maxence-cabiddu-maxence-cabiddu/vl-jepa
- **W&B Autoresearch**: https://wandb.ai/maxence-cabiddu-maxence-cabiddu/vl-jepa-autoresearch

## Documentation

- `FILE_MANIFEST.md` - Description de chaque fichier
- `program.md` - Instructions pour agents autonomes
- `CLAUDE.md` - Contexte technique

## Règles

1. **Autoresearch**: Modifier UNIQUEMENT `autoresearch/train.py`
2. **Training**: Utiliser les hyperparams trouvés par autoresearch
3. **Model code**: Ne pas toucher `vljepa/` sauf changement architecture
4. **Commits**: Un commit = une expérience (pour rollback)
5. **PAS de git**: `data/`, `checkpoints/`, `.env`, `results.tsv`

## License

MIT
