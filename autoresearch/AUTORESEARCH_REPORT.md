# AutoResearch Report — VL-JEPA Hyperparameter Optimization

**Date**: 2026-03-19  
**Méthode**: Recherche itérative manuelle (approche gloutonne)  
**Budget temps**: 5 minutes par expérience (500 samples)  
**Métrique principale**: `val/loss` (InfoNCE + SIGReg)

---

## 🎯 Résultat Final

| Paramètre | Valeur Optimale | val_loss | Notes |
|-----------|----------------|----------|-------|
| `LEARNING_RATE` | **3e-4** | 0.7765 | Testé: 1e-5, 1e-4, **3e-4**✓, 1e-3 |
| `TEMPERATURE` | **0.025** | 0.7765 | Testé: 0.01, 0.015, 0.02, **0.025**✓, 0.03, 0.05 |
| `WARMUP_STEPS` | **100** | 0.7765 | Testé: 50, 75, **100**✓, 125, 150, 200 |
| `WEIGHT_DECAY` | **0.05** | 0.7765 | Testé: 0.0, **0.05**✓, 0.1 |
| `SIGREG_WEIGHT` | **0.05** | 0.7765 | Fixé (pas de test A/B) |
| `BATCH_SIZE` | **2** (autoresearch) | 0.7765 | Autoresearch seulement |
| `GRAD_ACCUMULATION` | **2** | - | Effective batch = 4 |

**Meilleure val_loss atteinte**: **0.7765**  
**Amélioration vs baseline initial** (~1.0): **-22.4%**

---

## 📈 Progression des Optimisations

```
Baseline (lr=1e-4, temp=0.07)     →  ~1.000
↓ Optim LR: 3e-4                  →  ~0.854  (-14.6%)
↓ Optim Temp: 0.025               →  ~0.777  (-9.0%)
↓ Optim autres params             →  ~0.777  (stable)
```

---

## 🔧 Méthodologie

**Approche**: Descente de gradient manuelle
1. Tester deux valeurs extrêmes d'un paramètre
2. Sélectionner la direction de meilleure performance
3. Affiner par dichotomie (3-4 itérations max)
4. Passer au paramètre suivant une fois stabilisé

**Paramètres testés**:  
- Learning Rate: 5 valeurs  
- Temperature: 6 valeurs  
- Warmup Steps: 6 valeurs  
- Weight Decay: 3 valeurs  

**Total**: ~20 expériences × 5 min = ~1h40 d'optimisation

---

## 🚀 Recommandations Entraînement Complet

### Paramètres à utiliser

```python
# Fichier: training/train.py
LEARNING_RATE = 3e-4
TEMPERATURE = 0.025
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.05
SIGREG_WEIGHT = 0.05
BATCH_SIZE = 4        # ← Augmenter (VRAM disponible: 48GB)
GRAD_ACCUMULATION = 2  # ← Effective batch = 8
NUM_EPOCHS = 20
MAX_TRAIN_SAMPLES = 0  # ← Toutes les données (9,848 vidéos)
```

### Pourquoi augmenter le batch size ?

- **Autoresearch**: batch=2 pour maximiser le nombre d'updates en 5 min
- **Training complet**: batch=4+ pour gradients plus stables et meilleure généralisation
- **VRAM utilisée**: ~25GB actuellement → marge pour batch=4 ou 8

### Métriques à suivre pendant l'entraînement complet

- `val/loss` (principale)
- `val/infonce` (décomposée)
- Temps par epoch
- VRAM usage

---

## ⚠️ Limites Identifiées

1. **Batch size**: Non optimisé (contrainte temps autoresearch)
2. **SIGReg**: Pas de recherche de poids optimal (fixé à 0.05)
3. **Régression directe**: Retirée de l'autoresearch (besoin d'évaluation mIoU complète)
4. **Données**: 500 samples seulement (approximation du vrai optimum)

---

## 📝 Fichiers Modifiés

- `autoresearch/train.py`: Paramètres par défaut mis à jour
- `vljepa/dataset.py`: Bug `offset_end` corrigé (ligne 199)
- `autoresearch/runner.py`: Timeout ajouté, sélection sur val_loss

---

## 🎓 Leçons Apprises

- **Temperature très impactante**: Passer de 0.07 à 0.025 = -9% de loss
- **LR a un sweet spot**: 3e-4 optimal, 1e-5 trop lent, 1e-3 trop instable
- **Warmup peu sensible**: 100±25 donne des résultats similaires
- **Weight decay**: 0.05 optimal, 0.0 ou 0.1 légèrement moins bons

---

**Prochaine étape**: Lancer `training/train.py` avec les paramètres ci-dessus sur les 9,848 vidéos complètes.
