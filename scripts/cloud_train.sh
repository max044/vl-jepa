#!/bin/bash
# Script simple pour lancer l'entraînement complet sur le cloud
# Usage: bash scripts/cloud_train.sh

set -e

echo "=== VL-JEPA Training Complet ==="
echo ""

# Vérifier GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Pas de GPU détecté - entraînement impossible"
    exit 1
fi

echo "✓ GPU disponible:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo ""

# Vérifier que les données sont présentes
if [ ! -d "data/Charades_v1_480" ]; then
    echo "❌ Données non trouvées dans data/Charades_v1_480"
    echo "Exécutez d'abord: uv run training/download_data.py"
    exit 1
fi

echo "✓ Données trouvées"
echo ""

# Lancer l'entraînement
echo "Démarrage de l'entraînement complet..."
echo ""

uv run training/train.py 2>&1 | tee training/run.log

echo ""
echo "=== Entraînement terminé ==="
echo ""
echo "Checkpoints sauvegardés dans: checkpoints/"
echo "Logs: training/run.log"
echo ""
echo "Pour évaluer le meilleur modèle:"
echo "  bash scripts/cloud_eval.sh checkpoints/best_e*.pt"
