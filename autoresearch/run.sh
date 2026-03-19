#!/bin/bash
# Script simple pour lancer une expérience autoresearch sur le cloud
# Usage: bash autoresearch/run.sh

set -e

echo "=== VL-JEPA Autoresearch ==="
echo "Lancement de l'expérience..."
echo ""

# Vérifier qu'on est sur une instance cloud ou local avec GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU disponible:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "⚠ Pas de GPU détecté - entraînement sera lent"
fi

echo ""
echo "Démarrage de l'entraînement (5 minutes)..."
echo ""

# Lancer l'entraînement avec UV
cd "$(dirname "$0")/.."
uv run autoresearch/train.py 2>&1 | tee autoresearch/run.log

echo ""
echo "=== Expérience terminée ==="
echo "Résultats dans: autoresearch/run.log"
echo ""
echo "Pour voir le val_loss:"
echo "  grep '^val_loss:' autoresearch/run.log"
