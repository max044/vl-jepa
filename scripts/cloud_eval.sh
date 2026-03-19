#!/bin/bash
# Script simple pour lancer l'évaluation sur le cloud
# Usage: bash scripts/cloud_eval.sh [checkpoint_path]

CHECKPOINT="${1:-checkpoints/best_e*.pt}"

set -e

echo "=== VL-JEPA Évaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo ""

# Vérifier que le checkpoint existe
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint non trouvé: $CHECKPOINT"
    echo ""
    echo "Checkpoints disponibles:"
    ls -lh checkpoints/*.pt 2>/dev/null || echo "  Aucun"
    exit 1
fi

echo "✓ Checkpoint trouvé"
echo ""

# Lancer l'évaluation
echo "Démarrage de l'évaluation..."
echo ""

uv run training/eval.py --checkpoint "$CHECKPOINT" 2>&1 | tee training/eval.log

echo ""
echo "=== Évaluation terminée ==="
echo "Résultats dans: training/eval.log"
