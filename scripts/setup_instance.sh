#!/bin/bash
# Setup initial d'une instance VastAI pour VL-JEPA
# À exécuter une fois sur une nouvelle instance

set -e

echo "=== Setup Instance VL-JEPA ==="
echo ""

# Vérifier qu'on est bien sur l'instance
if [ ! -d "/root/vl-jepa" ]; then
    echo "❌ /root/vl-jepa non trouvé - êtes-vous sur l'instance cloud?"
    exit 1
fi

cd /root/vl-jepa

echo "1. Vérification des données..."
if [ ! -d "data/Charades_v1_480" ] || [ -z "$(ls -A data/Charades_v1_480/*.mp4 2>/dev/null | head -1)" ]; then
    echo "   ❌ Vidéos non trouvées"
    echo "   Téléchargement des vidéos..."
    # Les vidéos devraient déjà être là, sinon on peut les télécharger
    echo "   (Les vidéos devraient être déjà présentes sur l'instance)"
else
    echo "   ✓ Vidéos trouvées: $(ls data/Charades_v1_480/*.mp4 | wc -l) fichiers"
fi

echo ""
echo "2. Préparation autoresearch..."
if [ ! -d "data/autoresearch" ]; then
    mkdir -p data/autoresearch
    ln -sf /root/vl-jepa/data/Charades_v1_480 data/autoresearch/Charades_v1_480
    cp data/charades_sta_train.txt data/autoresearch/ 2>/dev/null || echo "   Annotations déjà présentes"
    cp data/charades_sta_test.txt data/autoresearch/ 2>/dev/null || echo "   Annotations test déjà présentes"
fi
echo "   ✓ Autoresearch prêt"

echo ""
echo "3. Vérification W&B..."
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "   ✓ Variables d'environnement chargées"
else
    echo "   ⚠ Fichier .env non trouvé - créez-le avec WANDB_API_KEY"
fi

echo ""
echo "4. Test de l'environnement..."
if command -v uv &> /dev/null; then
    echo "   ✓ UV installé"
else
    echo "   ⚠ UV non trouvé - installation..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo ""
echo "=== Setup terminé ==="
echo ""
echo "Prochaines étapes:"
echo "  1. Lancer autoresearch: bash autoresearch/run.sh"
echo "  2. Ou lancer training:   bash scripts/cloud_train.sh"
echo ""
