#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# VL-JEPA Cloud Bootstrap Script
# Run this ONCE on a fresh GPU instance (Vast.ai / RunPod / Lambda)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║       VL-JEPA Cloud Bootstrap            ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. System info ──────────────────────────────────────────────
echo "▸ GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU detected)"
echo ""

# ── 2. Install uv (fast Python package manager) ────────────────
if ! command -v uv &> /dev/null; then
    echo "▸ Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "▸ uv already installed: $(uv --version)"
fi

# ── 3. Clone or update repo ────────────────────────────────────
REPO_DIR="${REPO_DIR:-$HOME/vl-jepa}"

if [ -d "$REPO_DIR" ]; then
    echo "▸ Repo exists at $REPO_DIR — pulling latest..."
    cd "$REPO_DIR"
    git pull --ff-only
else
    echo "▸ Cloning repo..."
    git clone https://github.com/max044/vl-jepa.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

# ── 4. Install Python dependencies ─────────────────────────────
echo ""
echo "▸ Installing dependencies with uv..."
uv sync

# ── 5. Setup environment variables ─────────────────────────────
if [ -f .env ]; then
    echo "▸ Loading .env..."
    set -a; source .env; set +a
else
    echo "⚠  No .env found. Create one from .env.example:"
    echo "   cp .env.example .env && nano .env"
fi

# ── 6. Logins (W&B, HF) ──────────────────────────────────────────
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "▸ Logging into W&B..."
    uv run wandb login "$WANDB_API_KEY" 2>/dev/null || true
    echo "  ✓ W&B configured"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    echo "▸ Logging into Hugging Face..."
    uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "  ✓ HF configured"
fi

# ── 7. Download Data from HF ──────────────────────────────────
# On utilise maintenant exclusivement le repo HF de l'utilisateur
# qui contient à la fois les vidéos et les annotations.
export HF_DATASET_ID="max044/Charades_v1_480"

echo ""
echo "▸ Downloading dataset from HF: $HF_DATASET_ID..."

# On télécharge tout dans le dossier data/
# Les fichiers .txt iront dans data/
# Le dossier Charades_v1_480/ iront dans data/Charades_v1_480/
uv run huggingface-cli download "$HF_DATASET_ID" --local-dir data --repo-type dataset

echo "✓ Dataset ready in data/"

# ── 8. Check for video data ────────────────────────────────────
if [ ! -d data/Charades_v1_480 ] || [ -z "$(ls -A data/Charades_v1_480 2>/dev/null)" ]; then
    echo ""
    echo "⚠  Videos not found at data/Charades_v1_480/ after HF download. Something went wrong."
    echo "   Please check the HF dataset or download manually."
else
    VIDEO_COUNT=$(ls data/Charades_v1_480/*.mp4 2>/dev/null | wc -l)
    echo "▸ Found $VIDEO_COUNT videos in data/Charades_v1_480/"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "✓ Bootstrap complete!"
echo ""
echo "Next steps:"
echo "  1. Ensure videos are in data/Charades_v1_480/"
echo "  2. Set WANDB_API_KEY in .env"
echo "  3. Run: bash scripts/train_cloud.sh"
echo "═══════════════════════════════════════════"
