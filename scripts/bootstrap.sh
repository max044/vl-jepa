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

# ── 6. W&B login ───────────────────────────────────────────────
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "▸ Logging into W&B..."
    uv run wandb login "$WANDB_API_KEY" 2>/dev/null || true
    echo "  ✓ W&B configured"
else
    echo "⚠  WANDB_API_KEY not set. Set it in .env to enable experiment tracking."
fi

# ── 7. Download annotations (if not present) ───────────────────
if [ ! -f data/charades_sta_train.txt ]; then
    echo ""
    echo "▸ Downloading Charades-STA annotations..."
    uv run python download_annotations.py
else
    echo "▸ Annotations already present."
fi

# ── 8. Check for video data ────────────────────────────────────
if [ ! -d data/Charades_v1_480 ] || [ -z "$(ls -A data/Charades_v1_480 2>/dev/null)" ]; then
    echo ""
    echo "⚠  Videos not found at data/Charades_v1_480/"
    echo "   Download from: https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
    echo "   Then extract to data/Charades_v1_480/"
    echo ""
    echo "   Quick command:"
    echo "   wget -P data/ https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
    echo "   unzip data/Charades_v1_480.zip -d data/"
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
