#!/bin/bash
"""
Setup script for cloud training instances.

This script downloads the complete Charades-STA dataset from HF Storage (XET)
to the local instance for fast training. Uses hf sync for efficient transfer.

Usage:
    bash scripts/setup_cloud_data.sh

Environment variables:
    HF_TOKEN: HuggingFace token (required for private bucket)
    DATA_DIR: Target directory (default: ./data)
"""

set -e

# Configuration
DATA_DIR="${DATA_DIR:-./data}"
BUCKET_URL="hf://buckets/max044/charades-sta-storage/Charades_v1_480"
VIDEOS_DIR="$DATA_DIR/Charades_v1_480"

echo "================================"
echo "Charades-STA Cloud Data Setup"
echo "================================"
echo "Target directory: $VIDEOS_DIR"
echo ""

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Private bucket access may fail."
    echo "Set it with: export HF_TOKEN=your_token"
fi

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$VIDEOS_DIR"

# Check if hf CLI is available
if ! command -v hf &> /dev/null; then
    echo "Installing HuggingFace CLI..."
    pip install -q huggingface-hub
fi

echo ""
echo "Downloading dataset from HF Storage (XET)..."
echo "This may take 10-30 minutes depending on your connection."
echo ""

# Sync from bucket to local
hf sync "$BUCKET_URL" "$VIDEOS_DIR" --progress

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "Videos: $(ls $VIDEOS_DIR/*.mp4 2>/dev/null | wc -l) files"
echo "Train annotations: $(wc -l < $VIDEOS_DIR/charades_sta_train.txt 2>/dev/null || echo 'Not found')"
echo "Test annotations: $(wc -l < $VIDEOS_DIR/charades_sta_test.txt 2>/dev/null || echo 'Not found')"
echo ""
echo "You can now run training with:"
echo "  python train.py"
