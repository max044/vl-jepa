#!/bin/bash
# Quick launch script for Vast.ai

set -e

# Load env variables
export VASTAI_API_KEY=$(grep VASTAI_API_KEY .env | cut -d= -f2)
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)

echo "=================================="
echo "VL-JEPA Cloud Launch"
echo "=================================="

# Find cheapest RTX 4090
echo "Searching for cheapest RTX 4090..."
OFFER=$(uv run vastai search offers "gpu_name=RTX_4090" "dph <= 0.60" -o dph 2>&1 | head -2 | tail -1)

if [ -z "$OFFER" ]; then
    echo "No RTX 4090 found, trying RTX 3090..."
    OFFER=$(uv run vastai search offers "gpu_name=RTX_3090" "dph <= 0.40" -o dph 2>&1 | head -2 | tail -1)
fi

if [ -z "$OFFER" ]; then
    echo "No suitable GPU found!"
    exit 1
fi

INSTANCE_ID=$(echo $OFFER | awk '{print $1}')
PRICE=$(echo $OFFER | awk '{print $10}')

echo "Found instance: $INSTANCE_ID at $PRICE/hour"
echo ""

# Ask for confirmation
read -p "Launch instance? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create instance
echo "Creating instance..."
uv run vastai create instance $INSTANCE_ID \
    --disk 50 \
    --image vastai/pytorch:latest \
    --env "HF_TOKEN=$HF_TOKEN" \
    --onstart-cmd "#!/bin/bash
set -e
echo '=== Setup ==='
apt-get update -qq && apt-get install -y -qq git curl
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers accelerate huggingface-hub datasets sentence-transformers
pip install -q opencv-python-headless timm wandb
pip install -q 'huggingface-hub>=0.24.0'

if [ -n \"\$HF_TOKEN\" ]; then
    huggingface-cli login --token \"\$HF_TOKEN\"
fi

mkdir -p ~/data/Charades_v1_480
cd ~/data
hf sync hf://buckets/max044/charades-sta-storage/Charades_v1_480 Charades_v1_480 --progress || echo 'Using local mode'

cd ~
git clone https://github.com/max044/vl-jepa.git || (cd vl-jepa && git pull)
cd vl-jepa
mkdir -p autoresearch/results
echo '=== Ready ==='"

echo ""
echo "Instance created! Check with: uv run vastai show instances"
