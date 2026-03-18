# 🎥 VL-JEPA: Fast Video-Language Retrieval

A streamlined implementation of **Video-Language Joint Embedding Predictive Architecture** (VL-JEPA) for **Temporal Moment Retrieval**.

Instead of "describing" videos (generative), this model learns to **align** video segments with text in a shared embedding space. This makes searching through hours of video nearly instantaneous.

---

## 🧠 Architecture
- **Vision (X)**: Frozen `V-JEPA 2` (ViT-L).
- **Text (Y)**: Frozen `MiniLM` (all-MiniLM-L6-v2).
- **Alignment**: `Qwen 2.5 0.5B` tuned with **LoRA** (predicts text embeddings from video features).

---

## ⚡ Quick Start (Local)

1. **Install** (requires [uv](https://astral.sh/uv)):
   ```bash
   git clone https://github.com/max044/vl-jepa.git
   cd vl-jepa
   uv sync
   ```

2. **Prepare Data**:
   ```bash
   uv run download_annotations.py
   # Place your Charades videos in data/Charades_v1_480/
   ```

3. **Train**:
   ```bash
   uv run train.py --device mps # or cuda
   ```

---

## ☁️ Cloud GPU Training (Vast.ai)

### 🚀 Automated Method (Recommended)
We provide an automated launcher using the **Vast.ai Python SDK**. This script will find the cheapest GPU, launch it, sync your code/.env, bootstrap the environment, and **start training automatically**.

1. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Add your VASTAI_API_KEY, WANDB_API_KEY and HF_TOKEN to .env
   ```

2. **Launch & Train**:
   ```bash
   uv run python scripts/vast_launcher.py --gpu "RTX 4090"
   ```
   *The script will automatically start training. You can monitor progress with the provided `tail -f` command or via SSH.*

### 🎛️ Advanced Launcher Options
The `vast_launcher.py` script is highly flexible. You can use it to run evaluations, sweeps, or just prepare an instance.

- **Run a specific script** (e.g. an evaluation or sweep):
  ```bash
  uv run python scripts/vast_launcher.py --script "scripts/sweep.sh"
  ```
- **Skip dataset download** (useful if you stream data):
  ```bash
  uv run python scripts/vast_launcher.py --no-dataset
  ```
- **Prepare instance only** (bootstrap without running any script):
  ```bash
  uv run python scripts/vast_launcher.py --no-run
  ```

### 🛠️ Manual Method (Alternative)
If you prefer to set up the instance yourself on Vast.ai or RunPod:

1. **Initialize Instance**:
   Run this on your fresh GPU instance:
   ```bash
   curl -sSL https://raw.githubusercontent.com/max044/vl-jepa/main/scripts/bootstrap.sh | bash
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   nano .env  # Add WANDB_API_KEY and HF_TOKEN
   ```

3. **Run Training**:
   ```bash
   uv run download_annotations.py
   bash scripts/train_cloud.sh
   ```

---

## 🧪 Evaluation & Inference

Once training is finished, you can evaluate your model on the test set:

```bash
# Replace ID with your W&B run ID (e.g. 1a2b3c4d)
CHECKPOINT="max044/vl-jepa/model-ID:best" bash scripts/eval_cloud.sh
```

To run inference on a single video, use:
```bash
uv run infer.py --checkpoint path/to/model.pt --video data/test.mp4 --query "person opening door"
```

---

## 🔍 How it Works

1. **Training**: The model takes a video segment and its description. It uses **InfoNCE loss** to push the "correct" pairs together in the embedding space.
2. **Inference**: To find a moment (e.g., *"person opening a door"*):
    - We slide windows of various sizes (2s, 4s, 8s, 16s) across the video.
    - We compare each window's embedding to the query embedding.
    - We return the windows with the highest similarity scores.

---

## 📊 Monitoring
- **W&B**: Every run logs loss curves, GPU usage, and uploads checkpoints as tagged artifacts (`best`, `latest`).
- **Early Stopping**: Automated based on `val/loss`.
- **Metrics**: We use **mIoU** and **Recall@1** to measure how accurately the model finds the ground-truth timestamps.

---

## 📄 License
MIT
