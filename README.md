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

## ☁️ Cloud GPU Training (Vast.ai / RunPod)

We use **Lazy-Loading**: the training starts instantly. Videos are streamed from Hugging Face Hub only when needed.

1. **Initialize Instance**:
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

4. **Run Final Test**:
   ```bash
   # Replace ID with your W&B run ID
   CHECKPOINT="max044/vl-jepa/model-ID:best" bash scripts/eval_cloud.sh
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
