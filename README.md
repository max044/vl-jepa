# VL-JEPA: Simplified Video-Language Alignment

A simplified implementation of the Video-Language Joint Embedding Predictive
Architecture (VL-JEPA) for **Temporal Moment Retrieval** (Temporal Grounding).

This project uses **V-JEPA 2** for video understanding and **Qwen 2.5 0.5B** as
a predictor to align video features with language queries in a high-dimensional
embedding space.

## 🚀 Architecture

The model follows the JEPA framework by aligning video features (X) and text
descriptions (Y) through a predictor (P):

- **X-Encoder (Video)**: Frozen **V-JEPA 2** (ViT-L). High-fidelity hierarchical
  video features.
- **Y-Encoder (Text)**: Frozen **MiniLM** (all-MiniLM-L6-v2). Compact and
  efficient semantic text embeddings.
- **Predictor (Alignment)**: **Qwen 2.5 0.5B** with **LoRA** (Low-Rank
  Adaptation). Learns to predict the target text embedding from the joint
  video+query representation.

## 🛠️ Installation

This project uses `uv` for lightning-fast dependency management.

```bash
# Clone the repository
git clone https://github.com/max044/vl-jepa.git
cd vl-jepa

# Create environment and install dependencies
uv sync
```

## 📊 Data Preparation

The model is trained on the **Charades-STA** dataset for temporal grounding.

1. **Videos**: Download
   [Charades v1](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip)
   and place them in `data/Charades_v1_480`.
2. **Annotations**: Use `download_annotations.py` to download the annotations.

Structure:

```text
data/
├── Charades_v1_480/      # Video files (.mp4)
├── charades_sta_train.txt
└── charades_sta_test.txt
```

## 🏋️ Training

Start training with default hyperparameters:

```bash
# Regular training
uv run train.py

# Debug mode (small subset, only 2 epochs)
uv run train.py --debug --device mps
```

### Key Training Features:

- **Bidirectional InfoNCE Loss**: Maximizes mutual information between predicted
  and target embeddings.
- **LoRA Tuning**: Only 0.2% of the predictor parameters (Qwen) are trained,
  making it extremely memory-efficient.
- **MPS Support**: Optimized for Mac M1/M2/M3 chips.

## 🔍 Inference (Moment Retrieval)

Once trained, you can use the model to find specific moments in a video based on
a text query. The script uses a sliding window approach with NMS to find the
best matching segments.

```bash
# Example: Local inference
uv run infer.py \
    --video data/Charades_v1_480/3MSZA.mp4 \
    --query "person turns on the light" \
    --checkpoint checkpoints/best.pth \
    --device mps
```

## 🔍 Implementation Details

Unlike standard VLM (Visual-Language Models) that use generative heads, this
VL-JEPA implementation focuses on **embedding alignment**. This makes it an
order of magnitude faster for retrieval tasks (search) as embeddings can be
pre-computed and indexed using vector databases (Faiss, Milvus, Chroma).

## 📚 References

This implementation is based on the official VL-JEPA paper:

```bibtex
@misc{chen2026vljepajointembeddingpredictive,
      title={VL-JEPA: Joint Embedding Predictive Architecture for Vision-language}, 
      author={Delong Chen and Mustafa Shukor and Theo Moutakanni and Willy Chung and Jade Yu and Tejaswi Kasarla and Yejin Bang and Allen Bolourchi and Yann LeCun and Pascale Fung},
      year={2026},
      eprint={2512.10942},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.10942}, 
}
```

## 📄 License

MIT
