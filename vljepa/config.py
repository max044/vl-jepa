"""Configuration for VL-JEPA training and inference."""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    """All hyperparameters and paths for VL-JEPA."""

    # ── Device ──────────────────────────────────────────────
    device: str = ""  # auto-detected if empty

    # ── Model ────────────────────────────────────────────────────────────
    # X-Encoder: V-JEPA 2 ViT-L (frozen, ~300M)
    clip_model: str = "facebook/vjepa2-vitl-fpc64-256"

    # Predictor: Qwen 2.5 0.5B (LoRA)
    predictor_model: str = "Qwen/Qwen2.5-0.5B"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Y-Encoder: MiniLM (frozen, ~22M)
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Embedding and model dimensions
    x_dim: int = 1024         # V-JEPA ViT-L output dim
    predictor_dim: int = 896  # Qwen 2.5 0.5B hidden dim
    text_dim: int = 384       # MiniLM-L6-v2 output dim
    embed_dim: int = 384      # Shared projection target

    # ── Video ────────────────────────────────────────────────────────────
    num_frames: int = 16
    frame_size: int = 224     # V-JEPA input resolution

    # ── Training ─────────────────────────────────────────────────────────
    batch_size: int = 4       # Start small (increase if GPU RAM allows)
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 20
    warmup_steps: int = 200
    grad_clip: float = 1.0

    # Loss
    temperature: float = 0.07
    sigreg_weight: float = 0.1

    # ── Data ────────────────────────────────────────────────
    data_dir: str = "./data"
    videos_dir: str = "./data/Charades_v1_480"
    anno_train: str = "./data/charades_sta_train.txt"
    anno_test: str = "./data/charades_sta_test.txt"

    # ── Checkpoints ─────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 2  # save checkpoint every N epochs

    # ── Inference ───────────────────────────────────────────
    window_sizes: list[float] = field(default_factory=lambda: [2.0, 4.0, 8.0, 16.0])
    window_stride: float = 1.0
    nms_threshold: float = 0.5
    top_k: int = 5

    # ── Debug ───────────────────────────────────────────────
    debug: bool = False
    debug_samples: int = 100
    num_workers: int = 0  # 0 for MPS compatibility

    def __post_init__(self):
        if not self.device:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Ensure directories exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
