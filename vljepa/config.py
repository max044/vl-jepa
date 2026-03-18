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

    # Predictor: Qwen 3.5 0.8B (full fine-tune, no LoRA) - based on VL-JEPA paper
    # Using Qwen3.5-0.8B as it's newer and closer to Llama-3.2-1B in size
    predictor_model: str = "Qwen/Qwen3.5-0.8B"
    use_lora: bool = False  # No LoRA - full fine-tune as per paper
    predictor_layers: int = 8  # Last 8 layers as per paper (Section 3.1)
    use_bidirectional_attention: bool = True  # Disable causal mask as per paper

    # Y-Encoder: Qwen3-Embedding-0.6B (trainable) - better than EmbeddingGemma per ablation
    text_model: str = "Qwen/Qwen3-Embedding-0.6B"
    y_encoder_lr_multiplier: float = 0.05  # LR multiplier for Y-Encoder as per paper

    # Embedding and model dimensions (from paper)
    x_dim: int = 1024              # V-JEPA ViT-L output dim
    predictor_dim: int = 896       # Qwen hidden dim
    text_dim: int = 1024           # Qwen3-Embedding-0.6B hidden_size
    embed_dim: int = 1536          # Shared embedding space (as per paper)

    # ── Video ────────────────────────────────────────────────────────────
    num_frames: int = 16
    frame_size: int = 224     # V-JEPA input resolution

    # ── Training ─────────────────────────────────────────────────────────
    batch_size: int = 2       # Reduced due to larger model (Qwen3.5-0.8B)
    grad_accumulation: int = 2  # Effective batch = 4
    lr: float = 1e-4          # Lower LR for larger model
    weight_decay: float = 0.05
    epochs: int = 20
    warmup_steps: int = 500
    grad_clip: float = 1.0
    dtype: str = "bf16"       # BF16 for better numerical stability

    # Loss
    temperature: float = 0.07
    sigreg_weight: float = 0.1  # SIGReg weight - penalizes representation collapse

    # ── Data ────────────────────────────────────────────────
    data_dir: str = "./data"
    videos_dir: str = "./data/Charades_v1_480"
    anno_train: str = "./data/charades_sta_train.txt"
    anno_test: str = "./data/charades_sta_test.txt"
    val_split: float = 0.1  # % of training data to use for validation

    # HF Storage (XET) - faster alternative to dataset for cloud training
    use_hf_storage: bool = False  # Use HF dataset instead (bucket not accessible)
    hf_dataset_id: str = "max044/Charades_v1_480"  # HF Dataset for videos

    # ── Checkpoints ─────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 2  # save checkpoint every N epochs
    val_every: int = 2   # run validation every N epochs
    val_samples: int = 500  # limit validation samples for speed
    early_stopping_patience: int = 5  # Stop if no validation improvement for N epochs (-1 to disable)

    # ── Inference ───────────────────────────────────────────
    window_sizes: list[float] = field(default_factory=lambda: [2.0, 4.0, 8.0, 16.0])
    window_stride: float = 1.0
    nms_threshold: float = 0.5
    inference_batch_size: int = 32  # Batch size for sliding window proposals
    top_k: int = 5

    # ── Model Improvements (Optional) ────────────────────
    use_regression: bool = False
    regression_loss_weight: float = 1.0
    use_learnable_temp: bool = False
    
    # ── Query Tokenization ───────────────────────────────
    query_max_length: int = 512  # Max query tokens (as per paper)

    # ── Debug ───────────────────────────────────────────────
    debug: bool = False
    debug_samples: int = 100
    num_workers: int = 0  # 0 for MPS compatibility

    def auto_detect(self):
        """Auto-detect device if empty."""
        if not self.device:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        if self.device == "mps" and self.dtype == "bf16":
            self.dtype = "fp16"
        return self

    def __post_init__(self):
        self.auto_detect()
        # Ensure directories exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, data: dict):
        """Create a Config from a dictionary, filtering unknown keys."""
        valid_keys = {f.name for f in field_dict(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data).auto_detect()

    def apply_overrides(self, overrides: list[str]):
        """Apply CLI overrides in format 'key=value'."""
        for override in overrides:
            if "=" not in override:
                continue
            key, value = override.split("=", 1)
            if not hasattr(self, key):
                print(f"Warning: Unknown config key '{key}'")
                continue
            
            # Cast to the correct type
            current_val = getattr(self, key)
            if isinstance(current_val, bool):
                new_val = value.lower() in ("true", "1", "yes")
            elif isinstance(current_val, list):
                # Simple list casting (comma-separated or single value)
                if "," in value:
                    new_val = [type(current_val[0])(v.strip()) for v in value.split(",")]
                else:
                    new_val = [type(current_val[0])(value)]
            else:
                new_val = type(current_val)(value)
            
            setattr(self, key, new_val)
        return self.auto_detect()

from dataclasses import fields as field_dict
