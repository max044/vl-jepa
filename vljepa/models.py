"""VL-JEPA model components: V-JEPA 2 (X-Encoder), Qwen 2.5 (Predictor), MiniLM (Y-Encoder)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer
import numpy as np

from vljepa.config import Config


class XEncoder(nn.Module):
    """Frozen V-JEPA 2 Video Encoder.

    Extracts hierarchical video features.
    """

    def __init__(self, config: Config):
        super().__init__()
        # Load V-JEPA 2 model
        try:
            self.model = AutoModel.from_pretrained(config.clip_model, trust_remote_code=True)
        except Exception:
            print(f"Warning: Failed to load {config.clip_model}. Trying fallback 'facebook/vjepa-vit-h-14-224'.")
            self.model = AutoModel.from_pretrained("facebook/vjepa-vit-h-14-224", trust_remote_code=True)
            config.x_dim = self.model.config.hidden_size

        # Freeze
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # Move to device if needed
        self.model.to(config.device)

        self.hidden_size = config.x_dim

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode video frames.

        Args:
            pixel_values: (B, C, T, H, W) preprocessed frames (0-1 float, normalized)
        """
        if pixel_values.shape[1] == 3 and pixel_values.shape[2] > 3:
             # (B, C, T, H, W) -> (B, T, C, H, W)
             pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        try:
            outputs = self.model(pixel_values_videos=pixel_values)
        except TypeError:
             # Fallback
             outputs = self.model(pixel_values=pixel_values)

        last_hidden = outputs.last_hidden_state # (B, seq_len, hidden)
        sv = last_hidden.mean(dim=1) # (B, hidden)
        return sv

    def preprocess_frames(self, frames_batch: list[list], device: str = "cpu") -> torch.Tensor:
        """Preprocess frames."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1)

        padded = []
        for frames in frames_batch:
             if len(frames) == 0:
                 t = torch.zeros((16, 3, 224, 224), device=device)
                 padded.append(t)
                 continue

             # Stack to (T, H, W, 3)
             t = torch.tensor(np.stack(frames), dtype=torch.float32, device=device)
             
             # Permute to (T, 3, H, W)
             t = t.permute(0, 3, 1, 2) / 255.0
             
             # Resize
             t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
             
             padded.append(t)

        max_t = max((t.size(0) for t in padded), default=16)
        final_padded = []
        for t in padded:
             if t.size(0) < max_t:
                 pad = t[-1:].expand(max_t - t.size(0), -1, -1, -1)
                 t = torch.cat([t, pad], dim=0)
             final_padded.append(t)
        
        # Stack -> (B, T, 3, H, W)
        pixel_values = torch.stack(final_padded, dim=0) 
        
        # Input to V-JEPA 2 (via HF) usually expects (B, T, C, H, W)
        
        # Normalize (broadcasting T)
        # mean/std are (1, 3, 1, 1, 1). We need to align with (B, T, 3, H, W)
        # Permute to (B, 3, T, H, W) for normalization
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        pixel_values = (pixel_values - mean) / std
        
        # Permute back to (B, T, 3, H, W)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        
        return pixel_values


class QueryEncoder(nn.Module):
    """Tokenizer for Qwen."""

    def __init__(self, config: Config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.predictor_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(self, texts: list[str], device: str = "cpu") -> dict:
        return self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=64
        ).to(device)


class Predictor(nn.Module):
    """Qwen 2.5 0.5B Predictor with LoRA."""

    def __init__(self, config: Config):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            config.predictor_model,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        if config.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        self.visual_proj = nn.Linear(config.x_dim, config.predictor_dim)
        self.output_proj = nn.Linear(config.predictor_dim, config.embed_dim)

        # Move to device
        self.to(config.device)

    def forward(self, sv: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B = sv.size(0)
        sv_embeds = self.visual_proj(sv).unsqueeze(1) # (B, 1, predictor_dim)
        
        if hasattr(self.model, "base_model"):
             base = self.model.base_model.model
        else:
             base = self.model
             
        # Qwen2 uses model.embed_tokens
        # We try to access it via property or direct module
        if hasattr(base, "model"):
             embed_layer = base.model.embed_tokens
        elif hasattr(base, "embed_tokens"):
             embed_layer = base.embed_tokens
        else:
             # General fallback for AutoModel
             embed_layer = base.get_input_embeddings()

        inputs_embeds = embed_layer(input_ids)
        combined_embeds = torch.cat([sv_embeds, inputs_embeds], dim=1)
        
        ones = torch.ones((B, 1), device=sv.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([ones, attention_mask], dim=1)
        
        outputs = self.model(inputs_embeds=combined_embeds, attention_mask=combined_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        
        return self.output_proj(last_hidden)


class YEncoder(nn.Module):
    """Frozen MiniLM Y-Encoder."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.model = SentenceTransformer(config.text_model)
        self.projection = nn.Linear(config.text_dim, config.embed_dim)
        
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, texts: list[str], device: str = "cpu") -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True, device=device)
        # Clone to avoid "Inference tensors cannot be saved for backward" error
        return self.projection(embeddings.clone())


class VLJepa(nn.Module):
    """V-JEPA 2 + Qwen 2.5 + MiniLM."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.x_encoder = XEncoder(config)
        self.query_encoder = QueryEncoder(config)
        self.predictor = Predictor(config)
        self.y_encoder = YEncoder(config)

    def forward(self, pixel_values, query_ids, query_mask, target_texts):
        sv = self.x_encoder(pixel_values)
        sy_hat = self.predictor(sv, query_ids, query_mask)
        sy = self.y_encoder(target_texts, device=str(pixel_values.device))
        return sy_hat, sy

    def encode_video_query(self, pixel_values, query_ids, query_mask):
        sv = self.x_encoder(pixel_values)
        sy_hat = self.predictor(sv, query_ids, query_mask)
        return sy_hat

    def encode_text(self, texts, device="cpu"):
        return self.y_encoder(texts, device=device)

    def trainable_parameters(self):
        return list(self.predictor.parameters()) + list(self.y_encoder.projection.parameters())

    def count_parameters(self):
        def _count(m):
            return {
                "total": sum(p.numel() for p in m.parameters()),
                "trainable": sum(p.numel() for p in m.parameters() if p.requires_grad)
            }
        return {
            "x_encoder": _count(self.x_encoder),
            "predictor": _count(self.predictor),
            "y_encoder": _count(self.y_encoder)
        }
