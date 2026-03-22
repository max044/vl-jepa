"""VL-JEPA model components based on official paper architecture.

Architecture (from VL-JEPA paper Section 3.1):
- X-Encoder: V-JEPA 2 ViT-L (frozen, ~300M params)
- Predictor: Last 8 Transformer layers from Llama-like model, bi-directional attention
- Y-Encoder: Embedding model (trainable, lr * 0.05)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np
from typing import Optional

from vljepa.config import Config


class XEncoder(nn.Module):
    """Frozen V-JEPA 2 Video Encoder.

    Extracts hierarchical video features.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        try:
            self.model = AutoModel.from_pretrained(config.clip_model, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Failed to load {config.clip_model}: {e}")
            # Do NOT mutate config.x_dim anymore to avoid breaking Predictor
            self.model = AutoModel.from_pretrained("facebook/vjepa-vit-h-14-224", trust_remote_code=True)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.model.to(config.device)
        self.hidden_size = config.x_dim

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode video frames."""
        if pixel_values.shape[1] == 3 and pixel_values.shape[2] > 3:
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        try:
            outputs = self.model(pixel_values_videos=pixel_values)
        except TypeError:
            outputs = self.model(pixel_values=pixel_values)

        last_hidden = outputs.last_hidden_state
        sv = last_hidden.mean(dim=1)
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

            t = torch.tensor(np.stack(frames), dtype=torch.float32, device=device)
            t = t.permute(0, 3, 1, 2) / 255.0
            t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
            padded.append(t)

        max_t = max((t.size(0) for t in padded), default=16)
        final_padded = []
        for t in padded:
            if t.size(0) < max_t:
                pad = t[-1:].expand(max_t - t.size(0), -1, -1, -1)
                t = torch.cat([t, pad], dim=0)
            final_padded.append(t)

        pixel_values = torch.stack(final_padded, dim=0)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        pixel_values = (pixel_values - mean) / std
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        return pixel_values

    def preprocess_video(self, video_frames: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Optimized preprocessing for a full video on GPU."""
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.config.dtype, torch.float32)
        
        t = torch.tensor(video_frames, dtype=dtype, device=device)
        t = t.permute(0, 3, 1, 2) / 255.0
        t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
        
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
        t = (t - mean) / std
        return t


class QueryEncoder(nn.Module):
    """Tokenizer for Qwen."""

    def __init__(self, config: Config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.predictor_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(self, texts: list[str], device: str = "cpu") -> dict:
        return self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)

class RegressionHead(nn.Module):
    """Small MLP for temporal refinement."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2),
            nn.Sigmoid() # Bound offsets to [0, 1]
        )
    
    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(config.dtype, torch.float32)
        
        # Charger le text model directement (pas ForCausalLM)
        from transformers import AutoModelForCausalLM
        full_model = AutoModelForCausalLM.from_pretrained(
            config.predictor_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",  # pas flash attention pour l'instant
        )
        
        # Extraire le text model sous-jacent
        # Pour Qwen3.5-0.8B (ForCausalLM) : full_model.model
        self.text_model = full_model.model  # Qwen3_5TextModel
        
        if config.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.text_model = get_peft_model(self.text_model, lora_config)
        
        # Garder seulement les N dernières layers si besoin
        if config.predictor_layers > 0:
            n = len(self.text_model.layers)
            self.text_model.layers = self.text_model.layers[n - config.predictor_layers:]
        
        hidden_size = self.text_model.config.hidden_size
        self.visual_proj = nn.Linear(config.x_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, config.embed_dim)
        
        if config.use_regression:
            self.regression_head = RegressionHead(hidden_size)
        
        self.to(config.device)

    def forward(self, sv, input_ids, attention_mask):
        B = sv.size(0)
        device = sv.device
        
        # Projection visuelle → 1 token visuel
        sv_embeds = self.visual_proj(sv).unsqueeze(1)  # (B, 1, hidden)
        
        # Embeddings texte
        inputs_embeds = self.text_model.embed_tokens(input_ids)  # (B, seq, hidden)
        
        # Concaténation
        combined_embeds = torch.cat([sv_embeds, inputs_embeds], dim=1)  # (B, 1+seq, hidden)
        
        visual_mask = torch.ones(B, 1, device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)  # (B, 1+seq)
        
        # Laisser Qwen3_5TextModel gérer MRoPE, causal mask, layer_types
        outputs = self.text_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            use_cache=False,
        )
        
        hidden_states = outputs.last_hidden_state  # (B, 1+seq, hidden)
        
        # [NEW] Anchor-Token Pooling: Use the first token output (the visual anchor) 
        # instead of mean pooling. This token carries the most concentrated relational 
        # information between video and text.
        pooled = hidden_states[:, 0, :]
        
        # Keep global pooling ONLY for the similarity head (sy_hat) to match original JEPAL training if needed,
        # but here we unify to pooled (anchor) for consistency in regression.
        results = {"sy_hat": self.output_proj(pooled)}
        if hasattr(self, "regression_head"):
            results["offsets"] = self.regression_head(pooled)
        return results

class IntervalIoULoss(nn.Module):
    """Temporal IoU Loss for grounding intervals with GIoU and order penalty."""
    def forward(self, pred, target):
        # pred, target: (B, 2) in [0, 1] range [start_offset, end_offset]
        p_s, p_e = pred[:, 0], pred[:, 1]
        t_s, t_e = target[:, 0], target[:, 1]
        
        # 1. Order Penalty (Don't let the model get away with start > end)
        order_penalty = torch.clamp(p_s - p_e, min=0).mean()
        
        # 2. Validity fallback (for IoU calculation only)
        p_s_final = torch.min(p_s, p_e)
        p_e_final = torch.max(p_s, p_e)
        
        inter_s = torch.max(p_s_final, t_s)
        inter_e = torch.min(p_e_final, t_e)
        inter = (inter_e - inter_s).clamp(min=0)
        
        union_s = torch.min(p_s_final, t_s)
        union_e = torch.max(p_e_final, t_e)
        union = (union_e - union_s).clamp(min=1e-6)
        
        iou = inter / union
        
        # 3. GIoU: Add outer box penalty to guide non-overlapping intervals
        outer_s = torch.min(p_s_final, t_s)
        outer_e = torch.max(p_e_final, t_e)
        outer = (outer_e - outer_s).clamp(min=1e-6)
        
        giou = iou - (outer - union) / outer
        
        return (1 - giou.mean()) + 0.1 * order_penalty

class YEncoder(nn.Module):
    """Qwen3-Embedding-0.6B Y-Encoder (trainable with reduced LR).
    
    Based on VL-JEPA paper: trainable with learning rate multiplier of 0.05x.
    Uses last_token_pool as per Qwen3-Embedding documentation.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Handle dtype selection
        if config.dtype == "bf16":
            dtype = torch.bfloat16
        elif config.dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        self.model = AutoModel.from_pretrained(
            config.text_model,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
        model_config = AutoConfig.from_pretrained(config.text_model, trust_remote_code=True)
        text_hidden = getattr(model_config, 'hidden_size', 1024)
        
        # Geler le modèle textuel de base (seule la projection sera entraînable)
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.projection = nn.Linear(text_hidden, config.embed_dim)
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.text_model, 
            trust_remote_code=True,
            padding_side='left'
        )
        
        self.to(config.device)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Y-Encoder trainable params: {trainable:,} (lr * {config.y_encoder_lr_multiplier})")

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last token pooling as per Qwen3-Embedding documentation."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode(self, texts: list[str], device: str = "cpu") -> torch.Tensor:
        """Encode texts to embeddings using last token pooling."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        outputs = self.model(**inputs)
        
        embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        
        return self.projection(embeddings)
    
    def forward(self, texts: list[str], device: str = "cpu") -> torch.Tensor:
        return self.encode(texts, device)


class VLJepa(nn.Module):
    """VL-JEPA model based on official paper architecture."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        print("Loading X-Encoder (V-JEPA 2)...")
        self.x_encoder = XEncoder(config)
        
        print("Loading Query Encoder (Qwen tokenizer)...")
        self.query_encoder = QueryEncoder(config)
        
        print("Loading Predictor (Qwen 3.5 0.8B)...")
        self.predictor = Predictor(config)
        
        print("Loading Y-Encoder (Qwen3-Embedding-0.6B)...")
        self.y_encoder = YEncoder(config)
        
        if config.use_learnable_temp:
            init_val = np.log(1 / config.temperature)
            self.logit_scale = nn.Parameter(torch.ones([]) * init_val)

    def forward(self, pixel_values, query_ids, query_mask, target_texts):
        # VL-JEPA training alignment:
        # Predictor receives (video + caption) -> sy_hat
        # Y-Encoder receives (caption) -> sy
        # InfoNCE forces sy_hat to match sy using video features as the bridge.
        sv = self.x_encoder(pixel_values)
        results = self.predictor(sv, query_ids, query_mask)
        sy_hat = results["sy_hat"]
        sy = self.y_encoder(target_texts, device=str(pixel_values.device))
        
        return {
            "sy_hat": sy_hat,
            "sy": sy,
            "offsets": results.get("offsets"),
            "temperature": self.get_temperature()
        }

    def get_temperature(self):
        if hasattr(self, "logit_scale"):
            return 1.0 / self.logit_scale.exp()
        return torch.tensor(self.config.temperature, device=self.config.device)

    def encode_video_query(self, pixel_values, query_ids, query_mask):
        sv = self.x_encoder(pixel_values)
        results = self.predictor(sv, query_ids, query_mask)
        return results

    def encode_text(self, texts, device="cpu"):
        return self.y_encoder(texts, device=device)

    def trainable_parameters(self):
        params = list(self.predictor.parameters()) + list(self.y_encoder.projection.parameters())
        if hasattr(self, "logit_scale"):
            params.append(self.logit_scale)
        return params

    def trainable_parameters_regression(self):
        """Only the regression head parameters."""
        if hasattr(self.predictor, "regression_head"):
            return list(self.predictor.regression_head.parameters())
        return []
    
    def trainable_parameters_y_encoder(self):
        """Y-Encoder parameters for separate LR."""
        return list(self.y_encoder.parameters())  # tout le Y-Encoder
    
    def trainable_parameters_predictor(self):
        """Predictor parameters."""
        return list(self.predictor.parameters())

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