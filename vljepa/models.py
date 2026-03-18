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
        try:
            self.model = AutoModel.from_pretrained(config.clip_model, trust_remote_code=True)
        except Exception:
            print(f"Warning: Failed to load {config.clip_model}. Trying fallback 'facebook/vjepa-vit-h-14-224'.")
            self.model = AutoModel.from_pretrained("facebook/vjepa-vit-h-14-224", trust_remote_code=True)
            config.x_dim = self.model.config.hidden_size

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
        t = torch.tensor(video_frames, dtype=torch.float16, device=device)
        t = t[..., [2, 1, 0]]
        t = t.permute(0, 3, 1, 2) / 255.0
        t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)
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


class Predictor(nn.Module):
    """Qwen 3.5 0.8B Predictor with bi-directional attention.
    
    Based on VL-JEPA paper Section 3.1:
    - Uses last N transformer layers (default: 8)
    - Disables causal attention mask for bi-directional attention
    - Linear projections connect to vision and text embeddings
    - Average pooling on non-[PAD] tokens for output
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        if config.device == "cuda":
            dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16
        else:
            dtype = torch.float32
        
        model_config = AutoConfig.from_pretrained(config.predictor_model, trust_remote_code=True)
        
        self.model = AutoModel.from_pretrained(
            config.predictor_model,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
        hidden_size = getattr(model_config, 'hidden_size', 896)
        num_hidden_layers = getattr(model_config, 'num_hidden_layers', 24)
        
        num_layers = config.predictor_layers if config.predictor_layers > 0 else num_hidden_layers
        start_layer = num_hidden_layers - num_layers if config.predictor_layers > 0 else 0
        
        print(f"  Predictor: using layers {start_layer}-{num_hidden_layers-1} ({num_layers} layers)")
        
        # Get layers - handle different model structures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            embed_tokens = self.model.model.embed_tokens
            rotary_emb = self.model.model.rotary_emb
            norm = self.model.model.norm
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
            embed_tokens = self.model.embed_tokens
            rotary_emb = self.model.rotary_emb
            norm = self.model.norm
        else:
            raise ValueError(f"Cannot find layers in model type: {type(self.model)}")
        
        if config.predictor_layers > 0:
            self.transformer_layers = nn.ModuleList(list(layers)[start_layer:])
            self.norm = norm
            self.using_partial_layers = True
        else:
            self.transformer_layers = None
            self.using_partial_layers = False
        
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.visual_proj = nn.Linear(config.x_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, config.embed_dim)
        
        if config.use_regression:
            self.regression_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 2)
            )
        
        self.to(config.device)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Predictor trainable params: {trainable:,}")

    def forward(
        self, 
        sv: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass with bi-directional attention.
        
        Visual and query embeddings are concatenated and processed with
        bi-directional attention (causal mask disabled).
        """
        B = sv.size(0)
        device = sv.device
        
        sv_embeds = self.visual_proj(sv).unsqueeze(1)
        
        inputs_embeds = self.embed_tokens(input_ids)
        
        combined_embeds = torch.cat([sv_embeds, inputs_embeds], dim=1)
        
        visual_mask = torch.ones(B, 1, device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        
        seq_len = combined_embeds.size(1)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        
        hidden_states = combined_embeds
        
        if self.using_partial_layers:
            for layer in self.transformer_layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=combined_mask,
                    position_ids=position_ids,
                )[0]
            
            batch_indices = torch.arange(B, device=device)
            non_pad_mask = combined_mask.bool()
            sum_embeddings = (hidden_states * non_pad_mask.unsqueeze(-1)).sum(dim=1)
            sum_mask = non_pad_mask.sum(dim=1, keepdim=True).float()
            pooled = sum_embeddings / sum_mask.clamp(min=1)
        else:
            outputs = self.model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
            )
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states.mean(dim=1)
        
        results = {"sy_hat": self.output_proj(pooled)}
        if hasattr(self, "regression_head"):
            results["offsets"] = self.regression_head(pooled)
            
        return results


class YEncoder(nn.Module):
    """Qwen3-Embedding-0.6B Y-Encoder (trainable with reduced LR).
    
    Based on VL-JEPA paper: trainable with learning rate multiplier of 0.05x.
    Uses last_token_pool as per Qwen3-Embedding documentation.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        if config.device == "cuda":
            dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16
        else:
            dtype = torch.float32
        
        self.model = AutoModel.from_pretrained(
            config.text_model,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        
        model_config = AutoConfig.from_pretrained(config.text_model, trust_remote_code=True)
        text_hidden = getattr(model_config, 'hidden_size', 1024)
        
        self.projection = nn.Linear(text_hidden, config.embed_dim)
        
        self.projection.weight.requires_grad = True
        self.projection.bias.requires_grad = True
        
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

    @torch.no_grad()
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
    
    def trainable_parameters_y_encoder(self):
        """Y-Encoder parameters for separate LR."""
        return list(self.y_encoder.projection.parameters())
    
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