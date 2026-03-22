"""VL-JEPA model components — faithful to the official paper architecture.

Architecture (VL-JEPA paper, Section 3.1):
- X-Encoder : V-JEPA 2 ViT-L, frozen (~307M params)
- Predictor  : Last 8 Transformer layers of Llama-3.2-1B, bidirectional attention
               + LoRA (r=16) via inject_adapter_in_model
               + linear projection -> 1536-dim shared embedding space
- Y-Encoder  : EmbeddingGemma-300M, trainable at LR x 0.05
               + linear projection -> 1536-dim shared embedding space

Shared embedding dim: 1536 (fixed per paper).
Loss is computed in this shared space (bidirectional InfoNCE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer, LlamaModel

from vljepa.config import Config


# ---------------------------------------------------------------------------
# Constants (per paper)
# ---------------------------------------------------------------------------

EMBED_DIM         = 1536
MAX_TEXT_LEN      = 512
PREDICTOR_LAYERS  = 8
LLAMA_MODEL       = "meta-llama/Llama-3.2-1B"
GEMMA_EMBED_MODEL = "google/embeddinggemma-300m"


# ---------------------------------------------------------------------------
# X-Encoder — frozen V-JEPA 2 ViT-L
# ---------------------------------------------------------------------------

class XEncoder(nn.Module):
    """Frozen V-JEPA 2 video encoder. Never updated during training."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(config.dtype, torch.float32)
        self.model = AutoModel.from_pretrained(config.clip_model, trust_remote_code=True, torch_dtype=dtype)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(config.device)

        self.hidden_size = config.x_dim

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return mean-pooled video embedding (B, x_dim)."""
        pixel_values = pixel_values.to(self.model.device, dtype=next(self.model.parameters()).dtype)
        if pixel_values.shape[1] == 3 and pixel_values.shape[2] > 3:
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        outputs = self.model(pixel_values_videos=pixel_values)
        return outputs.last_hidden_state.mean(dim=1)

    def preprocess_frames(self, frames_batch: list, device: str = "cpu") -> torch.Tensor:
        """Normalise and pad a batch of raw frame lists -> (B, T, C, H, W)."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1)

        padded = []
        for frames in frames_batch:
            if len(frames) == 0:
                padded.append(torch.zeros((16, 3, 224, 224), device=device))
                continue
            t = torch.tensor(np.stack(frames), dtype=torch.float32, device=device)
            t = t.permute(0, 3, 1, 2) / 255.0
            t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            padded.append(t)

        max_t = max(t.size(0) for t in padded)
        final = []
        for t in padded:
            if t.size(0) < max_t:
                pad = t[-1:].expand(max_t - t.size(0), -1, -1, -1)
                t = torch.cat([t, pad], dim=0)
            final.append(t)

        pixel_values = torch.stack(final, dim=0)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        pixel_values = (pixel_values - mean) / std
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        return pixel_values

    def preprocess_video(self, frames_np: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Preprocess a full video (N, H, W, 3) RGB -> (N, 3, 224, 224) normalized."""
        mean       = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std        = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        chunk_size = 64
        processed  = []

        for i in range(0, len(frames_np), chunk_size):
            chunk = torch.tensor(frames_np[i : i + chunk_size], dtype=torch.float32, device=device)
            chunk = chunk.permute(0, 3, 1, 2) / 255.0
            chunk = F.interpolate(chunk, size=(224, 224), mode="bilinear", align_corners=False)
            chunk = (chunk - mean) / std
            processed.append(chunk)

        return torch.cat(processed, dim=0)  # (N, 3, 224, 224)


# ---------------------------------------------------------------------------
# Predictor — last 8 layers of Llama-3.2-1B, bidirectional + LoRA
# ---------------------------------------------------------------------------

def _make_bidirectional_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """(B, T) padding mask -> (B, 1, 1, T) additive mask. PAD=-inf, real=0."""
    mask = attention_mask[:, None, None, :].float() 
    return (1.0 - mask) * torch.finfo(torch.float32).min


class Predictor(nn.Module):
    """VL-JEPA Predictor (paper Section 3.1).

    Initialised from the last 8 layers of Llama-3.2-1B with bidirectional attention.
    We use LlamaModel.forward() with inputs_embeds and monkey-patch create_causal_mask
    to return our non-causal mask instead of the triangular one.

    LoRA (r=16) injected via inject_adapter_in_model (works on any nn.Module).
    Linear projection -> 1536-dim shared embedding space.
    """

    def __init__(self, config: Config):
        super().__init__()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(config.dtype, torch.float32)

        print(f"  Loading Llama-3.2-1B to extract last {PREDICTOR_LAYERS} layers...")
        full_llama: LlamaModel = LlamaModel.from_pretrained(
            LLAMA_MODEL,
            torch_dtype=dtype,
            attn_implementation="eager",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Keep only last PREDICTOR_LAYERS and update config accordingly
        full_llama.layers = nn.ModuleList(full_llama.layers[-PREDICTOR_LAYERS:])
        full_llama.config.num_hidden_layers = PREDICTOR_LAYERS
        self.llama = full_llama

        llama_hidden = self.llama.config.hidden_size

        if config.use_lora:
            from peft import inject_adapter_in_model, LoraConfig
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.llama = inject_adapter_in_model(lora_cfg, self.llama)
            for name, param in self.llama.named_parameters():
                param.requires_grad = "lora" in name

        # embed_tokens and norm are never trained
        for param in self.llama.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.llama.norm.parameters():
            param.requires_grad = False

        self.visual_proj = nn.Linear(config.x_dim, llama_hidden).to(dtype)
        self.output_proj = nn.Linear(llama_hidden, EMBED_DIM).to(dtype)

        self.to(config.device)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Predictor trainable params: {trainable / 1e6:.1f}M")

        self._verify_bidirectional()

    def _verify_bidirectional(self):
        """Vérifie la bidirectionnalité par différence de comportement causal vs non-causal.

        Place un signal sur le dernier token. En causal, token 0 ne peut pas le voir.
        En bidirectionnel, token 0 est influencé → sortie différente.
        """
        import transformers.models.llama.modeling_llama as llama_module

        device = next(self.llama.parameters()).device
        dtype  = next(self.llama.parameters()).dtype
        B, T   = 1, 6
        H      = self.llama.config.hidden_size

        dummy           = torch.zeros(B, T, H, device=device, dtype=dtype)
        dummy[0, -1, :] = 1.0

        # Forward bidirectionnel — create_causal_mask retourne None
        original_ccm = llama_module.create_causal_mask
        llama_module.create_causal_mask = lambda **kwargs: None

        with torch.no_grad():
            out_bidir = self.llama(inputs_embeds=dummy, use_cache=False).last_hidden_state

        # Forward causal — masque original restauré
        llama_module.create_causal_mask = original_ccm
        with torch.no_grad():
            out_causal = self.llama(inputs_embeds=dummy, use_cache=False).last_hidden_state

        diff = (out_bidir[0, 0] - out_causal[0, 0]).abs().sum().item()
        assert diff > 1e-4, f"Masque bidirectionnel indistinguable du causal (diff={diff:.6f})"
        print(f"  ✓ Bidirectional attention verified (diff vs causal={diff:.4f})")

    def forward(self, sv: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        import transformers.models.llama.modeling_llama as llama_module

        B      = sv.size(0)
        device = sv.device

        sv_embed      = self.visual_proj(sv).unsqueeze(1)
        text_embeds   = self.llama.embed_tokens(input_ids)
        combined      = torch.cat([sv_embed, text_embeds], dim=1)       # (B, 1+T, hidden)

        visual_mask   = torch.ones(B, 1, device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1) # (B, 1+T)

        # Monkey-patch create_causal_mask — replaced by our non-causal padding mask.
        # try/finally guarantees restoration even if forward raises.
        original_ccm = llama_module.create_causal_mask
        def _padding_only_mask(config, inputs_embeds, attention_mask, **kwargs):
            return _make_bidirectional_mask(combined_mask).to(inputs_embeds.dtype)
        llama_module.create_causal_mask = _padding_only_mask

        try:
            out = self.llama(
                inputs_embeds=combined,
                attention_mask=combined_mask,
                use_cache=False,
            )
        finally:
            llama_module.create_causal_mask = original_ccm

        hidden = out.last_hidden_state  # (B, 1+T, hidden)

        # Average pooling on non-PAD tokens (paper Section 3.1)
        mask_exp = combined_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled   = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-6)

        return {"sy_hat": self.output_proj(pooled)}


# ---------------------------------------------------------------------------
# Y-Encoder — EmbeddingGemma-300M, trainable at LR x 0.05
# ---------------------------------------------------------------------------

class YEncoder(nn.Module):
    """VL-JEPA Y-Encoder (paper Section 3.1).

    EmbeddingGemma-300M via SentenceTransformer (5 modules: Transformer, Pooling,
    Dense x2, Normalize). We use modules [0-3] (without final Normalize) and add
    our own linear projection 768 -> 1536.

    We bypass SentenceTransformer.encode() to preserve gradients, calling
    each module manually in forward().

    All parameters trainable; LR multiplier x0.05 applied in the optimizer.
    """

    GEMMA_OUT_DIM = 768  # output dim after Dense x2, before our projection

    def __init__(self, config: Config):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        print("  Loading EmbeddingGemma-300M (SentenceTransformer)...")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(config.dtype, torch.float32)
        st = SentenceTransformer(GEMMA_EMBED_MODEL, model_kwargs={"torch_dtype": dtype})

        # st pipeline: [0] Transformer, [1] Pooling, [2] Dense, [3] Dense, [4] Normalize
        # We keep [0-3] and add our own projection instead of [4]
        self.st_modules = nn.ModuleList([st[i] for i in range(4)])
        self.projection = nn.Linear(self.GEMMA_OUT_DIM, EMBED_DIM).to(dtype)

        self.to(config.device)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Y-Encoder trainable params: {trainable / 1e6:.1f}M  (lr x {config.y_encoder_lr_multiplier})")

    def encode_document(self, captions: list[str], device: str = "cpu") -> torch.Tensor:
        """Encode captions as documents, preserving gradients.

        st_modules[0].tokenize() prepends the document prompt automatically.
        We call each module manually to keep the computation graph intact.
        """
        features = self.st_modules[0].tokenize(captions)
        features = {k: v.to(device) for k, v in features.items()}

        # SentenceTransformer modules all accept and return a feature dict
        out = features
        for module in self.st_modules:
            out = module(out)

        pooled = out["sentence_embedding"]  # (B, 768) after Dense x2
        return self.projection(pooled)      # (B, EMBED_DIM)

    def forward(self, captions: list[str], device: str = "cpu") -> torch.Tensor:
        return self.encode_document(captions, device)


# ---------------------------------------------------------------------------
# QueryEncoder — thin wrapper around Llama tokenizer
# ---------------------------------------------------------------------------

class QueryEncoder(nn.Module):
    """Holds a reference to the Llama tokenizer already loaded by Predictor.

    Exists for API compatibility with train.py (model.query_encoder.tokenize()).
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def tokenize(self, texts: list, device: str = "cpu") -> dict:
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
        ).to(device)


# ---------------------------------------------------------------------------
# VLJepa — top-level model
# ---------------------------------------------------------------------------

class VLJepa(nn.Module):
    """VL-JEPA full model.

    Training objective (paper Section 3.2):
        Predictor(video + query)  ->  sy_hat
        Y-Encoder(caption)        ->  sy
        Bidirectional InfoNCE loss on (sy_hat, sy) in the 1536-dim shared space.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        print("Loading X-Encoder (V-JEPA 2 ViT-L, frozen)...")
        self.x_encoder = XEncoder(config)

        print("Loading Predictor (Llama-3.2-1B last 8 layers + LoRA)...")
        self.predictor = Predictor(config)

        print("Wiring Query Encoder (reusing Llama tokenizer)...")
        self.query_encoder = QueryEncoder(self.predictor.tokenizer)

        print("Loading Y-Encoder (EmbeddingGemma-300M)...")
        self.y_encoder = YEncoder(config)

        if config.use_learnable_temp:
            init_val = torch.tensor(config.temperature).log().neg()
            self.logit_scale = nn.Parameter(torch.ones([]) * init_val)

    def forward(self, pixel_values, query_ids, query_mask, target_texts) -> dict:
        device = pixel_values.device
        sv     = self.x_encoder(pixel_values)
        result = self.predictor(sv, query_ids, query_mask)
        sy     = self.y_encoder(target_texts, device=device)
        return {"sy_hat": result["sy_hat"], "sy": sy, "temperature": self.get_temperature()}

    def get_temperature(self) -> torch.Tensor:
        if hasattr(self, "logit_scale"):
            return 1.0 / self.logit_scale.exp()
        return torch.tensor(self.config.temperature, device=self.config.device)

    def encode_video_query(self, pixel_values, query_ids, query_mask) -> dict:
        return self.predictor(self.x_encoder(pixel_values), query_ids, query_mask)

    def encode_text(self, texts: list, device: str = "cpu") -> torch.Tensor:
        return self.y_encoder(texts, device=device)

    def count_parameters(self) -> dict:
        def _count(m):
            return {
                "total":     sum(p.numel() for p in m.parameters()),
                "trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
            }
        return {
            "x_encoder": _count(self.x_encoder),
            "predictor":  _count(self.predictor),
            "y_encoder":  _count(self.y_encoder),
        }


# ---------------------------------------------------------------------------
# Disabled — temporal regression (not used in current training phase)
# ---------------------------------------------------------------------------

# class RegressionHead(nn.Module): ...
# class IntervalIoULoss(nn.Module): ...