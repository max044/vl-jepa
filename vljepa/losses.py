"""Loss functions for VL-JEPA.

Core loss: bidirectional InfoNCE between predicted (sy_hat) and target (sy) embeddings.

Experimental (disabled):
- SIGReg regularization (Balestriero & LeCun, 2025 — https://arxiv.org/abs/2511.08544)
"""

import torch
import torch.nn.functional as F


def infonce_bidirectional(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE loss between predicted and target embeddings.

    Args:
        pred   : (B, D) predicted embeddings — output of Predictor
        target : (B, D) target embeddings   — output of Y-Encoder
        temperature: softmax temperature τ (paper uses 0.025)

    Returns scalar loss.
    """
    pred   = F.normalize(pred,   dim=-1)
    target = F.normalize(target, dim=-1)

    logits = pred @ target.T / temperature          # (B, B)
    labels = torch.arange(pred.size(0), device=pred.device)

    loss_fwd = F.cross_entropy(logits,   labels)   # pred   → target
    loss_bwd = F.cross_entropy(logits.T, labels)   # target → pred

    return (loss_fwd + loss_bwd) / 2


def vl_jepa_loss(
    sy_hat: torch.Tensor,
    sy: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict]:
    """Compute VL-JEPA training loss.

    Args:
        sy_hat      : (B, D) predicted embedding from Predictor
        sy          : (B, D) target embedding from Y-Encoder
        temperature : InfoNCE temperature

    Returns:
        loss      : scalar tensor
        loss_dict : metrics dict for logging
    """
    loss = infonce_bidirectional(sy_hat, sy, temperature)

    return loss, {
        "loss/infonce": loss.item(),
        "loss/total":   loss.item(),
    }


# ---------------------------------------------------------------------------
# Experimental — not used in current training
# ---------------------------------------------------------------------------

# class SIGReg(torch.nn.Module):
#     """Spectral independence regularization (Balestriero & LeCun, 2025).
#     Penalises collapse of the embedding distribution.
#     Re-enable by passing sigreg_weight > 0 and a SIGReg instance to vl_jepa_loss.
#     """
#     def __init__(self, knots=17):
#         super().__init__()
#         t = torch.linspace(0, 3, knots, dtype=torch.float32)
#         dt = 3 / (knots - 1)
#         weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
#         weights[[0, -1]] = dt
#         window = torch.exp(-t.square() / 2.0)
#         self.register_buffer("t", t)
#         self.register_buffer("phi", window)
#         self.register_buffer("weights", weights * window)
#
#     def forward(self, proj):
#         A = torch.randn(proj.size(-1), 256, device=proj.device)  # device-agnostic
#         A = A.div_(A.norm(p=2, dim=0))
#         x_t = (proj @ A).unsqueeze(-1) * self.t
#         err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
#         return (err @ self.weights * proj.size(-2)).mean()