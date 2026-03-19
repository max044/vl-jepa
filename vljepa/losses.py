"""Loss functions for VL-JEPA: bidirectional InfoNCE + SIGReg regularization.

SIGReg implementation based on LeJEPA (Balestriero & LeCun, 2025).
Reference: https://arxiv.org/abs/2511.08544
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def infonce_bidirectional(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE loss between predicted and target embeddings."""
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    logits = pred @ target.T / temperature

    labels = torch.arange(pred.size(0), device=pred.device)
    loss_fwd = F.cross_entropy(logits, labels)
    loss_bwd = F.cross_entropy(logits.T, labels)

    return (loss_fwd + loss_bwd) / 2


def vl_jepa_loss(
    sy_hat: torch.Tensor,
    sy: torch.Tensor,
    temperature: float | torch.Tensor = 0.07,
    sigreg_weight: float = 0.1,
    offsets: torch.Tensor | None = None,
    offset_targets: torch.Tensor | None = None,
    regression_weight: float = 1.0,
    sigreg_module: nn.Module | None = None,
) -> tuple[torch.Tensor, dict]:
    """Bidirectional InfoNCE loss with SIGReg regularization.
    
    Args:
        sy_hat: Predicted embeddings (B, D)
        sy: Target embeddings (B, D)
        temperature: InfoNCE temperature
        sigreg_weight: Weight for SIGReg regularization
        offsets: Predicted offsets (B, 2)
        offset_targets: Target offsets (B, 2)
        regression_weight: Weight for regression loss
        sigreg_module: Optional pre-initialized SIGReg module
    """
    infonce = infonce_bidirectional(sy_hat, sy, temperature)
    
    sigreg_value = torch.tensor(0.0, device=sy_hat.device)
    if sigreg_weight > 0 and sy_hat.size(0) > 1:
        if sigreg_module is not None:
            sigreg_value = sigreg_module(sy_hat) + sigreg_module(sy)
        else:
            sigreg_value = torch.tensor(0.0, device=sy_hat.device)
    
    total_loss = infonce + sigreg_weight * sigreg_value
    
    metrics = {
        "loss/infonce": infonce.item(),
        "loss/sigreg": sigreg_value.item() if sigreg_weight > 0 else 0.0,
    }

    if offsets is not None and offset_targets is not None:
        reg_loss = F.smooth_l1_loss(offsets, offset_targets)
        total_loss += regression_weight * reg_loss
        metrics["loss/regression"] = reg_loss.item()
    
    metrics["loss/total"] = total_loss.item()
    
    return total_loss, metrics
