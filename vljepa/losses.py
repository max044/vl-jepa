"""Loss functions for VL-JEPA: bidirectional InfoNCE + SIGReg regularization."""

import torch
import torch.nn.functional as F


def infonce_bidirectional(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE loss between predicted and target embeddings.

    Args:
        pred: predicted embeddings (B, D), L2-normalized inside.
        target: target embeddings (B, D), L2-normalized inside.
        temperature: scaling factor for logits.

    Returns:
        Scalar loss (average of forward + backward directions).
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    # Cosine similarity matrix (B, B)
    logits = pred @ target.T / temperature

    labels = torch.arange(pred.size(0), device=pred.device)
    loss_fwd = F.cross_entropy(logits, labels)
    loss_bwd = F.cross_entropy(logits.T, labels)

    return (loss_fwd + loss_bwd) / 2


def sigreg_loss(
    embeddings: torch.Tensor,
    lambda_reg: float = 0.1,
) -> torch.Tensor:
    """Regularize embeddings towards unit-variance isotropic distribution.

    Simplified SIGReg: penalizes deviation of the covariance from identity.
    """
    if embeddings.size(0) < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Center
    embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)

    # Covariance (D, D)
    B, D = embeddings.shape
    cov = (embeddings.T @ embeddings) / (B - 1)

    # Variance: encourage diagonal to be 1
    var_loss = F.relu(1.0 - cov.diagonal()).mean()

    # Covariance: decorrelate off-diagonal
    off_diag = cov - torch.diag(cov.diagonal())
    cov_loss = (off_diag ** 2).mean()

    return lambda_reg * (var_loss + cov_loss)


def vl_jepa_loss(
    sy_hat: torch.Tensor,
    sy: torch.Tensor,
    temperature: float | torch.Tensor = 0.07,
    sigreg_weight: float = 0.1,
    offsets: torch.Tensor | None = None,
    offset_targets: torch.Tensor | None = None,
    regression_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Bidirectional InfoNCE loss with optional regression loss.
    
    Args:
        sy_hat: Predicted embeddings (B, D)
        sy: Target embeddings (B, D)
        temperature: InfoNCE temperature
        sigreg_weight: Weight for signature regularization
        offsets: Predicted offsets (B, 2)
        offset_targets: Target offsets (B, 2)
        regression_weight: Weight for regression loss
    """
    # 1. InfoNCE Loss (bidirectional)
    infonce = infonce_bidirectional(sy_hat, sy, temperature)
    
    # 2. Signature Regularization (optional, helps avoid collapse)
    # Penalize low variance in embeddings
    if sy_hat.size(0) > 1:
        std_hat = sy_hat.std(dim=0).mean()
        std_y = sy.std(dim=0).mean()
        sigreg = - (std_hat + std_y)
    else:
        sigreg = torch.tensor(0.0, device=sy_hat.device)
    
    total_loss = infonce + sigreg_weight * sigreg
    
    metrics = {
        "loss/infonce": infonce.item(),
        "loss/sigreg": sigreg.item(),
    }

    # 3. Regression Loss (optional)
    if offsets is not None and offset_targets is not None:
        reg_loss = F.smooth_l1_loss(offsets, offset_targets)
        total_loss += regression_weight * reg_loss
        metrics["loss/regression"] = reg_loss.item()
    
    metrics["loss/total"] = total_loss.item()
    
    return total_loss, metrics
