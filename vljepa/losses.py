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
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
    sigreg_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined VL-JEPA training loss.

    Returns:
        total_loss: scalar tensor for backprop.
        metrics: dict with breakdown of loss components.
    """
    align = infonce_bidirectional(pred, target, temperature)
    reg_pred = sigreg_loss(pred, sigreg_weight)
    reg_target = sigreg_loss(target, sigreg_weight)

    total = align + reg_pred + reg_target

    metrics = {
        "loss/total": total.item(),
        "loss/infonce": align.item(),
        "loss/sigreg_pred": reg_pred.item(),
        "loss/sigreg_target": reg_target.item(),
    }

    return total, metrics
