from __future__ import annotations

import torch
import torch.nn.functional as F


def distance_bin_ce(
    *,
    logits: torch.Tensor,  # (B,L,L,B)
    targets: torch.Tensor,  # (B,L,L) long
    pair_mask: torch.Tensor,  # (B,L,L) bool
) -> torch.Tensor:
    """Cross-entropy loss over distance bins."""

    B = logits.size(-1)
    logits_flat = logits[pair_mask].reshape(-1, B)
    targets_flat = targets[pair_mask].reshape(-1)
    return F.cross_entropy(logits_flat, targets_flat)


def contact_bce(
    *,
    logits: torch.Tensor,  # (B,L,L)
    targets: torch.Tensor,  # (B,L,L) float{0,1}
    pair_mask: torch.Tensor,  # (B,L,L) bool
) -> torch.Tensor:
    logits_flat = logits[pair_mask].reshape(-1)
    targets_flat = targets[pair_mask].reshape(-1)
    return F.binary_cross_entropy_with_logits(logits_flat, targets_flat)


def torsion_sincos_mse(
    *,
    pred: torch.Tensor,  # (B,L,4)
    target: torch.Tensor,  # (B,L,4)
    residue_mask: torch.Tensor,  # (B,L) bool
) -> torch.Tensor:
    pred_flat = pred[residue_mask].reshape(-1, 4)
    target_flat = target[residue_mask].reshape(-1, 4)
    return F.mse_loss(pred_flat, target_flat)
