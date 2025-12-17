from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConstraintPredictions:
    """Structured outputs produced by the neural constraint predictor.

    Important: this intentionally excludes any direct XYZ coordinate prediction.
    """

    sequence: str
    distance_logits: torch.Tensor  # (B, L, L, num_bins)
    contact_logits: torch.Tensor  # (B, L, L)
    torsion_angles: torch.Tensor  # (B, L, 4) = (sin(phi), cos(phi), sin(psi), cos(psi))
    confidence: torch.Tensor | None  # (B, L)
    residue_mask: torch.Tensor  # (B, L) boolean, True for real residues
