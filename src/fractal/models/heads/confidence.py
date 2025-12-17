from __future__ import annotations

import torch
import torch.nn as nn


class ConfidenceHead(nn.Module):
    """Optional per-residue confidence/uncertainty head.

    This is deliberately lightweight and produces a single scalar per residue.
    Interpretation is task-dependent (e.g., pLDDT-style proxy, calibration target, etc.).
    """

    def __init__(self, *, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, residue_embeddings: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        conf = self.net(residue_embeddings).squeeze(-1)  # (B, L)
        neg_inf = torch.finfo(conf.dtype).min
        conf = conf.masked_fill(~residue_mask, neg_inf)
        return conf
