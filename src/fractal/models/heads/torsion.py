from __future__ import annotations

import torch
import torch.nn as nn


class TorsionAngleHead(nn.Module):
    """Predict backbone torsion angles (phi, psi) as sin/cos.

    Predicting angles via sin/cos avoids discontinuities at +/-pi.

    Output: (B, L, 4) = (sin(phi), cos(phi), sin(psi), cos(psi))
    """

    def __init__(self, *, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, residue_embeddings: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        angles = self.net(residue_embeddings)

        # Normalize each (sin,cos) pair to unit norm for stability.
        phi = angles[..., 0:2]
        psi = angles[..., 2:4]
        phi = phi / (phi.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        psi = psi / (psi.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        out = torch.cat([phi, psi], dim=-1)

        neg_inf = torch.finfo(out.dtype).min
        out = out.masked_fill(~residue_mask[..., None], neg_inf)
        return out
