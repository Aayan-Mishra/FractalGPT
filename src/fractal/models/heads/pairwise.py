from __future__ import annotations

import torch
import torch.nn as nn


class PairwiseConstraintHead(nn.Module):
    """Predict pairwise geometric constraints from per-residue embeddings.

    Outputs are structured tensors (distance-bin logits + contact logits).

    Memory efficiency: pairwise representations are computed in blocks along the
    i-dimension to avoid materializing an (L x L x D) tensor at once.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        pairwise_dim: int,
        hidden_dim: int,
        num_distance_bins: int,
        chunk_size: int = 64,
        checkpoint_chunks: bool = False,
        symmetric: bool = True,
    ):
        super().__init__()
        self.num_distance_bins = int(num_distance_bins)
        self.chunk_size = int(chunk_size)
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self.symmetric = bool(symmetric)

        self.q_proj = nn.Linear(input_dim, pairwise_dim)
        self.k_proj = nn.Linear(input_dim, pairwise_dim)

        self.mlp = nn.Sequential(
            nn.Linear(pairwise_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_distance_bins + 1),
        )

        self._checkpoint = None
        if self.checkpoint_chunks:
            from torch.utils.checkpoint import checkpoint

            self._checkpoint = checkpoint

    def forward(self, residue_embeddings: torch.Tensor, residue_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            residue_embeddings: (B, L, C)
            residue_mask: (B, L) boolean

        Returns:
            distance_logits: (B, L, L, num_bins)
            contact_logits: (B, L, L)
        """

        B, L, _ = residue_embeddings.shape
        device = residue_embeddings.device

        q = self.q_proj(residue_embeddings)  # (B, L, D)
        k = self.k_proj(residue_embeddings)  # (B, L, D)

        dist_logits = torch.empty((B, L, L, self.num_distance_bins), device=device, dtype=residue_embeddings.dtype)
        contact_logits = torch.empty((B, L, L), device=device, dtype=residue_embeddings.dtype)

        # Mask pairs involving padding.
        pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]  # (B, L, L)

        for i0 in range(0, L, self.chunk_size):
            i1 = min(L, i0 + self.chunk_size)
            q_chunk = q[:, i0:i1, :]  # (B, c, D)

            # Broadcast to (B, c, L, D)
            pair_repr = q_chunk[:, :, None, :] + k[:, None, :, :]

            if self._checkpoint is not None and self.training:
                out = self._checkpoint(self.mlp, pair_repr, use_reentrant=False)
            else:
                out = self.mlp(pair_repr)  # (B, c, L, num_bins+1)
            dist_logits[:, i0:i1, :, :] = out[..., : self.num_distance_bins]
            contact_logits[:, i0:i1, :] = out[..., self.num_distance_bins]

        # Apply mask by setting invalid pairs to large negative.
        neg_inf = torch.finfo(dist_logits.dtype).min
        dist_logits = dist_logits.masked_fill(~pair_mask[..., None], neg_inf)
        contact_logits = contact_logits.masked_fill(~pair_mask, neg_inf)

        if self.symmetric:
            dist_logits = 0.5 * (dist_logits + dist_logits.transpose(1, 2))
            contact_logits = 0.5 * (contact_logits + contact_logits.transpose(1, 2))

        return dist_logits, contact_logits
