import math

import torch

from fractal.geometry.folding import FoldingSettings, fold_from_constraints
from fractal.geometry.internal_coords import place_atom


def _make_sincos(angle: float) -> tuple[float, float]:
    return math.sin(angle), math.cos(angle)


def test_fold_from_constraints_runs_small():
    seq = "ACDEFGHIKL"
    L = len(seq)
    num_bins = 32

    # Zero torsions.
    phi_sc = torch.tensor([_make_sincos(0.0)] * L, dtype=torch.float32)
    psi_sc = torch.tensor([_make_sincos(0.0)] * L, dtype=torch.float32)
    tors = torch.cat([phi_sc, psi_sc], dim=-1)[None, :, :]  # (1,L,4)

    # Dummy contacts.
    contact_logits = torch.zeros((1, L, L), dtype=torch.float32)

    # Create a simple distogram peaked at mid-bin.
    dist_logits = torch.zeros((1, L, L, num_bins), dtype=torch.float32)
    dist_logits[..., num_bins // 2] = 5.0

    struct = fold_from_constraints(
        sequence=seq,
        distance_logits=dist_logits,
        torsion_angles=tors,
        contact_logits=contact_logits,
        confidence=None,
        steps=10,
        lr=5e-2,
    )

    assert struct.atoms.shape == (L, 4, 3)
    assert torch.isfinite(torch.tensor(struct.atoms)).all()
