import math

import torch

from fractal.geometry.internal_coords import place_atom


def test_place_atom_bond_length():
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    c = torch.tensor([1.0, 1.0, 0.0])

    d = place_atom(a, b, c, length=2.0, angle_rad=math.radians(60.0), dihedral_rad=torch.tensor(0.0))
    assert torch.isfinite(d).all()
    assert abs(torch.linalg.norm(d - c).item() - 2.0) < 1e-4
