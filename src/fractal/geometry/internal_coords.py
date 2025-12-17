from __future__ import annotations

import math

import torch


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True).clamp(min=1e-8))


def place_atom(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    length: float,
    angle_rad: float,
    dihedral_rad: torch.Tensor,
) -> torch.Tensor:
    """Place a new atom D given three previous atoms A,B,C.

    Uses standard internal coordinate construction.

    Args:
      a,b,c: (..., 3)
      length: |C-D|
      angle_rad: angle B-C-D
      dihedral_rad: dihedral A-B-C-D

    Returns:
      d: (..., 3)
    """

    bc = _normalize(c - b)
    cb = -bc

    n = _normalize(torch.cross(b - a, c - b, dim=-1))
    m = torch.cross(n, cb, dim=-1)

    # Local coordinates in the frame at C
    cos_ang = math.cos(angle_rad)
    sin_ang = math.sin(angle_rad)

    d_local = (
        cb * cos_ang
        + m * (sin_ang * torch.cos(dihedral_rad))[..., None]
        + n * (sin_ang * torch.sin(dihedral_rad))[..., None]
    )

    return c + float(length) * d_local
