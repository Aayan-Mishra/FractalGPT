from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


AA3_TO_AA1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


@dataclass(frozen=True)
class BackboneAtoms:
    """Backbone atom coordinates for a protein chain.

    Coordinates are in Angstrom.
    """

    sequence: str
    n: np.ndarray  # (L,3)
    ca: np.ndarray  # (L,3)
    c: np.ndarray  # (L,3)
    o: np.ndarray  # (L,3)
    ca_bfactor: np.ndarray | None  # (L,) or None


def dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Return dihedral angle (radians) for 4 points."""

    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1n = b1 / (np.linalg.norm(b1) + 1e-8)

    v = b0 - (b0 * b1n).sum() * b1n
    w = b2 - (b2 * b1n).sum() * b1n

    x = (v * w).sum()
    y = np.cross(b1n, v).dot(w)
    return float(np.arctan2(y, x))


def torsions_phi_psi(backbone: BackboneAtoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute phi/psi torsions.

    Returns:
      phi: (L,) radians (phi[0] undefined -> 0)
      psi: (L,) radians (psi[-1] undefined -> 0)
      mask: (L,) bool, True where defined
    """

    L = len(backbone.sequence)
    phi = np.zeros((L,), dtype=np.float32)
    psi = np.zeros((L,), dtype=np.float32)
    mask = np.zeros((L,), dtype=np.bool_)

    for i in range(1, L - 1):
        # phi(i) = dihedral(C_{i-1}, N_i, CA_i, C_i)
        phi[i] = dihedral(backbone.c[i - 1], backbone.n[i], backbone.ca[i], backbone.c[i])
        # psi(i) = dihedral(N_i, CA_i, C_i, N_{i+1})
        psi[i] = dihedral(backbone.n[i], backbone.ca[i], backbone.c[i], backbone.n[i + 1])
        mask[i] = True

    return phi, psi, mask


def distogram_bins(
    ca: np.ndarray,
    *,
    num_bins: int,
    d_min: float,
    d_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CA-CA distances and bin indices.

    Returns:
      dist: (L,L) float32
      bins: (L,L) int64 in [0, num_bins-1]
    """

    ca = ca.astype(np.float32)
    diff = ca[:, None, :] - ca[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1) + 1e-8).astype(np.float32)

    edges = np.linspace(d_min, d_max, num_bins + 1, dtype=np.float32)
    # bucketize into 0..num_bins-1
    bins = np.digitize(dist, edges[1:-1], right=False).astype(np.int64)
    bins = np.clip(bins, 0, num_bins - 1)

    return dist, bins


def contact_map(
    dist: np.ndarray,
    *,
    threshold: float = 8.0,
    min_seq_sep: int = 3,
) -> np.ndarray:
    L = dist.shape[0]
    cm = (dist <= float(threshold)).astype(np.float32)

    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    cm[sep < int(min_seq_sep)] = 0.0
    return cm


def torsion_sincos(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Pack torsions as sin/cos for stability.

    Output shape: (L,4) = (sin(phi), cos(phi), sin(psi), cos(psi))
    """

    out = np.zeros((phi.shape[0], 4), dtype=np.float32)
    out[:, 0] = np.sin(phi)
    out[:, 1] = np.cos(phi)
    out[:, 2] = np.sin(psi)
    out[:, 3] = np.cos(psi)
    return out
