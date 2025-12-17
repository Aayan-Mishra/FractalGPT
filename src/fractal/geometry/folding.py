from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from fractal.geometry.internal_coords import place_atom
from fractal.geometry.structure import BackboneStructure


@dataclass(frozen=True)
class FoldingSettings:
    num_bins: int
    d_min: float = 2.0
    d_max: float = 20.0
    bond_n_ca: float = 1.458
    bond_ca_c: float = 1.525
    bond_c_n: float = 1.329
    bond_c_o: float = 1.229

    angle_c_n_ca: float = 121.7
    angle_n_ca_c: float = 110.4
    angle_ca_c_n: float = 116.2
    angle_ca_c_o: float = 120.8

    omega_trans: float = 180.0


def _bin_centers(num_bins: int, d_min: float, d_max: float, device: torch.device) -> torch.Tensor:
    edges = torch.linspace(d_min, d_max, num_bins + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers


def _sin_cos_to_angle(sin_cos: torch.Tensor) -> torch.Tensor:
    # sin_cos (...,2) -> angle (...)
    s = sin_cos[..., 0]
    c = sin_cos[..., 1]
    return torch.atan2(s, c)


def _build_backbone_from_torsions(
    *,
    sequence_len: int,
    phi: torch.Tensor,
    psi: torch.Tensor,
    settings: FoldingSettings,
) -> torch.Tensor:
    """Deterministically construct backbone atoms from torsions.

    Returns:
      atoms: (L, 4, 3) in order N, CA, C, O
    """

    L = int(sequence_len)
    device = phi.device
    dtype = phi.dtype

    # Initialize first residue in a fixed reference frame.
    n0 = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
    ca0 = torch.tensor([settings.bond_n_ca, 0.0, 0.0], device=device, dtype=dtype)

    # Place C0 in xy-plane with given N-CA-C angle.
    ang = math.radians(settings.angle_n_ca_c)
    c0 = ca0 + torch.tensor(
        [settings.bond_ca_c * math.cos(math.pi - ang), settings.bond_ca_c * math.sin(math.pi - ang), 0.0],
        device=device,
        dtype=dtype,
    )

    atoms = torch.zeros((L, 4, 3), device=device, dtype=dtype)
    atoms[0, 0] = n0
    atoms[0, 1] = ca0
    atoms[0, 2] = c0

    # Place O0 with a fixed dihedral.
    o0 = place_atom(
        atoms[0, 0],
        atoms[0, 1],
        atoms[0, 2],
        length=settings.bond_c_o,
        angle_rad=math.radians(settings.angle_ca_c_o),
        dihedral_rad=torch.tensor(0.0, device=device, dtype=dtype),
    )
    atoms[0, 3] = o0

    omega = torch.tensor(math.radians(settings.omega_trans), device=device, dtype=dtype)

    for i in range(1, L):
        # N_i from psi_{i-1}
        n_i = place_atom(
            atoms[i - 1, 0],
            atoms[i - 1, 1],
            atoms[i - 1, 2],
            length=settings.bond_c_n,
            angle_rad=math.radians(settings.angle_ca_c_n),
            dihedral_rad=psi[i - 1],
        )

        ca_i = place_atom(
            atoms[i - 1, 1],
            atoms[i - 1, 2],
            n_i,
            length=settings.bond_n_ca,
            angle_rad=math.radians(settings.angle_c_n_ca),
            dihedral_rad=omega,
        )

        c_i = place_atom(
            atoms[i - 1, 2],
            n_i,
            ca_i,
            length=settings.bond_ca_c,
            angle_rad=math.radians(settings.angle_n_ca_c),
            dihedral_rad=phi[i],
        )

        o_i = place_atom(
            n_i,
            ca_i,
            c_i,
            length=settings.bond_c_o,
            angle_rad=math.radians(settings.angle_ca_c_o),
            dihedral_rad=torch.tensor(0.0, device=device, dtype=dtype),
        )

        atoms[i, 0] = n_i
        atoms[i, 1] = ca_i
        atoms[i, 2] = c_i
        atoms[i, 3] = o_i

    return atoms


def fold_from_constraints(
    *,
    sequence: str,
    distance_logits: torch.Tensor,
    torsion_angles: torch.Tensor,
    contact_logits: torch.Tensor,
    confidence: torch.Tensor | None,
    steps: int = 500,
    lr: float = 1e-2,
) -> BackboneStructure:
    """Deterministic folding/assembly driven by predicted constraints.

    This is a *geometry engine* that:
    - consumes distance distributions + torsion angles
    - performs CPU-based energy minimization
    - returns a valid backbone coordinate set

    Notes:
    - This initial implementation optimizes torsions (internal coordinates) on CPU.
    - Bond lengths/angles are enforced by construction; sterics are softly penalized.
    """

    torch.set_grad_enabled(True)

    # Geometry must run on CPU.
    device = torch.device("cpu")

    # Use only batch item 0 for now (CLI is single-seq).
    dist_logits = distance_logits[0].to(device)
    cont_logits = contact_logits[0].to(device)
    tors = torsion_angles[0].to(device)

    L = len(sequence)

    settings = FoldingSettings(num_bins=int(dist_logits.shape[-1]))

    # Convert distance distribution to expected distances.
    centers = _bin_centers(settings.num_bins, settings.d_min, settings.d_max, device)
    probs = torch.softmax(dist_logits.float(), dim=-1)
    d_target = (probs * centers).sum(dim=-1).to(torch.float32)  # (L,L)

    # Initial torsions from model (sin/cos).
    phi0 = _sin_cos_to_angle(tors[:, 0:2].float())
    psi0 = _sin_cos_to_angle(tors[:, 2:4].float())

    # First residue phi is undefined; last residue psi is undefined.
    if L > 0:
        phi0[0] = 0.0
        psi0[-1] = 0.0

    phi = torch.nn.Parameter(phi0.clone())
    psi = torch.nn.Parameter(psi0.clone())

    opt = torch.optim.Adam([phi, psi], lr=float(lr))

    # Pair weighting.
    cont_p = torch.sigmoid(cont_logits.float()).clamp(0.0, 1.0)
    pair_w = 0.5 + 0.5 * cont_p  # (L,L)

    if confidence is not None:
        conf = torch.sigmoid(confidence[0].to(device).float())
        pair_w = pair_w * (conf[:, None] * conf[None, :])

    # Ignore near-neighbor pairs where CA distances are dominated by chain geometry.
    sep = torch.arange(L, device=device)
    sep = (sep[None, :] - sep[:, None]).abs()
    valid_pairs = sep >= 3

    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)

        atoms = _build_backbone_from_torsions(sequence_len=L, phi=phi, psi=psi, settings=settings)
        ca = atoms[:, 1, :]  # (L,3)

        d_ca = torch.cdist(ca[None, :, :], ca[None, :, :]).squeeze(0).clamp(min=1e-6)

        # Distance restraint energy.
        e_dist = (pair_w[valid_pairs] * (d_ca[valid_pairs] - d_target[valid_pairs]).pow(2)).mean()

        # Soft CA-only steric repulsion (nonlocal).
        clash_thr = 3.5
        too_close = torch.relu(clash_thr - d_ca)
        e_clash = (too_close[valid_pairs].pow(2)).mean()

        # Regularize to stay near predicted torsions.
        e_reg = 0.05 * ((phi - phi0).pow(2).mean() + (psi - psi0).pow(2).mean())

        loss = e_dist + 0.1 * e_clash + e_reg
        loss.backward()
        opt.step()

    atoms_final = _build_backbone_from_torsions(sequence_len=L, phi=phi.detach(), psi=psi.detach(), settings=settings)
    atoms_np = atoms_final.detach().cpu().numpy().astype(np.float32)
    return BackboneStructure(sequence=sequence, atoms=atoms_np)
