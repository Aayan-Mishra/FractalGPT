"""Evaluation metrics for protein structure prediction.

Critical AlphaFold-style metrics:
- RMSD (Root Mean Square Deviation)
- TM-score (Template Modeling score)
- GDT-TS (Global Distance Test - Total Score)
- Contact prediction accuracy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class StructureMetrics:
    """Container for structure quality metrics."""
    
    rmsd: float
    tm_score: float | None = None
    gdt_ts: float | None = None
    contact_precision: float | None = None
    contact_recall: float | None = None


def compute_rmsd(
    pred_coords: np.ndarray | torch.Tensor,
    true_coords: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None = None,
) -> float:
    """Compute RMSD between predicted and true coordinates.
    
    Args:
        pred_coords: (L, 3) or (L, N, 3) array of predicted coordinates
        true_coords: (L, 3) or (L, N, 3) array of true coordinates
        mask: Optional (L,) boolean mask for valid residues
    
    Returns:
        RMSD in Angstroms
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.cpu().numpy()
    if isinstance(true_coords, torch.Tensor):
        true_coords = true_coords.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # If 3D (L, N, 3), use CA atoms only (index 1)
    if pred_coords.ndim == 3:
        pred_coords = pred_coords[:, 1, :]  # CA atoms
    if true_coords.ndim == 3:
        true_coords = true_coords[:, 1, :]
    
    if mask is not None:
        pred_coords = pred_coords[mask]
        true_coords = true_coords[mask]
    
    # Center both structures
    pred_centered = pred_coords - pred_coords.mean(axis=0, keepdims=True)
    true_centered = true_coords - true_coords.mean(axis=0, keepdims=True)
    
    # Optimal rotation using Kabsch algorithm
    H = pred_centered.T @ true_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation and compute RMSD
    pred_aligned = pred_centered @ R
    squared_diff = np.sum((pred_aligned - true_centered) ** 2, axis=-1)
    rmsd = float(np.sqrt(squared_diff.mean()))
    
    return rmsd


def compute_tm_score(
    pred_coords: np.ndarray | torch.Tensor,
    true_coords: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None = None,
) -> float:
    """Compute TM-score (Template Modeling score).
    
    TM-score is a length-independent metric in [0, 1] where:
    - TM-score < 0.17: random
    - TM-score ~ 0.5: same fold
    - TM-score > 0.5: same topology
    
    Args:
        pred_coords: (L, 3) or (L, N, 3) array of predicted coordinates
        true_coords: (L, 3) or (L, N, 3) array of true coordinates
        mask: Optional (L,) boolean mask for valid residues
    
    Returns:
        TM-score in [0, 1]
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.cpu().numpy()
    if isinstance(true_coords, torch.Tensor):
        true_coords = true_coords.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Use CA atoms only
    if pred_coords.ndim == 3:
        pred_coords = pred_coords[:, 1, :]
    if true_coords.ndim == 3:
        true_coords = true_coords[:, 1, :]
    
    if mask is not None:
        pred_coords = pred_coords[mask]
        true_coords = true_coords[mask]
    
    L = len(pred_coords)
    if L == 0:
        return 0.0
    
    # TM-score normalization factor
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)
    
    # Center structures
    pred_centered = pred_coords - pred_coords.mean(axis=0, keepdims=True)
    true_centered = true_coords - true_coords.mean(axis=0, keepdims=True)
    
    # Optimal rotation (Kabsch)
    H = pred_centered.T @ true_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Align and compute distances
    pred_aligned = pred_centered @ R
    distances = np.linalg.norm(pred_aligned - true_centered, axis=-1)
    
    # TM-score formula
    tm_score = float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))
    
    return tm_score


def compute_gdt_ts(
    pred_coords: np.ndarray | torch.Tensor,
    true_coords: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None = None,
) -> float:
    """Compute GDT-TS (Global Distance Test - Total Score).
    
    GDT-TS is the average of GDT scores at distance cutoffs 1, 2, 4, 8 Å.
    It measures the percentage of residues under each cutoff.
    
    Args:
        pred_coords: (L, 3) or (L, N, 3) array of predicted coordinates
        true_coords: (L, 3) or (L, N, 3) array of true coordinates
        mask: Optional (L,) boolean mask for valid residues
    
    Returns:
        GDT-TS score in [0, 100]
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.cpu().numpy()
    if isinstance(true_coords, torch.Tensor):
        true_coords = true_coords.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Use CA atoms only
    if pred_coords.ndim == 3:
        pred_coords = pred_coords[:, 1, :]
    if true_coords.ndim == 3:
        true_coords = true_coords[:, 1, :]
    
    if mask is not None:
        pred_coords = pred_coords[mask]
        true_coords = true_coords[mask]
    
    L = len(pred_coords)
    if L == 0:
        return 0.0
    
    # Center structures
    pred_centered = pred_coords - pred_coords.mean(axis=0, keepdims=True)
    true_centered = true_coords - true_coords.mean(axis=0, keepdims=True)
    
    # Optimal rotation (Kabsch)
    H = pred_centered.T @ true_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Align and compute distances
    pred_aligned = pred_centered @ R
    distances = np.linalg.norm(pred_aligned - true_centered, axis=-1)
    
    # GDT-TS: average percentage under 1, 2, 4, 8 Å
    cutoffs = [1.0, 2.0, 4.0, 8.0]
    gdt_scores = []
    for cutoff in cutoffs:
        under_cutoff = (distances < cutoff).sum()
        gdt_scores.append(100.0 * under_cutoff / L)
    
    gdt_ts = float(np.mean(gdt_scores))
    return gdt_ts


def compute_contact_metrics(
    pred_contacts: np.ndarray | torch.Tensor,
    true_contacts: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None = None,
    threshold: float = 8.0,
) -> tuple[float, float]:
    """Compute contact prediction precision and recall.
    
    Args:
        pred_contacts: (L, L) predicted contact probabilities or binary contacts
        true_contacts: (L, L) true contact map (binary or distances)
        mask: Optional (L,) boolean mask for valid residues
        threshold: Distance threshold in Angstroms for defining contacts
    
    Returns:
        (precision, recall) tuple
    """
    if isinstance(pred_contacts, torch.Tensor):
        pred_contacts = pred_contacts.cpu().numpy()
    if isinstance(true_contacts, torch.Tensor):
        true_contacts = true_contacts.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if mask is not None:
        pair_mask = mask[:, None] & mask[None, :]
        pred_contacts = pred_contacts[pair_mask]
        true_contacts = true_contacts[pair_mask]
    
    # Binarize if needed
    if pred_contacts.max() <= 1.0:
        pred_binary = (pred_contacts > 0.5).astype(float)
    else:
        pred_binary = (pred_contacts < threshold).astype(float)
    
    if true_contacts.max() > 1.0:
        true_binary = (true_contacts < threshold).astype(float)
    else:
        true_binary = true_contacts
    
    # Compute precision and recall
    true_positives = (pred_binary * true_binary).sum()
    predicted_positives = pred_binary.sum()
    actual_positives = true_binary.sum()
    
    precision = float(true_positives / max(predicted_positives, 1e-8))
    recall = float(true_positives / max(actual_positives, 1e-8))
    
    return precision, recall


def evaluate_structure(
    pred_coords: np.ndarray | torch.Tensor,
    true_coords: np.ndarray | torch.Tensor,
    pred_contacts: np.ndarray | torch.Tensor | None = None,
    true_contacts: np.ndarray | torch.Tensor | None = None,
    mask: np.ndarray | torch.Tensor | None = None,
) -> StructureMetrics:
    """Compute comprehensive structure quality metrics.
    
    Args:
        pred_coords: (L, 3) or (L, N, 3) predicted coordinates
        true_coords: (L, 3) or (L, N, 3) true coordinates
        pred_contacts: Optional (L, L) predicted contacts
        true_contacts: Optional (L, L) true contacts
        mask: Optional (L,) residue mask
    
    Returns:
        StructureMetrics with all computed metrics
    """
    rmsd = compute_rmsd(pred_coords, true_coords, mask)
    tm_score = compute_tm_score(pred_coords, true_coords, mask)
    gdt_ts = compute_gdt_ts(pred_coords, true_coords, mask)
    
    contact_precision = None
    contact_recall = None
    if pred_contacts is not None and true_contacts is not None:
        contact_precision, contact_recall = compute_contact_metrics(
            pred_contacts, true_contacts, mask
        )
    
    return StructureMetrics(
        rmsd=rmsd,
        tm_score=tm_score,
        gdt_ts=gdt_ts,
        contact_precision=contact_precision,
        contact_recall=contact_recall,
    )
