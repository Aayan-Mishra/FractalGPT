from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestRow:
    """A single dataset sample pointing to precomputed supervision tensors."""

    id: str
    tokens_path: str
    residue_mask_path: str
    dist_targets_path: str
    contact_targets_path: str
    torsion_targets_path: str
    confidence_targets_path: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ManifestRow":
        # Allow manifests to contain extra metadata keys (e.g. structure_path, sequence_path).
        allowed = {
            "id",
            "tokens_path",
            "residue_mask_path",
            "dist_targets_path",
            "contact_targets_path",
            "torsion_targets_path",
            "confidence_targets_path",
        }
        filtered = {k: d[k] for k in allowed if k in d}
        missing = {
            k
            for k in (
                "id",
                "tokens_path",
                "residue_mask_path",
                "dist_targets_path",
                "contact_targets_path",
                "torsion_targets_path",
            )
            if k not in filtered
        }
        if missing:
            raise KeyError(f"Manifest row missing required keys: {sorted(missing)}")
        return cls(**filtered)


class PrecomputedConstraintDataset(Dataset):
    """Dataset for precomputed constraint supervision.

    Design intent:
    - Heavy preprocessing (PDB → distances/contacts/torsions) happens offline.
    - Training is I/O-efficient and deterministic.

    Each sample is expected to be stored as small torch tensors on disk.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.rows = [
            ManifestRow.from_dict(json.loads(line))
            for line in self.manifest_path.read_text().splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        tokens = torch.load(row.tokens_path)
        residue_mask = torch.load(row.residue_mask_path)
        dist_targets = torch.load(row.dist_targets_path)
        contact_targets = torch.load(row.contact_targets_path)
        torsion_targets = torch.load(row.torsion_targets_path)
        confidence_targets = (
            torch.load(row.confidence_targets_path) if row.confidence_targets_path else None
        )
        return {
            "tokens": tokens,
            "residue_mask": residue_mask,
            "dist_targets": dist_targets,
            "contact_targets": contact_targets,
            "torsion_targets": torsion_targets,
            "confidence_targets": confidence_targets,
        }


def collate_precomputed_constraints(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for variable-length protein sequences.
    
    Pads all tensors to the maximum length in the batch.
    """
    # Find max length in batch
    max_len = max(b["residue_mask"].size(0) for b in batch)
    
    # Prepare output tensors
    batch_size = len(batch)
    tokens_list = []
    residue_mask_list = []
    dist_targets_list = []
    contact_targets_list = []
    torsion_targets_list = []
    confidence_targets_list = []
    
    for b in batch:
        L = b["residue_mask"].size(0)
        pad = max_len - L
        
        # Pad tokens (usually already includes padding tokens)
        tokens_list.append(torch.nn.functional.pad(b["tokens"], (0, pad), value=1))  # ESM pad token = 1
        
        # Pad residue mask
        residue_mask_list.append(torch.nn.functional.pad(b["residue_mask"], (0, pad), value=False))
        
        # Pad 2D targets (L, L) -> (max_len, max_len)
        dist_targets_list.append(
            torch.nn.functional.pad(b["dist_targets"], (0, pad, 0, pad), value=0)
        )
        contact_targets_list.append(
            torch.nn.functional.pad(b["contact_targets"], (0, pad, 0, pad), value=0.0)
        )
        
        # Pad 1D targets
        torsion_targets_list.append(
            torch.nn.functional.pad(b["torsion_targets"], (0, 0, 0, pad), value=0.0)
        )
        
        if b["confidence_targets"] is not None:
            confidence_targets_list.append(
                torch.nn.functional.pad(b["confidence_targets"], (0, pad), value=0.0)
            )
    
    result = {
        "tokens": torch.stack(tokens_list),
        "residue_mask": torch.stack(residue_mask_list),
        "dist_targets": torch.stack(dist_targets_list),
        "contact_targets": torch.stack(contact_targets_list),
        "torsion_targets": torch.stack(torsion_targets_list),
    }
    
    if confidence_targets_list:
        result["confidence_targets"] = torch.stack(confidence_targets_list)
    
    return result
