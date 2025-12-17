from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BackboneStructure:
    """Minimal backbone structure container.

    Stores backbone atoms for each residue in PDB atom order: N, CA, C, O.
    """

    sequence: str
    atoms: np.ndarray  # (L, 4, 3) float32

    def to_pdb(self, path: str | Path, *, chain_id: str = "A") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        atom_names = ["N", "CA", "C", "O"]
        resname = "ALA"  # placeholder; residue typing can be added later

        lines: list[str] = []
        serial = 1
        for i, aa in enumerate(self.sequence, start=1):
            # TODO: map aa->3-letter; keep placeholder deterministic for now.
            _ = aa
            for a_idx, atom in enumerate(atom_names):
                x, y, z = self.atoms[i - 1, a_idx]
                lines.append(
                    f"ATOM  {serial:5d} {atom:>4s} {resname:>3s} {chain_id}{i:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom[0]:>2s}"
                )
                serial += 1

        lines.append("END")
        p.write_text("\n".join(lines) + "\n")
