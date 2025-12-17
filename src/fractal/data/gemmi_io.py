from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fractal.data.structure_features import AA3_TO_AA1, BackboneAtoms


def _require_gemmi():
    try:
        import gemmi  # type: ignore

        return gemmi
    except Exception as e:  # pragma: no cover
        raise ImportError("gemmi is required for structure parsing. Install with: pip install -e '.[data]' ") from e


def _pick_model_chain(structure, chain_id: str | None):
    # Choose first model.
    model = structure[0]
    if chain_id is not None:
        for chain in model:
            if chain.name == chain_id:
                return chain
        raise ValueError(f"Chain '{chain_id}' not found")

    # Default: pick first polymer chain that looks like protein.
    for chain in model:
        if len(chain) == 0:
            continue
        # heuristic: presence of CA atoms
        for res in chain:
            if res.find_atom("CA", "*") is not None:
                return chain
    raise ValueError("No suitable protein chain found")


def load_backbone_atoms(path: str | Path, *, chain_id: str | None = None) -> BackboneAtoms:
    """Load a single chain backbone from PDB/mmCIF using gemmi."""

    gemmi = _require_gemmi()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    st = gemmi.read_structure(str(p))
    chain = _pick_model_chain(st, chain_id)

    seq: list[str] = []
    n_list: list[np.ndarray] = []
    ca_list: list[np.ndarray] = []
    c_list: list[np.ndarray] = []
    o_list: list[np.ndarray] = []
    b_list: list[float] = []

    for res in chain:
        # Skip waters and non-polymer residues.
        if res.is_water():
            continue

        name3 = res.name.upper()
        aa1 = AA3_TO_AA1.get(name3)
        if aa1 is None:
            # Skip non-canonical amino acids by default.
            continue

        n = res.find_atom("N", "*")
        ca = res.find_atom("CA", "*")
        c = res.find_atom("C", "*")
        o = res.find_atom("O", "*")
        if any(x is None for x in (n, ca, c, o)):
            continue

        seq.append(aa1)
        n_list.append(np.array([n.pos.x, n.pos.y, n.pos.z], dtype=np.float32))
        ca_list.append(np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=np.float32))
        c_list.append(np.array([c.pos.x, c.pos.y, c.pos.z], dtype=np.float32))
        o_list.append(np.array([o.pos.x, o.pos.y, o.pos.z], dtype=np.float32))
        b_list.append(float(getattr(ca, "b_iso", 0.0)))

    if len(seq) == 0:
        raise ValueError(f"No canonical residues parsed from {p}")

    return BackboneAtoms(
        sequence="".join(seq),
        n=np.stack(n_list, axis=0),
        ca=np.stack(ca_list, axis=0),
        c=np.stack(c_list, axis=0),
        o=np.stack(o_list, axis=0),
        ca_bfactor=np.array(b_list, dtype=np.float32),
    )
