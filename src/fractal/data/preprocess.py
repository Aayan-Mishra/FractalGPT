from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from fractal.data.gemmi_io import load_backbone_atoms
from fractal.data.structure_features import (
    contact_map,
    distogram_bins,
    torsion_sincos,
    torsions_phi_psi,
)
from fractal.ssl_utils import configure_ssl_certificates


_ESM_TOKENIZER_CACHE: dict[str, tuple[object, int]] = {}


def _get_esm_batch_converter(esm_checkpoint: str):
    """Return (batch_converter, padding_idx) for an ESM checkpoint.

    We intentionally cache this so preprocessing multiple structures doesn't
    repeatedly load the same model/alphabet (and re-trigger downloads).
    """

    if esm_checkpoint in _ESM_TOKENIZER_CACHE:
        return _ESM_TOKENIZER_CACHE[esm_checkpoint]

    configure_ssl_certificates()

    try:
        import esm  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("ESM not installed; install with: pip install -e '.[esm]' ") from e

    if not hasattr(esm.pretrained, esm_checkpoint):
        raise ValueError(f"Unknown esm checkpoint: {esm_checkpoint}")

    model, alphabet = getattr(esm.pretrained, esm_checkpoint)()
    # Free model weights ASAP; we only need alphabet+batch_converter.
    del model

    batch_converter = alphabet.get_batch_converter()
    pad = int(alphabet.padding_idx)
    _ESM_TOKENIZER_CACHE[esm_checkpoint] = (batch_converter, pad)
    return batch_converter, pad


@dataclass(frozen=True)
class PreprocessConfig:
    num_distance_bins: int = 64
    d_min: float = 2.0
    d_max: float = 20.0
    contact_threshold: float = 8.0
    min_contact_seq_sep: int = 3
    max_len: int = 1024


def _tokenize_with_esm(sequences: list[str], *, esm_checkpoint: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize with ESM alphabet.

    This loads the ESM checkpoint to access its alphabet. For large datasets you
    should run preprocessing in bigger batches.
    """

    batch_converter, pad = _get_esm_batch_converter(esm_checkpoint)
    data = [(f"seq{i}", s[:max_len]) for i, s in enumerate(sequences)]
    _, _, tokens = batch_converter(data)

    lens = (tokens != pad).sum(dim=1)
    max_L = int((lens - 2).max().item())

    residue_mask = torch.zeros((tokens.size(0), max_L), dtype=torch.bool)
    for i, l in enumerate(lens.tolist()):
        residue_mask[i, : (l - 2)] = True

    return tokens, residue_mask


def preprocess_structure(
    *,
    structure_path: str | Path,
    out_dir: str | Path,
    pdb_id: str,
    cfg: PreprocessConfig,
    chain_id: str | None,
    esm_checkpoint: str | None,
) -> dict:
    """Parse a single structure file and write tensor artifacts.

    Writes:
      - sequence.txt
      - torsion_targets.pt  (L,4)
      - dist_targets.pt     (L,L) long
      - contact_targets.pt  (L,L) float
      - confidence_targets.pt (L,) float  (optional)
      - tokens.pt / residue_mask.pt (if esm_checkpoint provided)

    Returns a manifest row dict.
    """

    out_dir = Path(out_dir) / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)

    bb = load_backbone_atoms(structure_path, chain_id=chain_id)
    seq = bb.sequence[: cfg.max_len]

    dist, dist_bins = distogram_bins(bb.ca[: cfg.max_len], num_bins=cfg.num_distance_bins, d_min=cfg.d_min, d_max=cfg.d_max)
    contacts = contact_map(dist, threshold=cfg.contact_threshold, min_seq_sep=cfg.min_contact_seq_sep)

    phi, psi, _ = torsions_phi_psi(bb)
    phi = phi[: cfg.max_len]
    psi = psi[: cfg.max_len]
    tors = torsion_sincos(phi, psi)  # (L,4)

    # Confidence target from CA B-factor (optional supervision; interpretation is dataset-dependent).
    conf = None
    if bb.ca_bfactor is not None:
        conf = bb.ca_bfactor[: cfg.max_len].astype("float32")

    (out_dir / "sequence.txt").write_text(seq + "\n")

    dist_targets = torch.from_numpy(dist_bins[: cfg.max_len, : cfg.max_len]).long()
    contact_targets = torch.from_numpy(contacts[: cfg.max_len, : cfg.max_len]).float()
    torsion_targets = torch.from_numpy(tors).float()

    torch.save(dist_targets, out_dir / "dist_targets.pt")
    torch.save(contact_targets, out_dir / "contact_targets.pt")
    torch.save(torsion_targets, out_dir / "torsion_targets.pt")

    if conf is not None:
        torch.save(torch.from_numpy(conf).float(), out_dir / "confidence_targets.pt")

    row = {
        "id": pdb_id,
        "structure_path": str(structure_path),
        "sequence_path": str(out_dir / "sequence.txt"),
        "dist_targets_path": str(out_dir / "dist_targets.pt"),
        "contact_targets_path": str(out_dir / "contact_targets.pt"),
        "torsion_targets_path": str(out_dir / "torsion_targets.pt"),
        "confidence_targets_path": str(out_dir / "confidence_targets.pt") if conf is not None else None,
    }

    if esm_checkpoint is not None:
        tokens, residue_mask = _tokenize_with_esm([seq], esm_checkpoint=esm_checkpoint, max_len=cfg.max_len)
        torch.save(tokens[0], out_dir / "tokens.pt")
        torch.save(residue_mask[0], out_dir / "residue_mask.pt")
        row["tokens_path"] = str(out_dir / "tokens.pt")
        row["residue_mask_path"] = str(out_dir / "residue_mask.pt")

    return row


def write_manifest(rows: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
