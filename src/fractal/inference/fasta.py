from __future__ import annotations

from pathlib import Path


def read_first_fasta_sequence(path: str | Path) -> tuple[str, str]:
    """Read the first FASTA record.

    Returns (header, sequence).

    Minimal parser suitable for research pipelines; upstream preprocessing can
    enforce canonical amino acid alphabets.
    """

    p = Path(path)
    header: str | None = None
    seq_parts: list[str] = []

    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                break
            header = line[1:].strip() or "sequence"
            continue
        seq_parts.append(line.replace(" ", ""))

    if header is None:
        raise ValueError(f"No FASTA header found in {p}")

    sequence = "".join(seq_parts).upper()
    if len(sequence) == 0:
        raise ValueError(f"Empty FASTA sequence in {p}")

    return header, sequence
