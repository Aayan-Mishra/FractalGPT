from __future__ import annotations

from pathlib import Path

from fractal.inference.fasta import read_first_fasta_sequence
from fractal.models.constraint_predictor import ConstraintPredictor, ConstraintPredictorConfig
from fractal.models.types import ConstraintPredictions


def predict_constraints_from_fasta(
    *,
    fasta_path: str | Path,
    esm_checkpoint: str,
    num_distance_bins: int,
    device: str,
    max_len: int,
    checkpoint_dir: str | Path | None = None,
) -> ConstraintPredictions:
    """FASTA → constraint predictor outputs.

    This is intentionally separate from deterministic folding.
    """

    _, seq = read_first_fasta_sequence(fasta_path)

    if checkpoint_dir is not None:
        model = ConstraintPredictor.from_pretrained(checkpoint_dir, device=device)
        return model.predict_from_sequence(seq, device=device)

    cfg = ConstraintPredictorConfig(
        esm_checkpoint=esm_checkpoint,
        num_distance_bins=num_distance_bins,
        max_len=max_len,
    )
    model = ConstraintPredictor(cfg).to(device)

    return model.predict_from_sequence(seq, device=device)
