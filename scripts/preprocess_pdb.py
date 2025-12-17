from __future__ import annotations

import random
from pathlib import Path

import typer

from fractal.data.preprocess import PreprocessConfig, preprocess_structure, write_manifest

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    raw_dir: Path = typer.Argument(..., help="Directory containing downloaded .pdb/.cif files."),
    out_dir: Path = typer.Option(Path("data/processed"), "--out-dir"),
    fmt: str = typer.Option("cif", help="Expected structure format: cif|pdb"),
    esm_checkpoint: str | None = typer.Option(
        None,
        "--esm-checkpoint",
        help="Optional: tokenize with ESM alphabet and write tokens.pt/residue_mask.pt",
    ),
    chain_id: str | None = typer.Option(None, "--chain", help="Optional chain id."),
    num_bins: int = typer.Option(64, help="Distance bins."),
    max_len: int = typer.Option(1024),
    seed: int = typer.Option(0),
    train_frac: float = typer.Option(0.9),
    val_frac: float = typer.Option(0.05),
):
    """Parse structures → precomputed supervision tensors + JSONL manifests."""

    cfg = PreprocessConfig(num_distance_bins=num_bins, max_len=max_len)

    paths = sorted(raw_dir.glob(f"*.{fmt}"))
    if len(paths) == 0:
        raise typer.BadParameter(f"No '*.{fmt}' files found in {raw_dir}")

    rows: list[dict] = []
    failures: list[dict] = []

    for p in paths:
        pdb_id = p.stem.lower()
        try:
            row = preprocess_structure(
                structure_path=p,
                out_dir=out_dir,
                pdb_id=pdb_id,
                cfg=cfg,
                chain_id=chain_id,
                esm_checkpoint=esm_checkpoint,
            )
            rows.append(row)
        except Exception as e:
            failures.append({"id": pdb_id, "path": str(p), "error": str(e)})

    rng = random.Random(seed)
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]

    write_manifest(train, out_dir / "train_manifest.jsonl")
    write_manifest(val, out_dir / "val_manifest.jsonl")
    write_manifest(test, out_dir / "test_manifest.jsonl")

    failures_path = out_dir / "failures.jsonl"
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.write_text(
        "\n".join([__import__("json").dumps(x) for x in failures]) + ("\n" if failures else "")
    )

    typer.echo(
        f"processed={n} train={len(train)} val={len(val)} test={len(test)} failures={len(failures)}"
    )


if __name__ == "__main__":
    app()
