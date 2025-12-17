from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    n: int = typer.Option(10_000, "--n", help="How many structures to use end-to-end."),
    ids_file: Path = typer.Option(Path("pdb_ids_10k.txt"), "--ids", help="PDB ID file (will be created if missing)."),
    raw_dir: Path = typer.Option(Path("data/raw/rcsb_10k"), "--raw-dir"),
    processed_dir: Path = typer.Option(Path("data/processed_10k"), "--processed-dir"),
    fmt: str = typer.Option("cif", "--fmt"),
    esm_checkpoint: str = typer.Option("esm2_t30_150M_UR50D", "--esm"),
    max_len: int = typer.Option(1024, "--max-len"),
    train_config: Path = typer.Option(Path("configs/train.yaml"), "--train-config"),
    device: str = typer.Option("cpu", "--device", help="cpu|cuda"),
    fasta: Path | None = typer.Option(None, "--fasta", help="Optional FASTA to run inference after training."),
    out_pdb: Path = typer.Option(Path("outputs/infer.pdb"), "--out", help="Inference output PDB."),
):
    """Run FRACTAL end-to-end: fetch IDs -> download -> preprocess -> train -> infer.

    Notes:
    - Downloading + preprocessing 10k structures can take hours.
    - Training can take hours/days depending on your GPU.
    """

    # 1) Get IDs (if missing)
    if not ids_file.exists():
        typer.echo(f"ID file missing; generating {ids_file} via RCSB search...")
        from scripts.fetch_pdb_ids import main as fetch_ids

        fetch_ids(out=ids_file, limit=n)  # type: ignore

    # 2) Download
    typer.echo(f"Downloading up to {n} structures into {raw_dir}...")
    from scripts.download_rcsb import main as download

    download(pdb_ids=ids_file, out_dir=raw_dir, fmt=fmt, ligands=None, log=raw_dir / "download_log.jsonl", max_ids=n, sample=True, seed=0)  # type: ignore

    # 3) Preprocess
    typer.echo(f"Preprocessing into {processed_dir}...")
    from scripts.preprocess_pdb import main as preprocess

    preprocess(
        raw_dir=raw_dir,
        out_dir=processed_dir,
        fmt=fmt,
        esm_checkpoint=esm_checkpoint,
        chain_id=None,
        num_bins=64,
        max_len=max_len,
        seed=0,
        train_frac=0.9,
        val_frac=0.05,
    )  # type: ignore

    # 4) Train (with updated config)
    typer.echo("Training...")
    from scripts.train import main as train

    train(config=train_config, resume_from=None)  # type: ignore

    # 5) Inference (optional)
    if fasta is not None:
        typer.echo(f"Running inference on {fasta}...")
        from fractal.cli import fold

        checkpoint = Path("models/trained/best")
        fold(
            fasta=fasta,
            out_pdb=out_pdb,
            viz=True,
            out_png=None,
            out_html=None,
            checkpoint=checkpoint,
            device=device,
            esm_checkpoint=esm_checkpoint,
            num_bins=64,
            max_len=max_len,
            geom_steps=500,
        )  # type: ignore


if __name__ == "__main__":
    app()
