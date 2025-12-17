from __future__ import annotations

import json
import random
from pathlib import Path

import typer

from fractal.data.rcsb import RCSBDownloader

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    pdb_ids: Path = typer.Argument(..., help="Text file with one PDB id per line."),
    out_dir: Path = typer.Option(Path("data/raw/rcsb"), "--out-dir"),
    fmt: str = typer.Option("cif", help="cif|pdb"),
    ligands: Path | None = typer.Option(None, "--ligands", help="Optional file with chemical component IDs."),
    log: Path = typer.Option(Path("data/raw/rcsb/download_log.jsonl"), "--log"),
    max_ids: int = typer.Option(0, "--max-ids", help="If >0, only download up to this many structures."),
    sample: bool = typer.Option(False, "--sample", help="If set, randomly sample max_ids (instead of taking first N)."),
    seed: int = typer.Option(0, "--seed", help="Random seed for --sample."),
    workers: int = typer.Option(8, "--workers", help="Parallel download workers."),
    retries: int = typer.Option(3),
    timeout_s: float = typer.Option(30.0),
):
    pdb_list = [l.strip() for l in pdb_ids.read_text().splitlines() if l.strip()]

    if max_ids and max_ids > 0:
        if sample:
            rng = random.Random(seed)
            if max_ids < len(pdb_list):
                pdb_list = rng.sample(pdb_list, k=max_ids)
        else:
            pdb_list = pdb_list[:max_ids]
    ligand_list = None
    if ligands is not None:
        ligand_list = [l.strip() for l in ligands.read_text().splitlines() if l.strip()]

    dl = RCSBDownloader(out_dir=out_dir, retries=retries, timeout_s=timeout_s)
    results = dl.download_many(pdb_list, fmt=fmt, ligands=ligand_list, log_path=log, workers=workers)

    ok = sum(1 for r in results if r.ok)
    fail = len(results) - ok
    typer.echo(json.dumps({"ok": ok, "failed": fail, "log": str(log)}))


if __name__ == "__main__":
    app()
