from __future__ import annotations

from pathlib import Path

import typer

from fractal.geometry.folding import fold_from_constraints
from fractal.inference.pipeline import predict_constraints_from_fasta
from fractal.visualization import render_structure_png, write_ngl_html

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    fasta: Path = typer.Argument(...),
    out: Path = typer.Option(..., "--out"),
    viz: bool = typer.Option(True, "--viz/--no-viz"),
    out_png: Path | None = typer.Option(None, "--out-png"),
    out_html: Path | None = typer.Option(None, "--out-html"),
    checkpoint: Path | None = typer.Option(None, "--checkpoint"),
    esm_checkpoint: str = typer.Option("esm2_t30_150M_UR50D"),
    device: str = typer.Option("cpu"),
    num_bins: int = typer.Option(64),
    max_len: int = typer.Option(1024),
    steps: int = typer.Option(500),
):
    constraints = predict_constraints_from_fasta(
        fasta_path=fasta,
        esm_checkpoint=esm_checkpoint,
        num_distance_bins=num_bins,
        device=device,
        max_len=max_len,
        checkpoint_dir=checkpoint,
    )
    struct = fold_from_constraints(
        sequence=constraints.sequence,
        distance_logits=constraints.distance_logits,
        torsion_angles=constraints.torsion_angles,
        contact_logits=constraints.contact_logits,
        confidence=constraints.confidence,
        steps=steps,
    )
    struct.to_pdb(out)

    if viz:
        png_path = out_png if out_png is not None else out.with_suffix(".png")
        html_path = out_html if out_html is not None else out.with_suffix(".html")

        try:
            write_ngl_html(pdb_path=out, out_html=html_path, title=f"FRACTAL: {out.name}")
        except Exception as e:
            typer.echo(f"[viz] Failed to write HTML viewer: {e}")

        try:
            render_structure_png(structure=struct, out_png=png_path)
        except Exception as e:
            typer.echo(f"[viz] Failed to write PNG (install '.[viz]'?): {e}")


if __name__ == "__main__":
    app()
