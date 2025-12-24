from __future__ import annotations

from pathlib import Path

import torch
import typer

from fractal.geometry.folding import fold_from_constraints
from fractal.inference.pipeline import predict_constraints_from_fasta
from fractal.utils import ensure_parent_dir
from fractal.visualization import render_structure_png, write_ngl_html

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("webui")
def webui(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (default: 127.0.0.1)."),
    port: int = typer.Option(8000, "--port", help="Bind port (default: 8000)."),
    reload: bool = typer.Option(True, "--reload/--no-reload", help="Auto-reload on code changes."),
    share: bool = typer.Option(False, "--share", help="Create a public URL via ngrok (for Kaggle/Colab)."),
):
    """Start the local FastAPI WebUI server.
    
    Use --share to get a public URL for cloud environments like Kaggle or Colab.
    """

    try:
        import uvicorn  # type: ignore
    except Exception as e:
        raise typer.BadParameter(
            "WebUI dependencies are not installed. Run: pip install -e '.[webui]'"
        ) from e

    if share:
        # Public tunnel mode for cloud environments
        try:
            from pyngrok import ngrok  # type: ignore
        except ImportError:
            typer.echo("Installing pyngrok for public URL...")
            import subprocess
            subprocess.check_call(["pip", "install", "pyngrok"])
            from pyngrok import ngrok  # type: ignore

        # Start ngrok tunnel
        public_url = ngrok.connect(port, bind_tls=True)
        typer.echo("")
        typer.echo("=" * 60)
        typer.echo("🌐 FRACTAL WebUI - Public URL")
        typer.echo("=" * 60)
        typer.echo(f"  Public URL: {public_url}")
        typer.echo(f"  Local URL:  http://{host}:{port}")
        typer.echo("=" * 60)
        typer.echo("")
        
        # In share mode, bind to 0.0.0.0 and disable reload
        uvicorn.run(
            "fractal.webui.app:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info",
        )
    else:
        # Local mode
        typer.echo("")
        typer.echo(f"🧬 FRACTAL WebUI running at http://{host}:{port}")
        typer.echo("")
        uvicorn.run(
            "fractal.webui.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )


@app.command("fold")
def fold(
    fasta: Path = typer.Argument(..., help="Input FASTA file."),
    out_pdb: Path | None = typer.Option(None, "--out", help="Output PDB path (defaults to input.pdb)."),
    viz: bool = typer.Option(True, "--viz/--no-viz", help="Also write HTML (3D) + PNG alongside PDB."),
    out_png: Path | None = typer.Option(None, "--out-png", help="Optional output PNG path."),
    out_html: Path | None = typer.Option(None, "--out-html", help="Optional output HTML (NGL viewer) path."),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", help="Path to a trained checkpoint directory (uses trained heads)."),
    device: str = typer.Option("cpu", help="Inference device: cpu|mps|cuda"),
    esm_checkpoint: str = typer.Option(
        "esm2_t30_150M_UR50D", help="ESM-2 backbone identifier."
    ),
    num_bins: int = typer.Option(64, help="Distance distribution bins."),
    max_len: int = typer.Option(1024, help="Max sequence length (truncate)."),
    geom_steps: int = typer.Option(500, help="Geometry optimization steps."),
):
    """FASTA → constraints (ESM2 heads) → deterministic geometry → PDB."""

    # Default output path to input.pdb if not specified
    if out_pdb is None:
        out_pdb = fasta.with_suffix(".pdb")

    ensure_parent_dir(out_pdb)

    constraints = predict_constraints_from_fasta(
        fasta_path=fasta,
        esm_checkpoint=esm_checkpoint,
        num_distance_bins=num_bins,
        device=device,
        max_len=max_len,
        checkpoint_dir=checkpoint,
    )

    structure = fold_from_constraints(
        sequence=constraints.sequence,
        distance_logits=constraints.distance_logits,
        torsion_angles=constraints.torsion_angles,
        contact_logits=constraints.contact_logits,
        confidence=constraints.confidence,
        steps=geom_steps,
    )

    structure.to_pdb(out_pdb)

    if viz:
        png_path = out_png if out_png is not None else out_pdb.with_suffix(".png")
        html_path = out_html if out_html is not None else out_pdb.with_suffix(".html")

        # Always try to write the HTML (no extra deps).
        try:
            write_ngl_html(pdb_path=out_pdb, out_html=html_path, title=f"FRACTAL: {out_pdb.name}")
        except Exception as e:
            typer.echo(f"[viz] Failed to write HTML viewer: {e}")

        # PNG requires matplotlib; fail gracefully.
        try:
            render_structure_png(structure=structure, out_png=png_path)
        except Exception as e:
            typer.echo(f"[viz] Failed to write PNG (install '.[viz]'?): {e}")


@app.command("predict")
def predict(
    fasta: Path = typer.Argument(..., help="Input FASTA file."),
    out_pt: Path = typer.Option(..., "--out", help="Output constraints .pt path."),
    viz: bool = typer.Option(
        False,
        "--viz/--no-viz",
        help="Also fold and write PDB + HTML (3D) + PNG alongside the .pt.",
    ),
    viz_steps: int = typer.Option(500, help="Folding steps if --viz is enabled."),
    out_pdb: Path | None = typer.Option(None, "--out-pdb", help="Optional output PDB path (viz mode)."),
    out_png: Path | None = typer.Option(None, "--out-png", help="Optional output PNG path (viz mode)."),
    out_html: Path | None = typer.Option(None, "--out-html", help="Optional output HTML path (viz mode)."),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", help="Path to a trained checkpoint directory (uses trained heads)."),
    device: str = typer.Option("cpu", help="Inference device: cpu|mps|cuda"),
    esm_checkpoint: str = typer.Option(
        "esm2_t30_150M_UR50D", help="ESM-2 backbone identifier."
    ),
    num_bins: int = typer.Option(64, help="Distance distribution bins."),
    max_len: int = typer.Option(1024, help="Max sequence length (truncate)."),
):
    """FASTA → constraints (.pt) without folding."""

    ensure_parent_dir(out_pt)
    constraints = predict_constraints_from_fasta(
        fasta_path=fasta,
        esm_checkpoint=esm_checkpoint,
        num_distance_bins=num_bins,
        device=device,
        max_len=max_len,
        checkpoint_dir=checkpoint,
    )

    payload = {
        "sequence": constraints.sequence,
        "distance_logits": constraints.distance_logits.cpu(),
        "contact_logits": constraints.contact_logits.cpu(),
        "torsion_angles": constraints.torsion_angles.cpu(),
        "confidence": constraints.confidence.cpu() if constraints.confidence is not None else None,
        "residue_mask": constraints.residue_mask.cpu(),
    }
    torch.save(payload, out_pt)

    if viz:
        pdb_path = out_pdb if out_pdb is not None else out_pt.with_suffix(".pdb")
        png_path = out_png if out_png is not None else out_pt.with_suffix(".png")
        html_path = out_html if out_html is not None else out_pt.with_suffix(".html")

        ensure_parent_dir(pdb_path)
        structure = fold_from_constraints(
            sequence=constraints.sequence,
            distance_logits=constraints.distance_logits,
            torsion_angles=constraints.torsion_angles,
            contact_logits=constraints.contact_logits,
            confidence=constraints.confidence,
            steps=viz_steps,
        )
        structure.to_pdb(pdb_path)

        try:
            write_ngl_html(pdb_path=pdb_path, out_html=html_path, title=f"FRACTAL: {pdb_path.name}")
        except Exception as e:
            typer.echo(f"[viz] Failed to write HTML viewer: {e}")

        try:
            render_structure_png(structure=structure, out_png=png_path)
        except Exception as e:
            typer.echo(f"[viz] Failed to write PNG (install '.[viz]'?): {e}")


if __name__ == "__main__":
    app()
