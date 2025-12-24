from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, FileResponse

from fractal.geometry.folding import fold_from_constraints
from fractal.inference.pipeline import predict_constraints_from_fasta
from fractal.utils import ensure_parent_dir
from fractal.visualization import write_ngl_html


@dataclass
class RunArtifacts:
    root: Path
    pdb_path: Path
    html_path: Path


_RUNS: dict[str, RunArtifacts] = {}


def _page(title: str, body_html: str) -> str:
    css = """
    :root {
        --bg: #0f0f0f;
        --panel: #151515;
        --panel-2: #0a0a0a;
        --border: #2a2a2a;
        --text: #f2f2f2;
        --muted: #b7b7b7;
        --accent: #65CBF3;
        --danger: #FF7D45;
        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font-family: var(--sans);
        line-height: 1.45;
    }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }

    .wrap { max-width: 1080px; margin: 0 auto; padding: 32px 24px 60px; }
    header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 24px;
        padding-bottom: 18px;
        border-bottom: 1px solid var(--border);
        flex-wrap: wrap;
    }
    h1 { font-size: 24px; margin: 0; letter-spacing: 0.3px; font-weight: 700; }
    .subtitle { color: var(--muted); font-size: 14px; margin: 6px 0 0; }
    .badge {
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
        border: 1px solid var(--border);
        padding: 8px 12px;
        border-radius: 8px;
        background: var(--panel-2);
        white-space: nowrap;
    }

    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0 24px; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }

    .card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px;
    }
    .card h2 { font-size: 13px; margin: 0 0 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .kv { display: grid; grid-template-columns: 170px 1fr; gap: 8px 12px; }
    @media (max-width: 520px) { .kv { grid-template-columns: 1fr; } }
    .k { color: var(--muted); font-size: 12px; }
    .v { font-family: var(--mono); font-size: 12px; }
    .ok { color: #7CFCB2; }
    .no { color: var(--danger); }

    form { margin-top: 14px; }
    fieldset {
        border: 1px solid var(--border);
        background: var(--panel);
        border-radius: 12px;
        padding: 20px;
        margin: 0 0 18px;
    }
    legend { padding: 0 10px; color: var(--muted); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 860px) { .row { grid-template-columns: 1fr; } }

    label { display: block; font-size: 12px; color: var(--muted); margin: 12px 0 8px; font-weight: 500; }
    input[type="text"], input[type="number"], select, textarea {
        width: 100%;
        border: 1px solid var(--border);
        background: var(--panel-2);
        color: var(--text);
        border-radius: 10px;
        padding: 11px 14px;
        outline: none;
        font-family: var(--mono);
        font-size: 13px;
        transition: border-color 0.2s;
    }
    input:focus, select:focus, textarea:focus { border-color: var(--accent); }
    textarea { font-family: var(--mono); resize: vertical; }
    input[type="file"] { width: 100%; color: var(--muted); font-size: 13px; }

    .help { color: var(--muted); font-size: 12px; margin-top: 8px; }
    .help code { font-family: var(--mono); color: var(--text); }

    .actions { display: flex; align-items: center; gap: 16px; margin-top: 20px; flex-wrap: wrap; }
    button {
        border: 1px solid var(--accent);
        background: linear-gradient(135deg, #0b2a33 0%, #0f3541 100%);
        color: var(--text);
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 700;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.2s;
    }
    button:hover { 
        border-color: var(--accent); 
        background: linear-gradient(135deg, #0f3541 0%, #134250 100%);
        box-shadow: 0 0 12px rgba(101, 203, 243, 0.3);
    }
    .muted { color: var(--muted); font-size: 12px; }

    pre {
        border: 1px solid var(--border);
        background: var(--panel-2);
        padding: 12px;
        border-radius: 12px;
        overflow: auto;
        font-family: var(--mono);
        font-size: 12px;
        color: var(--text);
    }
    """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">{body_html}</div>
</body>
</html>"""


def create_app() -> FastAPI:
    app = FastAPI(title="fractal-ui")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        cuda_ok = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_ok else 0

        # Choose a sensible default device for the dropdown.
        default_device = "cpu"
        if cuda_ok:
            default_device = "cuda"
        elif hasattr(torch.backends, "mps") and bool(torch.backends.mps.is_available()):
            default_device = "mps"

        ok_class = "ok" if cuda_ok else "no"
        body = f"""
<header>
<div>
    <h1>FRACTAL Web UI</h1>
    <div class="subtitle">FRACTAL inference + deterministic folding (PDB + HTML viewer)</div>
</div>
<div class="badge">Runs stored in <span style="color: var(--text)">ui_runs/</span></div>
</header>

<div class="grid">
<div class="card">
    <h2>System</h2>
    <div class="kv">
        <div class="k">CUDA available</div>
        <div class="v {ok_class}">{cuda_ok}</div>
        <div class="k">CUDA device count</div>
        <div class="v">{cuda_count}</div>
    </div>
    <div class="help">If CUDA is <code>False</code> on a GPU machine, you likely installed a CPU-only PyTorch wheel.</div>
</div>
<div class="card">
    <h2>Notes</h2>
    <div class="help">
        - Provide a trained checkpoint (local path) or a Hugging Face repo id.<br/>
        - If checkpoint is empty, heads are random (output will be meaningless).<br/>
        - The HTML viewer uses NGL in your browser.
    </div>
</div>
</div>

<form action="/run" method="post" enctype="multipart/form-data">
<fieldset>
    <legend>Model</legend>
    <div class="kv" style="background: var(--panel-2); padding: 12px; border-radius: 8px; margin-top: 4px;">
        <div class="k">Using model</div>
        <div class="v" style="color: var(--accent); font-weight: 600;">Fractal-Labs/FRACTAL-1-3B</div>
    </div>
    <div class="help" style="margin-top: 10px;">This model is automatically loaded from Hugging Face. No configuration needed.</div>
</fieldset>

<fieldset>
    <legend>Input</legend>
    <div class="row">
        <div>
            <label for="fasta_file">Upload FASTA file</label>
            <input id="fasta_file" type="file" name="fasta_file" accept=".fasta,.fa,.faa,.fna,text/plain" />
            <div class="help">If you upload a file, pasted text is ignored.</div>
        </div>
        <div>
            <label for="fasta_text">Paste FASTA</label>
            <textarea id="fasta_text" name="fasta_text" rows="10" placeholder=">protein\nMADEUPSEQUENCE..."></textarea>
        </div>
    </div>
</fieldset>

<fieldset>
    <legend>Options</legend>
    <div class="row">
        <div>
            <label for="device">Device</label>
            <select id="device" name="device">
                <option value="cpu" {"selected" if default_device == "cpu" else ""}>cpu</option>
                <option value="mps" {"selected" if default_device == "mps" else ""}>mps</option>
                <option value="cuda" {"selected" if default_device == "cuda" else ""}>cuda</option>
            </select>
        </div>
        <div>
            <label for="max_len">Max length</label>
            <input id="max_len" type="number" name="max_len" value="1024" min="32" max="4096" />
        </div>
    </div>

    <div class="row">
        <div>
            <label for="num_bins">Distance bins</label>
            <input id="num_bins" type="number" name="num_bins" value="64" min="16" max="256" />
        </div>
        <div>
            <label for="geom_steps">Folding steps</label>
            <input id="geom_steps" type="number" name="geom_steps" value="500" min="50" max="5000" />
        </div>
    </div>
    <div class="help" style="margin-top: 8px;">Defaults are tuned for a quick demo run.</div>
</fieldset>

<div class="actions">
    <button type="submit">Run inference</button>
    <div class="muted">Outputs: <span style="font-family: var(--mono)">output.pdb</span> + <span style="font-family: var(--mono)">output.html</span></div>
</div>
</form>
"""
        # THIS RETURN MUST BE INDENTED INSIDE index() FUNCTION
        return _page("FRACTAL Web UI", body)

    @app.post("/run", response_class=HTMLResponse)
    async def run(
        device: str = Form("cuda"),
        max_len: int = Form(1024),
        num_bins: int = Form(64),
        geom_steps: int = Form(500),
        fasta_text: str = Form(""),
        fasta_file: UploadFile | None = File(None),
    ) -> str:
        if (fasta_file is None or fasta_file.filename is None or fasta_file.filename == "") and not fasta_text.strip():
            raise HTTPException(status_code=400, detail="Provide a FASTA upload or pasted FASTA text.")

        run_id = uuid.uuid4().hex[:12]
        root = Path("ui_runs") / run_id
        root.mkdir(parents=True, exist_ok=True)

        fasta_path = root / "input.fasta"
        if fasta_file is not None and fasta_file.filename:
            raw = await fasta_file.read()
            try:
                txt = raw.decode("utf-8")
            except Exception:
                txt = raw.decode("latin-1")
            fasta_path.write_text(txt)
        else:
            fasta_path.write_text(fasta_text.strip() + "\n")

        out_pdb = root / "output.pdb"
        out_html = root / "output.html"
        ensure_parent_dir(out_pdb)

        # Fixed model: Fractal-Labs/FRACTAL-1-3B
        ckpt_arg: str | Path | None = "Fractal-Labs/FRACTAL-1-3B"

        try:
            constraints = predict_constraints_from_fasta(
                fasta_path=fasta_path,
                esm_checkpoint="esm2_t30_150M_UR50D",
                num_distance_bins=int(num_bins),
                device=str(device),
                max_len=int(max_len),
                checkpoint_dir=ckpt_arg,
            )
        except Exception as e:
            body = f"""
<header>
<div>
    <h1>Run failed</h1>
    <div class="subtitle">Inference crashed before folding</div>
</div>
<div class="badge">run_id: {run_id}</div>
</header>

<p class="muted">Error:</p>
<pre>{type(e).__name__}: {e}</pre>
<p><a href="/">Back</a></p>
"""
            return _page("Run failed", body)

        try:
            structure = fold_from_constraints(
                sequence=constraints.sequence,
                distance_logits=constraints.distance_logits,
                torsion_angles=constraints.torsion_angles,
                contact_logits=constraints.contact_logits,
                confidence=constraints.confidence,
                steps=int(geom_steps),
            )
            structure.to_pdb(out_pdb)
            write_ngl_html(pdb_path=out_pdb, out_html=out_html, title=f"FRACTAL: {run_id}")
        except Exception as e:
            body = f"""
<header>
<div>
    <h1>Run failed</h1>
    <div class="subtitle">Folding or serialization crashed after inference</div>
</div>
<div class="badge">run_id: {run_id}</div>
</header>

<p class="muted">Error:</p>
<pre>{type(e).__name__}: {e}</pre>
<p><a href="/">Back</a></p>
"""
            return _page("Run failed", body)

        _RUNS[run_id] = RunArtifacts(root=root, pdb_path=out_pdb, html_path=out_html)

        body = f"""
<header>
<div>
    <h1>Run complete</h1>
    <div class="subtitle">Artifacts generated successfully</div>
</div>
<div class="badge">run_id: {run_id}</div>
</header>

<div class="card">
<h2>Downloads</h2>
<div class="help">
    <a href="/download/{run_id}/output.pdb">Download PDB</a><br/>
    <a href="/download/{run_id}/output.html">Open HTML viewer</a>
</div>
</div>

<p><a href="/">Run another</a></p>
"""
        return _page("Run complete", body)

    @app.get("/download/{run_id}/{filename}")
    def download(run_id: str, filename: str):
        art = _RUNS.get(run_id)
        if art is None:
            raise HTTPException(status_code=404, detail="Unknown run_id")

        # Only allow the two known filenames.
        if filename == "output.pdb":
            return FileResponse(path=str(art.pdb_path), media_type="chemical/x-pdb")
        if filename == "output.html":
            return FileResponse(path=str(art.html_path), media_type="text/html")

        raise HTTPException(status_code=404, detail="Unknown file")

    return app


app = create_app()