from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fractal.geometry.structure import BackboneStructure


def write_ngl_html(*, pdb_path: str | Path, out_html: str | Path, title: str = "FRACTAL Structure") -> Path:
    """Write a self-contained HTML file that renders a PDB using NGL.

    This produces an interactive 3D viewer in the browser without requiring
    any Python visualization dependencies.
    """

    pdb_path = Path(pdb_path)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    pdb_text = pdb_path.read_text()
    pdb_json = json.dumps(pdb_text)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    html, body {{ height: 100%; margin: 0; font-family: monospace; }}
    #viewport {{ width: 100%; height: calc(100% - 60px); }}
    #info {{ 
      position: fixed; 
      bottom: 0; 
      left: 0; 
      width: 100%; 
      height: 60px; 
      background: rgba(0,0,0,0.85); 
      color: #0f0; 
      padding: 12px 20px; 
      box-sizing: border-box;
      font-size: 14px;
      line-height: 1.5;
      overflow: auto;
    }}
    #info span {{ color: #0ff; font-weight: bold; }}
  </style>
  <script src=\"https://unpkg.com/ngl@latest/dist/ngl.js\"></script>
</head>
<body>
  <div id=\"viewport\"></div>
  <div id=\"info\">Click on atoms to see details...</div>
  <script>
    const pdbText = {pdb_json};
    const blob = new Blob([pdbText], {{ type: 'text/plain' }});
    const stage = new NGL.Stage('viewport');
    const infoBox = document.getElementById('info');

    window.addEventListener('resize', function () {{ stage.handleResize(); }}, false);

    stage.loadFile(blob, {{ ext: 'pdb' }}).then(function (o) {{
      o.addRepresentation('cartoon', {{ colorScheme: 'chainindex' }});
      o.addRepresentation('ball+stick', {{ sele: 'hetero and not water' }});
      
      // Add click handler for atom picking
      stage.signals.clicked.add(function (pickingProxy) {{
        if (pickingProxy && pickingProxy.atom) {{
          const atom = pickingProxy.atom;
          const resname = atom.resname || 'UNK';
          const resno = atom.resno !== undefined ? atom.resno : '?';
          const chain = atom.chainname || 'A';
          const atomname = atom.atomname || '?';
          const element = atom.element || '?';
          
          infoBox.innerHTML = 
            '<span>Atom:</span> ' + atomname + 
            ' &nbsp;&nbsp;<span>Residue:</span> ' + resname + 
            ' &nbsp;&nbsp;<span>Position:</span> ' + resno + 
            ' &nbsp;&nbsp;<span>Chain:</span> ' + chain +
            ' &nbsp;&nbsp;<span>Element:</span> ' + element;
        }} else {{
          infoBox.innerHTML = 'Click on atoms to see details...';
        }}
      }});
      
      stage.autoView();
    }});
  </script>
</body>
</html>
"""

    out_html.write_text(html)
    return out_html


def render_structure_png(*, structure: "BackboneStructure", out_png: str | Path, dpi: int = 200) -> Path:
    """Render a quick-look PNG (3D scatter) of the folded backbone.

    Requires matplotlib. If matplotlib isn't installed, this raises ImportError.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PNG rendering requires matplotlib. Install optional deps: pip install -e '.[viz]'"
        ) from e

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Use CA atoms for a clean backbone trace.
    coords = np.asarray(structure.atoms, dtype=np.float32)
    ca = coords[:, 1, :]  # (L,3)

    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(ca[:, 0], ca[:, 1], ca[:, 2], linewidth=1.5)
    ax.scatter(ca[:, 0], ca[:, 1], ca[:, 2], s=6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("FRACTAL fold")

    # Rough equal aspect for 3D.
    mins = ca.min(axis=0)
    maxs = ca.max(axis=0)
    centers = 0.5 * (mins + maxs)
    span = float((maxs - mins).max())
    if span > 0:
        ax.set_xlim(centers[0] - span / 2, centers[0] + span / 2)
        ax.set_ylim(centers[1] - span / 2, centers[1] + span / 2)
        ax.set_zlim(centers[2] - span / 2, centers[2] + span / 2)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png
