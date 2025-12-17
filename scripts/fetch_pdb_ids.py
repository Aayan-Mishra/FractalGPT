from __future__ import annotations

import json
from pathlib import Path

import requests
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _query_payload(*, start: int, rows: int, method: str | None, max_resolution: float | None) -> dict:
    nodes = []

    if method is not None:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match",
                    "value": method,
                },
            }
        )

    if max_resolution is not None:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": float(max_resolution),
                },
            }
        )

    # Use a single terminal query if only one filter, otherwise group
    if len(nodes) == 1:
        query = nodes[0]
    else:
        query = {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes,
        }
    
    return {
        "query": query,
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": int(start), "rows": int(rows)},
        },
    }


@app.command()
def main(
    out: Path = typer.Option(Path("pdb_ids_10k.txt"), "--out", help="Output file for PDB IDs."),
    limit: int = typer.Option(10_000, "--limit", help="Number of PDB IDs to fetch."),
    method: str = typer.Option("X-RAY DIFFRACTION", "--method", help="Experimental method filter."),
    max_resolution: float | None = typer.Option(3.0, "--max-resolution", help="Max resolution cutoff (Angstrom)."),
    rows_per_page: int = typer.Option(1000, "--page-size", help="IDs per request (<= 10000)."),
    timeout_s: float = typer.Option(30.0, "--timeout", help="HTTP timeout seconds."),
):
    """Fetch PDB entry IDs from the RCSB Search API.

    This solves the practical problem that you currently only have ~5 IDs in pdb_ids.txt.
    """

    out.parent.mkdir(parents=True, exist_ok=True)

    url = "https://search.rcsb.org/rcsbsearch/v2/query"

    ids: list[str] = []
    start = 0
    page_size = max(1, min(int(rows_per_page), 10_000))

    while len(ids) < int(limit):
        payload = _query_payload(start=start, rows=page_size, method=method, max_resolution=max_resolution)
        r = requests.post(url, json=payload, timeout=float(timeout_s))
        r.raise_for_status()
        data = r.json()

        results = data.get("result_set", [])
        if not results:
            break

        for item in results:
            ident = item.get("identifier")
            if isinstance(ident, str) and len(ident) == 4:
                ids.append(ident)
                if len(ids) >= int(limit):
                    break

        start += page_size

    out.write_text("\n".join(ids) + "\n")
    typer.echo(json.dumps({"count": len(ids), "out": str(out)}))


if __name__ == "__main__":
    app()
