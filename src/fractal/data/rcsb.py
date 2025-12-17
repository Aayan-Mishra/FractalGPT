from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def normalize_pdb_id(pdb_id: str) -> str:
    pdb_id = pdb_id.strip().lower()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError(f"Invalid PDB id: {pdb_id!r}")
    return pdb_id


def normalize_chem_comp_id(comp_id: str) -> str:
    comp_id = comp_id.strip().upper()
    if len(comp_id) < 1 or len(comp_id) > 3 or not comp_id.isalnum():
        raise ValueError(f"Invalid chemical component id: {comp_id!r}")
    return comp_id


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    id: str
    path: str | None
    url: str
    error: str | None


class RCSBDownloader:
    """Download structures from RCSB PDB.

    Supports:
    - Structure files via files.rcsb.org/download/{pdb_id}.pdb|cif
    - Ligand chemical component definitions via files.rcsb.org/ligands/download/{comp_id}.cif

    Notes:
    - This module is intentionally IO-focused and does not parse structures.
    - Failures are captured as structured DownloadResult objects.
    """

    def __init__(
        self,
        *,
        out_dir: str | Path,
        timeout_s: float = 30.0,
        retries: int = 3,
        backoff_s: float = 1.0,
        user_agent: str = "FRACTAL/0.1 (+https://github.com)"
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _get(self, url: str) -> bytes:
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, timeout=self.timeout_s)
                if r.status_code == 200:
                    return r.content
                raise RuntimeError(f"HTTP {r.status_code}")
            except Exception as e:
                last_err = e
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff_s * (2**attempt))
        assert last_err is not None
        raise last_err

    def download_structure(self, pdb_id: str, *, fmt: str = "cif") -> DownloadResult:
        pdb_id_n = normalize_pdb_id(pdb_id)
        fmt_n = fmt.lower()
        if fmt_n not in {"pdb", "cif"}:
            raise ValueError("fmt must be 'pdb' or 'cif'")

        url = f"https://files.rcsb.org/download/{pdb_id_n}.{fmt_n}"
        out_path = self.out_dir / f"{pdb_id_n}.{fmt_n}"

        try:
            data = self._get(url)
            out_path.write_bytes(data)
            return DownloadResult(ok=True, id=pdb_id_n, path=str(out_path), url=url, error=None)
        except Exception as e:
            return DownloadResult(ok=False, id=pdb_id_n, path=None, url=url, error=str(e))

    def download_chem_comp(self, comp_id: str) -> DownloadResult:
        comp_id_n = normalize_chem_comp_id(comp_id)
        url = f"https://files.rcsb.org/ligands/download/{comp_id_n}.cif"
        out_path = self.out_dir / f"{comp_id_n}.cif"

        try:
            data = self._get(url)
            out_path.write_bytes(data)
            return DownloadResult(ok=True, id=comp_id_n, path=str(out_path), url=url, error=None)
        except Exception as e:
            return DownloadResult(ok=False, id=comp_id_n, path=None, url=url, error=str(e))

    def download_many(
        self,
        pdb_ids: list[str],
        *,
        fmt: str = "cif",
        ligands: list[str] | None = None,
        log_path: str | Path | None = None,
        workers: int = 8,
    ) -> list[DownloadResult]:
        results: list[DownloadResult] = []

        workers_i = max(1, int(workers))

        # Download structures in parallel for speed.
        with ThreadPoolExecutor(max_workers=workers_i) as ex:
            futures = {ex.submit(self.download_structure, pid, fmt=fmt): pid for pid in pdb_ids}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    pid = futures[fut]
                    # Should be rare; download_structure already catches most errors.
                    results.append(
                        DownloadResult(ok=False, id=str(pid).lower(), path=None, url="", error=str(e))
                    )

        if ligands:
            with ThreadPoolExecutor(max_workers=workers_i) as ex:
                futures = {ex.submit(self.download_chem_comp, cid): cid for cid in ligands}
                for fut in as_completed(futures):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        cid = futures[fut]
                        results.append(
                            DownloadResult(ok=False, id=str(cid).upper(), path=None, url="", error=str(e))
                        )

        if log_path is not None:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w") as f:
                for r in results:
                    f.write(json.dumps(r.__dict__) + "\n")

        return results
