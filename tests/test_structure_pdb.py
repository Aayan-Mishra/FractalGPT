import numpy as np

from fractal.geometry.structure import BackboneStructure


def test_to_pdb_writes(tmp_path):
    seq = "ACD"
    atoms = np.zeros((3, 4, 3), dtype=np.float32)
    s = BackboneStructure(sequence=seq, atoms=atoms)

    out = tmp_path / "x.pdb"
    s.to_pdb(out)
    txt = out.read_text()
    assert "ATOM" in txt
    assert txt.rstrip().endswith("END")
