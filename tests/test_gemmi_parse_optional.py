import pytest


gemmi = pytest.importorskip("gemmi")

from fractal.data.gemmi_io import load_backbone_atoms


def test_gemmi_parses_minimal_pdb(tmp_path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N",
                "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 10.00           C",
                "ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00 10.00           C",
                "ATOM      4  O   ALA A   1       2.000   2.600   0.000  1.00 10.00           O",
                "ATOM      5  N   GLY A   2       3.200   1.400   0.000  1.00 20.00           N",
                "ATOM      6  CA  GLY A   2       3.800   2.700   0.000  1.00 20.00           C",
                "ATOM      7  C   GLY A   2       5.300   2.700   0.000  1.00 20.00           C",
                "ATOM      8  O   GLY A   2       5.900   3.800   0.000  1.00 20.00           O",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    bb = load_backbone_atoms(pdb, chain_id="A")
    assert bb.sequence == "AG"
    assert bb.ca.shape == (2, 3)
    assert bb.ca_bfactor is not None
    assert bb.ca_bfactor.shape == (2,)
