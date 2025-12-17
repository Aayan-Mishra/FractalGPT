import pytest

from fractal.data.rcsb import normalize_chem_comp_id, normalize_pdb_id


def test_normalize_pdb_id():
    assert normalize_pdb_id("1ABC") == "1abc"
    with pytest.raises(ValueError):
        normalize_pdb_id("abc")


def test_normalize_chem_comp_id():
    assert normalize_chem_comp_id("atp") == "ATP"
    with pytest.raises(ValueError):
        normalize_chem_comp_id("")
