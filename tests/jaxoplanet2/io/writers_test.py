import pytest

from jaxoplanet2.io.params import load_params
from jaxoplanet2.io.writers import backup_params, write_param_values

TEXT = (
    "#name,value,fit,bounds,label,unit,truth\n"
    "#companion b,,,,,,\n"
    "b_rr,0.1,1,uniform 0 0.3,$R_b / R_\\star$,, # a note\n"
    "b_f_c,0,0,,f_c,,\n"
)


def test_only_value_cells_change(tmp_path):
    (tmp_path / "params.csv").write_text(TEXT)
    write_param_values(tmp_path / "params.csv", {"b_rr": 0.1234})
    lines = (tmp_path / "params.csv").read_text().splitlines()
    assert lines[2] == "b_rr,0.1234,1,uniform 0 0.3,$R_b / R_\\star$,, # a note"
    assert lines[:2] == TEXT.splitlines()[:2]
    assert lines[3] == "b_f_c,0,0,,f_c,,"
    assert load_params(tmp_path)["b_rr"].value == 0.1234


def test_unknown_names_are_an_error(tmp_path):
    (tmp_path / "params.csv").write_text(TEXT)
    with pytest.raises(KeyError, match="b_K"):
        write_param_values(tmp_path / "params.csv", {"b_K": 1.0})
    assert (tmp_path / "params.csv").read_text() == TEXT


def test_backup_is_made_once(tmp_path):
    (tmp_path / "params.csv").write_text(TEXT)
    assert backup_params(tmp_path) == tmp_path / "params.csv.orig"
    (tmp_path / "params.csv").write_text("changed")
    backup_params(tmp_path)
    assert (tmp_path / "params.csv.orig").read_text() == TEXT
