import pytest
from typer.testing import CliRunner

from jaxoplanet2.cli import app
from jaxoplanet2.init import TEMPLATE_FILES, init_directory

runner = CliRunner()


def test_init_creates_template_files(tmp_path):
    written = init_directory(tmp_path / "fit")
    assert sorted(p.name for p in written) == sorted(TEMPLATE_FILES)
    for name in TEMPLATE_FILES:
        assert (tmp_path / "fit" / name).read_text().startswith("#name,value")


def test_init_refuses_to_overwrite(tmp_path):
    init_directory(tmp_path)
    (tmp_path / "params.csv").write_text("#name,value\ncustom,1\n")
    with pytest.raises(FileExistsError, match="params.csv"):
        init_directory(tmp_path)
    assert "custom" in (tmp_path / "params.csv").read_text()


def test_init_overwrite_replaces_files(tmp_path):
    (tmp_path / "params.csv").write_text("#name,value\ncustom,1\n")
    init_directory(tmp_path, overwrite=True)
    assert "custom" not in (tmp_path / "params.csv").read_text()


def test_cli_init_reports_next_steps(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path / "fit")])
    assert result.exit_code == 0
    assert "tess.csv" in result.output
    assert (tmp_path / "fit" / "settings.csv").exists()


def test_cli_init_existing_files_exit_nonzero(tmp_path):
    init_directory(tmp_path)
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 1
    assert "--overwrite" in result.output
