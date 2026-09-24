from typer.testing import CliRunner

import jaxoplanet2
from jaxoplanet2.cli import app

runner = CliRunner()


def test_help_lists_program_name():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "jaxoplanet" in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_version_flag_prints_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert jaxoplanet2.__version__ in result.output
