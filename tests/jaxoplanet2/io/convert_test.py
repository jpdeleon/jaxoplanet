import math
import shutil
from pathlib import Path

import jax
import pytest
from typer.testing import CliRunner

from jaxoplanet2.cli import app
from jaxoplanet2.io.params import load_params
from jaxoplanet2.model.parameterization import companion_geometry
from tests.jaxoplanet2.native import from_allesfitter

jax.config.update("jax_enable_x64", True)
runner = CliRunner()
CASE = Path(__file__).parents[1] / "golden" / "cases" / "eccentric_rv_batman"
LEGACY = {
    "b_rr": 0.1,
    "b_rsuma": 0.12,
    "b_cosi": 0.05,
    "b_epoch": 2459000.3,
    "b_period": 4.0,
    "b_f_c": 0.3,
    "b_f_s": 0.4,
}


@pytest.fixture
def fit_dir(tmp_path):
    dest = tmp_path / "fit"
    shutil.copytree(CASE, dest, ignore=shutil.ignore_patterns("results"))
    return dest


def test_cli_converts_values_and_keeps_a_backup(fit_dir):
    original = (fit_dir / "params.csv").read_text()

    result = runner.invoke(app, ["convert-params", str(fit_dir)])

    assert result.exit_code == 0, result.output
    assert "b_duration: new prior uniform" in result.output
    assert "review them before fitting" in result.output
    assert (fit_dir / "params.csv.orig").read_text() == original
    params = load_params(fit_dir)
    expected = from_allesfitter(LEGACY)
    for name in ("radius_ratio", "duration", "impact_param", "time_transit"):
        assert params[f"b_{name}"].value == pytest.approx(expected[f"b_{name}"])
    # 12 digits: the conversion runs in float64 even before x64 is configured
    assert params["b_duration"].value == pytest.approx(expected["b_duration"], rel=1e-12)
    for legacy in ("b_rr", "b_rsuma", "b_cosi", "b_epoch"):
        assert legacy not in params


def test_converted_values_reproduce_the_allesfitter_orbit(fit_dir):
    runner.invoke(app, ["convert-params", str(fit_dir)])

    values = {p.name: p.value for p in load_params(fit_dir)}
    g = companion_geometry(values, "b")

    assert float(g.a_over_rstar) == pytest.approx(1.1 / 0.12, rel=1e-12)
    assert float(g.inclination) == pytest.approx(math.acos(0.05))


def test_second_conversion_is_a_no_op(fit_dir):
    runner.invoke(app, ["convert-params", str(fit_dir)])
    converted = (fit_dir / "params.csv").read_text()

    result = runner.invoke(app, ["convert-params", str(fit_dir)])

    assert result.exit_code == 0
    assert "already uses the native parameterization" in result.output
    assert (fit_dir / "params.csv").read_text() == converted


def test_non_transiting_geometry_is_an_error(fit_dir):
    path = fit_dir / "params.csv"
    path.write_text(path.read_text().replace("b_cosi,0.05,", "b_cosi,0.9,"))

    result = runner.invoke(app, ["convert-params", str(fit_dir)])

    assert result.exit_code == 1
    assert "do not transit" in result.output
    assert not (fit_dir / "params.csv.orig").exists()


def test_conversion_is_float64_even_with_x64_disabled(fit_dir):
    with jax.enable_x64(False):
        runner.invoke(app, ["convert-params", str(fit_dir)])

    duration = load_params(fit_dir)["b_duration"].value

    assert duration == pytest.approx(from_allesfitter(LEGACY)["b_duration"], rel=1e-12)
