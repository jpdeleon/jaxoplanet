import shutil

import jax
import matplotlib
import numpy as np
import pytest
from typer.testing import CliRunner

from jaxoplanet2.cli import app
from jaxoplanet2.fitdir import load_fit_directory
from jaxoplanet2.plots.initial_guess import (
    binned,
    companion_signal,
    phase_offset,
    show_initial_guess,
)
from tests.jaxoplanet2.native import native_golden

matplotlib.use("Agg")
jax.config.update("jax_enable_x64", True)

GOLDEN = native_golden()  # golden cases converted to native parameters
runner = CliRunner()


@pytest.fixture
def fit_dir(tmp_path):
    dest = tmp_path / "fit"
    shutil.copytree(
        GOLDEN / "eccentric_rv_ellc", dest, ignore=shutil.ignore_patterns("results")
    )
    return dest


def test_phase_offset_wraps_to_half_period():
    t = np.array([0.0, 1.0, 2.9, 3.1, -2.0])
    np.testing.assert_allclose(phase_offset(t, 0.0, 3.0), [0.0, 1.0, -0.1, 0.1, 1.0])


def test_binned_means_and_drops_empty_bins():
    x = np.array([0.1, 0.2, 2.5])
    y = np.array([1.0, 3.0, 5.0])
    centers, means = binned(x, y, np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(centers, [0.5, 2.5])
    np.testing.assert_allclose(means, [2.0, 5.0])


def test_companion_signals_add_up_to_the_total():
    fit = load_fit_directory(GOLDEN / "two_planets_dilution_exposure")
    t = fit.data["tess"].time
    values = fit.params.values()
    b = np.asarray(companion_signal(values, fit.settings, "tess", t, companion="b"))
    c = np.asarray(companion_signal(values, fit.settings, "tess", t, companion="c"))
    total = np.asarray(companion_signal(values, fit.settings, "tess", t))
    assert b.min() < -1e-3 and c.min() < -1e-3
    np.testing.assert_allclose(total, b + c, atol=1e-12)


def test_writes_one_figure_per_instrument(fit_dir):
    paths = show_initial_guess(fit_dir)
    assert sorted(p.name for p in paths) == [
        "initial_guess_harps.pdf",
        "initial_guess_tess.pdf",
    ]
    assert all(p.stat().st_size > 1000 for p in paths)


def test_file_extension_is_configurable(fit_dir):
    paths = show_initial_guess(fit_dir, file_extension="png")
    assert all(p.suffix == ".png" for p in paths)


def test_no_plot_writes_nothing(fit_dir):
    assert show_initial_guess(fit_dir, do_plot=False) == []
    assert not (fit_dir / "results").exists()


def test_invalid_extension(fit_dir):
    with pytest.raises(ValueError, match="bmp"):
        show_initial_guess(fit_dir, file_extension=".bmp")


def test_cli_show_initial_guess(fit_dir):
    result = runner.invoke(app, ["show-initial-guess", str(fit_dir), "-e", ".png"])
    assert result.exit_code == 0, result.output
    assert "initial_guess_tess.png" in result.output
    assert (fit_dir / "results" / "initial_guess_tess.png").exists()


def test_cli_quiet_suppresses_output(fit_dir):
    result = runner.invoke(app, ["show-initial-guess", str(fit_dir), "-q"])
    assert result.exit_code == 0
    assert result.output == ""


def test_cli_refuses_invalid_fit(fit_dir):
    (fit_dir / "tess.csv").write_text("1,nan,0.1\n")
    result = runner.invoke(app, ["show-initial-guess", str(fit_dir)])
    assert result.exit_code == 1
    assert "NaN" in result.output
