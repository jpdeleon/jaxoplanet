import numpy as np
import pytest
from typer.testing import CliRunner

from jaxoplanet2.cli import app
from jaxoplanet2.infer.mcmc import mcmc_fit
from jaxoplanet2.infer.mcmc_output import latex_row, mcmc_output, read_table
from jaxoplanet2.io.params import load_params
from tests.jaxoplanet2.infer.mcmc_test import TINY
from tests.jaxoplanet2.synthetic import make_fit_dir

runner = CliRunner()


@pytest.fixture(scope="module")
def sampled(tmp_path_factory):
    fit = make_fit_dir(tmp_path_factory.mktemp("out") / "fit", extra_settings=TINY)
    mcmc_fit(fit, quiet=True, progress_bar=False)
    return fit


def test_latex_row_formats_asymmetric_errors():
    x = np.concatenate(
        [np.full(16, 0.9), np.full(34, 1.0), np.full(34, 1.0), [1.2] * 16]
    )
    row = latex_row("$P$", x, "d")
    assert row.startswith("$P$ & $1.0")
    assert row.endswith("& d \\\\")


def test_writes_tables_and_figures(sampled):
    paths = mcmc_output(sampled, overwrite=True, file_extension="png")
    names = {p.name for p in paths}
    assert {"mcmc_table.csv", "mcmc_derived_table.csv", "mcmc_latex_table.txt",
            "mcmc_corner.png", "mcmc_fit_tess.png"} <= names  # fmt: skip
    table = read_table(sampled / "results" / "mcmc_table.csv")
    assert set(table) == {p.name for p in load_params(sampled).free}
    median, lower, upper = table["b_radius_ratio"]
    assert 0.05 < median < 0.15 and lower > 0 and upper > 0
    derived = (sampled / "results" / "mcmc_derived_table.csv").read_text()
    assert "b_T_tra_tot" in derived and "b_rsuma" in derived


def test_refuses_to_overwrite(sampled):
    mcmc_output(sampled, overwrite=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        mcmc_output(sampled)


def test_cli_mcmc_output(sampled):
    result = runner.invoke(app, ["mcmc-output", str(sampled), "-o", "-e", ".png"])
    assert result.exit_code == 0, result.output
    assert "mcmc_corner.png" in result.output


def test_cli_without_samples(tmp_path):
    fit = make_fit_dir(tmp_path / "fit")
    result = runner.invoke(app, ["mcmc-output", str(fit)])
    assert result.exit_code == 1
    assert "mcmc-fit" in result.output
