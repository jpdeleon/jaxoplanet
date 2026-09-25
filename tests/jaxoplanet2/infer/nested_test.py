import builtins

import numpy as np
import pytest
from typer.testing import CliRunner

from jaxoplanet2.cli import app
from jaxoplanet2.fitdir import load_fit_directory
from jaxoplanet2.infer import nested
from jaxoplanet2.infer.mcmc import load_samples
from jaxoplanet2.infer.mcmc_output import read_table
from tests.jaxoplanet2.synthetic import TRUTH, make_fit_dir

runner = CliRunner()
QUICK = "ns_nlive,40\nns_tol,1.0\nns_modus,dynamic\nns_bound,single\n"


def test_allesfitter_settings_map_to_jaxns(tmp_path):
    fit = load_fit_directory(make_fit_dir(tmp_path / "fit", extra_settings=QUICK))
    constructor, termination = nested.sampler_kwargs(fit)
    assert constructor == {"num_live_points": 40}
    assert termination == {"dlogZ": 1.0}


def test_defaults_are_left_to_numpyro(tmp_path):
    fit = load_fit_directory(make_fit_dir(tmp_path / "fit"))
    assert nested.sampler_kwargs(fit) == ({}, {})


def test_missing_jaxns_is_explained(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numpyro.contrib.nested_sampling":
            raise ImportError("no jaxns")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(nested.NestedError, match=r"jaxoplanet2\[ns\]"):
        nested._nested_sampler()


def test_cli_ns_fit_writes_evidence_and_tables(tmp_path):
    fit = make_fit_dir(tmp_path / "fit", extra_settings=QUICK)
    result = runner.invoke(app, ["ns-fit", str(fit), "-q"])
    assert result.exit_code == 0, result.output
    samples, meta = load_samples(fit, "ns")
    assert samples["b_radius_ratio"].shape == (1, nested.POSTERIOR_DRAWS)
    assert np.isfinite(meta["log_z"]) and meta["log_z_err"] > 0
    table = read_table(fit / "results" / "ns_table.csv")
    assert set(table) == set(meta["fitkeys"])
    assert (fit / "results" / "ns_corner.pdf").exists()
    again = runner.invoke(app, ["ns-output", str(fit), "-o", "-e", ".png", "-q"])
    assert again.exit_code == 0, again.output


@pytest.mark.slow
def test_nested_sampling_recovers_the_truth(tmp_path):
    fit = make_fit_dir(tmp_path / "fit")
    result = nested.ns_fit(fit, quiet=True)
    for name in ("b_radius_ratio", "b_time_transit", "b_period"):
        draws = result.samples[name].ravel()
        assert abs(draws.mean() - TRUTH[name]) < 3 * draws.std(), name
