import numpy as np
import pytest
from typer.testing import CliRunner

from jaxoplanet2.cli import app
from jaxoplanet2.infer.mcmc import (
    load_samples,
    mcmc_fit,
    run_config,
)
from jaxoplanet2.io.settings import parse_settings_text
from tests.jaxoplanet2.synthetic import TRUTH, make_fit_dir

runner = CliRunner()
# plumbing only: a low target_accept keeps these fast
TINY = (
    "mcmc_nwalkers,2\nmcmc_total_steps,30\nmcmc_burn_steps,20\n"
    "jx_nuts_target_accept,0.8\n"
)


def settings(extra=""):
    return parse_settings_text("companions_phot,b\ninst_phot,tess\n" + extra)


def test_nwalkers_become_chains_capped_by_devices():
    cfg = run_config(settings("mcmc_nwalkers,100\n"), device_count=8)
    assert cfg.num_chains == 8
    assert cfg.chain_method == "parallel"


def test_at_least_two_chains_on_a_single_device():
    cfg = run_config(settings("mcmc_nwalkers,100\n"), device_count=1)
    assert cfg.num_chains == 2
    assert cfg.chain_method == "sequential"


def test_more_chains_than_devices_run_sequentially():
    cfg = run_config(settings("mcmc_nwalkers,4\njx_num_chains,4\n"), device_count=1)
    assert cfg.num_chains == 4
    assert cfg.chain_method == "sequential"


def test_dense_mass_is_the_default_and_can_be_disabled():
    assert run_config(settings(), device_count=1).dense_mass is True
    cfg = run_config(settings("jx_nuts_dense_mass,False\n"), device_count=1)
    assert cfg.dense_mass is False


def test_steps_map_to_warmup_samples_and_thinning():
    cfg = run_config(
        settings("mcmc_total_steps,5000\nmcmc_burn_steps,2000\nmcmc_thin_by,5\n"),
        device_count=1,
    )
    assert (cfg.num_warmup, cfg.num_samples, cfg.thinning) == (2000, 3000, 5)


def test_jx_settings_reach_the_sampler():
    cfg = run_config(
        settings("jx_seed,7\njx_nuts_target_accept,0.95\njx_nuts_max_tree_depth,8\n"),
        device_count=1,
    )
    assert (cfg.seed, cfg.target_accept, cfg.max_tree_depth) == (7, 0.95, 8)


@pytest.fixture
def tiny_fit(tmp_path):
    return make_fit_dir(tmp_path / "fit", extra_settings=TINY)


def test_mcmc_fit_writes_samples_and_diagnostics(tiny_fit):
    result = mcmc_fit(tiny_fit, quiet=True, progress_bar=False)
    samples, meta = load_samples(tiny_fit)
    assert samples["b_rr"].shape == (2, 10)
    assert set(samples) == set(meta["fitkeys"])
    assert meta["num_chains"] == 2
    assert result.r_hat.keys() == samples.keys()
    text = (tiny_fit / "results" / "mcmc_diagnostics.txt").read_text()
    assert "r_hat" in text and "b_rr" in text


def test_cli_mcmc_fit(tiny_fit):
    result = runner.invoke(app, ["mcmc-fit", str(tiny_fit), "--no-progress"])
    assert result.exit_code == 0, result.output
    assert "mcmc-output" in result.output or "mcmc_table" in result.output
    assert (tiny_fit / "results" / "mcmc_samples.npz").exists()


def test_cli_mcmc_fit_rejects_invalid_directory(tiny_fit):
    (tiny_fit / "tess.csv").write_text("1,nan,0.1\n")
    result = runner.invoke(app, ["mcmc-fit", str(tiny_fit)])
    assert result.exit_code == 1


@pytest.mark.slow
def test_posterior_contains_the_truth(tmp_path):
    fit = make_fit_dir(
        tmp_path / "fit",
        extra_settings="mcmc_nwalkers,2\nmcmc_total_steps,300\nmcmc_burn_steps,150\n",
    )
    result = mcmc_fit(fit, quiet=True, progress_bar=False)
    samples, _ = load_samples(fit)
    for name in ("b_rr", "b_epoch", "b_period", "ln_err_flux_tess"):
        draws = samples[name].ravel()
        z = abs(draws.mean() - TRUTH[name]) / draws.std()
        assert z < 3, (name, z)
        assert result.r_hat[name] < 1.05
    assert result.divergences == 0  # with the default target_accept of 0.99
    assert np.isfinite(samples["b_rsuma"]).all()
