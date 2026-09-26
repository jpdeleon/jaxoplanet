"""End-to-end: inject a planet, run the whole CLI pipeline, recover the truth.

Slow (a real NUTS run); runs in CI's slow-tests job (JAXOPLANET2_SLOW=1).
"""

import subprocess
import sys

import numpy as np
import pytest
from typer.testing import CliRunner

from jaxoplanet2.cli import app
from jaxoplanet2.infer.mcmc import load_samples
from jaxoplanet2.infer.mcmc_output import read_table
from tests.jaxoplanet2.synthetic import TRUTH, make_fit_dir

runner = CliRunner()
# a deliberately wrong starting point; optimize must fix it before sampling
START = {
    "b_radius_ratio": 0.09,
    "b_duration": 0.115,
    "b_time_transit": 2459000.402,
    "b_K": 0.03,
}
SAMPLER = "mcmc_nwalkers,2\nmcmc_total_steps,500\nmcmc_burn_steps,300\n"
RECOVERED = (
    "b_radius_ratio",
    "b_duration",
    "b_impact_param",
    "b_time_transit",
    "b_period",
    "b_K",
    "ln_err_flux_tess",
)
MAX_SIGMA = 3.0
MAX_R_HAT = 1.05


def run(*args):
    result = runner.invoke(app, [*args])
    assert result.exit_code == 0, result.output
    return result.output


@pytest.mark.slow
def test_transit_and_rv_injection_is_recovered(tmp_path):
    fit = make_fit_dir(tmp_path / "fit", rv=True, start=START, extra_settings=SAMPLER)

    assert "OK" in run("validate", str(fit))
    run("show-initial-guess", str(fit), "-q")
    assert "accepted" in run("optimize", str(fit), "--skip-bounds-check")
    # a fresh process, so the CLI can expose CPU cores and run chains in parallel
    sampled = subprocess.run(
        [sys.executable, "-m", "jaxoplanet2", "mcmc-fit", str(fit), "--no-progress"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert sampled.returncode == 0, sampled.stdout + sampled.stderr
    assert "(parallel)" in sampled.stdout

    table = read_table(fit / "results" / "mcmc_table.csv")
    for name in RECOVERED:
        median, lower, upper = table[name]
        sigma = lower if TRUTH[name] < median else upper
        assert abs(median - TRUTH[name]) < MAX_SIGMA * sigma, (name, table[name])

    samples, meta = load_samples(fit)
    draws = meta["num_chains"] * meta["num_samples"]
    diverging = int(np.load(fit / "results" / "mcmc_samples.npz")["__diverging__"].sum())
    assert diverging <= 0.01 * draws
    from numpyro.diagnostics import split_gelman_rubin

    for name in RECOVERED:
        assert float(split_gelman_rubin(samples[name])) < MAX_R_HAT, name
    for name in ("mcmc_fit_tess.pdf", "mcmc_fit_harps.pdf", "mcmc_corner.pdf"):
        assert (fit / "results" / name).exists()
