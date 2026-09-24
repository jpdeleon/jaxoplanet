"""Golden parity test: jaxoplanet2 reproduces allesfitter's model.

Each ``cases/<name>`` directory is a complete allesfitter fit directory. The
matching ``<name>.npz`` holds allesfitter's model evaluated at the params.csv
values (see generate_golden.py). Here the same directory goes through the
jaxoplanet2 readers and model, and the two must agree.

Tolerances: batman is exact, so ``flux_model=batman`` cases must agree to 1 ppm.
ellc integrates numerically on a grid and is itself off by ~6 ppm, so ellc cases
get 10 ppm. RVs always come from ellc, good to ~2 cm/s; allow 5 cm/s (km/s data).
"""

from pathlib import Path

import jax
import numpy as np
import pytest

from jaxoplanet2.fitdir import load_fit_directory
from jaxoplanet2.io.data import load_datasets
from jaxoplanet2.io.params import load_params
from jaxoplanet2.io.settings import load_settings
from jaxoplanet2.model.numpyro_model import (
    initial_values,
    log_prob_parts,
    mean_components,
)
from jaxoplanet2.model.photometry import flux_model
from jaxoplanet2.model.rv import rv_model

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
CASES = sorted(p.name for p in (HERE / "cases").iterdir())
FLUX_ATOL = {"batman": 1e-6, "ellc": 1e-5}
RV_ATOL = 5e-5  # km/s
# model differences allowed in the likelihood comparison: batman light curves
# agree to <0.04 ppm; RVs always come from ellc (~2 cm/s)
LOGLIKE_MODEL_DELTA = {"flux": 1e-7, "rv": RV_ATOL}


def test_every_case_has_a_reference():
    assert CASES
    for name in CASES:
        assert (HERE / f"{name}.npz").is_file(), f"run generate_golden.py for {name}"


@pytest.mark.parametrize("name", CASES)
def test_model_matches_allesfitter(name):
    case = HERE / "cases" / name
    reference = np.load(HERE / f"{name}.npz")
    settings = load_settings(case)
    values = load_params(case).values()
    data = load_datasets(case, settings, values)
    backend = settings.raw.get("flux_model", "ellc")

    for inst in settings.inst_phot:
        model = np.asarray(flux_model(values, settings, inst, data[inst].time))
        np.testing.assert_allclose(model, reference[inst], atol=FLUX_ATOL[backend])
    for inst in settings.inst_rv:
        model = np.asarray(rv_model(values, settings, inst, data[inst].time))
        np.testing.assert_allclose(model, reference[inst], atol=RV_ATOL)


def _loglike_tolerance(residual, sigma, delta):
    """Largest log-likelihood change a model error of ``delta`` can cause."""
    return float(
        np.sum(np.abs(residual) * delta / sigma**2 + 0.5 * (delta / sigma) ** 2)
    )


@pytest.mark.parametrize("name", CASES)
def test_log_likelihood_matches_allesfitter(name):
    """Noise model, error normalisation, baselines and fast_fit all agree.

    Any difference must be explained by the (tested) model differences alone,
    so the tolerance is the bound those differences put on the likelihood.
    """
    case = HERE / "cases" / name
    reference = np.load(HERE / f"{name}.npz")
    fit = load_fit_directory(case)
    if fit.settings.raw.get("flux_model") != "batman":
        pytest.skip("ellc reference: its ~5 ppm model error dominates")
    values = fit.params.values()
    parts = log_prob_parts(fit, initial_values(fit))
    for inst, loglike in parts.per_instrument.items():
        data = fit.data[inst]
        mu, base, sigma = (
            np.asarray(x) for x in mean_components(values, fit.settings, data)
        )
        tol = _loglike_tolerance(
            data.y - mu - base, sigma, LOGLIKE_MODEL_DELTA[data.kind]
        )
        np.testing.assert_allclose(
            loglike, float(reference[f"{inst}_loglike"]), atol=tol
        )


@pytest.mark.parametrize("name", CASES)
def test_baselines_match_allesfitter(name):
    """Sampled, hybrid and GP (conditional mean) baselines at the data."""
    reference = np.load(HERE / f"{name}.npz")
    fit = load_fit_directory(HERE / "cases" / name)
    if fit.settings.raw.get("flux_model") != "batman":
        pytest.skip("ellc reference: its model error leaks into hybrid/GP baselines")
    values = fit.params.values()
    for inst, data in fit.data.items():
        base = np.asarray(mean_components(values, fit.settings, data)[1])
        np.testing.assert_allclose(base, reference[f"{inst}_baseline"], atol=1e-7)


@pytest.mark.parametrize("name", CASES)
def test_references_contain_a_real_signal(name):
    """Guard against a vacuous comparison (e.g. data that misses every transit)."""
    reference = np.load(HERE / f"{name}.npz")
    settings = load_settings(HERE / "cases" / name)
    for inst in settings.inst_phot:
        assert 1.0 - reference[inst].min() > 1e-3
    for inst in settings.inst_rv:
        assert np.ptp(reference[inst]) > 0.05
