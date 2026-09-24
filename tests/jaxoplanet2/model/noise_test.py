import jax
import numpy as np
import pytest
from scipy.stats import norm

from jaxoplanet2.io.data import Dataset
from jaxoplanet2.io.settings import parse_settings_text
from jaxoplanet2.model.noise import gaussian_loglike, white_noise_sigma

jax.config.update("jax_enable_x64", True)

SETTINGS = parse_settings_text(
    "companions_phot,b\ncompanions_rv,b\ninst_phot,tess\ninst_rv,harps\n"
)


def flux_data(yerr, yerr_mean=None):
    n = len(yerr)
    return Dataset("tess", "flux", np.arange(n), np.ones(n), yerr, yerr_mean)


def rv_data(yerr):
    n = len(yerr)
    return Dataset("harps", "rv", np.arange(n), np.zeros(n), yerr)


def test_flux_sigma_rescales_file_errors_like_allesfitter():
    d = flux_data(np.array([1.0, 3.0]))
    sigma = white_noise_sigma({"ln_err_flux_tess": np.log(0.002)}, SETTINGS, d)
    # err_scales = yerr / mean(yerr) = [0.5, 1.5]
    np.testing.assert_allclose(sigma, [0.001, 0.003])


def test_flux_sigma_uses_the_full_file_mean_after_fast_fit():
    d = flux_data(np.array([1.0, 1.0]), yerr_mean=2.0)
    sigma = white_noise_sigma({"ln_err_flux_tess": 0.0}, SETTINGS, d)
    np.testing.assert_allclose(sigma, [0.5, 0.5])


def test_rv_sigma_adds_jitter_in_quadrature():
    d = rv_data(np.array([0.003, 0.004]))
    sigma = white_noise_sigma({"ln_jitter_rv_harps": np.log(0.004)}, SETTINGS, d)
    np.testing.assert_allclose(sigma, [0.005, np.sqrt(0.004**2 + 0.004**2)])


@pytest.mark.parametrize(
    "dataset, key", [(flux_data(np.ones(2)), "ln_err_flux_tess"),
                     (rv_data(np.ones(2)), "ln_jitter_rv_harps")]
)  # fmt: skip
def test_missing_noise_parameter_is_explained(dataset, key):
    with pytest.raises(KeyError, match=key):
        white_noise_sigma({}, SETTINGS, dataset)


def test_gaussian_loglike_matches_scipy():
    r = np.array([0.1, -0.2, 0.05])
    sigma = np.array([0.1, 0.2, 0.3])
    expected = norm.logpdf(r, scale=sigma).sum()
    np.testing.assert_allclose(gaussian_loglike(r, sigma), expected)


def test_gaussian_loglike_is_differentiable_in_sigma():
    g = jax.grad(lambda s: gaussian_loglike(np.array([0.1]), s))(0.1)
    # d/ds [-r^2/(2s^2) - log s] = r^2/s^3 - 1/s = 0 at s = |r|
    assert g == pytest.approx(0.0, abs=1e-10)
