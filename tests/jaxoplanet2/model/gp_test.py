import jax
import numpy as np
import pytest
from tinygp import GaussianProcess
from tinygp.kernels import quasisep

from jaxoplanet2.io.data import Dataset
from jaxoplanet2.io.settings import parse_settings_text
from jaxoplanet2.model.gp import gp_kernel, residual_process

jax.config.update("jax_enable_x64", True)

T = np.linspace(0, 3, 400)
SIGMA = np.full(400, 1e-3)
DATA = Dataset("tess", "flux", T, np.ones(400), SIGMA)


def settings(kind):
    return parse_settings_text(
        f"companions_phot,b\ninst_phot,tess\nbaseline_flux_tess,{kind}\n"
    )


def test_matern32_maps_log_sigma_and_log_rho():
    k = gp_kernel(
        {"baseline_gp_matern32_lnsigma_flux_tess": np.log(2.0),
         "baseline_gp_matern32_lnrho_flux_tess": np.log(0.5)},
        "sample_GP_Matern32", "flux_tess",
    )  # fmt: skip
    ref = quasisep.Matern32(scale=0.5, sigma=2.0)
    np.testing.assert_allclose(k(T[:5], T[:5]), ref(T[:5], T[:5]))


def test_sho_matches_celerite_power_normalisation():
    # celerite: k(0) = S0 * w0 * Q
    S0, Q, w0 = 1e-6, 2.0, 3.0
    k = gp_kernel(
        {"baseline_gp_sho_lnS0_flux_tess": np.log(S0),
         "baseline_gp_sho_lnQ_flux_tess": np.log(Q),
         "baseline_gp_sho_lnomega0_flux_tess": np.log(w0)},
        "sample_GP_SHO", "flux_tess",
    )  # fmt: skip
    np.testing.assert_allclose(k(np.zeros(1), np.zeros(1))[0, 0], S0 * w0 * Q)


def test_real_term_is_an_exponential():
    a, c = 2e-6, 4.0
    k = gp_kernel(
        {"baseline_gp_real_lna_flux_tess": np.log(a),
         "baseline_gp_real_lnc_flux_tess": np.log(c)},
        "sample_GP_real", "flux_tess",
    )  # fmt: skip
    tau = np.array([0.0, 0.3])
    np.testing.assert_allclose(k(tau, np.zeros(1))[:, 0], a * np.exp(-c * tau))


def test_missing_hyperparameter_is_explained():
    with pytest.raises(KeyError, match="baseline_gp_real_lnc_flux_tess"):
        gp_kernel({"baseline_gp_real_lna_flux_tess": 0.0}, "sample_GP_real", "flux_tess")


def test_residual_process_uses_white_noise_and_optional_offset():
    values = {
        "baseline_gp_real_lna_flux_tess": np.log(1e-6),
        "baseline_gp_real_lnc_flux_tess": 0.0,
        "baseline_gp_offset_flux_tess": 0.002,
    }
    gp = residual_process(values, settings("sample_GP_real"), DATA, SIGMA)
    ref = GaussianProcess(
        quasisep.Exp(scale=1.0, sigma=1e-3), T, diag=SIGMA**2, mean=0.002
    )
    r = np.random.default_rng(0).normal(0, 1e-3, 400)
    np.testing.assert_allclose(gp.log_probability(r), ref.log_probability(r))


def test_log_probability_is_differentiable():
    def f(lna):
        values = {
            "baseline_gp_real_lna_flux_tess": lna,
            "baseline_gp_real_lnc_flux_tess": 0.0,
        }
        gp = residual_process(values, settings("sample_GP_real"), DATA, SIGMA)
        return gp.log_probability(np.zeros(400))

    assert np.isfinite(jax.grad(f)(np.log(1e-6)))
