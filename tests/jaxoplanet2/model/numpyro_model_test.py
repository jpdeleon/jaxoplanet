from pathlib import Path

import jax
import numpy as np
import pytest
from numpyro import handlers
from scipy.stats import norm, truncnorm, uniform

from jaxoplanet2.fitdir import load_fit_directory
from jaxoplanet2.io.priors import Normal, TruncNormal, Uniform
from jaxoplanet2.model.noise import gaussian_loglike, white_noise_sigma
from jaxoplanet2.model.numpyro_model import (
    build_model,
    initial_values,
    likelihood_site,
    log_prob_parts,
    mean_components,
)
from jaxoplanet2.model.photometry import flux_model
from jaxoplanet2.model.rv import rv_model

jax.config.update("jax_enable_x64", True)
GOLDEN = Path(__file__).parents[1] / "golden" / "cases"


def scipy_logpdf(prior, x):
    if isinstance(prior, Uniform):
        return uniform.logpdf(x, prior.lower, prior.upper - prior.lower)
    if isinstance(prior, Normal):
        return norm.logpdf(x, prior.mean, prior.sd)
    assert isinstance(prior, TruncNormal)
    a = (prior.lower - prior.mean) / prior.sd
    b = (prior.upper - prior.mean) / prior.sd
    return truncnorm.logpdf(x, a, b, prior.mean, prior.sd)


@pytest.fixture(scope="module")
def fit():
    return load_fit_directory(GOLDEN / "eccentric_rv_ellc")


def test_initial_values_are_the_free_params(fit):
    init = initial_values(fit)
    assert set(init) == {p.name for p in fit.params.free}
    assert init["b_rr"] == pytest.approx(0.1)


def test_log_prob_parts_match_independent_computation(fit):
    parts = log_prob_parts(fit, initial_values(fit))
    values = fit.params.values()

    expected_prior = sum(scipy_logpdf(p.prior, p.value) for p in fit.params.free)
    np.testing.assert_allclose(parts.log_prior, expected_prior, rtol=1e-10)

    expected_like = 0.0
    for inst, data in fit.data.items():
        model = flux_model if data.kind == "flux" else rv_model
        mu = model(values, fit.settings, inst, data.time)
        sigma = white_noise_sigma(values, fit.settings, data)
        expected_like += float(gaussian_loglike(data.y - mu, sigma))
    np.testing.assert_allclose(parts.log_likelihood, expected_like, rtol=1e-10)
    assert set(parts.per_instrument) == {"tess", "harps"}


def test_fixed_params_are_not_sampled(fit):
    tr = handlers.trace(handlers.seed(build_model(fit), 0)).get_trace()
    sampled = {
        k for k, v in tr.items() if v["type"] == "sample" and not v["is_observed"]
    }
    assert sampled == {p.name for p in fit.params.free}
    assert "b_f_c" not in tr


def test_log_density_is_differentiable(fit):
    from numpyro.infer.util import log_density

    model = build_model(fit)
    init = initial_values(fit)

    def f(params):
        return log_density(model, (), {}, params)[0]

    grads = jax.grad(f)(init)
    assert all(np.isfinite(float(g)) for g in grads.values())


def test_sample_offset_baseline_shifts_the_mean(tmp_path):
    import shutil

    shutil.copytree(GOLDEN / "circular_batman", tmp_path / "fit")
    settings = tmp_path / "fit" / "settings.csv"
    settings.write_text(
        settings.read_text().replace(
            "baseline_flux_tess,none", "baseline_flux_tess,sample_offset"
        )
    )
    with (tmp_path / "fit" / "params.csv").open("a") as f:
        f.write("baseline_offset_flux_tess,0.001,1,uniform -0.01 0.01,off,,\n")
    fit = load_fit_directory(tmp_path / "fit")
    data = fit.data["tess"]
    values = fit.params.values()
    mu, base, sigma = mean_components(values, fit.settings, data)
    np.testing.assert_allclose(base, 0.001)
    distribution, observed = likelihood_site(values, fit.settings, data)
    np.testing.assert_allclose(distribution.mean, np.asarray(mu) + 0.001)
    np.testing.assert_array_equal(observed, data.y)
    assert "baseline_offset_flux_tess" in initial_values(fit)


def test_gp_baseline_observes_the_residual(tmp_path):
    fit = load_fit_directory(GOLDEN / "gp_baselines")
    data = fit.data["ngts"]
    values = fit.params.values()
    distribution, observed = likelihood_site(values, fit.settings, data)
    mu = np.asarray(mean_components(values, fit.settings, data)[0])
    np.testing.assert_allclose(observed, data.y - mu)
    assert distribution.event_shape == (len(data),)
