import jax
import numpy as np
import numpy.polynomial.polynomial as poly
import pytest

from jaxoplanet2.io.data import Dataset
from jaxoplanet2.io.settings import parse_settings_text
from jaxoplanet2.model.baseline import deterministic_baseline

jax.config.update("jax_enable_x64", True)

RNG = np.random.default_rng(3)
TIME = np.sort(2459000.0 + RNG.uniform(0, 5, 200))
SIGMA = RNG.uniform(0.5e-3, 2e-3, 200)
RESIDUAL = 1e-3 * (TIME - TIME[0]) + 2e-3 + RNG.normal(0, 1e-3, 200)
DATA = Dataset("tess", "flux", TIME, np.ones(200), SIGMA)


def settings(baseline):
    return parse_settings_text(
        f"companions_phot,b\ninst_phot,tess\nbaseline_flux_tess,{baseline}\n"
    )


def evaluate(baseline, values=None):
    return np.asarray(
        deterministic_baseline(values or {}, settings(baseline), DATA, RESIDUAL, SIGMA)
    )


def test_none_is_zero():
    np.testing.assert_array_equal(evaluate("none"), np.zeros(200))


def test_sample_offset():
    np.testing.assert_allclose(
        evaluate("sample_offset", {"baseline_offset_flux_tess": 0.01}), 0.01
    )


def test_sample_linear_uses_allesfitters_normalised_time():
    b = evaluate(
        "sample_linear",
        {"baseline_offset_flux_tess": 0.01, "baseline_slope_flux_tess": 0.002},
    )
    x = (TIME - TIME[0]) / (TIME[-1] - TIME[0])
    np.testing.assert_allclose(b, 0.01 + 0.002 * x)


def test_hybrid_offset_is_allesfitters_one_over_sigma_average():
    weights = 1.0 / (SIGMA / SIGMA.mean())
    expected = np.average(RESIDUAL, weights=weights)
    np.testing.assert_allclose(evaluate("hybrid_offset"), expected, rtol=1e-12)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_hybrid_poly_matches_allesfitters_polyfit(order):
    # allesfitter scales time by x[-1] (not the range) and passes w = 1/sigma_rel
    x = (TIME - TIME[0]) / TIME[-1]
    w = 1.0 / (SIGMA / SIGMA.mean())
    expected = poly.polyval(x, poly.polyfit(x, RESIDUAL, order, w=w))
    np.testing.assert_allclose(evaluate(f"hybrid_poly_{order}"), expected, atol=1e-10)


def test_missing_parameter_is_explained():
    with pytest.raises(KeyError, match="baseline_offset_flux_tess"):
        evaluate("sample_offset")


def test_gp_baselines_are_not_deterministic():
    with pytest.raises(ValueError, match="GP"):
        evaluate("sample_GP_Matern32")


def test_hybrid_poly_is_differentiable_in_the_residual():
    s = settings("hybrid_poly_2")

    def f(shift):
        return deterministic_baseline({}, s, DATA, RESIDUAL + shift, SIGMA).sum()

    # a constant shift moves every fitted value by the same amount
    np.testing.assert_allclose(jax.grad(f)(0.0), 200.0, rtol=1e-8)
