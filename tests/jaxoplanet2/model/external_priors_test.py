import jax
import numpy as np
import pytest
from scipy.stats import norm

from jaxoplanet2.io.settings import parse_settings_text
from jaxoplanet2.model.external_priors import (
    DensityPrior,
    Star,
    companion_mass_g,
    density_prior,
    external_log_prior,
    implied_host_density,
    load_star,
)
from tests.jaxoplanet2.native import from_allesfitter

jax.config.update("jax_enable_x64", True)
TOI1448 = Star(0.38, (0.01, 0.01), 0.37, (0.02, 0.02))
TOI1448_RAW = {"b_rr": 0.069, "b_rsuma": 0.0119, "b_cosi": 0.0, "b_epoch": 0.0,
               "b_period": 8.112246}  # fmt: skip


def toi(**overrides):
    """TOI-1448 in native parameters (the references are allesfitter numbers)."""
    return from_allesfitter({**TOI1448_RAW, **overrides})


TOI1448_VALUES = toi()
SETTINGS = parse_settings_text("companions_phot,b\ninst_phot,tess\n")


def test_load_star_reads_allesfitter_format(tmp_path):
    (tmp_path / "params_star.csv").write_text(
        "#R_star,R_star_lerr,R_star_uerr,M_star,M_star_lerr,M_star_uerr,Teff_star\n"
        "#R_sun,R_sun,R_sun,M_sun,M_sun,M_sun,K\n0.38,0.01,0.02,0.37,0.02,0.03,3391\n"
    )
    assert load_star(tmp_path) == Star(
        0.38, (0.01, 0.02), 0.37, (0.02, 0.03), teff=3391.0, teff_err=(0.0, 0.0)
    )
    assert load_star(tmp_path / "nowhere") is None


def test_density_prior_matches_allesfitter_on_toi1448():
    # allesfitter2: external_priors['host_density'] = normal 9.5135 0.9424
    prior = density_prior(TOI1448)
    assert prior.mean == pytest.approx(9.513457817680493, rel=5e-3)
    assert prior.sd == pytest.approx(0.942401645061544, rel=2e-2)


def test_implied_density_matches_allesfitter_on_toi1448():
    rho, defined = implied_host_density(TOI1448_VALUES, TOI1448, "b")
    assert bool(defined)
    assert float(rho) == pytest.approx(208.37611001624103, rel=1e-4)


def test_companion_mass_solves_the_mass_function():
    # Earth around the Sun: K = 8.95 cm/s
    m = companion_mass_g(8.95e-5 / 100 * 100, 365.25, np.pi / 2, 0.0, 1.0)
    assert float(m) / 5.97e27 == pytest.approx(1.0, rel=0.01)


def test_rv_mass_lowers_the_implied_host_density():
    values = {**TOI1448_VALUES, "b_K": 0.05}
    with_mass, _ = implied_host_density(values, TOI1448, "b")
    without, _ = implied_host_density(TOI1448_VALUES, TOI1448, "b")
    assert float(with_mass) < float(without)


def test_large_planet_without_rv_has_no_density_prior():
    values = toi(b_rr=0.3)  # rr^3 = 0.027 > 0.01
    assert not bool(implied_host_density(values, TOI1448, "b")[1])


def test_external_log_prior_is_the_normal_logpdf():
    prior = DensityPrior(9.5, 0.9)
    lp = external_log_prior(TOI1448_VALUES, SETTINGS, prior, TOI1448)
    rho = float(implied_host_density(TOI1448_VALUES, TOI1448, "b")[0])
    assert float(lp) == pytest.approx(norm.logpdf(rho, 9.5, 0.9), rel=1e-10)


def test_density_prior_can_be_switched_off():
    s = parse_settings_text(
        "companions_phot,b\ninst_phot,tess\nuse_host_density_prior,False\n"
    )
    assert (
        float(external_log_prior(TOI1448_VALUES, s, DensityPrior(9.5, 0.9), TOI1448))
        == 0
    )


@pytest.mark.parametrize(
    "overrides",
    [{"b_f_c": 0.8, "b_f_s": 0.8}, {"b_f_c": 0.995, "b_f_s": 0.0}, {"dil_tess": 0.9995}],
)
def test_physical_limits_give_minus_infinity(overrides):
    values = toi(**overrides)
    assert external_log_prior(values, SETTINGS, None, None) == -np.inf


def test_gradient_is_finite_without_rv():
    def f(duration):
        v = {**TOI1448_VALUES, "b_duration": duration}
        return external_log_prior(v, SETTINGS, DensityPrior(9.5, 0.9), TOI1448)

    assert np.isfinite(jax.grad(f)(TOI1448_VALUES["b_duration"]))


def test_no_transit_geometry_gives_minus_infinity():
    k = TOI1448_VALUES["b_radius_ratio"]
    grazing_beyond = {**TOI1448_VALUES, "b_impact_param": 1.0 + k + 0.01}
    assert external_log_prior(grazing_beyond, SETTINGS, None, None) == -np.inf
    too_long = {**TOI1448_VALUES, "b_duration": 0.6 * TOI1448_VALUES["b_period"]}
    assert external_log_prior(too_long, SETTINGS, None, None) == -np.inf
