import math

import numpy as np
import pytest

from jaxoplanet2.infer.derived import derive, summarize
from jaxoplanet2.io.settings import parse_settings_text
from jaxoplanet2.model.external_priors import Star
from tests.jaxoplanet2.native import from_allesfitter

SETTINGS = parse_settings_text("companions_phot,b\ninst_phot,tess\n")
N = 2000


def draws(**values):
    """N identical native draws of a system given in allesfitter numbers."""
    native = from_allesfitter(values)
    return {k: np.full(N, v, dtype=float) for k, v in native.items()}


BASE = {"b_rr": 0.1, "b_rsuma": 0.11, "b_cosi": 0.05, "b_period": 3.0, "b_epoch": 0.0}


def test_summarize_gives_median_and_one_sigma_errors():
    x = np.random.default_rng(0).normal(5.0, 2.0, 200_000)
    median, lower, upper = summarize(x)
    assert median == pytest.approx(5.0, abs=0.02)
    assert lower == pytest.approx(2.0, rel=0.01)
    assert upper == pytest.approx(2.0, rel=0.01)


def test_geometry_of_a_circular_orbit():
    d = derive(draws(**BASE), SETTINGS, star=None)
    a_over_r = 1.1 / 0.11
    assert d["b_a/R_star"].values[0] == pytest.approx(a_over_r)
    assert d["b_R_star/a"].values[0] == pytest.approx(0.1)
    assert d["b_i"].values[0] == pytest.approx(math.degrees(math.acos(0.05)))
    assert d["b_rsuma"].values[0] == pytest.approx(0.11)
    assert d["b_cosi"].values[0] == pytest.approx(0.05)
    b = a_over_r * 0.05
    sin_i = math.sqrt(1 - 0.05**2)
    t14 = 3.0 / math.pi * math.asin(math.sqrt(1.1**2 - b**2) / a_over_r / sin_i)
    t23 = 3.0 / math.pi * math.asin(math.sqrt(0.9**2 - b**2) / a_over_r / sin_i)
    assert d["b_T_tra_tot"].values[0] == pytest.approx(t14 * 24)
    assert d["b_T_tra_full"].values[0] == pytest.approx(t23 * 24)
    assert d["b_host_density"].unit == "cgs"


def test_eccentric_orbit_recovers_allesfitters_geometry():
    d = derive(draws(**BASE, b_f_c=0.3, b_f_s=0.4), SETTINGS, star=None)
    e, w = 0.25, math.atan2(0.4, 0.3)
    assert d["b_e"].values[0] == pytest.approx(e)
    assert d["b_w"].values[0] == pytest.approx(math.degrees(w))
    assert d["b_rsuma"].values[0] == pytest.approx(0.11)
    assert d["b_cosi"].values[0] == pytest.approx(0.05)


def test_grazing_transit_has_no_full_duration():
    d = derive(draws(**{**BASE, "b_cosi": 0.095}), SETTINGS, star=None)
    assert np.isnan(d["b_T_tra_full"].values[0])


def test_physical_quantities_need_the_star():
    assert "b_R_companion_earth" not in derive(draws(**BASE), SETTINGS, star=None)
    star = Star(1.0, (0.0, 0.0), 1.0, (0.0, 0.0), teff=5772.0, teff_err=(0.0, 0.0))
    d = derive(draws(**BASE), SETTINGS, star=star)
    assert d["b_R_companion_earth"].values.mean() == pytest.approx(
        0.1 * 109.08, rel=1e-3
    )
    assert d["b_a_au"].values.mean() == pytest.approx(10.0 / 215.03, rel=1e-3)
    teq = 5772.0 * 0.7**0.25 * math.sqrt(0.1 / 2)
    assert d["b_Teq"].values.mean() == pytest.approx(teq, rel=1e-6)


def test_companion_mass_from_rv():
    s = parse_settings_text(
        "companions_phot,b\ncompanions_rv,b\ninst_phot,tess\ninst_rv,h\n"
    )
    star = Star(1.0, (0.0, 0.0), 1.0, (0.0, 0.0))
    # a hot Jupiter: K = 0.1 km/s at P = 3 d around the Sun is ~0.72 M_jup
    d = derive(draws(**BASE, b_K=0.1), s, star=star)
    assert d["b_M_companion_jup"].values.mean() == pytest.approx(0.72, rel=0.02)


def test_star_uncertainties_propagate():
    star = Star(1.0, (0.1, 0.1), 1.0, (0.0, 0.0))
    d = derive(draws(**BASE), SETTINGS, star=star, seed=1)
    spread = d["b_R_companion_earth"].values.std()
    assert spread == pytest.approx(0.1 * 109.08 * 0.1, rel=0.05)
