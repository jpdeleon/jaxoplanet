import jax
import numpy as np
import pytest
from scipy.optimize import brentq

from jaxoplanet2.io.settings import parse_settings_text
from jaxoplanet2.model.parameterization import mid_eclipse_offset
from jaxoplanet2.model.rv import rv_model

jax.config.update("jax_enable_x64", True)

SETTINGS = "companions_rv,b\ninst_rv,harps\n"


def settings(text=SETTINGS):
    return parse_settings_text(text)


def values(**overrides):
    base = {"b_epoch": 2459000.3, "b_period": 3.0, "b_K": 0.05}
    return {**base, **overrides}


def keplerian_rv(t, epoch, period, K, *, f_c=0.0, f_s=0.0, cosi=0.0):
    """Textbook stellar RV: K [cos(nu + w) + e cos w], with allesfitter's epoch."""
    e = f_c**2 + f_s**2
    w = np.arctan2(f_s, f_c) if e else np.pi / 2
    shift = float(mid_eclipse_offset(period, e, np.cos(w), np.sin(w), np.arccos(cosi)))
    f_conj = np.pi / 2 - w
    E_conj = 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(f_conj / 2), np.sqrt(1 + e) * np.cos(f_conj / 2)
    )
    t_peri = (epoch - shift) - period / (2 * np.pi) * (E_conj - e * np.sin(E_conj))
    out = []
    for ti in np.atleast_1d(t):
        M = 2 * np.pi * (ti - t_peri) / period
        M = (M + np.pi) % (2 * np.pi) - np.pi
        E = brentq(lambda E, M=M: E - e * np.sin(E) - M, -np.pi - 1, np.pi + 1)
        nu = 2 * np.arctan2(
            np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2)
        )
        out.append(K * (np.cos(nu + w) + e * np.cos(w)))
    return np.array(out)


def test_circular_rv_is_a_sine_through_the_epoch():
    v = values()
    t = v["b_epoch"] + np.linspace(0, 3, 301)
    got = np.asarray(rv_model(v, settings(), "harps", t))
    expected = -0.05 * np.sin(2 * np.pi * (t - v["b_epoch"]) / 3.0)
    np.testing.assert_allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize(
    "f_c, f_s, cosi", [(0.3, 0.4, 0.05), (-0.2, -0.5, 0.02), (0.5, 0.1, 0.0)]
)
def test_eccentric_rv_matches_textbook_keplerian(f_c, f_s, cosi):
    v = values(b_f_c=f_c, b_f_s=f_s, b_cosi=cosi, b_rr=0.1, b_rsuma=0.1)
    t = v["b_epoch"] + np.linspace(-1.5, 1.5, 61)
    got = np.asarray(rv_model(v, settings(), "harps", t))
    expected = keplerian_rv(t, v["b_epoch"], 3.0, 0.05, f_c=f_c, f_s=f_s, cosi=cosi)
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_companions_add_up():
    s = settings("companions_rv,b c\ninst_rv,harps\n")
    v = values(c_epoch=2459001.0, c_period=11.0, c_K=0.02)
    t = 2459000.0 + np.linspace(0, 10, 101)
    total = np.asarray(rv_model(v, s, "harps", t))
    b = np.asarray(rv_model(v, settings(), "harps", t))
    c = np.asarray(rv_model(v, settings("companions_rv,c\ninst_rv,harps\n"), "harps", t))
    np.testing.assert_allclose(total, b + c, atol=1e-12)


def test_no_rv_companions_gives_zero():
    s = settings("companions_phot,b\ninst_phot,tess\ninst_rv,harps\n")
    t = np.linspace(0, 1, 5)
    np.testing.assert_array_equal(rv_model(values(), s, "harps", t), np.zeros(5))


def test_exposure_smearing_reduces_amplitude():
    s = settings(SETTINGS + "t_exp_harps,0.5\nt_exp_n_int_harps,11\n")
    t = 2459000.3 + np.linspace(0, 3, 301)
    smeared = np.asarray(rv_model(values(), s, "harps", t))
    sharp = np.asarray(rv_model(values(), settings(), "harps", t))
    assert np.max(np.abs(smeared)) < np.max(np.abs(sharp))


def test_rv_model_is_differentiable():
    t = np.linspace(2459000.0, 2459003.0, 31)
    g = jax.grad(lambda K: rv_model(values(b_K=K), settings(), "harps", t).sum())(0.05)
    assert np.isfinite(g)
