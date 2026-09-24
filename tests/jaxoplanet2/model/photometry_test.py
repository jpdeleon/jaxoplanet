import jax
import numpy as np
import pytest

from jaxoplanet2.io.settings import parse_settings_text
from jaxoplanet2.model.exposure import exposure_nodes
from jaxoplanet2.model.parameterization import companion_orbit, mid_eclipse_offset
from jaxoplanet2.model.photometry import flux_model, ld_coefficients

batman = pytest.importorskip("batman")
jax.config.update("jax_enable_x64", True)

SETTINGS = "companions_phot,b\ninst_phot,tess\nhost_ld_law_tess,quad\n"


def settings(text=SETTINGS):
    return parse_settings_text(text)


def values(**overrides):
    base = {
        "b_rr": 0.1,
        "b_rsuma": 0.12,
        "b_cosi": 0.03,
        "b_epoch": 0.3,
        "b_period": 3.0,
        "host_ldc_q1_tess": 0.4,
        "host_ldc_q2_tess": 0.3,
    }
    return {**base, **overrides}


def batman_flux(t, v, u, t0_shift=0.0):
    p = batman.TransitParams()
    ecc = v.get("b_f_c", 0.0) ** 2 + v.get("b_f_s", 0.0) ** 2
    p.t0 = v["b_epoch"] - t0_shift
    p.per = v["b_period"]
    p.rp = v["b_rr"]
    p.a = (1 + v["b_rr"]) / v["b_rsuma"]
    p.inc = np.degrees(np.arccos(v["b_cosi"]))
    p.ecc = ecc
    p.w = (
        np.degrees(np.arctan2(v.get("b_f_s", 0.0), v.get("b_f_c", 0.0))) if ecc else 90.0
    )
    p.limb_dark = "quadratic"
    p.u = list(u)
    return batman.TransitModel(p, t).light_curve(p)


def test_quadratic_ld_from_kipping_q():
    u = ld_coefficients(values(), settings(), "tess")
    q1, q2 = 0.4, 0.3
    np.testing.assert_allclose(u, [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)])


def test_quadratic_ld_in_u_space():
    s = settings(SETTINGS + "host_ld_space_tess,u\n")
    u = ld_coefficients({"host_ldc_u1_tess": 0.3, "host_ldc_u2_tess": 0.1}, s, "tess")
    np.testing.assert_allclose(u, [0.3, 0.1])


def test_linear_ld():
    s = settings("companions_phot,b\ninst_phot,tess\nhost_ld_law_tess,lin\n")
    np.testing.assert_allclose(
        ld_coefficients({"host_ldc_q1_tess": 0.4}, s, "tess"), [0.4]
    )


def test_no_ld():
    s = settings("companions_phot,b\ninst_phot,tess\nhost_ld_law_tess,none\n")
    assert ld_coefficients({}, s, "tess").shape == (0,)


def test_missing_ld_parameter_is_explained():
    with pytest.raises(KeyError, match="host_ldc_q2_tess"):
        ld_coefficients({"host_ldc_q1_tess": 0.4}, settings(), "tess")


def test_circular_transit_matches_batman():
    v = values()
    t = np.linspace(0.1, 0.5, 2001)
    u = ld_coefficients(v, settings(), "tess")
    got = np.asarray(flux_model(v, settings(), "tess", t))
    np.testing.assert_allclose(got, batman_flux(t, v, u), atol=1e-7)


def test_eccentric_transit_matches_batman_at_conjunction_time():
    v = values(b_f_c=0.3, b_f_s=0.4, b_cosi=0.05, b_period=4.0)
    t = np.linspace(0.1, 0.5, 2001)
    u = ld_coefficients(v, settings(), "tess")
    e, w = 0.25, np.arctan2(0.4, 0.3)
    shift = float(mid_eclipse_offset(4.0, e, np.cos(w), np.sin(w), np.arccos(0.05)))
    got = np.asarray(flux_model(v, settings(), "tess", t))
    np.testing.assert_allclose(got, batman_flux(t, v, u, t0_shift=shift), atol=1e-7)


def test_bjd_epochs_keep_precision():
    v = values(b_epoch=2459000.3)
    t = 2459000.0 + np.linspace(0.1, 0.5, 2001)
    u = ld_coefficients(v, settings(), "tess")
    got = np.asarray(flux_model(v, settings(), "tess", t))
    ref = batman_flux(t - 2459000.0, values(), u)
    np.testing.assert_allclose(got, ref, atol=1e-7)


def test_multiple_companions_add_their_depths():
    s = settings("companions_phot,b c\ninst_phot,tess\nhost_ld_law_tess,quad\n")
    v = values(c_rr=0.05, c_rsuma=0.05, c_cosi=0.0, c_epoch=0.31, c_period=7.0)
    t = np.linspace(0.2, 0.4, 501)
    total = np.asarray(flux_model(v, s, "tess", t))
    only_b = np.asarray(flux_model(v, settings(), "tess", t))
    s_c = settings("companions_phot,c\ninst_phot,tess\nhost_ld_law_tess,quad\n")
    only_c = np.asarray(flux_model(v, s_c, "tess", t))
    np.testing.assert_allclose(total, 1 - (1 - only_b) - (1 - only_c), atol=1e-12)


def test_dilution_scales_the_depth():
    t = np.linspace(0.2, 0.4, 201)
    plain = np.asarray(flux_model(values(), settings(), "tess", t))
    diluted = np.asarray(flux_model(values(dil_tess=0.25), settings(), "tess", t))
    np.testing.assert_allclose(diluted - 1, 0.75 * (plain - 1), atol=1e-12)


def test_exposure_nodes_are_allesfitters_trapezoid():
    offsets, weights = exposure_nodes(0.02, 5)
    np.testing.assert_allclose(offsets, [-0.01, -0.005, 0.0, 0.005, 0.01])
    np.testing.assert_allclose(weights, [0.125, 0.25, 0.25, 0.25, 0.125])
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_long_exposures_are_smeared():
    s = settings(SETTINGS + "t_exp_tess,0.0204\nt_exp_n_int_tess,201\n")
    t = np.linspace(0.2, 0.4, 401)
    v = values()
    smeared = np.asarray(flux_model(v, s, "tess", t))
    # brute force: average the instantaneous model across the exposure
    fine = np.linspace(-0.0102, 0.0102, 2001)
    inst = np.asarray(flux_model(v, settings(), "tess", (t[:, None] + fine).ravel()))
    brute = np.trapezoid(inst.reshape(len(t), -1), fine, axis=1) / 0.0204
    # converges to the exact exposure average as n_int grows
    np.testing.assert_allclose(smeared, brute, atol=2e-7)
    assert smeared.min() > np.asarray(flux_model(v, settings(), "tess", t)).min()


def test_single_sub_exposure_is_instantaneous():
    s = settings(SETTINGS + "t_exp_tess,0.02\nt_exp_n_int_tess,1\n")
    t = np.linspace(0.2, 0.4, 51)
    np.testing.assert_allclose(
        flux_model(values(), s, "tess", t), flux_model(values(), settings(), "tess", t)
    )


def test_flux_model_is_differentiable_and_jittable():
    s = settings()
    t = np.linspace(0.25, 0.35, 21)

    @jax.jit
    def loss(rr):
        return flux_model(values(b_rr=rr), s, "tess", t).sum()

    g = jax.grad(loss)(0.1)
    assert np.isfinite(g) and g < 0  # bigger planet, less flux


def test_orbit_reused_from_parameterization():
    orbit = companion_orbit(values(), "b")
    assert float(orbit.radius) == pytest.approx(0.1)
