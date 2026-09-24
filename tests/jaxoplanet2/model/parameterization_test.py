import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from jaxoplanet2.model.parameterization import (
    companion_geometry,
    companion_orbit,
    host_density_cgs,
    mid_eclipse_offset,
)

jax.config.update("jax_enable_x64", True)

# (period, f_c, f_s, cosi, offset) from allesfitter.batman_backend.mid_eclipse_offset
ALLESFITTER_OFFSETS = [
    (4.0, 0.3, 0.4, 0.05, -0.00012570010559979057),
    (4.0, -0.2, -0.5, 0.02, 6.179538298732634e-05),
    (10.0, 0.5, 0.1, 0.08, -0.0020257285780578877),
    (2.5, 0.0, 0.6, 0.1, -9.71836152669408e-15),
    (7.0, -0.6, 0.2, 0.03, 0.00020522439962068206),
]


def values(**overrides):
    base = {
        "b_rr": 0.1,
        "b_rsuma": 0.12,
        "b_cosi": 0.03,
        "b_epoch": 2459000.3,
        "b_period": 3.0,
    }
    return {**base, **overrides}


def test_geometry_matches_allesfitter_definitions():
    g = companion_geometry(values(b_f_c=0.3, b_f_s=0.4), "b")
    assert g.a_over_rstar == pytest.approx(1.1 / 0.12)
    assert g.radius_1 == pytest.approx(0.12 / 1.1)
    assert g.eccentricity == pytest.approx(0.25)
    assert g.cos_omega == pytest.approx(0.6)
    assert g.sin_omega == pytest.approx(0.8)
    assert g.inclination == pytest.approx(np.arccos(0.03))


def test_geometry_defaults_circular_and_no_rv():
    g = companion_geometry(values(), "b")
    assert g.eccentricity == 0.0
    assert g.K == 0.0


def _offset(period, f_c, f_s, cosi):
    e = f_c**2 + f_s**2
    w = np.arctan2(f_s, f_c)
    return float(mid_eclipse_offset(period, e, np.cos(w), np.sin(w), np.arccos(cosi)))


def _exact_offset(period, f_c, f_s, cosi):
    """Brute-force reference: minimise the projected separation with scipy."""
    e, w, sin2_i = f_c**2 + f_s**2, np.arctan2(f_s, f_c), 1 - cosi**2

    def sep2(f):
        return ((1 - e**2) / (1 + e * np.cos(f))) ** 2 * (
            1 - np.sin(f + w) ** 2 * sin2_i
        )

    def mean_anomaly(f):
        E = 2 * np.arctan2(
            np.sqrt(1 - e) * np.sin(f / 2), np.sqrt(1 + e) * np.cos(f / 2)
        )
        return E - e * np.sin(E)

    f_conj = np.pi / 2 - w
    f_min = minimize_scalar(
        sep2, bracket=(f_conj - 0.3, f_conj, f_conj + 0.3), tol=1e-14
    ).x
    d = (mean_anomaly(f_min) - mean_anomaly(f_conj) + np.pi) % (2 * np.pi) - np.pi
    return period * d / (2 * np.pi)


@pytest.mark.parametrize("period, f_c, f_s, cosi, _", ALLESFITTER_OFFSETS)
def test_mid_eclipse_offset_is_exact(period, f_c, f_s, cosi, _):
    exact = _exact_offset(period, f_c, f_s, cosi)
    np.testing.assert_allclose(_offset(period, f_c, f_s, cosi), exact, atol=1e-8)


@pytest.mark.parametrize("period, f_c, f_s, cosi, expected", ALLESFITTER_OFFSETS)
def test_mid_eclipse_offset_matches_allesfitter(period, f_c, f_s, cosi, expected):
    # allesfitter searches a grid with parabolic refinement: good to ~2e-7 d
    np.testing.assert_allclose(_offset(period, f_c, f_s, cosi), expected, atol=2e-7)


def test_mid_eclipse_offset_is_zero_for_circular_orbits():
    assert mid_eclipse_offset(3.0, 0.0, 0.0, 1.0, 1.5) == pytest.approx(0.0, abs=1e-15)


def test_epoch_is_time_of_minimum_projected_separation():
    v = values(b_f_c=0.4, b_f_s=0.3, b_cosi=0.08, b_period=6.0)
    orbit = companion_orbit(v, "b")
    t = v["b_epoch"] + np.linspace(-0.01, 0.01, 20001)
    x, y, _ = orbit.relative_position(t)
    sep = np.hypot(np.asarray(x), np.asarray(y)).ravel()
    assert abs(t[np.argmin(sep)] - v["b_epoch"]) < 2e-6


def test_orbit_has_requested_scaled_semimajor_axis():
    v = values()
    orbit = companion_orbit(v, "b")
    np.testing.assert_allclose(orbit.semimajor / orbit.central_radius, 1.1 / 0.12)


def test_orbit_without_transit_parameters_is_rv_only():
    v = {"c_epoch": 10.0, "c_period": 5.0, "c_K": 0.01}
    orbit = companion_orbit(v, "c")
    rv = np.asarray(orbit.radial_velocity(np.linspace(0, 5, 5001)))
    assert np.max(np.abs(rv)) == pytest.approx(0.01, rel=1e-3)


def test_gradients_are_finite_at_zero_eccentricity():
    def f(f_c, f_s):
        v = values(b_f_c=f_c, b_f_s=f_s)
        orbit = companion_orbit(v, "b")
        x, y, _ = orbit.relative_position(jnp.asarray(v["b_epoch"] + 0.01))
        return jnp.sum(x**2 + y**2)

    grads = jax.grad(f, argnums=(0, 1))(0.0, 0.0)
    assert all(np.isfinite(g) for g in grads)


def test_orbit_is_jittable():
    @jax.jit
    def f(rr):
        orbit = companion_orbit(values(b_rr=rr, b_f_c=0.1, b_f_s=0.2), "b")
        return orbit.radius

    assert float(f(0.05)) == pytest.approx(0.05)


def test_host_density_from_kepler():
    # a/R* = 215.03 and P = 365.25 d is the Sun: 1.41 g/cm^3
    rsuma = 1.0 / 215.032
    rho = host_density_cgs(values(b_rr=0.0, b_rsuma=rsuma, b_period=365.25), "b")
    assert rho == pytest.approx(1.41, rel=0.01)
