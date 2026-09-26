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
    orbit_from_transit,
    transit_from_orbit,
)
from tests.jaxoplanet2.native import from_allesfitter

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
    return from_allesfitter({**base, **overrides})


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
    t = v["b_time_transit"] + np.linspace(-0.01, 0.01, 20001)
    x, y, _ = orbit.relative_position(t)
    sep = np.hypot(np.asarray(x), np.asarray(y)).ravel()
    assert abs(t[np.argmin(sep)] - v["b_time_transit"]) < 2e-6


def test_orbit_has_requested_scaled_semimajor_axis():
    v = values()
    orbit = companion_orbit(v, "b")
    np.testing.assert_allclose(orbit.semimajor / orbit.central_radius, 1.1 / 0.12)


def test_orbit_without_transit_parameters_is_rv_only():
    v = {"c_time_transit": 10.0, "c_period": 5.0, "c_K": 0.01}
    orbit = companion_orbit(v, "c")
    rv = np.asarray(orbit.radial_velocity(np.linspace(0, 5, 5001)))
    assert np.max(np.abs(rv)) == pytest.approx(0.01, rel=1e-3)


def test_gradients_are_finite_at_zero_eccentricity():
    def f(f_c, f_s):
        v = {**values(), "b_f_c": f_c, "b_f_s": f_s}  # traced: override after
        orbit = companion_orbit(v, "b")
        x, y, _ = orbit.relative_position(jnp.asarray(v["b_time_transit"] + 0.01))
        return jnp.sum(x**2 + y**2)

    grads = jax.grad(f, argnums=(0, 1))(0.0, 0.0)
    assert all(np.isfinite(g) for g in grads)


def test_orbit_is_jittable():
    @jax.jit
    def f(rr):
        v = {**values(b_f_c=0.1, b_f_s=0.2), "b_radius_ratio": rr}
        orbit = companion_orbit(v, "b")
        return orbit.radius

    assert float(f(0.05)) == pytest.approx(0.05)


def test_host_density_from_kepler():
    # a/R* = 215.03 and P = 365.25 d is the Sun: 1.41 g/cm^3
    rsuma = 1.0 / 215.032
    v = values(b_rr=0.0, b_rsuma=rsuma, b_cosi=0.0, b_period=365.25)  # must transit
    rho = host_density_cgs(v, "b")
    assert rho == pytest.approx(1.41, rel=0.01)


def test_fixed_zero_eccentricity_is_detected_statically():
    from jaxoplanet2.model.parameterization import is_fixed_circular

    assert is_fixed_circular(values(), "b")
    assert is_fixed_circular(values(b_f_c=0.0, b_f_s=0.0), "b")
    assert not is_fixed_circular(values(b_f_c=0.1), "b")
    jax.jit(lambda f_c: assert_not_circular({**values(), "b_f_c": f_c}))(0.0)


def assert_not_circular(v):
    from jaxoplanet2.model.parameterization import is_fixed_circular

    assert not is_fixed_circular(v, "b")  # a traced value may become non-zero
    return 0.0


def test_circular_fast_path_matches_the_general_orbit():
    t = values()["b_time_transit"] + np.linspace(-0.3, 0.3, 101)
    fast = companion_orbit(values(), "b").relative_position(t)

    def general_orbit(f_c):
        v = {**values(), "b_f_c": f_c, "b_f_s": f_c}
        return companion_orbit(v, "b").relative_position(t)

    general = jax.jit(general_orbit)(0.0)
    for a, b in zip(fast, general, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-10)


@pytest.mark.parametrize("ecc, sin_omega", [(0.0, 1.0), (0.25, 0.8), (0.4, -0.6)])
def test_transit_parameters_invert_exactly(ecc, sin_omega):
    k, a_over_r, inc, period = 0.1, 11.0, np.arccos(0.03), 3.2
    b, t14 = transit_from_orbit(k, a_over_r, inc, period, ecc=ecc, sin_omega=sin_omega)
    a2, inc2 = orbit_from_transit(k, b, t14, period, ecc=ecc, sin_omega=sin_omega)
    np.testing.assert_allclose([a2, inc2], [a_over_r, inc], rtol=1e-13)


def test_duration_is_winn_t14_for_a_circular_orbit():
    # T14 = P/pi asin(sqrt((1+k)^2 - b^2) / (a/R* sin i)), b = a/R* cos i
    k, a_over_r, cosi, period = 0.1, 11.0, 0.03, 3.2
    b = a_over_r * cosi
    expected = (
        period
        / np.pi
        * np.arcsin(np.sqrt((1 + k) ** 2 - b**2) / (a_over_r * np.sqrt(1 - cosi**2)))
    )
    g = companion_geometry(
        {"b_radius_ratio": k, "b_impact_param": b, "b_duration": expected,
         "b_time_transit": 0.0, "b_period": period},
        "b",
    )  # fmt: skip
    np.testing.assert_allclose(g.a_over_rstar, a_over_r, rtol=1e-13)
    np.testing.assert_allclose(np.cos(g.inclination), cosi, rtol=1e-12)
