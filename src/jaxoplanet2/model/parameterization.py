"""jaxoplanet-native transit parameters -> exact Keplerian orbits.

Each companion is described by jaxoplanet's ``TransitOrbit`` inputs, which are
what transit data constrain directly and therefore sample well:

- ``<c>_period``, ``<c>_time_transit`` (mid-transit time),
- ``<c>_duration`` (total transit duration T14, first to fourth contact),
- ``<c>_impact_param`` (b), ``<c>_radius_ratio`` (k = Rp/R*),

plus ``<c>_f_c = sqrt(e) cos(w)``, ``<c>_f_s = sqrt(e) sin(w)`` and ``<c>_K``.
``TransitOrbit`` itself moves the planet in a straight line on a circular orbit;
to stay exact (curved, eccentric, RV-consistent), these parameters are mapped
onto a Keplerian orbit instead by inverting Winn (2010) eqs. 7 and 14:

    b = (a/R*) cos i (1 - e^2) / (1 + e sin w)
    T14 = P/pi asin(sqrt((1 + k)^2 - b^2) / ((a/R*) sin i))
          * sqrt(1 - e^2) / (1 + e sin w)

which is exact for circular orbits. ``time_transit`` is the time of minimum
projected separation (allesfitter's epoch); jaxoplanet's Keplerian
``time_transit`` is inferior conjunction, and :func:`mid_eclipse_offset` bridges
the two for eccentric, inclined orbits. Every companion gets its own central
body sized to its own a/R*, as allesfitter models companions independently.

Everything here is pure JAX, so it can be jitted and differentiated.
"""

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxoplanet.orbits.keplerian import Central, OrbitalBody, System

# a/R* used for RV-only companions, whose RV does not depend on it
RV_ONLY_A_OVER_RSTAR = 10.0
NEWTON_STEPS = 8
MAX_NEWTON_STEP = 0.5  # radians of true anomaly
# G in cgs with the jaxoplanet time unit (days): rho = 3 pi (a/R*)^3 / (G P^2)
G_CGS = 6.6743e-8
SECONDS_PER_DAY = 86400.0
# the native transit parameters, as named by jaxoplanet's TransitOrbit
TRANSIT_PARAMS = ("radius_ratio", "duration", "impact_param", "time_transit", "period")


class Geometry(NamedTuple):
    a_over_rstar: jax.Array
    radius_1: jax.Array  # R*/a, allesfitter's radius_1
    rr: jax.Array  # radius ratio k
    impact_param: jax.Array
    duration: jax.Array  # T14; NaN for companions without a transit
    inclination: jax.Array
    eccentricity: jax.Array
    cos_omega: jax.Array
    sin_omega: jax.Array
    time_transit: jax.Array
    period: jax.Array
    K: jax.Array

    @property
    def rsuma(self) -> jax.Array:
        """allesfitter's (R* + Rp)/a."""
        return (1.0 + self.rr) / self.a_over_rstar


def eccentricity_factors(ecc, sin_omega) -> tuple[jax.Array, jax.Array]:
    """(b factor (1-e^2)/(1+e sin w), duration factor sqrt(1-e^2)/(1+e sin w))."""
    denom = 1.0 + ecc * sin_omega
    return (1.0 - ecc**2) / denom, jnp.sqrt(1.0 - ecc**2) / denom


def orbit_from_transit(k, b, duration, period, *, ecc, sin_omega):
    """(a/R*, inclination) that give the transit (k, b, T14) on this orbit."""
    b_factor, t_factor = eccentricity_factors(ecc, sin_omega)
    chord = jnp.sqrt(jnp.maximum((1.0 + k) ** 2 - b**2, 0.0))
    a_sin_i = chord / jnp.sin(jnp.pi * duration / (period * t_factor))
    a_cos_i = b / b_factor
    return jnp.hypot(a_sin_i, a_cos_i), jnp.arctan2(a_sin_i, a_cos_i)


def transit_from_orbit(k, a_over_rstar, inclination, period, *, ecc, sin_omega):
    """(b, T14): the forward map of :func:`orbit_from_transit`."""
    b_factor, t_factor = eccentricity_factors(ecc, sin_omega)
    b = a_over_rstar * jnp.cos(inclination) * b_factor
    chord = jnp.sqrt(jnp.maximum((1.0 + k) ** 2 - b**2, 0.0))
    arg = chord / (a_over_rstar * jnp.sin(inclination))
    return b, period / jnp.pi * jnp.arcsin(jnp.clip(arg, -1.0, 1.0)) * t_factor


def companion_geometry(values: Mapping[str, jax.Array | float], c: str) -> Geometry:
    def get(name: str, default: float | None = None) -> jax.Array:
        key = f"{c}_{name}"
        if key not in values:
            if default is None:
                raise KeyError(f"missing parameter '{key}'")
            return jnp.asarray(default, dtype=float)
        return jnp.asarray(values[key], dtype=float)

    f_c, f_s = get("f_c", 0.0), get("f_s", 0.0)
    ecc = f_c**2 + f_s**2
    # double-where keeps gradients finite at e = 0, where omega is undefined
    positive = ecc > 0
    sqrt_e = jnp.sqrt(jnp.where(positive, ecc, 1.0))
    cos_omega = jnp.where(positive, f_c / sqrt_e, 0.0)
    sin_omega = jnp.where(positive, f_s / sqrt_e, 1.0)
    k, b, period = get("radius_ratio", 0.0), get("impact_param", 0.0), get("period")
    if f"{c}_duration" in values:
        duration = get("duration")
        a_over_rstar, inclination = orbit_from_transit(
            k, b, duration, period, ecc=ecc, sin_omega=sin_omega
        )
    else:  # RV-only companion: the transit geometry does not enter the model
        duration = jnp.asarray(jnp.nan)
        a_over_rstar = jnp.asarray(RV_ONLY_A_OVER_RSTAR)
        inclination = jnp.asarray(jnp.pi / 2)
    return Geometry(
        a_over_rstar=a_over_rstar,
        radius_1=1.0 / a_over_rstar,
        rr=k,
        impact_param=b,
        duration=duration,
        inclination=inclination,
        eccentricity=ecc,
        cos_omega=cos_omega,
        sin_omega=sin_omega,
        time_transit=get("time_transit"),
        period=period,
        K=get("K", 0.0),
    )


def _mean_anomaly(true_anomaly: jax.Array, ecc: jax.Array) -> jax.Array:
    half = 0.5 * true_anomaly
    ecc_anomaly = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 - ecc) * jnp.sin(half), jnp.sqrt(1.0 + ecc) * jnp.cos(half)
    )
    return ecc_anomaly - ecc * jnp.sin(ecc_anomaly)


def mid_eclipse_offset(
    period: jax.Array | float,
    ecc: jax.Array | float,
    cos_omega: jax.Array | float,
    sin_omega: jax.Array | float,
    inclination: jax.Array | float,
) -> jax.Array:
    """Time of minimum projected separation minus time of inferior conjunction.

    Zero for circular or edge-on orbits. allesfitter finds the minimum on a grid;
    here a few Newton steps on d(sep^2)/df = 0 do the same, differentiably.
    """
    omega = jnp.arctan2(sin_omega, cos_omega)
    sin2_i = jnp.sin(inclination) ** 2

    def sep2(f: jax.Array) -> jax.Array:
        r = (1.0 - ecc**2) / (1.0 + ecc * jnp.cos(f))
        return r**2 * (1.0 - jnp.sin(f + omega) ** 2 * sin2_i)

    d1, d2 = jax.grad(sep2), jax.grad(jax.grad(sep2))
    f_conj = 0.5 * jnp.pi - omega

    def newton(f: jax.Array, _: None) -> tuple[jax.Array, None]:
        curvature = d2(f)
        step = jnp.where(
            curvature > 0, -d1(f) / jnp.where(curvature > 0, curvature, 1.0), 0.0
        )
        return f + jnp.clip(step, -MAX_NEWTON_STEP, MAX_NEWTON_STEP), None

    f_min, _ = jax.lax.scan(newton, f_conj, None, length=NEWTON_STEPS)
    d_mean = _mean_anomaly(f_min, ecc) - _mean_anomaly(f_conj, ecc)
    d_mean = (d_mean + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    return period * d_mean / (2.0 * jnp.pi)


def is_fixed_circular(values: Mapping[str, jax.Array | float], c: str) -> bool:
    """True if ``f_c`` and ``f_s`` are known, before tracing, to be zero.

    Then the orbit is circular for every sample, the mid-eclipse offset is
    exactly zero, and jaxoplanet's cheaper circular code path can be used.
    """
    for name in (f"{c}_f_c", f"{c}_f_s"):
        value = values.get(name, 0.0)
        if isinstance(value, jax.core.Tracer) or float(value) != 0.0:
            return False
    return True


def companion_orbit(values: Mapping[str, jax.Array | float], c: str) -> OrbitalBody:
    """The jaxoplanet orbit of companion ``c``, in units of the host radius."""
    g = companion_geometry(values, c)
    central = Central.from_orbital_properties(
        period=g.period, semimajor=g.a_over_rstar, radius=1.0
    )
    common = {
        "period": g.period,
        "inclination": g.inclination,
        "radius": g.rr,
        "radial_velocity_semiamplitude": g.K,
    }
    if is_fixed_circular(values, c):
        system = System(central).add_body(time_transit=g.time_transit, **common)
        return system.bodies[0]
    offset = mid_eclipse_offset(
        g.period, g.eccentricity, g.cos_omega, g.sin_omega, g.inclination
    )
    system = System(central).add_body(
        time_transit=g.time_transit - offset,
        eccentricity=g.eccentricity,
        cos_omega_peri=g.cos_omega,
        sin_omega_peri=g.sin_omega,
        **common,
    )
    return system.bodies[0]


def host_density_cgs(values: Mapping[str, jax.Array | float], c: str) -> jax.Array:
    """Host density implied by the transit's a/R* and ``period`` (Kepler's law).

    Neglects the companion's mass, as allesfitter does for photometry-only fits.
    """
    g = companion_geometry(values, c)
    period_s = g.period * SECONDS_PER_DAY
    return 3.0 * jnp.pi * g.a_over_rstar**3 / (G_CGS * period_s**2)
