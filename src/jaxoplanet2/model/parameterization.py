"""allesfitter parameters -> jaxoplanet orbits.

allesfitter describes each companion by ``rr = Rp/R*``, ``rsuma = (R* + Rp)/a``,
``cosi``, ``epoch``, ``period``, ``f_c = sqrt(e) cos(w)``, ``f_s = sqrt(e) sin(w)``
and ``K``. Two conventions matter when mapping these onto jaxoplanet:

- allesfitter's ``epoch`` is the time of *minimum projected separation* (mid
  eclipse), while jaxoplanet's ``time_transit`` is inferior conjunction. They
  differ for eccentric, inclined orbits; :func:`mid_eclipse_offset` bridges them.
- Every companion gets its own central body sized so that ``a/R*`` matches its
  ``rsuma``, just as allesfitter models each companion independently.

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


class Geometry(NamedTuple):
    a_over_rstar: jax.Array
    radius_1: jax.Array  # R*/a, allesfitter's radius_1
    rr: jax.Array
    inclination: jax.Array
    eccentricity: jax.Array
    cos_omega: jax.Array
    sin_omega: jax.Array
    epoch: jax.Array
    period: jax.Array
    K: jax.Array


def companion_geometry(values: Mapping[str, jax.Array | float], c: str) -> Geometry:
    def get(name: str, default: float | None = None) -> jax.Array:
        key = f"{c}_{name}"
        if key not in values:
            if default is None:
                raise KeyError(f"missing parameter '{key}'")
            return jnp.asarray(default, dtype=float)
        return jnp.asarray(values[key], dtype=float)

    rr = get("rr", 0.0)
    if f"{c}_rsuma" in values:
        rsuma = get("rsuma")
        a_over_rstar = (1.0 + rr) / rsuma
    else:
        a_over_rstar = jnp.asarray(RV_ONLY_A_OVER_RSTAR)
    f_c, f_s = get("f_c", 0.0), get("f_s", 0.0)
    ecc = f_c**2 + f_s**2
    # double-where keeps gradients finite at e = 0, where omega is undefined
    positive = ecc > 0
    sqrt_e = jnp.sqrt(jnp.where(positive, ecc, 1.0))
    return Geometry(
        a_over_rstar=a_over_rstar,
        radius_1=1.0 / a_over_rstar,
        rr=rr,
        inclination=jnp.arccos(get("cosi", 0.0)),
        eccentricity=ecc,
        cos_omega=jnp.where(positive, f_c / sqrt_e, 0.0),
        sin_omega=jnp.where(positive, f_s / sqrt_e, 1.0),
        epoch=get("epoch"),
        period=get("period"),
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


def companion_orbit(values: Mapping[str, jax.Array | float], c: str) -> OrbitalBody:
    """The jaxoplanet orbit of companion ``c``, in units of the host radius."""
    g = companion_geometry(values, c)
    offset = mid_eclipse_offset(
        g.period, g.eccentricity, g.cos_omega, g.sin_omega, g.inclination
    )
    central = Central.from_orbital_properties(
        period=g.period, semimajor=g.a_over_rstar, radius=1.0
    )
    system = System(central).add_body(
        period=g.period,
        time_transit=g.epoch - offset,
        inclination=g.inclination,
        eccentricity=g.eccentricity,
        cos_omega_peri=g.cos_omega,
        sin_omega_peri=g.sin_omega,
        radius=g.rr,
        radial_velocity_semiamplitude=g.K,
    )
    return system.bodies[0]


def host_density_cgs(values: Mapping[str, jax.Array | float], c: str) -> jax.Array:
    """Host density implied by ``rsuma`` and ``period`` (Kepler's third law).

    Neglects the companion's mass, as allesfitter does for photometry-only fits.
    """
    g = companion_geometry(values, c)
    period_s = g.period * SECONDS_PER_DAY
    return 3.0 * jnp.pi * g.a_over_rstar**3 / (G_CGS * period_s**2)
