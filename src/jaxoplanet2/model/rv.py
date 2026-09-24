"""Keplerian radial velocities per instrument, following allesfitter.

The host's RV is the sum over ``companions_rv`` of each companion's Keplerian
signal with semi-amplitude ``<c>_K`` (in the data's units, usually km/s).
Systemic offsets and trends are baselines, handled elsewhere.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from jaxoplanet2.io.settings import Settings
from jaxoplanet2.model.exposure import integrate_exposure
from jaxoplanet2.model.parameterization import companion_orbit

Values = Mapping[str, jax.Array | float]


def _instantaneous_rv(values: Values, settings: Settings, time: jax.Array) -> jax.Array:
    rv = jnp.zeros_like(time)
    for c in settings.companions_rv:
        rv = rv + companion_orbit(values, c).radial_velocity(time)
    return rv


def rv_model(
    values: Values, settings: Settings, inst: str, time: np.ndarray | jax.Array
) -> jax.Array:
    """Model RV of instrument ``inst`` at ``time`` (no systemic offset)."""
    return integrate_exposure(
        lambda t: _instantaneous_rv(values, settings, t),
        time,
        settings.t_exp[inst],
        settings.t_exp_n_int[inst],
    )
