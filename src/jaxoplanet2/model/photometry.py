"""Transit light curves per instrument, following allesfitter's conventions.

- Limb darkening is sampled in Kipping's q-space (default) or directly in u-space.
- Companions are independent: ``flux = 1 - sum_c (1 - flux_c)``.
- ``dil_<inst>`` dilutes the transit depth: ``flux = 1 + (flux - 1) (1 - dil)``.
- Long exposures (``t_exp_<inst>``, ``t_exp_n_int_<inst>``) are integrated with
  allesfitter's trapezoid nodes.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet2.io.settings import Settings
from jaxoplanet2.model.exposure import integrate_exposure
from jaxoplanet2.model.parameterization import companion_orbit

Values = Mapping[str, jax.Array | float]


def _require(values: Values, key: str, why: str) -> jax.Array:
    if key not in values:
        raise KeyError(f"missing parameter '{key}' ({why})")
    return jnp.asarray(values[key], dtype=float)


def ld_coefficients(values: Values, settings: Settings, inst: str) -> jax.Array:
    """Polynomial limb-darkening coefficients ``u`` for jaxoplanet."""
    law = settings.ld_law[inst]
    if law is None or law.lower() == "none":
        return jnp.zeros(0)
    space = settings.ld_space[inst]
    why = f"host_ld_law_{inst}={law}, host_ld_space_{inst}={space}"
    if law == "lin":
        return _require(values, f"host_ldc_{space}1_{inst}", why)[None]
    if space == "u":
        return jnp.stack(
            [_require(values, f"host_ldc_u{i}_{inst}", why) for i in (1, 2)]
        )
    q1 = _require(values, f"host_ldc_q1_{inst}", why)
    q2 = _require(values, f"host_ldc_q2_{inst}", why)
    sqrt_q1 = jnp.sqrt(q1)
    return jnp.stack([2.0 * sqrt_q1 * q2, sqrt_q1 * (1.0 - 2.0 * q2)])


def _instantaneous_flux(values: Values, settings: Settings, inst: str, time):
    u = ld_coefficients(values, settings, inst)
    depth = 0.0
    for c in settings.companions_phot:
        orbit = companion_orbit(values, c)
        depth = depth - limb_dark_light_curve(orbit, u)(time)
    return 1.0 - depth


def flux_model(
    values: Values, settings: Settings, inst: str, time: np.ndarray | jax.Array
) -> jax.Array:
    """Normalised model flux of instrument ``inst`` at ``time``."""
    flux = integrate_exposure(
        lambda t: _instantaneous_flux(values, settings, inst, t),
        time,
        settings.t_exp[inst],
        settings.t_exp_n_int[inst],
    )
    dil = values.get(f"dil_{inst}", 0.0)
    return 1.0 + (flux - 1.0) * (1.0 - dil)
