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
from jaxoplanet2.model.ttv import TtvWindows

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


def flux_model(
    values: Values,
    settings: Settings,
    inst: str,
    time: np.ndarray | jax.Array,
    ttv: Mapping[str, TtvWindows] | None = None,
) -> jax.Array:
    """Normalised model flux of instrument ``inst`` at ``time``.

    ``ttv`` (from :func:`jaxoplanet2.model.ttv.ttv_windows`) shifts each observed
    transit of those companions by its own TTV. Companions' depths add linearly,
    so each is integrated over the exposure on its own (shifted) time axis.
    """
    u = ld_coefficients(values, settings, inst)
    times = np.asarray(time, dtype=float)
    depth = 0.0
    for c in settings.companions_phot:
        light_curve = limb_dark_light_curve(companion_orbit(values, c), u)
        t_c, inside = jnp.asarray(times), 1.0
        if ttv and c in ttv:
            offset, inside = ttv[c].shift(values, times)
            t_c = t_c - offset
        dip = integrate_exposure(
            light_curve, t_c, settings.t_exp[inst], settings.t_exp_n_int[inst]
        )
        depth = depth - dip * inside
    dil = values.get(f"dil_{inst}", 0.0)
    return 1.0 - depth * (1.0 - dil)
