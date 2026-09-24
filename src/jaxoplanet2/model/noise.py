"""White-noise models (``error_<kind>_<inst> = sample``), as in allesfitter.

- Photometry: the file's errors only set *relative* weights. The absolute level
  is sampled: ``sigma = yerr / mean(yerr) * exp(ln_err_flux_<inst>)``.
- RV: the file's errors are kept and a jitter is added in quadrature:
  ``sigma = sqrt(yerr^2 + exp(ln_jitter_rv_<inst>)^2)``.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from jaxoplanet2.io.data import Dataset
from jaxoplanet2.io.settings import Settings

Values = Mapping[str, jax.Array | float]
LOG_2PI = jnp.log(2.0 * jnp.pi)


def _require(values: Values, key: str) -> jax.Array:
    if key not in values:
        raise KeyError(f"missing parameter '{key}' (needed by error_*_... = sample)")
    return jnp.asarray(values[key], dtype=float)


def white_noise_sigma(values: Values, settings: Settings, data: Dataset) -> jax.Array:
    """Per-point standard deviation of the white noise for ``data``."""
    del settings  # only 'sample' errors exist; kept for a uniform model API
    if data.kind == "flux":
        scale = jnp.exp(_require(values, f"ln_err_flux_{data.inst}"))
        # normalised by the mean error of the data being fitted, i.e. after
        # fast_fit windowing (verified against allesfitter's calculate_yerr_w)
        return data.yerr / jnp.mean(data.yerr) * scale
    jitter = jnp.exp(_require(values, f"ln_jitter_rv_{data.inst}"))
    return jnp.sqrt(data.yerr**2 + jitter**2)


def gaussian_loglike(residual: jax.Array, sigma: jax.Array) -> jax.Array:
    """Independent Gaussian log-likelihood, summed over points."""
    return -0.5 * jnp.sum((residual / sigma) ** 2 + 2.0 * jnp.log(sigma) + LOG_2PI)
