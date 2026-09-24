"""Instrumental baselines, following allesfitter's definitions.

- ``sample_*`` baselines have sampled parameters (see ``requirements.BASELINE_PARAMS``).
- ``hybrid_*`` baselines are profiled: fitted analytically to the residuals of the
  current model, exactly as allesfitter does, so they add no parameters.
- ``sample_GP_*`` baselines are Gaussian processes; they change the likelihood
  rather than the mean and live in :mod:`jaxoplanet2.model.gp`.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from jaxoplanet2.io.data import Dataset
from jaxoplanet2.io.settings import Settings

Values = Mapping[str, jax.Array | float]
HYBRID_POLY = "hybrid_poly_"


def is_gp(baseline: str) -> bool:
    return baseline.lower().startswith("sample_gp_")


def _param(values: Values, name: str, baseline: str) -> jax.Array:
    if name not in values:
        raise KeyError(f"missing parameter '{name}' (needed by {baseline})")
    return jnp.asarray(values[name], dtype=float)


def _hybrid_poly(
    time: jax.Array, residual: jax.Array, sigma: jax.Array, order: int
) -> jax.Array:
    # allesfitter: numpy polyfit with w = 1/(sigma/mean(sigma)), i.e. weighted
    # least squares. The fitted values do not depend on how time is scaled, so
    # use a well-conditioned [0, 1] axis instead of allesfitter's t/t[-1].
    x = (time - time[0]) / (time[-1] - time[0])
    vander = jnp.vander(x, order + 1, increasing=True)
    w = 1.0 / sigma
    coeffs, *_ = jnp.linalg.lstsq(vander * w[:, None], residual * w)
    return vander @ coeffs


def deterministic_baseline(
    values: Values,
    settings: Settings,
    data: Dataset,
    residual: jax.Array,
    sigma: jax.Array,
) -> jax.Array:
    """Baseline at the data's time stamps.

    ``residual`` is data minus astrophysical model; ``sigma`` the white noise.
    """
    baseline = settings.baseline[(data.kind, data.inst)]
    kind = baseline.lower()
    suffix = f"{data.kind}_{data.inst}"
    time = jnp.asarray(data.time)
    if kind == "none":
        return jnp.zeros_like(time)
    if kind == "sample_offset":
        return jnp.full_like(time, _param(values, f"baseline_offset_{suffix}", baseline))
    if kind == "sample_linear":
        offset = _param(values, f"baseline_offset_{suffix}", baseline)
        slope = _param(values, f"baseline_slope_{suffix}", baseline)
        return offset + slope * (time - time[0]) / (time[-1] - time[0])
    if kind == "hybrid_offset":
        # np.average(residual, weights=1/(sigma/mean(sigma))): 1/sigma, not 1/sigma^2
        w = 1.0 / sigma
        return jnp.full_like(time, jnp.sum(w * residual) / jnp.sum(w))
    if kind.startswith(HYBRID_POLY):
        return _hybrid_poly(time, residual, sigma, int(kind[len(HYBRID_POLY) :]))
    if is_gp(kind):
        raise ValueError(f"{baseline} is a GP baseline, not a deterministic one")
    raise ValueError(f"unknown baseline '{baseline}'")
