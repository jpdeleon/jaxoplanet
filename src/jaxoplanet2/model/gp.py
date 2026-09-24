"""Gaussian-process baselines (``sample_GP_*``), as in allesfitter.

allesfitter models the residuals ``data - model`` with a celerite GP whose
diagonal is the white-noise variance. The same kernels exist in tinygp's
quasiseparable module, which scales linearly with the number of points:

- ``sample_GP_Matern32``: log_sigma, log_rho -> ``Matern32(scale=rho, sigma)``.
  (celerite's Matern32Term is an eps=0.01 approximation; tinygp's is exact.)
- ``sample_GP_SHO``: log_S0, log_Q, log_omega0 -> ``SHO(omega0, Q)`` with
  ``sigma**2 = S0 * omega0 * Q`` (celerite's power normalisation).
- ``sample_GP_real``: log_a, log_c -> ``a exp(-c tau)`` = ``Exp(1/c, sqrt(a))``.
- ``sample_GP_complex``: log_a..log_d -> ``Celerite(a, b, c, d)``, identical to
  celerite's ComplexTerm ``exp(-c tau) [a cos(d tau) + b sin(d tau)]``.

An optional ``baseline_gp_offset_<kind>_<inst>`` is the GP mean.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from tinygp import GaussianProcess
from tinygp.kernels import quasisep

from jaxoplanet2.io.data import Dataset
from jaxoplanet2.io.settings import Settings

Values = Mapping[str, jax.Array | float]


def _param(values: Values, name: str) -> jax.Array:
    if name not in values:
        raise KeyError(f"missing GP hyperparameter '{name}'")
    return jnp.asarray(values[name], dtype=float)


def gp_kernel(values: Values, baseline: str, suffix: str) -> quasisep.Quasisep:
    """The quasiseparable kernel of ``baseline`` for ``<kind>_<inst>``."""
    kind = baseline.lower()

    def p(name: str) -> jax.Array:
        return _param(values, f"baseline_gp_{name}_{suffix}")

    if kind == "sample_gp_matern32":
        return quasisep.Matern32(
            scale=jnp.exp(p("matern32_lnrho")), sigma=jnp.exp(p("matern32_lnsigma"))
        )
    if kind == "sample_gp_sho":
        S0, Q, w0 = (
            jnp.exp(p("sho_lnS0")),
            jnp.exp(p("sho_lnQ")),
            jnp.exp(p("sho_lnomega0")),
        )
        return quasisep.SHO(omega=w0, quality=Q, sigma=jnp.sqrt(S0 * w0 * Q))
    if kind == "sample_gp_real":
        a, c = jnp.exp(p("real_lna")), jnp.exp(p("real_lnc"))
        return quasisep.Exp(scale=1.0 / c, sigma=jnp.sqrt(a))
    if kind == "sample_gp_complex":
        # celerite's ComplexTerm: exp(-c tau) [a cos(d tau) + b sin(d tau)]
        a, b, c, d = (jnp.exp(p(f"complex_ln{x}")) for x in "abcd")
        return quasisep.Celerite(a=a, b=b, c=c, d=d)
    raise ValueError(f"'{baseline}' is not a GP baseline")


def residual_process(
    values: Values, settings: Settings, data: Dataset, sigma: jax.Array
) -> GaussianProcess:
    """GP over ``data - model`` with the white noise on the diagonal."""
    baseline = settings.baseline[(data.kind, data.inst)]
    suffix = f"{data.kind}_{data.inst}"
    mean = values.get(f"baseline_gp_offset_{suffix}", 0.0)
    return GaussianProcess(
        gp_kernel(values, baseline, suffix),
        jnp.asarray(data.time),
        diag=sigma**2,
        mean=jnp.asarray(mean, dtype=float),
    )
