"""Process-wide JAX configuration driven by the ``jx_*`` settings."""

import jax

from jaxoplanet2.io.settings import Settings


def configure_jax(settings: Settings) -> None:
    # BJD epochs (~2.46e6 d) need float64 to resolve seconds
    jax.config.update("jax_enable_x64", settings.jx.x64)
