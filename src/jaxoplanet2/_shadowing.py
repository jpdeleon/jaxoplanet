"""Detect an upstream ``jaxoplanet`` install that would clash with this fork.

The ``jaxoplanet2`` distribution ships its own copy of the ``jaxoplanet`` module.
Installing the upstream ``jaxoplanet`` distribution into the same environment
makes the two overwrite each other's files, so we warn loudly when we see it.
"""

import warnings
from collections.abc import Callable, Iterable
from importlib import metadata
from typing import Any

UPSTREAM_DISTRIBUTION = "jaxoplanet"


class UpstreamShadowingWarning(UserWarning):
    """The upstream jaxoplanet distribution is installed next to jaxoplanet2."""


def upstream_installed(
    distributions: Callable[[], Iterable[Any]] = metadata.distributions,
) -> bool:
    for dist in distributions():
        name = dist.metadata["Name"] or ""
        if name.lower() == UPSTREAM_DISTRIBUTION:
            return True
    return False


def warn_if_upstream_installed(
    distributions: Callable[[], Iterable[Any]] = metadata.distributions,
) -> None:
    if upstream_installed(distributions):
        warnings.warn(
            "Both 'jaxoplanet2' and the upstream 'jaxoplanet' distribution are "
            "installed. They both provide the 'jaxoplanet' module and overwrite "
            "each other's files; uninstall one of them.",
            UpstreamShadowingWarning,
            stacklevel=2,
        )
