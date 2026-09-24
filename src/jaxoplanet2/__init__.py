"""allesfitter-style transit and RV fitting powered by jaxoplanet."""

__all__ = ["__version__"]

from jaxoplanet2._metadata import __version__ as __version__
from jaxoplanet2._shadowing import warn_if_upstream_installed

warn_if_upstream_installed()
