from importlib import metadata

try:
    __version__ = metadata.version("jaxoplanet2")
except metadata.PackageNotFoundError:  # running from a source tree
    __version__ = "0+unknown"
