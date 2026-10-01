"""The one exception type ``jaxoplanet init`` reports to the user."""


class PrepareError(ValueError):
    """A fit directory cannot be prepared (bad input, missing data, ...)."""
