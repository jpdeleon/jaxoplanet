"""allesfitter's prior grammar (the ``bounds`` column of params.csv).

``uniform lo hi``, ``normal mu sd`` and ``trunc_normal lo hi mu sd``.
"""

import math
import warnings
from dataclasses import dataclass

import numpyro.distributions as dist


class PriorError(ValueError):
    """A ``bounds`` entry cannot be parsed or is invalid."""


@dataclass(frozen=True)
class Uniform:
    lower: float
    upper: float

    @property
    def support(self) -> tuple[float, float]:
        return (self.lower, self.upper)

    def describe(self) -> str:
        return f"uniform {self.lower!r} {self.upper!r}"


@dataclass(frozen=True)
class Normal:
    mean: float
    sd: float

    @property
    def support(self) -> tuple[float, float]:
        return (-math.inf, math.inf)

    def describe(self) -> str:
        return f"normal {self.mean!r} {self.sd!r}"


@dataclass(frozen=True)
class TruncNormal:
    lower: float
    upper: float
    mean: float
    sd: float

    @property
    def support(self) -> tuple[float, float]:
        return (self.lower, self.upper)

    def describe(self) -> str:
        return f"trunc_normal {self.lower!r} {self.upper!r} {self.mean!r} {self.sd!r}"


Prior = Uniform | Normal | TruncNormal

_ARITY = {
    "uniform": (Uniform, 2),
    "normal": (Normal, 2),
    "trunc_normal": (TruncNormal, 4),
}


def parse_bounds(text: str) -> Prior:
    tokens = text.split()
    if not tokens:
        raise PriorError("empty bounds; expected e.g. 'uniform 0 1'")
    kind = tokens[0].lower()
    if kind not in _ARITY:
        raise PriorError(
            f"unknown prior '{tokens[0]}'; use uniform, normal or trunc_normal"
        )
    cls, arity = _ARITY[kind]
    if len(tokens) - 1 < arity:
        raise PriorError(f"'{text}': {kind} needs {arity} numbers")
    if len(tokens) - 1 > arity:
        # allesfitter silently drops trailing numbers (e.g. an asymmetric
        # upper error); do the same for compatibility, but say so
        warnings.warn(
            f"bounds '{text}': {kind} takes {arity} numbers, ignoring the rest "
            "(as allesfitter does)",
            UserWarning,
            stacklevel=2,
        )
    try:
        numbers = [float(t) for t in tokens[1 : arity + 1]]
    except ValueError as e:
        raise PriorError(f"'{text}': {e}") from e
    if not all(math.isfinite(x) for x in numbers):
        raise PriorError(f"'{text}': all numbers must be finite")
    prior = cls(*numbers)
    _validate(prior, text)
    return prior


def _validate(prior: Prior, text: str) -> None:
    if isinstance(prior, (Uniform, TruncNormal)) and not prior.lower < prior.upper:
        raise PriorError(f"'{text}': lower bound must be below upper bound")
    if isinstance(prior, (Normal, TruncNormal)) and not prior.sd > 0:
        raise PriorError(f"'{text}': standard deviation must be positive")


def to_distribution(prior: Prior) -> dist.Distribution:
    if isinstance(prior, Uniform):
        return dist.Uniform(prior.lower, prior.upper)
    if isinstance(prior, Normal):
        return dist.Normal(prior.mean, prior.sd)
    return dist.TruncatedNormal(
        loc=prior.mean, scale=prior.sd, low=prior.lower, high=prior.upper
    )
