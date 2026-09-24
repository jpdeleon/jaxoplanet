"""``shift_epoch``: move each epoch (and its prior) into the middle of the data.

A reference epoch near the data centre decorrelates epoch and period. This is a
port of allesfitter's ``Basement.change_epoch``, including how it shifts each
kind of epoch prior by ``N`` periods.
"""

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from jaxoplanet2.io.data import Dataset
from jaxoplanet2.io.params import Param, ParamTable
from jaxoplanet2.io.priors import Normal, Prior, TruncNormal, Uniform
from jaxoplanet2.io.settings import Settings


class EpochShiftError(ValueError):
    """allesfitter cannot shift this combination of epoch/period priors."""


def first_epoch(time: np.ndarray, epoch: float, period: float, width: float) -> float:
    """First transit (including egress) at or after the start of the data."""
    start = float(np.min(time))
    first = epoch + width / 2.0
    if start <= first:
        first -= math.floor((first - start) / period) * period
    else:
        first += math.ceil((start - first) / period) * period
    return first - width / 2.0


def mid_epoch(
    time: np.ndarray, epoch: float, period: float, width: float
) -> tuple[float, int]:
    """(epoch of the transit nearest the data centre, periods shifted)."""
    n_half = round((np.max(time) - np.min(time)) / 2.0 / period)
    mid = first_epoch(time, epoch, period, width) + n_half * period
    return mid, round((mid - epoch) / period)


def _shift_uniform_bounds(lo: float, hi: float, n: int, p_lo: float, p_hi: float):
    # widen by the period bounds; they swap roles when shifting backwards
    return (lo + n * p_lo, hi + n * p_hi) if n > 0 else (lo + n * p_hi, hi + n * p_lo)


def _translate(prior: Prior, delta: float) -> Prior:
    if isinstance(prior, Uniform):
        return Uniform(prior.lower + delta, prior.upper + delta)
    if isinstance(prior, Normal):
        return Normal(prior.mean + delta, prior.sd)
    return TruncNormal(
        prior.lower + delta, prior.upper + delta, prior.mean + delta, prior.sd
    )


def shift_prior(
    epoch_prior: Prior, period_prior: Prior | None, n: int, period: float
) -> Prior:
    """The epoch prior after moving the epoch by ``n`` periods."""
    e, p = epoch_prior, period_prior
    if p is None:  # fixed period: an exact translation
        return _translate(e, n * period)
    if isinstance(e, Uniform) and isinstance(p, Uniform):
        return Uniform(*_shift_uniform_bounds(e.lower, e.upper, n, p.lower, p.upper))
    if isinstance(e, Normal) and isinstance(p, Normal):
        return Normal(e.mean + n * p.mean, math.hypot(e.sd, n * p.sd))
    if isinstance(e, TruncNormal) and isinstance(p, TruncNormal):
        lo, hi = _shift_uniform_bounds(e.lower, e.upper, n, p.lower, p.upper)
        return TruncNormal(lo, hi, e.mean + n * p.mean, math.hypot(e.sd, n * p.sd))
    if isinstance(e, Uniform) and isinstance(p, (Normal, TruncNormal)):
        step = n * (period + p.sd)
        return Uniform(e.lower + step, e.upper + step)
    raise EpochShiftError(
        f"shift_epoch cannot combine a {type(e).__name__} epoch prior with a "
        f"{type(p).__name__} period prior (allesfitter cannot either); "
        "use matching priors or set shift_epoch,False"
    )


def _epoch_times(settings: Settings, data: Mapping[str, Dataset], c: str) -> np.ndarray:
    chosen = settings.inst_for_epoch[c]
    insts = settings.inst_all if chosen == "all" else tuple(chosen.split())
    return np.concatenate([data[i].time for i in insts])


def shift_epochs(
    settings: Settings, params: ParamTable, data: Mapping[str, Dataset]
) -> tuple[ParamTable, dict[str, int]]:
    """Params with every companion's epoch moved to the data centre."""
    width = settings.fast_fit_width or 0.0
    table, shifts = params, {}
    for c in settings.companions_all:
        epoch, period = params[f"{c}_epoch"], params[f"{c}_period"]
        time = _epoch_times(settings, data, c)
        new_epoch, n = mid_epoch(time, epoch.value, period.value, width)
        shifts[c] = n
        if n == 0:
            continue
        prior = epoch.prior
        if prior is not None:
            prior = shift_prior(prior, period.prior, n, period.value)
        table = _replace_param(table, replace(epoch, value=new_epoch, prior=prior))
    return table, shifts


def _replace_param(table: ParamTable, new: Param) -> ParamTable:
    return ParamTable(tuple(new if p.name == new.name else p for p in table))
