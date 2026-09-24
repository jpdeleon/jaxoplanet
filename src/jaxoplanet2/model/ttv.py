"""Transit-timing variations (``fit_ttvs``), as in allesfitter.

allesfitter lists the transits of each photometric companion that the data
actually cover (``get_tmid_observed_transits``: linear-ephemeris mid-times with
data within +-fast_fit_width/2), numbered 1, 2, ... in time order. Transit N is
then modelled with its mid-time shifted by ``<companion>_ttv_transit_N``; points
outside every window see no transit of that companion. The windows are fixed at
load time from the params.csv epoch and period, as in allesfitter.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxoplanet2.io.data import Dataset
from jaxoplanet2.io.epochs import first_epoch
from jaxoplanet2.io.params import ParamTable
from jaxoplanet2.io.settings import Settings

TTV_NAME = re.compile(r"(?P<c>[^_]+)_ttv_transit_(?P<n>\d+)")


class TtvError(ValueError):
    """params.csv does not match the transits the data cover."""


def observed_transits(
    time: np.ndarray, epoch: float, period: float, width: float
) -> tuple[float, ...]:
    """Mid-times (linear ephemeris) of the transits the data cover."""
    time = np.sort(time)
    first = first_epoch(time, epoch, period, width)
    n = int((time[-1] - first) / period) + 1
    tmids = first + period * np.arange(n)
    covered = [
        t for t in tmids if np.any((time >= t - width / 2) & (time <= t + width / 2))
    ]
    return tuple(float(t) for t in covered)


@dataclass(frozen=True)
class TtvWindows:
    companion: str
    tmids: tuple[float, ...]
    width: float

    def index(self, time: np.ndarray) -> np.ndarray:
        """Window number (0-based) of each time stamp, -1 if in none."""
        time = np.asarray(time, dtype=float)
        idx = np.full(time.shape, -1)
        for i, t in enumerate(self.tmids):
            idx[(time >= t - self.width / 2) & (time <= t + self.width / 2)] = i
        return idx

    def shift(
        self, values: Mapping[str, jax.Array | float], time: np.ndarray
    ) -> tuple[jax.Array, np.ndarray]:
        """(per-point TTV offset, per-point in-window mask) for this companion."""
        idx = self.index(time)
        offsets = jnp.stack(
            [
                jnp.asarray(values[f"{self.companion}_ttv_transit_{i + 1}"], dtype=float)
                for i in range(len(self.tmids))
            ]
        )
        inside = idx >= 0
        return jnp.where(inside, offsets[np.maximum(idx, 0)], 0.0), inside


def _ttv_rows(params: ParamTable, c: str) -> list[int]:
    numbers = []
    for p in params:
        m = TTV_NAME.fullmatch(p.name)
        if m and m["c"] == c:
            numbers.append(int(m["n"]))
    return sorted(numbers)


def ttv_windows(
    settings: Settings, params: ParamTable, data: Mapping[str, Dataset]
) -> dict[str, TtvWindows]:
    """Windows per photometric companion; checks params.csv has one row each."""
    if not settings.fit_ttvs:
        return {}
    width = settings.fast_fit_width
    time = np.concatenate([data[i].time for i in settings.inst_phot])
    windows = {}
    for c in settings.companions_phot:
        tmids = observed_transits(
            time, params[f"{c}_epoch"].value, params[f"{c}_period"].value, width
        )
        rows = _ttv_rows(params, c)
        if rows != list(range(1, len(tmids) + 1)):
            raise TtvError(
                f"fit_ttvs=True: the data cover {len(tmids)} transits of {c} "
                f"(fast_fit_width={width}), so params.csv needs rows "
                f"{c}_ttv_transit_1..{len(tmids)}; it has {len(rows)}. Regenerate "
                "them with the current fast_fit_width, epoch and period."
            )
        windows[c] = TtvWindows(c, tmids, width)
    return windows
