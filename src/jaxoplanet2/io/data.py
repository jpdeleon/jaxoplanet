"""Read ``<inst>.csv`` data files and apply allesfitter's ``fast_fit`` windowing."""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np

from jaxoplanet2.io.settings import Settings

Kind = Literal["flux", "rv"]


class DataError(ValueError):
    """A data file is malformed or unusable."""


@dataclass(frozen=True)
class Dataset:
    """One instrument's time series; arrays are read-only."""

    inst: str
    kind: Kind
    time: np.ndarray
    y: np.ndarray
    yerr: np.ndarray

    def __post_init__(self) -> None:
        for name in ("time", "y", "yerr"):
            arr = np.array(getattr(self, name), dtype=float)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    def __len__(self) -> int:
        return len(self.time)

    def select(self, mask: np.ndarray) -> "Dataset":
        return Dataset(
            self.inst, self.kind, self.time[mask], self.y[mask], self.yerr[mask]
        )


def load_dataset(fit_dir: str | Path, inst: str, kind: Kind) -> Dataset:
    path = Path(fit_dir) / f"{inst}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no {inst}.csv in {Path(fit_dir)}")
    columns = _read_columns(path)
    time, y, yerr = columns
    _validate(path.name, time, y, yerr)
    return Dataset(inst, kind, time, y, yerr)


def _read_columns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for raw_line in path.read_text().splitlines():
        # genfromtxt(comments="#") semantics, as in allesfitter
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        cells = line.split(",")
        if len(cells) < 3:
            raise DataError(f"{path.name} needs 3 columns: time, value, error")
        try:
            rows.append([float(c) for c in cells[:3]])
        except ValueError:
            if not rows:  # a plain (pandas-style) header row
                continue
            raise DataError(f"{path.name}: cannot parse line '{line}'") from None
    if not rows:
        raise DataError(f"{path.name} contains no data")
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _validate(name: str, time: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> None:
    if not np.all(np.isfinite(time * y * yerr)):
        raise DataError(f"{name} contains NaN/inf values; remove those rows")
    if np.any(yerr <= 0):
        raise DataError(f"{name}: all uncertainties must be positive")
    steps = np.diff(time)
    if np.any(steps < 0):
        raise DataError(f"{name}: time is not sorted; sort the file by time")
    if np.any(steps == 0):
        warnings.warn(f"{name} has repeated time stamps", UserWarning, stacklevel=3)


def fast_fit_mask(
    time: np.ndarray, ephemerides: Sequence[tuple[float, float]], width: float
) -> np.ndarray:
    """Points within ``width / 2`` of any transit of any (epoch, period)."""
    mask = np.zeros(len(time), dtype=bool)
    for epoch, period in ephemerides:
        n = np.round((time - epoch) / period)
        mask |= np.abs(time - (epoch + n * period)) <= width / 2
    return mask


def load_datasets(
    fit_dir: str | Path, settings: Settings, values: Mapping[str, float]
) -> Mapping[str, Dataset]:
    """Load every instrument, applying ``fast_fit`` to the photometry.

    ``values`` supplies the ``<companion>_epoch`` / ``<companion>_period`` guesses
    that define the transit windows.
    """
    data = {}
    for inst in settings.inst_all:
        kind: Kind = "flux" if inst in settings.inst_phot else "rv"
        dataset = load_dataset(fit_dir, inst, kind)
        if kind == "flux" and settings.fast_fit and settings.companions_phot:
            ephemerides = [
                (values[f"{c}_time_transit"], values[f"{c}_period"])
                for c in settings.companions_phot
            ]
            mask = fast_fit_mask(dataset.time, ephemerides, settings.fast_fit_width)
            if not mask.any():
                raise DataError(
                    f"{inst}.csv has no in-transit data within fast_fit_width; "
                    "check the epoch/period guesses"
                )
            dataset = dataset.select(mask)
        data[inst] = dataset
    return MappingProxyType(data)
