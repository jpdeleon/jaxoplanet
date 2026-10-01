"""Read transit parameters and the light curve from a quicklook TLS ``.h5`` file.

`quicklook <https://github.com/jpdeleon/quicklook>`_ saves its
``transitleastsquares`` results (``TIC123_s1_tls.h5``, plus ``*_tls_p2.h5`` ...
for further candidates of its iterative search). ``jaxoplanet init --h5`` seeds
a companion from them, and reuses the embedded light curve when it is exactly
the sector and pipeline being prepared.

Two layouts are read, as quicklook's own loader does: scalars JSON-encoded in a
``_scalar_data`` attribute (current), or stored as file attributes with tuples
as deepdish-style groups (legacy).
"""

import json
from dataclasses import dataclass

import numpy as np

MISSION_BJD_OFFSET = {"tess": 2_457_000.0, "k2": 2_454_833.0, "kepler": 2_454_833.0}
_LEGACY_ATTR_SKIP = frozenset(
    {"CLASS", "DEEPDISH_IO_VERSION", "PYTABLES_FORMAT_VERSION", "TITLE", "VERSION"}
)


@dataclass(frozen=True)
class H5LightCurve:
    time: np.ndarray  # BTJD (mission time), not BJD
    flux: np.ndarray
    flux_err: np.ndarray
    pipeline: str | None
    sector: int | None
    exptime: float | None  # s


def _scalars(h5file) -> dict:
    if "_scalar_data" in h5file.attrs:
        return json.loads(h5file.attrs["_scalar_data"])
    return {k: v for k, v in h5file.attrs.items() if k not in _LEGACY_ATTR_SKIP}


def _legacy_tuple(group) -> tuple | None:
    items = []
    while f"i{len(items)}" in group.attrs:
        items.append(group.attrs[f"i{len(items)}"])
    return tuple(items) if items else None


def _value(h5file, scalars: dict, key: str):
    import h5py

    if key in scalars:
        return scalars[key]
    if key not in h5file:
        return None
    obj = h5file[key]
    return _legacy_tuple(obj) if isinstance(obj, h5py.Group) else obj[()]


def _fractional_depth(raw) -> float | None:
    """TLS ``depth`` (in-transit relative flux, near 1) -> dimming (near 0)."""
    if raw is None:
        return None
    value = float(raw)
    if value > 1.0:
        value /= 1e3
    return value if value < 0.5 else 1.0 - value


def _optional(x, scale: float = 1.0, offset: float = 0.0) -> float | None:
    return None if x is None else float(x) * scale + offset


def read_transit_params(path: str, mission: str = "tess") -> dict[str, float | None]:
    """--period/--epoch/--duration/--depth (and *_err) values, in d, BJD, h, ppm.

    TLS reports no epoch or duration error; those stay ``None``.
    """
    import h5py

    offset = MISSION_BJD_OFFSET.get(mission.lower(), MISSION_BJD_OFFSET["tess"])
    with h5py.File(path, "r") as f:
        s = _scalars(f)
        get = {
            k: _value(f, s, k)
            for k in (
                "period",
                "period_uncertainty",
                "T0",
                "duration",
                "depth",
                "depth_mean",
            )
        }
    depth = _fractional_depth(get["depth"])
    depth_mean = get["depth_mean"]
    return {
        "period": _optional(get["period"]),
        "period_err": _optional(get["period_uncertainty"]),
        "epoch": _optional(get["T0"], offset=offset),
        "epoch_err": None,
        "duration": _optional(get["duration"], scale=24.0),
        "duration_err": None,
        "depth": None if depth is None else depth * 1e6,
        "depth_err": float(depth_mean[1]) * 1e6
        if depth_mean is not None and len(depth_mean) == 2
        else None,
    }


def read_lightcurve(path: str) -> H5LightCurve | None:
    """The raw, normalized light curve TLS ran on; ``None`` if not embedded."""
    import h5py

    with h5py.File(path, "r") as f:
        s = _scalars(f)
        get = {
            k: _value(f, s, k)
            for k in ("time_raw", "flux_raw", "err_raw", "pipeline", "sector", "exptime")
        }
    if any(get[k] is None for k in ("time_raw", "flux_raw", "err_raw")):
        return None
    return H5LightCurve(
        time=np.asarray(get["time_raw"], dtype=float),
        flux=np.asarray(get["flux_raw"], dtype=float),
        flux_err=np.asarray(get["err_raw"], dtype=float),
        pipeline=None if get["pipeline"] is None else str(get["pipeline"]).lower(),
        sector=None if get["sector"] is None else int(get["sector"]),
        exptime=_optional(get["exptime"]),
    )
