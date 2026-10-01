"""Search, download and save light curves with lightkurve for ``jaxoplanet init``.

A "segment" is a TESS sector, K2 campaign or Kepler quarter; labels stay strings
so K2's split campaigns (``11a``/``11b``) survive. Light curves are normalized
per segment and stitched, never flattened (that could remove transits).
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jaxoplanet2.prepare.errors import PrepareError
from jaxoplanet2.prepare.files import Dilution
from jaxoplanet2.prepare.h5 import H5LightCurve

log = logging.getLogger("jaxoplanet2.prepare")

BJD_OFFSET = {"tess": 2457000.0, "k2": 2454833.0, "kepler": 2454833.0}
NONINTERACTIVE_ENVS = ("JAXOPLANET_NONINTERACTIVE", "ALLESFITTER_NONINTERACTIVE")
SEGMENT_WORDS = {"tess": "sector", "k2": "campaign", "kepler": "quarter"}
PLACEHOLDER_FLUX_ERR = 1.0
EXPTIME_ATOL = 0.5  # s


@dataclass(frozen=True)
class Table:
    """A light curve as it goes into ``<inst>.csv``: full BJD, sorted, finite."""

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray

    def __len__(self) -> int:
        return len(self.time)


@dataclass(frozen=True)
class Request:
    """One instrument to download."""

    inst: str
    pipeline: str
    exptime: float | None  # s; None: whatever the pipeline has
    lc_type: str  # pdcsap or sap (QLP always uses sap)
    quality_bitmask: str
    sigma: float | None
    mission: str
    segment_flag: str  # default, first, last, all_sector, multi_sector
    segments: tuple[str, ...] = ()

    @property
    def flux_column(self) -> str:
        return "sap_flux" if self.pipeline.lower() == "qlp" else f"{self.lc_type}_flux"


@dataclass(frozen=True)
class Downloaded:
    table: Table
    exptime: float
    dilution: Dilution | None
    files: tuple[Path, ...]


def segment_word(mission: str, plural: bool = False) -> str:
    return SEGMENT_WORDS.get(str(mission).lower(), "sector") + ("s" if plural else "")


def parse_segment_label(mission_str) -> str:
    """'TESS Sector 82' -> '82'; 'K2 Campaign 11a' -> '11a'."""
    return str(mission_str).split()[-1]


def segments_match(a, b) -> bool:
    """Numeric labels compare by value ('01' == 1); others as strings."""
    sa, sb = str(a).strip(), str(b).strip()
    if sa.isdigit() and sb.isdigit():
        return int(sa) == int(sb)
    return sa == sb


def natural_segment_key(s) -> tuple[float, str]:
    s = str(s)
    n = len(s) - len(s.lstrip("0123456789"))
    return (int(s[:n]) if n else float("inf"), s[n:])


def segment_flag(segments: list[str] | None) -> str:
    if segments is None:
        return "default"
    return {("all",): "all_sector", ("-1",): "last", ("0",): "first"}.get(
        tuple(segments), "multi_sector"
    )


def _plain(arr) -> np.ndarray:
    """Strip astropy units and masks down to a float ndarray (masked -> NaN)."""
    arr = getattr(arr, "value", arr)
    arr = arr.filled(np.nan) if hasattr(arr, "filled") else arr
    return np.asarray(arr, dtype=float)


def _segment_of(lc) -> str:
    for attr in ("sector", "campaign", "quarter"):
        value = getattr(lc, attr, None)
        if value:
            return str(value)
    return "?"


def safe_stitch(collection):
    """Normalize each segment by its finite median and stitch.

    Avoids lightkurve's ``normalize`` (astropy's masked nanmedian fails on
    all-NaN segments) and drops segments that cannot be normalized. Returns
    ``None`` when nothing usable is left.
    """
    import lightkurve as lk

    if not isinstance(collection, lk.LightCurveCollection):
        return collection
    good = []
    for one in collection:
        seg = _segment_of(one)
        flux = _plain(one.flux)
        finite = flux[np.isfinite(flux)]
        med = float(np.median(finite)) if finite.size >= 2 else np.nan
        if not np.isfinite(med) or med == 0.0:
            log.warning(f"Dropping segment {seg} (n_finite={finite.size}, median={med})")
            continue
        norm = one.copy()
        norm.flux = one.flux / med
        if getattr(one, "flux_err", None) is not None:
            norm.flux_err = one.flux_err / med
        good.append(norm)
    if not good:
        log.error("All segments dropped: nothing to stitch.")
        return None
    return lk.LightCurveCollection(good).stitch(corrector_func=lambda x: x)


def _noninteractive() -> bool:
    return any(os.environ.get(env) for env in NONINTERACTIVE_ENVS)


def choose_exptime(search_result, unique_exptimes, context: str, expected_n=None):
    """Resolve several exposure times (e.g. 20 s and 120 s SPOC) in one search.

    Fails with a hint naming ``-e``, unless ``JAXOPLANET_NONINTERACTIVE`` is set:
    then the shortest cadence (preferring one with ``expected_n`` products) is
    picked. Returns ``(narrowed search result, exposure time)``.
    """
    choices = sorted(float(e) for e in unique_exptimes)
    listed = ", ".join(f"{c:g}" for c in choices)
    if not _noninteractive():
        raise PrepareError(
            f"Multiple exposure times are available for {context} ({listed} s). "
            f"Re-run with one, e.g. -e {choices[0]:g}, or set "
            f"{NONINTERACTIVE_ENVS[0]}=1 to auto-select the shortest cadence."
        )
    exptimes = search_result.table.to_pandas().exptime.to_numpy(dtype=float)
    chosen = choices[0]
    if expected_n is not None:
        for c in choices:
            if int(np.isclose(exptimes, c, atol=EXPTIME_ATOL).sum()) == expected_n:
                chosen = c
                break
    log.warning(
        f"Multiple exposure times for {context} ({listed} s); auto-selected -e "
        f"{chosen:g} ({NONINTERACTIVE_ENVS[0]} set). Pass -e/--exptime to override."
    )
    return search_result[list(np.isclose(exptimes, chosen, atol=EXPTIME_ATOL))], chosen


def to_table(lc, mission: str) -> Table:
    """Full-BJD time, flux and error of finite points, sorted in time.

    An all-NaN ``flux_err`` (QLP, TARS) becomes 1: every point weighs the same
    and the fitted ``ln_err_flux_<inst>`` sets the scale.
    """
    time = _plain(lc.time.value) + BJD_OFFSET[mission]
    flux = _plain(lc.flux)
    flux_err = _plain(lc.flux_err)
    if not np.any(np.isfinite(flux_err)):
        log.error("flux_err is all NaN; setting it to 1 (ln_err_flux sets the scale).")
        flux_err = np.full_like(flux, PLACEHOLDER_FLUX_ERR)
    keep = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    order = np.argsort(time[keep], kind="stable")
    table = Table(time[keep][order], flux[keep][order], flux_err[keep][order])
    if len(table) == 0:
        raise PrepareError("the light curve has no finite points")
    return table


def write_table(table: Table, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([table.time, table.flux, table.flux_err])
    np.savetxt(path, rows, delimiter=",", fmt="%.10f")
    log.info(f"Saved: {path} ({len(table):,} points)")
    return path


def lightcurve_from_h5(h5lc: H5LightCurve):
    import lightkurve as lk
    from astropy.time import Time

    time = Time(h5lc.time, format="btjd", scale="tdb")
    return lk.LightCurve(time=time, flux=h5lc.flux, flux_err=h5lc.flux_err)


def h5_matches_segment(h5lc: H5LightCurve | None, segment, pipeline: str) -> bool:
    """Reuse the h5 light curve only for its exact sector and pipeline.

    An h5 predating quicklook's pipeline record is trusted on the sector alone.
    """
    if h5lc is None or h5lc.sector is None or not segments_match(h5lc.sector, segment):
        return False
    if h5lc.pipeline is None:
        log.warning(f"h5 light curve records no pipeline; assuming {pipeline}.")
        return True
    return h5lc.pipeline == pipeline.lower()
