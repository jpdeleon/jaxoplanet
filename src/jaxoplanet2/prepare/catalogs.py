"""Catalog lookups for ``jaxoplanet init``: ExoFOP TOI/CTOI, NExSci, TIC, tess-point.

Tables are cached under ``$JAXOPLANET2_CACHE_DIR`` (default
``~/.cache/jaxoplanet2``) and refreshed with ``-u/--update_db``. Heavy
dependencies (pandas, astropy, astroquery, tess-point) are imported lazily so
``import jaxoplanet2`` stays light.
"""

import json
import logging
import math
import os
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from jaxoplanet2.prepare.errors import PrepareError
from jaxoplanet2.prepare.priors import Ephemeris, Star

log = logging.getLogger("jaxoplanet2.prepare")

EXOFOP = "https://exofop.ipac.caltech.edu/tess"
TOI_URL = f"{EXOFOP}/download_toi.php?sort=toi&output=csv"
CTOI_URL = f"{EXOFOP}/download_ctoi.php?sort=ctoi&output=csv"
ALIAS_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Lookup/nph-aliaslookup.py?objname="
CACHE_ENV = "JAXOPLANET2_CACHE_DIR"
TIMEOUT = 60  # s
NEXSCI_EPOCH_ERR = 0.1  # d; pscomppars has no usable epoch error
NEXSCI_DEPTH_ERR = 1_000.0  # ppm
R_EARTH_IN_R_SUN = 0.009168
# (key, catalog column, prompt, must be positive) of a raw TIC ephemeris
EPHEMERIS_FIELDS = (
    ("period", "Period (days)", "Porb (d): ", True),
    ("period_err", "Period (days) err", "Porb err (d): ", False),
    ("epoch", "Epoch (BJD)", "Epoch (BJD): ", True),
    ("epoch_err", "Epoch (BJD) err", "Epoch err (BJD): ", False),
    ("duration", "Duration (hours)", "Tdur (h): ", True),
    ("duration_err", "Duration (hours) err", "Tdur err (h): ", False),
    ("depth", "Depth (ppm)", "Depth (ppm): ", True),
    ("depth_err", "Depth (ppm) err", "Depth err (ppm): ", False),
)
EPHEMERIS_KEYS = tuple(key for key, *_ in EPHEMERIS_FIELDS)


class CatalogError(PrepareError):
    """A target is not in the catalog, or the catalog cannot be read."""


@dataclass(frozen=True)
class Target:
    name: str  # e.g. TOI-1097, TIC-123, HIP67522; also the output directory
    source: str  # tfop, ctoi, custom, nexsci
    planets: tuple[Ephemeris, ...]
    tic_id: int | None = None
    ra: float | None = None  # deg
    dec: float | None = None
    # NExSci only: host properties and planet radii (R_earth) per companion
    nexsci_host: Mapping[str, float] = field(default_factory=dict)
    planet_radii: tuple[tuple[float, float], ...] = ()


def cache_dir() -> Path:
    path = Path(os.environ.get(CACHE_ENV, Path.home() / ".cache" / "jaxoplanet2"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _float(x) -> float:
    try:
        value = float(np.ma.filled(x, np.nan))
    except (TypeError, ValueError):
        return math.nan
    return value


def _cached_csv(name: str, download: Callable, update: bool):
    import pandas as pd

    path = cache_dir() / name
    if path.exists() and not update:
        log.info(f"Loaded: {path} (use --update_db to refresh)")
        return pd.read_csv(path)
    log.info(f"Downloading {name} ...")
    df = download()
    df.to_csv(path, index=False)
    log.info(f"Saved: {path}")
    return df


def load_tois(update: bool = False):
    """ExoFOP TOI table without TFOPWG false positives."""
    import pandas as pd

    df = _cached_csv("TOIs.csv", lambda: pd.read_csv(TOI_URL), update)
    return df[df["TFOPWG Disposition"] != "FP"].sort_values("TOI")


def load_ctois(update: bool = False):
    """ExoFOP CTOI table (user FPs removed), columns renamed like the TOI table."""
    import pandas as pd

    df = _cached_csv("CTOIs.csv", lambda: pd.read_csv(CTOI_URL), update)
    df = df.drop_duplicates()
    df = df[df["User Disposition"] != "FP"]
    renames = (
        ("Error", "err"),
        (" ppm", " (ppm)"),
        ("Transit Epoch", "Epoch"),
        ("hrs", "hours"),
    )
    columns = list(df.columns)
    for old, new in renames:
        columns = [c.replace(old, new) for c in columns]
    return df.set_axis(columns, axis=1).sort_values("CTOI")


def load_nexsci(update: bool = False):
    """NExSci ``pscomppars`` (transiting planets only)."""

    def download():
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

        table = NasaExoplanetArchive.query_criteria(
            table="pscomppars", where="discoverymethod like 'Transit'"
        )
        return table.to_pandas()

    return _cached_csv("nexsci_pscomppars.csv", download, update)


def _read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return json.loads(response.read())


def tfop_info(name: str) -> dict:
    """ExoFOP-TESS target summary (coordinates, TIC ID, ...)."""
    query = urllib.parse.quote(name.replace(" ", ""))
    try:
        return _read_json(f"{EXOFOP}/target.php?id={query}&json")
    except (OSError, ValueError) as e:
        raise CatalogError(f"ExoFOP-TESS has no target {name!r}: {e}") from e


def name_aliases(name: str) -> tuple[str, ...]:
    """Every NExSci alias of a system (TOI, TIC, HD, EPIC, ...)."""
    try:
        data = _read_json(ALIAS_URL + urllib.parse.quote(name))
        aliases = data["system"]["system_info"]["alias_set"]["aliases"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise CatalogError(f"NExSci alias lookup of {name!r} failed: {e}") from e
    return tuple(str(a) for a in aliases)


def alias_with_prefix(name: str, prefix: str) -> str:
    for alias in name_aliases(name):
        if alias.lower().startswith(prefix.lower()):
            return alias
    raise CatalogError(f"no {prefix} alias for {name}")


def ephemeris_from_row(row: Mapping) -> Ephemeris:
    return Ephemeris(
        *(_float(row.get(column, math.nan)) for _, column, _, _ in EPHEMERIS_FIELDS)
    )


def resolve_ephemeris(
    values: Mapping[str, float | None], prompt: Callable[[str], str] = input
) -> Ephemeris:
    """A raw target's ephemeris: given values, prompting only for missing ones.

    Every given value is validated before the first prompt, so a bad command
    line fails at once.
    """
    checked = {}
    for key, _column, _label, positive in EPHEMERIS_FIELDS:
        value = values.get(key)
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            relation = "positive" if positive else "non-negative"
            raise CatalogError(
                f"--{key.replace('_', '-')} must be a finite {relation} number"
            )
        checked[key] = value
    resolved = [
        checked[key] if key in checked else float(prompt(label))
        for key, _column, label, _ in EPHEMERIS_FIELDS
    ]
    return Ephemeris(*resolved)


def merge_ephemeris_values(
    cli: Mapping[str, float | None], fallback: Mapping[str, float | None]
) -> dict[str, float | None]:
    """``cli`` with its missing fields filled from ``fallback`` (e.g. an h5 file)."""
    return {
        key: cli.get(key) if cli.get(key) is not None else fallback.get(key)
        for key in EPHEMERIS_KEYS
    }


def _normalize(s: str) -> str:
    return str(s).lower().replace(" ", "").replace("-", "").replace("_", "")


def _coords_hms(df) -> tuple[float, float]:
    from astropy.coordinates import SkyCoord

    ra, dec = df[["RA", "Dec"]].values[0]
    coord = SkyCoord(ra, dec, unit=("hourangle", "deg"))
    return float(coord.ra.deg), float(coord.dec.deg)


def _from_toi(toi: int, update: bool) -> Target:
    df = load_tois(update)
    log.info("Using parameters from the TOI database (use --update_db to update).")
    log.info(f"To use published parameters in NExSci, use -name TOI-{toi}")
    rows = df[df["TOI"].astype(str).str.split(".").str[0] == str(toi)]
    if rows.empty:
        raise CatalogError(f"TOI {toi} is not in the TOI database")
    rows = rows.reset_index(drop=True)
    ra, dec = _coords_hms(rows)
    return Target(
        name=f"TOI-{str(toi).zfill(4)}",
        source="tfop",
        planets=tuple(ephemeris_from_row(r) for _, r in rows.iterrows()),
        tic_id=int(rows["TIC ID"].iloc[0]),
        ra=ra,
        dec=dec,
    )


def _from_ctoi(ctoi: int, update: bool) -> Target:
    df = load_ctois(update)
    log.info("Using parameters from the CTOI database (use --update_db to update).")
    rows = df[df["TIC ID"] == int(ctoi)].reset_index(drop=True)
    if rows.empty:
        raise CatalogError(f"TIC {ctoi} is not in the CTOI database")
    ra, dec = _coords_hms(rows)
    return Target(
        name=f"CTOI-{ctoi}",
        source="ctoi",
        planets=tuple(ephemeris_from_row(r) for _, r in rows.iterrows()),
        tic_id=int(ctoi),
        ra=ra,
        dec=dec,
    )


def _quadrature(row: Mapping, column: str) -> float:
    return math.hypot(_float(row.get(f"{column}err1")), _float(row.get(f"{column}err2")))


def _nexsci_ephemeris(row: Mapping) -> Ephemeris:
    return Ephemeris(
        period=_float(row["pl_orbper"]),
        period_err=_quadrature(row, "pl_orbper"),
        epoch=_float(row["pl_tranmid"]),
        epoch_err=NEXSCI_EPOCH_ERR,
        duration=_float(row["pl_trandur"]),
        duration_err=_quadrature(row, "pl_trandur"),
        depth=_float(row["pl_trandep"]) / 100 * 1e6,  # % -> ppm
        depth_err=NEXSCI_DEPTH_ERR,
    )


def _from_nexsci(name: str, update: bool) -> Target:
    df = load_nexsci(update)
    log.info("Using parameters from the NExSci database (use --update_db to update).")
    hosts = df["hostname"].astype(str).map(_normalize)
    match = hosts == _normalize(name)
    if not match.any():
        log.info(f"Hostname '{name}' not found directly; resolving NExSci aliases...")
        try:
            for alias in name_aliases(name):
                if (hosts == _normalize(alias)).any():
                    log.info(f"Matched NExSci hostname via alias: {alias}")
                    match = hosts == _normalize(alias)
                    break
        except CatalogError as e:
            log.error(str(e))
    if not match.any():
        raise CatalogError(f"hostname {name} is not in the NExSci database")
    rows = df[match].reset_index(drop=True)
    first = rows.iloc[0]
    host_keys = ("st_teff", "st_logg", "st_rad", "st_mass")
    host = {k: _float(first.get(k)) for k in host_keys}
    host.update({f"{k}_err": _quadrature(first, k) for k in host_keys})
    radii = tuple(
        (_float(r.get("pl_rade")), _quadrature(r, "pl_rade")) for _, r in rows.iterrows()
    )
    return Target(
        name=name.strip().replace(" ", ""),
        source="nexsci",
        planets=tuple(_nexsci_ephemeris(r) for _, r in rows.iterrows()),
        nexsci_host=host,
        planet_radii=radii,
    )


def find_target(
    *,
    toi: int | None = None,
    ctoi: int | None = None,
    tic: int | None = None,
    name: str | None = None,
    update: bool = False,
    tic_ephemeris: Ephemeris | None = None,
) -> Target:
    """Look up exactly one of ``toi``, ``ctoi``, ``tic`` or ``name``.

    A raw ``tic`` has no catalog ephemeris; it comes from ``tic_ephemeris``.
    """
    if toi:
        return _from_toi(toi, update)
    if ctoi:
        return _from_ctoi(ctoi, update)
    if tic:
        if tic_ephemeris is None:
            raise CatalogError("a raw TIC target needs its period/epoch/duration/depth")
        log.info("Using the given ephemeris and TIC stellar parameters.")
        return Target(
            name=f"TIC-{tic}", source="custom", planets=(tic_ephemeris,), tic_id=tic
        )
    if name:
        return _from_nexsci(name, update)
    raise CatalogError("one of -toi, -ctoi, -tic or -name is required")


def with_coordinates(target: Target) -> Target:
    """Fill in the TIC ID and coordinates of a TIC or NExSci target from ExoFOP."""
    if target.ra is not None and target.tic_id is not None:
        return target
    info = tfop_info(target.name if target.source != "custom" else f"TIC{target.tic_id}")
    try:
        return replace(
            target,
            ra=float(info["coordinates"]["ra"]),
            dec=float(info["coordinates"]["dec"]),
            tic_id=int(info["basic_info"]["tic_id"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise CatalogError(f"ExoFOP-TESS has no coordinates for {target.name}") from e


def tess_sectors(tic_id: int, ra: float, dec: float) -> np.ndarray:
    """Sectors in which TESS observed (or will observe) the target."""
    from tess_stars2px import tess_stars2px_function_entry

    result = tess_stars2px_function_entry(tic_id, ra, dec)
    return np.asarray(result[3], dtype=int)


TIC_COLUMNS = (
    "Teff",
    "e_Teff",
    "logg",
    "e_logg",
    "MH",
    "e_MH",
    "rad",
    "e_rad",
    "mass",
    "e_mass",
)


def tic_star(tic_id: int) -> Star:
    """Stellar parameters from the TESS Input Catalog (NaN where missing)."""
    from astroquery.mast import Catalogs

    table = Catalogs.query_criteria(catalog="Tic", ID=int(tic_id))
    if len(table) == 0:
        raise CatalogError(f"TIC {tic_id} is not in the TESS Input Catalog")
    row = table[0]
    return Star(*(_float(row[c]) for c in TIC_COLUMNS))


def nexsci_star(host: Mapping[str, float]) -> Star:
    """Stellar parameters of a NExSci host; [Fe/H] is left unknown (NaN)."""
    return Star(
        teff=host["st_teff"],
        teff_err=host["st_teff_err"],
        logg=host["st_logg"],
        logg_err=host["st_logg_err"],
        feh=math.nan,
        feh_err=math.nan,
        radius=host["st_rad"],
        radius_err=host["st_rad_err"],
        mass=host["st_mass"],
        mass_err=host["st_mass_err"],
    )


def radius_ratio_from_radii(
    planet_radius: tuple[float, float], star_radius: float
) -> tuple[float, float]:
    """(k, sigma_k) from a planet radius (R_earth) and the stellar radius (R_sun)."""
    rp, rp_err = planet_radius
    return rp * R_EARTH_IN_R_SUN / star_radius, rp_err * R_EARTH_IN_R_SUN / star_radius
