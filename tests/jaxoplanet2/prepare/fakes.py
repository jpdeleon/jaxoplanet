"""Offline stand-ins for MAST (lightkurve searches) and the catalogs."""

import numpy as np
import pandas as pd

PERIOD = 3.0
EPOCH_BTJD = 1325.5  # a transit inside sector 1
DEPTH = 0.01
DURATION_H = 2.4
SECTOR_START = {1: 1325.0, 2: 1353.0, 3: 1381.0}
SECTOR_LENGTH = 20.0  # d


def transit_flux(time_btjd, epoch=EPOCH_BTJD, period=PERIOD):
    phase = (time_btjd - epoch + 0.5 * period) % period - 0.5 * period
    return np.where(np.abs(phase) < DURATION_H / 48, 1 - DEPTH, 1.0)


def make_lc(sector: int, exptime: float, flux_column: str, pipeline: str, seed=0):
    import astropy.units as u
    import lightkurve as lk
    from astropy.time import Time

    rng = np.random.default_rng(seed + sector)
    cadence = exptime / 86400
    time = np.arange(SECTOR_START[sector], SECTOR_START[sector] + SECTOR_LENGTH, cadence)
    scale = 1000.0 if flux_column == "pdcsap_flux" else 1050.0
    flux = scale * (transit_flux(time) + rng.normal(0, 1e-3, time.size))
    err = np.full_like(flux, np.nan if pipeline == "qlp" else scale * 1e-3)
    meta = {"SECTOR": sector, "FLUX_ORIGIN": flux_column}
    if pipeline == "spoc":
        meta |= {"CROWDSAP": 0.98 - 0.01 * sector, "FLFRCSAP": 0.9}
    return lk.TessLightCurve(
        time=Time(time, format="btjd", scale="tdb"),
        flux=flux * u.electron / u.s,
        flux_err=err * u.electron / u.s,
        meta=meta,
    )


class FakeTable:
    def __init__(self, rows):
        self.rows = rows

    def to_pandas(self):
        return pd.DataFrame(self.rows)


class FakeSearchResult:
    """The slice of ``lightkurve.SearchResult`` jaxoplanet init uses."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.downloads = []

    mission = property(
        lambda self: [f"TESS Sector {r['sector']:02d}" for r in self.rows]
    )
    author = property(lambda self: [r["author"] for r in self.rows])
    table = property(lambda self: FakeTable(self.rows))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, mask):
        return FakeSearchResult(
            r for r, keep in zip(self.rows, mask, strict=True) if keep
        )

    def download_all(self, flux_column, quality_bitmask):
        import lightkurve as lk

        return lk.LightCurveCollection(
            [
                make_lc(r["sector"], r["exptime"], flux_column, r["author"].lower())
                for r in self.rows
            ]
        )


def fake_search(products):
    """``lightkurve.search_lightcurve`` over (sector, author, exptime) products."""
    rows = [{"sector": s, "author": a, "exptime": float(e)} for s, a, e in products]

    def search(query, author=None, exptime=None, mission=None):
        keep = [
            r
            for r in rows
            if (author is None or r["author"].lower() == author.lower())
            and (exptime is None or abs(r["exptime"] - exptime) < 0.5)
        ]
        return FakeSearchResult(keep)

    return search


def toi_table(toi=6715, tic=123456789, planets=1):
    rows = []
    for i in range(planets):
        rows.append(
            {
                "TOI": f"{toi}.0{i + 1}",
                "TIC ID": tic,
                "RA": "05:00:00.0",
                "Dec": "-30:00:00.0",
                "TFOPWG Disposition": "PC",
                "Period (days)": PERIOD * (i + 1),
                "Period (days) err": 1e-4,
                "Epoch (BJD)": EPOCH_BTJD + 2457000,
                "Epoch (BJD) err": 1e-3,
                "Duration (hours)": DURATION_H,
                "Duration (hours) err": 0.1,
                "Depth (ppm)": DEPTH * 1e6,
                "Depth (ppm) err": 100.0,
            }
        )
    return pd.DataFrame(rows)
