"""Download one instrument's light curve, plot it, and save ``<inst>.csv``.

Mirrors prepare_allesfit's per-instrument loop: pick the requested segments
(first/last/all/listed) of one pipeline, stitch them, optionally sigma-clip,
plot (SPOC: PDCSAP vs SAP), record SPOC crowding as a dilution, and write the
data file the fit reads.
"""

import logging
from dataclasses import dataclass

import numpy as np

from jaxoplanet2.prepare.errors import PrepareError
from jaxoplanet2.prepare.h5 import H5LightCurve
from jaxoplanet2.prepare.lightcurves import (
    Request,
    choose_exptime,
    h5_matches_segment,
    lightcurve_from_h5,
    natural_segment_key,
    parse_segment_label,
    safe_stitch,
    segment_word,
    segments_match,
)

log = logging.getLogger("jaxoplanet2.prepare")


@dataclass(frozen=True)
class Selection:
    rows: object | None  # lightkurve SearchResult to download; None: use lc
    segments: tuple[str, ...]
    exptime: float
    lc: object | None = None  # the h5 light curve, when reused

    @property
    def label(self) -> str:
        """Segment part of file names, e.g. "1s2s3"."""
        return "s".join(self.segments)


def _exptimes(result) -> np.ndarray:
    return result.table.to_pandas().exptime.unique()


def search(query_name: str, req: Request):
    """The requested pipeline's search result, after logging what exists."""
    import lightkurve as lk

    everything = lk.search_lightcurve(query_name, mission=req.mission)
    if len(everything) == 0:
        raise PrepareError(f"no light curves found for {query_name}")
    pipelines = {a.lower() for a in everything.author}
    log.info(f"Available pipelines: {sorted(pipelines)}")
    log.info(f"Available exposure times: {_exptimes(everything)}")
    if req.pipeline.lower() not in pipelines:
        raise PrepareError(f"pipeline={req.pipeline} not in {sorted(pipelines)}")
    result = lk.search_lightcurve(
        query_name, author=req.pipeline, exptime=req.exptime, mission=req.mission
    )
    if not result:
        raise PrepareError(f"no light curve for pipeline={req.pipeline}; check inputs")
    return result


def _mask(n: int, keep) -> list[bool]:
    return [bool(keep(i)) for i in range(n)]


def _select_all(result, req: Request, unique: list[str]) -> Selection:
    words = segment_word(req.mission, plural=True)
    log.info(f"Using {req.pipeline.upper()} in {len(unique)} {words}: {unique}")
    exptime = req.exptime
    if len(_exptimes(result)) > 1:
        result, exptime = choose_exptime(
            result, _exptimes(result), f"all {words}", expected_n=len(unique)
        )
    exptime = _exptimes(result)[0] if exptime is None else exptime
    return Selection(result, tuple(unique), float(exptime))


def _from_h5(h5lc: H5LightCurve, segment: str, exptime) -> Selection:
    log.info(f"Reusing the light curve embedded in the h5 file for segment {segment}.")
    lc = lightcurve_from_h5(h5lc).normalize()
    return Selection(None, (segment,), float(h5lc.exptime or exptime), lc)


def _select_listed(result, req: Request, sectors: list[str], h5lc) -> Selection:
    word = segment_word(req.mission)
    wanted = req.segments
    rows = result[
        _mask(len(sectors), lambda i: any(segments_match(sectors[i], s) for s in wanted))
    ]
    if len(rows) == 0:
        available = sorted(set(sectors), key=natural_segment_key)
        raise PrepareError(
            f"{req.pipeline.upper()} has no light curve for {word}={list(wanted)}; "
            f"try {word}={available}"
        )
    exptime = req.exptime
    if len(wanted) > len(rows):
        raise PrepareError(f"not every {word} in {list(wanted)} has exptime={exptime} s")
    if len(wanted) < len(rows):
        rows, exptime = choose_exptime(
            rows, _exptimes(rows), f"the given {word}s", expected_n=len(wanted)
        )
    exptime = _exptimes(rows)[0] if exptime is None else exptime
    if len(wanted) == 1 and h5_matches_segment(h5lc, wanted[0], req.pipeline):
        return _from_h5(h5lc, wanted[0], exptime)
    log.info(f"Using {req.pipeline.upper()} in {word}s {list(wanted)} ({exptime} s).")
    return Selection(rows, tuple(wanted), float(exptime))


def _select_one(result, req: Request, sectors: list[str], h5lc) -> Selection:
    i = 0 if req.segment_flag == "first" else len(sectors) - 1
    segment = sectors[i]
    rows = result[_mask(len(sectors), lambda j: j == i)]
    exptime = _exptimes(rows)[0] if req.exptime is None else req.exptime
    if h5_matches_segment(h5lc, segment, req.pipeline):
        return _from_h5(h5lc, segment, exptime)
    log.info(f"Using {req.pipeline.upper()} in {segment_word(req.mission)} {segment}.")
    return Selection(rows, (segment,), float(exptime))


def select(result, req: Request, h5lc: H5LightCurve | None = None) -> Selection:
    sectors = [parse_segment_label(s) for s in result.mission]
    unique = sorted(set(sectors), key=natural_segment_key)
    if req.segment_flag == "all_sector":
        return _select_all(result, req, unique)
    if req.segment_flag == "multi_sector":
        return _select_listed(result, req, sectors, h5lc)
    return _select_one(result, req, sectors, h5lc)


def _stitched(rows, req: Request, flux_column: str):
    collection = rows.download_all(
        flux_column=flux_column, quality_bitmask=req.quality_bitmask
    )
    if collection is None:
        raise PrepareError(f"downloading the {flux_column} light curves failed")
    lc = safe_stitch(collection)
    if lc is None:
        raise PrepareError("no usable segments downloaded")
    return collection, lc


def _check_header(lc, sel: Selection, req: Request) -> None:
    """A single downloaded segment must be the one requested."""
    if len(sel.segments) != 1:
        return  # a stitched light curve keeps only the first segment's header
    for attr in ("sector", "campaign", "quarter"):
        value = getattr(lc, attr, None)
        if value and not segments_match(value, sel.segments[0]):
            word = segment_word(req.mission)
            raise PrepareError(
                f"{word}={value} in the header, but {sel.label} requested"
            )


def _clip(lc, sigma: float | None):
    if not sigma:
        return lc
    clipped = lc.remove_outliers(sigma=sigma)
    if len(clipped) < len(lc):
        log.info(f"Removed {len(lc) - len(clipped)} outliers using sigma={sigma}.")
    return clipped


@dataclass(frozen=True)
class Fluxes:
    lc: object  # the light curve to fit
    pdcsap: object | None = None  # SPOC only, for the PDCSAP/SAP comparison
    sap: object | None = None
    header_source: object | None = None  # carries CROWDSAP


def fetch(sel: Selection, req: Request) -> Fluxes:
    if sel.rows is None:
        return Fluxes(_clip(sel.lc, req.sigma))
    collection, lc = _stitched(sel.rows, req, req.flux_column)
    log.info("The light curves were not flattened, to keep the transits.")
    _check_header(lc, sel, req)
    if req.pipeline.lower() != "spoc":
        return Fluxes(_clip(lc, req.sigma))
    if req.flux_column == "pdcsap_flux":
        pdc_collection, pdcsap = collection, lc
    else:
        pdc_collection, pdcsap = _stitched(sel.rows, req, "pdcsap_flux")
    _, sap = _stitched(sel.rows, req, "sap_flux")
    return Fluxes(
        _clip(lc, req.sigma),
        _clip(pdcsap, req.sigma),
        _clip(sap, req.sigma),
        pdc_collection,
    )
