"""Per-instrument outputs of ``jaxoplanet init``: plots, crowding, data files."""

import logging
from pathlib import Path

import numpy as np

from jaxoplanet2.prepare.download import Fluxes, Selection, fetch, search, select
from jaxoplanet2.prepare.errors import PrepareError
from jaxoplanet2.prepare.files import Dilution
from jaxoplanet2.prepare.h5 import H5LightCurve
from jaxoplanet2.prepare.lightcurves import (
    Downloaded,
    Request,
    choose_exptime,
    natural_segment_key,
    parse_segment_label,
    safe_stitch,
    segment_word,
    segments_match,
    to_table,
    write_table,
)

log = logging.getLogger("jaxoplanet2.prepare")

CONTAMINATION_HEADER = """\
# SPOC photometric contamination -> jaxoplanet2 dilution
#
# Header keywords (per FITS extension):
#   CROWDSAP : target_flux / total_flux in the optimal aperture, in [0, 1]
#   FLFRCSAP : fraction of target PSF flux captured by the aperture
#
# dil_<inst> is the contaminant fraction of the measured flux:
#
#   contratio = (1 - CROWDSAP) / CROWDSAP     # contaminant / target
#   dilution  = contratio / (1 + contratio)   # contaminant / (contaminant + target)
#             = 1 - CROWDSAP                  # algebraically identical
#
# params.csv carries a commented normal-prior dil_<inst> row centred on the
# median below; uncomment it (and comment the uniform row) to use it.
#"""


def _header_float(meta, key: str) -> float | None:
    try:
        return float(meta[key]) if meta.get(key) is not None else None
    except (TypeError, ValueError):
        return None


def spoc_crowding(lc_or_collection) -> list[tuple[str, float, float | None]]:
    """(segment, CROWDSAP, FLFRCSAP) of every segment whose header has them."""
    import lightkurve as lk

    is_collection = isinstance(lc_or_collection, lk.LightCurveCollection)
    items = list(lc_or_collection) if is_collection else [lc_or_collection]
    rows = []
    for one in items:
        meta = getattr(one, "meta", None) or {}
        crowdsap = _header_float(meta, "CROWDSAP")
        if crowdsap is None:
            continue
        seg = next(
            (
                str(v)
                for a in ("sector", "campaign", "quarter")
                if (v := getattr(one, a, None))
            ),
            "?",
        )
        rows.append((seg, crowdsap, _header_float(meta, "FLFRCSAP")))
    return rows


def write_contamination(rows, path: Path, mission: str) -> Dilution | None:
    """Write the crowding table; returns the median/scatter dilution."""
    if not rows:
        return None
    lines = [
        CONTAMINATION_HEADER,
        f"# {segment_word(mission)}\tCROWDSAP\tFLFRCSAP\tcontratio\tdilution",
    ]
    for seg, crowd, flfrc in rows:
        contratio = (1.0 - crowd) / crowd if crowd > 0 else float("inf")
        flf = "NA" if flfrc is None else f"{flfrc:.6f}"
        lines.append(f"{seg}\t{crowd:.6f}\t{flf}\t{contratio:.6f}\t{1.0 - crowd:.6f}")
    crowds = np.array([r[1] for r in rows])
    dils = 1.0 - crowds
    lines += [
        "",
        f"# Median across {len(rows)} segment(s):",
        f"median_CROWDSAP\t{np.median(crowds):.6f}",
        f"median_dilution\t{np.median(dils):.6f}",
    ]
    path.write_text("\n".join(lines) + "\n")
    log.info(f"Saved: {path}")
    return Dilution(float(np.median(dils)), float(np.std(dils)))


def _title(req: Request, sel: Selection) -> str:
    word = segment_word(req.mission).capitalize()
    return f"{word}={sel.label}\nexptime={int(sel.exptime)}s"


def _plot(fluxes: Fluxes, req: Request, sel: Selection, stem: Path) -> Path:
    import matplotlib.pyplot as plt

    if fluxes.pdcsap is not None and len(fluxes.pdcsap) == len(fluxes.sap):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fluxes.pdcsap.scatter(ax=axs[0], zorder=2, label="PDCSAP", c="C0")
        fluxes.sap.scatter(ax=axs[0], zorder=1, label="SAP", c="C1")
        (fluxes.pdcsap - fluxes.sap).scatter(ax=axs[1], label="difference", c="k")
        axs[0].set_title(_title(req, sel))
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        fluxes.lc.scatter(ax=ax, label=req.pipeline)
        ax.set_title(_title(req, sel))
    path = stem.with_suffix(".png")
    fig.savefig(path)
    plt.close(fig)
    log.info(f"Saved: {path}")
    return path


def _stem(
    outdir: Path, target: str, req: Request, sel: Selection, spoc_pair: bool
) -> Path:
    kind = req.flux_column.split("_")[0] if spoc_pair else req.pipeline
    return outdir / f"{target}_{req.mission}_{kind}_s{sel.label}_exp{int(sel.exptime)}s"


def download_instrument(
    query_name: str,
    req: Request,
    outdir: Path,
    target_name: str,
    h5lc: H5LightCurve | None = None,
) -> Downloaded:
    """Download ``req``, then write ``<inst>.csv``, a plot and (SPOC) crowding."""
    sel = select(search(query_name, req), req, h5lc)
    fluxes = fetch(sel, req)
    spoc_pair = fluxes.pdcsap is not None and len(fluxes.pdcsap) == len(fluxes.sap)
    stem = _stem(outdir, target_name, req, sel, spoc_pair)
    files = [_plot(fluxes, req, sel, stem)]
    dilution = None
    if fluxes.header_source is not None:
        rows = spoc_crowding(fluxes.header_source)
        contam = outdir / f"{req.inst}_spoc_contamination.txt"
        dilution = write_contamination(rows, contam, req.mission)
        if dilution is None:
            log.info("CROWDSAP is not in the SPOC header; no contamination file.")
        else:
            files.append(contam)
    elif req.pipeline.lower() == "spoc":
        log.info("The h5 light curve has no FITS header: no SPOC crowding/dilution.")
    table = to_table(fluxes.lc, req.mission)
    files.append(write_table(table, stem.with_suffix(".csv")))
    files.append(write_table(table, outdir / f"{req.inst}.csv"))
    return Downloaded(table, sel.exptime, dilution, tuple(files))


def _lc_only_segments(sectors: list[str], wanted: tuple[str, ...]) -> list[str]:
    unique = sorted(set(sectors), key=natural_segment_key)
    if wanted == ("all",):
        return unique
    if wanted in (("-1",), ("0",)):
        return [unique[-1 if wanted == ("-1",) else 0]]
    return list(wanted)


def download_lc_only(query_name: str, label: str, req: Request, basedir: Path) -> Path:
    """``--lc-only``: save just ``<label>_<pipeline>_s<segments>_exp<t>s.csv``."""
    import lightkurve as lk

    result = lk.search_lightcurve(
        query_name, author=req.pipeline, exptime=req.exptime, mission=req.mission
    )
    if not result:
        raise PrepareError(f"no light curve found for pipeline={req.pipeline}")
    sectors = [parse_segment_label(s) for s in result.mission]
    use = _lc_only_segments(sectors, req.segments)
    mask = [any(segments_match(s, u) for u in use) for s in sectors]
    if not any(mask):
        word = segment_word(req.mission).capitalize()
        available = sorted(set(sectors), key=natural_segment_key)
        raise PrepareError(
            f"{word} {list(req.segments)} not available; available: {available}"
        )
    rows = result[mask]
    exptimes = rows.table.to_pandas().exptime.unique()
    exptime = exptimes[0] if req.exptime is None else req.exptime
    if len(use) != len(rows):
        rows, exptime = choose_exptime(
            rows, exptimes, f"{segment_word(req.mission)}={use}", expected_n=len(use)
        )
    lc = safe_stitch(
        rows.download_all(
            flux_column=req.flux_column, quality_bitmask=req.quality_bitmask
        )
    )
    if lc is None:
        raise PrepareError("no usable segments downloaded")
    table = to_table(lc, req.mission)
    name = f"{label}_{req.pipeline}_s{'_'.join(use)}_exp{int(exptime)}s.csv"
    return write_table(table, basedir / name)
