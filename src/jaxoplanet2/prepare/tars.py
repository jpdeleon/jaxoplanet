"""Teach lightkurve to read TARS (TESS All-Sky Rotation Survey) light curves.

lightkurve rejects TARS HLSPs ("Not recognized as a supported data product"):
they are identified only by ``HLSPID='TARS'`` and hold just ``TIME`` and
``FLUX``. :func:`patch_lightkurve` registers a reader so ``-p tars`` downloads
work. TARS has no error column: ``flux_err`` is NaN, which ``jaxoplanet init``
replaces by 1 (the fitted ``ln_err_flux_<inst>`` then sets the scale).
"""

import importlib
import os

import numpy as np

_PATCH_FLAG = "_jaxoplanet2_tars_patched"


def _is_tars_header(header) -> bool:
    return str(header.get("HLSPID", "")).strip().upper() == "TARS"


def read_tars(path):
    """A TARS ``*_lc.fits`` file as a ``lightkurve.TessLightCurve`` (BTJD, TDB)."""
    import astropy.units as u
    from astropy.io import fits
    from astropy.time import Time
    from lightkurve import TessLightCurve

    with fits.open(path) as hdulist:
        primary = hdulist[0].header
        if not _is_tars_header(primary):
            raise ValueError(
                f"{os.path.basename(str(path))!r} is not a TARS light curve"
            )
        lc_hdu = hdulist[1]
        missing = [c for c in ("TIME", "FLUX") if c not in lc_hdu.columns.names]
        if missing:
            raise ValueError(f"TARS light curve lacks the {missing} column(s)")
        time = np.asarray(lc_hdu.data["TIME"], dtype=float)
        flux = np.asarray(lc_hdu.data["FLUX"], dtype=float)
    meta_keys = {
        "OBJECT": "HLSPTARG",
        "LABEL": "HLSPTARG",
        "TARGETID": "TICID",
        "TICID": "TICID",
        "SECTOR": "SECTOR",
        "CAMERA": "CAMERA",
        "CCD": "CCD",
        "EXPTIME": "EXPTIME",
        "RA": "RA_TARG",
        "DEC": "DEC_TARG",
    }
    meta = {k: primary.get(v) for k, v in meta_keys.items()}
    return TessLightCurve(
        time=Time(time, format="btjd", scale="tdb"),
        flux=flux * u.dimensionless_unscaled,
        flux_err=np.full_like(flux, np.nan) * u.dimensionless_unscaled,
        meta={"MISSION": "TESS", "AUTHOR": "TARS", "FLUX_ORIGIN": "TARS", **meta},
    )


def is_tars_file(path_or_url) -> bool:
    """True for a readable TARS product; never raises."""
    from astropy.io import fits

    try:
        with fits.open(path_or_url) as hdulist:
            return _is_tars_header(hdulist[0].header)
    except Exception:
        return False


def patch_lightkurve() -> bool:
    """Route TARS files in ``lightkurve.read`` (and so ``download_all``) to
    :func:`read_tars`. Idempotent; returns False if already patched.

    ``read`` is imported by value into several lightkurve namespaces, so each
    binding is replaced.
    """
    import lightkurve
    import lightkurve.io as lk_io
    import lightkurve.io.detect as lk_detect
    import lightkurve.search as lk_search
    from astropy.io import registry
    from lightkurve.lightcurve import LightCurve

    if getattr(lightkurve, _PATCH_FLAG, False):
        return False
    lk_read = importlib.import_module("lightkurve.io.read")

    def _reader(path_or_url, **_kwargs):  # flux_column/quality_bitmask: n/a
        return read_tars(path_or_url)

    registry.register_reader("tars", LightCurve, _reader, force=True)
    original_detect = lk_detect.detect_filetype
    original_read = lk_read.read

    def detect_filetype(hdulist):
        try:
            if _is_tars_header(hdulist[0].header):
                return "TARS"
        except Exception:
            pass
        return original_detect(hdulist)

    def read(path_or_url, **kwargs):
        if is_tars_file(path_or_url):
            return _reader(path_or_url)
        return original_read(path_or_url, **kwargs)

    for module in (lk_detect, lk_read):
        if hasattr(module, "detect_filetype"):
            module.detect_filetype = detect_filetype
    for module in (lk_read, lk_io, lk_search, lightkurve):
        if hasattr(module, "read"):
            module.read = read
    setattr(lightkurve, _PATCH_FLAG, True)
    return True
