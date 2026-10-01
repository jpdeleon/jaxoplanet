import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from jaxoplanet2.prepare.h5 import read_lightcurve, read_transit_params  # noqa: E402


def _write_current(path, lightcurve=True):
    scalars = {
        "period": 3.0,
        "period_uncertainty": 1e-3,
        "T0": 1325.5,
        "duration": 0.1,
        "depth": 0.99,
        "depth_mean": [0.99, 2e-4],
        "pipeline": "SPOC",
        "sector": 1,
        "exptime": 120,
    }
    with h5py.File(path, "w") as f:
        if lightcurve:
            f.create_dataset("time_raw", data=np.arange(3.0))
            f.create_dataset("flux_raw", data=np.ones(3))
            f.create_dataset("err_raw", data=np.full(3, 1e-3))
        f.attrs["_scalar_data"] = json.dumps(scalars)


def test_read_transit_params_current_layout(tmp_path):
    path = tmp_path / "TIC1_s1_tls.h5"
    _write_current(path)
    values = read_transit_params(str(path))
    assert values["period"] == 3.0
    assert values["epoch"] == pytest.approx(2458325.5)
    assert values["duration"] == pytest.approx(2.4)
    assert values["depth"] == pytest.approx(1e4)
    assert values["depth_err"] == pytest.approx(200.0)
    assert values["epoch_err"] is None and values["duration_err"] is None


def test_read_transit_params_legacy_layout(tmp_path):
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as f:
        f.attrs.update(
            {"period": 5.0, "T0": 10.0, "duration": 0.2, "depth": 990.0, "TITLE": "x"}
        )
        group = f.create_group("depth_mean")
        group.attrs.update({"i0": 0.99, "i1": 1e-4})
    values = read_transit_params(str(path), mission="kepler")
    assert values["epoch"] == pytest.approx(2454843.0)
    assert values["depth"] == pytest.approx(1e4)  # ppt-style TLS depth
    assert values["depth_err"] == pytest.approx(100.0)
    assert values["period_err"] is None


def test_read_lightcurve(tmp_path):
    path = tmp_path / "a.h5"
    _write_current(path)
    lc = read_lightcurve(str(path))
    assert (lc.pipeline, lc.sector, lc.exptime) == ("spoc", 1, 120.0)
    assert lc.flux.tolist() == [1.0, 1.0, 1.0]
    _write_current(path, lightcurve=False)
    assert read_lightcurve(str(path)) is None


def _write_tars(path, hlspid="TARS"):
    from astropy.io import fits

    primary = fits.PrimaryHDU()
    primary.header.update({"HLSPID": hlspid, "TICID": 7, "SECTOR": 1})
    columns = [
        fits.Column(name="TIME", format="D", array=np.arange(1325.0, 1330.0)),
        fits.Column(name="FLUX", format="D", array=np.ones(5)),
    ]
    fits.HDUList([primary, fits.BinTableHDU.from_columns(columns)]).writeto(path)


def test_tars_reader_and_lightkurve_patch(tmp_path):
    lk = pytest.importorskip("lightkurve")
    from jaxoplanet2.prepare.tars import is_tars_file, patch_lightkurve, read_tars

    path = tmp_path / "hlsp_tars_lc.fits"
    _write_tars(path)
    lc = read_tars(path)
    assert lc.meta["AUTHOR"] == "TARS"
    assert np.all(np.isnan(lc.flux_err.value))
    assert is_tars_file(path) and not is_tars_file(tmp_path / "missing.fits")
    patch_lightkurve()
    assert patch_lightkurve() is False  # idempotent
    assert len(lk.read(path)) == 5
    other = tmp_path / "other.fits"
    _write_tars(other, hlspid="QLP")
    with pytest.raises(ValueError, match="not a TARS"):
        read_tars(other)
