import numpy as np
import pytest

lk = pytest.importorskip("lightkurve")

from jaxoplanet2.prepare import lightcurves as lcs  # noqa: E402
from jaxoplanet2.prepare.errors import PrepareError  # noqa: E402
from jaxoplanet2.prepare.h5 import H5LightCurve  # noqa: E402
from jaxoplanet2.prepare.products import spoc_crowding, write_contamination  # noqa: E402

from . import fakes  # noqa: E402


def test_segment_helpers():
    assert lcs.parse_segment_label("K2 Campaign 11a") == "11a"
    assert lcs.segments_match("01", 1) and not lcs.segments_match("11a", "11b")
    assert sorted(["11b", "2", "11a", "x"], key=lcs.natural_segment_key) == [
        "2",
        "11a",
        "11b",
        "x",
    ]
    assert lcs.segment_word("k2", plural=True) == "campaigns"
    flags = [lcs.segment_flag(s) for s in (None, ["all"], ["-1"], ["0"], ["3", "4"])]
    assert flags == ["default", "all_sector", "last", "first", "multi_sector"]


def test_request_flux_column():
    req = lcs.Request("tess", "QLP", None, "pdcsap", "default", None, "tess", "last")
    assert req.flux_column == "sap_flux"
    assert (
        lcs.Request(
            "t", "spoc", None, "sap", "default", None, "tess", "last"
        ).flux_column
        == "sap_flux"
    )


def test_safe_stitch_normalizes_and_drops_empty_segments():
    good = fakes.make_lc(1, 120, "pdcsap_flux", "spoc")
    empty = fakes.make_lc(2, 120, "pdcsap_flux", "spoc")
    empty.flux = empty.flux * np.nan
    stitched = lcs.safe_stitch(lk.LightCurveCollection([good, empty]))
    assert len(stitched) == len(good)
    assert np.nanmedian(stitched.flux.value) == pytest.approx(1.0, abs=1e-3)
    assert lcs.safe_stitch(lk.LightCurveCollection([empty])) is None
    assert lcs.safe_stitch(good) is good


def test_to_table_replaces_missing_errors_and_sorts():
    lc = fakes.make_lc(1, 600, "sap_flux", "qlp")
    table = lcs.to_table(lc[::-1], "tess")
    assert np.all(np.diff(table.time) > 0)
    assert table.time[0] == pytest.approx(fakes.SECTOR_START[1] + 2457000)
    assert np.all(table.flux_err == 1.0)
    lc.flux = lc.flux * np.nan
    with pytest.raises(PrepareError):
        lcs.to_table(lc, "tess")


def test_choose_exptime(monkeypatch):
    result = fakes.fake_search([(1, "SPOC", 20), (1, "SPOC", 120), (2, "SPOC", 120)])(
        "x"
    )
    monkeypatch.delenv("JAXOPLANET_NONINTERACTIVE", raising=False)
    monkeypatch.delenv("ALLESFITTER_NONINTERACTIVE", raising=False)
    with pytest.raises(PrepareError, match="-e 20"):
        lcs.choose_exptime(result, [20.0, 120.0], "all sectors")
    monkeypatch.setenv("ALLESFITTER_NONINTERACTIVE", "1")
    narrowed, chosen = lcs.choose_exptime(result, [20.0, 120.0], "x", expected_n=2)
    assert (chosen, len(narrowed)) == (120.0, 2)
    assert lcs.choose_exptime(result, [20.0, 120.0], "x")[1] == 20.0


def test_h5_matches_segment():
    h5lc = H5LightCurve(np.arange(3.0), np.ones(3), np.ones(3), "spoc", 5, 120.0)
    assert lcs.h5_matches_segment(h5lc, "05", "SPOC")
    assert not lcs.h5_matches_segment(h5lc, "5", "qlp")
    assert not lcs.h5_matches_segment(h5lc, "6", "spoc")
    assert not lcs.h5_matches_segment(None, "5", "spoc")
    unknown = H5LightCurve(h5lc.time, h5lc.flux, h5lc.flux_err, None, 5, None)
    assert lcs.h5_matches_segment(unknown, "5", "qlp")
    assert len(lcs.lightcurve_from_h5(h5lc)) == 3


def test_contamination_table(tmp_path):
    collection = lk.LightCurveCollection(
        [fakes.make_lc(s, 1800, "pdcsap_flux", "spoc") for s in (1, 2)]
    )
    rows = spoc_crowding(collection)
    assert [r[0] for r in rows] == ["1", "2"]
    dilution = write_contamination(rows, tmp_path / "c.txt", "tess")
    assert dilution.median == pytest.approx(0.035)
    assert "median_dilution\t0.035000" in (tmp_path / "c.txt").read_text()
    assert write_contamination([], tmp_path / "none.txt", "tess") is None
    assert spoc_crowding(fakes.make_lc(1, 1800, "sap_flux", "qlp")) == []
