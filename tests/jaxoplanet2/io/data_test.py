import numpy as np
import pytest

from jaxoplanet2.io.data import (
    DataError,
    Dataset,
    fast_fit_mask,
    load_dataset,
    load_datasets,
)
from jaxoplanet2.io.settings import parse_settings_text


def write(path, text):
    path.write_text(text)
    return path


def test_loads_headerless_three_columns(tmp_path):
    write(tmp_path / "tess.csv", "1.0,1.001,0.001\n2.0,0.999,0.002\n")
    d = load_dataset(tmp_path, "tess", "flux")
    assert d.inst == "tess"
    assert d.kind == "flux"
    np.testing.assert_array_equal(d.time, [1.0, 2.0])
    np.testing.assert_array_equal(d.y, [1.001, 0.999])
    np.testing.assert_array_equal(d.yerr, [0.001, 0.002])
    assert len(d) == 2


def test_hash_header_and_extra_columns(tmp_path):
    write(
        tmp_path / "tess.csv", "#time,flux,flux_err,airmass\n1,1,0.1,1.2\n2,1,0.1,1.3\n"
    )
    d = load_dataset(tmp_path, "tess", "flux")
    np.testing.assert_array_equal(d.time, [1, 2])


def test_plain_header_row(tmp_path):
    write(tmp_path / "harps.csv", "BJD,RV,RV_err\n1,0.01,0.002\n2,0.02,0.002\n")
    d = load_dataset(tmp_path, "harps", "rv")
    np.testing.assert_array_equal(d.y, [0.01, 0.02])


def test_comment_lines_are_skipped(tmp_path):
    write(tmp_path / "tess.csv", "# my notes\n1,1,0.1\n# more\n2,1,0.1\n")
    assert len(load_dataset(tmp_path, "tess", "flux")) == 2


def test_inline_comments_are_stripped(tmp_path):
    write(tmp_path / "tess.csv", "1,1,0.1 # first\n2,1,0.1\n")
    np.testing.assert_array_equal(
        load_dataset(tmp_path, "tess", "flux").yerr, [0.1, 0.1]
    )


def test_arrays_are_read_only(tmp_path):
    write(tmp_path / "tess.csv", "1,1,0.1\n2,1,0.1\n")
    d = load_dataset(tmp_path, "tess", "flux")
    with pytest.raises(ValueError):
        d.y[0] = 5.0


@pytest.mark.parametrize(
    "text, match",
    [
        ("1,nan,0.1\n2,1,0.1\n", "NaN"),
        ("1,1,0\n2,1,0.1\n", "positive"),
        ("1,1,-0.1\n2,1,0.1\n", "positive"),
        ("2,1,0.1\n1,1,0.1\n", "sorted"),
        ("1,1\n2,1\n", "3 columns"),
        ("", "no data"),
    ],
)
def test_invalid_data(tmp_path, text, match):
    write(tmp_path / "tess.csv", text)
    with pytest.raises(DataError, match=match):
        load_dataset(tmp_path, "tess", "flux")


def test_repeated_timestamps_warn(tmp_path):
    write(tmp_path / "tess.csv", "1,1,0.1\n1,1,0.1\n2,1,0.1\n")
    with pytest.warns(UserWarning, match="repeated"):
        load_dataset(tmp_path, "tess", "flux")


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="tess.csv"):
        load_dataset(tmp_path, "tess", "flux")


def test_select_returns_new_dataset():
    d = Dataset("tess", "flux", np.arange(4.0), np.ones(4), np.full(4, 0.1))
    sub = d.select(np.array([True, False, True, False]))
    np.testing.assert_array_equal(sub.time, [0.0, 2.0])
    assert len(d) == 4


def test_fast_fit_mask_keeps_windows_around_each_transit():
    time = np.linspace(0, 10, 1001)
    mask = fast_fit_mask(time, [(1.0, 3.0)], width=0.4)
    kept = time[mask]
    for tmid in (1.0, 4.0, 7.0, 10.0):
        near = np.abs(kept - tmid) <= 0.2 + 1e-12
        assert near.any()
    assert np.all(np.min(np.abs(kept[:, None] - [1, 4, 7, 10]), axis=1) <= 0.2 + 1e-9)


def test_fast_fit_mask_handles_epoch_outside_data_and_multiple_companions():
    time = np.linspace(100, 110, 2001)
    mask = fast_fit_mask(time, [(0.5, 2.0), (0.0, 5.0)], width=0.2)
    kept = time[mask]
    assert np.any(np.isclose(kept, 100.5, atol=1e-3))
    assert np.any(np.isclose(kept, 105.0, atol=1e-3))


def test_load_datasets_applies_fast_fit_to_photometry_only(tmp_path):
    t = np.linspace(0, 10, 101)
    rows = "\n".join(f"{x},1,0.001" for x in t)
    write(tmp_path / "tess.csv", rows)
    write(tmp_path / "harps.csv", rows)
    settings = parse_settings_text(
        "companions_phot,b\ncompanions_rv,b\ninst_phot,tess\ninst_rv,harps\n"
        "fast_fit,True\nfast_fit_width,0.5\n"
    )
    data = load_datasets(tmp_path, settings, {"b_epoch": 1.0, "b_period": 3.0})
    assert len(data["harps"]) == 101
    assert 0 < len(data["tess"]) < 101


def test_load_datasets_fast_fit_without_transits_fails(tmp_path):
    write(tmp_path / "tess.csv", "0.0,1,0.001\n0.1,1,0.001\n")
    settings = parse_settings_text(
        "companions_phot,b\ninst_phot,tess\nfast_fit,True\nfast_fit_width,0.01\n"
    )
    with pytest.raises(DataError, match="in-transit"):
        load_datasets(tmp_path, settings, {"b_epoch": 0.5, "b_period": 10.0})


def test_load_datasets_without_fast_fit_keeps_everything(tmp_path):
    write(tmp_path / "tess.csv", "0.0,1,0.001\n0.1,1,0.001\n")
    settings = parse_settings_text("companions_phot,b\ninst_phot,tess\n")
    data = load_datasets(tmp_path, settings, {"b_epoch": 0.5, "b_period": 10.0})
    assert len(data["tess"]) == 2
