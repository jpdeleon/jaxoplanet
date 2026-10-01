import math

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from jaxoplanet2.prepare import catalogs  # noqa: E402
from jaxoplanet2.prepare.catalogs import CatalogError  # noqa: E402
from jaxoplanet2.prepare.priors import Ephemeris  # noqa: E402

from . import fakes  # noqa: E402

FULL = {
    "period": 3.0,
    "period_err": 1e-4,
    "epoch": 2459000.0,
    "epoch_err": 1e-3,
    "duration": 2.0,
    "duration_err": 0.1,
    "depth": 1000.0,
    "depth_err": 50.0,
}


def test_resolve_ephemeris_prompts_only_for_missing_values():
    asked = []
    values = {**FULL, "epoch_err": None, "duration_err": None}
    eph = catalogs.resolve_ephemeris(values, lambda label: asked.append(label) or "0.5")
    assert asked == ["Epoch err (BJD): ", "Tdur err (h): "]
    assert eph == Ephemeris(**{**FULL, "epoch_err": 0.5, "duration_err": 0.5})


@pytest.mark.parametrize(
    ("key", "value"), [("period", 0.0), ("depth_err", -1.0), ("epoch", math.inf)]
)
def test_resolve_ephemeris_rejects_bad_values_before_prompting(key, value):
    with pytest.raises(CatalogError, match=f"--{key.replace('_', '-')}"):
        catalogs.resolve_ephemeris({key: value}, pytest.fail)


def test_merge_prefers_the_command_line():
    merged = catalogs.merge_ephemeris_values(
        {"period": 3.0}, {"period": 9.0, "depth": 5.0}
    )
    assert merged["period"] == 3.0
    assert merged["depth"] == 5.0
    assert merged["epoch"] is None


def test_find_target_from_toi(monkeypatch):
    monkeypatch.setattr(catalogs, "load_tois", lambda update: fakes.toi_table(planets=2))
    target = catalogs.find_target(toi=6715)
    assert target.name == "TOI-6715"
    assert target.tic_id == 123456789
    assert target.ra == pytest.approx(75.0)
    assert [p.period for p in target.planets] == [3.0, 6.0]
    with pytest.raises(CatalogError, match="TOI 1"):
        catalogs.find_target(toi=1)


def test_find_target_from_ctoi_renames_columns(monkeypatch, tmp_path):
    monkeypatch.setenv(catalogs.CACHE_ENV, str(tmp_path))
    row = {
        "CTOI": 42.01,
        "TIC ID": 42,
        "RA": "05:00:00",
        "Dec": "-30:00:00",
        "User Disposition": "PC",
        "Period (days)": 3.0,
        "Period (days) Error": 1e-4,
        "Transit Epoch (BJD)": 2459000.0,
        "Transit Epoch (BJD) Error": 1e-3,
        "Duration (hrs)": 2.0,
        "Duration (hrs) Error": 0.1,
        "Depth ppm": 1000.0,
        "Depth ppm Error": 50.0,
    }
    pd.DataFrame([row, {**row, "CTOI": 42.02, "User Disposition": "FP"}]).to_csv(
        tmp_path / "CTOIs.csv", index=False
    )
    target = catalogs.find_target(ctoi=42)
    assert target.name == "CTOI-42"
    assert target.planets == (Ephemeris(**FULL),)


def test_find_target_from_tic_needs_an_ephemeris():
    with pytest.raises(CatalogError):
        catalogs.find_target(tic=42)
    target = catalogs.find_target(tic=42, tic_ephemeris=Ephemeris(**FULL))
    assert (target.name, target.source, target.tic_id) == ("TIC-42", "custom", 42)
    with pytest.raises(CatalogError):
        catalogs.find_target()


def _nexsci_row(host):
    return {
        "hostname": host,
        "pl_orbper": 3.0,
        "pl_orbpererr1": 1e-4,
        "pl_orbpererr2": -1e-4,
        "pl_tranmid": 2459000.0,
        "pl_trandur": 2.0,
        "pl_trandurerr1": 0.1,
        "pl_trandurerr2": -0.1,
        "pl_trandep": 0.1,
        "pl_rade": 11.0,
        "pl_radeerr1": 0.5,
        "pl_radeerr2": -0.5,
        "st_teff": 5000.0,
        "st_tefferr1": 50.0,
        "st_tefferr2": -50.0,
        "st_logg": 4.5,
        "st_loggerr1": 0.1,
        "st_loggerr2": -0.1,
        "st_rad": 0.9,
        "st_raderr1": 0.03,
        "st_raderr2": -0.03,
        "st_mass": 0.9,
        "st_masserr1": 0.04,
        "st_masserr2": -0.04,
    }


def test_find_target_from_nexsci_via_alias(monkeypatch):
    table = pd.DataFrame([_nexsci_row("HIP 67522")])
    monkeypatch.setattr(catalogs, "load_nexsci", lambda update: table)
    monkeypatch.setattr(catalogs, "name_aliases", lambda name: ("TOI-2", "HIP 67522"))
    target = catalogs.find_target(name="TOI 2")
    assert target.name == "TOI2"
    assert target.planets[0].depth == pytest.approx(1000.0)
    assert target.planets[0].period_err == pytest.approx(math.hypot(1e-4, 1e-4))
    star = catalogs.nexsci_star(target.nexsci_host)
    assert (star.radius, math.isnan(star.feh)) == (0.9, True)
    k, _ = catalogs.radius_ratio_from_radii(target.planet_radii[0], star.radius)
    assert k == pytest.approx(11.0 * catalogs.R_EARTH_IN_R_SUN / 0.9)


def test_find_target_from_nexsci_unknown_host(monkeypatch):
    monkeypatch.setattr(
        catalogs, "load_nexsci", lambda update: pd.DataFrame([_nexsci_row("X")])
    )

    def no_aliases(name):
        raise CatalogError("offline")

    monkeypatch.setattr(catalogs, "name_aliases", no_aliases)
    with pytest.raises(CatalogError, match="NExSci"):
        catalogs.find_target(name="Y")


def test_with_coordinates_asks_exofop(monkeypatch):
    info = {"coordinates": {"ra": "10.5", "dec": "-5"}, "basic_info": {"tic_id": "7"}}
    monkeypatch.setattr(catalogs, "tfop_info", lambda name: info)
    target = catalogs.with_coordinates(catalogs.Target("TIC-7", "custom", (), tic_id=7))
    assert (target.ra, target.dec, target.tic_id) == (10.5, -5.0, 7)
    monkeypatch.setattr(catalogs, "tfop_info", lambda name: {})
    with pytest.raises(CatalogError):
        catalogs.with_coordinates(catalogs.Target("X", "nexsci", ()))


def test_alias_with_prefix(monkeypatch):
    monkeypatch.setattr(
        catalogs, "name_aliases", lambda name: ("K2-100", "EPIC 211990866")
    )
    assert catalogs.alias_with_prefix("K2-100", "epic") == "EPIC 211990866"
    with pytest.raises(CatalogError):
        catalogs.alias_with_prefix("K2-100", "HD")


def test_cached_tables_are_reused_unless_updated(monkeypatch, tmp_path):
    monkeypatch.setenv(catalogs.CACHE_ENV, str(tmp_path))
    calls = []

    def download():
        calls.append(1)
        return pd.DataFrame({"a": [1]})

    catalogs._cached_csv("t.csv", download, update=False)
    catalogs._cached_csv("t.csv", download, update=False)
    catalogs._cached_csv("t.csv", download, update=True)
    assert len(calls) == 2


def test_float_handles_masked_and_bad_values():
    assert math.isnan(catalogs._float(np.ma.masked))
    assert math.isnan(catalogs._float("n/a"))
    assert catalogs._float("1.5") == 1.5
