import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lightkurve")

from jaxoplanet2.io.params import load_params  # noqa: E402
from jaxoplanet2.io.settings import load_settings  # noqa: E402
from jaxoplanet2.prepare import catalogs  # noqa: E402
from jaxoplanet2.prepare.errors import PrepareError  # noqa: E402
from jaxoplanet2.prepare.options import build_parser  # noqa: E402
from jaxoplanet2.prepare.priors import Star  # noqa: E402
from jaxoplanet2.prepare.run import run  # noqa: E402
from jaxoplanet2.validate import validate  # noqa: E402

from . import fakes  # noqa: E402

SUN = Star(5772, 50, 4.44, 0.05, 0.0, 0.1, 1.0, 0.02, 1.0, 0.03)
TWO_SECTORS = [(1, "SPOC", 120), (2, "SPOC", 120), (1, "QLP", 600), (2, "SPOC", 20)]


@pytest.fixture
def offline(monkeypatch, tmp_path):
    import lightkurve

    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv(catalogs.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.delenv("JAXOPLANET_NONINTERACTIVE", raising=False)
    monkeypatch.delenv("ALLESFITTER_NONINTERACTIVE", raising=False)
    monkeypatch.setattr(catalogs, "load_tois", lambda update=False: fakes.toi_table())
    monkeypatch.setattr(catalogs, "tic_star", lambda tic: SUN)
    monkeypatch.setattr(catalogs, "tess_sectors", lambda *a: np.array([1, 2, 3]))
    monkeypatch.setattr(lightkurve, "search_lightcurve", fakes.fake_search(TWO_SECTORS))
    return tmp_path


def _args(tmp_path, *argv):
    return build_parser().parse_args([str(tmp_path / "runs"), *argv])


def _no_prompt(label):
    raise AssertionError(f"unexpected prompt {label!r}")


def test_toi_builds_a_valid_fit_directory(offline, monkeypatch):
    claret = lambda **kw: (0.4, 0.1, 0.3, 0.1)  # noqa: E731
    monkeypatch.setitem(sys.modules, "limbdark", SimpleNamespace(claret=claret))
    outdir = run(_args(offline, "-toi", "6715", "-s", "1", "2", "-e", "120"), _no_prompt)

    assert outdir == offline / "runs" / "TOI-6715"
    for name in (
        "params.csv",
        "settings.csv",
        "params_star.csv",
        "run.sh",
        "tess.csv",
        "tess_spoc_contamination.txt",
        "TOI-6715.log",
        "TOI-6715_tess_pdcsap_s1s2_exp120s.png",
    ):
        assert (outdir / name).is_file(), name
    params = load_params(outdir)
    assert params["b_duration"].value == pytest.approx(fakes.DURATION_H / 24, rel=1e-3)
    assert params["b_radius_ratio"].value == pytest.approx(0.1, rel=1e-3)
    text = (outdir / "params.csv").read_text()
    assert "#host_ldc_q1_tess,0.40,1,normal 0.40 0.10" in text
    assert "#dil_tess,0.035000,1,normal 0.035000 0.010000" in text  # CROWDSAP .97,.96
    settings = load_settings(outdir)
    assert settings.t_exp["tess"] == pytest.approx(120 / 86400, rel=1e-3)
    assert validate(outdir).ok


def test_existing_directory_needs_overwrite(offline):
    args = ["-toi", "6715", "-s", "1", "-e", "120", "--no-validate"]
    run(_args(offline, *args))
    with pytest.raises(PrepareError, match="--overwrite"):
        run(_args(offline, *args))
    run(_args(offline, *args, "-o"))


def test_mixed_exposure_times_need_a_choice(offline, monkeypatch):
    args = _args(offline, "-toi", "6715", "-s", "all", "--no-validate")
    with pytest.raises(PrepareError, match="-e 20"):
        run(args)
    monkeypatch.setenv("JAXOPLANET_NONINTERACTIVE", "1")
    outdir = run(args)
    settings = load_settings(outdir)
    assert settings.t_exp["tess"] == pytest.approx(120 / 86400, rel=1e-3)


def test_unobserved_sector_is_rejected(offline):
    with pytest.raises(PrepareError, match="not observed"):
        run(_args(offline, "-toi", "6715", "-s", "9"))


def test_raw_tic_qlp_with_ttvs(offline, monkeypatch):
    monkeypatch.setattr(
        catalogs,
        "tfop_info",
        lambda name: {
            "coordinates": {"ra": 75.0, "dec": -30.0},
            "basic_info": {"tic_id": 42},
        },
    )
    ephemeris = [
        "--period",
        "3",
        "--period-err",
        "0.0001",
        "--epoch",
        str(fakes.EPOCH_BTJD + 2457000),
        "--epoch-err",
        "0.001",
        "--duration",
        "2.4",
        "--duration-err",
        "0.1",
        "--depth",
        "10000",
        "--depth-err",
        "100",
    ]
    argv = ["-tic", "42", "-s", "1", "-p", "qlp", "--ttv", "--no-validate", *ephemeris]
    outdir = run(_args(offline, *argv))

    assert outdir.name == "TIC-42"
    data = np.loadtxt(outdir / "tess.csv", delimiter=",")
    assert np.all(data[:, 2] == 1.0)  # QLP has no errors
    ttv_rows = [n for n in load_params(outdir).names if "_ttv_transit_" in n]
    assert len(ttv_rows) == 7  # 20 d of sector 1 at P = 3 d
    assert load_settings(outdir).fit_ttvs
    assert not (outdir / "tess_spoc_contamination.txt").exists()
    assert validate(outdir).ok


def test_single_pipeline_warns_about_undownloaded_instruments(offline):
    argv = [
        "-toi",
        "6715",
        "-s",
        "1",
        "-e",
        "120",
        "-f",
        "tess",
        "kepler",
        "--no-validate",
    ]
    outdir = run(_args(offline, *argv))
    assert "kepler.csv does not exist" in (outdir / "TOI-6715.log").read_text()
    assert "t_exp_kepler,0.001389" in (outdir / "settings.csv").read_text()


def test_lc_only_saves_just_the_light_curve(offline):
    paths = run(_args(offline, "-tic", "42", "-s", "2", "-e", "120", "--lc-only"))
    assert [p.name for p in paths] == ["TIC-42_spoc_s2_exp120s.csv"]
    assert not (offline / "runs" / "TIC-42").exists()


def test_missing_stellar_radius_needs_interactive(offline, monkeypatch):
    unknown_radius = Star(5772, 50, 4.44, 0.05, 0.0, 0.1, np.nan, np.nan, 1.0, 0.03)
    monkeypatch.setattr(catalogs, "tic_star", lambda tic: unknown_radius)
    args = ["-toi", "6715", "-s", "1", "-e", "120", "--no-validate"]
    with pytest.raises(PrepareError, match="Rstar"):
        run(_args(offline, *args))
    outdir = run(_args(offline, *args, "-i", "-o"), lambda label: "0.9")
    assert "use_host_density_prior,False" in (outdir / "settings.csv").read_text()
    assert (
        (outdir / "params_star.csv").read_text().splitlines()[-1].startswith("0.90,0.10")
    )


def _h5(path, sector=1, pipeline="spoc"):
    import json

    h5py = pytest.importorskip("h5py")
    time = np.arange(fakes.SECTOR_START[1], fakes.SECTOR_START[1] + 20, 120 / 86400)
    scalars = {
        "period": 3.0,
        "period_uncertainty": 1e-4,
        "T0": fakes.EPOCH_BTJD,
        "duration": 0.1,
        "depth": 0.99,
        "pipeline": pipeline,
        "sector": sector,
        "exptime": 120,
    }
    with h5py.File(path, "w") as f:
        f.create_dataset("time_raw", data=time)
        f.create_dataset("flux_raw", data=fakes.transit_flux(time))
        f.create_dataset("err_raw", data=np.full(time.size, 1e-3))
        f.attrs["_scalar_data"] = json.dumps(scalars)
    return str(path)


def test_tic_from_h5_reuses_its_light_curve(offline, monkeypatch):
    monkeypatch.setattr(
        catalogs,
        "tfop_info",
        lambda name: {
            "coordinates": {"ra": 75.0, "dec": -30.0},
            "basic_info": {"tic_id": 42},
        },
    )
    asked = []
    prompt = lambda label: asked.append(label) or "0.01"  # noqa: E731
    argv = ["-tic", "42", "-s", "1", "--h5", _h5(offline / "a.h5"), "--no-validate"]
    outdir = run(_args(offline, *argv), prompt)
    assert asked == ["Epoch err (BJD): ", "Tdur err (h): ", "Depth err (ppm): "]
    assert "embedded in the h5" in (outdir / "TIC-42.log").read_text()
    assert not (outdir / "tess_spoc_contamination.txt").exists()
    assert load_params(outdir)["b_period"].value == 3.0


def test_toi_with_h5_adds_a_companion(offline):
    h5 = _h5(offline / "p2.h5", sector=2)
    argv = [
        "-toi",
        "6715",
        "-s",
        "1",
        "-e",
        "120",
        "--h5",
        h5,
        "--period",
        "7",
        "--epoch-err",
        "0.01",
        "--depth-err",
        "50",
        "--duration-err",
        "0.1",
        "--no-validate",
    ]
    outdir = run(_args(offline, *argv), _no_prompt)
    params = load_params(outdir)
    assert params["c_period"].value == 7.0
    assert load_settings(outdir).companions_phot == ("b", "c")


def test_nexsci_target(offline, monkeypatch):
    target = catalogs.Target(
        "HIP67522",
        "nexsci",
        catalogs.find_target(toi=6715).planets,
        nexsci_host={
            "st_teff": 5700.0,
            "st_teff_err": 50.0,
            "st_logg": 4.4,
            "st_logg_err": 0.1,
            "st_rad": 1.0,
            "st_rad_err": 0.02,
            "st_mass": 1.0,
            "st_mass_err": 0.03,
        },
        planet_radii=((11.0, 0.5),),
    )
    monkeypatch.setattr(catalogs, "find_target", lambda **kw: target)
    monkeypatch.setattr(
        catalogs,
        "tfop_info",
        lambda name: {
            "coordinates": {"ra": 1.0, "dec": 2.0},
            "basic_info": {"tic_id": 5},
        },
    )
    outdir = run(
        _args(offline, "-name", "HIP 67522", "-s", "2", "-e", "120", "--no-validate")
    )
    assert outdir.name == "HIP67522"
    assert "[Fe/H]=0.0+/-0.1" in (outdir / "HIP67522.log").read_text()


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["-toi", "6715", "-s", "1", "-p", "tglc"], "pipeline=tglc"),
        (["-toi", "6715", "-s", "3"], "no light curve for sector"),
        (["-toi", "6715", "-s", "1", "2", "-e", "600", "-p", "qlp"], "not every sector"),
    ],
)
def test_download_errors(offline, argv, match):
    with pytest.raises(PrepareError, match=match):
        run(_args(offline, *argv))
