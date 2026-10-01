import pytest
from typer.testing import CliRunner

from jaxoplanet2.cli import app
from jaxoplanet2.prepare.errors import PrepareError
from jaxoplanet2.prepare.options import base_dir, build_parser, has_target, plan

runner = CliRunner()


def _plan(*argv):
    return plan(build_parser().parse_args(list(argv)))


def test_same_flags_as_prepare_allesfit():
    args = build_parser().parse_args(
        [
            "-toi",
            "1097",
            "-s",
            "1",
            "2",
            "-e",
            "120",
            "600",
            "-p",
            "spoc",
            "qlp",
            "-f",
            "spoc120",
            "qlp600",
            "-sig",
            "5",
            "-qb",
            "hard",
            "-i",
            "-u",
            "-o",
        ]
    )
    assert has_target(args)
    p = plan(args)
    assert p.downloads == (("spoc120", "spoc", 120.0), ("qlp600", "qlp", 600.0))
    assert p.segments == ("1", "2")


def test_plan_defaults_and_broadcasts():
    p = _plan("-toi", "1", "-e", "120", "-f", "a", "b")
    assert (p.insts, p.pipelines, p.exptimes, p.segments) == (
        ("a", "b"),
        ("spoc",),
        (120.0,),
        None,
    )
    assert _plan("-name", "K2-18", "-m", "k2").segments == ("-1",)
    assert _plan("-name", "Kepler-1", "-m", "kepler", "-q", "3").segments == ("3",)


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["-toi", "1", "-bp", "tess"], "--bandpass"),
        (["-toi", "1", "--stellar-var-gp-sho"], "stellar-var"),
        (["-toi", "1", "-p", "spoc", "qlp"], "--pipeline has 2"),
        (
            ["-toi", "1", "-p", "spoc", "qlp", "-f", "a", "b", "-e", "1", "2", "3"],
            "--exptime has 3",
        ),
        (["-ctoi", "1", "--h5", "x.h5"], "--h5"),
        (["-toi", "1", "-c", "5"], "-s/--sector for TESS"),
        (["-toi", "1", "--lc-only"], "--lc-only"),
    ],
)
def test_plan_rejects(argv, match):
    with pytest.raises(PrepareError, match=match):
        _plan(*argv)


def test_base_dir():
    parse = build_parser().parse_args
    assert base_dir(parse(["-toi", "1"])) == "."
    assert base_dir(parse(["out", "-toi", "1"])) == "out"
    assert base_dir(parse(["-toi", "1", "-dir", "d"])) == "d"
    with pytest.raises(PrepareError):
        base_dir(parse(["a", "-toi", "1", "-dir", "b"]))


def test_cli_init_help_lists_prepare_options():
    result = runner.invoke(app, ["init", "-h"])
    assert result.exit_code == 0
    for flag in ("-toi", "--sector", "--exptime", "--h5", "--lc-only"):
        assert flag in result.output


def test_cli_init_needs_a_directory_or_target():
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 1
    assert "-toi" in result.output


def test_cli_init_reports_prepare_errors():
    result = runner.invoke(app, ["init", "-toi", "1", "-bp", "tess"])
    assert result.exit_code == 1
    assert "not supported by jaxoplanet2" in result.output


def test_cli_init_with_target_runs_prepare(monkeypatch, tmp_path):
    import jaxoplanet2.prepare.run as prepare_run

    seen = {}
    monkeypatch.setattr(
        prepare_run, "run", lambda args: seen.setdefault("a", args) and tmp_path
    )
    result = runner.invoke(app, ["init", str(tmp_path), "-toi", "1097", "-s", "all"])
    assert result.exit_code == 0, result.output
    assert seen["a"].toi == 1097 and seen["a"].sector == ["all"]
    assert f"wrote {tmp_path}" in result.output
