"""Command line of ``jaxoplanet init``, identical to allesfitter's prepare_allesfit.py.

``jaxoplanet init DIR`` alone writes template files into DIR. With a target
(``-toi``, ``-ctoi``, ``-tic`` or ``-name``) it builds ``DIR/<target>/`` from
catalogs and downloaded light curves instead, e.g.::

    jaxoplanet init -toi 1097 -s all -e 120
    jaxoplanet init -tic 273586149 -s -1 -p qlp --period 3.1 --epoch 2459000.1 ...
    jaxoplanet init -name "HIP 67522" -o -i --debug
"""

from argparse import SUPPRESS, ArgumentParser, Namespace
from dataclasses import dataclass

from jaxoplanet2.prepare.catalogs import EPHEMERIS_KEYS
from jaxoplanet2.prepare.errors import PrepareError

MISSIONS = ("tess", "k2", "kepler")
EPILOG = (
    "Without -toi/-ctoi/-tic/-name, DIR gets template params.csv and settings.csv. "
    "With a target, the fit directory is DIR/<target> (DIR defaults to -dir, "
    "then '.'). Put DIR before multi-value options such as -s 1 2."
)


def _add_target_options(ap: ArgumentParser) -> None:
    target = ap.add_mutually_exclusive_group()
    target.add_argument("-toi", help="TOI ID", type=int)
    target.add_argument("-ctoi", help="CTOI ID (its TIC ID)", type=int)
    target.add_argument("-tic", help="TIC ID (give its ephemeris, or --h5)", type=int)
    target.add_argument("-name", help="host name in NExSci, e.g. 'HIP 67522'", type=str)
    segment = ap.add_mutually_exclusive_group()
    segment.add_argument(
        "-s",
        "--sector",
        nargs="+",
        default=None,
        help="TESS sector(s); -1: most recent, 0: first, all: all",
    )
    segment.add_argument(
        "-c", "--campaign", default=None, help="K2 campaign; -1: most recent, all: all"
    )
    segment.add_argument(
        "-q", "--quarter", default=None, help="Kepler quarter; -1: most recent, all: all"
    )


def _add_data_options(ap: ArgumentParser) -> None:
    ap.add_argument(
        "-e",
        "--exptime",
        type=float,
        nargs="+",
        default=None,
        help="exposure time(s) in s, one per --pipeline or one shared "
        "(default: auto-detected)",
    )
    ap.add_argument(
        "-p",
        "--pipeline",
        type=str,
        nargs="+",
        default=["spoc"],
        help="pipeline(s) (default: spoc); one per --filename downloads "
        "every instrument, e.g. -f spoc120 qlp600 -p spoc qlp -e 120 600",
    )
    ap.add_argument(
        "-f",
        "--filename",
        type=str,
        nargs="+",
        default=None,
        help="instrument name(s), i.e. <inst>.csv (default: the mission)",
    )
    ap.add_argument("-m", "--mission", type=str, default="tess", choices=MISSIONS)
    ap.add_argument(
        "-lc",
        "--lc_type",
        type=str,
        default="pdcsap",
        choices=("pdcsap", "sap"),
        help="flux column (default: pdcsap; QLP always uses sap)",
    )
    ap.add_argument(
        "-sig", "--sigma", type=float, default=None, help="sigma for removing outliers"
    )
    ap.add_argument(
        "-qb",
        "--quality",
        type=str,
        default="default",
        choices=("none", "default", "hard", "hardest"),
    )


def _add_ephemeris_options(ap: ArgumentParser) -> None:
    units = {"period": "d", "epoch": "BJD", "duration": "h", "depth": "ppm"}
    for key in EPHEMERIS_KEYS:
        quantity = key.removesuffix("_err")
        what = "uncertainty" if key.endswith("_err") else "value"
        ap.add_argument(
            f"--{key.replace('_', '-')}",
            type=float,
            default=None,
            help=f"transit {quantity} {what} of a raw TIC target ({units[quantity]})",
        )
    ap.add_argument(
        "--h5",
        type=str,
        default=None,
        help="quicklook TLS .h5 file seeding a -tic ephemeris, or one "
        "more -toi companion; explicit --period etc. take priority",
    )


def _add_run_options(ap: ArgumentParser) -> None:
    ap.add_argument("-dir", type=str, default=None, help="base directory (default: .)")
    ap.add_argument(
        "-i", "--interactive", action="store_true", help="prompt for missing values"
    )
    ap.add_argument(
        "-u",
        "--update_db",
        action="store_true",
        help="refresh the cached TOI/CTOI/NExSci tables",
    )
    ap.add_argument("-o", "--overwrite", action="store_true", help="overwrite files")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--lc-only",
        action="store_true",
        help="only download and save the light curve(s)",
    )
    ap.add_argument(
        "--ttv",
        action="store_true",
        help="add <c>_ttv_transit_N rows (one per observed transit) and set fit_ttvs",
    )
    ap.add_argument(
        "--no-validate",
        action="store_true",
        help="skip 'jaxoplanet validate' of the new directory",
    )
    # allesfitter options jaxoplanet2 cannot honour: rejected with a clear message
    ap.add_argument("-bp", "--bandpass", nargs="+", default=None, help=SUPPRESS)
    ap.add_argument(
        "--stellar-var-gp-sho",
        "--stellar_var_gp_sho",
        dest="stellar_var_gp_sho",
        action="store_true",
        help=SUPPRESS,
    )


def build_parser() -> ArgumentParser:
    ap = ArgumentParser(
        prog="jaxoplanet init",
        description="Create a fit directory: template files, or a target's "
        "catalog-seeded params/settings and downloaded light curves.",
        epilog=EPILOG,
    )
    ap.add_argument("dir_path", nargs="?", default=None, metavar="DIR")
    _add_target_options(ap)
    _add_data_options(ap)
    _add_ephemeris_options(ap)
    _add_run_options(ap)
    return ap


def has_target(args: Namespace) -> bool:
    return any((args.toi, args.ctoi, args.tic, args.name))


def base_dir(args: Namespace) -> str:
    if args.dir_path and args.dir and args.dir_path != args.dir:
        raise PrepareError(f"DIR ({args.dir_path}) and -dir ({args.dir}) disagree")
    return args.dir_path or args.dir or "."


@dataclass(frozen=True)
class Plan:
    """Validated, resolved command-line choices."""

    mission: str
    insts: tuple[str, ...]
    pipelines: tuple[str, ...]  # one per downloaded instrument (insts[:n])
    exptimes: tuple[float | None, ...]
    segments: tuple[str, ...] | None  # None: most recent

    @property
    def downloads(self) -> tuple[tuple[str, str, float | None], ...]:
        return tuple(zip(self.insts, self.pipelines, self.exptimes, strict=False))


def _segments(args: Namespace, mission: str) -> tuple[str, ...] | None:
    if args.sector is not None:
        return tuple(str(s) for s in args.sector)
    other = {"k2": args.campaign, "kepler": args.quarter}.get(mission)
    if mission == "tess":
        if args.campaign is not None or args.quarter is not None:
            raise PrepareError("use -s/--sector for TESS")
        return None
    return (str(other),) if other is not None else ("-1",)


def _exptimes(args: Namespace, n: int) -> tuple[float | None, ...]:
    if args.exptime is None:
        return (None,) * n
    if len(args.exptime) == 1:
        return tuple(args.exptime) * n
    if len(args.exptime) != n:
        raise PrepareError(
            f"--exptime has {len(args.exptime)} entries but --pipeline has {n}; "
            "give one per --pipeline, or a single shared value"
        )
    return tuple(args.exptime)


def plan(args: Namespace) -> Plan:
    """Check option combinations prepare_allesfit accepts, and resolve them."""
    if args.bandpass is not None:
        raise PrepareError(
            "--bandpass (chromatic radius ratios) is not supported by jaxoplanet2; "
            "instruments share one <c>_radius_ratio"
        )
    if args.stellar_var_gp_sho:
        raise PrepareError(
            "--stellar-var-gp-sho is not supported by jaxoplanet2; use the default "
            "per-instrument GP baseline (baseline_flux_<inst>,sample_GP_SHO also works)"
        )
    mission = args.mission.lower()
    insts = tuple(args.filename) if args.filename else (mission,)
    pipelines = tuple(args.pipeline)
    if len(pipelines) > 1 and len(pipelines) != len(insts):
        raise PrepareError(
            f"--pipeline has {len(pipelines)} entries but --filename has {len(insts)}; "
            "give one --pipeline per --filename, or one to download only the first"
        )
    if args.h5 is not None and not (args.tic or args.toi):
        raise PrepareError("--h5 is only used together with -tic or -toi")
    segments = _segments(args, mission)
    if args.lc_only and segments is None:
        raise PrepareError("--lc-only needs -s/--sector (or -c/-q)")
    return Plan(mission, insts, pipelines, _exptimes(args, len(pipelines)), segments)
