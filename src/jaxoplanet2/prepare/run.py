"""``jaxoplanet init -toi/-ctoi/-tic/-name``: build a fit directory from catalogs.

The port of allesfitter's prepare_allesfit.py. It writes ``params.csv`` (native
transit parameters), ``settings.csv``, ``params_star.csv``, ``run.sh``,
``<inst>.csv`` light curves, plots and a log into ``<base>/<target>/``.
"""

import logging
import math
import sys
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from jaxoplanet2.model.ttv import observed_transits
from jaxoplanet2.prepare import catalogs, files, priors
from jaxoplanet2.prepare.catalogs import Target
from jaxoplanet2.prepare.errors import PrepareError
from jaxoplanet2.prepare.h5 import H5LightCurve, read_lightcurve, read_transit_params
from jaxoplanet2.prepare.lightcurves import Downloaded, Request, Table, segment_flag
from jaxoplanet2.prepare.options import Plan, base_dir, plan
from jaxoplanet2.prepare.priors import CompanionPriors, Star

log = logging.getLogger("jaxoplanet2.prepare")

Prompt = Callable[[str], str]
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
CANONICAL_OUTPUTS = ("params.csv", "settings.csv", "run.sh")
LD_BANDS = {"tess": "T", "k2": "Kp", "kepler": "Kp"}
LIMBDARK_HINT = "pip install git+https://github.com/john-livingston/limbdark.git"
FAST_FIT_WIDTH = 1.0 / 3.0  # d, as in the written settings.csv


class _StderrHandler(logging.StreamHandler):
    """Writes to the current ``sys.stderr``, which test runners may swap."""

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, _value) -> None:
        pass


def configure_logging(debug: bool) -> None:
    log.setLevel(logging.DEBUG if debug else logging.INFO)
    if not any(isinstance(h, _StderrHandler) for h in log.handlers):
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S"))
        log.addHandler(handler)
    log.propagate = False


def _log_to_file(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
    return handler


@dataclass(frozen=True)
class Seed:
    """What --h5 contributes: ephemeris values and possibly a light curve."""

    values: dict[str, float | None]
    lightcurve: H5LightCurve | None


def _cli_ephemeris(args: Namespace) -> dict[str, float | None]:
    return {key: getattr(args, key) for key in catalogs.EPHEMERIS_KEYS}


def _h5_seed(args: Namespace, mission: str) -> Seed | None:
    if args.h5 is None:
        return None
    values = read_transit_params(args.h5, mission=mission)
    lc = read_lightcurve(args.h5) if mission == "tess" else None
    if mission == "tess" and lc is None:
        log.info(f"{args.h5} has no raw light curve; downloading normally.")
    return Seed(values, lc)


def _target(args: Namespace, seed: Seed | None, prompt: Prompt) -> Target:
    cli = _cli_ephemeris(args)
    tic_ephemeris = None
    if args.tic:
        values = catalogs.merge_ephemeris_values(cli, seed.values) if seed else cli
        tic_ephemeris = catalogs.resolve_ephemeris(values, prompt)
    target = catalogs.find_target(
        toi=args.toi,
        ctoi=args.ctoi,
        tic=args.tic,
        name=args.name,
        update=args.update_db,
        tic_ephemeris=tic_ephemeris,
    )
    if seed is not None and args.toi:
        extra = catalogs.resolve_ephemeris(
            catalogs.merge_ephemeris_values(cli, seed.values), prompt
        )
        target = replace(target, planets=(*target.planets, extra))
    return target


def _check_tess_segments(target: Target, segments: tuple[str, ...] | None) -> Target:
    target = catalogs.with_coordinates(target)
    observed = catalogs.tess_sectors(target.tic_id, target.ra, target.dec)
    log.info(f"TESS sectors of {target.name}: {observed.tolist()}")
    if segment_flag(list(segments) if segments else None) != "multi_sector":
        return target
    missing = [
        s for s in segments if not s.lstrip("-").isdigit() or int(s) not in observed
    ]
    if missing:
        raise PrepareError(
            f"{target.name} was not observed in sector={missing}; "
            f"try sector={observed.tolist()}"
        )
    return target


def _outdir(base: str, target: Target, overwrite: bool) -> Path:
    outdir = Path(base, target.name)
    if outdir.is_dir():
        collisions = sorted({p.name for p in outdir.iterdir()} & set(CANONICAL_OUTPUTS))
        if collisions and not overwrite:
            raise PrepareError(
                f"{outdir} already contains {collisions}; use --overwrite"
            )
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


@dataclass(frozen=True)
class Asker:
    """Fills unknown values: prompted with --interactive, else a default or an error."""

    interactive: bool
    prompt: Prompt = input

    def number(self, label: str) -> float:
        if not self.interactive:
            raise PrepareError(f"{label} is unknown; use --interactive to enter it")
        return float(self.prompt(f"{label}: "))

    def fill(self, value: float, label: str, default: float | None = None) -> float:
        if math.isfinite(value):
            return value
        if default is not None and not self.interactive:
            return default
        return self.number(label)


def _star(target: Target, ask: Asker) -> tuple[Star, bool]:
    """Stellar parameters with gaps filled; and whether the host density is usable."""
    if target.source == "nexsci":
        star = catalogs.nexsci_star(target.nexsci_host)
    else:
        star = catalogs.tic_star(target.tic_id)
    density_prior = math.isfinite(star.radius) and math.isfinite(star.mass)
    star = Star(
        teff=ask.fill(star.teff, "Teff"),
        teff_err=ask.fill(star.teff_err, "Teff err", 500.0),
        logg=ask.fill(star.logg, "logg"),
        logg_err=ask.fill(star.logg_err, "logg err", 0.1),
        feh=ask.fill(star.feh, "[Fe/H]", 0.0),
        feh_err=ask.fill(star.feh_err, "[Fe/H] err", 0.1),
        radius=ask.fill(star.radius, "Rstar [Rsun]"),
        radius_err=_nan_to(star.radius_err, 0.1),
        mass=ask.fill(star.mass, "Mstar [Msun]"),
        mass_err=_nan_to(star.mass_err, 0.1),
    )
    log.info(
        f"Teff={star.teff:.0f}+/-{star.teff_err:.0f} K, logg={star.logg:.2f}+/-"
        f"{star.logg_err:.2f}, [Fe/H]={star.feh}+/-{star.feh_err}, "
        f"Rs={star.radius:.2f}+/-{star.radius_err:.2f}, "
        f"Ms={star.mass:.2f}+/-{star.mass_err:.2f}"
    )
    return star, density_prior


def _limb_darkening(star: Star, mission: str) -> files.LimbDarkening | None:
    """Claret quadratic coefficients in q space (commented rows); optional."""
    try:
        import limbdark
    except ImportError:
        log.warning(
            f"limbdark is not installed: no theoretical LD rows ({LIMBDARK_HINT})"
        )
        return None
    q1, q1_err, q2, q2_err = limbdark.claret(
        band=LD_BANDS[mission],
        teff=star.teff,
        uteff=star.teff_err,
        logg=star.logg,
        ulogg=star.logg_err,
        feh=star.feh,
        ufeh=star.feh_err,
        law="quadratic",
        transform=True,
    )
    return files.LimbDarkening(float(q1), float(q1_err), float(q2), float(q2_err))


def _radius_ratio(eph, i: int, target: Target, star: Star, ask: Asker):
    """(k, sigma_k) from the depth, else prompted, else from NExSci radii."""
    if eph.depth > 0:
        k, k_err = priors.radius_ratio_from_depth(eph.depth, _nan_to(eph.depth_err, 0.0))
        if math.isfinite(k):
            return k, k_err
    c = files.COMPANION_LETTERS[i]
    if not ask.interactive and i < len(target.planet_radii):
        if math.isfinite(target.planet_radii[i][0]):
            return catalogs.radius_ratio_from_radii(target.planet_radii[i], star.radius)
    k_ppt = ask.number(f"Planet {c} Rp/Rs (ppt)")
    return k_ppt / 1e3, ask.number(f"Planet {c} Rp/Rs err (ppt)") / 1e3


def _nan_to(x: float, default: float) -> float:
    return x if math.isfinite(x) else default


def _ephemeris(eph: priors.Ephemeris, c: str, ask: Asker) -> priors.Ephemeris:
    if eph.period > 0 and eph.epoch > 0:
        return eph
    if not ask.interactive:
        raise PrepareError(f"planet {c} has no period/epoch; use --interactive")
    return replace(
        eph,
        period=ask.number("Porb"),
        period_err=ask.number("Porb err"),
        epoch=ask.number("Epoch"),
        epoch_err=ask.number("Epoch err"),
    )


def _companions(
    target: Target, star: Star, ask: Asker, seed=0
) -> dict[str, CompanionPriors]:
    rng = np.random.default_rng(seed)
    out = {}
    for i, catalog_eph in enumerate(target.planets):
        c = files.COMPANION_LETTERS[i]
        eph = _ephemeris(catalog_eph, c, ask)
        k, k_err = _radius_ratio(eph, i, target, star, ask)
        p = priors.companion_priors(eph, k, k_err, star, rng=rng)
        log.info(
            f"{c}: P={p.period:.6f} d, T0={p.time_transit:.6f}, k={p.radius_ratio:.4f}, "
            f"T14={p.duration * 24:.2f} h ({target.source}), "
            f"{p.duration_from_density * 24:.2f} h from the stellar density at b=0.5"
        )
        out[c] = p
    return out


def _query_name(target: Target, args: Namespace) -> str:
    if target.tic_id is not None and (args.toi or args.ctoi or args.tic):
        return f"TIC {target.tic_id}"
    if args.name and args.name.lower().startswith("k2"):
        try:
            return catalogs.alias_with_prefix(args.name, "epic")
        except catalogs.CatalogError as e:
            log.info(str(e))
    return args.name


def _request(args: Namespace, p: Plan, inst: str, pipeline: str, exptime) -> Request:
    return Request(
        inst=inst,
        pipeline=pipeline,
        exptime=exptime,
        lc_type=args.lc_type,
        quality_bitmask=args.quality,
        sigma=args.sigma,
        mission=p.mission,
        segment_flag=segment_flag(list(p.segments) if p.segments else None),
        segments=p.segments or (),
    )


def _download_all(args, p: Plan, target, outdir, seed) -> dict[str, Downloaded]:
    from jaxoplanet2.prepare.products import download_instrument
    from jaxoplanet2.prepare.tars import patch_lightkurve

    patch_lightkurve()
    query = _query_name(target, args)
    h5lc = seed.lightcurve if seed else None
    return {
        inst: download_instrument(
            query, _request(args, p, inst, pipeline, exptime), outdir, target.name, h5lc
        )
        for inst, pipeline, exptime in p.downloads
    }


def _existing_table(outdir: Path, inst: str) -> Table | None:
    path = outdir / f"{inst}.csv"
    if not path.is_file():
        log.warning(
            f"{path} does not exist: add it before fitting (only "
            f"{inst!r}'s first --filename is downloaded with one --pipeline)"
        )
        return None
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    return Table(data[:, 0], data[:, 1], data[:, 2])


def _ttv_counts(companions, tables: list[Table]) -> dict[str, int]:
    time = np.sort(np.concatenate([t.time for t in tables]))
    counts = {}
    for c, p in companions.items():
        counts[c] = len(
            observed_transits(time, p.time_transit, p.period, FAST_FIT_WIDTH)
        )
        log.info(f"TTV: {c} has {counts[c]} transits with data")
    return counts


def _gp_bounds(companions, tables: dict[str, Table]) -> dict[str, priors.GpBounds]:
    protect = priors.gp_protective_duration(p.duration for p in companions.values())
    out = {}
    for inst, table in tables.items():
        bounds = priors.dataset_gp_bounds(table.time, table.flux, protect)
        if bounds is not None:
            log.info(
                f"{inst}: GP/noise priors from the data (RMS={bounds.rms:.2e}, "
                f"cadence={bounds.cadence * 86400:.0f} s, "
                f"baseline={bounds.baseline:.2f} d)"
            )
            out[inst] = bounds
    return out


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    log.info(f"Saved: {path}")
    return path


def _validate(outdir: Path) -> None:
    from jaxoplanet2.validate import validate

    report = validate(outdir)
    for line in report.warnings:
        log.warning(f"validate: {line}")
    for line in report.errors:
        log.error(f"validate: {line}")
    if report.ok:
        log.info(f"validate: OK. Next: 'jaxoplanet show-initial-guess {outdir}'")


def _lc_only(args: Namespace, p: Plan) -> list[Path]:
    from jaxoplanet2.prepare.products import download_lc_only
    from jaxoplanet2.prepare.tars import patch_lightkurve

    patch_lightkurve()
    if args.tic:
        query, label = f"TIC {args.tic}", f"TIC-{args.tic}"
    elif args.toi:
        query, label = f"TOI {args.toi}", f"TOI-{str(args.toi).zfill(4)}"
    elif args.ctoi:
        query, label = f"TIC {args.ctoi}", f"CTOI-{args.ctoi}"
    else:
        query, label = args.name, args.name.strip().replace(" ", "")
    return [
        download_lc_only(
            query,
            label,
            _request(args, p, p.insts[0], pipeline, exptime),
            Path(base_dir(args)),
        )
        for pipeline, exptime in zip(p.pipelines, p.exptimes, strict=True)
    ]


def run(args: Namespace, prompt: Prompt = input) -> Path | list[Path]:
    """Prepare the fit directory; returns it (``--lc-only``: the saved files)."""
    configure_logging(args.debug)
    p = plan(args)
    if args.lc_only:
        return _lc_only(args, p)
    seed = _h5_seed(args, p.mission)
    target = _target(args, seed, prompt)
    log.debug(f"target: {target}")
    if p.mission == "tess":
        target = _check_tess_segments(target, p.segments)
    outdir = _outdir(base_dir(args), target, args.overwrite)
    handler = _log_to_file(outdir / f"{target.name}.log")
    try:
        return _prepare(args, p, (target, outdir, seed), Asker(args.interactive, prompt))
    finally:
        log.removeHandler(handler)
        handler.close()


def _prepare(args, p: Plan, where: tuple[Target, Path, Seed | None], ask: Asker) -> Path:
    target, outdir, seed = where
    star, density_prior = _star(target, ask)
    ld = _limb_darkening(star, p.mission)
    companions = _companions(target, star, ask)
    downloads = _download_all(args, p, target, outdir, seed)
    tables = {inst: d.table for inst, d in downloads.items()}
    for inst in p.insts:
        if inst not in tables and (table := _existing_table(outdir, inst)) is not None:
            tables[inst] = table
    ttv = _ttv_counts(companions, list(tables.values())) if args.ttv else None
    params = files.params_csv(
        companions,
        p.insts,
        ld=ld,
        gp_bounds=_gp_bounds(companions, tables),
        dilution={i: d.dilution for i, d in downloads.items() if d.dilution},
        ttv_counts=ttv,
    )
    t_exp = {inst: d.exptime for inst, d in downloads.items()}
    settings = files.settings_csv(
        tuple(companions), p.insts, t_exp, density_prior, fit_ttvs=bool(args.ttv)
    )
    _write(outdir / "params.csv", params)
    _write(outdir / "settings.csv", settings)
    _write(outdir / "params_star.csv", files.params_star_csv(star))
    _write(outdir / "run.sh", files.RUN_SH).chmod(0o755)
    if not args.no_validate:
        _validate(outdir)
    return outdir
