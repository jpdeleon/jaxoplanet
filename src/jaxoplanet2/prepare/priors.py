"""Initial values and priors for a new fit, from catalog values and the data.

Pure numpy: everything here is deterministic given its inputs (and ``rng``), so
the network-facing parts of ``jaxoplanet init`` stay thin. The transit priors
are for jaxoplanet's native parameters (radius_ratio, duration, impact_param,
time_transit, period); the catalog T14 seeds ``duration`` directly, so unlike
allesfitter's (rsuma, cosi) no impact-parameter assumption leaks into it.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from jaxoplanet2.io.convert import DURATION_PRIOR_FACTOR
from jaxoplanet2.prepare.errors import PrepareError

N_SAMPLES = 10_000
QUANTILES_3SIG = (0.135, 50.0, 99.865)
R_SUN_CM = 6.957e10
M_SUN_G = 1.9884754153381438e33
G_CGS = 6.6743e-8
SECONDS_PER_DAY = 86400.0
HOURS_PER_DAY = 24.0
MAX_RADIUS_RATIO = 0.5  # brown-dwarf regime
INITIAL_IMPACT_PARAM = 0.5  # median of the transit-conditioned b ~ U(0, 1)
FALLBACK_DURATION_ERR_HOURS = 1.0
FALLBACK_EPOCH_ERR_DAYS = 0.1
FALLBACK_PERIOD_REL_ERR = 1e-3
# ln sigma_flux upper bound: exp(-3) ~ 5% relative scatter
DEFAULT_LN_ERR = (-6.0, -10.0, -3.0)
DEFAULT_GP_LNSIGMA = (-5.0, -10.0, -3.0)
# GP ln rho upper bound: exp(5) ~ 148 d, beyond any TESS/Kepler systematic
DEFAULT_GP_LNRHO = (0.0, -1.0, 5.0)
MIN_GP_POINTS = 20


class PriorError(PrepareError):
    """Catalog values cannot seed a physical transit."""


@dataclass(frozen=True)
class Ephemeris:
    """One companion as catalogs report it (NaN where unknown)."""

    period: float  # d
    period_err: float
    epoch: float  # BJD
    epoch_err: float
    duration: float  # h
    duration_err: float
    depth: float  # ppm
    depth_err: float


@dataclass(frozen=True)
class Star:
    teff: float  # K
    teff_err: float
    logg: float  # cgs
    logg_err: float
    feh: float  # dex
    feh_err: float
    radius: float  # R_sun
    radius_err: float
    mass: float  # M_sun
    mass_err: float


@dataclass(frozen=True)
class CompanionPriors:
    """Initial values and prior bounds of one companion's native parameters."""

    radius_ratio: float
    radius_ratio_upper: float
    duration: float  # d
    duration_bounds: tuple[float, float]
    impact_param: float
    impact_param_upper: float
    time_transit: float
    time_transit_err: float
    period: float
    period_err: float
    duration_from_density: float  # d; NaN without a usable stellar density


@dataclass(frozen=True)
class GpBounds:
    """(initial, lower, upper) of each per-instrument noise/GP log-parameter."""

    ln_err: tuple[float, float, float]
    lnsigma: tuple[float, float, float]
    lnrho: tuple[float, float, float]
    rms: float = math.nan
    cadence: float = math.nan  # d
    baseline: float = math.nan  # d


DEFAULT_GP_BOUNDS = GpBounds(DEFAULT_LN_ERR, DEFAULT_GP_LNSIGMA, DEFAULT_GP_LNRHO)


def _finite_positive(x: float) -> bool:
    return x is not None and math.isfinite(x) and x > 0


def _error_or(value: float, fallback: float) -> float:
    return value if _finite_positive(value) else fallback


def stellar_density(mass, radius):
    """Mean density in g/cm^3 of a sphere (M in M_sun, R in R_sun)."""
    volume = 4.0 / 3.0 * np.pi * (np.asarray(radius) * R_SUN_CM) ** 3
    return np.asarray(mass) * M_SUN_G / volume


def a_over_rstar(density, period):
    """Kepler's law: a/R* from the stellar density (g/cm^3) and period (d)."""
    p_sec = np.asarray(period) * SECONDS_PER_DAY
    return (G_CGS * np.asarray(density) * p_sec**2 / (3.0 * np.pi)) ** (1.0 / 3.0)


def circular_duration(period, a_rstar, radius_ratio, impact_param):
    """T14 (d) of a circular orbit (Winn 2010 eq. 14); NaN if it does not transit."""
    a = np.asarray(a_rstar, dtype=float)
    b = np.asarray(impact_param, dtype=float)
    chord2 = (1.0 + np.asarray(radius_ratio)) ** 2 - b**2
    sin_i = np.sqrt(np.clip(1.0 - (b / a) ** 2, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        arg = np.sqrt(np.where(chord2 > 0, chord2, np.nan)) / (a * sin_i)
        arg = np.where(arg <= 1.0, arg, np.nan)
        return np.asarray(period) / np.pi * np.arcsin(arg)


def radius_ratio_upper(rr_max: float) -> float:
    """3-sigma radius ratio rounded up to 0.1, +0.05 headroom, capped at 0.5."""
    return min(MAX_RADIUS_RATIO, math.ceil(rr_max * 10) / 10 + 0.05)


def _round(x: float) -> float:
    return float(f"{x:.4g}")


def duration_bounds(duration: float, period: float) -> tuple[float, float]:
    """uniform [T/3, 3T] like convert-params, kept below P/2 so it inverts."""
    upper = min(duration * DURATION_PRIOR_FACTOR, 0.5 * period)
    return _round(duration / DURATION_PRIOR_FACTOR), _round(upper)


def companion_priors(
    eph: Ephemeris,
    radius_ratio: float,
    radius_ratio_err: float,
    star: Star,
    rng: np.random.Generator | None = None,
) -> CompanionPriors:
    """Native-parameter priors of one companion.

    The radius-ratio upper bound comes from its 3-sigma spread. The duration is
    the catalog T14 when there is one, else the T14 a circular orbit at b=0.5
    has around a star of the catalog density.
    """
    rng = np.random.default_rng() if rng is None else rng
    if not (_finite_positive(eph.period) and _finite_positive(eph.epoch)):
        raise PriorError(f"period={eph.period}, epoch={eph.epoch} must be positive")
    if not _finite_positive(radius_ratio):
        raise PriorError(f"radius ratio {radius_ratio} must be positive")
    rr_err = _error_or(radius_ratio_err, 0.0)
    rr_samples = rng.normal(radius_ratio, rr_err, N_SAMPLES)
    rr_max = float(np.percentile(rr_samples, QUANTILES_3SIG[2]))
    rr_upper = radius_ratio_upper(rr_max)

    density = stellar_density(star.mass, star.radius)
    t14_density = float(
        circular_duration(
            eph.period,
            a_over_rstar(density, eph.period),
            radius_ratio,
            INITIAL_IMPACT_PARAM,
        )
    )
    duration = eph.duration / HOURS_PER_DAY
    if not _finite_positive(duration):
        duration = t14_density
    if not _finite_positive(duration):
        raise PriorError(
            "no catalog duration, and the stellar density gives no transit; "
            "pass --duration"
        )
    return CompanionPriors(
        radius_ratio=radius_ratio,
        radius_ratio_upper=rr_upper,
        duration=duration,
        duration_bounds=duration_bounds(duration, eph.period),
        impact_param=INITIAL_IMPACT_PARAM,
        impact_param_upper=_round(1.0 + rr_upper),
        time_transit=eph.epoch,
        time_transit_err=_error_or(eph.epoch_err, FALLBACK_EPOCH_ERR_DAYS),
        period=eph.period,
        period_err=_error_or(eph.period_err, FALLBACK_PERIOD_REL_ERR * eph.period),
        duration_from_density=t14_density,
    )


def radius_ratio_from_depth(depth_ppm: float, depth_err_ppm: float):
    """(k, sigma_k) from a transit depth, as prepare_allesfit computes them.

    sigma_k = sqrt(sigma_depth) deliberately overestimates the error: it only
    sets how far above k the uniform radius-ratio prior reaches.
    """
    return math.sqrt(depth_ppm / 1e6), math.sqrt(depth_err_ppm / 1e6)


def gp_protective_duration(durations_days: Iterable[float], fallback=0.1) -> float:
    """Longest finite positive duration (d): a shared GP must spare every transit."""
    d = np.asarray(list(durations_days), dtype=float)
    valid = d[np.isfinite(d) & (d > 0)]
    return float(valid.max()) if valid.size else float(fallback)


def dataset_gp_bounds(time, flux, duration_days: float) -> GpBounds | None:
    """Noise and Matern-3/2 GP priors grounded in a light curve.

    - ``ln_err_flux``: centred on the point-to-point RMS, never above 10%;
    - GP ``ln sigma``: from RMS/10 (weaker is useless) to min(5%, 100 RMS);
    - GP ``ln rho``: above 2 cadences and T14/2 (so the GP cannot fit a
      transit), below the observing baseline.

    ``None`` when there are too few points to estimate the noise.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f)
    if int(ok.sum()) < MIN_GP_POINTS:
        return None
    order = np.argsort(t[ok])
    t, f = t[ok][order], f[ok][order]
    diff = np.diff(f)
    rms = 1.4826 * float(np.median(np.abs(diff - np.median(diff)))) / math.sqrt(2.0)
    if not _finite_positive(rms):
        rms = float(np.nanstd(f))
    rms = max(rms, 1e-6)
    cadence = float(np.median(np.diff(t)))
    baseline = float(t[-1] - t[0])
    duration_days = max(float(duration_days), 1.0 / HOURS_PER_DAY)

    ln_err = math.log(rms)
    err_lo = ln_err - 3.0
    err_hi = min(ln_err + 2.0, math.log(0.10))
    if err_hi <= err_lo:
        err_hi = err_lo + 1.0

    sigma_lo_phys = max(rms / 10.0, 1e-7)
    sig_lo = math.log(sigma_lo_phys)
    sig_hi = math.log(min(0.05, 100.0 * rms))
    if sig_hi <= sig_lo:
        sig_hi = sig_lo + 2.0
    sig = math.log(max(3.0 * rms, 2.0 * sigma_lo_phys))
    sig = min(max(sig, sig_lo + 0.1), sig_hi - 0.1)

    # 2% margins keep the bounds inside the limits after rounding to 3 decimals
    rho_lo_phys = 1.02 * max(2.0 * cadence, 0.5 * duration_days)
    rho_hi_phys = 0.98 * baseline
    if rho_hi_phys <= rho_lo_phys:
        rho_hi_phys = 4.0 * rho_lo_phys
    rho_lo, rho_hi = math.log(rho_lo_phys), math.log(rho_hi_phys)
    return GpBounds(
        ln_err=(ln_err, err_lo, err_hi),
        lnsigma=(sig, sig_lo, sig_hi),
        lnrho=(0.5 * (rho_lo + rho_hi), rho_lo, rho_hi),
        rms=rms,
        cadence=cadence,
        baseline=baseline,
    )
