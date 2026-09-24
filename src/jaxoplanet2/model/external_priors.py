"""allesfitter's "external priors": terms that are not per-parameter priors.

- Host density: ``params_star.csv`` (R*, M*) implies a stellar density; every
  transiting companion implies one too, via Kepler's law and ``rsuma``. With
  ``use_host_density_prior`` the two are tied by a normal prior.
- Physical limits: ``e < 1``, no collision (``e < 1 - rsuma``), ``dil < 0.999``.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from jaxoplanet2.io.settings import Settings
from jaxoplanet2.model.parameterization import companion_geometry, host_density_cgs

Values = Mapping[str, jax.Array | float]
PARAMS_STAR_FILE = "params_star.csv"
R_SUN_CM = 6.957e10
M_SUN_G = 1.9884754153381438e33
G_CGS = 6.6743e-8
SECONDS_PER_DAY = 86400.0
KM_TO_CM = 1e5
MAX_DILUTION = 0.999
DENSITY_SAMPLES = 100_000
MASS_ITERATIONS = 30
# photometry-only fits neglect the companion's density if rr^3 < this
NEGLIGIBLE_RR_CUBED = 0.01


@dataclass(frozen=True)
class Star:
    radius: float  # R_sun
    radius_err: tuple[float, float]
    mass: float  # M_sun
    mass_err: tuple[float, float]
    teff: float | None = None  # K
    teff_err: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class DensityPrior:
    mean: float  # g/cm^3
    sd: float


def load_star(fit_dir: str | Path) -> Star | None:
    path = Path(fit_dir) / PARAMS_STAR_FILE
    if not path.is_file():
        return None
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    header = [c.strip() for c in lines[0].lstrip("#").split(",")]
    rows = [ln for ln in lines[1:] if not ln.startswith("#")]
    if not rows:
        raise ValueError(f"{PARAMS_STAR_FILE} has no data row")
    row = dict(zip(header, (c.strip() for c in rows[0].split(",")), strict=False))
    try:
        return Star(
            radius=float(row["R_star"]),
            radius_err=(float(row["R_star_lerr"]), float(row["R_star_uerr"])),
            mass=float(row["M_star"]),
            mass_err=(float(row["M_star_lerr"]), float(row["M_star_uerr"])),
            **_teff(row),
        )
    except (KeyError, ValueError) as e:
        raise ValueError(f"{PARAMS_STAR_FILE}: cannot read {e}") from e


def _teff(row: dict[str, str]) -> dict:
    if not row.get("Teff_star"):
        return {}
    errs = (row.get("Teff_star_lerr") or "0", row.get("Teff_star_uerr") or "0")
    return {"teff": float(row["Teff_star"]), "teff_err": tuple(map(float, errs))}


def split_normal(median: float, err: tuple[float, float], rng, size: int):
    # allesfitter fits a skew normal; for symmetric errors (69 of 71 local fits)
    # that is exactly this normal, and a split normal is close otherwise
    lo, hi = abs(err[0]), abs(err[1])
    z = rng.standard_normal(size)
    return median + np.where(z < 0, lo, hi) * z


def density_prior(star: Star, seed: int = 0) -> DensityPrior:
    """allesfitter's host-density prior: MC over R* and M*, normal in rho."""
    rng = np.random.default_rng(seed)
    radius = split_normal(star.radius, star.radius_err, rng, DENSITY_SAMPLES)
    mass = split_normal(star.mass, star.mass_err, rng, DENSITY_SAMPLES)
    rho = mass * M_SUN_G / (4.0 / 3.0 * math.pi * (radius * R_SUN_CM) ** 3)
    p16, p50, p84 = np.percentile(rho, [16, 50, 84])
    return DensityPrior(mean=float(p50), sd=float(max(p50 - p16, p84 - p50)))


def companion_mass_g(K_kms, period_d, inclination, ecc, host_mass_msun) -> jax.Array:
    """Companion mass from the RV semi-amplitude (exact mass function).

    Solved in solar masses: in grams (M + m)^2 ~ 1e66 overflows float32.
    """
    f_grams = (
        (period_d * SECONDS_PER_DAY) * (K_kms * KM_TO_CM) ** 3 / (2 * math.pi * G_CGS)
    )
    f = f_grams * (1 - ecc**2) ** 1.5 / M_SUN_G
    m = jnp.zeros_like(jnp.asarray(K_kms, dtype=float))
    for _ in range(MASS_ITERATIONS):  # m^3 sin^3 i = f (M + m)^2
        m = jnp.cbrt(f * (host_mass_msun + m) ** 2) / jnp.sin(inclination)
    return m * M_SUN_G


def implied_host_density(
    values: Values, star: Star, c: str
) -> tuple[jax.Array, jax.Array]:
    """(host density from companion ``c``'s orbit, whether it is defined).

    allesfitter's calc_rho_host: with an RV mass the companion's own density is
    subtracted; photometry-only fits neglect it if rr^3 < 0.01, else no prior.
    Branches use ``jnp.where`` so this works on traced values.
    """
    g = companion_geometry(values, c)
    rho_kepler = host_density_cgs(values, c)
    with_mass = (g.K > 0) & (g.rr > 0)
    safe_rr = jnp.where(g.rr > 0, g.rr, 1.0)
    safe_K = jnp.where(with_mass, g.K, 1.0)  # keeps cbrt's gradient finite at K=0
    mass = companion_mass_g(safe_K, g.period, g.inclination, g.eccentricity, star.mass)
    rho_comp = mass / (4.0 / 3.0 * math.pi * (safe_rr * star.radius * R_SUN_CM) ** 3)
    rho = jnp.where(with_mass, rho_kepler - g.rr**3 * rho_comp, rho_kepler)
    defined = with_mass | ((g.rr > 0) & (g.rr**3 < NEGLIGIBLE_RR_CUBED))
    return rho, defined


def external_log_prior(
    values: Values,
    settings: Settings,
    prior: DensityPrior | None,
    star: Star | None,
) -> jax.Array:
    lp = jnp.asarray(0.0)
    if prior is not None and star is not None and settings.use_host_density_prior:
        for c in settings.companions_phot:
            rho, defined = implied_host_density(values, star, c)
            z = (rho - prior.mean) / prior.sd
            term = norm.logpdf(0.0) - math.log(prior.sd) - 0.5 * z**2
            lp = lp + jnp.where(defined, term, 0.0)
    for c in settings.companions_all:
        g = companion_geometry(values, c)
        allowed = g.eccentricity < 1.0
        if f"{c}_rsuma" in values:
            allowed = allowed & (g.eccentricity < 1.0 - values[f"{c}_rsuma"])
        lp = jnp.where(allowed, lp, -jnp.inf)
    for inst in settings.inst_phot:
        lp = jnp.where(values.get(f"dil_{inst}", 0.0) > MAX_DILUTION, -jnp.inf, lp)
    return lp
