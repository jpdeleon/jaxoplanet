"""Derived parameters from posterior samples, following allesfitter's deriver.

Geometry (always): R*/a, a/R*, Rp/a, allesfitter's rsuma and cos i, i, e, w,
the total/full durations (T14 is sampled; T23 from Winn 2010 eq. 16) and the
host density implied by the orbit. With params_star.csv also Rp, a, Teq
(albedo 0.3, emissivity 1, as allesfitter assumes) and, with an RV
semi-amplitude, the companion mass. Stellar parameters are drawn per sample, so
their uncertainties propagate.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import numpy as np

from jaxoplanet2.io.settings import Settings
from jaxoplanet2.model.external_priors import Star, companion_mass_g, split_normal
from jaxoplanet2.model.parameterization import (
    G_CGS,
    SECONDS_PER_DAY,
    companion_geometry,
    eccentricity_factors,
)

R_EARTH_PER_R_SUN = 109.076
R_JUP_PER_R_SUN = 9.731
AU_PER_R_SUN = 1.0 / 215.032
M_EARTH_G = 5.9722e27
M_JUP_G = 1.89813e30
ALBEDO, EMISSIVITY = 0.3, 1.0
HOURS = 24.0


@dataclass(frozen=True)
class Derived:
    label: str
    unit: str
    values: np.ndarray


def summarize(x: np.ndarray) -> tuple[float, float, float]:
    """(median, lower 1-sigma error, upper 1-sigma error) from percentiles."""
    p16, p50, p84 = np.nanpercentile(x, [16, 50, 84])
    return float(p50), float(p50 - p16), float(p84 - p50)


def _tex(symbol: str, sub: str) -> str:
    return f"${symbol}_\\mathrm{{{sub}}}$"


class _Draws:
    """Flat posterior draws with allesfitter's defaults for absent parameters."""

    def __init__(self, samples: Mapping[str, np.ndarray], c: str):
        self.samples, self.c = samples, c
        self.n = len(next(iter(samples.values())))

    def __call__(self, name: str, default: float) -> np.ndarray:
        key = f"{self.c}_{name}"
        if key in self.samples:
            return np.asarray(self.samples[key], dtype=float)
        return np.full(self.n, default)

    def geometry(self):
        names = ("radius_ratio", "duration", "impact_param", "time_transit",
                 "period", "f_c", "f_s", "K")  # fmt: skip
        values = {
            f"{self.c}_{n}": self.samples[f"{self.c}_{n}"]
            for n in names
            if f"{self.c}_{n}" in self.samples
        }
        return companion_geometry(values, self.c)


def _full_duration(g) -> np.ndarray:
    """T23 (Winn 2010 eq. 16), in hours; NaN for grazing transits."""
    _, t_factor = eccentricity_factors(g.eccentricity, g.sin_omega)
    reach = 1.0 - g.rr
    arg = np.sqrt(np.clip(reach**2 - g.impact_param**2, 0, None)) / (
        g.a_over_rstar * np.sin(g.inclination)
    )
    t = g.period / math.pi * np.arcsin(np.clip(arg, -1, 1)) * t_factor * HOURS
    return np.where(reach > np.abs(g.impact_param), t, np.nan)


def _geometry(get: _Draws) -> dict[str, Derived]:
    c = get.c
    g = jax.tree_util.tree_map(np.asarray, get.geometry())
    omega = np.arctan2(g.sin_omega, g.cos_omega)
    rho = 3 * math.pi * g.a_over_rstar**3 / (G_CGS * (g.period * SECONDS_PER_DAY) ** 2)
    return {
        f"{c}_R_star/a": Derived(f"$R_\\star/a_\\mathrm{{{c}}}$", "", g.radius_1),
        f"{c}_a/R_star": Derived(f"$a_\\mathrm{{{c}}}/R_\\star$", "", g.a_over_rstar),
        f"{c}_R_companion/a": Derived(_tex("R", c) + "/a", "", g.rr / g.a_over_rstar),
        f"{c}_rsuma": Derived(f"$(R_\\star + R_\\mathrm{{{c}}})/a$", "", g.rsuma),
        f"{c}_cosi": Derived(f"$\\cos i_\\mathrm{{{c}}}$", "", np.cos(g.inclination)),
        f"{c}_i": Derived(_tex("i", c), "deg", np.degrees(g.inclination)),
        f"{c}_e": Derived(_tex("e", c), "", g.eccentricity),
        f"{c}_w": Derived(_tex("w", c), "deg", np.degrees(omega)),
        f"{c}_T_tra_tot": Derived(_tex("T", f"tot;{c}"), "h", g.duration * HOURS),
        f"{c}_T_tra_full": Derived(_tex("T", f"full;{c}"), "h", _full_duration(g)),
        f"{c}_host_density": Derived(_tex("\\rho", f"\\star;{c}"), "cgs", rho),
    }


def _physical(get: _Draws, geo, star: Star, rng, has_rv: bool) -> dict[str, Derived]:
    c, n = get.c, get.n
    r_star = split_normal(star.radius, star.radius_err, rng, n)
    m_star = split_normal(star.mass, star.mass_err, rng, n)
    radius = get("radius_ratio", 0.0) * r_star
    a_over_r = geo[f"{c}_a/R_star"].values
    out = {
        f"{c}_R_companion_earth": Derived(
            _tex("R", c), "R_earth", radius * R_EARTH_PER_R_SUN
        ),
        f"{c}_R_companion_jup": Derived(_tex("R", c), "R_jup", radius * R_JUP_PER_R_SUN),
        f"{c}_a_au": Derived(_tex("a", c), "au", a_over_r * r_star * AU_PER_R_SUN),
    }
    if star.teff is not None:
        teff = split_normal(star.teff, star.teff_err, rng, n)
        teq = teff * ((1 - ALBEDO) / EMISSIVITY) ** 0.25 / np.sqrt(2 * a_over_r)
        out[f"{c}_Teq"] = Derived(_tex("T", f"eq;{c}"), "K", teq)
    if has_rv:
        mass = np.asarray(
            companion_mass_g(
                get("K", 0.0),
                get("period", np.nan),
                np.radians(geo[f"{c}_i"].values),
                geo[f"{c}_e"].values,
                m_star,
            )
        )
        label = _tex("M", c)
        out[f"{c}_M_companion_earth"] = Derived(label, "M_earth", mass / M_EARTH_G)
        out[f"{c}_M_companion_jup"] = Derived(label, "M_jup", mass / M_JUP_G)
    return out


def derive(
    samples: Mapping[str, np.ndarray],
    settings: Settings,
    star: Star | None,
    seed: int = 0,
) -> dict[str, Derived]:
    """Derived parameters for every companion; ``samples`` are flat arrays."""
    rng = np.random.default_rng(seed)
    out: dict[str, Derived] = {}
    for c in settings.companions_all:
        get = _Draws(samples, c)
        geo = _geometry(get)
        out.update(geo)
        if star is not None:
            out.update(_physical(get, geo, star, rng, c in settings.companions_rv))
    return out
