"""Derived parameters from posterior samples, following allesfitter's deriver.

Geometry (always): R*/a, a/R*, Rp/a, i, e, w, the transit impact parameter and
total/full durations (Winn 2010, eqs. 7, 14, 16) and the host density implied by
the orbit. With params_star.csv also Rp, a, Teq (albedo 0.3, emissivity 1, as
allesfitter assumes) and, with an RV semi-amplitude, the companion mass. Stellar
parameters are drawn per sample, so their uncertainties propagate.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from jaxoplanet2.io.settings import Settings
from jaxoplanet2.model.external_priors import Star, companion_mass_g, split_normal
from jaxoplanet2.model.parameterization import G_CGS, SECONDS_PER_DAY

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


def _duration(*, period, a_over_r, k, b, sin_i, ecc_factor, sign) -> np.ndarray:
    """Winn (2010) eq. 14 (sign=+1, total) or 16 (sign=-1, full), in hours."""
    reach = 1 + sign * k
    arg = np.sqrt(np.clip(reach**2 - b**2, 0, None)) / (a_over_r * sin_i)
    t = period / math.pi * np.arcsin(np.clip(arg, -1, 1)) * ecc_factor * HOURS
    return np.where(reach > np.abs(b), t, np.nan)


def _geometry(get: _Draws) -> dict[str, Derived]:
    c = get.c
    rr, rsuma, cosi = get("rr", 0.0), get("rsuma", np.nan), get("cosi", 0.0)
    period, f_c, f_s = get("period", np.nan), get("f_c", 0.0), get("f_s", 0.0)
    ecc = f_c**2 + f_s**2
    w = np.where(ecc > 0, np.arctan2(f_s, f_c), math.pi / 2)
    a_over_r = (1 + rr) / rsuma
    b = a_over_r * cosi * (1 - ecc**2) / (1 + ecc * np.sin(w))
    shape = {
        "period": period,
        "a_over_r": a_over_r,
        "k": rr,
        "b": b,
        "sin_i": np.sqrt(1 - cosi**2),
        "ecc_factor": np.sqrt(1 - ecc**2) / (1 + ecc * np.sin(w)),
    }
    rho = 3 * math.pi * a_over_r**3 / (G_CGS * (period * SECONDS_PER_DAY) ** 2)
    return {
        f"{c}_R_star/a": Derived(f"$R_\\star/a_\\mathrm{{{c}}}$", "", 1 / a_over_r),
        f"{c}_a/R_star": Derived(f"$a_\\mathrm{{{c}}}/R_\\star$", "", a_over_r),
        f"{c}_R_companion/a": Derived(_tex("R", c) + "/a", "", rr / a_over_r),
        f"{c}_i": Derived(_tex("i", c), "deg", np.degrees(np.arccos(cosi))),
        f"{c}_e": Derived(_tex("e", c), "", ecc),
        f"{c}_w": Derived(_tex("w", c), "deg", np.degrees(w)),
        f"{c}_b_tra": Derived(_tex("b", f"tra;{c}"), "", b),
        f"{c}_T_tra_tot": Derived(
            _tex("T", f"tot;{c}"), "h", _duration(**shape, sign=+1)
        ),
        f"{c}_T_tra_full": Derived(
            _tex("T", f"full;{c}"), "h", _duration(**shape, sign=-1)
        ),
        f"{c}_host_density": Derived(_tex("\\rho", f"\\star;{c}"), "cgs", rho),
    }


def _physical(get: _Draws, geo, star: Star, rng, has_rv: bool) -> dict[str, Derived]:
    c, n = get.c, get.n
    r_star = split_normal(star.radius, star.radius_err, rng, n)
    m_star = split_normal(star.mass, star.mass_err, rng, n)
    radius = get("rr", 0.0) * r_star
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
