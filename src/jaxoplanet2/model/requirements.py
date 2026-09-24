"""Which params.csv entries a given settings.csv needs, uses, or cannot honour.

This is the single source of truth for parameter names: the model reads exactly
these, and ``validate`` checks a fit directory against them.
"""

import re
from dataclasses import dataclass

from jaxoplanet2.io.params import ParamTable
from jaxoplanet2.io.settings import Settings

# baseline type -> (required, optional) parameter prefixes; the full name is
# f"{prefix}_{kind}_{inst}", e.g. baseline_offset_flux_tess
BASELINE_PARAMS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "none": ((), ()),
    "hybrid_offset": ((), ()),
    "sample_offset": (("baseline_offset",), ()),
    "sample_linear": (("baseline_offset", "baseline_slope"), ()),
    "sample_gp_matern32": (
        ("baseline_gp_matern32_lnsigma", "baseline_gp_matern32_lnrho"),
        ("baseline_gp_offset",),
    ),
    "sample_gp_sho": (
        ("baseline_gp_sho_lnS0", "baseline_gp_sho_lnQ", "baseline_gp_sho_lnomega0"),
        ("baseline_gp_offset",),
    ),
    "sample_gp_real": (
        ("baseline_gp_real_lna", "baseline_gp_real_lnc"),
        ("baseline_gp_offset",),
    ),
    "sample_gp_complex": (
        (
            "baseline_gp_complex_lna",
            "baseline_gp_complex_lnb",
            "baseline_gp_complex_lnc",
            "baseline_gp_complex_lnd",
        ),
        ("baseline_gp_offset",),
    ),
}

# ellc/allesfitter parameters for physics jaxoplanet2 does not model. They are
# harmless while fixed at a neutral value and unsupported otherwise; the value
# None means "harmless at any fixed value" (e.g. gravity darkening of a sphere).
NEUTRAL_EXTRAS: tuple[tuple[re.Pattern[str], float | None], ...] = (
    (re.compile(r"[^_]+_sbratio_.+"), 0.0),
    (re.compile(r"[^_]+_geom_albedo_.+"), 0.0),
    (re.compile(r"[^_]+_phase_curve_.+"), 0.0),
    (re.compile(r"[^_]+_gdc_.+"), None),
    # dilution of an instrument that is not photometric (a leftover row)
    (re.compile(r"dil_.+"), 0.0),
)
TTV_PARAM = re.compile(r"[^_]+_ttv_transit_\d+")


def baseline_params(kind: str, inst: str, baseline: str) -> tuple[list, list]:
    key = baseline.lower()
    if key.startswith("hybrid_poly_"):
        key = "hybrid_offset"  # analytic, no sampled parameters
    required, optional = BASELINE_PARAMS[key]
    return (
        [f"{p}_{kind}_{inst}" for p in required],
        [f"{p}_{kind}_{inst}" for p in optional],
    )


def _ld_params(settings: Settings, inst: str) -> list[str]:
    law = settings.ld_law[inst]
    if law is None:
        return []
    space = settings.ld_space[inst]
    if law == "lin":
        return [f"host_ldc_{space}1_{inst}"]
    return [f"host_ldc_{space}{n}_{inst}" for n in (1, 2)]


def required_params(settings: Settings) -> dict[str, str]:
    """Parameter name -> why it is needed."""
    req: dict[str, str] = {}
    for c in settings.companions_phot:
        for p in ("rr", "rsuma", "cosi", "epoch", "period"):
            req[f"{c}_{p}"] = f"transit model of companion {c}"
    for c in settings.companions_rv:
        for p in ("epoch", "period", "K"):
            req[f"{c}_{p}"] = f"RV model of companion {c}"
    for inst in settings.inst_phot:
        for name in _ld_params(settings, inst):
            req[name] = f"host_ld_law_{inst}={settings.ld_law[inst]}"
        req[f"ln_err_flux_{inst}"] = f"error_flux_{inst}=sample"
    for inst in settings.inst_rv:
        req[f"ln_jitter_rv_{inst}"] = f"error_rv_{inst}=sample"
    for (kind, inst), baseline in settings.baseline.items():
        for name in baseline_params(kind, inst, baseline)[0]:
            req[name] = f"baseline_{kind}_{inst}={baseline}"
    return req


def optional_params(settings: Settings) -> set[str]:
    names = set()
    for c in settings.companions_all:
        names |= {f"{c}_{p}" for p in ("f_c", "f_s", "K", "rr", "rsuma", "cosi")}
    names |= {f"dil_{inst}" for inst in settings.inst_phot}
    for (kind, inst), baseline in settings.baseline.items():
        names |= set(baseline_params(kind, inst, baseline)[1])
    return names


@dataclass(frozen=True)
class ParamCheck:
    missing: tuple[str, ...]
    unsupported: tuple[str, ...]  # present, but jaxoplanet2 cannot honour them
    ignored: tuple[str, ...]  # present and harmless, but not used


def _neutral_extra(name: str, value: float, fit: bool) -> bool | None:
    """True: harmless extra; False: unsupported extra; None: not an extra."""
    for pattern, neutral in NEUTRAL_EXTRAS:
        if pattern.fullmatch(name):
            return not fit and (neutral is None or value == neutral)
    return None


def check_params(settings: Settings, params: ParamTable) -> ParamCheck:
    required = required_params(settings)
    known = set(required) | optional_params(settings)
    missing = tuple(name for name in required if name not in params)
    unsupported, ignored = [], []
    for p in params:
        if p.name in known:
            continue
        if TTV_PARAM.fullmatch(p.name):
            # with fit_ttvs, load_fit_directory already matched them to transits
            if not settings.fit_ttvs:
                ignored.append(p.name)
            continue
        verdict = _neutral_extra(p.name, p.value, p.fit)
        (ignored if verdict else unsupported).append(p.name)
    return ParamCheck(missing, tuple(unsupported), tuple(ignored))
