"""Text of the files ``jaxoplanet init`` writes: params, settings, params_star, run.sh.

Every builder is a pure function of already-resolved values, so the generated
fit directory can be checked without any network access. Only settings and
parameters jaxoplanet2 supports are active; alternatives are left commented.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from jaxoplanet2.prepare.priors import DEFAULT_GP_BOUNDS, CompanionPriors, GpBounds, Star

COMPANION_LETTERS = tuple("bcdefghijk")
SECTION = (
    "###############################################################################,"
)
TTV_HALF_WIDTH = 0.05  # d: +-72 min
DILUTION_SIGMA_FLOOR = 0.01  # CROWDSAP carries no formal uncertainty
MCMC_CHAINS = 2  # NUTS: mcmc_nwalkers is the number of chains


@dataclass(frozen=True)
class LimbDarkening:
    """Theoretical (Claret) quadratic coefficients in Kipping's q space."""

    q1: float
    q1_err: float
    q2: float
    q2_err: float


@dataclass(frozen=True)
class Dilution:
    """SPOC CROWDSAP-derived dilution: median and scatter over segments."""

    median: float
    std: float


def _lines(lines: Sequence[str]) -> str:
    return "".join(f"{line}\n" for line in lines)


def _companion_rows(c: str, p: CompanionPriors, insts: Sequence[str]) -> list[str]:
    lo, hi = p.duration_bounds
    rows = [
        f"#companion {c}: jaxoplanet's native transit parameters,,,,,,",
        f"{c}_radius_ratio,{p.radius_ratio:.4f},1,uniform 0 {p.radius_ratio_upper:.4f},"
        f"$R_{c} / R_\\star$,,",
        f"{c}_duration,{p.duration:.5f},1,uniform {lo:g} {hi:g},$T_{{14;{c}}}$,d,",
        f"{c}_impact_param,{p.impact_param:.2f},1,uniform 0 {p.impact_param_upper:g},"
        f"$b_{c}$,,",
        f"{c}_time_transit,{p.time_transit:.6f},1,"
        f"normal {p.time_transit:.6f} {p.time_transit_err:.6f},$T_{{0;{c}}}$,BJD,",
        f"{c}_period,{p.period:.6f},1,normal {p.period:.6f} {p.period_err:.6f},"
        f"$P_{c}$,d,",
        f"{c}_f_c,0,0,uniform -1 1,$\\sqrt{{e_{c}}} \\cos{{\\omega_{c}}}$,,",
        f"{c}_f_s,0,0,uniform -1 1,$\\sqrt{{e_{c}}} \\sin{{\\omega_{c}}}$,,",
    ]
    rows += [f"#{c}_sbratio_{inst},0,0,uniform 0 1,$J$,," for inst in insts]
    return rows


def _dilution_rows(inst: str, dilution: Dilution | None) -> list[str]:
    label = f"$D_\\mathrm{{0; {inst}}}$"
    rows = []
    if dilution is not None:
        sd = max(dilution.std, DILUTION_SIGMA_FLOOR)
        m = dilution.median
        rows.append(f"#dil_{inst},{m:.6f},1,normal {m:.6f} {sd:.6f},{label} (SPOC),,")
    rows.append(f"dil_{inst},0,0,uniform 0 1,{label},,")
    return rows


def _ld_rows(inst: str, ld: LimbDarkening | None) -> list[str]:
    rows = []
    for n in (1, 2):
        label = f"$q_{{{n}; \\mathrm{{{inst}}}}}$"
        if ld is not None:
            q, err = (ld.q1, ld.q1_err) if n == 1 else (ld.q2, ld.q2_err)
            rows.append(
                f"#host_ldc_q{n}_{inst},{q:.2f},1,normal {q:.2f} {err:.2f},{label},,"
            )
        rows.append(f"host_ldc_q{n}_{inst},0.5,1,uniform 0 1,{label},,")
    return rows


def _log_row(name: str, bounds: tuple[float, float, float], label: str, unit="") -> str:
    init, lo, hi = bounds
    return f"{name},{init:.3f},1,uniform {lo:.3f} {hi:.3f},{label},{unit},"


def _noise_rows(inst: str, gp: GpBounds) -> list[str]:
    return [
        _log_row(
            f"ln_err_flux_{inst}",
            gp.ln_err,
            f"$\\ln{{\\sigma_\\mathrm{{{inst}}}}}$",
            "rel. flux",
        )
    ]


def _baseline_rows(inst: str, gp: GpBounds) -> list[str]:
    return [
        f"#baseline_gp_offset_flux_{inst},0,1,uniform -0.1 0.1,"
        f"$\\mathrm{{offset ({inst})}}$,,",
        _log_row(
            f"baseline_gp_matern32_lnsigma_flux_{inst}",
            gp.lnsigma,
            f"$\\mathrm{{gp \\ln \\sigma ({inst})}}$",
        ),
        _log_row(
            f"baseline_gp_matern32_lnrho_flux_{inst}",
            gp.lnrho,
            f"$\\mathrm{{gp \\ln \\rho ({inst})}}$",
        ),
        f"#baseline_offset_flux_{inst},0,1,uniform -0.01 0.01,"
        f"$\\mathrm{{offset ({inst})}}$,,",
    ]


def ttv_rows(c: str, n_transits: int) -> list[str]:
    if n_transits == 0:
        return []
    rows = [f"#TTV companion {c},,,,,,"]
    rows += [
        f"{c}_ttv_transit_{j},0,1,uniform -{TTV_HALF_WIDTH} {TTV_HALF_WIDTH},"
        f"TTV$_\\mathrm{{{c};{j}}}$,d,"
        for j in range(1, n_transits + 1)
    ]
    return rows


def params_csv(
    companions: Mapping[str, CompanionPriors],
    insts: Sequence[str],
    *,
    ld: LimbDarkening | None = None,
    gp_bounds: Mapping[str, GpBounds] | None = None,
    dilution: Mapping[str, Dilution] | None = None,
    ttv_counts: Mapping[str, int] | None = None,
) -> str:
    gp_bounds = gp_bounds or {}
    dilution = dilution or {}
    rows = ["#name,value,fit,bounds,label,unit,truth"]
    for c, p in companions.items():
        rows += _companion_rows(c, p, insts)
    rows.append("#dilution per instrument,,,,,,")
    for inst in insts:
        rows += _dilution_rows(inst, dilution.get(inst))
    rows.append("#limb darkening coefficients per instrument,,,,,,")
    for inst in insts:
        rows += _ld_rows(inst, ld)
    rows.append("#errors per instrument,,,,,,")
    for inst in insts:
        rows += _noise_rows(inst, gp_bounds.get(inst, DEFAULT_GP_BOUNDS))
    rows.append("#baseline per instrument,,,,,,")
    for inst in insts:
        rows += _baseline_rows(inst, gp_bounds.get(inst, DEFAULT_GP_BOUNDS))
    for c, n in (ttv_counts or {}).items():
        rows += ttv_rows(c, n)
    return _lines(rows)


def _header(title: str) -> list[str]:
    return [SECTION, f"# {title},", SECTION]


def _exposure_rows(insts: Sequence[str], t_exp: Mapping[str, float]) -> list[str]:
    rows = ["# crucial only for long (>600 s) exposure times,"]
    fallback = t_exp[insts[0]]
    for inst in insts:
        seconds = round(t_exp.get(inst, fallback))
        days = seconds / 86400.0
        if days >= 1:
            raise ValueError(f"exposure time of {inst} ({seconds} s) exceeds a day")
        rows += [f"t_exp_{inst},{days:.6f}", f"#t_exp_n_int_{inst},10"]
    return rows


def settings_csv(
    companions: Sequence[str],
    insts: Sequence[str],
    t_exp: Mapping[str, float],
    host_density_prior: bool,
    fit_ttvs: bool = False,
) -> str:
    rows = ["#name,value", *_header("General settings")]
    rows += [
        f"companions_phot,{' '.join(companions)}",
        "companions_rv,",
        f"inst_phot,{' '.join(insts)}",
        "inst_rv,",
        *_header("Fit performance settings"),
        "fast_fit,True",
        "fast_fit_width,0.3333333333333333",
        "shift_epoch,True",
    ]
    for c in companions:
        rows += [f"inst_for_{c}_epoch,all", f"#inst_for_{c}_epoch,{' '.join(insts)}"]
    rows += [
        *_header("MCMC settings (NUTS: nwalkers -> chains; burn -> warmup)"),
        f"mcmc_nwalkers,{MCMC_CHAINS}",
        "mcmc_total_steps,2000",
        "mcmc_burn_steps,1000",
        "mcmc_thin_by,1",
        *_header("Nested sampling settings (jaxoplanet ns-fit)"),
        "#ns_nlive,1000",
        "#ns_tol,0.0001",
        *_header("Limb darkening law per object and instrument"),
        *(f"host_ld_law_{inst},quad" for inst in insts),
        *_header("Exposure interpolation settings"),
        *_exposure_rows(insts, t_exp),
        *_header("Baseline settings per instrument"),
    ]
    for inst in insts:
        rows += [
            f"#baseline_flux_{inst},sample_offset",
            f"#baseline_flux_{inst},sample_linear",
            f"#baseline_flux_{inst},hybrid_poly_2",
            f"baseline_flux_{inst},sample_GP_Matern32",
        ]
    rows += [
        *_header("Error settings per instrument"),
        *(f"error_flux_{inst},sample" for inst in insts),
        *_header("Host density prior (needs params_star.csv)"),
        f"use_host_density_prior,{host_density_prior}",
        *_header("Fit TTVs"),
        f"fit_ttvs,{fit_ttvs}",
        *_header("jaxoplanet2-only settings (all optional)"),
        "# jx_x64,True",
        "# jx_seed,42",
        "# jx_num_chains,2",
        "# jx_nuts_dense_mass,True",
        "# jx_nuts_target_accept,0.99",
        "# jx_nuts_max_tree_depth,10",
    ]
    return _lines(rows)


def params_star_csv(star: Star) -> str:
    return _lines(
        [
            "#R_star,R_star_lerr,R_star_uerr,M_star,M_star_lerr,M_star_uerr,"
            "Teff_star,Teff_star_lerr,Teff_star_uerr",
            "#R_sun,R_sun,R_sun,M_sun,M_sun,M_sun,K,K,K",
            f"{star.radius:.2f},{star.radius_err:.2f},{star.radius_err:.2f},"
            f"{star.mass:.2f},{star.mass_err:.2f},{star.mass_err:.2f},"
            f"{star.teff:.0f},{star.teff_err:.0f},{star.teff_err:.0f}",
        ]
    )


RUN_SH = """#!/usr/bin/env bash
# Fit this directory with jaxoplanet2; uncomment the steps you want.
set -euo pipefail
cd "$(dirname "$0")"

### check settings, params and data, and evaluate the initial model
jaxoplanet validate .
### show the fit using the initial guess
jaxoplanet show-initial-guess .

### global optimization (updates params.csv after asking)
#jaxoplanet optimize .

### MCMC sampling (NUTS)
#jaxoplanet mcmc-fit .
#jaxoplanet mcmc-output . --overwrite

### nested sampling for evidence / model comparison (pip install jaxoplanet2[ns])
#jaxoplanet ns-fit .
#jaxoplanet ns-output . --overwrite
"""
