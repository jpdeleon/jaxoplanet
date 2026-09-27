"""``jaxoplanet validate``: check a fit directory before spending time on a fit."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jaxoplanet2._jax import configure_jax
from jaxoplanet2.fitdir import FitDirectory, load_fit_directory
from jaxoplanet2.model.numpyro_model import initial_values, log_prob_parts
from jaxoplanet2.model.requirements import check_params


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(path: str | Path, allow_unsupported: bool = False) -> ValidationReport:
    report = ValidationReport()
    try:
        fit = load_fit_directory(path, allow_unsupported=allow_unsupported)
    except (OSError, ValueError) as e:
        report.errors.append(str(e))
        return report
    configure_jax(fit.settings)
    _describe(fit, report)
    _check_parameters(fit, report, allow_unsupported)
    if report.ok:
        _check_likelihood(fit, report)
    return report


def _describe(fit: FitDirectory, report: ValidationReport) -> None:
    s = fit.settings
    report.info.append(
        f"companions: phot {list(s.companions_phot)}, rv {list(s.companions_rv)}"
    )
    for inst, d in fit.data.items():
        report.info.append(f"{inst}: {len(d)} {d.kind} points")
    report.info.append(
        f"parameters: {len(fit.params.free)} free, {len(fit.params.fixed)} fixed"
    )


def _check_parameters(
    fit: FitDirectory, report: ValidationReport, allow_unsupported: bool
) -> None:
    check = check_params(fit.settings, fit.params)
    if check.legacy:
        report.errors.append(
            "params.csv uses allesfitter's transit parameters "
            f"({', '.join(check.legacy)}); jaxoplanet2 samples jaxoplanet's native "
            "ones (radius_ratio, duration, impact_param, time_transit, period). "
            f"Run 'jaxoplanet convert-params {fit.path}' to convert the file."
        )
        return
    for name in check.missing:
        report.errors.append(f"params.csv is missing '{name}'")
    unsupported = [f"params.csv: '{n}' is not supported" for n in check.unsupported]
    (report.warnings if allow_unsupported else report.errors).extend(unsupported)
    if check.ignored:
        report.warnings.append(f"ignored (not used): {', '.join(check.ignored)}")


def _check_likelihood(fit: FitDirectory, report: ValidationReport) -> None:
    parts = log_prob_parts(fit, initial_values(fit))
    for inst, ll in parts.per_instrument.items():
        if not np.isfinite(ll):
            report.errors.append(
                f"{inst}: log-likelihood is not finite at initial values"
            )
    if not np.isfinite(parts.log_prior):
        report.errors.append("log-prior is not finite at initial values")
    if report.ok:
        report.info.append(f"log-prior at initial values: {parts.log_prior:.3f}")
        report.info.append(
            f"log-likelihood at initial values: {parts.log_likelihood:.3f}"
        )
