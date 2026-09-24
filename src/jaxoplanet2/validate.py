"""``jaxoplanet validate``: check a fit directory before spending time on a fit."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jaxoplanet2._jax import configure_jax
from jaxoplanet2.fitdir import FitDirectory, load_fit_directory
from jaxoplanet2.model.noise import gaussian_loglike, white_noise_sigma
from jaxoplanet2.model.photometry import flux_model
from jaxoplanet2.model.requirements import check_params
from jaxoplanet2.model.rv import rv_model


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
    for name in check.missing:
        report.errors.append(f"params.csv is missing '{name}'")
    unsupported = [f"params.csv: '{n}' is not supported" for n in check.unsupported]
    (report.warnings if allow_unsupported else report.errors).extend(unsupported)
    if check.ignored:
        report.warnings.append(f"ignored (not used): {', '.join(check.ignored)}")


def _check_likelihood(fit: FitDirectory, report: ValidationReport) -> None:
    values = fit.params.values()
    total = 0.0
    for inst, data in fit.data.items():
        model = flux_model if data.kind == "flux" else rv_model
        mu = model(values, fit.settings, inst, data.time)
        sigma = white_noise_sigma(values, fit.settings, data)
        ll = float(gaussian_loglike(data.y - mu, sigma))
        if not np.isfinite(ll):
            report.errors.append(
                f"{inst}: log-likelihood is not finite at initial values"
            )
        total += ll
    if report.ok:
        report.info.append(f"log-likelihood at initial values: {total:.3f}")
