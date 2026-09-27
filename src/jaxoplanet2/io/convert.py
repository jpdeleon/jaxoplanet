"""Convert allesfitter's (rr, rsuma, cosi, epoch) to jaxoplanet-native parameters.

- ``<c>_rr`` -> ``<c>_radius_ratio``: identical
- ``<c>_epoch`` -> ``<c>_time_transit``: identical (mid-transit time)
- ``<c>_rsuma`` -> ``<c>_duration``: T14 from (k, a/R*, i, P, e, w)
- ``<c>_cosi`` -> ``<c>_impact_param``: b from (a/R*, i, e, w)

Values and truths convert exactly (the model inverts the same equations).
Priors on ``rr`` and ``epoch`` carry over; a prior on ``rsuma`` or ``cosi`` has no
equivalent in (duration, b), so those get broad uniform priors, which the
caller should review.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import jax

from jaxoplanet2.io.params import (
    DEFAULT_COLUMNS,
    PARAMS_FILE,
    Param,
    ParamTable,
    header_columns,
    load_params,
)
from jaxoplanet2.io.priors import Prior, Uniform
from jaxoplanet2.io.settings import Settings, load_settings
from jaxoplanet2.io.writers import backup_params
from jaxoplanet2.model.parameterization import transit_from_orbit

LEGACY_TRANSIT = ("rr", "rsuma", "cosi", "epoch")
RENAMED = {"rr": "radius_ratio", "epoch": "time_transit"}
DURATION_PRIOR_FACTOR = 3.0  # duration prior: uniform [T/3, 3T]


class ConversionError(ValueError):
    """params.csv cannot be converted to the native parameterization."""


def is_legacy(name: str, companions) -> bool:
    return any(name == f"{c}_{p}" for c in companions for p in LEGACY_TRANSIT)


@dataclass(frozen=True)
class Conversion:
    table: ParamTable
    replaced: Mapping[str, Param]  # old name -> new Param (first row of a pair)
    extra: Mapping[str, Param]  # old name -> second new Param it also produces
    notes: tuple[str, ...]


def _value(params: ParamTable, name: str, default: float) -> float:
    return params[name].value if name in params else default


def _truth(params: ParamTable, name: str, default: float) -> float | None:
    if name not in params:
        return default
    return params[name].truth


def _transit(k, rsuma, cosi, period, *, f_c, f_s) -> tuple[float, float]:
    ecc = f_c**2 + f_s**2
    sin_omega = f_s / math.sqrt(ecc) if ecc > 0 else 1.0
    with jax.enable_x64(True):  # written values must not carry float32 noise
        b, t14 = transit_from_orbit(
            k, (1 + k) / rsuma, math.acos(cosi), period, ecc=ecc, sin_omega=sin_omega
        )
    return float(b), float(t14)


def _round(x: float) -> float:
    """Readable prior bounds: 4 significant digits."""
    return float(f"{x:.4g}")


def _upper(prior: Prior | None, fallback: float) -> float:
    if prior is None or not math.isfinite(prior.support[1]):
        return fallback
    return prior.support[1]


def _convert_companion(params: ParamTable, c: str) -> tuple[dict, dict, list[str]]:
    get = {n: _value(params, f"{c}_{n}", d) for n, d in
           (("rr", 0.0), ("period", math.nan), ("f_c", 0.0), ("f_s", 0.0))}  # fmt: skip
    rsuma, cosi = params[f"{c}_rsuma"], params[f"{c}_cosi"]
    b, t14 = _transit(
        get["rr"], rsuma.value, cosi.value, get["period"], f_c=get["f_c"], f_s=get["f_s"]
    )
    if not (math.isfinite(b) and math.isfinite(t14) and t14 > 0):
        raise ConversionError(
            f"{c}: rsuma={rsuma.value}, cosi={cosi.value} do not transit"
        )
    truths = [_truth(params, f"{c}_{n}", d) for n, d in
              (("rr", 0.0), ("rsuma", None), ("cosi", None), ("period", None),
               ("f_c", 0.0), ("f_s", 0.0))]  # fmt: skip
    b_truth = t_truth = None
    if all(t is not None for t in truths):
        k, rs, ci, period, f_c, f_s = truths
        b_truth, t_truth = _transit(k, rs, ci, period, f_c=f_c, f_s=f_s)
    k_max = _upper(params[f"{c}_rr"].prior if f"{c}_rr" in params else None, get["rr"])
    duration = Param(
        name=f"{c}_duration",
        value=t14,
        fit=rsuma.fit,
        prior=Uniform(
            _round(t14 / DURATION_PRIOR_FACTOR), _round(t14 * DURATION_PRIOR_FACTOR)
        )
        if rsuma.fit
        else None,
        label=f"$T_{{14;{c}}}$",
        unit="d",
        truth=t_truth,
    )
    impact = Param(
        name=f"{c}_impact_param",
        value=b,
        fit=cosi.fit,
        prior=Uniform(0.0, _round(1.0 + max(k_max, get["rr"]))) if cosi.fit else None,
        label=f"$b_{{{c}}}$",
        unit="",
        truth=b_truth,
    )
    notes = []
    if rsuma.fit:
        notes.append(f"{c}_duration: new prior {duration.prior.describe()}")
    if cosi.fit:
        notes.append(f"{c}_impact_param: new prior {impact.prior.describe()}")
    return {f"{c}_rsuma": duration}, {f"{c}_cosi": impact}, notes


def to_native(params: ParamTable, settings: Settings) -> Conversion:
    """The native equivalent of an allesfitter-parameterised table."""
    replaced, second, notes = {}, {}, []
    for c in settings.companions_all:
        for old, new in RENAMED.items():
            if f"{c}_{old}" in params:
                p = params[f"{c}_{old}"]
                replaced[p.name] = replace(p, name=f"{c}_{new}")
        has = [f"{c}_{n}" in params for n in ("rsuma", "cosi")]
        if all(has):
            dur, imp, companion_notes = _convert_companion(params, c)
            replaced.update(dur)
            second.update(imp)
            notes += companion_notes
        elif any(has):
            raise ConversionError(f"{c}: rsuma and cosi must both be present")
    rows = []
    for p in params:
        if p.name in replaced:
            rows.append(replaced[p.name])
        elif p.name in second:
            rows.append(second[p.name])
        else:
            rows.append(p)
    return Conversion(ParamTable(tuple(rows)), replaced, second, tuple(notes))


def _row(p: Param, columns: tuple[str, ...]) -> str:
    cells = {
        "name": p.name,
        "value": repr(float(p.value)),
        "fit": "1" if p.fit else "0",
        "bounds": p.prior.describe() if p.prior is not None else "",
        "label": p.label,
        "unit": p.unit,
        "truth": "" if p.truth is None else repr(float(p.truth)),
    }
    return ",".join(cells.get(col, "") for col in columns)


def convert_params_file(fit_dir: str | Path) -> tuple[str, ...]:
    """Rewrite ``params.csv`` in place (backup: params.csv.orig); returns notes.

    Only the four converted rows change; comments and other rows are kept.
    """
    fit_dir = Path(fit_dir)
    settings = load_settings(fit_dir, allow_unsupported=True)
    params = load_params(fit_dir)
    if not any(is_legacy(p.name, settings.companions_all) for p in params):
        return ("params.csv already uses the native parameterization",)
    conversion = to_native(params, settings)
    new_rows = {**conversion.replaced, **conversion.extra}
    path = fit_dir / PARAMS_FILE
    columns = DEFAULT_COLUMNS
    out = []
    for line in path.read_text().splitlines():
        header = header_columns(line.strip())
        if header is not None:
            columns = header
        name = line.split(",", 1)[0].strip()
        out.append(_row(new_rows[name], columns) if name in new_rows else line)
    backup_params(fit_dir)
    path.write_text("\n".join(out) + "\n")
    return conversion.notes
