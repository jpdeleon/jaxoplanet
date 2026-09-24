"""Parse allesfitter's ``params.csv`` into an immutable :class:`ParamTable`."""

import math
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from jaxoplanet2.io.priors import Normal, Prior, PriorError, TruncNormal, parse_bounds

PARAMS_FILE = "params.csv"
DEFAULT_COLUMNS = ("name", "value", "fit", "bounds", "label", "unit", "truth")
OPTIONAL_COLUMNS = ("truth", "init_err", "coupled_with")
REQUIRED_COLUMNS = ("name", "value", "fit")
FAR_FROM_PRIOR_SIGMA = 3.0


class ParamsError(ValueError):
    """params.csv is malformed or inconsistent."""


@dataclass(frozen=True)
class Param:
    name: str
    value: float
    fit: bool
    prior: Prior | None
    label: str = ""
    unit: str = ""
    truth: float | None = None


@dataclass(frozen=True)
class ParamTable:
    params: tuple[Param, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for p in self.params:
            if p.name in seen:
                raise ParamsError(f"params.csv: '{p.name}' is defined twice")
            seen.add(p.name)

    def __getitem__(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def __contains__(self, name: object) -> bool:
        return any(p.name == name for p in self.params)

    def __iter__(self) -> Iterator[Param]:
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    @property
    def free(self) -> tuple[Param, ...]:
        return tuple(p for p in self.params if p.fit)

    @property
    def fixed(self) -> tuple[Param, ...]:
        return tuple(p for p in self.params if not p.fit)

    def values(self) -> dict[str, float]:
        return {p.name: p.value for p in self.params}

    def with_values(self, values: Mapping[str, float]) -> "ParamTable":
        unknown = set(values) - set(self.names)
        if unknown:
            raise ParamsError(f"unknown parameters: {', '.join(sorted(unknown))}")
        return ParamTable(
            tuple(
                replace(p, value=float(values[p.name])) if p.name in values else p
                for p in self.params
            )
        )


def load_params(fit_dir: str | Path, filename: str = PARAMS_FILE) -> ParamTable:
    path = Path(fit_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"no {filename} in {Path(fit_dir)}")
    return parse_params_text(path.read_text())


def parse_params_text(text: str) -> ParamTable:
    columns = DEFAULT_COLUMNS
    params = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        header = _header_columns(line)
        if header is not None:
            columns = header
            continue
        # genfromtxt(comments="#") semantics: drop inline comments too
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            params.append(_parse_row(line, columns))
        except (ParamsError, PriorError) as e:
            raise ParamsError(f"params.csv line {lineno}: {e}") from e
    return ParamTable(tuple(params))


def _header_columns(line: str) -> tuple[str, ...] | None:
    body = line.lstrip("\\#").strip()
    if not body.startswith("name,"):
        return None
    columns = tuple(c.strip() for c in body.split(","))
    known = set(DEFAULT_COLUMNS) | set(OPTIONAL_COLUMNS)
    unknown = [c for c in columns if c and c not in known]
    if unknown:
        raise ParamsError(f"params.csv: unknown column(s) {', '.join(unknown)}")
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ParamsError(f"params.csv: missing column(s) {', '.join(missing)}")
    return columns


def _parse_row(line: str, columns: tuple[str, ...]) -> Param:
    cells = [c.strip() for c in line.split(",")]
    extra = [c for c in cells[len(columns) :] if c]
    if extra:
        raise ParamsError(f"more cells than columns ({', '.join(extra)})")
    row = dict(zip(columns, cells + [""] * (len(columns) - len(cells)), strict=True))

    name = row["name"]
    if not name:
        raise ParamsError("missing parameter name")
    if row.get("coupled_with"):
        raise ParamsError(f"{name}: coupled_with is not supported by jaxoplanet2")
    value = _float(row["value"], f"{name}: value")
    fit = _fit_flag(row["fit"], name)

    prior = None
    if fit:
        if not row.get("bounds"):
            raise ParamsError(f"{name}: fit=1 needs bounds (e.g. 'uniform 0 1')")
        prior = parse_bounds(row["bounds"])
        _check_initial_value(name, value, prior)

    truth = row.get("truth", "")
    return Param(
        name=name,
        value=value,
        fit=fit,
        prior=prior,
        label=row.get("label", ""),
        unit=row.get("unit", ""),
        truth=_float(truth, f"{name}: truth") if truth else None,
    )


def _fit_flag(text: str, name: str) -> bool:
    # allesfitter reads the column as a number, so "1.0" is as good as "1"
    try:
        flag = float(text)
    except ValueError:
        flag = math.nan
    if flag not in (0.0, 1.0):
        raise ParamsError(f"{name}: fit must be 0 or 1, got '{text}'")
    return flag == 1.0


def _float(text: str, what: str) -> float:
    try:
        return float(text)
    except ValueError as e:
        raise ParamsError(f"{what} '{text}' is not a number") from e


def _check_initial_value(name: str, value: float, prior: Prior) -> None:
    lower, upper = prior.support
    if not lower <= value <= upper:
        raise ParamsError(
            f"{name}: initial value {value} lies outside its bounds [{lower}, {upper}]"
        )
    if isinstance(prior, (Normal, TruncNormal)):
        if abs(value - prior.mean) > FAR_FROM_PRIOR_SIGMA * prior.sd:
            warnings.warn(
                f"params.csv: initial value of {name} lies more than 3 sigma "
                "from its prior",
                UserWarning,
                stacklevel=5,
            )
