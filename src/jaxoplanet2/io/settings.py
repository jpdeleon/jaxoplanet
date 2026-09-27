"""Parse allesfitter's ``settings.csv`` into an immutable :class:`Settings`."""

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

from jaxoplanet2.io._settings_schema import (
    BOOLS,
    EMPTY,
    FALSY,
    TRUTHY,
    Kind,
    match_rule,
)

SETTINGS_FILE = "settings.csv"
T = TypeVar("T")

# allesfitter's defaults for keys the user may leave out
DEFAULT_FAST_FIT_WIDTH = 8.0 / 24.0
DEFAULT_LD_LAW = "quad"
DEFAULT_MCMC = {"nwalkers": 100, "total_steps": 2000, "burn_steps": 1000, "thin_by": 1}
LEGACY_PREFIXES = (
    ("planets", "companions"),
    ("ld_law", "host_ld_law"),
    ("use_stellar_density_prior", "use_host_density_prior"),
    ("use_stellar_density", "use_host_density_prior"),
)


class SettingsError(ValueError):
    """settings.csv is malformed or internally inconsistent."""


class UnsupportedSettingsError(SettingsError):
    """settings.csv asks for features jaxoplanet2 does not implement."""

    def __init__(self, problems: Mapping[str, str]):
        self.keys = tuple(problems)
        lines = [f"  - {key}: {reason}" for key, reason in problems.items()]
        super().__init__(
            "settings.csv uses features jaxoplanet2 does not support:\n"
            + "\n".join(lines)
            + "\nRemove them, or pass allow_unsupported=True "
            "(--allow-unsupported) to ignore them."
        )


@dataclass(frozen=True)
class McmcSettings:
    nwalkers: int
    total_steps: int
    burn_steps: int
    thin_by: int

    @property
    def num_warmup(self) -> int:
        return self.burn_steps

    @property
    def num_samples(self) -> int:
        return self.total_steps - self.burn_steps


@dataclass(frozen=True)
class JxSettings:
    """jaxoplanet2-only knobs, all prefixed ``jx_`` in settings.csv."""

    x64: bool = True
    seed: int = 42
    num_chains: int | None = None  # None: derive from mcmc_nwalkers
    # 0.99: zero divergences on a synthetic transit, vs 47/600 at 0.9 (see #23)
    nuts_target_accept: float = 0.99
    nuts_max_tree_depth: int = 10
    # transit posteriors are correlated (e.g. radius_ratio, impact_param): a dense
    # mass matrix needs ~2.6x fewer leapfrog steps than a diagonal one
    nuts_dense_mass: bool = True
    optimizer: str = "lbfgs"


@dataclass(frozen=True)
class Settings:
    raw: Mapping[str, str]
    companions_phot: tuple[str, ...]
    companions_rv: tuple[str, ...]
    inst_phot: tuple[str, ...]
    inst_rv: tuple[str, ...]
    fast_fit: bool
    fast_fit_width: float
    shift_epoch: bool
    inst_for_epoch: Mapping[str, str]
    use_host_density_prior: bool
    fit_ttvs: bool
    mcmc: McmcSettings
    jx: JxSettings
    ld_law: Mapping[str, str | None]
    ld_space: Mapping[str, str]
    t_exp: Mapping[str, float | None]
    t_exp_n_int: Mapping[str, int | None]
    baseline: Mapping[tuple[str, str], str]
    error: Mapping[tuple[str, str], str]

    @property
    def companions_all(self) -> tuple[str, ...]:
        return _unique(self.companions_phot + self.companions_rv)

    @property
    def inst_all(self) -> tuple[str, ...]:
        return _unique(self.inst_phot + self.inst_rv)

    def kinds(self, inst: str) -> tuple[str, ...]:
        """The data kinds ('flux', 'rv') an instrument provides."""
        return tuple(
            kind
            for kind, insts in (("flux", self.inst_phot), ("rv", self.inst_rv))
            if inst in insts
        )


def load_settings(fit_dir: str | Path, allow_unsupported: bool = False) -> Settings:
    path = Path(fit_dir) / SETTINGS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no {SETTINGS_FILE} in {Path(fit_dir)}")
    return parse_settings_text(path.read_text(), allow_unsupported=allow_unsupported)


def parse_settings_text(text: str, allow_unsupported: bool = False) -> Settings:
    raw = _read_rows(text)
    lists = {k: _as_list(raw.get(k)) for k in _LIST_KEYS}
    _check_structure(lists)
    _check_supported(raw, lists, allow_unsupported)
    return _build(MappingProxyType(raw), lists)


_LIST_KEYS = ("companions_phot", "companions_rv", "inst_phot", "inst_rv")


def _read_rows(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # allesfitter reads with genfromtxt(comments="#"): inline comments go
        line = line.split("#", 1)[0].strip()
        key, _, value = line.partition(",")
        key, value = _rename_legacy(key.strip()), value.strip()
        if key in rows:
            raise SettingsError(f"'{key}' appears more than once in settings.csv")
        rows[key] = value
    return rows


def _rename_legacy(key: str) -> str:
    for old, new in LEGACY_PREFIXES:
        if key.startswith(old) and not key.startswith(new):
            renamed = new + key[len(old) :]
            warnings.warn(
                f"settings.csv: '{key}' is outdated, reading it as '{renamed}'",
                DeprecationWarning,
                stacklevel=4,
            )
            return renamed
    return key


def _as_list(value: str | None) -> tuple[str, ...]:
    if value is None or value.lower() in EMPTY:
        return ()
    return tuple(value.split())


def _unique(items: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _check_structure(lists: Mapping[str, tuple[str, ...]]) -> None:
    if not (lists["inst_phot"] or lists["inst_rv"]):
        raise SettingsError("settings.csv needs at least one of inst_phot / inst_rv")
    for kind in ("phot", "rv"):
        if lists[f"companions_{kind}"] and not lists[f"inst_{kind}"]:
            raise SettingsError(
                f"companions_{kind} is set but inst_{kind} lists no instruments"
            )


def _check_supported(
    raw: Mapping[str, str],
    lists: Mapping[str, tuple[str, ...]],
    allow_unsupported: bool,
) -> None:
    names = {
        "inst": set(lists["inst_phot"] + lists["inst_rv"]),
        "comp": set(lists["companions_phot"] + lists["companions_rv"]),
    }
    problems: dict[str, str] = {}
    for key, value in raw.items():
        found = match_rule(key)
        if found is None:
            problems[key] = f"unknown setting (value '{value}')"
            continue
        rule, m = found
        if rule.group and rule.kind is Kind.SUPPORTED:
            name = m.group(rule.group)
            if name not in names[rule.group]:
                # allesfitter ignores leftovers for instruments/companions that
                # are not part of this fit; so do we, but say so.
                listed = "inst_phot/inst_rv" if rule.group == "inst" else "companions"
                warnings.warn(
                    f"settings.csv: ignoring '{key}': '{name}' is not in {listed}",
                    UserWarning,
                    stacklevel=4,
                )
                continue
        if rule.kind is Kind.OFF_ONLY and rule.allowed is FALSY:
            # allesfitter's set_bool: anything but true/1 (e.g. "No") is off
            if value.lower() in TRUTHY:
                problems[key] = f"'{value}' is only supported while switched off"
            continue
        if rule.allowed is not None and value.lower() not in rule.allowed:
            off_only = rule.kind is Kind.OFF_ONLY
            what = "only supported while switched off" if off_only else "unsupported"
            problems[key] = f"'{value}' is {what}"
    if not problems:
        return
    if not allow_unsupported:
        raise UnsupportedSettingsError(problems)
    warnings.warn(str(UnsupportedSettingsError(problems)), UserWarning, stacklevel=4)


def _typed(raw: Mapping[str, str], key: str, cast: Callable[[str], T], default: T) -> T:
    value = raw.get(key)
    if value is None or value.lower() in EMPTY:
        return default
    try:
        return cast(value)
    except ValueError as e:
        raise SettingsError(f"settings.csv: cannot read '{key}' = '{value}'") from e


def _bool(value: str) -> bool:
    if value.lower() not in BOOLS:
        raise ValueError(value)
    return value.lower() in ("true", "1")


def _build(raw: Mapping[str, str], lists: Mapping[str, tuple[str, ...]]) -> Settings:
    inst_all = _unique(lists["inst_phot"] + lists["inst_rv"])
    companions = _unique(lists["companions_phot"] + lists["companions_rv"])
    kinds = [("flux", i) for i in lists["inst_phot"]] + [
        ("rv", i) for i in lists["inst_rv"]
    ]
    return Settings(
        raw=raw,
        **lists,
        fast_fit=_typed(raw, "fast_fit", _bool, False),
        fast_fit_width=_typed(raw, "fast_fit_width", float, DEFAULT_FAST_FIT_WIDTH),
        shift_epoch=_typed(raw, "shift_epoch", _bool, True),
        inst_for_epoch=MappingProxyType(
            {c: _typed(raw, f"inst_for_{c}_epoch", str, "all") for c in companions}
        ),
        use_host_density_prior=_typed(raw, "use_host_density_prior", _bool, True),
        fit_ttvs=_typed(raw, "fit_ttvs", _bool, False),
        mcmc=_build_mcmc(raw),
        jx=_build_jx(raw),
        ld_law=MappingProxyType({i: _ld_law(raw, i) for i in inst_all}),
        ld_space=_per_inst(raw, "host_ld_space_", inst_all, str, "q"),
        t_exp=_per_inst(raw, "t_exp_", inst_all, float, None),
        t_exp_n_int=_per_inst(raw, "t_exp_n_int_", inst_all, int, None),
        baseline=MappingProxyType(
            {(k, i): _typed(raw, f"baseline_{k}_{i}", str, "none") for k, i in kinds}
        ),
        error=MappingProxyType(
            {(k, i): _typed(raw, f"error_{k}_{i}", str, "sample") for k, i in kinds}
        ),
    )


def _ld_law(raw: Mapping[str, str], inst: str) -> str | None:
    # allesfitter2: empty/missing means 'quad'; only an explicit 'none' disables LD
    value = raw.get(f"host_ld_law_{inst}", "")
    if not value:
        return DEFAULT_LD_LAW
    return None if value.lower() == "none" else value


def _per_inst(
    raw: Mapping[str, str],
    prefix: str,
    insts: tuple[str, ...],
    cast: Callable[[str], T],
    default: T,
) -> Mapping[str, T]:
    return MappingProxyType({i: _typed(raw, prefix + i, cast, default) for i in insts})


def _build_mcmc(raw: Mapping[str, str]) -> McmcSettings:
    values = {k: _typed(raw, f"mcmc_{k}", int, v) for k, v in DEFAULT_MCMC.items()}
    for key, value in values.items():
        if value < 1:
            raise SettingsError(f"settings.csv: mcmc_{key} must be positive")
    if values["burn_steps"] >= values["total_steps"]:
        raise SettingsError(
            "settings.csv: mcmc_burn_steps must be smaller than mcmc_total_steps"
        )
    return McmcSettings(**values)


def _build_jx(raw: Mapping[str, str]) -> JxSettings:
    d = JxSettings()
    return JxSettings(
        x64=_typed(raw, "jx_x64", _bool, d.x64),
        seed=_typed(raw, "jx_seed", int, d.seed),
        num_chains=_typed(raw, "jx_num_chains", int, d.num_chains),
        nuts_target_accept=_typed(
            raw, "jx_nuts_target_accept", float, d.nuts_target_accept
        ),
        nuts_max_tree_depth=_typed(
            raw, "jx_nuts_max_tree_depth", int, d.nuts_max_tree_depth
        ),
        nuts_dense_mass=_typed(raw, "jx_nuts_dense_mass", _bool, d.nuts_dense_mass),
        optimizer=_typed(raw, "jx_optimizer", str, d.optimizer),
    )
