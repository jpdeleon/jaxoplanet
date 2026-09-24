"""Which allesfitter ``settings.csv`` keys jaxoplanet2 understands.

Each key is classified by the first matching :class:`Rule`:

- *supported*: the value is used by jaxoplanet2 (``allowed`` restricts values);
- *ignored*: meaningless for a JAX backend (e.g. ``multiprocess``), accepted as is;
- *off-only*: a feature jaxoplanet2 lacks; accepted only while it is switched off.

Anything else is unsupported and reported, so a fit never silently drops a
feature the user asked for.
"""

import re
from dataclasses import dataclass
from enum import Enum


class Kind(Enum):
    SUPPORTED = "supported"
    IGNORED = "ignored"
    OFF_ONLY = "off-only"


BASELINES = frozenset(
    {
        "none",
        "sample_offset",
        "sample_linear",
        "hybrid_offset",
        *(f"hybrid_poly_{n}" for n in range(10)),
        "sample_gp_matern32",
        "sample_gp_sho",
        "sample_gp_real",
    }
)
BOOLS = frozenset({"true", "false", "1", "0"})
EMPTY = frozenset({"", "none"})


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern[str]
    kind: Kind
    # lower-cased values the key may take; None means any value
    allowed: frozenset[str] | None = None
    # name of the regex group holding an instrument or companion to validate
    group: str | None = None


def _rule(
    regex: str,
    kind: Kind = Kind.SUPPORTED,
    allowed: frozenset[str] | None = None,
    group: str | None = None,
) -> Rule:
    return Rule(re.compile(regex + r"\Z"), kind, allowed, group)


S, I, O = Kind.SUPPORTED, Kind.IGNORED, Kind.OFF_ONLY

# Order matters: the first matching rule wins, so specific patterns come first.
RULES: tuple[Rule, ...] = (
    _rule(r"companions_(phot|rv)"),
    _rule(r"inst_(phot|rv)"),
    _rule(r"(fast_fit|shift_epoch|use_host_density_prior|fit_ttvs)", allowed=BOOLS),
    _rule(r"fast_fit_width"),
    _rule(r"inst_for_(?P<comp>.+)_epoch", group="comp"),
    _rule(r"mcmc_(nwalkers|total_steps|burn_steps|thin_by)"),
    _rule(r"ns_.+"),
    _rule(r"jx_x64", allowed=BOOLS),
    _rule(r"jx_(seed|num_chains|nuts_target_accept|nuts_max_tree_depth|optimizer)"),
    _rule(r"host_ld_law_(?P<inst>.+)", allowed=EMPTY | {"quad", "lin"}, group="inst"),
    _rule(r"host_ld_space_(?P<inst>.+)", allowed=frozenset({"q", "u"}), group="inst"),
    _rule(r"t_exp_n_int_(?P<inst>.+)", group="inst"),
    _rule(r"t_exp_(?P<inst>.+)", group="inst"),
    _rule(r"baseline_(flux|rv)_.+_against", O, allowed=frozenset({"time"})),
    _rule(r"baseline_(flux|rv)_(?P<inst>.+)", allowed=BASELINES, group="inst"),
    _rule(r"error_(flux|rv)_(?P<inst>.+)", allowed=frozenset({"sample"}), group="inst"),
    # meaningless for jaxoplanet2: parallelism, ellc numerics, emcee internals
    _rule(r"multiprocess(_cores)?", I),
    _rule(r"(host|[^_]+)_(grid|shape)_.+", I),
    _rule(r"(exact_grav|print_progress|flux_model|phase_curve_style)", I),
    _rule(r"mcmc_(pre_run_loops|pre_run_steps|moves)", I),
    # features jaxoplanet2 does not have: fine while switched off
    _rule(r"(secondary_eclipse|phase_curve|phase_variations|mask_transit)", O,
          allowed=frozenset({"false", "0"})),
    _rule(r"N_flares", O, allowed=EMPTY | {"0"}),
    _rule(r"(host|[^_]+)_N_spots_.+", O, allowed=EMPTY | {"0"}),
    _rule(r"stellar_var_(flux|rv)", O, allowed=EMPTY),
    _rule(r"(host|[^_]+)_flux_weighted_.+", O, allowed=frozenset({"false", "0"})),
    _rule(r"[^_]+_ld_law_.+", O, allowed=EMPTY),
)  # fmt: skip


def match_rule(key: str) -> tuple[Rule, re.Match[str]] | None:
    for rule in RULES:
        m = rule.pattern.match(key)
        if m is not None:
            return rule, m
    return None
