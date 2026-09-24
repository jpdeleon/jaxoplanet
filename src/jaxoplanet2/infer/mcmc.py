"""``mcmc-fit``: NUTS sampling of the posterior with numpyro.

allesfitter's emcee settings map onto NUTS as follows (see issue #1):

- ``mcmc_nwalkers`` -> number of chains, capped by the available devices but
  at least two (override with ``jx_num_chains``); NUTS needs far fewer chains
  than emcee needs walkers.
- ``mcmc_burn_steps`` -> warmup, ``mcmc_total_steps - mcmc_burn_steps`` ->
  samples per chain, ``mcmc_thin_by`` -> thinning.

Samples are stored in ``results/mcmc_samples.npz`` as (chain, draw) arrays in
the fit's internal frame (epochs shifted by ``shift_epoch``, as allesfitter
reports them); the metadata records the shifts.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np
import numpyro
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import MCMC, NUTS, init_to_value

from jaxoplanet2._jax import configure_jax
from jaxoplanet2._metadata import __version__
from jaxoplanet2.fitdir import load_fit_directory
from jaxoplanet2.io.settings import Settings
from jaxoplanet2.model.numpyro_model import build_model, initial_values
from jaxoplanet2.validate import validate

SAMPLES_FILE = "mcmc_samples.npz"
DIAGNOSTICS_FILE = "mcmc_diagnostics.txt"
META_KEY = "__meta__"
DIVERGING_KEY = "__diverging__"
R_HAT_WARN = 1.01
MIN_CHAINS = 2


class McmcError(ValueError):
    """The fit directory cannot be sampled."""


def enable_host_devices(count: int | None = None) -> None:
    """Expose CPU cores as JAX devices so chains can run in parallel.

    Only effective before JAX initialises its backend (i.e. before the first
    computation), so the CLI calls it first thing.
    """
    numpyro.set_host_device_count(count or os.cpu_count() or 1)


@dataclass(frozen=True)
class RunConfig:
    num_chains: int
    num_warmup: int
    num_samples: int
    thinning: int
    chain_method: str
    target_accept: float
    max_tree_depth: int
    dense_mass: bool
    seed: int


def run_config(settings: Settings, device_count: int) -> RunConfig:
    mcmc, jx = settings.mcmc, settings.jx
    # at least two chains, so r_hat can compare them, even on a single device
    chains = jx.num_chains or min(mcmc.nwalkers, max(device_count, MIN_CHAINS))
    return RunConfig(
        num_chains=chains,
        num_warmup=mcmc.num_warmup,
        num_samples=mcmc.num_samples,
        thinning=mcmc.thin_by,
        # vmapped ("vectorized") NUTS runs chains in lockstep, each waiting for
        # the deepest tree; sequential is much faster on a single device
        chain_method="parallel" if chains <= device_count else "sequential",
        target_accept=jx.nuts_target_accept,
        max_tree_depth=jx.nuts_max_tree_depth,
        dense_mass=jx.nuts_dense_mass,
        seed=jx.seed,
    )


@dataclass(frozen=True)
class McmcResult:
    samples: dict[str, np.ndarray]  # (chain, draw)
    r_hat: dict[str, float]
    ess: dict[str, float]
    divergences: int
    wallclock_s: float


def mcmc_fit(
    fit_dir: str | Path,
    *,
    quiet: bool = False,
    progress_bar: bool = True,
    allow_unsupported: bool = False,
) -> McmcResult:
    report = validate(fit_dir, allow_unsupported=allow_unsupported)
    if not report.ok:
        raise McmcError("\n".join(report.errors))
    fit = load_fit_directory(fit_dir, allow_unsupported=allow_unsupported)
    configure_jax(fit.settings)
    cfg = run_config(fit.settings, jax.local_device_count())
    kernel = NUTS(
        build_model(fit),
        target_accept_prob=cfg.target_accept,
        max_tree_depth=cfg.max_tree_depth,
        dense_mass=cfg.dense_mass,
        init_strategy=init_to_value(values=initial_values(fit)),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.num_chains,
        thinning=cfg.thinning,
        chain_method=cfg.chain_method,
        progress_bar=progress_bar,
    )
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(cfg.seed), extra_fields=("diverging",))
    # JAX dispatches asynchronously: wait for the draws before timing
    grouped = jax.block_until_ready(mcmc.get_samples(group_by_chain=True))
    wallclock = time.perf_counter() - t0

    names = [p.name for p in fit.params.free]
    samples = {n: np.asarray(grouped[n]) for n in names}
    diverging = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"])
    result = McmcResult(
        samples=samples,
        r_hat={n: float(split_gelman_rubin(v)) for n, v in samples.items()},
        ess={n: float(effective_sample_size(v)) for n, v in samples.items()},
        divergences=int(diverging.sum()),
        wallclock_s=wallclock,
    )
    meta = {
        "fitkeys": names,
        "epoch_shifts": dict(fit.epoch_shifts),
        "jaxoplanet2_version": __version__,
        "wallclock_s": wallclock,
        **asdict(cfg),
    }
    _save(fit.results, result, diverging, meta)
    if not quiet:
        print(_diagnostics_text(result, cfg))
    return result


def _save(results: Path, result: McmcResult, diverging, meta) -> None:
    results.mkdir(parents=True, exist_ok=True)
    np.savez(
        results / SAMPLES_FILE,
        **result.samples,
        **{DIVERGING_KEY: diverging, META_KEY: np.array(json.dumps(meta))},
    )
    cfg = RunConfig(**{k: meta[k] for k in RunConfig.__dataclass_fields__})
    (results / DIAGNOSTICS_FILE).write_text(_diagnostics_text(result, cfg) + "\n")


def _diagnostics_text(result: McmcResult, cfg: RunConfig) -> str:
    lines = [
        f"NUTS: {cfg.num_chains} chains ({cfg.chain_method}), {cfg.num_warmup} "
        f"warmup + {cfg.num_samples} samples, thinning {cfg.thinning}; "
        f"{result.wallclock_s:.1f} s",
        f"divergences: {result.divergences}",
        f"{'parameter':32s} {'r_hat':>8s} {'ess':>10s}",
    ]
    for name in result.samples:
        flag = "  <-- not converged" if result.r_hat[name] > R_HAT_WARN else ""
        lines.append(
            f"{name:32s} {result.r_hat[name]:8.4f} {result.ess[name]:10.1f}{flag}"
        )
    return "\n".join(lines)


def load_samples(
    fit_dir: str | Path, sampler: str = "mcmc"
) -> tuple[dict[str, np.ndarray], dict]:
    """(samples by parameter as (chain, draw) arrays, run metadata).

    ``sampler`` is ``mcmc`` or ``ns`` (nested sampling stores equal-weight draws
    as a single "chain").
    """
    name = f"{sampler}_samples.npz"
    path = Path(fit_dir) / "results" / name
    if not path.is_file():
        raise FileNotFoundError(f"no {name}; run 'jaxoplanet {sampler}-fit' first")
    with np.load(path) as data:
        meta = json.loads(str(data[META_KEY]))
        samples = {name: data[name] for name in meta["fitkeys"]}
    return samples, meta
