"""``ns-fit``: nested sampling with jaxns (numpyro's NestedSampler wrapper).

Gives the Bayesian evidence ln Z for model comparison, like allesfitter's
dynesty-based ``ns_fit``. allesfitter's settings map as follows:

- ``ns_nlive`` -> number of live points (default: 25 per free parameter)
- ``ns_tol``   -> the ``dlogZ`` termination tolerance (default 1e-4)
- ``ns_modus``, ``ns_bound``, ``ns_sample`` are dynesty internals; jaxns has no
  equivalent and they are ignored.

Needs the optional dependency: ``pip install "jaxoplanet2[ns]"``.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

from jaxoplanet2._jax import configure_jax
from jaxoplanet2._metadata import __version__
from jaxoplanet2.fitdir import FitDirectory, load_fit_directory
from jaxoplanet2.infer.mcmc import DIVERGING_KEY, META_KEY
from jaxoplanet2.model.numpyro_model import build_model
from jaxoplanet2.validate import validate

SAMPLES_FILE = "ns_samples.npz"
DIAGNOSTICS_FILE = "ns_diagnostics.txt"
POSTERIOR_DRAWS = 4000


class NestedError(ValueError):
    """The fit cannot be run with nested sampling."""


@dataclass(frozen=True)
class NestedResult:
    samples: dict[str, np.ndarray]  # equal-weight posterior draws, shape (1, n)
    log_z: float
    log_z_err: float
    wallclock_s: float


def sampler_kwargs(fit: FitDirectory) -> tuple[dict, dict]:
    """(jaxns constructor kwargs, termination kwargs) from settings.csv."""
    raw = fit.settings.raw
    constructor, termination = {}, {}
    if raw.get("ns_nlive"):
        constructor["num_live_points"] = int(raw["ns_nlive"])
    if raw.get("ns_tol"):
        termination["dlogZ"] = float(raw["ns_tol"])
    return constructor, termination


def _nested_sampler():
    try:
        from numpyro.contrib.nested_sampling import NestedSampler
    except ImportError as e:  # jaxns missing
        raise NestedError(
            'nested sampling needs jaxns: pip install "jaxoplanet2[ns]"'
        ) from e
    return NestedSampler


def ns_fit(
    fit_dir: str | Path, *, quiet: bool = False, allow_unsupported: bool = False
) -> NestedResult:
    report = validate(fit_dir, allow_unsupported=allow_unsupported)
    if not report.ok:
        raise NestedError("\n".join(report.errors))
    fit = load_fit_directory(fit_dir, allow_unsupported=allow_unsupported)
    configure_jax(fit.settings)
    constructor, termination = sampler_kwargs(fit)
    sampler = _nested_sampler()(
        build_model(fit),
        # copies: numpyro fills in its defaults (incl. JAX devices) in place
        constructor_kwargs=dict(constructor),
        termination_kwargs=dict(termination),
    )
    seed = fit.settings.jx.seed
    t0 = time.perf_counter()
    sampler.run(jax.random.PRNGKey(seed))
    draws = sampler.get_samples(jax.random.PRNGKey(seed + 1), POSTERIOR_DRAWS)
    draws = jax.block_until_ready(draws)
    wallclock = time.perf_counter() - t0

    names = [p.name for p in fit.params.free]
    results = sampler._results  # jaxns results; numpyro exposes no public accessor
    result = NestedResult(
        samples={n: np.asarray(draws[n])[None, :] for n in names},
        log_z=float(results.log_Z_mean),
        log_z_err=float(results.log_Z_uncert),
        wallclock_s=wallclock,
    )
    _save(fit, result, names, constructor, termination)
    if not quiet:
        print(_summary(result))
    return result


def _summary(result: NestedResult) -> str:
    return (
        f"nested sampling: ln Z = {result.log_z:.3f} +- {result.log_z_err:.3f}; "
        f"{POSTERIOR_DRAWS} posterior draws; {result.wallclock_s:.1f} s"
    )


def _save(fit, result, names, constructor, termination) -> None:
    fit.results.mkdir(parents=True, exist_ok=True)
    meta = {
        "fitkeys": names,
        "log_z": result.log_z,
        "log_z_err": result.log_z_err,
        "epoch_shifts": dict(fit.epoch_shifts),
        "constructor_kwargs": constructor,
        "termination_kwargs": termination,
        "jaxoplanet2_version": __version__,
        "wallclock_s": result.wallclock_s,
    }
    np.savez(
        fit.results / SAMPLES_FILE,
        **result.samples,
        **{
            DIVERGING_KEY: np.zeros((1, POSTERIOR_DRAWS), bool),
            META_KEY: np.array(json.dumps(meta)),
        },
    )
    (fit.results / DIAGNOSTICS_FILE).write_text(_summary(result) + "\n")
