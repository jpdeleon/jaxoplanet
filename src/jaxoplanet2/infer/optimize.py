"""``optimize``: find the posterior maximum to warm-start the samplers.

Mirrors allesfitter2's ``optimize``: maximise the log posterior (priors,
likelihood and external priors) over the free parameters, apply its acceptance
gates, and on acceptance back up params.csv to params.csv.orig and rewrite the
fitted values in place, so the next command starts from the optimum.

Local methods use JAX gradients. ``differential_evolution`` / ``dual_annealing``
search globally inside the prior bounds (normal priors clipped at 5 sigma) and
are followed by an L-BFGS-B refinement.
"""

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density
from scipy import optimize as scipy_optimize

from jaxoplanet2._jax import configure_jax
from jaxoplanet2.fitdir import FitDirectory, load_fit_directory
from jaxoplanet2.io.params import PARAMS_FILE
from jaxoplanet2.io.priors import Normal, Prior, TruncNormal, Uniform
from jaxoplanet2.io.writers import backup_params, write_param_values
from jaxoplanet2.model.numpyro_model import build_model
from jaxoplanet2.validate import validate

GRADIENT_METHODS = ("L-BFGS-B", "TNC", "SLSQP")
LOCAL_METHODS = (*GRADIENT_METHODS, "Nelder-Mead", "Powell")
GLOBAL_METHODS = ("differential_evolution", "dual_annealing")
NORMAL_SIGMA_CLIP = 5.0
BOUND_MARGIN = 1e-4  # fraction of the prior width counted as "on the bound"
RESTART_JITTER = 0.5  # in units of each parameter's prior scale
REJECTED = 1e300  # objective value for points with a non-finite posterior
SAVE_FILE = "optimize_save.json"
OPTIMIZED_PARAMS_FILE = "params_optimized.csv"


class OptimizeError(ValueError):
    """The fit directory or the requested method cannot be optimised."""


@dataclass
class OptimizeResult:
    """JSON-serialisable summary, as allesfitter2's ``optimize_save.json``."""

    method: str
    accepted: bool
    success: bool
    lnprob_initial: float
    lnprob_opt: float
    delta_lnprob: float
    theta_initial: list
    theta_opt: list  # in the params.csv frame (epoch shifts undone)
    fitkeys: list
    n_restarts: int
    restart_lnprobs: list
    nfev: int
    wallclock_s: float
    reject_reason: str = ""
    bounds: list = field(default_factory=list)


def _curvature_scale(logp, theta0: np.ndarray, priors: list[Prior]) -> np.ndarray:
    """Per-parameter step scale: the local posterior width 1/sqrt(-H_ii).

    Falls back to a prior-based scale where the curvature is not usable.
    """
    diag = np.diag(np.asarray(jax.hessian(logp)(jnp.asarray(theta0))))
    fallback = np.array([_prior_scale(p) for p in priors])
    with np.errstate(divide="ignore", invalid="ignore"):
        width = 1.0 / np.sqrt(-diag)
    usable = np.isfinite(width) & (width > 0)
    return np.where(usable, np.minimum(width, fallback), fallback)


def _prior_scale(prior: Prior) -> float:
    if isinstance(prior, Uniform):
        return (prior.upper - prior.lower) / 10.0
    if isinstance(prior, TruncNormal):
        return min(prior.sd, (prior.upper - prior.lower) / 10.0)
    return prior.sd


def _bounds(prior: Prior, clip: bool) -> tuple[float | None, float | None]:
    if isinstance(prior, Normal):
        if not clip:
            return (None, None)
        half = NORMAL_SIGMA_CLIP * prior.sd
        return (prior.mean - half, prior.mean + half)
    return prior.support


class Objective:
    """Log posterior over the free parameters, in scaled coordinates.

    ``theta = theta0 + scale * x`` puts every parameter on a similar footing,
    which matters for BJD epochs next to limb-darkening coefficients.
    """

    def __init__(self, fit: FitDirectory):
        free = fit.params.free
        self.fitkeys = [p.name for p in free]
        self.theta0 = np.array([p.value for p in free])
        self.priors = [p.prior for p in free]
        model = build_model(fit)
        keys = self.fitkeys

        def logp(theta):
            return log_density(model, (), {}, dict(zip(keys, theta, strict=True)))[0]

        self.scale = _curvature_scale(logp, self.theta0, self.priors)

        self._logp = jax.jit(logp)
        self._value_and_grad = jax.jit(jax.value_and_grad(logp))
        self.nfev = 0
        # L-BFGS-B's first step is the raw gradient; dividing the objective by
        # its initial gradient norm makes that step ~1 posterior width
        _, grad0 = self._value_and_grad(jnp.asarray(self.theta0))
        norm0 = float(np.linalg.norm(np.asarray(grad0) * self.scale))
        self.f_scale = norm0 if np.isfinite(norm0) and norm0 > 1 else 1.0

    def theta(self, x: np.ndarray) -> np.ndarray:
        return self.theta0 + self.scale * np.asarray(x)

    def lnprob(self, theta: np.ndarray) -> float:
        value = float(self._logp(jnp.asarray(theta)))
        return value if np.isfinite(value) else -np.inf

    def negative(self, x: np.ndarray) -> float:
        self.nfev += 1
        value = self.lnprob(self.theta(x))
        return -value / self.f_scale if np.isfinite(value) else REJECTED

    def negative_with_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        self.nfev += 1
        value, grad = self._value_and_grad(jnp.asarray(self.theta(x)))
        if not (np.isfinite(value) and np.all(np.isfinite(grad))):
            return REJECTED, np.zeros_like(x)  # makes the line search back off
        return -float(value) / self.f_scale, -np.asarray(
            grad
        ) * self.scale / self.f_scale

    def x_bounds(self, clip: bool) -> list[tuple[float | None, float | None]]:
        out = []
        for prior, t0, s in zip(self.priors, self.theta0, self.scale, strict=True):
            lo, hi = _bounds(prior, clip)
            out.append(
                (
                    None if lo is None else (lo - t0) / s,
                    None if hi is None else (hi - t0) / s,
                )
            )
        return out

    def on_bounds(self, theta: np.ndarray) -> list[str]:
        hits = []
        for name, prior, value in zip(self.fitkeys, self.priors, theta, strict=True):
            if isinstance(prior, Normal):
                continue
            lo, hi = prior.support
            margin = BOUND_MARGIN * (hi - lo)
            if value <= lo + margin or value >= hi - margin:
                hits.append(name)
        return hits


def _run_once(obj: Objective, method: str, x0: np.ndarray, maxiter: int | None, seed):
    options = {"maxiter": maxiter} if maxiter else {}
    if method in GLOBAL_METHODS:
        bounds = obj.x_bounds(clip=True)
        search = getattr(scipy_optimize, method)
        kwargs = {"seed": seed, "maxiter": maxiter or 1000}
        if method == "differential_evolution":
            kwargs["x0"] = np.clip(x0, *np.array(bounds).T)
        res = search(obj.negative, bounds, **kwargs)
        x0 = res.x
        method = "L-BFGS-B"  # refine the global optimum with gradients
    jac = method in GRADIENT_METHODS
    fun = obj.negative_with_grad if jac else obj.negative
    bounds = obj.x_bounds(clip=False)
    res = scipy_optimize.minimize(
        fun, x0, jac=jac, method=method, bounds=bounds, options=options
    )
    return res.x, obj.lnprob(obj.theta(res.x)), bool(res.success)


def _starts(obj: Objective, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    starts = [np.zeros(len(obj.fitkeys))]
    bounds = np.array(
        [(-np.inf if lo is None else lo, np.inf if hi is None else hi)
         for lo, hi in obj.x_bounds(clip=False)]
    )  # fmt: skip
    for _ in range(n - 1):
        x = rng.normal(0.0, RESTART_JITTER, len(obj.fitkeys))
        starts.append(np.clip(x, bounds[:, 0], bounds[:, 1]))
    return starts


def _to_params_frame(fit: FitDirectory, obj: Objective, theta: np.ndarray):
    """Undo shift_epoch with the optimised period (allesfitter2's convention)."""
    values = dict(zip(obj.fitkeys, theta.tolist(), strict=True))
    for c, n in fit.epoch_shifts.items():
        if n and f"{c}_epoch" in values:
            period = values.get(f"{c}_period", fit.params[f"{c}_period"].value)
            values[f"{c}_epoch"] -= n * period
    return values


def _gate(
    obj, lnp0, lnp, theta, *, restart_lnprobs, threshold, consistency, check_bounds
) -> str:
    if not np.isfinite(lnp):
        return "optimum has a non-finite log posterior"
    if lnp - lnp0 < threshold:
        return f"improvement {lnp - lnp0:.2f} < threshold {threshold:.2f}"
    hits = obj.on_bounds(theta) if check_bounds else []
    if hits:
        return f"optimum on prior bounds: {', '.join(hits)}"
    finite = [lp for lp in restart_lnprobs if np.isfinite(lp)]
    if len(finite) > 1 and max(finite) - min(finite) > consistency:
        return f"restart spread {max(finite) - min(finite):.2f} > {consistency:.2f}"
    return ""


def optimize(
    fit_dir: str | Path,
    *,
    method: str = "L-BFGS-B",
    n_restarts: int = 1,
    seed: int = 42,
    maxiter: int | None = None,
    update_params: bool = True,
    quiet: bool = False,
    allow_unsupported: bool = False,
    improvement_threshold: float | None = None,
    consistency_threshold: float = 1.0,
    skip_bounds_check: bool = False,
) -> OptimizeResult:
    if method not in LOCAL_METHODS + GLOBAL_METHODS:
        known = ", ".join(LOCAL_METHODS + GLOBAL_METHODS)
        raise OptimizeError(f"unknown method '{method}'; use one of {known}")
    report = validate(fit_dir, allow_unsupported=allow_unsupported)
    if not report.ok:
        raise OptimizeError("\n".join(report.errors))
    fit = load_fit_directory(fit_dir, allow_unsupported=allow_unsupported)
    configure_jax(fit.settings)
    obj = Objective(fit)
    lnp0 = obj.lnprob(obj.theta0)
    threshold = improvement_threshold
    if threshold is None:  # allesfitter2's default
        threshold = 0.5 * len(obj.fitkeys)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    starts = _starts(obj, n_restarts, rng)
    runs = [_run_once(obj, method, x0, maxiter, seed) for x0 in starts]
    best_x, best_lnp, _ = max(runs, key=lambda r: r[1])
    theta = obj.theta(best_x)
    restart_lnprobs = [float(r[1]) for r in runs]
    reject = _gate(
        obj,
        lnp0,
        best_lnp,
        theta,
        restart_lnprobs=restart_lnprobs,
        threshold=threshold,
        consistency=consistency_threshold,
        check_bounds=not skip_bounds_check,
    )
    values = _to_params_frame(fit, obj, theta)

    result = OptimizeResult(
        method=method,
        accepted=not reject,
        success=any(r[2] for r in runs),
        lnprob_initial=float(lnp0),
        lnprob_opt=float(best_lnp),
        delta_lnprob=float(best_lnp - lnp0),
        theta_initial=obj.theta0.tolist(),
        theta_opt=[values[k] for k in obj.fitkeys],
        fitkeys=obj.fitkeys,
        n_restarts=n_restarts,
        restart_lnprobs=restart_lnprobs,
        nfev=obj.nfev,
        wallclock_s=time.perf_counter() - t0,
        reject_reason=reject,
        bounds=[list(p.support) for p in obj.priors],
    )
    _save(fit, result, values, update_params)
    if not quiet:
        _report(result)
    return result


def _save(fit: FitDirectory, result: OptimizeResult, values, update_params) -> None:
    fit.results.mkdir(parents=True, exist_ok=True)
    (fit.results / SAVE_FILE).write_text(json.dumps(asdict(result), indent=2))
    optimized = fit.results / OPTIMIZED_PARAMS_FILE
    shutil.copy2(fit.path / PARAMS_FILE, optimized)
    write_param_values(optimized, values)
    if result.accepted and update_params:
        backup_params(fit.path)
        write_param_values(fit.path / PARAMS_FILE, values)


def _report(result: OptimizeResult) -> None:
    status = "accepted" if result.accepted else f"rejected ({result.reject_reason})"
    print(
        f"optimize [{result.method}]: ln posterior {result.lnprob_initial:.3f} -> "
        f"{result.lnprob_opt:.3f} (delta {result.delta_lnprob:+.3f}), {status}; "
        f"{result.nfev} evaluations in {result.wallclock_s:.1f} s"
    )
