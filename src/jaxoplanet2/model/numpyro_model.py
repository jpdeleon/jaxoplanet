"""The full probabilistic model of a fit directory, as a numpyro model.

Free parameters are sampled from their params.csv priors, fixed ones enter as
constants. Every instrument contributes one observed site ``obs_<inst>`` whose
distribution is the white-noise Gaussian around model + baseline.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

from jaxoplanet2.fitdir import FitDirectory
from jaxoplanet2.io.data import Dataset
from jaxoplanet2.io.priors import to_distribution
from jaxoplanet2.io.settings import Settings
from jaxoplanet2.model.noise import white_noise_sigma
from jaxoplanet2.model.photometry import flux_model
from jaxoplanet2.model.rv import rv_model

Values = Mapping[str, jax.Array | float]
OBS_PREFIX = "obs_"


def signal(values: Values, settings: Settings, data: Dataset) -> jax.Array:
    """Astrophysical model at the data's time stamps (flux or RV)."""
    model = flux_model if data.kind == "flux" else rv_model
    return model(values, settings, data.inst, data.time)


def observation(values: Values, settings: Settings, data: Dataset) -> dist.Distribution:
    """Distribution of ``data.y`` given the parameter values."""
    mu = signal(values, settings, data)
    sigma = white_noise_sigma(values, settings, data)
    baseline = settings.baseline[(data.kind, data.inst)]
    if baseline.lower() != "none":
        raise NotImplementedError(
            f"baseline_{data.kind}_{data.inst}={baseline} is not implemented yet"
        )
    return dist.Normal(mu, sigma)


def sample_values(fit: FitDirectory) -> dict[str, jax.Array | float]:
    values: dict[str, jax.Array | float] = {p.name: p.value for p in fit.params.fixed}
    for p in fit.params.free:
        assert p.prior is not None
        values[p.name] = numpyro.sample(p.name, to_distribution(p.prior))
    return values


def build_model(fit: FitDirectory) -> Callable[[], None]:
    def model() -> None:
        values = sample_values(fit)
        for inst, data in fit.data.items():
            numpyro.sample(
                OBS_PREFIX + inst,
                observation(values, fit.settings, data),
                obs=jnp.asarray(data.y),
            )

    return model


def initial_values(fit: FitDirectory) -> dict[str, float]:
    return {p.name: p.value for p in fit.params.free}


@dataclass(frozen=True)
class LogProb:
    log_prior: float
    log_likelihood: float
    per_instrument: Mapping[str, float]

    @property
    def total(self) -> float:
        return self.log_prior + self.log_likelihood


def log_prob_parts(fit: FitDirectory, params: Mapping[str, float]) -> LogProb:
    """Log prior, log likelihood and per-instrument likelihoods at ``params``."""
    model = handlers.substitute(build_model(fit), data=dict(params))
    trace = handlers.trace(model).get_trace()
    prior, per_inst = 0.0, {}
    for name, site in trace.items():
        if site["type"] != "sample":
            continue
        lp = float(jnp.sum(site["fn"].log_prob(site["value"])))
        if site["is_observed"]:
            per_inst[name.removeprefix(OBS_PREFIX)] = lp
        else:
            prior += lp
    return LogProb(prior, sum(per_inst.values()), per_inst)
