"""Finite exposure times: average a model over each exposure like allesfitter.

allesfitter evaluates the model on ``n_int`` sub-exposure nodes spread evenly over
``t_exp`` and combines them with trapezoid weights.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


def exposure_nodes(t_exp: float, n_int: int) -> tuple[np.ndarray, np.ndarray]:
    """allesfitter's trapezoid sub-exposure offsets and weights."""
    offsets = (np.arange(n_int) / (n_int - 1.0) - 0.5) * t_exp
    weights = np.ones(n_int) / (n_int - 1.0)
    weights[[0, -1]] *= 0.5
    return offsets, weights


def integrate_exposure(
    model: Callable[[jax.Array], jax.Array],
    time: jax.Array,
    t_exp: float | None,
    n_int: int | None,
) -> jax.Array:
    """``model`` averaged over each exposure; instantaneous if not smeared."""
    time = jnp.asarray(time, dtype=float)
    if not (t_exp and n_int and n_int > 1):
        return model(time)
    offsets, weights = exposure_nodes(t_exp, n_int)
    grid = time[:, None] + offsets[None, :]
    return model(grid.ravel()).reshape(grid.shape) @ weights
