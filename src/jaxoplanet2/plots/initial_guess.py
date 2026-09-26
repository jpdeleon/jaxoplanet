"""``show-initial-guess``: data with the model at the params.csv values.

One figure per instrument: the full time series with the model, its residuals,
and one phase-folded panel per companion in which the other companions' signals
have been removed.
"""

from dataclasses import replace
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from jaxoplanet2._jax import configure_jax  # noqa: E402
from jaxoplanet2.fitdir import FitDirectory, load_fit_directory  # noqa: E402
from jaxoplanet2.io.settings import Settings  # noqa: E402
from jaxoplanet2.model.numpyro_model import mean_components  # noqa: E402
from jaxoplanet2.model.parameterization import companion_geometry  # noqa: E402
from jaxoplanet2.model.photometry import flux_model  # noqa: E402
from jaxoplanet2.model.rv import rv_model  # noqa: E402
from jaxoplanet2.validate import validate  # noqa: E402

EXTENSIONS = ("pdf", "png", "jpg", "svg", "webp")
FOLD_POINTS = 1000
FOLD_BINS = 40
MIN_TRANSIT_WINDOW = 0.05  # days
HOURS = 24.0


def phase_offset(time: np.ndarray, epoch: float, period: float) -> np.ndarray:
    """Time since the nearest transit, in (-period/2, period/2]."""
    return (time - epoch + period / 2) % period - period / 2


def _only(settings: Settings, companion: str | None) -> Settings:
    if companion is None:
        return settings
    keep = (companion,)
    return replace(
        settings,
        companions_phot=keep if companion in settings.companions_phot else (),
        companions_rv=keep if companion in settings.companions_rv else (),
    )


def companion_signal(
    values, settings, inst, time, *, companion=None, ttv=None
) -> jax.Array:
    """Model signal (flux - 1, or RV) of one companion, or of all if None."""
    s = _only(settings, companion)
    if inst in settings.inst_phot:
        return flux_model(values, s, inst, time, ttv) - 1.0
    return rv_model(values, s, inst, time)


def binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray):
    """Mean of ``y`` in each bin of ``x``; empty bins are dropped."""
    counts, _ = np.histogram(x, edges)
    sums, _ = np.histogram(x, edges, weights=y)
    keep = counts > 0
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers[keep], sums[keep] / counts[keep]


def _fold_window(values, settings, inst, companion) -> float:
    g = companion_geometry(values, companion)
    period = float(g.period)
    if inst in settings.inst_rv:
        return period / 2
    # ~1.5 x the total duration of a central transit
    duration = period / (np.pi * float(g.a_over_rstar))
    return min(period / 2, max(1.5 * duration, MIN_TRANSIT_WINDOW))


def _companions(settings: Settings, inst: str) -> tuple[str, ...]:
    if inst in settings.inst_phot:
        return settings.companions_phot
    return settings.companions_rv


def plot_instrument(
    fit: FitDirectory, inst: str, title: str = "initial guess"
) -> plt.Figure:
    values = fit.params.values()
    data = fit.data[inst]
    settings = fit.settings
    total = np.asarray(companion_signal(values, settings, inst, data.time, ttv=fit.ttv))
    level = 1.0 if data.kind == "flux" else 0.0
    instrumental = np.asarray(mean_components(values, settings, data, fit.ttv)[1])
    model = level + total + instrumental
    ylabel = "relative flux" if data.kind == "flux" else "RV"
    companions = _companions(settings, inst) or (None,)

    fig = plt.figure(figsize=(4 * max(len(companions), 2), 8), layout="constrained")
    grid = GridSpec(3, len(companions), figure=fig, height_ratios=(2, 1, 2))
    ax_data = fig.add_subplot(grid[0, :])
    ax_res = fig.add_subplot(grid[1, :], sharex=ax_data)
    ax_data.errorbar(data.time, data.y, data.yerr, fmt=".", color="0.6", ms=2, zorder=0)
    ax_data.plot(data.time, model, "C0-", lw=1)
    ax_data.set(ylabel=ylabel, title=f"{inst}: {title}")
    ax_res.errorbar(data.time, data.y - model, data.yerr, fmt=".", ms=2)
    ax_res.axhline(0.0, color="C0")
    ax_res.set(xlabel="time [BJD]", ylabel="residuals")

    for col, c in enumerate(companions):
        ax = fig.add_subplot(grid[2, col])
        if c is None:
            ax.set_axis_off()
            continue
        _plot_fold(
            ax, fit, inst, c, remove=total + instrumental, level=level, ylabel=ylabel
        )
    return fig


def _plot_fold(ax, fit, inst, c, *, remove, level, ylabel) -> None:
    """Phase-fold companion ``c``; ``remove`` is everything but its own signal."""
    values, settings, data = fit.params.values(), fit.settings, fit.data[inst]
    epoch, period = float(values[f"{c}_time_transit"]), float(values[f"{c}_period"])
    own = np.asarray(
        companion_signal(values, settings, inst, data.time, companion=c, ttv=fit.ttv)
    )
    y = data.y - (remove - own)  # baseline and the other companions
    window = _fold_window(values, settings, inst, c)
    scale = HOURS if data.kind == "flux" else 1.0 / period
    time = data.time
    if c in fit.ttv and data.kind == "flux":  # align each transit on its own TTV
        time = time - np.asarray(fit.ttv[c].shift(values, time)[0])
    dt = phase_offset(time, epoch, period)
    near = np.abs(dt) <= window
    ax.errorbar(dt[near] * scale, y[near], data.yerr[near], fmt=".", color="0.6", ms=2)
    if data.kind == "flux":
        edges = np.linspace(-window, window, FOLD_BINS + 1)
        xb, yb = binned(dt[near], y[near], edges)
        ax.plot(xb * scale, yb, "o", color="C1", ms=4, zorder=3)
    fine = np.linspace(-window, window, FOLD_POINTS)
    model = np.asarray(
        companion_signal(values, settings, inst, epoch + fine, companion=c)
    )
    ax.plot(fine * scale, level + model, "C0-")
    xlabel = "hours from mid-transit" if data.kind == "flux" else "phase"
    ax.set(xlabel=xlabel, ylabel=ylabel, title=f"companion {c}")


def show_initial_guess(
    fit_dir: str | Path,
    *,
    do_plot: bool = True,
    file_extension: str = "pdf",
    allow_unsupported: bool = False,
) -> list[Path]:
    """Validate ``fit_dir`` and plot every instrument; returns the files written."""
    ext = file_extension.lstrip(".").lower()
    if ext not in EXTENSIONS:
        raise ValueError(f"unsupported figure format '{file_extension}'")
    report = validate(fit_dir, allow_unsupported=allow_unsupported)
    if not report.ok:
        raise ValueError("\n".join(report.errors))
    if not do_plot:
        return []
    fit = load_fit_directory(fit_dir, allow_unsupported=allow_unsupported)
    configure_jax(fit.settings)
    fit.results.mkdir(parents=True, exist_ok=True)
    paths = []
    for inst in fit.settings.inst_all:
        fig = plot_instrument(fit, inst)
        path = fit.results / f"initial_guess_{inst}.{ext}"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths
