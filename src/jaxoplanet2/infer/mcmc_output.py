"""``mcmc-output``: tables and figures from ``results/mcmc_samples.npz``.

Writes, like allesfitter's mcmc_output:

- ``mcmc_table.csv``: median and 1-sigma errors of every fitted parameter
- ``mcmc_derived_table.csv``: derived parameters (see :mod:`.derived`)
- ``mcmc_latex_table.txt``: both tables as LaTeX rows
- ``mcmc_corner.<ext>`` and ``mcmc_fit_<inst>.<ext>``: posterior corner plot and
  the data with the model at the posterior medians
"""

from dataclasses import replace
from pathlib import Path

import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jaxoplanet2._jax import configure_jax  # noqa: E402
from jaxoplanet2.fitdir import FitDirectory, load_fit_directory  # noqa: E402
from jaxoplanet2.infer.derived import Derived, derive, summarize  # noqa: E402
from jaxoplanet2.infer.mcmc import load_samples  # noqa: E402
from jaxoplanet2.plots.initial_guess import EXTENSIONS, plot_instrument  # noqa: E402

TABLE_FILE = "mcmc_table.csv"


def _fmt(x: float) -> str:
    return repr(float(x))


def write_table(path: Path, fit: FitDirectory, flat: dict[str, np.ndarray]) -> None:
    rows = ["#name,median,lower_error,upper_error,label,unit"]
    for p in fit.params.free:
        med, lo, hi = summarize(flat[p.name])
        rows.append(f"{p.name},{_fmt(med)},{_fmt(lo)},{_fmt(hi)},{p.label},{p.unit}")
    path.write_text("\n".join(rows) + "\n")


def read_table(path: str | Path) -> dict[str, tuple[float, float, float]]:
    """name -> (median, lower_error, upper_error) from a written table."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, med, lo, hi, *_ = line.split(",")
        out[name] = (float(med), float(lo), float(hi))
    return out


def write_derived(path: Path, derived: dict[str, Derived]) -> None:
    rows = ["#name,median,lower_error,upper_error,label,unit"]
    for name, d in derived.items():
        med, lo, hi = summarize(d.values)
        rows.append(f"{name},{_fmt(med)},{_fmt(lo)},{_fmt(hi)},{d.label},{d.unit}")
    path.write_text("\n".join(rows) + "\n")


def latex_row(label: str, values: np.ndarray, unit: str) -> str:
    med, lo, hi = summarize(values)
    digits = max(0, 2 - int(np.floor(np.log10(max(min(lo, hi), 1e-300)))))
    value = f"${med:.{digits}f}_{{-{lo:.{digits}f}}}^{{+{hi:.{digits}f}}}$"
    return f"{label} & {value} & {unit} \\\\"


def write_latex(path, fit, flat, derived) -> None:
    lines = ["% fitted parameters"]
    lines += [
        latex_row(p.label or p.name, flat[p.name], p.unit) for p in fit.params.free
    ]
    lines.append("% derived parameters")
    lines += [latex_row(d.label, d.values, d.unit) for d in derived.values()]
    path.write_text("\n".join(lines) + "\n")


def _corner(path: Path, fit: FitDirectory, flat: dict[str, np.ndarray]) -> None:
    names = [p.name for p in fit.params.free]
    data = np.column_stack([flat[n] for n in names])
    labels = [p.label or p.name for p in fit.params.free]
    fig = corner.corner(data, labels=labels, quantiles=(0.16, 0.5, 0.84))
    fig.savefig(path)
    plt.close(fig)


def posterior_output(
    fit_dir: str | Path,
    sampler: str,
    *,
    file_extension: str = "pdf",
    overwrite: bool = False,
    allow_unsupported: bool = False,
) -> list[Path]:
    """Tables and figures from ``results/<sampler>_samples.npz``.

    ``sampler`` is ``mcmc`` or ``ns``; every file is prefixed with it, as in
    allesfitter (``mcmc_table.csv``, ``ns_table.csv``, ...).
    """
    ext = file_extension.lstrip(".").lower()
    if ext not in EXTENSIONS:
        raise ValueError(f"unsupported figure format '{file_extension}'")
    fit = load_fit_directory(fit_dir, allow_unsupported=allow_unsupported)
    configure_jax(fit.settings)
    table = fit.results / f"{sampler}_table.csv"
    if table.exists() and not overwrite:
        raise FileExistsError(f"{table} exists; pass overwrite=True (-o) to redo")
    samples, _ = load_samples(fit_dir, sampler)
    flat = {name: np.asarray(v).ravel() for name, v in samples.items()}
    derived = derive(flat, fit.settings, fit.star, seed=fit.settings.jx.seed)

    derived_path = fit.results / f"{sampler}_derived_table.csv"
    latex_path = fit.results / f"{sampler}_latex_table.txt"
    write_table(table, fit, flat)
    write_derived(derived_path, derived)
    write_latex(latex_path, fit, flat, derived)
    written = [table, derived_path, latex_path]

    corner_path = fit.results / f"{sampler}_corner.{ext}"
    _corner(corner_path, fit, flat)
    written.append(corner_path)

    medians = {name: float(np.median(v)) for name, v in flat.items()}
    posterior = replace(fit, params=fit.params.with_values(medians))
    for inst in fit.settings.inst_all:
        fig = plot_instrument(posterior, inst, title="posterior median")
        path = fit.results / f"{sampler}_fit_{inst}.{ext}"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


def mcmc_output(fit_dir: str | Path, **kwargs) -> list[Path]:
    """Tables and figures from the MCMC samples (see :func:`posterior_output`)."""
    return posterior_output(fit_dir, "mcmc", **kwargs)
