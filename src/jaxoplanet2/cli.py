"""The ``jaxoplanet`` command line interface.

Mirrors the allesfitter CLI (``show-initial-guess``, ``optimize``, ``mcmc-fit``,
...) so existing fit directories and habits carry over unchanged.
"""

import typer

from jaxoplanet2._metadata import __version__

app = typer.Typer(
    name="jaxoplanet",
    help="allesfitter-style transit and RV fitting powered by jaxoplanet",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"jaxoplanet2 v{__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    pass


@app.command()
def init(
    dir_path: str = typer.Argument(..., help="directory to create the fit in"),
    overwrite: bool = typer.Option(
        False, "--overwrite", "-o", help="replace existing params/settings files"
    ),
) -> None:
    """Create template params.csv and settings.csv in a fit directory."""
    from jaxoplanet2.init import init_directory

    try:
        written = init_directory(dir_path, overwrite=overwrite)
    except FileExistsError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e
    for path in written:
        typer.echo(f"wrote {path}")
    typer.echo(
        "Next: add your data as <inst>.csv (e.g. tess.csv: time,flux,flux_err), "
        f"edit the templates, then run 'jaxoplanet show-initial-guess {dir_path}'."
    )


@app.command()
def validate(
    dir_path: str = typer.Argument(..., help="path to the fit directory"),
    allow_unsupported: bool = typer.Option(
        False,
        "--allow-unsupported",
        help="warn about unsupported settings/params instead of failing",
    ),
) -> None:
    """Check settings, params and data, and evaluate the initial model."""
    from jaxoplanet2.validate import validate as _validate

    report = _validate(dir_path, allow_unsupported=allow_unsupported)
    for line in report.info:
        typer.echo(line)
    for line in report.warnings:
        typer.echo(f"WARNING: {line}")
    for line in report.errors:
        typer.echo(f"ERROR: {line}")
    if not report.ok:
        raise typer.Exit(1)
    typer.echo("OK")


@app.command()
def show_initial_guess(
    dir_path: str = typer.Argument(..., help="path to the fit directory"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    no_plot: bool = typer.Option(False, "--no-plot", help="skip generating figures"),
    file_extension: str = typer.Option(
        ".pdf", "--file-extension", "-e", help="figure format: pdf, png, jpg, svg, webp"
    ),
    allow_unsupported: bool = typer.Option(False, "--allow-unsupported"),
) -> None:
    """Plot the data with the model at the params.csv values."""
    from jaxoplanet2.plots.initial_guess import show_initial_guess as _show

    try:
        paths = _show(
            dir_path,
            do_plot=not no_plot,
            file_extension=file_extension,
            allow_unsupported=allow_unsupported,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e
    if not quiet:
        for path in paths:
            typer.echo(f"wrote {path}")


@app.command()
def optimize(  # noqa: PLR0917 (typer maps each option to a parameter)
    dir_path: str = typer.Argument(..., help="path to the fit directory"),
    method: str = typer.Option(
        "L-BFGS-B",
        "--method",
        "-m",
        help="L-BFGS-B, TNC, SLSQP, Nelder-Mead, Powell, differential_evolution, "
        "dual_annealing",
    ),
    restarts: int = typer.Option(1, "--restarts", "-n", help="number of restarts"),
    seed: int = typer.Option(42, "--seed", help="random seed"),
    maxfevals: int | None = typer.Option(
        None, "--maxfevals", help="per-restart iteration budget"
    ),
    no_update: bool = typer.Option(
        False, "--no-update", help="never rewrite params.csv, even if accepted"
    ),
    skip_bounds_check: bool = typer.Option(
        False,
        "--skip-bounds-check",
        help="accept optima on a prior bound (e.g. cosi=0 for a central transit)",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    allow_unsupported: bool = typer.Option(False, "--allow-unsupported"),
) -> None:
    """Maximise the posterior to warm-start MCMC / nested sampling."""
    from jaxoplanet2.infer.optimize import OptimizeError, optimize as _optimize
    from jaxoplanet2.plots.initial_guess import show_initial_guess as _show

    try:
        result = _optimize(
            dir_path,
            method=method,
            n_restarts=restarts,
            seed=seed,
            maxiter=maxfevals,
            update_params=not no_update,
            skip_bounds_check=skip_bounds_check,
            quiet=quiet,
            allow_unsupported=allow_unsupported,
        )
    except OptimizeError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e
    if result.accepted and not no_update:
        if not quiet:
            typer.echo("Optimization accepted; refreshing initial-guess plots.")
        _show(dir_path, allow_unsupported=allow_unsupported)


@app.command()
def mcmc_fit(
    dir_path: str = typer.Argument(..., help="path to the fit directory"),
    no_progress: bool = typer.Option(False, "--no-progress", help="hide progress bars"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    allow_unsupported: bool = typer.Option(False, "--allow-unsupported"),
) -> None:
    """Sample the posterior with NUTS (numpyro)."""
    from jaxoplanet2.infer.mcmc import (
        McmcError,
        enable_host_devices,
        mcmc_fit as _mcmc_fit,
    )

    enable_host_devices()  # before anything touches the JAX backend
    try:
        _mcmc_fit(
            dir_path,
            quiet=quiet,
            progress_bar=not no_progress,
            allow_unsupported=allow_unsupported,
        )
    except McmcError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e
    if not quiet:
        typer.echo(f"Run 'jaxoplanet mcmc-output {dir_path}' for tables and plots.")


def main() -> None:
    app()
