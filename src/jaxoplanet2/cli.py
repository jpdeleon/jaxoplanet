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


def main() -> None:
    app()
