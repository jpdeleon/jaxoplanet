"""Scaffold a new fit directory with template ``params.csv`` and ``settings.csv``."""

from importlib import resources
from pathlib import Path

TEMPLATE_FILES = ("params.csv", "settings.csv")


def init_directory(path: str | Path, overwrite: bool = False) -> list[Path]:
    """Copy the template files into ``path``, creating it if needed.

    Raises:
        FileExistsError: If a template file already exists and ``overwrite`` is
            False. Nothing is written in that case.
    """
    target = Path(path)
    destinations = [target / name for name in TEMPLATE_FILES]
    existing = [p.name for p in destinations if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{', '.join(existing)} already exist in {target}; "
            "pass overwrite=True (--overwrite) to replace them"
        )

    target.mkdir(parents=True, exist_ok=True)
    templates = resources.files("jaxoplanet2") / "templates"
    for dest in destinations:
        dest.write_text((templates / dest.name).read_text())
    return destinations
