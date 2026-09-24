"""Load a whole allesfitter-format fit directory in one go."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from jaxoplanet2.io.data import Dataset, load_datasets
from jaxoplanet2.io.params import ParamTable, load_params
from jaxoplanet2.io.settings import Settings, load_settings

RESULTS_DIR = "results"


@dataclass(frozen=True)
class FitDirectory:
    path: Path
    settings: Settings
    params: ParamTable
    data: Mapping[str, Dataset]

    @property
    def results(self) -> Path:
        return self.path / RESULTS_DIR


def load_fit_directory(
    path: str | Path, allow_unsupported: bool = False
) -> FitDirectory:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"fit directory {path} does not exist")
    settings = load_settings(path, allow_unsupported=allow_unsupported)
    params = load_params(path)
    data = load_datasets(path, settings, params.values())
    return FitDirectory(path, settings, params, data)
