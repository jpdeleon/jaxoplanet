"""Load a whole allesfitter-format fit directory in one go."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from jaxoplanet2.io.data import Dataset, load_datasets
from jaxoplanet2.io.epochs import shift_epochs
from jaxoplanet2.io.params import ParamTable, load_params
from jaxoplanet2.io.settings import Settings, load_settings
from jaxoplanet2.model.external_priors import (
    DensityPrior,
    Star,
    density_prior,
    load_star,
)
from jaxoplanet2.model.ttv import TtvWindows, ttv_windows

RESULTS_DIR = "results"


@dataclass(frozen=True)
class FitDirectory:
    path: Path
    settings: Settings
    params: ParamTable
    data: Mapping[str, Dataset]
    star: Star | None = None
    density_prior: DensityPrior | None = None
    # periods each epoch was moved by shift_epoch (allesfitter's change_epoch)
    epoch_shifts: Mapping[str, int] = field(default_factory=dict)
    ttv: Mapping[str, TtvWindows] = field(default_factory=dict)

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
    shifts: dict[str, int] = {}
    if settings.shift_epoch:
        params, shifts = shift_epochs(settings, params, data)
    star = load_star(path)
    prior = density_prior(star) if star and settings.use_host_density_prior else None
    ttv = MappingProxyType(ttv_windows(settings, params, data))
    return FitDirectory(
        path, settings, params, data, star, prior, MappingProxyType(shifts), ttv
    )
