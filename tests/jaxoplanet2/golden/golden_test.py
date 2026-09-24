"""Golden parity test: jaxoplanet2 reproduces allesfitter's model.

Each ``cases/<name>`` directory is a complete allesfitter fit directory. The
matching ``<name>.npz`` holds allesfitter's model evaluated at the params.csv
values (see generate_golden.py). Here the same directory goes through the
jaxoplanet2 readers and model, and the two must agree.

Tolerances: batman is exact, so ``flux_model=batman`` cases must agree to 1 ppm.
ellc integrates numerically on a grid and is itself off by ~6 ppm, so ellc cases
get 10 ppm. RVs always come from ellc, good to ~2 cm/s; allow 5 cm/s (km/s data).
"""

from pathlib import Path

import jax
import numpy as np
import pytest

from jaxoplanet2.io.data import load_datasets
from jaxoplanet2.io.params import load_params
from jaxoplanet2.io.settings import load_settings
from jaxoplanet2.model.photometry import flux_model
from jaxoplanet2.model.rv import rv_model

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
CASES = sorted(p.name for p in (HERE / "cases").iterdir())
FLUX_ATOL = {"batman": 1e-6, "ellc": 1e-5}
RV_ATOL = 5e-5  # km/s


def test_every_case_has_a_reference():
    assert CASES
    for name in CASES:
        assert (HERE / f"{name}.npz").is_file(), f"run generate_golden.py for {name}"


@pytest.mark.parametrize("name", CASES)
def test_model_matches_allesfitter(name):
    case = HERE / "cases" / name
    reference = np.load(HERE / f"{name}.npz")
    settings = load_settings(case)
    values = load_params(case).values()
    data = load_datasets(case, settings, values)
    backend = settings.raw.get("flux_model", "ellc")

    for inst in settings.inst_phot:
        model = np.asarray(flux_model(values, settings, inst, data[inst].time))
        np.testing.assert_allclose(model, reference[inst], atol=FLUX_ATOL[backend])
    for inst in settings.inst_rv:
        model = np.asarray(rv_model(values, settings, inst, data[inst].time))
        np.testing.assert_allclose(model, reference[inst], atol=RV_ATOL)


@pytest.mark.parametrize("name", CASES)
def test_references_contain_a_real_signal(name):
    """Guard against a vacuous comparison (e.g. data that misses every transit)."""
    reference = np.load(HERE / f"{name}.npz")
    settings = load_settings(HERE / "cases" / name)
    for inst in settings.inst_phot:
        assert 1.0 - reference[inst].min() > 1e-3
    for inst in settings.inst_rv:
        assert np.ptp(reference[inst]) > 0.05
