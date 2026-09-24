"""The shipped examples stay valid as the code evolves (no data download)."""

from pathlib import Path

import pytest

from jaxoplanet2.io.params import load_params
from jaxoplanet2.io.settings import load_settings
from jaxoplanet2.model.requirements import check_params

EXAMPLES = Path(__file__).parents[2] / "examples" / "jaxoplanet2"


@pytest.mark.parametrize("example", sorted(p.name for p in EXAMPLES.iterdir()))
def test_example_settings_and_params_are_consistent(example):
    settings = load_settings(EXAMPLES / example)
    check = check_params(settings, load_params(EXAMPLES / example))
    assert check.missing == ()
    assert check.unsupported == ()
