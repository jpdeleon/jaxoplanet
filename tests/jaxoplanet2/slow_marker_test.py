import os

import pytest

from tests.jaxoplanet2.conftest import SLOW_ENV_VAR


@pytest.mark.slow
def test_slow_tests_only_run_when_enabled():
    assert os.environ.get(SLOW_ENV_VAR) == "1"
