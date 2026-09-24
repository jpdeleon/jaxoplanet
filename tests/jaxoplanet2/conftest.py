import os

import pytest

SLOW_ENV_VAR = "JAXOPLANET2_SLOW"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", f"slow: end-to-end fits; run only when {SLOW_ENV_VAR}=1"
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get(SLOW_ENV_VAR) == "1":
        return
    skip_slow = pytest.mark.skip(reason=f"set {SLOW_ENV_VAR}=1 to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
