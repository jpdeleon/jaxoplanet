import warnings
from types import SimpleNamespace

import pytest

from jaxoplanet2._shadowing import (
    UpstreamShadowingWarning,
    upstream_installed,
    warn_if_upstream_installed,
)


def _dists(*names):
    return lambda: [SimpleNamespace(metadata={"Name": n}) for n in names]


def test_upstream_installed_detects_jaxoplanet():
    assert upstream_installed(_dists("numpy", "jaxoplanet"))


def test_upstream_installed_ignores_jaxoplanet2():
    assert not upstream_installed(_dists("numpy", "jaxoplanet2"))


def test_upstream_installed_is_case_insensitive():
    assert upstream_installed(_dists("JaxOplanet"))


def test_warns_when_upstream_installed():
    with pytest.warns(UpstreamShadowingWarning, match="overwrite"):
        warn_if_upstream_installed(_dists("jaxoplanet", "jaxoplanet2"))


def test_silent_when_only_jaxoplanet2_installed():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_upstream_installed(_dists("jaxoplanet2"))
