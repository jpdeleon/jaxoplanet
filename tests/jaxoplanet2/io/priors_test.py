import math

import numpy as np
import numpyro.distributions as dist
import pytest

from jaxoplanet2.io.priors import (
    Normal,
    PriorError,
    TruncNormal,
    Uniform,
    parse_bounds,
    to_distribution,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("uniform 0 0.1000", Uniform(0.0, 0.1)),
        ("uniform  -10   -1", Uniform(-10.0, -1.0)),
        ("normal 2459169.619940 0.001420", Normal(2459169.61994, 0.00142)),
        ("trunc_normal 0 1 0.43 0.07", TruncNormal(0.0, 1.0, 0.43, 0.07)),
        ("Uniform 0 1", Uniform(0.0, 1.0)),
    ],
)
def test_parse_bounds(text, expected):
    assert parse_bounds(text) == expected


@pytest.mark.parametrize(
    "text, match",
    [
        ("", "empty"),
        ("gaussian 0 1", "gaussian"),
        ("uniform 0", "2 numbers"),
        ("trunc_normal 0 1 0.5", "4 numbers"),
        ("uniform 0 abc", "abc"),
        ("uniform 1 0", "lower"),
        ("uniform 1 1", "lower"),
        ("normal 0 0", "positive"),
        ("trunc_normal 0 1 0.5 -1", "positive"),
        ("trunc_normal 1 0 0.5 0.1", "lower"),
        ("uniform 0 inf", "finite"),
    ],
)
def test_parse_bounds_errors(text, match):
    with pytest.raises(PriorError, match=match):
        parse_bounds(text)


def test_extra_numbers_are_ignored_with_warning_like_allesfitter():
    with pytest.warns(UserWarning, match="ignoring"):
        prior = parse_bounds("trunc_normal 0 1 0.0134 0.0010 0.0012")
    assert prior == TruncNormal(0.0, 1.0, 0.0134, 0.0010)


def test_support():
    assert Uniform(0, 1).support == (0, 1)
    assert Normal(0, 1).support == (-math.inf, math.inf)
    assert TruncNormal(-1, 2, 0, 1).support == (-1, 2)


def test_to_distribution_uniform():
    d = to_distribution(Uniform(0.0, 2.0))
    assert isinstance(d, dist.Uniform)
    np.testing.assert_allclose(d.log_prob(1.0), -np.log(2.0))


def test_to_distribution_normal():
    d = to_distribution(Normal(1.0, 2.0))
    np.testing.assert_allclose(d.mean, 1.0)
    np.testing.assert_allclose(d.variance, 4.0)


def test_to_distribution_trunc_normal_respects_bounds():
    d = to_distribution(TruncNormal(0.0, 1.0, 0.43, 0.07))
    assert np.isfinite(d.log_prob(0.5))
    assert d.log_prob(1.5) == -np.inf


def test_describe_round_trips():
    for prior in (Uniform(0, 0.1), Normal(1, 2), TruncNormal(0, 1, 0.43, 0.07)):
        assert parse_bounds(prior.describe()) == prior


def test_loguniform_prior():
    from jaxoplanet2.io.priors import LogUniform

    prior = parse_bounds("loguniform 1e-4 1e-1")
    assert prior == LogUniform(1e-4, 1e-1)
    assert parse_bounds(prior.describe()) == prior
    d = to_distribution(prior)
    # density 1 / (x ln(hi/lo))
    np.testing.assert_allclose(
        d.log_prob(1e-2), -np.log(1e-2 * np.log(1e-1 / 1e-4)), rtol=1e-6
    )


@pytest.mark.parametrize("text", ["loguniform 0 1", "loguniform -1 1", "loguniform 2 1"])
def test_loguniform_errors(text):
    with pytest.raises(PriorError):
        parse_bounds(text)
