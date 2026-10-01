import math

import numpy as np
import pytest

from jaxoplanet2.prepare.priors import (
    DEFAULT_GP_BOUNDS,
    Ephemeris,
    PriorError,
    Star,
    a_over_rstar,
    circular_duration,
    companion_priors,
    dataset_gp_bounds,
    duration_bounds,
    gp_protective_duration,
    radius_ratio_from_depth,
    radius_ratio_upper,
    stellar_density,
)

SUN = Star(5772, 50, 4.44, 0.05, 0.0, 0.1, 1.0, 0.02, 1.0, 0.03)


def _eph(**kw):
    base = dict(
        period=3.0,
        period_err=1e-4,
        epoch=2459000.0,
        epoch_err=1e-3,
        duration=2.4,
        duration_err=0.1,
        depth=10_000.0,
        depth_err=100.0,
    )
    return Ephemeris(**{**base, **kw})


def test_solar_density_and_earth_orbit():
    rho = stellar_density(1.0, 1.0)
    assert rho == pytest.approx(1.41, rel=0.01)
    assert a_over_rstar(rho, 365.25) == pytest.approx(215, rel=0.01)


def test_circular_duration_of_central_transit_matches_winn():
    a, k, period = 10.0, 0.1, 3.0
    expected = period / math.pi * math.asin((1 + k) / a)
    assert circular_duration(period, a, k, 0.0) == pytest.approx(expected)


def test_circular_duration_is_nan_without_transit():
    assert math.isnan(circular_duration(3.0, 10.0, 0.1, 1.2))


def test_radius_ratio_upper_rounds_up_and_caps():
    assert radius_ratio_upper(0.12) == pytest.approx(0.25)
    assert radius_ratio_upper(0.9) == 0.5


def test_duration_bounds_stay_below_half_period():
    assert duration_bounds(0.1, 3.0) == (0.03333, 0.3)
    assert duration_bounds(1.0, 2.0)[1] == 1.0


def test_radius_ratio_from_depth():
    k, k_err = radius_ratio_from_depth(10_000, 100)
    assert k == pytest.approx(0.1)
    assert k_err == pytest.approx(0.01)


def test_companion_priors_use_catalog_duration():
    p = companion_priors(_eph(), 0.1, 0.01, SUN, rng=np.random.default_rng(0))
    assert p.duration == pytest.approx(0.1)
    assert p.duration_bounds == duration_bounds(0.1, 3.0)
    assert p.impact_param == 0.5
    assert p.impact_param_upper == pytest.approx(1 + p.radius_ratio_upper)
    assert p.time_transit_err == 1e-3
    assert math.isfinite(p.duration_from_density)


def test_companion_priors_fall_back_to_density_duration_and_errors():
    eph = _eph(duration=math.nan, epoch_err=math.nan, period_err=0.0)
    p = companion_priors(eph, 0.1, math.nan, SUN, rng=np.random.default_rng(0))
    assert p.duration == pytest.approx(p.duration_from_density)
    assert p.time_transit_err == 0.1
    assert p.period_err == pytest.approx(3e-3)


UNKNOWN_STAR = Star(*[math.nan] * 10)


@pytest.mark.parametrize(
    ("eph", "k", "star"),
    [
        (_eph(period=math.nan), 0.1, SUN),
        (_eph(), 0.0, SUN),
        (_eph(duration=math.nan), 0.1, UNKNOWN_STAR),
    ],
)
def test_companion_priors_reject_unphysical_input(eph, k, star):
    with pytest.raises(PriorError):
        companion_priors(eph, k, 0.01, star)


def test_gp_protective_duration_takes_the_longest():
    assert gp_protective_duration([0.1, math.nan, 0.2, -1]) == 0.2
    assert gp_protective_duration([]) == 0.1


def test_dataset_gp_bounds_follow_the_data():
    rng = np.random.default_rng(1)
    time = np.arange(0, 27, 2 / 1440)
    flux = 1 + rng.normal(0, 1e-3, time.size)
    b = dataset_gp_bounds(time, flux, 0.1)
    assert b.rms == pytest.approx(1e-3, rel=0.1)
    assert b.ln_err[1] < math.log(1e-3) < b.ln_err[2] <= math.log(0.1)
    assert b.lnsigma[1] < b.lnsigma[0] < b.lnsigma[2]
    assert math.exp(b.lnrho[1]) > 0.5 * 0.1
    assert math.exp(b.lnrho[2]) < 27


def test_dataset_gp_bounds_need_enough_points():
    assert dataset_gp_bounds(np.arange(5.0), np.ones(5), 0.1) is None


def test_dataset_gp_bounds_handle_constant_flux():
    b = dataset_gp_bounds(np.arange(100.0), np.ones(100), 0.1)
    assert b.rms == 1e-6
    assert DEFAULT_GP_BOUNDS.rms != DEFAULT_GP_BOUNDS.rms  # NaN placeholder
