import math

import numpy as np
import pytest

from jaxoplanet2.io.data import Dataset
from jaxoplanet2.io.epochs import (
    EpochShiftError,
    first_epoch,
    mid_epoch,
    shift_epochs,
    shift_prior,
)
from jaxoplanet2.io.params import parse_params_text
from jaxoplanet2.io.priors import Normal, TruncNormal, Uniform
from jaxoplanet2.io.settings import parse_settings_text


def test_first_epoch_moves_forward_and_backward_onto_the_data():
    time = np.array([100.0, 120.0])
    assert first_epoch(time, 0.5, 3.0, 0.0) == pytest.approx(102.5)
    assert first_epoch(time, 200.5, 3.0, 0.0) == pytest.approx(101.5)


def test_first_epoch_keeps_a_transit_whose_egress_is_in_the_data():
    # transit at 99.9 has its egress (width/2 = 0.2) after the data start
    assert first_epoch(np.array([100.0, 110.0]), 99.9, 3.0, 0.4) == pytest.approx(99.9)


def test_mid_epoch_is_near_the_data_centre():
    time = np.linspace(100, 130, 100)
    mid, n = mid_epoch(time, 1.0, 3.0, 0.0)
    assert abs(mid - 115.0) <= 1.5
    assert mid == pytest.approx(1.0 + n * 3.0)


def test_shift_matches_allesfitter_on_toi1448():
    # reference: allesfitter2 change_epoch on DONE/mdwarfs_hori/allesfit/TOI-1448
    new = shift_prior(
        Normal(2458713.3374, 0.0015), Normal(8.112246, 0.000018), 72, 8.112246
    )
    assert new.mean == pytest.approx(2459297.419112, abs=1e-6)
    assert new.sd == pytest.approx(0.001982325906605672, rel=1e-12)


@pytest.mark.parametrize("n", [3, -3])
def test_uniform_uniform_widens_with_period_bounds(n):
    new = shift_prior(Uniform(9.9, 10.1), Uniform(2.9, 3.1), n, 3.0)
    lo, hi = (2.9, 3.1) if n > 0 else (3.1, 2.9)
    assert new == Uniform(9.9 + n * lo, 10.1 + n * hi)


def test_trunc_normal_pair():
    new = shift_prior(
        TruncNormal(9.0, 11.0, 10.0, 0.1), TruncNormal(2.0, 4.0, 3.0, 0.01), 2, 3.0
    )
    assert new == TruncNormal(13.0, 19.0, 16.0, math.hypot(0.1, 0.02))


@pytest.mark.parametrize(
    "period_prior", [Normal(3.0, 0.01), TruncNormal(2, 4, 3.0, 0.01)]
)
def test_uniform_epoch_with_gaussian_period(period_prior):
    new = shift_prior(Uniform(9.9, 10.1), period_prior, 2, 3.0)
    assert new == Uniform(9.9 + 2 * 3.01, 10.1 + 2 * 3.01)


def test_fixed_period_translates_exactly():
    assert shift_prior(Normal(10.0, 0.1), None, 2, 3.0) == Normal(16.0, 0.1)
    assert shift_prior(Uniform(9, 11), None, -1, 3.0) == Uniform(6.0, 8.0)
    assert shift_prior(TruncNormal(9, 11, 10, 0.1), None, 1, 3.0) == TruncNormal(
        12.0, 14.0, 13.0, 0.1
    )


def test_mismatched_priors_are_rejected_like_allesfitter():
    with pytest.raises(EpochShiftError, match="Normal epoch"):
        shift_prior(Normal(10.0, 0.1), Uniform(2.9, 3.1), 2, 3.0)


def test_shift_epochs_updates_value_and_prior_and_reports_shifts():
    settings = parse_settings_text("companions_phot,b\ninst_phot,tess\n")
    params = parse_params_text(
        "#name,value,fit,bounds,label,unit\n"
        "b_time_transit,10.0,1,normal 10.0 0.001,,\nb_period,3.0,1,normal 3.0 0.0001,,\n"
    )
    t = np.linspace(100, 130, 300)
    data = {"tess": Dataset("tess", "flux", t, np.ones(300), np.full(300, 1e-3))}
    shifted, shifts = shift_epochs(settings, params, data)
    n = shifts["b"]
    assert n > 0
    assert shifted["b_time_transit"].value == pytest.approx(10.0 + n * 3.0)
    assert shifted["b_time_transit"].prior == Normal(
        10.0 + n * 3.0, math.hypot(0.001, n * 1e-4)
    )
    assert params["b_time_transit"].value == 10.0  # the input table is untouched


def test_loguniform_epoch_prior_cannot_be_shifted():
    from jaxoplanet2.io.priors import LogUniform

    with pytest.raises(EpochShiftError, match="loguniform"):
        shift_prior(LogUniform(1.0, 2.0), Uniform(2.9, 3.1), 2, 3.0)
