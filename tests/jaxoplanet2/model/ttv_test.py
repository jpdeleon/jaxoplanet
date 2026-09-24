import jax
import numpy as np
import pytest

from jaxoplanet2.fitdir import load_fit_directory
from jaxoplanet2.model.photometry import flux_model
from jaxoplanet2.model.ttv import TtvError, TtvWindows, observed_transits
from jaxoplanet2.validate import validate
from tests.jaxoplanet2.synthetic import TRUTH, make_fit_dir

jax.config.update("jax_enable_x64", True)
EPOCH, PERIOD = TRUTH["b_epoch"], TRUTH["b_period"]


def test_observed_transits_skip_data_gaps():
    # data around transits 0, 1 and 3 (none near transit 2)
    time = np.concatenate(
        [EPOCH + k * PERIOD + np.linspace(-0.1, 0.1, 50) for k in (0, 1, 3)]
    )
    tmids = observed_transits(time, EPOCH, PERIOD, width=0.3)
    np.testing.assert_allclose(tmids, [EPOCH, EPOCH + PERIOD, EPOCH + 3 * PERIOD])


def test_window_index():
    w = TtvWindows("b", (10.0, 13.0), 0.4)
    np.testing.assert_array_equal(
        w.index(np.array([9.9, 10.3, 12.85, 20.0])), [0, -1, 1, -1]
    )


def ttv_fit(tmp_path, ttvs=(0.0, 0.0, 0.0), rows=None):
    fit = make_fit_dir(
        tmp_path / "fit", extra_settings="fit_ttvs,True\nfast_fit_width,0.4\n"
    )
    n = len(ttvs) if rows is None else rows
    with (fit / "params.csv").open("a") as f:
        for i in range(n):
            f.write(f"b_ttv_transit_{i + 1},{ttvs[i]},1,uniform -0.05 0.05,ttv,d,\n")
    return fit


def test_row_count_must_match_the_covered_transits(tmp_path):
    fit = ttv_fit(tmp_path, rows=2)
    with pytest.raises(TtvError, match="b_ttv_transit_1..3"):
        load_fit_directory(fit)


def test_zero_ttvs_reproduce_the_linear_ephemeris(tmp_path):
    fit = load_fit_directory(ttv_fit(tmp_path))
    t = fit.data["tess"].time
    values = fit.params.values()
    with_ttv = flux_model(values, fit.settings, "tess", t, fit.ttv)
    linear = flux_model(values, fit.settings, "tess", t)
    np.testing.assert_allclose(with_ttv, linear, atol=1e-12)


def test_each_transit_moves_by_its_own_ttv(tmp_path):
    shift = 0.004  # ~6 minutes, on the second transit only
    fit = load_fit_directory(ttv_fit(tmp_path, ttvs=(0.0, shift, 0.0)))
    t = fit.data["tess"].time
    values = fit.params.values()
    got = np.asarray(flux_model(values, fit.settings, "tess", t, fit.ttv))
    idx = fit.ttv["b"].index(t)
    linear_shifted = np.asarray(flux_model(values, fit.settings, "tess", t - shift))
    linear = np.asarray(flux_model(values, fit.settings, "tess", t))
    np.testing.assert_allclose(got[idx == 1], linear_shifted[idx == 1], atol=1e-12)
    np.testing.assert_allclose(got[idx != 1], linear[idx != 1], atol=1e-12)


def test_ttv_gradient(tmp_path):
    fit = load_fit_directory(ttv_fit(tmp_path))
    t = fit.data["tess"].time
    base = fit.params.values()

    def f(ttv2):
        return flux_model(
            {**base, "b_ttv_transit_2": ttv2}, fit.settings, "tess", t, fit.ttv
        ).sum()

    assert abs(float(jax.grad(f)(0.001))) > 0


def test_ttv_fit_validates(tmp_path):
    report = validate(ttv_fit(tmp_path))
    assert report.ok, report.errors
    assert not report.warnings
