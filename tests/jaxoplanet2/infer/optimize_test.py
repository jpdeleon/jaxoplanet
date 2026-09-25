import json

import numpy as np
import pytest
from typer.testing import CliRunner

from jaxoplanet2.cli import app
from jaxoplanet2.infer.optimize import OptimizeError, optimize
from jaxoplanet2.io.params import load_params
from tests.jaxoplanet2.synthetic import TRUTH, make_fit_dir

runner = CliRunner()
START = {
    "b_radius_ratio": 0.08,
    "b_duration": 0.12,
    "b_time_transit": 2459000.405,
    "ln_err_flux_tess": -6.5,
}


@pytest.fixture
def fit_dir(tmp_path):
    return make_fit_dir(tmp_path / "fit", start=START)


def test_recovers_the_truth_and_updates_params(fit_dir):
    result = optimize(fit_dir, quiet=True)
    assert result.accepted, result.reject_reason
    assert result.delta_lnprob > 0.5 * len(result.fitkeys)
    params = load_params(fit_dir)
    k = params["b_radius_ratio"].value
    assert k == pytest.approx(TRUTH["b_radius_ratio"], abs=0.003)
    t0 = params["b_time_transit"].value
    assert t0 == pytest.approx(TRUTH["b_time_transit"], abs=5e-4)
    assert (fit_dir / "params.csv.orig").exists()
    assert "0.08" in (fit_dir / "params.csv.orig").read_text()


def test_writes_json_summary_and_optimized_table(fit_dir):
    result = optimize(fit_dir, quiet=True, update_params=False)
    saved = json.loads((fit_dir / "results" / "optimize_save.json").read_text())
    assert saved["accepted"] == result.accepted
    assert saved["fitkeys"] == list(result.fitkeys)
    optimized = load_params(fit_dir / "results", "params_optimized.csv")
    k = optimized["b_radius_ratio"].value
    assert k == pytest.approx(TRUTH["b_radius_ratio"], abs=0.003)


def test_no_update_leaves_params_untouched(fit_dir):
    before = (fit_dir / "params.csv").read_text()
    optimize(fit_dir, quiet=True, update_params=False)
    assert (fit_dir / "params.csv").read_text() == before
    assert not (fit_dir / "params.csv.orig").exists()


def test_rejects_when_it_cannot_beat_the_start(tmp_path):
    fit_dir = make_fit_dir(tmp_path / "fit")  # starts at the truth
    optimize(fit_dir, quiet=True)  # moves to the noise optimum
    before = (fit_dir / "params.csv").read_text()
    result = optimize(fit_dir, quiet=True)
    assert not result.accepted
    assert "improvement" in result.reject_reason
    assert (fit_dir / "params.csv").read_text() == before


def test_shifted_epoch_is_written_back_in_the_original_frame(tmp_path):
    start = {"b_time_transit": 2459000.405 - 10 * 3.2}  # data start 10 orbits later
    prior = "b_time_transit,{},1,uniform 2458968.3 2458968.5"
    fit_dir = make_fit_dir(tmp_path / "fit", start=start, shift_epoch=True)
    text = (fit_dir / "params.csv").read_text().splitlines()
    text = [
        prior.format(start["b_time_transit"]) + ",t0,,"
        if ln.startswith("b_time_transit")
        else ln
        for ln in text
    ]
    (fit_dir / "params.csv").write_text("\n".join(text) + "\n")
    result = optimize(fit_dir, quiet=True)
    assert result.accepted, result.reject_reason
    epoch = load_params(fit_dir)["b_time_transit"].value
    period = load_params(fit_dir)["b_period"].value
    # cycle-equivalent to the truth, but still near the user's original epoch
    assert epoch == pytest.approx(TRUTH["b_time_transit"] - 10 * period, abs=1e-3)


def test_optimum_on_a_bound_is_rejected_unless_skipped(tmp_path):
    # a central transit: the optimum sits at the cosi = 0 prior edge
    fit_dir = make_fit_dir(tmp_path / "fit", start={**START, "b_impact_param": 0.0})
    text = (
        (fit_dir / "params.csv")
        .read_text()
        .replace(
            "b_impact_param,0.0,1,uniform 0.0 1.2",
            "b_impact_param,0.0,1,uniform 0.0 0.001",
        )
    )
    (fit_dir / "params.csv").write_text(text)
    result = optimize(fit_dir, quiet=True, update_params=False)
    assert "prior bounds" in result.reject_reason
    result = optimize(fit_dir, quiet=True, update_params=False, skip_bounds_check=True)
    assert result.accepted


def test_differential_evolution_then_refine(fit_dir):
    result = optimize(fit_dir, method="differential_evolution", quiet=True, maxiter=30)
    assert result.method == "differential_evolution"
    assert np.isfinite(result.lnprob_opt)


def test_unknown_method(fit_dir):
    with pytest.raises(OptimizeError, match="cmaes"):
        optimize(fit_dir, method="cmaes", quiet=True)


def test_cli_optimize(fit_dir):
    result = runner.invoke(app, ["optimize", str(fit_dir), "-n", "2", "--seed", "1"])
    assert result.exit_code == 0, result.output
    assert "accepted" in result.output
    assert (fit_dir / "results" / "initial_guess_tess.pdf").exists()
