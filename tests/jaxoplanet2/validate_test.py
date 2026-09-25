import shutil
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from jaxoplanet2.cli import app
from jaxoplanet2.io.params import parse_params_text
from jaxoplanet2.io.settings import parse_settings_text
from jaxoplanet2.model.requirements import check_params, required_params
from jaxoplanet2.validate import validate
from tests.jaxoplanet2.native import native_golden

GOLDEN = native_golden()  # golden cases converted to native parameters
runner = CliRunner()


def settings(text):
    return parse_settings_text(text)


def table(rows):
    return parse_params_text("#name,value,fit,bounds,label,unit\n" + rows)


PHOT = "companions_phot,b\ninst_phot,tess\n"
FULL_PHOT_PARAMS = (
    "b_radius_ratio,0.1,0,,,\nb_duration,0.1,0,,,\nb_impact_param,0,0,,,\n"
    "b_time_transit,1,0,,,\nb_period,3,0,,,\n"
    "host_ldc_q1_tess,0.4,0,,,\nhost_ldc_q2_tess,0.3,0,,,\nln_err_flux_tess,-7,0,,,\n"
)


def test_required_params_for_photometry():
    req = required_params(settings(PHOT))
    for name in (
        "b_radius_ratio",
        "b_duration",
        "b_impact_param",
        "b_time_transit",
        "b_period",
    ):
        assert name in req
    assert "host_ldc_q1_tess" in req
    assert "ln_err_flux_tess" in req


def test_required_params_for_rv_and_baselines():
    s = settings("companions_rv,b\ninst_rv,harps\nbaseline_rv_harps,sample_linear\n")
    req = required_params(s)
    for name in ("b_time_transit", "b_period", "b_K", "ln_jitter_rv_harps"):
        assert name in req
    assert "baseline_offset_rv_harps" in req
    assert "baseline_slope_rv_harps" in req
    assert "b_radius_ratio" not in req


def test_required_ld_follows_law_and_space():
    s = settings(PHOT + "host_ld_space_tess,u\n")
    assert "host_ldc_u1_tess" in required_params(s)
    s = settings(PHOT + "host_ld_law_tess,none\n")
    assert not any(k.startswith("host_ldc") for k in required_params(s))


def test_gp_baseline_requirements():
    s = settings(PHOT + "baseline_flux_tess,sample_GP_SHO\n")
    req = required_params(s)
    assert {"baseline_gp_sho_lnS0_flux_tess", "baseline_gp_sho_lnQ_flux_tess",
            "baseline_gp_sho_lnomega0_flux_tess"} <= set(req)  # fmt: skip


def test_check_params_complete_table_is_clean():
    check = check_params(settings(PHOT), table(FULL_PHOT_PARAMS))
    assert check.missing == ()
    assert check.unsupported == ()


def test_check_params_reports_missing():
    rows = FULL_PHOT_PARAMS.replace("ln_err_flux_tess,-7,0,,,\n", "")
    check = check_params(settings(PHOT), table(rows))
    assert check.missing == ("ln_err_flux_tess",)


def test_neutral_ellc_extras_are_ignored():
    rows = FULL_PHOT_PARAMS + "b_sbratio_tess,0,0,,,\nhost_gdc_tess,0.3,0,,,\n"
    check = check_params(settings(PHOT), table(rows))
    assert check.unsupported == ()
    assert set(check.ignored) == {"b_sbratio_tess", "host_gdc_tess"}


def test_leftover_dilution_of_other_instruments_is_ignored_at_zero():
    rows = FULL_PHOT_PARAMS + "dil_zess,0,0,,,\n"
    assert check_params(settings(PHOT), table(rows)).ignored == ("dil_zess",)


def test_fitted_or_nonzero_unsupported_extras_are_reported():
    rows = FULL_PHOT_PARAMS + (
        "b_sbratio_tess,0.1,0,,,\n"
        "b_phase_curve_beaming_tess,1,1,uniform 0 10,,\n"
        "mystery_param,1,0,,,\n"
    )
    check = check_params(settings(PHOT), table(rows))
    assert set(check.unsupported) == {
        "b_sbratio_tess",
        "b_phase_curve_beaming_tess",
        "mystery_param",
    }


def test_optional_params_are_accepted():
    rows = (
        FULL_PHOT_PARAMS + "b_f_c,0,0,,,\nb_f_s,0,0,,,\ndil_tess,0.1,0,,,\nb_K,0,0,,,\n"
    )
    check = check_params(settings(PHOT), table(rows))
    assert check.unsupported == ()


def test_ttv_params_are_ignored_unless_fit_ttvs():
    rows = FULL_PHOT_PARAMS + "b_ttv_transit_1,0,1,uniform -0.1 0.1,,\n"
    check = check_params(settings(PHOT), table(rows))
    assert check.ignored == ("b_ttv_transit_1",)


@pytest.mark.parametrize("case", ["circular_batman", "eccentric_rv_ellc"])
def test_golden_cases_validate(case):
    report = validate(GOLDEN / case)
    assert report.ok, report.errors
    assert any("log-likelihood" in line for line in report.info)


def test_validate_reports_missing_params(tmp_path):
    shutil.copytree(GOLDEN / "circular_batman", tmp_path / "fit")
    params = tmp_path / "fit" / "params.csv"
    params.write_text(
        "\n".join(ln for ln in params.read_text().splitlines() if "ln_err" not in ln)
    )
    report = validate(tmp_path / "fit")
    assert not report.ok
    assert any("ln_err_flux_tess" in e for e in report.errors)


def test_validate_reports_unsupported_settings(tmp_path):
    shutil.copytree(GOLDEN / "circular_batman", tmp_path / "fit")
    with (tmp_path / "fit" / "settings.csv").open("a") as f:
        f.write("secondary_eclipse,True\n")
    assert not validate(tmp_path / "fit").ok
    assert validate(tmp_path / "fit", allow_unsupported=True).ok


def test_validate_reports_bad_data(tmp_path):
    shutil.copytree(GOLDEN / "circular_batman", tmp_path / "fit")
    (tmp_path / "fit" / "tess.csv").write_text("1,nan,0.1\n")
    report = validate(tmp_path / "fit")
    assert any("NaN" in e for e in report.errors)


def test_validate_reports_non_finite_likelihood(tmp_path):
    shutil.copytree(GOLDEN / "circular_batman", tmp_path / "fit")
    params = tmp_path / "fit" / "params.csv"
    text = params.read_text().replace(
        "ln_err_flux_tess,-7.0,1,uniform -15 0", "ln_err_flux_tess,1000.0,0,"
    )
    params.write_text(text)
    report = validate(tmp_path / "fit")
    assert any("not finite" in e for e in report.errors)


def test_validate_missing_directory(tmp_path):
    report = validate(tmp_path / "nope")
    assert not report.ok


def test_cli_validate_ok_and_failure(tmp_path):
    ok = runner.invoke(app, ["validate", str(GOLDEN / "circular_batman")])
    assert ok.exit_code == 0, ok.output
    assert "OK" in ok.output
    shutil.copytree(GOLDEN / "circular_batman", tmp_path / "fit")
    (tmp_path / "fit" / "tess.csv").write_text("1,nan,0.1\n")
    bad = runner.invoke(app, ["validate", str(tmp_path / "fit")])
    assert bad.exit_code == 1
    assert "NaN" in bad.output


def test_initial_values_are_used(tmp_path):
    report = validate(GOLDEN / "circular_batman")
    ll = [line for line in report.info if "log-likelihood" in line]
    assert ll and np.isfinite(float(ll[0].split()[-1]))


def test_allesfitter_parameters_point_to_convert_params(tmp_path):
    rows = (
        "b_rr,0.1,0,,,\nb_rsuma,0.1,0,,,\nb_cosi,0,0,,,\nb_epoch,1,0,,,\n"
        "b_period,3,0,,,\n"
    )
    check = check_params(settings(PHOT), table(rows))
    assert set(check.legacy) == {"b_rr", "b_rsuma", "b_cosi", "b_epoch"}
    shutil.copytree(
        Path(__file__).parent / "golden" / "cases" / "circular_batman", tmp_path / "fit"
    )
    report = validate(tmp_path / "fit")
    assert not report.ok
    assert any("convert-params" in e for e in report.errors)
