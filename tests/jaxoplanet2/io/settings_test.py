import warnings

import pytest

from jaxoplanet2.init import init_directory
from jaxoplanet2.io.settings import (
    SettingsError,
    UnsupportedSettingsError,
    load_settings,
    parse_settings_text,
)

EXAMPLE = """#name,value
###############################################################################,
# General settings,
###############################################################################,
companions_phot,b
companions_rv,b c
inst_phot,tess
inst_rv,harps
multiprocess,True
multiprocess_cores,40
fast_fit,True
fast_fit_width,0.3333333333333333
#fast_fit_width,0.5
secondary_eclipse,False
phase_curve,False
shift_epoch,True
inst_for_b_epoch,all
mcmc_nwalkers,100
mcmc_total_steps,2000
mcmc_burn_steps,1000
mcmc_thin_by,2
ns_modus,dynamic
ns_nlive,1000
host_ld_law_tess,quad
# t_exp_tess,0.0208333
baseline_flux_tess,sample_GP_Matern32
baseline_rv_harps,sample_offset
error_flux_tess,sample
use_host_density_prior,True
fit_ttvs,False
host_grid_tess,very_sparse
"""


def parse(text, **kwargs):
    return parse_settings_text(text, **kwargs)


def test_parses_real_allesfitter_settings():
    s = parse(EXAMPLE)
    assert s.companions_phot == ("b",)
    assert s.companions_rv == ("b", "c")
    assert s.companions_all == ("b", "c")
    assert s.inst_phot == ("tess",)
    assert s.inst_rv == ("harps",)
    assert s.inst_all == ("tess", "harps")
    assert s.fast_fit is True
    assert s.fast_fit_width == pytest.approx(1 / 3)
    assert s.shift_epoch is True
    assert s.inst_for_epoch["b"] == "all"
    assert s.use_host_density_prior is True
    assert s.fit_ttvs is False


def test_mcmc_settings_map_to_nuts():
    m = parse(EXAMPLE).mcmc
    assert (m.nwalkers, m.total_steps, m.burn_steps, m.thin_by) == (100, 2000, 1000, 2)
    assert m.num_warmup == 1000
    assert m.num_samples == 1000


def test_per_instrument_settings():
    s = parse(EXAMPLE)
    assert s.ld_law["tess"] == "quad"
    assert s.ld_space["tess"] == "q"
    assert s.baseline[("flux", "tess")] == "sample_GP_Matern32"
    assert s.baseline[("rv", "harps")] == "sample_offset"
    assert s.error[("flux", "tess")] == "sample"
    assert s.error[("rv", "harps")] == "sample"
    assert s.t_exp["tess"] is None
    assert s.t_exp_n_int["tess"] is None


def test_defaults_match_allesfitter():
    s = parse("companions_phot,b\ninst_phot,tess\n")
    assert s.fast_fit is False
    assert s.fast_fit_width == pytest.approx(8 / 24)
    assert s.shift_epoch is True
    assert s.use_host_density_prior is True
    assert s.baseline[("flux", "tess")] == "none"
    assert s.error[("flux", "tess")] == "sample"
    assert s.ld_law["tess"] == "quad"
    assert s.inst_for_epoch["b"] == "all"
    assert (s.mcmc.nwalkers, s.mcmc.total_steps) == (100, 2000)


def test_explicit_none_disables_limb_darkening():
    s = parse("companions_phot,b\ninst_phot,tess\nhost_ld_law_tess,None\n")
    assert s.ld_law["tess"] is None


def test_empty_ld_law_means_quad_like_allesfitter2():
    s = parse("companions_phot,b\ninst_phot,tess\nhost_ld_law_tess,\n")
    assert s.ld_law["tess"] == "quad"


def test_jx_settings_defaults_and_overrides():
    s = parse("companions_phot,b\ninst_phot,tess\n")
    assert s.jx.x64 is True
    assert s.jx.seed == 42
    assert s.jx.num_chains is None
    s = parse("companions_phot,b\ninst_phot,tess\njx_seed,7\njx_num_chains,4\n")
    assert s.jx.seed == 7
    assert s.jx.num_chains == 4


def test_exposure_settings():
    s = parse(
        "companions_phot,b\ninst_phot,kepler\nt_exp_kepler,0.0204\n"
        "t_exp_n_int_kepler,10\n"
    )
    assert s.t_exp["kepler"] == pytest.approx(0.0204)
    assert s.t_exp_n_int["kepler"] == 10


def test_legacy_keywords_are_renamed():
    with pytest.warns(DeprecationWarning, match="planets_phot"):
        s = parse("planets_phot,b\ninst_phot,tess\nld_law_tess,quad\n")
    assert s.companions_phot == ("b",)
    assert s.ld_law["tess"] == "quad"


def test_unsupported_keys_are_reported_together():
    text = "companions_phot,b\ninst_phot,tess\nN_flares,2\nphase_curve,True\n"
    with pytest.raises(UnsupportedSettingsError) as excinfo:
        parse(text)
    msg = str(excinfo.value)
    assert "N_flares" in msg
    assert "phase_curve" in msg
    assert excinfo.value.keys == ("N_flares", "phase_curve")


def test_unsupported_values_are_reported():
    text = "companions_phot,b\ninst_phot,tess\nbaseline_flux_tess,hybrid_spline\n"
    with pytest.raises(UnsupportedSettingsError, match="hybrid_spline"):
        parse(text)


def test_allow_unsupported_downgrades_to_warning():
    text = "companions_phot,b\ninst_phot,tess\nN_flares,2\n"
    with pytest.warns(UserWarning, match="N_flares"):
        s = parse(text, allow_unsupported=True)
    assert s.raw["N_flares"] == "2"


def test_switched_off_features_are_accepted():
    text = (
        "companions_phot,b\ninst_phot,tess\nN_flares,0\nphase_curve,False\n"
        "host_N_spots_tess,0\nstellar_var_flux,none\nflux_model,batman\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parse(text)


def test_inline_comments_are_stripped_like_genfromtxt():
    s = parse(
        "companions_phot,b\ninst_phot,tess\nbaseline_flux_tess,sample_GP_SHO #Matern32\n"
    )
    assert s.baseline[("flux", "tess")] == "sample_GP_SHO"


@pytest.mark.parametrize("value", ["No", "no", "False", "0", "off"])
def test_off_only_switches_use_allesfitter_set_bool(value):
    s = parse("companions_rv,b\ninst_rv,harps\nb_flux_weighted_harps," + value + "\n")
    assert s.raw["b_flux_weighted_harps"] == value


def test_off_only_switch_turned_on_is_unsupported():
    with pytest.raises(UnsupportedSettingsError, match="flux_weighted"):
        parse("companions_rv,b\ninst_rv,harps\nb_flux_weighted_harps,True\n")


def test_keys_for_unlisted_instruments_are_ignored_with_warning():
    with pytest.warns(UserWarning, match="host_ld_law_kepler"):
        s = parse("companions_phot,b\ninst_phot,tess\nhost_ld_law_kepler,quad\n")
    assert "kepler" not in s.ld_law


def test_keys_for_unlisted_companions_are_ignored_with_warning():
    with pytest.warns(UserWarning, match="inst_for_c_epoch"):
        s = parse("companions_phot,b\ninst_phot,tess\ninst_for_c_epoch,all\n")
    assert "c" not in s.inst_for_epoch


def test_use_stellar_density_is_legacy_alias():
    with pytest.warns(DeprecationWarning, match="use_stellar_density"):
        s = parse("companions_phot,b\ninst_phot,tess\nuse_stellar_density,False\n")
    assert s.use_host_density_prior is False


def test_use_stellar_density_prior_is_legacy_alias():
    with pytest.warns(DeprecationWarning, match="use_stellar_density_prior"):
        s = parse("companions_phot,b\ninst_phot,tess\nuse_stellar_density_prior,False\n")
    assert s.use_host_density_prior is False


def test_invalid_numbers_fail_with_key_name():
    with pytest.raises(SettingsError, match="mcmc_nwalkers"):
        parse("companions_phot,b\ninst_phot,tess\nmcmc_nwalkers,lots\n")


def test_burn_steps_must_be_smaller_than_total():
    with pytest.raises(SettingsError, match="mcmc_burn_steps"):
        parse(
            "companions_phot,b\ninst_phot,tess\n"
            "mcmc_total_steps,100\nmcmc_burn_steps,100\n"
        )


def test_needs_at_least_one_instrument():
    with pytest.raises(SettingsError, match="inst_phot"):
        parse("companions_phot,b\n")


def test_companions_need_matching_instruments():
    with pytest.raises(SettingsError, match="inst_rv"):
        parse("companions_phot,b\ncompanions_rv,b\ninst_phot,tess\n")


def test_duplicate_keys_are_an_error():
    with pytest.raises(SettingsError, match="fast_fit"):
        parse("companions_phot,b\ninst_phot,tess\nfast_fit,True\nfast_fit,False\n")


def test_settings_are_immutable():
    s = parse(EXAMPLE)
    with pytest.raises(AttributeError):
        s.fast_fit = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        s.raw["fast_fit"] = "False"  # type: ignore[index]


def test_load_settings_reads_template(tmp_path):
    init_directory(tmp_path)
    s = load_settings(tmp_path)
    assert s.companions_phot == ("b",)
    assert s.inst_phot == ("tess",)
    assert s.baseline[("flux", "tess")] == "sample_offset"


def test_load_settings_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="settings.csv"):
        load_settings(tmp_path)
