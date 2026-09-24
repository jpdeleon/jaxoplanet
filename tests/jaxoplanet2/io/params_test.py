import pytest

from jaxoplanet2.init import init_directory
from jaxoplanet2.io.params import ParamsError, load_params, parse_params_text
from jaxoplanet2.io.priors import Normal, TruncNormal, Uniform

EXAMPLE = r"""#name,value,fit,bounds,label,unit,truth
#companion b astrophysical params,,,,,,
b_rr,0.0237,1,uniform 0 0.1000,$R_b / R_\star$,,
b_epoch,2459169.619940,1,normal 2459169.619940 0.001420,$T_{0;b}$,BJD,
b_f_c,0,0,uniform 0.0 0.0,$\sqrt{e_b} \cos{\omega_b}$,,
host_ldc_q1_tess,0.43,1,trunc_normal 0 1 0.43 0.07,$q_{1; \mathrm{tess}}$,,0.4
#baseline_gp_offset_flux_tess,0,1,uniform -0.1 0.1,$\mathrm{gp}$,,
"""


def test_parses_real_allesfitter_params():
    t = parse_params_text(EXAMPLE)
    assert t.names == ("b_rr", "b_epoch", "b_f_c", "host_ldc_q1_tess")
    rr = t["b_rr"]
    assert rr.value == pytest.approx(0.0237)
    assert rr.fit is True
    assert rr.prior == Uniform(0.0, 0.1)
    assert rr.label == r"$R_b / R_\star$"
    assert rr.unit == ""
    assert rr.truth is None
    assert t["b_epoch"].prior == Normal(2459169.61994, 0.00142)
    assert t["b_epoch"].unit == "BJD"
    assert t["host_ldc_q1_tess"].prior == TruncNormal(0, 1, 0.43, 0.07)
    assert t["host_ldc_q1_tess"].truth == pytest.approx(0.4)


def test_fixed_params_have_no_prior():
    t = parse_params_text(EXAMPLE)
    assert t["b_f_c"].fit is False
    assert t["b_f_c"].prior is None
    assert [p.name for p in t.free] == ["b_rr", "b_epoch", "host_ldc_q1_tess"]
    assert [p.name for p in t.fixed] == ["b_f_c"]


def test_header_without_truth_column():
    t = parse_params_text(
        "#name,value,fit,bounds,label,unit\nb_rr,0.1,1,uniform 0 1,r,\n"
    )
    assert t["b_rr"].truth is None


def test_backslash_header_and_leading_comment():
    text = "#See the tutorial\n\\#name,value,fit,bounds,label,unit\nb_rr,0.1,0,,r,\n"
    assert parse_params_text(text).names == ("b_rr",)


def test_missing_header_uses_default_columns():
    t = parse_params_text("b_rr,0.1,1,uniform 0 1,r,,0.09\n")
    assert t["b_rr"].truth == pytest.approx(0.09)


def test_inline_comments_are_stripped_like_genfromtxt():
    t = parse_params_text(
        "#name,value,fit,bounds,label,unit\nb_rr,0.1,1,uniform 0 1,r, # was 0.2\n"
    )
    assert t["b_rr"].unit == ""


def test_values_mapping_and_contains():
    t = parse_params_text(EXAMPLE)
    assert "b_rr" in t
    assert "b_K" not in t
    assert t.values()["b_f_c"] == 0.0
    assert len(t) == 4


def test_with_values_returns_new_table():
    t = parse_params_text(EXAMPLE)
    t2 = t.with_values({"b_rr": 0.03})
    assert t2["b_rr"].value == 0.03
    assert t["b_rr"].value == pytest.approx(0.0237)
    with pytest.raises(ParamsError, match="b_K"):
        t.with_values({"b_K": 1.0})


def test_params_are_immutable():
    t = parse_params_text(EXAMPLE)
    with pytest.raises(AttributeError):
        t["b_rr"].value = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "row, match",
    [
        ("b_rr,abc,1,uniform 0 1,r,", "value"),
        ("b_rr,0.1,2,uniform 0 1,r,", "fit"),
        ("b_rr,0.1,1,,r,", "bounds"),
        ("b_rr,0.1,1,gaussian 0 1,r,", "gaussian"),
        ("b_rr,0.5,1,uniform 0 0.1,r,", "outside"),
        ("b_rr,0.5,1,trunc_normal 0 0.1 0.05 0.01,r,", "outside"),
        (",0.5,1,uniform 0 1,r,", "name"),
    ],
)
def test_row_errors_name_the_line(row, match):
    with pytest.raises(ParamsError, match=match) as excinfo:
        parse_params_text("#name,value,fit,bounds,label,unit\n" + row + "\n")
    assert "line 2" in str(excinfo.value)


@pytest.mark.parametrize("flag, expected", [("1.0", True), ("0.0", False)])
def test_float_fit_flags_are_accepted(flag, expected):
    t = parse_params_text(
        f"#name,value,fit,bounds,label,unit\nb_rr,0.1,{flag},uniform 0 1,r,\n"
    )
    assert t["b_rr"].fit is expected


def test_far_from_normal_prior_warns():
    with pytest.warns(UserWarning, match="3 sigma"):
        parse_params_text("#name,value,fit,bounds,label,unit\nb_x,10,1,normal 0 1,x,\n")


def test_duplicate_names_are_an_error():
    text = "#name,value,fit,bounds,label,unit\nb_rr,0.1,0,,r,\nb_rr,0.2,0,,r,\n"
    with pytest.raises(ParamsError, match="b_rr"):
        parse_params_text(text)


def test_coupled_with_column_is_unsupported():
    text = "#name,value,fit,bounds,label,unit,truth,coupled_with\nb_rr,0.1,0,,r,,,c_rr\n"
    with pytest.raises(ParamsError, match="coupled_with"):
        parse_params_text(text)


def test_empty_coupled_with_column_is_fine():
    text = "#name,value,fit,bounds,label,unit,truth,coupled_with\nb_rr,0.1,0,,r,,,\n"
    assert parse_params_text(text).names == ("b_rr",)


def test_unknown_column_is_an_error():
    with pytest.raises(ParamsError, match="colour"):
        parse_params_text("#name,value,fit,bounds,colour\nb_rr,0.1,0,,red\n")


def test_load_params_reads_template(tmp_path):
    init_directory(tmp_path)
    t = load_params(tmp_path)
    assert "b_rr" in t
    assert all(p.prior is not None for p in t.free)


def test_load_params_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="params.csv"):
        load_params(tmp_path)
