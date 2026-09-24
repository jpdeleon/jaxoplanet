"""Build synthetic allesfitter-format fit directories with a known truth.

The data are simulated with jaxoplanet2's own model (parity with allesfitter is
covered by the golden tests), so fits can be checked against the truth.
"""

from pathlib import Path

import jax
import numpy as np

from jaxoplanet2.io.settings import parse_settings_text
from jaxoplanet2.model.photometry import flux_model
from jaxoplanet2.model.rv import rv_model

jax.config.update("jax_enable_x64", True)

TRUTH = {
    "b_rr": 0.1,
    "b_rsuma": 0.1,
    "b_cosi": 0.02,
    "b_epoch": 2459000.4,
    "b_period": 3.2,
    "b_K": 0.02,
    "host_ldc_q1_tess": 0.4,
    "host_ldc_q2_tess": 0.3,
    "ln_err_flux_tess": np.log(1e-3),
    # above the 3 m/s errors, so the jitter is constrained (a sub-error jitter
    # leaves a flat likelihood plateau towards ln_jitter -> -15 that NUTS
    # can only cross with very deep trees)
    "ln_jitter_rv_harps": np.log(6e-3),
    "baseline_offset_rv_harps": 0.01,
}
PRIORS = {
    "b_rr": "uniform 0.0 0.3",
    "b_rsuma": "uniform 0.0 0.5",
    "b_cosi": "uniform 0.0 1.0",
    "b_epoch": "uniform 2459000.3 2459000.5",
    "b_period": "uniform 3.1 3.3",
    "b_K": "uniform 0.0 0.1",
    "host_ldc_q1_tess": "uniform 0.0 1.0",
    "host_ldc_q2_tess": "uniform 0.0 1.0",
    "ln_err_flux_tess": "uniform -15.0 0.0",
    "ln_jitter_rv_harps": "uniform -15.0 0.0",
    "baseline_offset_rv_harps": "uniform -0.1 0.1",
}


def make_fit_dir(
    path: Path,
    *,
    rv: bool = False,
    start: dict | None = None,
    shift_epoch: bool = False,
    extra_settings: str = "",
    seed: int = 0,
) -> Path:
    """Write settings/params/data to ``path``; params start at ``start``."""
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    settings_text = (
        "#name,value\ncompanions_phot,b\n"
        f"companions_rv,{'b' if rv else ''}\ninst_phot,tess\n"
        f"inst_rv,{'harps' if rv else ''}\n"
        f"shift_epoch,{shift_epoch}\nuse_host_density_prior,False\n"
        "host_ld_law_tess,quad\n"
        + ("baseline_rv_harps,sample_offset\n" if rv else "")
        + extra_settings
    )
    (path / "settings.csv").write_text(settings_text)
    settings = parse_settings_text(settings_text)

    names = [n for n in TRUTH if rv or not n.endswith("harps") and n != "b_K"]
    values = {**{n: TRUTH[n] for n in names}, **(start or {})}
    rows = ["#name,value,fit,bounds,label,unit,truth"]
    rows += [
        f"{n},{float(values[n])!r},1,{PRIORS[n]},{n},,{float(TRUTH[n])!r}" for n in names
    ]
    (path / "params.csv").write_text("\n".join(rows) + "\n")

    # three transits of 2-min photometry, plus RVs across the orbit
    t = np.concatenate(
        [np.arange(-0.2, 0.2, 2 / 1440) + TRUTH["b_epoch"] + k * TRUTH["b_period"]
         for k in range(3)]
    )  # fmt: skip
    truth = {n: TRUTH[n] for n in names}
    flux = np.asarray(flux_model(truth, settings, "tess", t))
    _write(path / "tess.csv", t, flux + rng.normal(0, 1e-3, t.size), 1e-3)
    if rv:
        tr = np.sort(TRUTH["b_epoch"] + rng.uniform(0, 30, 40))
        signal = np.asarray(rv_model(truth, settings, "harps", tr))
        noise = rng.normal(
            0, np.hypot(3e-3, np.exp(TRUTH["ln_jitter_rv_harps"])), tr.size
        )
        _write(path / "harps.csv", tr, signal + 0.01 + noise, 3e-3)
    return path


def _write(path: Path, t: np.ndarray, y: np.ndarray, err: float) -> None:
    rows = (f"{float(a)!r},{float(b)!r},{err!r}" for a, b in zip(t, y, strict=True))
    path.write_text("\n".join(rows) + "\n")
