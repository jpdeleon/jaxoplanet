"""Regenerate the allesfitter reference models used by golden_test.py.

Needs an environment with allesfitter (and ellc/batman) installed; CI does not
run this, it only compares against the committed ``<case>.npz`` files::

    python tests/jaxoplanet2/golden/generate_golden.py

For each fit directory in ``cases/`` this initialises allesfitter, evaluates its
model at the params.csv values on every instrument's time stamps, and stores the
result together with the allesfitter version that produced it.
"""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def reference_models(case_dir: Path) -> dict[str, np.ndarray]:
    import allesfitter
    from allesfitter import computer, config

    config.init(str(case_dir), quiet=True)
    basement = config.BASEMENT
    params = computer.update_params(basement.theta_0)
    out = {"allesfitter_version": np.array(allesfitter.__version__)}
    for key, insts in (("flux", "inst_phot"), ("rv", "inst_rv")):
        for inst in basement.settings[insts]:
            model = computer.calculate_model(params, inst, key)
            out[inst] = np.asarray(model)
            out[f"{inst}_loglike"] = np.asarray(_loglike(basement, params, inst, key))
            out[f"{inst}_baseline"] = np.asarray(
                computer.calculate_baseline(params, inst, key, model=model)
            )
    return out


def _loglike(basement, params, inst, key):
    """allesfitter's log-likelihood of one instrument (white noise or GP)."""
    from allesfitter import computer

    model = computer.calculate_model(params, inst, key)
    yerr_w = computer.calculate_yerr_w(params, inst, key)
    residuals = basement.data[inst][key] - model
    if "GP" in basement.settings[f"baseline_{key}_{inst}"]:
        gp = computer.baseline_get_gp(params, inst, key)
        gp.compute(basement.data[inst]["time"], yerr=yerr_w)
        return gp.log_likelihood(residuals)
    baseline = computer.calculate_baseline(params, inst, key, model=model, yerr_w=yerr_w)
    residuals = residuals - baseline
    return -0.5 * np.sum(residuals**2 / yerr_w**2 + np.log(2 * np.pi * yerr_w**2))


def main() -> None:
    warnings.simplefilter("ignore")
    for case_dir in sorted((HERE / "cases").iterdir()):
        models = reference_models(case_dir)
        shutil.rmtree(case_dir / "results", ignore_errors=True)  # allesfitter's logs
        np.savez(HERE / f"{case_dir.name}.npz", **models)
        insts = ", ".join(k for k in models if k != "allesfitter_version")
        print(f"{case_dir.name}: {insts}", file=sys.stderr)


if __name__ == "__main__":
    main()
