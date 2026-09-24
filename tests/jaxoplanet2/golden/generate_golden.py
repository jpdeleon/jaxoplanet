"""Regenerate the allesfitter reference models used by golden_test.py.

Needs an environment with allesfitter (and ellc/batman) installed; CI does not
run this, it only compares against the committed ``<case>.npz`` files::

    python tests/jaxoplanet2/golden/generate_golden.py

For each fit directory in ``cases/`` this initialises allesfitter, evaluates its
model at the params.csv values on every instrument's time stamps, and stores the
result together with the allesfitter version that produced it.
"""

from __future__ import annotations

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
    for inst in basement.settings["inst_phot"]:
        out[inst] = np.asarray(computer.calculate_model(params, inst, "flux"))
    for inst in basement.settings["inst_rv"]:
        out[inst] = np.asarray(computer.calculate_model(params, inst, "rv"))
    return out


def main() -> None:
    warnings.simplefilter("ignore")
    for case_dir in sorted((HERE / "cases").iterdir()):
        models = reference_models(case_dir)
        np.savez(HERE / f"{case_dir.name}.npz", **models)
        insts = ", ".join(k for k in models if k != "allesfitter_version")
        print(f"{case_dir.name}: {insts}", file=sys.stderr)


if __name__ == "__main__":
    main()
