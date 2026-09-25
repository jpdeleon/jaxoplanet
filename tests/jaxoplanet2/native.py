"""Write test inputs with familiar allesfitter numbers, run them natively.

``from_allesfitter`` turns ``<c>_rr / _rsuma / _cosi / _epoch`` into jaxoplanet's
native ``radius_ratio / duration / impact_param / time_transit`` with the exact
forward map, leaving every other key untouched.
"""

import atexit
import functools
import math
import shutil
import tempfile
from pathlib import Path

import jax

from jaxoplanet2.io.convert import convert_params_file
from jaxoplanet2.model.parameterization import transit_from_orbit


def from_allesfitter(values: dict) -> dict:
    out = dict(values)
    companions = {k[: -len("_epoch")] for k in values if k.endswith("_epoch")}
    for c in companions:
        out[f"{c}_time_transit"] = out.pop(f"{c}_epoch")
        if f"{c}_rr" in out:
            out[f"{c}_radius_ratio"] = out.pop(f"{c}_rr")
        if f"{c}_rsuma" not in out:
            continue
        k = out.get(f"{c}_radius_ratio", 0.0)
        f_c, f_s = out.get(f"{c}_f_c", 0.0), out.get(f"{c}_f_s", 0.0)
        ecc = f_c**2 + f_s**2
        with jax.ensure_compile_time_eval():  # stays concrete inside jax.jit
            b, t14 = transit_from_orbit(
                k,
                (1 + k) / out.pop(f"{c}_rsuma"),
                math.acos(out.pop(f"{c}_cosi", 0.0)),
                out[f"{c}_period"],
                ecc=ecc,
                sin_omega=f_s / math.sqrt(ecc) if ecc > 0 else 1.0,
            )
        out[f"{c}_impact_param"], out[f"{c}_duration"] = float(b), float(t14)
    return out


_GOLDEN = Path(__file__).parent / "golden" / "cases"


@functools.cache
def native_golden() -> Path:
    """A converted copy of the golden cases (allesfitter format in the repo)."""
    root = Path(tempfile.mkdtemp(prefix="jaxoplanet2_golden_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    for case in _GOLDEN.iterdir():
        dest = root / case.name
        shutil.copytree(case, dest, ignore=shutil.ignore_patterns("results"))
        convert_params_file(dest)
    return root
