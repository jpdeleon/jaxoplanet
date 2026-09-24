"""Write results back into allesfitter-format files without disturbing them."""

import shutil
from collections.abc import Mapping
from pathlib import Path

from jaxoplanet2.io.params import PARAMS_FILE

BACKUP_SUFFIX = ".orig"


def backup_params(fit_dir: str | Path) -> Path:
    """Copy params.csv to params.csv.orig, once: the backup keeps the original."""
    src = Path(fit_dir) / PARAMS_FILE
    dst = src.with_name(PARAMS_FILE + BACKUP_SUFFIX)
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def write_param_values(path: str | Path, values: Mapping[str, float]) -> None:
    """Replace the value cell of the named rows; every other byte is kept."""
    path = Path(path)
    lines = path.read_text().splitlines(keepends=True)
    pending = dict(values)
    out = []
    for line in lines:
        name, sep, rest = line.partition(",")
        if line.lstrip().startswith("#") or name.strip() not in pending or not sep:
            out.append(line)
            continue
        _, _, tail = rest.partition(",")
        out.append(f"{name},{float(pending.pop(name.strip()))!r},{tail}")
    if pending:
        raise KeyError(f"not in {path.name}: {', '.join(sorted(pending))}")
    path.write_text("".join(out))
