"""Generate the pages that come from code, so they cannot drift from it.

Run by the mkdocs-gen-files plugin at build time:

- ``cli.md``: the typer CLI reference (same text as ``jaxoplanet --help``),
- ``examples/kepler1627.md``: the example's README,
- ``api/``: one mkdocstrings page per public ``jaxoplanet2`` module.
"""

from pathlib import Path

import mkdocs_gen_files
import typer
from typer.cli import get_docs_for_click
from typer.main import get_command

from jaxoplanet2.cli import app

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "src" / "jaxoplanet2"
EXAMPLE = ROOT / "examples" / "jaxoplanet2" / "kepler1627" / "README.md"
# entry points and CLI glue: documented by cli.md instead
SKIP = {"cli", "__main__"}


def cli_reference() -> str:
    command = get_command(app)
    return get_docs_for_click(
        obj=command,
        ctx=typer.Context(command),
        name="jaxoplanet",
        title="CLI reference",
    )


def api_modules() -> list[tuple[str, ...]]:
    modules = []
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if any(p.startswith("_") for p in parts) or parts[-1] in SKIP:
            continue
        modules.append(parts)
    return modules


with mkdocs_gen_files.open("cli.md", "w") as f:
    f.write(cli_reference())

with mkdocs_gen_files.open("examples/kepler1627.md", "w") as f:
    f.write(EXAMPLE.read_text())
mkdocs_gen_files.set_edit_path("examples/kepler1627.md", EXAMPLE.relative_to(ROOT))

nav = mkdocs_gen_files.Nav()
for parts in api_modules():
    page = Path("api", *parts[1:]).with_suffix(".md") if len(parts) > 1 else None
    page = page or Path("api", "index.md")
    nav[parts[1:] or ("jaxoplanet2",)] = page.relative_to("api").as_posix()
    with mkdocs_gen_files.open(page, "w") as f:
        f.write(f"::: {'.'.join(parts)}\n")
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as f:
    f.writelines(nav.build_literate_nav())
