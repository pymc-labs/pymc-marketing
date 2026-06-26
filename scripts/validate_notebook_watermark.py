"""Validate that all Jupyter notebooks have a watermark cell as their last cell.

Used as a pre-commit hook to ensure every notebook includes a watermark
with the standard flags and required packages.

Usage:
    python scripts/validate_notebook_watermark.py [notebooks...]

If no notebooks are given, scans docs/source/ recursively for ``.ipynb`` files
(excluding ``dev/`` directories), matching the pre-commit hook's file matcher.
Directories are expanded to include all ``.ipynb`` files within.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
DOC_SOURCE = HERE.parent / "docs" / "source"

REQUIRED_FLAGS = {"-n", "-u", "-v", "-iv", "-w"}
REQUIRED_PACKAGES: set[str] = set()

EXCLUDE_SUBDIRS = {"dev"}
EXCLUDE_DIR_PREFIXES = {"."}

WATERMARK_RE = re.compile(
    r"-p (?P<packages>[a-zA-Z_][a-zA-Z0-9_.]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_.]*)*)"
)

IMPORT_RE = re.compile(r"(?:^|\n)\s*(?:import pymc_marketing|from pymc_marketing)")


def is_excluded(path: Path) -> bool:
    return any(
        part in EXCLUDE_SUBDIRS or any(part.startswith(p) for p in EXCLUDE_DIR_PREFIXES)
        for part in path.parts
    )


def load_notebook(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _imports_pymc_marketing(data: dict) -> bool:
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if IMPORT_RE.search(source):
            return True
    return False


def _check_imported_packages(notebook_data: dict, packages: set[str]) -> str | None:
    if _imports_pymc_marketing(notebook_data) and "pymc_marketing" not in packages:
        return "Missing pymc_marketing in -p (notebook imports pymc_marketing)"
    return None


def check_notebook(path: Path) -> str | None:
    try:
        data = load_notebook(path)
    except Exception as e:
        return f"Failed to load notebook: {e}"

    cells = data.get("cells", [])
    if not cells:
        return "No cells found"

    last_cell = cells[-1]

    if last_cell.get("cell_type") != "code":
        return "Last cell is not a code cell"

    source = "".join(last_cell.get("source", []))

    if "%load_ext watermark" not in source and "%reload_ext watermark" not in source:
        return "Missing %load_ext watermark in last cell"

    watermark_line = None
    for line in source.split("\n"):
        if line.startswith("%watermark"):
            watermark_line = line.strip()
            break

    if watermark_line is None:
        return "Missing %watermark line in last cell"

    missing_flags = REQUIRED_FLAGS - set(watermark_line.split())
    if missing_flags:
        flags_str = " ".join(sorted(missing_flags))
        return f"Missing required watermark flags: {flags_str}"

    match = WATERMARK_RE.search(watermark_line)
    if not match:
        return "Missing -p flag with packages in watermark (e.g. -p pymc_marketing,pytensor)"

    actual_packages = {p.strip() for p in match.group("packages").split(",")}

    if REQUIRED_PACKAGES:
        missing = REQUIRED_PACKAGES - actual_packages
        if missing:
            return f"Missing required packages in -p: {sorted(missing)}"

    imported_error = _check_imported_packages(data, actual_packages)
    if imported_error:
        return imported_error

    return None


def collect_notebooks() -> list[Path]:
    return sorted(p for p in DOC_SOURCE.rglob("*.ipynb") if not is_excluded(p))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (HERE.parent / path)


def _expand(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.ipynb"))
    return [path]


def _rel(path: Path) -> Path:
    try:
        return path.relative_to(HERE.parent)
    except ValueError:
        return path


def main() -> int:
    if len(sys.argv) > 1:
        notebooks = sorted(
            {resolved for p in sys.argv[1:] for resolved in _expand(_resolve(Path(p)))}
        )
    else:
        notebooks = collect_notebooks()

    failed: list[tuple[Path, str]] = []

    for path in notebooks:
        if is_excluded(path):
            continue
        error = check_notebook(path)
        if error is not None:
            failed.append((_rel(path), error))

    if failed:
        print(f"Watermark validation failed for {len(failed)} notebook(s):")
        print()
        for path, message in failed:
            print(f"  {path}")
            print(f"    {message}")
        print()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
