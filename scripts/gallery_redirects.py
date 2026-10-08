"""Validate the notebook redirect map and turn it into relative HTML targets.

``docs/source/gallery/redirects.yaml`` is the only redirect map (#3118).
It is empty until a migration PR moves a notebook. Targets are docnames.
The Sphinx build turns each pair into a relative HTML URL so the reader
stays on the language and version they requested.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml

REDIRECTS_FILE = "gallery/redirects.yaml"
REDIRECT_TEMPLATE = "_templates/redirect.html"
SOURCE_PREFIX = "notebooks/"
DESTINATION_PREFIXES = (
    "getting_started/notebooks/",
    "guide/notebooks/",
    "gallery/notebooks/",
)
_DOC_SUFFIXES = (".ipynb", ".md", ".rst")
_FORBIDDEN_SEGMENTS = frozenset({"en", "es", "stable", "latest"})
_HOSTS = ("pymc-marketing.io", "pymc-marketing.readthedocs.io", "readthedocs.io")
_DOCNAME_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./-"
)


@dataclass(frozen=True)
class Redirect:
    """One moved notebook: old docname to new docname."""

    source: str
    target: str


def source_file(source_dir: Path, docname: str) -> Path | None:
    """Return the source file for a docname, if one exists.

    Uses the gallery resolver so a redirect and a card agree on what a
    published notebook is.
    """
    from gallery_pages import notebook_file

    return notebook_file(source_dir, docname)


def relative_target(source: str, target: str) -> str:
    """Return the HTML URL of ``target`` relative to the old page.

    The result has no leading slash, no host, and no language or version
    prefix. Read the Docs serves the built file under ``/{lang}/{version}/``,
    so the browser keeps the language and version the reader requested.
    """
    relative = os.path.relpath(f"{target}.html", Path(source).parent.as_posix())
    return Path(relative).as_posix()


def render_redirect_html(to_uri: str, template: str) -> str:
    """Fill the redirect template. ``to_uri`` is substituted unchanged."""
    return Template(template).substitute(to_uri=to_uri)


def sitemap_excludes(entries: list[Redirect]) -> list[str]:
    """Return sitemap patterns for redirect pages.

    sphinx-sitemap records ``pagename.html``. A redirect page is not a
    Sphinx document, and this pattern is the second lock that keeps the
    old path out of the sitemap.
    """
    return [f"{entry.source}.html" for entry in entries]


def sphinx_redirects(source_dir: Path) -> tuple[dict[str, str], list[str], list[str]]:
    """Return ``(redirects, sitemap excludes, errors)`` for ``conf.py``.

    ``redirects`` maps an old docname to a relative HTML target.
    On any error the maps are empty so a bad file cannot emit a page.
    """
    entries, errors = load_redirects(source_dir)
    errors.extend(validate_entries(source_dir, entries))
    if errors:
        return {}, [], errors
    redirects = {
        entry.source: relative_target(entry.source, entry.target) for entry in entries
    }
    return redirects, sitemap_excludes(entries), []


def validate_redirect_map(source_dir: Path) -> list[str]:
    """Return every structural and coverage error in the redirect map."""
    entries, errors = load_redirects(source_dir)
    errors.extend(validate_entries(source_dir, entries))
    return errors


def load_redirects(source_dir: Path) -> tuple[list[Redirect], list[str]]:
    """Load redirect entries. Malformed rows are reported, not returned."""
    path = source_dir / REDIRECTS_FILE
    if not path.is_file():
        return [], [f"redirect map is missing: {REDIRECTS_FILE}"]
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        return [], [f"redirect map is not valid YAML: {exc}"]
    if not isinstance(data, list):
        return [], ["redirect map must be a list"]
    entries: list[Redirect] = []
    errors: list[str] = []
    for index, raw in enumerate(data, start=1):
        entry, error = _parse_entry(index, raw)
        if error:
            errors.append(error)
        elif entry is not None:
            entries.append(entry)
    return entries, errors


def validate_entries(source_dir: Path, entries: list[Redirect]) -> list[str]:
    """Return coverage errors for entries that already parsed."""
    errors: list[str] = []
    seen: dict[str, int] = {}
    for entry in entries:
        seen[entry.source] = seen.get(entry.source, 0) + 1
        if _outside_gallery(entry.source):
            errors.append(
                f"redirect source is not an example-gallery docname: {entry.source}"
            )
        if not entry.target.startswith(DESTINATION_PREFIXES):
            errors.append(f"redirect target is not a part-page docname: {entry.target}")
        if source_file(source_dir, entry.source) is not None:
            errors.append(f"redirect source is still a source file: {entry.source}")
        if source_file(source_dir, entry.target) is None:
            errors.append(f"redirect target has no source file: {entry.target}")
        if _forbidden_prefix(entry.source) or _forbidden_prefix(entry.target):
            errors.append(
                "redirect is absolute or contains a language or "
                f"version prefix: {entry.source} -> {entry.target}"
            )
    for source, count in sorted(seen.items()):
        if count > 1:
            errors.append(f"redirect source is listed more than once: {source}")
    return errors


def _outside_gallery(docname: str) -> bool:
    parts = docname.split("/")
    return not docname.startswith(SOURCE_PREFIX) or "dev" in parts or len(parts) < 3


def _parse_entry(index: int, raw: object) -> tuple[Redirect | None, str | None]:
    label = f"redirect entry {index}"
    if not isinstance(raw, dict):
        return None, f"{label} must be a mapping with from and to"
    unknown = sorted(set(raw) - {"from", "to"})
    if unknown:
        joined = ", ".join(unknown)
        return None, f"{label} has unknown keys: {joined}"
    source = raw.get("from")
    target = raw.get("to")
    if not isinstance(source, str) or not isinstance(target, str):
        return None, f"{label} must use docname strings for from and to"
    source_error = _docname_error(source)
    target_error = _docname_error(target)
    if source_error or target_error:
        return None, f"{label} {source_error or target_error}"
    return Redirect(source, target), None


def _docname_error(value: str) -> str | None:
    if not value or value != value.strip():
        return "must use docname strings for from and to"
    if value.startswith(("/", ".")) or "\\" in value or "://" in value or "//" in value:
        return "must be a relative docname, not a URL"
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return "must be a relative docname, not a URL"
    if value.endswith(_DOC_SUFFIXES):
        return "must be a docname with no suffix"
    if any(host in value.lower() for host in _HOSTS):
        return "must be a relative docname, not a URL"
    if any(char not in _DOCNAME_CHARS for char in value):
        return "must be a relative docname, not a URL"
    return None


def _forbidden_prefix(docname: str) -> bool:
    """Return whether a docname is absolute or carries language or version."""
    lowered = docname.lower()
    if docname.startswith(("/", ".")) or "://" in docname or "//" in docname:
        return True
    if any(host in lowered for host in _HOSTS):
        return True
    if any(part in _FORBIDDEN_SEGMENTS for part in docname.split("/")):
        return True
    padded = f"/{docname.strip('/')}/"
    markers = ("/en/", "/es/", "/stable/", "/latest/")
    return any(marker in padded for marker in markers)
