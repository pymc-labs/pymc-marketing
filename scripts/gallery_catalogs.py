"""Keep Spanish catalogs attached to notebook docnames.

The site is published in English and Spanish (#3122). A notebook move is
recorded in ``gallery/redirects.yaml``. If that notebook already has a
Spanish catalog, the catalog has to move in the same change. A notebook
that never had a catalog is not an error, and a stale catalog that is not
the source of a redirect is left alone.
"""

from pathlib import Path

LOCALE_RELATIVE = Path("../../locales")
CATALOG_LANGUAGE = "es"
SHELL_PAGES = (
    "getting_started/intro_guides",
    "guide/technical_guides",
    "gallery/gallery",
)


def catalog_dir(source_dir: Path) -> Path:
    """Return the Spanish message catalog directory for a docs source tree.

    This is ``locale_dirs`` from ``docs/source/conf.py`` plus ``es/LC_MESSAGES``.
    """
    return (source_dir / LOCALE_RELATIVE / CATALOG_LANGUAGE / "LC_MESSAGES").resolve()


def validate_catalogs(source_dir: Path) -> list[str]:
    """Return catalog errors for the docs split.

    A source tree with no locale directory is valid. The gallery unit fixtures
    have none, and a notebook that never had a catalog must not fail a move.
    """
    messages = catalog_dir(source_dir)
    if not messages.is_dir():
        return []
    errors = missing_shell_catalogs(messages)
    errors.extend(forgotten_catalog_moves(source_dir, messages))
    return errors


def missing_shell_catalogs(messages: Path) -> list[str]:
    """Return an error for each new part page that has no Spanish catalog."""
    errors: list[str] = []
    for docname in SHELL_PAGES:
        path = _catalog_file(messages, docname)
        if path is None or not _catalog_exists(path):
            errors.append(
                "Spanish catalog missing for new page: "
                f"locales/es/LC_MESSAGES/{docname}.po"
            )
    return errors


def forgotten_catalog_moves(
    source_dir: Path, messages: Path | None = None
) -> list[str]:
    """Return an error when a redirect left the old Spanish catalog behind.

    No error when the old catalog does not exist. That notebook never had a
    translation, or the catalog was already moved.
    """
    catalog_root = messages if messages is not None else catalog_dir(source_dir)
    if not catalog_root.is_dir():
        return []
    from gallery_redirects import load_redirects

    entries, _load_errors = load_redirects(source_dir)
    errors: list[str] = []
    for entry in entries:
        old = _catalog_file(catalog_root, entry.source)
        if old is not None and _catalog_exists(old):
            errors.append(
                "Spanish catalog was not moved with the notebook: "
                f"locales/es/LC_MESSAGES/{entry.source}.po still exists; "
                f"git mv it to locales/es/LC_MESSAGES/{entry.target}.po "
                "in the same PR. Do not regenerate the catalog."
            )
    return errors


def _catalog_exists(path: Path) -> bool:
    """Return whether ``path`` is a catalog with that exact spelling.

    macOS APFS reports ``sbg.po`` as existing when only ``sBG.po`` is present.
    Those are different docnames. Linux CI will not treat them as the same file.
    """
    if not path.parent.is_dir():
        return False
    return any(
        entry.name == path.name and entry.is_file() for entry in path.parent.iterdir()
    )


def _catalog_file(catalog_root: Path, docname: str) -> Path | None:
    if not docname or ".." in docname.split("/"):
        return None
    path = (catalog_root / f"{docname}.po").resolve()
    try:
        path.relative_to(catalog_root)
    except ValueError:
        return None
    return path
