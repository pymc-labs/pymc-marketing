#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Tests for Spanish catalogs on the notebook split."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from gallery_catalogs import (  # noqa: E402
    SHELL_PAGES,
    catalog_dir,
    forgotten_catalog_moves,
    missing_shell_catalogs,
    validate_catalogs,
)
from gallery_pages import PART_PAGES, check_gallery  # noqa: E402

SOURCE = ROOT / "docs" / "source"
MESSAGES = ROOT / "locales" / "es" / "LC_MESSAGES"
CASE_STUDY = "notebooks/mmm/mmm_case_study"
CASE_TITLE = "Caso de Estudio Completo de MMM"
PAGES = (
    "getting_started/intro_guides",
    "guide/technical_guides",
    "gallery/gallery",
)


def _indexes() -> dict[str, str]:
    return {
        "getting_started/index.md": "```{toctree}\n:hidden:\n\nintro_guides\n```\n",
        "guide/index.md": "```{toctree}\n:hidden:\n\ntechnical_guides\n```\n",
        "index.md": "```{toctree}\n:hidden:\n\ngallery/gallery\n```\n",
    }


def _tree(
    tmp_path: Path,
    *,
    notebooks: list[str],
    redirects: str = "[]\n",
    part: str | None = None,
) -> Path:
    source = tmp_path / "docs" / "source"
    (source / "gallery").mkdir(parents=True)
    cards = ""
    for docname in notebooks:
        cards += f"      - title: Example\n        notebook: {docname}\n"
        if part is not None:
            cards += f"        part: {part}\n"
    (source / "gallery" / "gallery.yaml").write_text(
        f"sections:\n  - title: Models\n    cards:\n{cards}"
    )
    (source / "gallery" / "redirects.yaml").write_text(redirects)
    for rel, text in _indexes().items():
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for docname in notebooks:
        path = source / f"{docname}.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    return source


def _write_po(path: Path, msgid: str = "Title", msgstr: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'msgid "{msgid}"\nmsgstr "{msgstr}"\n')


def _shell_catalogs(source: Path) -> Path:
    messages = catalog_dir(source)
    for docname in SHELL_PAGES:
        _write_po(messages / f"{docname}.po", msgid=docname)
    return messages


def test_shell_pages_match_the_part_pages() -> None:
    """The catalog check and the generator name the same three pages."""
    assert set(SHELL_PAGES) == set(PART_PAGES.values())
    assert "gallery/holding" not in SHELL_PAGES


def test_missing_locale_directory_is_not_an_error(tmp_path: Path) -> None:
    """Gallery fixtures have no locale directory and must stay valid."""
    source = _tree(tmp_path, notebooks=["notebooks/mmm/quickstart"])
    assert validate_catalogs(source) == []


def test_missing_shell_catalog_is_reported(tmp_path: Path) -> None:
    """A locale tree without the new part-page catalogs fails."""
    messages = tmp_path / "es"
    messages.mkdir()
    errors = missing_shell_catalogs(messages)
    assert len(errors) == 3
    assert all(error.startswith("Spanish catalog missing") for error in errors)
    assert "getting_started/intro_guides.po" in errors[0]


def test_forgotten_catalog_names_the_git_mv(tmp_path: Path) -> None:
    """An old catalog left behind a redirect is an error, with the new path."""
    source = _tree(tmp_path, notebooks=[])
    (source / "gallery" / "redirects.yaml").write_text(
        "- from: notebooks/mmm/mmm_case_study\n"
        "  to: gallery/notebooks/mmm/mmm_case_study\n"
    )
    messages = tmp_path / "messages"
    _write_po(messages / f"{CASE_STUDY}.po", msgstr="Caso")
    errors = forgotten_catalog_moves(source, messages)
    assert len(errors) == 1
    assert "mmm_case_study.po still exists" in errors[0]
    assert "git mv" in errors[0]
    assert "gallery/notebooks/mmm/mmm_case_study.po" in errors[0]


def test_notebook_that_never_had_a_catalog_is_not_an_error(tmp_path: Path) -> None:
    """No old .po means the move did not forget a translation."""
    source = _tree(tmp_path, notebooks=[])
    (source / "gallery" / "redirects.yaml").write_text(
        "- from: notebooks/mmm/mmm_case_study2\n"
        "  to: gallery/notebooks/mmm/mmm_case_study2\n"
    )
    messages = tmp_path / "messages"
    messages.mkdir()
    _write_po(messages / "notebooks/clv/sBG.po", msgstr="Modelo sBG")
    assert forgotten_catalog_moves(source, messages) == []


def test_differently_cased_catalog_is_not_the_old_file(tmp_path: Path) -> None:
    """``sBG.po`` is a stale catalog, not the catalog for ``sbg``."""
    source = _tree(tmp_path, notebooks=[])
    (source / "gallery" / "redirects.yaml").write_text(
        "- from: notebooks/clv/sbg\n  to: gallery/notebooks/clv/sbg\n"
    )
    messages = tmp_path / "messages"
    _write_po(messages / "notebooks/clv/sBG.po", msgstr="Modelo sBG")
    assert forgotten_catalog_moves(source, messages) == []


def test_catalog_path_cannot_escape_the_locale_tree(tmp_path: Path) -> None:
    """A redirect docname cannot point the check outside the catalog root."""
    source = _tree(tmp_path, notebooks=[])
    (source / "gallery" / "redirects.yaml").write_text(
        "- from: ../secrets\n  to: gallery/notebooks/mmm/case\n"
    )
    messages = tmp_path / "messages"
    messages.mkdir()
    assert forgotten_catalog_moves(source, messages) == []


def test_check_gallery_fails_while_the_old_catalog_remains(tmp_path: Path) -> None:
    """The gallery hook reports the forgotten catalog and does not write."""
    new = "gallery/notebooks/mmm/mmm_case_study"
    source = _tree(
        tmp_path,
        notebooks=[new],
        part="case",
        redirects=(
            "- from: notebooks/mmm/mmm_case_study\n"
            "  to: gallery/notebooks/mmm/mmm_case_study\n"
        ),
    )
    messages = _shell_catalogs(source)
    old = messages / f"{CASE_STUDY}.po"
    _write_po(old, msgstr=CASE_TITLE)
    _write_po(messages / "notebooks/clv/sBG.po", msgstr="Modelo sBG")
    code, found = check_gallery(source, write=True)
    assert code == 1
    assert any("mmm_case_study.po still exists" in message for message in found)
    assert not any("sBG.po" in message for message in found)
    assert not (source / "getting_started" / "intro_guides.md").is_file()

    old.unlink()
    code, found = check_gallery(source, write=True)
    assert code == 0, found


def test_check_gallery_requires_shell_catalogs_when_locales_exist(
    tmp_path: Path,
) -> None:
    """Creating the locale directory makes the three part-page catalogs required."""
    source = _tree(tmp_path, notebooks=["notebooks/mmm/quickstart"])
    catalog_dir(source).mkdir(parents=True)
    code, found = check_gallery(source)
    assert code == 1
    assert any("intro_guides.po" in message for message in found)
    assert any("technical_guides.po" in message for message in found)
    assert any("gallery/gallery.po" in message for message in found)


def _canonicals(html: str) -> list[str]:
    return re.findall(r'<link rel="canonical" href="([^"]*)"', html)


def _hreflangs(html: str) -> dict[str, str]:
    return dict(
        re.findall(
            r'<link rel="alternate" hreflang="([^"]*)" href="([^"]*)"',
            html,
        )
    )


def test_language_switcher_on_the_three_new_pages(tmp_path: Path) -> None:
    """Canonical and hreflang stay on the requested language, with no double slash."""
    pytest.importorskip("myst_parser")
    pytest.importorskip("labs_sphinx_theme")
    src = tmp_path / "source"
    for docname in PAGES:
        path = src / f"{docname}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {docname}\n")
    (src / "index.md").write_text(
        "# Home\n\n```{toctree}\n:hidden:\n\n" + "\n".join(PAGES) + "\n```\n"
    )
    (src / "conf.py").write_text(
        "extensions = ['myst_parser']\n"
        "html_theme = 'labs_sphinx_theme'\n"
        "html_context = {\n"
        "    'github_user': 'pymc-labs',\n"
        "    'github_repo': 'pymc-marketing',\n"
        "    'github_version': 'main',\n"
        "    'doc_path': 'docs/source/',\n"
        "    'baseurl': 'https://www.pymc-marketing.io',\n"
        "    'rtd_version': 'stable',\n"
        "    'translations': ['en', 'es'],\n"
        "}\n"
        "master_doc = 'index'\n"
    )
    for language in ("en", "es"):
        out = tmp_path / f"build-{language}"
        completed = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "html",
                "-W",
                "--keep-going",
                "-q",
                "-D",
                f"language={language}",
                str(src),
                str(out),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        for docname in PAGES:
            page = (out / f"{docname}.html").read_text()
            expected = f"https://www.pymc-marketing.io/{language}/stable/{docname}.html"
            assert expected in _canonicals(page)
            assert _hreflangs(page) == {
                "en": f"https://www.pymc-marketing.io/en/stable/{docname}.html",
                "es": f"https://www.pymc-marketing.io/es/stable/{docname}.html",
            }
            for href in [_canonicals(page)[0], *_hreflangs(page).values()]:
                assert "//" not in href.removeprefix("https://")


def test_spanish_build_shows_the_existing_case_study_title(tmp_path: Path) -> None:
    """The case-study catalog still translates its title. The notebook is unmoved."""
    pytest.importorskip("myst_parser")
    notebook = SOURCE / f"{CASE_STUDY}.ipynb"
    catalog = MESSAGES / f"{CASE_STUDY}.po"
    assert notebook.is_file()
    assert CASE_TITLE in catalog.read_text()

    src = tmp_path / "source"
    page = src / f"{CASE_STUDY}.md"
    page.parent.mkdir(parents=True)
    page.write_text("# MMM End-to-End Case Study\n")
    (src / "index.md").write_text(
        "```{toctree}\n:hidden:\n\nnotebooks/mmm/mmm_case_study\n```\n"
    )
    (src / "conf.py").write_text(
        "extensions = ['myst_parser']\n"
        "gettext_compact = False\n"
        "locale_dirs = ['../locales']\n"
        "language = 'es'\n"
        "master_doc = 'index'\n"
    )
    dest = tmp_path / "locales" / "es" / "LC_MESSAGES" / "notebooks" / "mmm"
    dest.mkdir(parents=True)
    shutil.copy(catalog, dest / "mmm_case_study.po")
    out = tmp_path / "build"
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "-W",
            "--keep-going",
            "-q",
            str(src),
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    html = (out / f"{CASE_STUDY}.html").read_text()
    assert CASE_TITLE in html
    assert "MMM End-to-End Case Study" not in html


def test_repo_keeps_two_languages_and_does_not_backfill() -> None:
    """English stays the source. Only the three shell catalogs were added."""
    assert not (ROOT / "locales" / "en").exists()
    rtd = (ROOT / ".readthedocs.yml").read_text()
    html_commands = [
        line.strip()
        for line in rtd.splitlines()
        if "sphinx" in line and "-b html" in line
    ]
    assert html_commands == [
        "- python -m sphinx -T -W --keep-going -j 3 -b html "
        "-d _build/doctrees -D language=$READTHEDOCS_LANGUAGE "
        "docs/source $READTHEDOCS_OUTPUT/html"
    ]
    conf = (SOURCE / "conf.py").read_text()
    assert '"translations": ["en", "es"]' in conf
    assert "locales/en" not in conf

    for docname in SHELL_PAGES:
        catalog = (MESSAGES / f"{docname}.po").read_text()
        assert f"../source/{docname}.md" in catalog
        assert 'msgid "No notebooks are tagged for this page yet."' in catalog
    gallery = (MESSAGES / "gallery" / "gallery.po").read_text()
    assert 'msgstr "Galería de Ejemplo"' in gallery
    assert 'msgid "No notebooks are tagged for this page yet."\nmsgstr ""' in gallery
    intro = (MESSAGES / "getting_started" / "intro_guides.po").read_text()
    assert 'msgid "Introductory guides"\nmsgstr ""' in intro
    assert (MESSAGES / "notebooks" / "clv" / "sBG.po").is_file()
    assert not (MESSAGES / "notebooks" / "mmm" / "mmm_case_study2.po").exists()

    git = shutil.which("git")
    assert git is not None
    tracked = subprocess.run(  # noqa: S603
        [git, "ls-files", "locales/es/LC_MESSAGES/notebooks"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    on_disk = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (MESSAGES / "notebooks").rglob("*.po")
    )
    assert on_disk == sorted(tracked)
