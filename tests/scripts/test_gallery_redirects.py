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
"""Tests for the notebook redirect map."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urljoin

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from gallery_pages import check_gallery  # noqa: E402
from gallery_redirects import (  # noqa: E402
    load_redirects,
    relative_target,
    render_redirect_html,
    sphinx_redirects,
    validate_redirect_map,
)

SOURCE = ROOT / "docs" / "source"
OLD = "notebooks/mmm/case_study"
NEW = "gallery/notebooks/mmm/case_study"
TEMPLATE = (SOURCE / "_templates" / "redirect.html").read_text()
FORBIDDEN = (
    "/en/",
    "/es/",
    "/stable/",
    "/latest/",
    "https://www.pymc-marketing.io",
    "pymc-marketing.readthedocs.io",
)


def _map(source: str = OLD, target: str = NEW) -> str:
    return f"- from: {source}\n  to: {target}\n"


def _tree(tmp_path: Path, mapping: str, *, target_exists: bool = True) -> Path:
    source = tmp_path / "source"
    (source / "gallery").mkdir(parents=True)
    (source / "gallery" / "gallery.yaml").write_text("sections: []\n")
    (source / "gallery" / "redirects.yaml").write_text(mapping)
    if target_exists:
        path = source / f"{NEW}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Moved\n")
    return source


def test_repository_map_is_empty() -> None:
    """This issue lands the mechanism, not the first moved notebook."""
    entries, errors = load_redirects(SOURCE)
    assert errors == []
    assert entries == []
    code, messages = check_gallery(SOURCE)
    assert code == 0, messages


def test_empty_map_is_valid(tmp_path: Path) -> None:
    source = _tree(tmp_path, "[]\n", target_exists=False)
    assert validate_redirect_map(source) == []
    redirects, excludes, errors = sphinx_redirects(source)
    assert errors == []
    assert redirects == {}
    assert excludes == []


def test_relative_target_keeps_language_and_version() -> None:
    target = relative_target(OLD, NEW)
    assert target == "../../gallery/notebooks/mmm/case_study.html"
    for forbidden in FORBIDDEN:
        assert forbidden not in target
    for page in (
        "https://www.pymc-marketing.io/en/latest/notebooks/mmm/case_study.html",
        "https://www.pymc-marketing.io/en/stable/notebooks/mmm/case_study.html",
        "https://www.pymc-marketing.io/es/latest/notebooks/mmm/case_study.html",
        "https://www.pymc-marketing.io/es/stable/notebooks/mmm/case_study.html",
    ):
        resolved = urljoin(page, target)
        assert resolved.endswith("/gallery/notebooks/mmm/case_study.html")
        assert "/en/" in resolved if "/en/" in page else "/es/" in resolved
        assert "/latest/" in resolved if "/latest/" in page else "/stable/" in resolved


def test_template_emits_relative_refresh_and_canonical() -> None:
    target = relative_target(OLD, NEW)
    html = render_redirect_html(target, TEMPLATE)
    assert f"url={target}" in html
    assert f'href="{target}"' in html
    assert html.count('rel="canonical"') == 1
    for forbidden in FORBIDDEN:
        assert forbidden not in html


@pytest.mark.parametrize(
    ("mapping", "fragment"),
    [
        (
            _map(),
            "redirect source is still a source file",
        ),
        (
            _map(target="gallery/notebooks/mmm/missing"),
            "redirect target has no source file",
        ),
        (
            _map(
                target="https://www.pymc-marketing.io/en/stable/gallery/notebooks/mmm/case_study"
            ),
            "must be a relative docname, not a URL",
        ),
        (
            _map(target="/en/stable/gallery/notebooks/mmm/case_study"),
            "must be a relative docname, not a URL",
        ),
        (
            _map(target="gallery/notebooks/en/case_study"),
            "language or version prefix",
        ),
        (
            _map(target="en/stable/gallery/notebooks/mmm/case_study"),
            "language or version prefix",
        ),
        (
            _map(source="getting_started/notebooks/mmm/case_study"),
            "not an example-gallery docname",
        ),
        (
            "- from: notebooks/mmm/case_study\n  to: gallery/notebooks/mmm/case_study\n"
            "- from: notebooks/mmm/case_study\n  to: guide/notebooks/mmm/case_study\n",
            "listed more than once",
        ),
    ],
)
def test_check_rejects_bad_redirect(
    tmp_path: Path, mapping: str, fragment: str
) -> None:
    source = _tree(tmp_path, mapping)
    (source / f"{OLD}.ipynb").parent.mkdir(parents=True, exist_ok=True)
    (source / f"{OLD}.ipynb").write_text("{}\n")
    # The still-a-source-file case needs the old file. The other cases must
    # not, or that error would hide the one under test.
    if fragment != "redirect source is still a source file":
        (source / f"{OLD}.ipynb").unlink()
    errors = validate_redirect_map(source)
    assert any(fragment in error for error in errors), errors
    redirects, excludes, sphinx_errors = sphinx_redirects(source)
    assert redirects == {}
    assert excludes == []
    assert sphinx_errors


def test_valid_entry_is_relative_and_excluded(tmp_path: Path) -> None:
    source = _tree(tmp_path, _map())
    redirects, excludes, errors = sphinx_redirects(source)
    assert errors == []
    assert redirects == {OLD: "../../gallery/notebooks/mmm/case_study.html"}
    assert excludes == [f"{OLD}.html"]
    for part in ("getting_started", "guide", "gallery"):
        target = f"{part}/notebooks/mmm/case_study"
        mapped = tmp_path / part / "source"
        (mapped / "gallery").mkdir(parents=True)
        (mapped / "gallery" / "redirects.yaml").write_text(_map(target=target))
        page = mapped / f"{target}.md"
        page.parent.mkdir(parents=True)
        page.write_text("# Moved\n")
        found, part_excludes, problems = sphinx_redirects(mapped)
        assert problems == [], problems
        assert found[OLD].endswith(f"{target}.html")
        assert not found[OLD].startswith("/")
        assert "://" not in found[OLD]
        assert part_excludes == [f"{OLD}.html"]


def _write_sphinx_src(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    (src / "gallery" / "notebooks" / "mmm").mkdir(parents=True)
    (src / "_templates").mkdir()
    (src / "gallery" / "redirects.yaml").write_text(_map())
    (src / f"{NEW}.md").write_text("# Case study\n\nMoved.\n")
    (src / "_templates" / "redirect.html").write_text(TEMPLATE)
    (src / "index.md").write_text(
        "# Home\n\n```{toctree}\n:hidden:\n\ngallery/notebooks/mmm/case_study\n```\n"
    )
    (src / "conf.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        "from gallery_redirects import REDIRECT_TEMPLATE, sphinx_redirects\n"
        "extensions = ['myst_parser', 'sphinx_reredirects', 'sphinx_sitemap']\n"
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
        "site_url = 'https://www.pymc-marketing.io/'\n"
        "sitemap_url_scheme = '{lang}stable/{link}'\n"
        "_redirects, _excludes, _errors = sphinx_redirects(Path(__file__).parent)\n"
        "assert not _errors, _errors\n"
        "redirects = _redirects\n"
        "redirect_html_template_file = REDIRECT_TEMPLATE\n"
        "sitemap_excludes = list(_excludes)\n"
        "master_doc = 'index'\n"
    )
    return src


def _canonicals(html: str) -> list[str]:
    return re.findall(r'<link rel="canonical" href="([^"]*)"', html)


def test_docs_build_emits_relative_redirect_for_en_and_es(tmp_path: Path) -> None:
    """A real Sphinx build writes the redirect page and keeps it out of the sitemap."""
    pytest.importorskip("sphinx_reredirects")
    pytest.importorskip("sphinx_sitemap")
    pytest.importorskip("myst_parser")
    pytest.importorskip("labs_sphinx_theme")
    src = _write_sphinx_src(tmp_path)
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
                "-j",
                "2",
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
        redirect = (out / f"{OLD}.html").read_text()
        target = "../../gallery/notebooks/mmm/case_study.html"
        assert f"url={target}" in redirect
        assert _canonicals(redirect) == [target]
        for forbidden in FORBIDDEN:
            assert forbidden not in redirect
        served = (
            "https://www.pymc-marketing.io/"
            f"{language}/latest/notebooks/mmm/case_study.html"
        )
        assert urljoin(served, target) == (
            "https://www.pymc-marketing.io/"
            f"{language}/latest/gallery/notebooks/mmm/case_study.html"
        )
        page = (out / f"{NEW}.html").read_text()
        expected = (
            "https://www.pymc-marketing.io/"
            f"{language}/stable/gallery/notebooks/mmm/case_study.html"
        )
        canonicals = _canonicals(page)
        assert expected in canonicals
        assert all("//" not in href.removeprefix("https://") for href in canonicals)
        old_canonical = (
            "https://www.pymc-marketing.io/"
            f"{language}/stable/notebooks/mmm/case_study.html"
        )
        assert old_canonical not in canonicals
        sitemap = (out / "sitemap.xml").read_text()
        old_loc = (
            "https://www.pymc-marketing.io/"
            f"{language}/stable/notebooks/mmm/case_study.html"
        )
        new_loc = (
            "https://www.pymc-marketing.io/"
            f"{language}/stable/gallery/notebooks/mmm/case_study.html"
        )
        assert old_loc not in sitemap
        assert new_loc in sitemap


def test_repo_conf_registers_the_empty_map() -> None:
    pytest.importorskip("sphinx_reredirects")
    pytest.importorskip("labs_sphinx_theme")
    pytest.importorskip("pymc_marketing")
    spec = importlib.util.spec_from_file_location(
        "pymc_marketing_docs_conf", SOURCE / "conf.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.redirects == {}
    assert "sphinx_reredirects" in module.extensions
    assert module.redirect_html_template_file == "_templates/redirect.html"
    assert module.sitemap_excludes[:5] == [
        "search.html",
        "genindex.html",
        "py-modindex.html",
        "api/generated/classmethods/*",
        "api/generated/classattributes/*",
    ]
