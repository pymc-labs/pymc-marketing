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
"""Tests for nav shells that stay valid while part-page grids are empty."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import gallery_pages  # noqa: E402
from gallery_pages import (  # noqa: E402
    part_cross_link_errors,
    validate_nav_shells,
)

SOURCE = ROOT / "docs" / "source"
_NAV_FILES = (
    "index.md",
    "getting_started/index.md",
    "guide/index.md",
)


def _copy_nav(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    for rel in _NAV_FILES:
        dest = source / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text((SOURCE / rel).read_text())
    return source


def _replace(source: Path, rel: str, old: str, new: str) -> None:
    path = source / rel
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def test_published_nav_shells_match_the_empty_grid_contract():
    """The repository nav keeps all three parts reachable and honest."""
    assert part_cross_link_errors() == []
    assert validate_nav_shells(SOURCE) == []


def test_fixture_without_quick_links_is_not_a_nav_shell(tmp_path: Path):
    """Synthetic gallery trees do not have to carry the homepage cards."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "index.md").write_text("# Docs\n")
    assert validate_nav_shells(source) == []


def test_example_gallery_card_cannot_claim_the_full_library(tmp_path: Path):
    """The old 'worked examples' card is the failure this shell replaces."""
    source = _copy_nav(tmp_path)
    _replace(
        source,
        "index.md",
        "Business cases only. This page is not a list of every notebook.",
        "Worked examples with real and synthetic data, showing how the library works.",
    )
    errors = validate_nav_shells(source)
    assert any("business cases" in error for error in errors)


def test_technical_guides_card_must_say_the_grid_fills_in(tmp_path: Path):
    """An empty Technical Guides card has to say the notebooks were not deleted."""
    source = _copy_nav(tmp_path)
    _replace(
        source,
        "index.md",
        "The library fills in as notebooks are tagged. An empty page is expected until then.",
        "Feature how-tos for the library.",
    )
    errors = validate_nav_shells(source)
    assert any("fills in" in error for error in errors)


def test_api_reference_card_stays_unchanged(tmp_path: Path):
    """The fourth card is additive. The API card is not rewritten."""
    source = _copy_nav(tmp_path)
    _replace(
        source,
        "index.md",
        "Assumes familiarity with the key concepts.",
        "Assumes no familiarity with the key concepts.",
    )
    errors = validate_nav_shells(source)
    assert any("API Reference card changed" in error for error in errors)


def test_intro_guides_entry_must_be_titled(tmp_path: Path):
    """A bare docname would show the page H1, not the locked nav label."""
    source = _copy_nav(tmp_path)
    _replace(
        source,
        "getting_started/index.md",
        "Intro Guides <intro_guides>",
        "intro_guides",
    )
    errors = validate_nav_shells(source)
    assert any("Intro Guides <intro_guides>" in error for error in errors)


def test_bass_stays_a_plain_link(tmp_path: Path):
    """Putting the untagged Bass notebook back in a toctree is a second home."""
    source = _copy_nav(tmp_path)
    _replace(
        source,
        "getting_started/index.md",
        "Intro Guides <intro_guides>",
        "Intro Guides <intro_guides>\nBass <../notebooks/bass/bass_example>",
    )
    errors = validate_nav_shells(source)
    assert any(
        "toctree-includes notebooks/bass/bass_example" in error for error in errors
    )

    source = _copy_nav(tmp_path)
    _replace(
        source,
        "getting_started/index.md",
        "See the {doc}`Bass diffusion model quickstart </notebooks/bass/bass_example>`.",
        "The Bass quickstart is not linked.",
    )
    errors = validate_nav_shells(source)
    assert any("plain Bass link is missing" in error for error in errors)


def test_guide_keeps_prose_toctrees(tmp_path: Path):
    """Technical Guides is added. Benefits, MMM, CLV, and Customer Choice stay."""
    source = _copy_nav(tmp_path)
    _replace(source, "guide/index.md", ":caption: Customer Choice\n", "")
    errors = validate_nav_shells(source)
    assert any("dropped caption Customer Choice" in error for error in errors)


def test_nav_rejects_a_heading_fragment(tmp_path: Path):
    """Heading fragments do not resolve when myst_heading_anchors is 0."""
    source = _copy_nav(tmp_path)
    _replace(
        source,
        "guide/index.md",
        "The library fills in as notebooks are tagged.",
        "See {ref}`Technical guides <technical-guides#technical-guides>`.",
    )
    errors = validate_nav_shells(source)
    assert any("heading fragment" in error for error in errors)


def test_part_cross_links_reject_a_custom_heading(monkeypatch: pytest.MonkeyPatch):
    """Cross-links use the three page anchors, not a heading slug."""
    copy = {part: dict(values) for part, values in gallery_pages._PAGE_COPY.items()}
    copy["case"]["links"] = "See {ref}`example gallery <gallery>`."
    monkeypatch.setattr(gallery_pages, "_PAGE_COPY", copy)
    errors = part_cross_link_errors()
    assert any(
        "case cross-links must be bare part anchors" in error for error in errors
    )
