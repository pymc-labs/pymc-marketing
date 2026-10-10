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
"""Tests for the notebook part-page generator."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_gallery  # noqa: E402
from gallery_pages import (  # noqa: E402
    check_gallery,
    collect_cards,
    load_gallery,
    missing_extracted_thumbnails,
    render_pages,
)

SOURCE = ROOT / "docs" / "source"


def _indexes() -> dict[str, str]:
    return {
        "getting_started/index.md": "```{toctree}\n:hidden:\n\nintro_guides\n```\n",
        "guide/index.md": "```{toctree}\n:hidden:\n\ntechnical_guides\n```\n",
        "index.md": "```{toctree}\n:hidden:\n\ngallery/gallery\n```\n",
    }


def write_source(
    tmp_path: Path,
    yaml_text: str,
    notebooks: list[str],
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Build a tiny docs source tree with the three part pages included."""
    source = tmp_path / "source"
    (source / "gallery").mkdir(parents=True, exist_ok=True)
    (source / "gallery" / "gallery.yaml").write_text(yaml_text)
    defaults = {**_indexes(), "gallery/redirects.yaml": "[]\n"}
    for rel, text in {**defaults, **(extra_files or {})}.items():
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for docname in notebooks:
        path = source / f"{docname}.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    return source


def _yaml(cards: str) -> str:
    return f"sections:\n  - title: Marketing Mix Models\n    cards:\n{cards}"


def _card(title: str, notebook: str, part: str | None = None) -> str:
    lines = [f"      - title: {title}", f"        notebook: {notebook}"]
    if part is not None:
        lines.append(f"        part: {part}")
    return "\n".join(lines) + "\n"


def test_missing_part_renders_empty_pages_and_one_holding_home(tmp_path: Path):
    """An untagged notebook stays off every part page and has one home."""
    source = write_source(
        tmp_path,
        _yaml(_card("Quickstart", "notebooks/mmm/mmm_quickstart")),
        ["notebooks/mmm/mmm_quickstart"],
    )
    data, errors = load_gallery(source / "gallery" / "gallery.yaml")
    assert errors == []
    cards, structural = collect_cards(data)
    assert structural == []
    pages = render_pages(cards)

    for docname in (
        "getting_started/intro_guides",
        "guide/technical_guides",
        "gallery/gallery",
    ):
        page = pages[docname]
        assert "No notebooks are tagged for this page yet." in page
        assert ":::{grid-item-card}" not in page
        assert "\n## " not in page
        assert "```{toctree}" not in page
    assert "This page is not a list of every notebook." in pages["gallery/gallery"]
    assert "/notebooks/**" not in pages["gallery/gallery"]
    holding = pages["gallery/holding"]
    assert holding.startswith("---\norphan: true\n---\n")
    assert "/notebooks/mmm/mmm_quickstart" in holding
    assert ":glob:" not in holding
    code, messages = check_gallery(source, write=True)
    assert code == 0, messages
    code, messages = check_gallery(source)
    assert code == 0, messages


def test_section_cards_are_kept_beside_subsections(tmp_path: Path):
    """Section-level cards are not dropped when subsections exist."""
    yaml_text = (
        "sections:\n"
        "  - title: Marketing Mix Models\n"
        "    cards:\n"
        "      - title: Section card\n"
        "        notebook: notebooks/mmm/section_card\n"
        "    subsections:\n"
        "      - title: How-tos\n"
        "        cards:\n"
        "          - title: Sub card\n"
        "            notebook: notebooks/mmm/sub_card\n"
        "            part: guide\n"
    )
    source = write_source(
        tmp_path,
        yaml_text,
        ["notebooks/mmm/section_card", "notebooks/mmm/sub_card"],
    )
    data, errors = load_gallery(source / "gallery" / "gallery.yaml")
    assert errors == []
    cards, structural = collect_cards(data)
    assert structural == []
    assert {card.docname for card in cards} == {
        "notebooks/mmm/section_card",
        "notebooks/mmm/sub_card",
    }
    pages = render_pages(cards)
    guide = pages["guide/technical_guides"]
    assert "## Marketing Mix Models" in guide
    assert "### How-tos" in guide
    assert "/notebooks/mmm/sub_card" in guide
    assert "section_card" not in guide
    holding = pages["gallery/holding"]
    assert "/notebooks/mmm/section_card" in holding
    assert "sub_card" not in holding
    code, messages = check_gallery(source, write=True)
    assert code == 0, messages


def test_unknown_part_fails_before_writing(tmp_path: Path):
    """An unknown part is an error, and check mode does not write pages."""
    source = write_source(
        tmp_path,
        _yaml(_card("Quickstart", "notebooks/mmm/mmm_quickstart", "tutorial")),
        ["notebooks/mmm/mmm_quickstart"],
    )
    code, messages = check_gallery(source)
    assert code == 1
    assert any("unknown part 'tutorial'" in message for message in messages)
    assert not (source / "gallery" / "holding.md").exists()


def test_duplicate_notebook_fails(tmp_path: Path):
    """The same docname in two cards is an error."""
    source = write_source(
        tmp_path,
        _yaml(
            _card("Quickstart", "notebooks/mmm/mmm_quickstart")
            + _card("Again", "notebooks/mmm/mmm_quickstart")
        ),
        ["notebooks/mmm/mmm_quickstart"],
    )
    code, messages = check_gallery(source)
    assert code == 1
    assert any("listed more than once" in message for message in messages)


def test_missing_file_and_missing_yaml_entry_fail(tmp_path: Path):
    """A yaml path with no file, and a published notebook with no card, both fail."""
    source = write_source(
        tmp_path,
        _yaml(_card("Missing", "notebooks/mmm/missing")),
        ["notebooks/mmm/extra", "notebooks/mmm/dev/draft"],
    )
    code, messages = check_gallery(source)
    assert code == 1
    assert any("yaml docname has no file: notebooks/mmm/missing" in m for m in messages)
    assert any("notebooks/mmm/extra" in m for m in messages)
    assert not any("dev/draft" in m for m in messages)


def test_future_notebook_root_is_required(tmp_path: Path):
    """A notebook under a part-page notebooks directory must be listed."""
    source = write_source(
        tmp_path,
        _yaml(_card("Quickstart", "notebooks/mmm/mmm_quickstart")),
        ["notebooks/mmm/mmm_quickstart", "guide/notebooks/mmm/new_guide"],
    )
    code, messages = check_gallery(source)
    assert code == 1
    assert any("guide/notebooks/mmm/new_guide" in message for message in messages)


def test_angle_bracket_toctree_entry_is_a_second_home(tmp_path: Path):
    """The old Bass entry form is a second home. A body link is not."""
    notebook = "notebooks/mmm/mmm_quickstart"
    source = write_source(
        tmp_path,
        _yaml(_card("Quickstart", notebook)),
        [notebook],
        extra_files={
            "getting_started/index.md": (
                "```{toctree}\n:hidden:\n\n"
                "intro_guides\n"
                "Bass Diffusion Model Quickstart <../notebooks/mmm/mmm_quickstart>\n"
                "```\n"
            )
        },
    )
    code, messages = check_gallery(source)
    assert code == 1
    assert any("2 toctree homes" in message for message in messages)
    assert any("getting_started/index" in message for message in messages)
    assert any("gallery/holding" in message for message in messages)

    source = write_source(
        tmp_path,
        _yaml(_card("Quickstart", notebook)),
        [notebook],
        extra_files={
            "getting_started/index.md": (
                "```{toctree}\n:hidden:\n\nintro_guides\n```\n\n"
                "See the {doc}`Quickstart </notebooks/mmm/mmm_quickstart>`.\n"
            )
        },
    )
    code, messages = check_gallery(source, write=True)
    assert code == 0, messages


def test_glob_toctree_is_a_second_home(tmp_path: Path):
    """The old hidden ``/notebooks/**`` glob cannot share a notebook."""
    source = write_source(
        tmp_path,
        _yaml(_card("Quickstart", "notebooks/mmm/mmm_quickstart")),
        ["notebooks/mmm/mmm_quickstart"],
        extra_files={
            "index.md": (
                "```{toctree}\n:hidden:\n\ngallery/gallery\n```\n\n"
                "```{toctree}\n:hidden:\n:glob:\n\n/notebooks/**\n```\n"
            )
        },
    )
    code, messages = check_gallery(source)
    assert code == 1
    assert any("2 toctree homes" in message for message in messages)


def test_tagged_notebook_moves_home_and_keeps_a_doc_link(tmp_path: Path):
    """A case card links by docname and is absent from the holding page."""
    docname = "gallery/notebooks/mmm/mmm_case_study"
    source = write_source(
        tmp_path,
        _yaml(_card("Case study", docname, "case")),
        [docname],
    )
    data, _ = load_gallery(source / "gallery" / "gallery.yaml")
    pages = render_pages(collect_cards(data)[0])
    gallery = pages["gallery/gallery"]
    assert f":link: /{docname}" in gallery
    assert ":link-type: doc" in gallery
    assert ":img-top: images/mmm_case_study.png" in gallery
    assert "No notebooks are tagged for this page yet." not in gallery
    assert docname not in pages["gallery/holding"]
    assert "```{toctree}" not in pages["gallery/holding"]
    intro = render_pages(
        collect_cards(
            {
                "sections": [
                    {
                        "title": "MMM",
                        "cards": [
                            {
                                "title": "Quickstart",
                                "notebook": "notebooks/mmm/mmm_quickstart",
                                "part": "intro",
                            }
                        ],
                    }
                ]
            }
        )[0]
    )["getting_started/intro_guides"]
    assert ":img-top: ../gallery/images/mmm_quickstart.png" in intro
    assert ":link: /notebooks/mmm/mmm_quickstart" in intro
    code, messages = check_gallery(source, write=True)
    assert code == 0, messages


def test_part_page_must_be_in_a_handwritten_toctree(tmp_path: Path):
    """Dropping intro_guides from the index fails. Listing holding also fails."""
    notebook = "notebooks/mmm/mmm_quickstart"
    source = write_source(
        tmp_path,
        _yaml(_card("Quickstart", notebook)),
        [notebook],
        extra_files={"getting_started/index.md": "# Getting Started\n"},
    )
    code, messages = check_gallery(source)
    assert any("getting_started/intro_guides" in message for message in messages)
    assert code == 1

    source = write_source(
        tmp_path,
        _yaml(_card("Quickstart", notebook)),
        [notebook],
        extra_files={
            "index.md": "```{toctree}\n:hidden:\n\ngallery/gallery\ngallery/holding\n```\n"
        },
    )
    code, messages = check_gallery(source)
    assert code == 1
    assert any("holding page is in a hand-written toctree" in m for m in messages)


def test_check_does_not_require_or_write_thumbnails(tmp_path: Path):
    """Check mode reports sync drift and leaves the tree untouched."""
    source = write_source(
        tmp_path,
        _yaml(_card("Quickstart", "notebooks/mmm/mmm_quickstart")),
        ["notebooks/mmm/mmm_quickstart"],
    )
    code, messages = check_gallery(source)
    assert code == 1
    assert any("out of sync" in message for message in messages)
    assert not (source / "gallery" / "holding.md").exists()
    assert not (source / "gallery" / "images").exists()
    assert missing_extracted_thumbnails(
        [source / "notebooks" / "mmm" / "mmm_quickstart.ipynb"],
        source / "gallery" / "images",
    ) == ["mmm_quickstart"]


def test_cli_check_does_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The ``--check`` flag exits non-zero without creating pages."""
    source = write_source(
        tmp_path,
        _yaml(_card("Quickstart", "notebooks/mmm/mmm_quickstart")),
        ["notebooks/mmm/mmm_quickstart"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_gallery.py",
            "--check",
            "--no-thumbnails",
            "--source-dir",
            str(source),
        ],
    )
    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exc:
        generate_gallery.main()
    assert exc.value.code == 1
    assert "out of sync" in caplog.text
    assert not (source / "gallery" / "holding.md").exists()


def test_real_source_checks_clean():
    """The repository tree has one home per notebook and in-sync pages."""
    code, messages = check_gallery(SOURCE)
    assert code == 0, messages


def test_pilot_tags_only_the_case_study():
    """The example gallery shows the moved case study and no other notebook."""
    data, errors = load_gallery(SOURCE / "gallery" / "gallery.yaml")
    assert errors == []
    assert data is not None
    cards, structural = collect_cards(data)
    assert structural == []
    tagged = [(card.docname, card.part) for card in cards if card.part is not None]
    assert tagged == [("gallery/notebooks/mmm/mmm_case_study", "case")]
    assert sum(card.part is None for card in cards) == 67
    pages = render_pages(cards)
    gallery = pages["gallery/gallery"]
    assert ":link: /gallery/notebooks/mmm/mmm_case_study" in gallery
    assert ":img-top: images/mmm_case_study.png" in gallery
    assert ":link-type: doc" in gallery
    assert "mmm_case_study2" not in gallery
    assert "mmm_chronos" not in gallery
    assert "No notebooks are tagged for this page yet." not in gallery
    assert "```{toctree}" in gallery
    assert "/gallery/notebooks/mmm/mmm_case_study" in gallery
    assert (
        "No notebooks are tagged for this page yet."
        in pages["getting_started/intro_guides"]
    )
    assert (
        "No notebooks are tagged for this page yet." in pages["guide/technical_guides"]
    )
    assert "```{toctree}" not in pages["getting_started/intro_guides"]
    assert "```{toctree}" not in pages["guide/technical_guides"]
    holding = pages["gallery/holding"]
    assert "gallery/notebooks/mmm/mmm_case_study" not in holding
    assert "/notebooks/mmm/mmm_case_study\n" not in holding
    assert "/notebooks/mmm/mmm_case_study2" in holding
    assert (SOURCE / "gallery" / "gallery.md").read_text() == gallery
    assert (SOURCE / "gallery" / "holding.md").read_text() == holding


@pytest.mark.parametrize(
    ("part", "needle"),
    [
        ("intro", "(intro_guides)="),
        ("guide", "(technical_guides)="),
        ("case", "(gallery)="),
    ],
)
def test_part_pages_keep_stable_anchors(part: str, needle: str):
    """The three cross-link anchors survive an empty render."""
    pages = render_pages([])
    docname = {
        "intro": "getting_started/intro_guides",
        "guide": "guide/technical_guides",
        "case": "gallery/gallery",
    }[part]
    assert needle in pages[docname]
    assert "{ref}" in pages[docname]
