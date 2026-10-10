"""Render the three notebook part pages and the untagged holding page.

``gallery.yaml`` is the source of truth (#1617, #3119). ``part`` is optional.
An empty part page is expected while cards are still untagged. Each notebook
has one toctree home. Sphinx 9.1 only logs a second home at INFO, so the
check lives here.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PARTS = ("intro", "guide", "case")
PART_PAGES = {
    "intro": "getting_started/intro_guides",
    "guide": "guide/technical_guides",
    "case": "gallery/gallery",
}
HOLDING_PAGE = "gallery/holding"
GENERATED_PAGES = (*PART_PAGES.values(), HOLDING_PAGE)
NOTEBOOK_ROOTS = (
    "notebooks",
    "getting_started/notebooks",
    "guide/notebooks",
    "gallery/notebooks",
)
_DOC_SUFFIXES = (".ipynb", ".md", ".rst")
_FENCES = (
    re.compile(
        r"^[ \t]*(?P<fence>`{3,})\{toctree\}[^\n]*\n"
        r"(?P<body>.*?)^[ \t]*(?P=fence)[ \t]*$",
        re.MULTILINE | re.DOTALL,
    ),
    re.compile(
        r"^[ \t]*(?P<fence>:{3,})\{toctree\}[^\n]*\n"
        r"(?P<body>.*?)^[ \t]*(?P=fence)[ \t]*$",
        re.MULTILINE | re.DOTALL,
    ),
)
_PAGE_COPY = {
    "intro": {
        "anchor": "intro_guides",
        "title": "Introductory guides",
        "rules": (
            "A notebook appears here only after its card sets `part: intro`.",
            "Untagged notebooks stay at their current URL.",
        ),
        "links": "The other parts are {ref}`technical_guides` and {ref}`gallery`.",
    },
    "guide": {
        "anchor": "technical_guides",
        "title": "Technical guides",
        "rules": (
            "A notebook appears here only after its card sets `part: guide`.",
            "Untagged notebooks stay at their current URL.",
        ),
        "links": "The other parts are {ref}`intro_guides` and {ref}`gallery`.",
    },
    "case": {
        "anchor": "gallery",
        "title": "Example Gallery",
        "rules": (
            "A notebook appears here only after its card sets `part: case`.",
            "Untagged notebooks stay at their current URL.",
            "This page is not a list of every notebook.",
        ),
        "links": (
            "The other parts are {ref}`intro_guides` and {ref}`technical_guides`."
        ),
    },
}
_EMPTY = "No notebooks are tagged for this page yet."
_HOLDING_RULES = (
    "The build includes each notebook exactly once.",
    "A notebook stays here, at its current URL, until its card sets `part`.",
)


@dataclass(frozen=True)
class Card:
    """One gallery.yaml card, normalized to a source-root docname."""

    title: str
    docname: str
    thumb: str
    part: str | None
    section: str
    subsection: str | None


def notebook_docname(value: object) -> str | None:
    """Return a source-root docname, or None when the value is not one."""
    if not isinstance(value, str):
        return None
    text = value.strip().replace("\\", "/")
    if not text or ".." in text.split("/"):
        return None
    for suffix in _DOC_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.strip("/") or None


def notebook_file(source_dir: Path, docname: str) -> Path | None:
    """Resolve a docname to an existing source file."""
    if not docname or ".." in docname.split("/"):
        return None
    base = source_dir / docname
    for suffix in _DOC_SUFFIXES:
        path = base.with_suffix(suffix)
        if path.is_file():
            return path
    return None


def load_gallery(yaml_path: Path) -> tuple[dict | None, list[str]]:
    """Load gallery.yaml. The first item is None when the file is unusable."""
    if not yaml_path.is_file():
        return None, [f"gallery.yaml is missing: {yaml_path}"]
    data = yaml.safe_load(yaml_path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("sections"), list):
        return None, ["gallery.yaml must be a mapping with a sections list"]
    return data, []


def collect_cards(data: dict) -> tuple[list[Card], list[str]]:
    """Walk sections into cards.

    A section may list its own cards and subsections. Both are kept.
    Dropping the section cards would hide those notebooks from their page.
    """
    cards: list[Card] = []
    errors: list[str] = []
    for section in data["sections"]:
        if not isinstance(section, dict):
            errors.append("gallery.yaml section is not a mapping")
            continue
        section_title = str(section.get("title", ""))
        subsections = section.get("subsections")
        groups = _card_groups(section_title, subsections, section, errors)
        for subsection, raw_cards in groups:
            if not isinstance(raw_cards, list):
                errors.append(f"cards in {section_title!r} must be a list")
                continue
            for raw in raw_cards:
                card, error = _card_from_raw(raw, section_title, subsection)
                if error:
                    errors.append(error)
                elif card is not None:
                    cards.append(card)
    return cards, errors


def _card_groups(
    section_title: str,
    subsections: object,
    section: dict,
    errors: list[str],
) -> list[tuple[str | None, object]]:
    """Return ``(subsection title, cards)`` groups.

    Section-level cards stay even when the section also has subsections.
    """
    if not subsections:
        return [(None, section.get("cards", []))]
    groups: list[tuple[str | None, object]] = []
    if "cards" in section:
        groups.append((None, section.get("cards")))
    if not isinstance(subsections, list):
        errors.append(f"subsections of {section_title!r} must be a list")
        return groups
    for sub in subsections:
        if not isinstance(sub, dict):
            errors.append(f"subsection of {section_title!r} is not a mapping")
            continue
        groups.append((str(sub.get("title", "")), sub.get("cards", [])))
    return groups


def _card_from_raw(
    raw: object, section_title: str, subsection: str | None
) -> tuple[Card | None, str | None]:
    if not isinstance(raw, dict):
        return None, f"card in {section_title!r} is not a mapping"
    title = str(raw.get("title", ""))
    if "notebook" not in raw or raw.get("notebook") in (None, ""):
        return None, f"card {title!r} in {section_title!r} is missing a notebook"
    docname = notebook_docname(raw.get("notebook"))
    if docname is None:
        return None, (
            f"card {title!r} has an invalid notebook path: {raw.get('notebook')!r}"
        )
    part = raw.get("part")
    if part is not None:
        part = str(part)
    thumb = raw.get("thumb")
    thumb_name = (
        str(thumb)
        if isinstance(thumb, str) and thumb.strip()
        else f"{Path(docname).name}.png"
    )
    return (
        Card(
            title=title or Path(docname).name,
            docname=docname,
            thumb=thumb_name,
            part=part,
            section=section_title,
            subsection=subsection,
        ),
        None,
    )


def part_counts(cards: list[Card]) -> dict[str, int]:
    """Count cards by part. Unknown parts are omitted."""
    counts = {part: 0 for part in PARTS}
    counts["untagged"] = 0
    for card in cards:
        if card.part in counts:
            counts[card.part] += 1
        elif card.part is None:
            counts["untagged"] += 1
    return counts


def published_notebooks(source_dir: Path) -> set[str]:
    """Docnames of published notebooks. ``dev/`` drafts are not published."""
    found: set[str] = set()
    for root in NOTEBOOK_ROOTS:
        base = source_dir / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.ipynb"):
            if _ignored_source(path, source_dir):
                continue
            found.add(path.relative_to(source_dir).with_suffix("").as_posix())
    return found


def _ignored_source(path: Path, source_dir: Path) -> bool:
    """Mirror ``exclude_patterns`` in ``docs/source/conf.py``."""
    rel = path.relative_to(source_dir)
    if "dev" in rel.parts or ".ipynb_checkpoints" in rel.parts:
        return True
    if rel.as_posix() == "gallery/README.md":
        return True
    return rel.parts[0] in {"build", "jupyter_execute", "jupyter_cache"}


def relative_image(page_docname: str, thumb: str) -> str:
    """Image path relative to the page that emits the card."""
    thumb_rel = thumb.replace("\\", "/").lstrip("/")
    target = f"gallery/images/{thumb_rel}"
    page_dir = Path(page_docname).parent.as_posix()
    if page_dir == ".":
        return target
    return Path(os.path.relpath(target, page_dir)).as_posix()


def _render_grid(page_docname: str, cards: list[Card]) -> list[str]:
    block = ["::::{grid} 1 2 3 3", ":gutter: 3", ""]
    for index, card in enumerate(cards):
        block.append(f":::{{grid-item-card}} {card.title}")
        block.append(f":img-top: {relative_image(page_docname, card.thumb)}")
        # Leading slash: the doc role joins a relative target onto the
        # emitting page. Regeneration rewrites this from the yaml docname.
        block.append(f":link: /{card.docname}")
        block.append(":link-type: doc")
        block.append(":::")
        if index < len(cards) - 1:
            block.append("")
    block.extend(["::::", ""])
    return block


def _render_sections(page_docname: str, cards: list[Card]) -> list[str]:
    lines: list[str] = []
    groups: list[tuple[str, str | None, list[Card]]] = []
    for card in cards:
        same_group = (
            groups
            and groups[-1][0] == card.section
            and groups[-1][1] == card.subsection
        )
        if same_group:
            groups[-1][2].append(card)
        else:
            groups.append((card.section, card.subsection, [card]))
    current_section: str | None = None
    for section, subsection, group in groups:
        if section != current_section:
            if section:
                lines.extend([f"## {section}", ""])
            current_section = section
        if subsection:
            lines.extend([f"### {subsection}", ""])
        lines.extend(_render_grid(page_docname, group))
    return lines


def _render_toctree(docnames: list[str]) -> list[str]:
    if not docnames:
        return []
    return [
        "```{toctree}",
        ":hidden:",
        "",
        *[f"/{docname}" for docname in docnames],
        "```",
        "",
    ]


def _unique_docnames(cards: list[Card]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for card in cards:
        if card.docname in seen:
            continue
        seen.add(card.docname)
        names.append(card.docname)
    return names


def _finalize(lines: list[str]) -> str:
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def render_pages(cards: list[Card]) -> dict[str, str]:
    """Render the three part pages and the holding page from cards."""
    pages: dict[str, str] = {}
    for part, docname in PART_PAGES.items():
        copy = _PAGE_COPY[part]
        selected = [card for card in cards if card.part == part]
        lines = [f"({copy['anchor']})=", "", f"# {copy['title']}", ""]
        lines.extend(copy["rules"])
        lines.extend(["", copy["links"], ""])
        if selected:
            lines.extend(_render_sections(docname, selected))
            lines.extend(_render_toctree(_unique_docnames(selected)))
        else:
            lines.extend([_EMPTY, ""])
        pages[docname] = _finalize(lines)
    untagged = [card for card in cards if card.part is None]
    holding = ["---", "orphan: true", "---", "", "# Untagged notebooks", ""]
    holding.extend(_HOLDING_RULES)
    holding.append("")
    holding.extend(_render_toctree(_unique_docnames(untagged)))
    pages[HOLDING_PAGE] = _finalize(holding)
    return pages


def _document_text(path: Path) -> str:
    if path.suffix != ".ipynb":
        return path.read_text()
    try:
        notebook = json.loads(path.read_text())
    except json.JSONDecodeError:
        return ""
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        chunks.append(source)
    return "\n".join(chunks)


def handwritten_documents(source_dir: Path) -> dict[str, str]:
    """Source documents except the pages this script writes."""
    documents: dict[str, str] = {}
    for pattern in ("*.md", "*.ipynb", "*.rst"):
        for path in source_dir.rglob(pattern):
            if _ignored_source(path, source_dir):
                continue
            docname = path.relative_to(source_dir).with_suffix("").as_posix()
            if docname in GENERATED_PAGES:
                continue
            documents[docname] = _document_text(path)
    return documents


def toctree_targets(text: str, page_docname: str) -> list[tuple[str, bool]]:
    """Return ``(target, glob)`` for each toctree entry.

    A ``{doc}`` role outside a toctree fence is not an entry. ``Title <doc>``
    contributes the docname only.
    """
    found: list[tuple[str, bool]] = []
    for pattern in _FENCES:
        for match in pattern.finditer(text):
            globbing = False
            for line in match.group("body").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(":"):
                    if stripped.startswith(":glob:"):
                        globbing = True
                    continue
                resolved = _resolve_entry(page_docname, _entry_target(stripped))
                if resolved:
                    found.append((resolved, globbing))
    return found


def _entry_target(line: str) -> str:
    match = re.search(r"<([^>]+)>\s*$", line)
    return match.group(1).strip() if match else line


def _resolve_entry(page_docname: str, entry: str) -> str:
    text = entry.strip().replace("\\", "/")
    for suffix in _DOC_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    if text.startswith("/"):
        return _norm_docname(text)
    parent = Path(page_docname).parent.as_posix()
    if parent == ".":
        return _norm_docname(text)
    return _norm_docname(f"{parent}/{text}")


def _norm_docname(path: str) -> str:
    parts: list[str] = []
    for part in path.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _glob_regex(pattern: str) -> re.Pattern[str]:
    escaped = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    return re.compile(f"^{escaped}$")


def _matching_docnames(
    targets: list[tuple[str, bool]], candidates: set[str]
) -> set[str]:
    matched: set[str] = set()
    for target, globbing in targets:
        if globbing:
            regex = _glob_regex(target)
            matched.update(name for name in candidates if regex.match(name))
        elif target in candidates:
            matched.add(target)
    return matched


def _is_orphan(text: str) -> bool:
    if not text.startswith("---"):
        return False
    end = text.find("\n---", 3)
    if end < 0:
        return False
    return bool(re.search(r"^orphan:\s*true\s*$", text[:end], re.MULTILINE))


def validate_gallery(
    source_dir: Path, cards: list[Card], pages: dict[str, str]
) -> list[str]:
    """Return every structural, coverage, and toctree-home error."""
    errors: list[str] = []
    seen: dict[str, int] = {}
    for card in cards:
        seen[card.docname] = seen.get(card.docname, 0) + 1
        if card.part is not None and card.part not in PARTS:
            errors.append(f"unknown part {card.part!r} on {card.docname}")
        if notebook_file(source_dir, card.docname) is None:
            errors.append(f"yaml docname has no file: {card.docname}")
    for docname, count in sorted(seen.items()):
        if count > 1:
            errors.append(f"notebook listed more than once: {docname}")

    listed = set(seen)
    missing = published_notebooks(source_dir) - listed
    for docname in sorted(missing):
        errors.append(f"published notebook missing from gallery.yaml: {docname}")

    placed = {
        card.docname
        for card in cards
        if (card.part is None or card.part in PARTS)
        and notebook_file(source_dir, card.docname) is not None
    }
    homes: dict[str, set[str]] = {docname: set() for docname in placed}
    for docname, text in pages.items():
        for name in _matching_docnames(toctree_targets(text, docname), placed):
            homes[name].add(docname)
    handwritten = handwritten_documents(source_dir)
    for docname, text in handwritten.items():
        for name in _matching_docnames(toctree_targets(text, docname), placed):
            homes[name].add(docname)
    for docname in sorted(placed):
        parents = homes[docname]
        if len(parents) > 1:
            joined = ", ".join(sorted(parents))
            errors.append(
                f"notebook has {len(parents)} toctree homes: {docname} ({joined})"
            )
        elif not parents:
            errors.append(f"notebook has no toctree home: {docname}")

    page_names = set(GENERATED_PAGES)
    included: dict[str, set[str]] = {name: set() for name in page_names}
    for docname, text in handwritten.items():
        matched = _matching_docnames(toctree_targets(text, docname), page_names)
        for name in matched:
            included[name].add(docname)
    for docname in PART_PAGES.values():
        if not included[docname]:
            errors.append(f"part page is not in a hand-written toctree: {docname}")
    if included[HOLDING_PAGE]:
        joined = ", ".join(sorted(included[HOLDING_PAGE]))
        errors.append(f"holding page is in a hand-written toctree: {joined}")
    if not _is_orphan(pages.get(HOLDING_PAGE, "")):
        errors.append("holding page is not marked orphan")
    return errors


def check_gallery(source_dir: Path, *, write: bool = False) -> tuple[int, list[str]]:
    """Validate and, unless ``write`` is set, compare generated pages to disk.

    Check mode never writes. Validation failures also skip the write.
    """
    data, messages = load_gallery(source_dir / "gallery" / "gallery.yaml")
    if data is None:
        return 1, messages
    cards, structural = collect_cards(data)
    messages.extend(structural)
    pages = render_pages(cards)
    messages.extend(validate_gallery(source_dir, cards, pages))
    if messages:
        return 1, messages
    stale = [
        docname
        for docname, text in pages.items()
        if not (source_dir / f"{docname}.md").is_file()
        or (source_dir / f"{docname}.md").read_text() != text
    ]
    if stale and not write:
        joined = ", ".join(stale)
        messages.append(
            "generated pages are out of sync: "
            f"{joined}. Run `python scripts/generate_gallery.py` to regenerate."
        )
        return 1, messages
    if write:
        for docname, text in pages.items():
            path = source_dir / f"{docname}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.is_file() or path.read_text() != text:
                path.write_text(text)
                logger.info("Wrote %s", path.relative_to(source_dir))
    return 0, messages


def missing_extracted_thumbnails(
    notebook_paths: list[Path], image_dir: Path
) -> list[str]:
    """Return stems whose extracted thumbnail is missing.

    Custom ``thumb:`` overrides are not produced by extraction, so they are
    not required. ``--check`` does not call this.
    """
    return [
        path.stem
        for path in notebook_paths
        if not (image_dir / f"{path.stem}.png").is_file()
    ]
