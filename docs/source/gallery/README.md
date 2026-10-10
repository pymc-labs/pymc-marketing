# PyMC-Marketing Example Gallery

This directory holds the example-gallery page and the manifest for every published notebook.

## How the pages are built

`gallery.yaml` is the single source of truth for sections, optional subsections, card titles, and notebook docnames.
`scripts/generate_gallery.py` renders four pages from it:

- `getting_started/intro_guides.md` for cards with `part: intro`
- `guide/technical_guides.md` for cards with `part: guide`
- `gallery/gallery.md` for cards with `part: case`
- `gallery/holding.md` for cards that omit `part`

`part` is optional. Leave it off while a notebook still mixes an introduction, a feature how-to, and a business case.
An empty part page is expected during that transition. The notebook stays on the holding page, at its current URL, until the card sets `part`.
Each notebook has one toctree home. The pre-commit hook `gallery-in-sync` fails the commit if a notebook is missing, duplicated, tagged with an unknown part, listed in more than one toctree, if `gallery/redirects.yaml` is invalid, or if a moved notebook left its Spanish catalog at the old path.

`notebook` is a docname relative to `docs/source/`, with no suffix. A notebook that has not moved yet is `notebooks/<category>/<stem>`.
After it is tagged, the docname becomes `getting_started/notebooks/<category>/<stem>`, `guide/notebooks/<category>/<stem>`, or `gallery/notebooks/<category>/<stem>`.
Do not put the same notebook in two cards.

## Nav shells

Getting Started links `Intro Guides <intro_guides>`. Guide links `Technical Guides <technical_guides>`. The homepage quick links are Getting Started, Technical Guides, Example Gallery, and API Reference.

Those entries stay while the card grids are empty. The Example Gallery card says business cases, not every notebook. The Technical Guides card, and the two section indexes, say the library fills in as notebooks are tagged. An empty page is expected. Do not put the Bass notebook back in a toctree. Cross-links between the three parts use `{ref}`intro_guides``, `{ref}`technical_guides``, and `{ref}`gallery`` only. Heading fragments do not resolve.

## Adding a new example

1. Add the notebook under `docs/source/notebooks/<category>/`. `dev/` drafts are ignored.
2. Add a card under the relevant section in `gallery.yaml`. Omit `part` unless the notebook is already one part.

   ```yaml
   - title: My New Example
     notebook: notebooks/mmm/my_new_example
   ```

   The thumbnail defaults to `images/<stem>.png`. Set an optional `thumb:` field to override it.
3. Run `python scripts/generate_gallery.py` to regenerate the four pages and extract the thumbnail from the first image cell.
   If the notebook has no image cell, the default logo is used.
4. Commit `gallery.yaml` and the regenerated pages. Do not commit `images/`. Those files are gitignored, and `--check` does not require them.

## Checking sync without writing

```
python scripts/generate_gallery.py --check --no-thumbnails
```

This does not write pages or thumbnails, and it does not require thumbnail files to exist.
It fails when a generated page is out of sync, when a published notebook is missing from the yaml, when the yaml lists a missing file, when a notebook has two toctree homes, when `gallery/redirects.yaml` is invalid, when a redirect's old Spanish catalog is still at the old path, or when those nav shells no longer match this contract.
A missing `part` is not a failure. An empty redirect map is valid. A notebook that never had a Spanish catalog is not a failure. An empty card grid is not a failure.

## Moving a notebook

One notebook per PR. Copy this checklist. Do not edit the Read the Docs dashboard. Do not edit `docs/source/index.md`, `llms.txt`, or other notebooks' absolute URLs to the notebook you moved. The redirect keeps those working on the version that contains the PR. Stable does not have the new path until release, and stable still serves the old file until then.

Classify before moving. If the notebook is one part, move the whole file and do not rewrite its prose. If it mixes an introduction, a feature how-to, and a business case, split that one source in the PR so each published piece is exactly one part. Do not pick the least-wrong tag. A split of one source is still one PR. The pilot, `mmm_case_study`, is one business case (PyData Global 2022 / Hajime Takeda dataset), not a feature how-to. It moved whole, with `part: case`. It was not split, and its text was not edited.

1. Create the destination directory, then `git mv` the notebook. `git mv` does not create parent directories.

   - `intro` → `docs/source/getting_started/notebooks/<category>/<stem>.ipynb`
   - `guide` → `docs/source/guide/notebooks/<category>/<stem>.ipynb`
   - `case` → `docs/source/gallery/notebooks/<category>/<stem>.ipynb`

   The docname is that path relative to `docs/source/`, with no suffix. The `notebooks/` segment is required. `guide/mmm/` already holds prose.

2. In `gallery.yaml`, set that card's `notebook` to the new docname and set `part` to `intro`, `guide`, or `case`. Do not tag any other card in the same PR.

3. Add one entry to `gallery/redirects.yaml`. `from` is the docname the example gallery publishes today (`notebooks/<category>/<stem>`). `to` is the new docname. Neither field is a URL. Do not include a language (`/en/`, `/es/`) or a version (`/stable/`, `/latest/`), a leading slash, or a host.

   ```yaml
   - from: notebooks/mmm/mmm_case_study
     to: gallery/notebooks/mmm/mmm_case_study
   ```

   The docs build writes a redirect page at the old docname. The refresh and canonical targets are relative to that page. For the pilot that target is `../../gallery/notebooks/mmm/mmm_case_study.html`. The built HTML contains neither `/en/` nor `https://www.pymc-marketing.io`, so a reader stays on the language and version they requested. The weekly link checker follows the redirect to a 200. A 404 on a `from` path is a failed migration. The old path is a redirect, not a document, and it is excluded from the sitemap.

4. Fix links inside the moved notebook that use a relative path which breaks because the file changed directory. That includes image paths into `docs/source/gallery/images/` and sibling notebook links. The pilot had none: its only links are external URLs. Links that point at other notebooks by the old absolute `https://www.pymc-marketing.io/en/stable/notebooks/...` URL stay. Those notebooks have not moved, and stable still serves them.

5. Run `python scripts/generate_gallery.py --no-thumbnails`. Commit the regenerated pages (`gallery/gallery.md`, `gallery/holding.md`, and a part page only if its grid changed). Do not commit `images/`. The card image path is by stem (`:img-top: images/<stem>.png` from `gallery/gallery.md`) and does not change when the notebook moves. `--check` does not require the PNG. Read the Docs runs `python scripts/generate_gallery.py` before `sphinx-build`, which is what makes the thumbnail readable. The new page is in that part page's toctree only. The holding toctree must not also include it.

6. If `locales/es/LC_MESSAGES/<old docname>.po` exists, create the new parent directory and `git mv` the catalog to `locales/es/LC_MESSAGES/<new docname>.po` in the same PR. Do not regenerate it from scratch. If no `.po` exists, do not create one.

7. Run `sphinx-intl update` so the `#:` locations follow the new path. Confirm a pre-existing `msgstr` is still present and that the message itself was not marked fuzzy. The catalog header stays `#, fuzzy`; that is not a translated message. Do not mark a message fuzzy to skip the check. The gettext filename is the path from the repository root and needs the `.ipynb` suffix. Sphinx looks that argument up from the working directory, not from `docs/source`. A bare docname, and a filename without the `docs/source/` prefix, are not files, and Sphinx skips them. Write the pots to `docs/gettext`, the sibling of `docs/source`. Locations are relative to that directory, so they stay `../source/<docname>.ipynb`, which is what every catalog already uses. An output directory outside the repository records an absolute location. `docs/gettext` is gitignored. Do not pass `-W`: a missing gitignored thumbnail and the pre-existing macOS autosummary stub fail the build and write no pots. Sphinx still reads the whole environment, so delete every pot except the moved docname before `sphinx-intl update`. Otherwise the update creates catalogs for notebooks that never had one.

   ```bash
   rm -rf docs/gettext
   sphinx-build docs/source docs/gettext -b gettext "docs/source/<new-docname>.ipynb"
   find docs/gettext -name '*.pot' ! -path "docs/gettext/<new-docname>.pot" -delete
   sphinx-intl update -p docs/gettext -l es --locale-dir locales
   rm -rf docs/gettext
   ```

   The pilot's title `msgstr` is still `Caso de Estudio Completo de MMM`, and that message is not fuzzy. The catalog header stays `#, fuzzy`. This catalog was last extracted on 2025-10-07, so the update also refreshed it against the current notebook. Strings whose English changed were marked fuzzy by that update, and their old translation is kept as the suggestion. New strings were added untranslated. Do not clear those flags by hand, and do not mark a message fuzzy yourself to skip the check. A pure move of a catalog that already matches the notebook only rewrites the `#:` locations and the `POT-Creation-Date`.

8. Do not edit unrelated `msgstr` URLs. Do not backfill catalogs for notebooks that never had one. Do not delete stale catalogs, including `notebooks/clv/sBG.po` and the `notebooks/clv/dev/` catalogs that no longer match a notebook. Do not retarget `/en/stable/` URLs inside `msgstr`. Redirects cover those 404s. Leave the part-page catalogs alone. Untranslated strings on a regenerated page fall back to English.

`--check` fails if `from` is still a source file, if `to` has no source file, if `to` is absolute or contains a language or version prefix, or if `locales/es/LC_MESSAGES/<from>.po` still exists. It does not fail when that notebook never had a `.po`. A missing `part` is not a failure. An empty card grid is not a failure.

`check-added-large-files` excludes `docs/source/notebooks/` and `docs/source/{getting_started,guide,gallery}/notebooks/`. A moved notebook is still a notebook. Do not drop those excludes. `ruff` ignores `B018` and `D103` on `docs/source/notebooks/*` only. The same ignores have to cover the three new notebook homes, or a moved notebook fails `B018` for a display expression. The pilot's cell that displays `channel_contribution_share` does. Do not edit the notebook to satisfy ruff.

English stays the source. Do not add `locales/en/`, and do not add a third language. The three part pages already have catalogs under `locales/es/LC_MESSAGES/`. Empty `msgstr` entries on those pages are fine until a translator fills them.

## Thumbnails

- PNG, roughly 4:3, around 600x450 pixels.
- Filename matches the notebook stem unless overridden with `thumb:`.
- Extraction runs only on the write pass. `--check` skips it.
- The grid layout uses the [Sphinx Design](https://sphinx-design.readthedocs.io/en/latest/grids.html) extension.
