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

A migration PR adds one entry to `gallery/redirects.yaml` and removes the old notebook file in the same change. It does not edit the Read the Docs dashboard.
`from` is the docname the example gallery publishes today. `to` is the new docname under `getting_started/notebooks/`, `guide/notebooks/`, or `gallery/notebooks/`.
Neither field is a URL. Do not include a language (`/en/`, `/es/`) or a version (`/stable/`, `/latest/`).

```yaml
- from: notebooks/mmm/mmm_case_study
  to: gallery/notebooks/mmm/mmm_case_study
```

The docs build writes a redirect page at the old docname. The refresh and canonical targets are relative to that page, so a reader stays on the language and version they requested.
The weekly link checker follows the redirect to a 200. A 404 on a `from` path is a failed migration.
`--check` fails if `from` is still a source file, if `to` has no source file, or if `to` is absolute or contains a language or version prefix.
It also fails if `locales/es/LC_MESSAGES/<from>.po` still exists. It does not fail when that notebook never had a `.po`.

Move the Spanish catalog in the same PR. English stays the source. Do not add `locales/en/`, and do not add a third language.

1. Move the source to the new docname.
2. If `locales/es/LC_MESSAGES/<old docname>.po` exists, `git mv` it to `locales/es/LC_MESSAGES/<new docname>.po` in the same PR. Do not regenerate it from scratch.
3. Run `sphinx-intl update` so the `#:` locations follow the new path. Confirm a pre-existing `msgstr` is still present. Do not mark the catalog fuzzy as a way to skip the check. Limit the gettext build to the moved docname, and use a clean pot directory, so the update does not create catalogs for notebooks that never had one.

   ```bash
   pot=$(mktemp -d)
   sphinx-build docs/source "$pot" -b gettext "docs/source/<new-docname>"
   sphinx-intl update -p "$pot" -l es --locale-dir locales
   ```
4. If no `.po` exists, do not create one in the migration PR.
5. Do not edit unrelated `msgstr` URLs.

Do not backfill catalogs for notebooks that never had one. Do not delete stale catalogs, including `notebooks/clv/sBG.po` and the `notebooks/clv/dev/` catalogs that no longer match a notebook. Do not retarget `/en/stable/` URLs inside `msgstr`. Redirects cover those 404s.

The three part pages already have catalogs under `locales/es/LC_MESSAGES/`. Empty `msgstr` entries on those pages are fine until a translator fills them.

## Thumbnails

- PNG, roughly 4:3, around 600x450 pixels.
- Filename matches the notebook stem unless overridden with `thumb:`.
- Extraction runs only on the write pass. `--check` skips it.
- The grid layout uses the [Sphinx Design](https://sphinx-design.readthedocs.io/en/latest/grids.html) extension.
