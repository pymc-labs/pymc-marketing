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
Each notebook has one toctree home. The pre-commit hook `gallery-in-sync` fails the commit if a notebook is missing, duplicated, tagged with an unknown part, listed in more than one toctree, or if `gallery/redirects.yaml` is invalid.

`notebook` is a docname relative to `docs/source/`, with no suffix. A notebook that has not moved yet is `notebooks/<category>/<stem>`.
After it is tagged, the docname becomes `getting_started/notebooks/<category>/<stem>`, `guide/notebooks/<category>/<stem>`, or `gallery/notebooks/<category>/<stem>`.
Do not put the same notebook in two cards.

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
It fails when a generated page is out of sync, when a published notebook is missing from the yaml, when the yaml lists a missing file, when a notebook has two toctree homes, or when `gallery/redirects.yaml` is invalid.
A missing `part` is not a failure. An empty redirect map is valid.

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

## Thumbnails

- PNG, roughly 4:3, around 600x450 pixels.
- Filename matches the notebook stem unless overridden with `thumb:`.
- Extraction runs only on the write pass. `--check` skips it.
- The grid layout uses the [Sphinx Design](https://sphinx-design.readthedocs.io/en/latest/grids.html) extension.
