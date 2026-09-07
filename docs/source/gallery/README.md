# PyMC-Marketing Example Gallery

This directory contains the gallery view for the example notebooks.

## How the gallery is built

`gallery.yaml` is the single source of truth for the gallery layout:
sections, optional subsections, per-card title, and the notebook path.
`scripts/generate_gallery.py` renders `gallery.md` from `gallery.yaml`
and extracts thumbnails from each notebook into `images/`. The pre-commit
hook `gallery-in-sync` blocks commits that leave `gallery.md` out of sync
with `gallery.yaml`.

## Adding a new example

1. Add the notebook under `docs/source/notebooks/<category>/`.
2. Add a card entry under the relevant section in `gallery.yaml`, e.g.
   ```yaml
   - title: My New Example
     notebook: mmm/my_new_example
   ```
   `notebook` is the path relative to `docs/source/notebooks/`, no
   extension. The thumbnail defaults to `images/<stem>.png`. Set an
   optional `thumb:` field to override.
3. Run `python scripts/generate_gallery.py` to regenerate `gallery.md`
   and extract the thumbnail from the first image cell of the notebook.
   If the notebook has no image cell, the default logo is used.
4. Commit `gallery.yaml`, the regenerated `gallery.md`, and the new
   `images/<stem>.png`.

## Checking sync without writing

```
python scripts/generate_gallery.py --check --no-thumbnails
```

Fails when `gallery.md` is out of sync with `gallery.yaml`, when a
notebook on disk is missing from the yaml, or when the yaml lists a
notebook that no longer exists. `dev/` drafts are excluded (matches
`exclude_patterns` in `conf.py`).

## Thumbnails

- PNG, roughly 4:3, around 600x450 pixels.
- Filename matches the notebook stem unless overridden with `thumb:` in
  the yaml.
- The grid layout uses the [Sphinx Design](https://sphinx-design.readthedocs.io/en/latest/grids.html)
  extension.
