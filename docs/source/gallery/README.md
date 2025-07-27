# PyMC-Marketing Example Gallery

This directory contains the gallery view for the PyMC-Marketing example notebooks.

## Gallery Structure

The gallery displays thumbnails and links to all example notebooks, organized by category.

## Adding New Examples to the Gallery

When adding new example notebooks:

1. Add the notebook to the appropriate directory in `docs/source/notebooks/`
2. Update the gallery entry in `gallery.md`
3. Create a thumbnail image for the notebook (ideally a screenshot of a key visualization from the notebook) and place it in the `images/` directory
4. Run `python create_gallery_images.py` to generate a placeholder thumbnail if you don't have a specific image

## Gallery Images

The gallery uses thumbnail images to provide visual navigation. For best results:

- Images should be in PNG format
- Images should have a 4:3 aspect ratio
- Size should be approximately 600x450 pixels
- Names should match the notebook filename

## Updating the Gallery

The gallery is structured using the Sphinx Design extension's grid layout. See the [Sphinx Design documentation](https://sphinx-design.readthedocs.io/en/latest/grids.html) for more information on customizing the grid layout.
