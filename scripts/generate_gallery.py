#!/usr/bin/env python3
"""
Sphinx plugin to generate a gallery for notebooks

Modified from the pytensor project, which was modified from the pymc project,
which modified the seaborn project, which modified the mpld3 project.
"""

import base64
import json
import os
import shutil
import tempfile
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import image

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("WARNING: Matplotlib not available. Using fallback for gallery generation.")
    MATPLOTLIB_AVAILABLE = False

# Define directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "docs" / "source" / "notebooks"
GALLERY_DIR = PROJECT_ROOT / "docs" / "source" / "gallery"
GALLERY_IMG_DIR = GALLERY_DIR / "images"

# Create gallery images directory if it doesn't exist
GALLERY_IMG_DIR.mkdir(exist_ok=True, parents=True)

# Default image in case we can't extract one from a notebook
DEFAULT_IMG_LOC = PROJECT_ROOT / "docs" / "source" / "_static" / "flat_logo.png"

# Map folder names to display titles
folder_title_map = {
    "mmm": "Marketing Mix Models (MMM)",
    "clv": "Customer Lifetime Value (CLV) Models",
    "customer_choice": "Customer Choice Models",
    "general": "General Tutorials",
}


def create_thumbnail(infile, outfile, width=275, height=275, cx=0.5, cy=0.5, border=4):
    """
    Create a thumbnail of the given image file

    Parameters
    ----------
    infile : str
        The path to the input image file
    outfile : str
        The path to save the thumbnail
    width, height : int
        The width and height of the thumbnail in pixels
    cx, cy : float
        The center position of the crop as a fraction of the image size
    border : int
        The size of the border in pixels
    """
    if not MATPLOTLIB_AVAILABLE:
        # If matplotlib is not available, just copy the default image
        shutil.copy(DEFAULT_IMG_LOC, outfile)
        return

    if not os.path.exists(infile):
        print(f"Warning: Input file {infile} does not exist")
        # Copy default image
        shutil.copy(DEFAULT_IMG_LOC, outfile)
        return

    try:
        im = image.imread(infile)
        rows, cols = im.shape[:2]
        size = min(rows, cols)

        if size == cols:
            xslice = slice(0, size)
            ymin = min(max(0, int(cx * rows - size // 2)), rows - size)
            yslice = slice(ymin, ymin + size)
        else:
            yslice = slice(0, size)
            xmin = min(max(0, int(cx * cols - size // 2)), cols - size)
            xslice = slice(xmin, xmin + size)

        thumb = im[yslice, xslice]

        # Add a border
        if len(thumb.shape) == 3:  # Color image
            thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
            thumb[:, :border, :3] = thumb[:, -border:, :3] = 0
        else:  # Grayscale image
            thumb[:border, :] = thumb[-border:, :] = 0
            thumb[:, :border] = thumb[:, -border:] = 0

        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

        ax = fig.add_axes(
            [0, 0, 1, 1], aspect="auto", frameon=False, xticks=[], yticks=[]
        )
        ax.imshow(thumb, aspect="auto", resample=True, interpolation="bilinear")
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)
        print(f"Created thumbnail: {outfile}")
    except Exception as e:
        print(f"Error creating thumbnail for {infile}: {e}")
        # Copy default image
        shutil.copy(DEFAULT_IMG_LOC, outfile)


class NotebookProcessor:
    """
    Process a notebook to extract images and create thumbnails
    """

    def __init__(self, notebook_path, category, temp_dir):
        self.notebook_path = Path(notebook_path)
        self.category = category
        self.name = self.notebook_path.stem
        self.temp_dir = temp_dir

        # Create category thumbnail directory in temp dir
        self.thumb_dir = Path(temp_dir) / category
        self.thumb_dir.mkdir(exist_ok=True)

        # Define thumbnail and gallery image paths
        self.thumb_path = self.thumb_dir / f"{self.name}.png"
        self.gallery_img_path = GALLERY_IMG_DIR / f"{self.name}.png"

    def extract_first_image(self):
        """
        Extract the first image from the notebook
        """
        if not MATPLOTLIB_AVAILABLE:
            # If matplotlib is not available, just copy the default image
            shutil.copy(DEFAULT_IMG_LOC, self.gallery_img_path)
            print(
                f"Using default image for {self.notebook_path.name} (matplotlib not available)"
            )
            return False

        temp_img_path = Path(self.temp_dir) / f"{self.name}_temp.png"

        try:
            with open(self.notebook_path, encoding="utf-8") as f:
                notebook = json.load(f)

            # Look for the first image output
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue

                for output in cell.get("outputs", []):
                    if "data" in output and "image/png" in output["data"]:
                        # Found an image
                        img_data = output["data"]["image/png"]
                        img_bytes = base64.b64decode(img_data)

                        # Save the image temporarily
                        with open(temp_img_path, "wb") as img_file:
                            img_file.write(img_bytes)

                        # Create a thumbnail from the extracted image
                        create_thumbnail(temp_img_path, self.thumb_path)

                        # Copy the thumbnail to the gallery images directory
                        shutil.copy(self.thumb_path, self.gallery_img_path)

                        # Clean up temporary file
                        if temp_img_path.exists():
                            temp_img_path.unlink()

                        return True

            # No image found, use default
            print(f"No image found in {self.notebook_path}")
            create_thumbnail(DEFAULT_IMG_LOC, self.thumb_path)
            shutil.copy(self.thumb_path, self.gallery_img_path)
            return False

        except Exception as e:
            print(f"Error processing {self.notebook_path}: {e}")
            # Use default image
            create_thumbnail(DEFAULT_IMG_LOC, self.thumb_path)
            shutil.copy(self.thumb_path, self.gallery_img_path)
            return False


def find_notebooks(notebook_dir=NOTEBOOK_DIR):
    """
    Find all notebooks in the notebook directory and return them by category
    """
    notebooks_by_category = {}

    # Check if notebook directory exists
    if not notebook_dir.exists():
        print(f"Warning: Notebook directory {notebook_dir} does not exist.")
        return notebooks_by_category

    # Find all notebook categories (directories in notebook_dir)
    try:
        categories = [
            d
            for d in os.listdir(notebook_dir)
            if os.path.isdir(os.path.join(notebook_dir, d)) and not d.startswith(".")
        ]
    except Exception as e:
        print(f"Error listing directory {notebook_dir}: {e}")
        return notebooks_by_category

    for category in categories:
        category_path = os.path.join(notebook_dir, category)
        try:
            # Get all .ipynb files in the category directory
            notebook_paths = [
                Path(os.path.join(category_path, nb))
                for nb in os.listdir(category_path)
                if nb.endswith(".ipynb") and not nb.startswith(".")
            ]
            notebooks_by_category[category] = notebook_paths
        except Exception as e:
            print(f"Error listing notebooks in {category_path}: {e}")
            notebooks_by_category[category] = []

    return notebooks_by_category


def main():
    """Main function to process notebooks and create thumbnails"""
    print("Starting gallery generation...")

    # Check if default image exists
    if not os.path.exists(DEFAULT_IMG_LOC):
        print(f"Warning: Default image {DEFAULT_IMG_LOC} does not exist.")
        try:
            # Create a simple default image if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                print("Creating a default image...")
                os.makedirs(os.path.dirname(DEFAULT_IMG_LOC), exist_ok=True)
                plt.figure(figsize=(4, 3))
                plt.text(
                    0.5, 0.5, "PyMC Marketing", ha="center", va="center", fontsize=14
                )
                plt.savefig(DEFAULT_IMG_LOC)
                plt.close()
            else:
                print("Cannot create default image (matplotlib not available)")
        except Exception as e:
            print(f"Error creating default image: {e}")

    # Create a temporary directory for thumbnails
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory for thumbnails: {temp_dir}")

        # Find all notebooks
        notebooks_by_category = find_notebooks()

        # Process each notebook
        success_count = 0
        total_count = 0

        for category, notebook_list in notebooks_by_category.items():
            print(f"Processing category: {category}")
            for notebook_path in notebook_list:
                processor = NotebookProcessor(notebook_path, category, temp_dir)
                if processor.extract_first_image():
                    success_count += 1
                total_count += 1

        print(
            f"\nSuccessfully extracted images from {success_count} out of {total_count} notebooks."
        )
        print(f"Gallery images are stored in {GALLERY_IMG_DIR}")

    # The temporary directory is automatically deleted when the context manager exits
    print("Temporary thumbnail directory has been cleaned up")

    # Check if _thumbnails directory exists and remove it if it does
    thumbnails_dir = PROJECT_ROOT / "docs" / "source" / "_thumbnails"
    if thumbnails_dir.exists():
        print(f"Removing old _thumbnails directory: {thumbnails_dir}")
        shutil.rmtree(thumbnails_dir)


if __name__ == "__main__":
    main()
