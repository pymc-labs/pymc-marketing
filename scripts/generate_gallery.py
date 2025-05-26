#!/usr/bin/env python3
"""
Sphinx plugin to generate a gallery for notebooks

Modified from the pytensor project, which was modified from the pymc project,
which modified the seaborn project, which modified the mpld3 project.
"""

import base64
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import image

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Using fallback for gallery generation.")
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


def create_thumbnail(
    infile: str | Path,
    outfile: str | Path,
    width: int = 275,
    height: int = 275,
    cx: float = 0.5,
    cy: float = 0.5,
    border: int = 4,
) -> None:
    """
    Create a thumbnail of the given image file

    Parameters
    ----------
    infile : str or Path
        The path to the input image file
    outfile : str or Path
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
        logger.warning(f"Input file {infile} does not exist")
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
        logger.info(f"Created thumbnail: {outfile}")
    except Exception as e:
        logger.error(f"Error creating thumbnail for {infile}: {e}")
        # Copy default image
        shutil.copy(DEFAULT_IMG_LOC, outfile)


class NotebookProcessor:
    """
    Process a notebook to extract images and create thumbnails
    """

    def __init__(
        self, notebook_path: str | Path, category: str, temp_dir: str | Path
    ) -> None:
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

    def _use_default_image(self, reason: str = "") -> bool:
        """Create thumbnail from default image and copy to gallery directory"""
        if reason:
            logger.info(reason)
        create_thumbnail(DEFAULT_IMG_LOC, self.thumb_path)
        shutil.copy(self.thumb_path, self.gallery_img_path)
        return False

    def extract_first_image(self) -> bool:
        """
        Extract the first image from the notebook

        Returns
        -------
        bool
            True if an image was successfully extracted, False otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            # If matplotlib is not available, just copy the default image
            shutil.copy(DEFAULT_IMG_LOC, self.gallery_img_path)
            logger.info(
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
            return self._use_default_image(f"No image found in {self.notebook_path}")

        except Exception as e:
            logger.error(f"Error processing {self.notebook_path}: {e}")
            return self._use_default_image()


def find_notebooks(notebook_dir: Path = NOTEBOOK_DIR) -> dict[str, list[Path]]:
    """
    Find all notebooks in the notebook directory and return them by category

    Parameters
    ----------
    notebook_dir : Path
        The directory containing notebooks organized in subdirectories by category

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping category names to lists of notebook paths
    """
    notebooks_by_category: dict[str, list[Path]] = {}

    # Check if notebook directory exists
    if not notebook_dir.exists():
        logger.warning(f"Notebook directory {notebook_dir} does not exist.")
        return notebooks_by_category

    # Find all notebook categories (directories in notebook_dir)
    try:
        categories = [
            d
            for d in os.listdir(notebook_dir)
            if os.path.isdir(os.path.join(notebook_dir, d)) and not d.startswith(".")
        ]
    except Exception as e:
        logger.error(f"Error listing directory {notebook_dir}: {e}")
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
            logger.error(f"Error listing notebooks in {category_path}: {e}")
            notebooks_by_category[category] = []

    return notebooks_by_category


def create_default_image() -> None:
    """Create a default image if it doesn't exist"""
    if not os.path.exists(DEFAULT_IMG_LOC):
        logger.warning(f"Default image {DEFAULT_IMG_LOC} does not exist.")
        try:
            # Create a simple default image if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                logger.info("Creating a default image...")
                os.makedirs(os.path.dirname(DEFAULT_IMG_LOC), exist_ok=True)
                plt.figure(figsize=(4, 3))
                plt.text(
                    0.5, 0.5, "PyMC-Marketing", ha="center", va="center", fontsize=14
                )
                plt.savefig(DEFAULT_IMG_LOC)
                plt.close()
            else:
                logger.warning("Cannot create default image (matplotlib not available)")
        except Exception as e:
            logger.error(f"Error creating default image: {e}")


def process_notebooks(temp_dir: str | Path) -> tuple[int, int]:
    """
    Process all notebooks and create thumbnails

    Parameters
    ----------
    temp_dir : str or Path
        Path to temporary directory for storing intermediate files

    Returns
    -------
    Tuple[int, int]
        Tuple containing (success_count, total_count)
    """
    # Find all notebooks
    notebooks_by_category = find_notebooks()

    # Process each notebook
    success_count = 0
    total_count = 0

    for category, notebook_list in notebooks_by_category.items():
        logger.info(f"Processing category: {category}")
        for notebook_path in notebook_list:
            processor = NotebookProcessor(notebook_path, category, temp_dir)
            if processor.extract_first_image():
                success_count += 1
            total_count += 1

    logger.info(
        f"\nSuccessfully extracted images from {success_count} out of {total_count} notebooks."
    )
    logger.info(f"Gallery images are stored in {GALLERY_IMG_DIR}")
    return success_count, total_count


def main() -> None:
    """Main function to process notebooks and create thumbnails"""
    logger.info("Starting gallery generation...")

    # Check if default image exists and create if needed
    create_default_image()

    # Create a temporary directory for thumbnails
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temporary directory for thumbnails: {temp_dir}")

        # Process notebooks
        process_notebooks(temp_dir)

    # The temporary directory is automatically deleted when the context manager exits
    logger.info("Temporary thumbnail directory has been cleaned up")

    # Check if _thumbnails directory exists and remove it if it does
    thumbnails_dir = PROJECT_ROOT / "docs" / "source" / "_thumbnails"
    if thumbnails_dir.exists():
        logger.info(f"Removing old _thumbnails directory: {thumbnails_dir}")
        shutil.rmtree(thumbnails_dir)


if __name__ == "__main__":
    main()
