import logging
import os

import papermill as pm
from tqdm import tqdm

KERNEL_NAME: str = "python3"
NOTEBOOKS_PATH: str = "docs/source/notebooks"
OUTPUT_PATH: str = "scripts/run_notebooks/outputs"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_notebooks_names() -> list[str]:
    return [
        "clv/bg_nbd.ipynb",
        "clv/clv_quickstart.ipynb",
        "clv/gamma_gamma.ipynb",
        "clv/sBG.ipynb",
        "mmm/mmm_budget_allocation_example.ipynb",
        "mmm/mmm_example.ipynb",
        "mmm/mmm_lift_test.ipynb",
    ]


def get_notebooks_paths() -> list[str]:
    return [
        f"{NOTEBOOKS_PATH}/{notebook_name}" for notebook_name in get_notebooks_names()
    ]


def verify_notebooks_exist() -> None:
    for notebook_path in get_notebooks_paths():
        if not os.path.exists(notebook_path):
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")


def make_output_directory() -> None:
    os.makedirs(OUTPUT_PATH, exist_ok=True)


def run_notebook(notebook_path: str) -> None:
    output_path = construct_output_path(notebook_path)
    pm.execute_notebook(
        input_path=notebook_path,
        output_path=output_path,
        kernel_name=KERNEL_NAME,
    )


def construct_output_path(notebook_path: str) -> str:
    return f"{OUTPUT_PATH}/{notebook_path.split('/')[-1]}"


if __name__ == "__main__":
    setup_logging()
    logging.info("Starting notebook runner")

    make_output_directory()
    logging.info("Output directory created")

    notebooks_paths = get_notebooks_paths()
    verify_notebooks_exist()

    for notebook_path in tqdm(notebooks_paths):
        logging.info(f"Running notebook: {notebook_path}")
        run_notebook(notebook_path)

    logging.info("Notebooks run successfully!")
