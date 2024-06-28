import logging
import os

import papermill
from tqdm import tqdm

KERNEL_NAME: str = "python3"
NOTEBOOKS_PATH: str = "docs/source/notebooks"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_notebooks_names() -> list[str]:
    return [
        "clv/bg_nbd.ipynb",
        # "clv/clv_quickstart.ipynb",
        "clv/gamma_gamma.ipynb",
        "clv/pareto_nbd.ipynb",
        "clv/sBG.ipynb",
        "mmm/mmm_budget_allocation_example.ipynb",
        "mmm/mmm_components.ipynb",
        "mmm/mmm_example.ipynb",
        "mmm/mmm_lift_test.ipynb",
        "mmm/mmm_tvp_example.ipynb",
    ]


def get_notebooks_paths() -> list[str]:
    return [
        f"{NOTEBOOKS_PATH}/{notebook_name}" for notebook_name in get_notebooks_names()
    ]


def verify_notebooks_exist() -> None:
    for notebook_path in get_notebooks_paths():
        if not os.path.exists(notebook_path):
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")


def get_cwd_from_notebook_path(notebook_path: str) -> str:
    return notebook_path.rsplit("/", 1)[0]


def run_notebook(notebook_path: str) -> None:
    cwd = get_cwd_from_notebook_path(notebook_path)
    papermill.execute_notebook(
        input_path=notebook_path,
        output_path=None,
        kernel_name=KERNEL_NAME,
        cwd=cwd,
    )


if __name__ == "__main__":
    setup_logging()
    logging.info("Starting notebook runner")

    notebooks_paths = get_notebooks_paths()
    verify_notebooks_exist()

    for notebook_path in tqdm(notebooks_paths):
        logging.info(f"Running notebook: {notebook_path}")
        run_notebook(notebook_path)

    logging.info("Notebooks run successfully!")
