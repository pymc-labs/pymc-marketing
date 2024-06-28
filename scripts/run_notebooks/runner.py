import logging
from pathlib import Path

import papermill
from joblib import Parallel, delayed
from tqdm import tqdm

KERNEL_NAME: str = "python3"
NOTEBOOKS_PATH = Path("docs/source/notebooks")
NOTEBOOKS_SKIP: list[str] = [
    "clv_quickstart.ipynb",
    "mmm_budget_allocation_example.ipynb",
    "mmm_tvp_example.ipynb",
]
NOTEBOOKS: list[Path] = list(NOTEBOOKS_PATH.glob("*/*.ipynb"))
NOTEBOOKS = [nb for nb in NOTEBOOKS if nb.name not in NOTEBOOKS_SKIP]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_cwd_from_notebook_path(notebook_path: Path) -> str:
    return str(notebook_path).rsplit("/", 1)[0]


def run_notebook(notebook_path: Path) -> None:
    cwd = get_cwd_from_notebook_path(notebook_path)
    logging.info(f"Running notebook: {notebook_path.name}")
    papermill.execute_notebook(
        input_path=str(notebook_path),
        output_path=None,
        kernel_name=KERNEL_NAME,
        cwd=cwd,
    )


if __name__ == "__main__":
    setup_logging()
    logging.info("Starting notebook runner")
    Parallel(n_jobs=-1)(
        delayed(run_notebook)(notebook_path) for notebook_path in tqdm(NOTEBOOKS)
    )

    logging.info("Notebooks run successfully!")
