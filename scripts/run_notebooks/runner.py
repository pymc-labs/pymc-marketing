"""Script to run all notebooks in the docs/source/notebooks directory."""

import argparse
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypedDict
from uuid import uuid4

import papermill
from joblib import Parallel, delayed
from nbformat.notebooknode import NotebookNode
from papermill.iorw import load_notebook_node, write_ipynb

HERE = Path(__file__).parent

KERNEL_NAME: str = "python3"
DOC_SOURCE = Path("docs/source")
NOTEBOOKS_PATH = DOC_SOURCE / "notebooks"
NOTEBOOKS: list[Path] = list(NOTEBOOKS_PATH.glob("*/*.ipynb"))
NOTEBOOKS.append(DOC_SOURCE / "guide" / "benefits" / "model_deployment.ipynb")

INJECTED_CODE_FILE = HERE / "injected.py"
INJECTED_CODE = INJECTED_CODE_FILE.read_text()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def generate_random_id() -> str:
    return str(uuid4())


def inject_pymc_sample_mock_code(cells: list) -> None:
    cells.insert(
        0,
        NotebookNode(
            id=f"code-injection-{generate_random_id()}",
            execution_count=sum(map(ord, "Mock pm.sample")),
            cell_type="code",
            metadata={"tags": []},
            outputs=[],
            source=INJECTED_CODE,
        ),
    )


def mock_run(notebook_path: Path) -> None:
    nb = load_notebook_node(str(notebook_path))
    inject_pymc_sample_mock_code(nb.cells)
    with NamedTemporaryFile(suffix=".ipynb") as f:
        write_ipynb(nb, f.name)
        desc = f"Mocked {notebook_path.name}"
        papermill.execute_notebook(
            input_path=f.name,
            output_path=None,
            progress_bar=dict(desc=desc),
            kernel_name=KERNEL_NAME,
            cwd=notebook_path.parent,
        )


def actual_run(notebook_path: Path) -> None:
    papermill.execute_notebook(
        input_path=notebook_path,
        output_path=None,
        kernel_name=KERNEL_NAME,
        progress_bar={"desc": f"Running {notebook_path.name}"},
        cwd=notebook_path.parent,
    )


def run_notebook(notebook_path: Path, mock: bool = True) -> None:
    logging.info(f"Running notebook: {notebook_path.name}")
    run = mock_run if mock else actual_run

    try:
        run(notebook_path)
    except Exception as e:
        logging.error(f"Error running notebook: {notebook_path.name}")
        raise e


class RunParams(TypedDict):
    notebook_path: Path
    mock: bool


def run_parameters(notebook_paths: list[Path]) -> list[RunParams]:
    def to_mock(notebook_path: Path) -> RunParams:
        return RunParams(notebook_path=notebook_path, mock=True)

    return [to_mock(notebook_path) for notebook_path in notebook_paths]


def parse_args():
    parser = argparse.ArgumentParser(description="Run notebooks.")
    parser.add_argument(
        "--notebooks",
        nargs="+",
        type=str,
        help="List of notebooks to run. If not provided, all notebooks will be run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    notebooks_to_run = NOTEBOOKS

    args = parse_args()
    if args.notebooks:
        notebooks_to_run = [Path(notebook) for notebook in args.notebooks]

    setup_logging()
    logging.info("Starting notebook runner")
    logging.info(f"Notebooks to run: {notebooks_to_run}")
    Parallel(n_jobs=-1)(
        delayed(run_notebook)(**run_params)
        for run_params in run_parameters(notebooks_to_run)
    )

    logging.info("Notebooks run successfully!")
