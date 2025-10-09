"""Script to run all notebooks in the docs/source/notebooks directory.

Examples
--------
Run all the notebooks in the documentation:

```terminal
python scripts/run_notebooks/runner.py
```

Run all the notebooks in docs/mmm and docs/clv:

```terminal
python scripts/run_notebooks/runner.py --notebooks mmm clv
```

Run all notebooks except those in docs/mmm and docs/clv:

```terminal
python scripts/run_notebooks/runner.py --exclude-dirs mmm clv
```

Run notebooks from index 2 to 5 (3rd to 5th notebook) in notebooks/mmm directory:

```terminal
python scripts/run_notebooks/runner.py --notebooks mmm --start-idx 2 --end-idx 5
```

"""

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
    parser.add_argument(
        "--exclude-dirs",
        nargs="+",
        type=str,
        help="List of directories to exclude (e.g., mmm clv)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Index of the notebook to start from (inclusive).",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Index of the notebook to end at (exclusive).",
    )
    parser.add_argument(
        "--parallel/no-parallel",
        dest="parallel",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def expand_directories(notebooks):
    expanded = []
    for notebook in notebooks:
        path = NOTEBOOKS_PATH / notebook
        if path.is_dir():
            logging.info(f"Expanding directory: {path}")
            expanded.extend(path.glob("*.ipynb"))
        else:
            expanded.append(notebook)
    return expanded


if __name__ == "__main__":
    notebooks_to_run = NOTEBOOKS

    args = parse_args()
    if args.notebooks:
        notebooks_to_run = [Path(notebook) for notebook in args.notebooks]

    notebooks_to_run = expand_directories(notebooks_to_run)

    if args.exclude_dirs:
        exclude_set = set(args.exclude_dirs)
        notebooks_to_run = [
            nb for nb in notebooks_to_run if nb.parent.name not in exclude_set
        ]

    notebooks_to_run = sorted(notebooks_to_run)

    notebooks_to_run = notebooks_to_run[args.start_idx : args.end_idx]

    def parallel_run():
        return Parallel(n_jobs=-1)(
            delayed(run_notebook)(**run_params)
            for run_params in run_parameters(notebooks_to_run)
        )

    def sequential_run():
        return [
            run_notebook(**run_params)
            for run_params in run_parameters(notebooks_to_run)
        ]

    run = parallel_run if args.parallel else sequential_run

    setup_logging()
    logging.info("Starting notebook runner")
    logging.info(f"Notebooks to run: {notebooks_to_run}")
    results = run()
    del results
    import gc

    gc.collect()

    logging.info("Notebooks run successfully!")
