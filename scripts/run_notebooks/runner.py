"""Script to run all notebooks in the docs/source/notebooks directory."""

import logging
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import papermill
from joblib import Parallel, delayed
from nbformat.notebooknode import NotebookNode
from papermill.iorw import load_notebook_node, write_ipynb
from tqdm import tqdm

KERNEL_NAME: str = "python3"
DOC_SOURCE = Path("docs/source")
NOTEBOOKS_PATH = DOC_SOURCE / "notebooks"
NOTEBOOKS_SKIP: list[str] = [
    "mmm_tvp_example.ipynb",  # This notebook takes too long to run
]
NOTEBOOKS: list[Path] = list(NOTEBOOKS_PATH.glob("*/*.ipynb"))
NOTEBOOKS = [nb for nb in NOTEBOOKS if nb.name not in NOTEBOOKS_SKIP]
NOTEBOOKS.append(DOC_SOURCE / "guide" / "benefits" / "model_deployment.ipynb")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


INJECTED_CODE = """
import pymc as pm
import arviz as az
import xarray as xr
import numpy as np

def mock_sample(*args, **kwargs):
    model = kwargs.get("model", None)
    samples = 10
    idata = pm.sample_prior_predictive(model=model, samples=samples)
    idata.add_groups(posterior=idata.prior)

    # Create mock sample stats with diverging data
    if "sample_stats" not in idata:
        n_chains = 1
        n_draws = samples
        sample_stats = xr.Dataset({
            "diverging": xr.DataArray(
                np.zeros((n_chains, n_draws), dtype=int),
                dims=("chain", "draw"),
            )
        })
        idata.add_groups(sample_stats=sample_stats)

    del idata.prior
    if "prior_predictive" in idata:
        del idata.prior_predictive
    return idata

pm.sample = mock_sample
"""


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
        papermill.execute_notebook(
            input_path=f.name,
            output_path=None,
            progress_bar=dict(desc=notebook_path.name),
            kernel_name=KERNEL_NAME,
            cwd=notebook_path.parent,
        )


def actual_run(notebook_path: Path) -> None:
    papermill.execute_notebook(
        input_path=notebook_path,
        output_path=None,
        kernel_name=KERNEL_NAME,
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


if __name__ == "__main__":
    SLICE = slice(-2, None)
    MOCK = True

    NOTEBOOKS = NOTEBOOKS[SLICE]

    setup_logging()
    logging.info("Starting notebook runner")
    logging.info(f"Notebooks to run: {NOTEBOOKS}")
    run = partial(run_notebook, mock=MOCK)
    Parallel(n_jobs=-1)(
        delayed(run)(notebook_path) for notebook_path in tqdm(NOTEBOOKS)
    )

    logging.info("Notebooks run successfully!")
