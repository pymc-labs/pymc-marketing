#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Integration test for ``taste_profiles`` running through papermill.

The unit tests in ``test_taste_profiles.py`` exercise the public functions
against an Agg matplotlib backend, but they do not go through the actual
notebook-execution pipeline (kernel startup, cell ordering, output capture,
error propagation). This integration test fills that gap.

Follows the project's house style for notebook execution:

* `papermill.execute_notebook` as the entry point (matches
  ``scripts/run_notebooks/runner.py``).
* `pm.sample` is replaced by ``pymc.testing.mock_sample`` so the notebook
  completes in ~1s instead of the ~15s a real fit would take. The same
  mocking idiom lives in ``scripts/run_notebooks/injected.py`` and is
  inlined here so the test does not import from ``scripts/``.

A mocked posterior has the right *shape* but is not informative, so the
assertions in the notebook focus on the things the integration test is
supposed to cover: shapes, dtype, contract invariants (rows summing to 1,
Gini in [0, 1]). Numerical-accuracy assertions belong in
``test_taste_profiles.py`` where the model is genuinely fit.
"""

import nbformat
import papermill
import pytest
from papermill.iorw import write_ipynb

# Mock injection identical to the four operative lines of
# scripts/run_notebooks/injected.py. Inlined here so this test does not
# depend on the scripts/ layout.
MOCK_INJECTION = """\
from functools import partial
import numpy as np
import pymc as pm
import pymc.testing

def _mock_diverging(size):
    return np.zeros(size, dtype=int)

pm.sample = partial(
    pymc.testing.mock_sample,
    sample_stats={"diverging": _mock_diverging},
)
"""

NOTEBOOK_BODY = """\
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.constrained_layout.use"] = True

from matplotlib.figure import Figure

from pymc_marketing.customer_choice import (
    BayesianBLP, generate_blp_panel, taste_profiles,
)

# --- fit a small multi-dim model (pm.sample is mocked) -------------------
df, truth = generate_blp_panel(
    T=10, J=3, K=2, L=2,
    true_alpha=-2.0, true_beta=np.array([0.8, 1.2]),
    sigma_alpha=0.5, sigma_beta=np.array([0.4, 0.5]),
    instrument_strength=0.7, price_xi_corr=0.6,
    market_size=2_000, n_dgp_draws=1_000,
    random_seed=42, return_truth=True,
)
model = BayesianBLP(
    market_data=df,
    characteristics=truth["characteristic_cols"],
    instruments=truth["instrument_cols"],
    random_coef_on=["price", *truth["characteristic_cols"]],
    n_mc_draws=60,
    random_seed=0,
)
model.fit(draws=20, tune=20, chains=2, progressbar=False, random_seed=0)

# --- consumer_taste_grid -------------------------------------------------
grid = taste_profiles.consumer_taste_grid(model)
assert grid.shape == (model.n_mc_draws, 3)
assert list(grid.columns) == list(model._random_coef_names)

# --- buyer_nu_posterior --------------------------------------------------
nu_bar = taste_profiles.buyer_nu_posterior(model, n_samples=30)
assert nu_bar.shape == (30, model._M, 3)

# --- brand_buyer_nu for every dim ----------------------------------------
for dim in range(3):
    bn = taste_profiles.brand_buyer_nu(model, n_samples=30, dim=dim)
    assert bn.shape == (model._M, model._J)

# --- demand_concentration_gini ------------------------------------------
gini = taste_profiles.demand_concentration_gini(model, n_samples=30)
assert gini.shape == (30, model._M)
assert (gini >= -1e-9).all() and (gini <= 1.0 + 1e-9).all()

# --- taste_type_demand_share contract: bucket columns sum to 1 ----------
profiles = taste_profiles.taste_type_demand_share(model, n_samples=30)
assert len(profiles) == model._M
bucket_cols = ["sensitive_pct", "modal_pct", "insensitive_pct"]
sums = profiles[bucket_cols].sum(axis=1)
assert np.allclose(sums.values, 1.0, atol=1e-9)

# --- plotters return Figure objects --------------------------------------
for fig in [
    taste_profiles.plot_taste_profile_stacked(model, n_samples=30),
    taste_profiles.plot_buyer_profile_heatmap(model, n_samples=30),
    taste_profiles.plot_brand_buyer_heatmap(model, n_samples=30, dim=0),
    taste_profiles.plot_demand_concentration(model, n_samples=30),
]:
    assert isinstance(fig, Figure)
    plt.close(fig)

print("OK: all taste_profiles functions executed through papermill")
"""


def _build_notebook(*sources: str) -> nbformat.NotebookNode:
    """Return a multi-cell notebook (one cell per ``sources`` entry)."""
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(s) for s in sources]
    return nb


def test_taste_profiles_round_trip_through_papermill(tmp_path):
    """Build → write → execute via papermill → assert success sentinel.

    Runs in ~1s because ``pm.sample`` is mocked. The test is not marked
    slow; it is intended to run on the default ``pytest`` invocation.
    """
    nb = _build_notebook(MOCK_INJECTION, NOTEBOOK_BODY)
    in_path = tmp_path / "taste_profiles_smoke.ipynb"
    write_ipynb(nb, str(in_path))

    out_nb = papermill.execute_notebook(
        input_path=str(in_path),
        output_path=None,
        kernel_name="python3",
        progress_bar=False,
        cwd=str(tmp_path),
    )

    streams = [
        "".join(out.get("text", ""))
        for cell in out_nb.cells
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream"
    ]
    combined = "".join(streams)
    assert "OK: all taste_profiles functions executed through papermill" in combined, (
        f"success marker missing from notebook stdout; got:\n{combined[-500:]}"
    )


def test_taste_profiles_round_trip_propagates_cell_errors(tmp_path):
    """A failing assertion inside the notebook surfaces as a test failure.

    Without this guard, a silent regression in any taste_profiles function
    could appear as 'no exception, no success marker' which the success
    assertion above would catch as a missing-marker error. This test pins
    the failure path explicitly: errors raised inside cells become
    ``papermill.PapermillExecutionError`` at the test boundary.
    """
    nb = _build_notebook(
        MOCK_INJECTION,
        "raise RuntimeError('intentional failure for the integration test')\n",
    )
    in_path = tmp_path / "intentional_fail.ipynb"
    write_ipynb(nb, str(in_path))

    with pytest.raises(papermill.PapermillExecutionError, match="intentional failure"):
        papermill.execute_notebook(
            input_path=str(in_path),
            output_path=None,
            kernel_name="python3",
            progress_bar=False,
            cwd=str(tmp_path),
        )
