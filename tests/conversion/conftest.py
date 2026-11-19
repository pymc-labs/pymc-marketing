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
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.conversion import FunnelModel, to_funnel_dataset

STAGES = ["visit", "product_view", "add_to_cart", "checkout", "purchase"]
STEP_PROBABILITIES = {
    "control": [0.60, 0.30, 0.55, 0.70],
    "treatment": [0.60, 0.36, 0.55, 0.72],
}
FIT_KWARGS = {
    "chains": 2,
    "draws": 500,
    "tune": 500,
    "cores": 1,
    "random_seed": 1,
    "progressbar": False,
}


def simulate_stage_counts(
    step_probabilities: dict, n_entrants: int, seed: int
) -> pd.DataFrame:
    """Simulate the stage counts of a closed funnel for each group."""
    rng = np.random.default_rng(seed)
    rows = []
    for group, probabilities in step_probabilities.items():
        count = n_entrants
        rows.append({"stage": STAGES[0], "variant": group, "count": count})
        for stage, probability in zip(STAGES[1:], probabilities, strict=False):
            count = rng.binomial(count, probability)
            rows.append({"stage": stage, "variant": group, "count": count})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def ab_dataset() -> xr.Dataset:
    counts = simulate_stage_counts(STEP_PROBABILITIES, n_entrants=8_000, seed=0)
    return to_funnel_dataset(counts, stage_order=STAGES)


@pytest.fixture(scope="session")
def fitted_model(ab_dataset) -> FunnelModel:
    model = FunnelModel()
    model.fit(ab_dataset, **FIT_KWARGS)
    return model
