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
import matplotlib

matplotlib.use("Agg")

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

SEED = 42


@pytest.fixture(scope="module")
def cv_results_idata():
    """Minimal az.InferenceData for MMMCVPlotSuite tests.

    Three folds over 30 daily dates:
      fold_0 — train 0-19, test 20-29
      fold_1 — train 0-24, test 25-29
      fold_2 — train 0-29, test [] (degenerate fold, no test rows)
    """
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    cv_labels = ["fold_0", "fold_1", "fold_2"]
    channels = ["tv", "radio"]
    n_chains, n_draws = 2, 50

    posterior_ds = xr.Dataset(
        {
            "beta_channel": xr.DataArray(
                rng.normal(size=(3, n_chains, n_draws, 2)),
                dims=["cv", "chain", "draw", "channel"],
                coords={
                    "cv": cv_labels,
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "channel": channels,
                },
            )
        }
    )

    pp_ds = xr.Dataset(
        {
            "y_original_scale": xr.DataArray(
                rng.normal(100, 10, size=(3, n_chains, n_draws, 30)),
                dims=["cv", "chain", "draw", "date"],
                coords={
                    "cv": cv_labels,
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "date": dates,
                },
            )
        }
    )

    fold_specs = [(20, 20), (25, 25), (30, 30)]
    meta_arr = np.empty(3, dtype=object)
    for i, (train_end, test_start) in enumerate(fold_specs):
        X_train = pd.DataFrame({"date": dates[:train_end]})
        y_train = pd.Series(rng.normal(100, 10, size=train_end), name="y")
        if test_start < 30:
            X_test = pd.DataFrame({"date": dates[test_start:]})
            y_test = pd.Series(rng.normal(100, 10, size=30 - test_start), name="y")
        else:
            X_test = pd.DataFrame({"date": pd.DatetimeIndex([])})
            y_test = pd.Series([], name="y", dtype=float)
        meta_arr[i] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    cv_metadata_ds = xr.Dataset(
        {
            "metadata": xr.DataArray(
                meta_arr,
                dims=["cv"],
                coords={"cv": cv_labels},
            )
        }
    )

    return az.InferenceData(
        posterior=posterior_ds,
        posterior_predictive=pp_ds,
        cv_metadata=cv_metadata_ds,
    )


@pytest.fixture(scope="module")
def cv_plot(cv_results_idata):
    from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

    return MMMCVPlotSuite(cv_results_idata)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestInit:
    def test_stores_cv_data(self, cv_results_idata):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

        suite = MMMCVPlotSuite(cv_results_idata)
        assert suite.cv_data is cv_results_idata

    def test_raises_type_error_for_non_idata(self):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

        with pytest.raises(TypeError, match=r"az\.InferenceData"):
            MMMCVPlotSuite({"not": "idata"})

    def test_raises_value_error_without_cv_metadata(self):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

        bad = az.InferenceData(posterior=xr.Dataset())
        with pytest.raises(ValueError, match="cv_metadata"):
            MMMCVPlotSuite(bad)
