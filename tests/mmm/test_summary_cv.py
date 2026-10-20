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
"""Tests for MMMCVSummaryFactory summary DataFrames."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite
from pymc_marketing.mmm.summary.cv import MMMCVSummaryFactory

SEED = 42


@pytest.fixture(scope="module")
def cv_results_idata():
    """Minimal xr.DataTree for MMMCVSummaryFactory tests."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    cv_labels = ["fold_0", "fold_1"]
    channels = ["tv", "radio"]
    n_chains, n_draws = 2, 10

    posterior_ds = xr.Dataset(
        {
            "beta_channel": xr.DataArray(
                rng.normal(size=(2, n_chains, n_draws, len(channels))),
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
                rng.normal(100, 10, size=(2, n_chains, n_draws, len(dates))),
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

    fold_specs = [(8, 8), (10, 10)]
    meta_arr = np.empty(2, dtype=object)
    for i, (train_end, test_start) in enumerate(fold_specs):
        X_train = pd.DataFrame({"date": dates[:train_end]})
        y_train = pd.Series(rng.normal(100, 10, size=train_end), name="y")
        X_test = pd.DataFrame({"date": dates[test_start:]})
        y_test = pd.Series(rng.normal(100, 10, size=len(dates) - test_start), name="y")
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

    return xr.DataTree.from_dict(
        {
            "/posterior": posterior_ds,
            "/posterior_predictive": pp_ds,
            "/cv_metadata": cv_metadata_ds,
        }
    )


@pytest.fixture(scope="module")
def cv_summary_factory(cv_results_idata):
    return MMMCVSummaryFactory(cv_results_idata)


class TestMMMCVSummarySchemas:
    """Test MMMCVSummaryFactory DataFrame schemas."""

    def test_predictions_schema(self, cv_summary_factory):
        """Test predictions returns DataFrame with correct schema."""
        df = cv_summary_factory.predictions(hdi_probs=[0.94])

        required_columns = {"cv", "date", "split", "mean", "median", "observed"}
        assert required_columns.issubset(set(df.columns))
        assert "abs_error_94_lower" in df.columns
        assert "abs_error_94_upper" in df.columns
        assert len(df) > 0

    def test_param_stability_schema(self, cv_summary_factory):
        """Test param_stability returns DataFrame with correct schema."""
        df = cv_summary_factory.param_stability(
            var_names=["beta_channel"],
            hdi_probs=[0.94],
        )

        required_columns = {"cv", "variable", "mean", "median"}
        assert required_columns.issubset(set(df.columns))
        assert "abs_error_94_lower" in df.columns
        assert "abs_error_94_upper" in df.columns
        assert set(df["variable"].unique()) == {"beta_channel"}

    def test_crps_schema(self, cv_summary_factory):
        """Test crps returns DataFrame with correct schema."""
        df = cv_summary_factory.crps()

        required_columns = {"split", "cv", "mean_crps"}
        assert required_columns.issubset(set(df.columns))
        assert len(df) > 0


class TestMMMCVPlotSuiteSummary:
    """Test MMMCVPlotSuite.summary property."""

    def test_plot_suite_summary_returns_factory(self, cv_results_idata):
        """MMMCVPlotSuite.summary returns MMMCVSummaryFactory bound to cv_data."""
        plot_suite = MMMCVPlotSuite(cv_results_idata)
        summary = plot_suite.summary
        assert isinstance(summary, MMMCVSummaryFactory)
        assert summary.cv_data is cv_results_idata
