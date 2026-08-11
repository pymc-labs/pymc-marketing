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
"""Edge-case tests for summary helpers and factories (codecov patch coverage)."""

pytest_plugins = [
    "tests.mmm.test_summary",
    "tests.mmm.test_summary_budget",
    "tests.mmm.test_summary_cv",
]

from unittest.mock import MagicMock, patch  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from pymc_marketing.data.idata import MMMIDataWrapper  # noqa: E402
from pymc_marketing.mmm.summary import MMMSummaryFactory  # noqa: E402
from pymc_marketing.mmm.summary_budget import BudgetSummaryFactory  # noqa: E402
from pymc_marketing.mmm.summary_cv import (  # noqa: E402
    MMMCVSummaryFactory,
    _crps_for_split,
    _pred_matrix_for_rows,
)
from pymc_marketing.mmm.summary_helpers import (  # noqa: E402
    StatsHelper,
    compute_summary_stats_with_hdi,
    compute_waterfall_components,
    prepare_sensitivity_data,
)
from pymc_marketing.mmm.summary_sensitivity import sensitivity_analysis  # noqa: E402
from pymc_marketing.mmm.time_slice_cross_validation import (  # noqa: E402
    TimeSliceCrossValidator,
)
from pymc_marketing.mmm.xarray_utils import _apply_aggregation  # noqa: E402


@pytest.fixture
def simple_wrapper(mock_mmm_idata_wrapper):
    """Alias for the shared mock wrapper."""
    return mock_mmm_idata_wrapper


class TestSummaryHelpersEdgeCases:
    def test_compute_waterfall_components_raises_when_empty(self, simple_wrapper):
        with patch.object(
            simple_wrapper,
            "get_contributions",
            return_value=xr.Dataset(),
        ):
            with pytest.raises(ValueError, match="No contribution data found"):
                compute_waterfall_components(simple_wrapper)

    def test_prepare_sensitivity_data_absolute_spend(self, simple_wrapper):
        sweep = np.linspace(0.5, 1.5, 3)
        sa_da = xr.DataArray(
            np.random.default_rng(1).normal(size=(10, 3)),
            dims=("sample", "sweep"),
            coords={"sample": np.arange(10), "sweep": sweep},
        )
        sweep_x, processed = prepare_sensitivity_data(
            sa_da,
            simple_wrapper,
            x_sweep_axis="absolute",
            apply_cost_per_unit=True,
        )
        assert set(processed.dims) == {"chain", "draw", "sweep"}
        assert float(sweep_x.sel(sweep=sweep[0]).item()) != float(sweep[0])

    def test_prepare_sensitivity_data_absolute_channel_data(self, simple_wrapper):
        sweep = np.linspace(0.5, 1.5, 3)
        sa_da = xr.DataArray(
            np.random.default_rng(2).normal(size=(10, 3)),
            dims=("sample", "sweep"),
            coords={"sample": np.arange(10), "sweep": sweep},
        )
        sweep_x, _ = prepare_sensitivity_data(
            sa_da,
            simple_wrapper,
            x_sweep_axis="absolute",
            apply_cost_per_unit=False,
        )
        assert sweep_x.ndim == 1

    def test_stats_helper_invalid_hdi_prob(self):
        with pytest.raises(ValueError, match="HDI probability must be between 0 and 1"):
            StatsHelper().validate_hdi_probs([1.5])

    def test_stats_helper_unknown_output_format(self):
        with pytest.raises(ValueError, match="Unknown output_format"):
            StatsHelper().convert_output(pd.DataFrame({"a": [1]}), "spark")  # type: ignore[arg-type]

    def test_compute_summary_stats_with_hdi_sample_dim(self):
        da = xr.DataArray(
            np.random.default_rng(3).normal(size=(20, 2)),
            dims=("sample", "channel"),
            coords={"sample": np.arange(20), "channel": ["A", "B"]},
        )
        df = compute_summary_stats_with_hdi(da, hdi_probs=[0.94])
        assert {"channel", "mean", "median", "abs_error_94_lower"}.issubset(df.columns)

    def test_compute_summary_stats_with_hdi_invalid_dims(self):
        da = xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0, 1]})
        with pytest.raises(ValueError, match="must have either"):
            compute_summary_stats_with_hdi(da, hdi_probs=[0.94])

    def test_stats_helper_polars_output(self):
        df = StatsHelper().convert_output(pd.DataFrame({"a": [1]}), "polars")
        assert hasattr(df, "to_pandas")

    def test_stats_helper_polars_missing_raises(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "polars", None)
        with pytest.raises(ImportError, match="Polars is required"):
            StatsHelper().convert_output(pd.DataFrame({"a": [1]}), "polars")


class TestXarrayUtilsEdgeCases:
    def test_apply_aggregation_rejects_multiple_ops(self):
        da = xr.DataArray([1, 2, 3], dims=("x",))
        with pytest.raises(ValueError, match="Only a single aggregation operation"):
            _apply_aggregation(da, {"sum": "x", "mean": "x"})

    def test_apply_aggregation_mean(self):
        da = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=("channel", "date"),
            coords={"channel": ["A", "B"], "date": [0, 1]},
        )
        result = _apply_aggregation(da, {"mean": "date"})
        assert result.dims == ("channel",)

    def test_apply_aggregation_unknown_op(self):
        da = xr.DataArray([1, 2, 3], dims=("x",))
        with pytest.raises(ValueError, match="Unknown aggregation operation"):
            _apply_aggregation(da, {"median": "x"})


class TestSummarySensitivityEdgeCases:
    def test_missing_sensitivity_group(self, simple_wrapper):
        with pytest.raises(ValueError, match="sensitivity_analysis"):
            sensitivity_analysis(simple_wrapper)

    def test_missing_sensitivity_variable(self, simple_wrapper):
        simple_wrapper.idata.sensitivity_analysis = xr.Dataset(
            {"other": xr.DataArray([1.0], dims=("sweep",), coords={"sweep": [1.0]})}
        )
        with pytest.raises(ValueError, match="'x' not found"):
            sensitivity_analysis(simple_wrapper)

    def test_absolute_axis_multidim_sweep_x(self, simple_wrapper):
        sweep = np.linspace(0.5, 1.5, 3)
        regions = ["A", "B"]
        channels = ["TV", "Radio", "Social"]
        sa_da = xr.DataArray(
            np.random.default_rng(4).normal(size=(10, 3, 2, 3)),
            dims=("sample", "sweep", "region", "channel"),
            coords={
                "sample": np.arange(10),
                "sweep": sweep,
                "region": regions,
                "channel": channels,
            },
        )
        simple_wrapper.idata.sensitivity_analysis = xr.Dataset({"x": sa_da})
        df = sensitivity_analysis(
            simple_wrapper,
            aggregation={"sum": "channel"},
            x_sweep_axis="absolute",
        )
        assert "sweep_x" in df.columns
        assert len(df) > 0

    def test_absolute_axis_multidim_sweep_x_by_region(self, mock_mmm_idata_wrapper):
        """Cover multi-dimensional sweep_x merge path in _add_sweep_x_column."""
        sweep = np.linspace(0.5, 1.5, 3)
        regions = ["A", "B"]
        channels = ["TV", "Radio", "Social"]
        sa_da = xr.DataArray(
            np.random.default_rng(5).normal(size=(10, 3, 2, 3)),
            dims=("sample", "sweep", "region", "channel"),
            coords={
                "sample": np.arange(10),
                "sweep": sweep,
                "region": regions,
                "channel": channels,
            },
        )
        mock_mmm_idata_wrapper.idata.sensitivity_analysis = xr.Dataset({"x": sa_da})
        df = sensitivity_analysis(
            mock_mmm_idata_wrapper,
            dims={"region": ["A"]},
            x_sweep_axis="absolute",
        )
        assert "sweep_x" in df.columns
        assert set(df["region"].unique()) == {"A"}


class TestMMMSummaryEdgeCases:
    def test_prior_predictive_missing_variable(self, mock_mmm_idata_with_prior):
        factory = MMMSummaryFactory(mock_mmm_idata_with_prior)
        with patch(
            "pymc_marketing.mmm.summary.get_prior_for_plot",
            return_value=xr.Dataset({"y": xr.DataArray([1.0])}),
        ):
            with pytest.raises(AttributeError, match="y_original_scale"):
                factory.prior_predictive(original_scale=True)

    def test_residuals_distribution_invalid_quantile(self, mock_mmm_idata_with_prior):
        factory = MMMSummaryFactory(mock_mmm_idata_with_prior)
        with pytest.raises(ValueError, match="Each quantile must be in"):
            factory.residuals_distribution(quantiles=[1.5])

    def test_residuals_distribution_invalid_aggregation(
        self, mock_mmm_idata_with_prior
    ):
        factory = MMMSummaryFactory(mock_mmm_idata_with_prior)
        with pytest.raises(ValueError, match="aggregation not found"):
            factory.residuals_distribution(aggregation=["missing_dim"])

    def test_residuals_distribution_str_aggregation(self, mock_mmm_idata_with_prior):
        factory = MMMSummaryFactory(mock_mmm_idata_with_prior)
        df = factory.residuals_distribution(aggregation="date")
        assert len(df) > 0

    def test_prior_vs_posterior_missing_prior(self, mock_mmm_idata_wrapper):
        factory = MMMSummaryFactory(mock_mmm_idata_wrapper)
        with pytest.raises(ValueError, match="No prior group found"):
            factory.prior_vs_posterior(var_names=["adstock_alpha"])

    def test_prior_vs_posterior_missing_posterior(self, simple_dates, simple_channels):
        rng = np.random.default_rng(5)
        idata = xr.DataTree.from_dict(
            {
                "/prior": xr.Dataset(
                    {
                        "beta": xr.DataArray(
                            rng.normal(size=(2, 5)),
                            dims=("chain", "draw"),
                        )
                    }
                )
            }
        )
        wrapper = MMMIDataWrapper(idata, schema=None, validate_on_init=False)
        factory = MMMSummaryFactory(wrapper)
        with pytest.raises(ValueError, match="No posterior group found"):
            factory.prior_vs_posterior(var_names=["beta"])

    def test_prior_vs_posterior_str_var_name(self, mock_mmm_idata_with_prior):
        factory = MMMSummaryFactory(mock_mmm_idata_with_prior)
        df = factory.prior_vs_posterior(var_names="adstock_alpha", num_points=10)
        assert set(df["variable"].unique()) == {"adstock_alpha"}

    def test_prior_vs_posterior_large_var_names_warning(
        self, mock_mmm_idata_with_prior
    ):
        factory = MMMSummaryFactory(mock_mmm_idata_with_prior)
        many_vars = [f"var_{i}" for i in range(25)]
        prior = {
            name: mock_mmm_idata_with_prior.idata.posterior["adstock_alpha"]
            for name in many_vars
        }
        post = {
            name: mock_mmm_idata_with_prior.idata.posterior["adstock_alpha"]
            for name in many_vars
        }
        mock_mmm_idata_with_prior.idata.prior = xr.Dataset(prior)
        mock_mmm_idata_with_prior.idata.posterior = xr.Dataset(
            {**dict(mock_mmm_idata_with_prior.idata.posterior.data_vars), **post}
        )
        with pytest.warns(UserWarning, match="Summarizing 25 variables"):
            factory.prior_vs_posterior(num_points=5)

    def test_prior_vs_posterior_missing_in_prior(self, mock_mmm_idata_with_prior):
        factory = MMMSummaryFactory(mock_mmm_idata_with_prior)
        with pytest.raises(ValueError, match="not found in prior"):
            factory.prior_vs_posterior(var_names=["missing_var"])

    def test_prior_vs_posterior_missing_in_posterior(self):
        idata = xr.DataTree.from_dict(
            {
                "/prior": xr.Dataset(
                    {
                        "only_prior": xr.DataArray(
                            np.ones((2, 5)),
                            dims=("chain", "draw"),
                        )
                    }
                ),
                "/posterior": xr.Dataset(
                    {
                        "other": xr.DataArray(
                            np.ones((2, 5)),
                            dims=("chain", "draw"),
                        )
                    }
                ),
            }
        )
        factory = MMMSummaryFactory(
            MMMIDataWrapper(idata, schema=None, validate_on_init=False)
        )
        with pytest.raises(ValueError, match="not found in posterior"):
            factory.prior_vs_posterior(var_names=["only_prior"])

    def test_prior_vs_posterior_skips_degenerate_facets(self):
        idata = xr.DataTree.from_dict(
            {
                "/prior": xr.Dataset(
                    {
                        "beta": xr.DataArray(
                            np.ones((1, 1)),
                            dims=("chain", "draw"),
                        )
                    }
                ),
                "/posterior": xr.Dataset(
                    {
                        "beta": xr.DataArray(
                            np.linspace(0, 1, 10).reshape(2, 5),
                            dims=("chain", "draw"),
                        )
                    }
                ),
            }
        )
        factory = MMMSummaryFactory(
            MMMIDataWrapper(idata, schema=None, validate_on_init=False)
        )
        with pytest.warns(UserWarning, match="fewer than 2 finite samples"):
            with pytest.raises(ValueError, match="No prior vs posterior density rows"):
                factory.prior_vs_posterior(var_names=["beta"], num_points=5)

    def test_prior_vs_posterior_skips_kde_failure(self, simple_dates):
        idata = xr.DataTree.from_dict(
            {
                "/prior": xr.Dataset(
                    {
                        "beta": xr.DataArray(
                            np.ones((2, 20)),
                            dims=("chain", "draw"),
                        )
                    }
                ),
                "/posterior": xr.Dataset(
                    {
                        "beta": xr.DataArray(
                            np.ones((2, 20)) * 2,
                            dims=("chain", "draw"),
                        )
                    }
                ),
            }
        )
        factory = MMMSummaryFactory(
            MMMIDataWrapper(idata, schema=None, validate_on_init=False)
        )
        with pytest.warns(UserWarning, match="degenerate samples"):
            with pytest.raises(ValueError, match="No prior vs posterior density rows"):
                factory.prior_vs_posterior(var_names=["beta"], num_points=5)


class TestBudgetSummaryEdgeCases:
    def test_contribution_over_time_missing_channel_dim(
        self, budget_contribution_samples
    ):
        bad = budget_contribution_samples.drop_dims("channel")
        with pytest.raises(ValueError, match="Expected 'channel' dimension"):
            BudgetSummaryFactory.contribution_over_time(bad)

    def test_contribution_over_time_missing_sample_dims(
        self, budget_contribution_samples
    ):
        dates = budget_contribution_samples.coords["date"]
        channels = budget_contribution_samples.coords["channel"]
        bad = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    np.ones((len(dates), len(channels))),
                    dims=("date", "channel"),
                    coords={"date": dates, "channel": channels},
                )
            }
        )
        with pytest.raises(ValueError, match="Expected 'sample' or"):
            BudgetSummaryFactory.contribution_over_time(bad)

    def test_contribution_over_time_missing_variable(self, budget_contribution_samples):
        bad = budget_contribution_samples.drop_vars(
            "channel_contribution_original_scale"
        )
        with pytest.raises(ValueError, match="channel_contribution_original_scale"):
            BudgetSummaryFactory.contribution_over_time(bad)


class TestCVSummaryEdgeCases:
    @staticmethod
    def _pp_without_y(cv_results_idata):
        pp = cv_results_idata["/posterior_predictive"].to_dataset()
        return pp.drop_vars("y_original_scale")

    def test_predictions_missing_y_original_scale(self, cv_results_idata):
        bad = xr.DataTree.from_dict(
            {
                "/posterior": cv_results_idata["/posterior"].to_dataset(),
                "/posterior_predictive": self._pp_without_y(cv_results_idata),
                "/cv_metadata": cv_results_idata["/cv_metadata"].to_dataset(),
            }
        )
        factory = MMMCVSummaryFactory(bad)
        with pytest.raises(ValueError, match="y_original_scale"):
            factory.predictions()

    def test_param_stability_missing_posterior(self, cv_results_idata):
        bad = xr.DataTree.from_dict(
            {
                "/posterior_predictive": cv_results_idata[
                    "/posterior_predictive"
                ].to_dataset(),
                "/cv_metadata": cv_results_idata["/cv_metadata"].to_dataset(),
            }
        )
        factory = MMMCVSummaryFactory(bad)
        with pytest.raises(ValueError, match="no 'posterior' group"):
            factory.param_stability()

    def test_param_stability_missing_cv_dim(self, cv_results_idata):
        posterior = xr.Dataset(
            {
                "beta_channel": xr.DataArray(
                    np.ones((2, 10, 2)),
                    dims=("chain", "draw", "channel"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "channel": ["tv", "radio"],
                    },
                )
            }
        )
        bad = xr.DataTree.from_dict(
            {
                "/posterior": posterior,
                "/posterior_predictive": cv_results_idata[
                    "/posterior_predictive"
                ].to_dataset(),
                "/cv_metadata": cv_results_idata["/cv_metadata"].to_dataset(),
            }
        )
        factory = MMMCVSummaryFactory(bad)
        with pytest.raises(ValueError, match="No 'cv' coordinate"):
            factory.param_stability()

    def test_predictions_with_dims_filter(self, cv_results_idata):
        factory = MMMCVSummaryFactory(cv_results_idata)
        date = cv_results_idata["/posterior_predictive"].coords["date"].values[0]
        df = factory.predictions(hdi_probs=[0.94], dims={"date": [date]})
        assert len(df) > 0

    def test_param_stability_with_dims_filter(self, cv_results_idata):
        factory = MMMCVSummaryFactory(cv_results_idata)
        df = factory.param_stability(
            var_names=["beta_channel"],
            dims={"channel": ["tv"]},
        )
        assert set(df["channel"].unique()) == {"tv"}

    def test_param_stability_empty_when_no_matching_vars(self, cv_results_idata):
        factory = MMMCVSummaryFactory(cv_results_idata)
        df = factory.param_stability(var_names=["nonexistent"])
        assert len(df) == 0

    def test_crps_missing_posterior_predictive(self, cv_results_idata):
        bad = xr.DataTree.from_dict(
            {
                "/posterior": cv_results_idata["/posterior"].to_dataset(),
                "/posterior_predictive": self._pp_without_y(cv_results_idata),
                "/cv_metadata": cv_results_idata["/cv_metadata"].to_dataset(),
            }
        )
        factory = MMMCVSummaryFactory(bad)
        with pytest.raises(ValueError, match="y_original_scale"):
            factory.crps()

    def test_pred_matrix_for_rows_scalar_selection(self, cv_results_idata):
        rows = pd.DataFrame(
            {
                "date": [
                    cv_results_idata["/posterior_predictive"].coords["date"].values[0]
                ]
            }
        )
        mat = _pred_matrix_for_rows(cv_results_idata, "fold_0", rows)
        assert mat.ndim == 2

    def test_crps_for_split_returns_nan_on_failure(self, cv_results_idata):
        with pytest.warns(UserWarning, match="CRPS computation failed"):
            result = _crps_for_split(
                cv_results_idata,
                "fold_0",
                pd.DataFrame({"date": ["1900-01-01"]}),
                pd.Series([1.0]),
                {},
            )
        assert np.isnan(result)


class TestTimeSliceCVSummaryProperty:
    def test_summary_raises_without_cv_idata(self):
        cv = TimeSliceCrossValidator.__new__(TimeSliceCrossValidator)
        fold_mock = MagicMock()
        fold_mock.idata = None
        cv._cv_results = [fold_mock]
        with pytest.raises(ValueError, match="cv_idata is not available"):
            cv.summary
