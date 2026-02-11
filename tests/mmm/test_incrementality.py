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
"""Tests for Incrementality module - counterfactual analysis with carryover."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pymc.model.fgraph import clone_model as cm

from pymc_marketing.mmm.incrementality import Incrementality
from pymc_marketing.mmm.utils import _convert_frequency_to_timedelta


class TestIncrementalityBasics:
    """Test basic functionality and API surface."""

    def test_incrementality_property_exists(self, simple_fitted_mmm):
        """Test that .incrementality property exists on fitted MMM."""
        assert hasattr(simple_fitted_mmm, "incrementality")
        incr = simple_fitted_mmm.incrementality
        assert isinstance(incr, Incrementality)

    def test_incrementality_has_required_methods(self, simple_fitted_mmm):
        """Test that Incrementality has all required methods."""
        incr = simple_fitted_mmm.incrementality
        assert hasattr(incr, "compute_incremental_contribution")


class TestComputeIncrementalContribution:
    """Test core compute_incremental_contribution function."""

    def test_output_shape_with_frequency_original(self, simple_fitted_mmm):
        """Test output has correct shape for original frequency."""
        incr = simple_fitted_mmm.incrementality
        result = incr.compute_incremental_contribution(frequency="original")

        n_samples = (
            simple_fitted_mmm.idata.posterior.dims["chain"]
            * simple_fitted_mmm.idata.posterior.dims["draw"]
        )
        n_dates = len(simple_fitted_mmm.idata.fit_data.date)
        n_channels = len(simple_fitted_mmm.channel_columns)

        assert result.dims == ("sample", "date", "channel")
        assert result.sizes["sample"] == n_samples
        assert result.sizes["date"] == n_dates
        assert result.sizes["channel"] == n_channels

    def test_output_shape_with_frequency_monthly(self, simple_fitted_mmm):
        """Test output aggregates correctly for monthly frequency."""
        incr = simple_fitted_mmm.incrementality
        result = incr.compute_incremental_contribution(frequency="monthly")

        assert "date" in result.dims
        assert result.sizes["date"] > 0
        dates = pd.to_datetime(result.date.values)
        # Check that dates are month-end dates
        for d in dates:
            period = pd.Period(d, freq="M")
            assert d == period.to_timestamp(how="end")

    def test_output_shape_with_frequency_all_time(self, simple_fitted_mmm):
        """Test output has no date dimension for all_time frequency."""
        incr = simple_fitted_mmm.incrementality
        result = incr.compute_incremental_contribution(frequency="all_time")

        assert "date" not in result.dims
        assert result.dims == ("sample", "channel")

    def test_hierarchical_dimensions_preserved(self, panel_fitted_mmm):
        """Test that custom dimensions (geo, country) are preserved in output."""
        incr = panel_fitted_mmm.incrementality
        result = incr.compute_incremental_contribution(frequency="monthly")

        assert "country" in result.dims
        assert result.sizes["country"] == len(panel_fitted_mmm.model.coords["country"])

    def test_counterfactual_factor_zero_gives_total_incrementality(
        self, simple_fitted_mmm
    ):
        """Test that factor=0.0 computes total incrementality (zero-out)."""
        incr = simple_fitted_mmm.incrementality
        result = incr.compute_incremental_contribution(
            frequency="all_time",
            counterfactual_spend_factor=0.0,
        )

        mean_incremental = result.mean(dim="sample")
        assert (mean_incremental > 0).any()

    def test_counterfactual_factor_1_01_gives_marginal_incrementality(
        self, simple_fitted_mmm
    ):
        """Test that factor=1.01 computes marginal incrementality."""
        incr = simple_fitted_mmm.incrementality
        result_marginal = incr.compute_incremental_contribution(
            frequency="all_time",
            counterfactual_spend_factor=1.01,
        )
        result_total = incr.compute_incremental_contribution(
            frequency="all_time",
            counterfactual_spend_factor=0.0,
        )

        mean_marginal = result_marginal.mean(dim="sample")
        mean_total = result_total.mean(dim="sample")
        assert (mean_marginal < mean_total).any()

    def test_negative_counterfactual_factor_raises_error(self, simple_fitted_mmm):
        """Test that negative counterfactual factor raises ValueError."""
        incr = simple_fitted_mmm.incrementality
        with pytest.raises(
            ValueError, match="counterfactual_spend_factor must be >= 0"
        ):
            incr.compute_incremental_contribution(
                frequency="all_time",
                counterfactual_spend_factor=-0.5,
            )

    def test_include_carryover_affects_results(self, simple_fitted_mmm):
        """Test that include_carryover flag changes results."""
        incr = simple_fitted_mmm.incrementality
        result_with = incr.compute_incremental_contribution(
            frequency="all_time",
            include_carryover=True,
        )
        result_without = incr.compute_incremental_contribution(
            frequency="all_time",
            include_carryover=False,
        )
        assert not xr.DataArray.equals(result_with, result_without)

    def test_period_start_end_filters_dates(self, simple_fitted_mmm):
        """Test that period_start and period_end filter date range."""
        incr = simple_fitted_mmm.incrementality
        all_dates = pd.to_datetime(simple_fitted_mmm.idata.fit_data.date.values)
        mid_date = all_dates[len(all_dates) // 2]

        result = incr.compute_incremental_contribution(
            frequency="original",
            period_start=all_dates[0],
            period_end=mid_date,
        )

        result_dates = pd.to_datetime(result.date.values)
        assert result_dates[-1] <= mid_date
        assert len(result_dates) < len(all_dates)

    def test_num_samples_subsamples_posterior(self, simple_fitted_mmm):
        """Test that num_samples reduces sample dimension."""
        incr = simple_fitted_mmm.incrementality

        result_full = incr.compute_incremental_contribution(frequency="all_time")
        result_sub = incr.compute_incremental_contribution(
            frequency="all_time",
            num_samples=10,
            random_state=42,
        )

        assert result_sub.sizes["sample"] == 10
        assert result_sub.sizes["sample"] < result_full.sizes["sample"]

    def test_random_state_makes_subsampling_reproducible(self, simple_fitted_mmm):
        """Test that random_state ensures reproducible subsampling."""
        incr = simple_fitted_mmm.incrementality

        result1 = incr.compute_incremental_contribution(
            frequency="all_time",
            num_samples=10,
            random_state=42,
        )
        result2 = incr.compute_incremental_contribution(
            frequency="all_time",
            num_samples=10,
            random_state=42,
        )

        xr.testing.assert_allclose(result1, result2)

    def test_original_scale_flag_affects_magnitude(self, simple_fitted_mmm):
        """Test that original_scale flag changes result magnitude."""
        incr = simple_fitted_mmm.incrementality

        result_original = incr.compute_incremental_contribution(
            frequency="all_time",
            original_scale=True,
        )
        result_scaled = incr.compute_incremental_contribution(
            frequency="all_time",
            original_scale=False,
        )

        target_scale = simple_fitted_mmm.data.get_target_scale()
        if target_scale != 1.0:
            assert not xr.DataArray.equals(result_original, result_scaled)


class TestHelperMethods:
    """Test helper methods used by compute_incremental_contribution."""

    def test_create_period_groups_all_time(self, simple_fitted_mmm):
        """Test period grouping for all_time frequency."""
        incr = simple_fitted_mmm.incrementality
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")

        periods = incr._create_period_groups(start, end, "all_time")

        assert len(periods) == 1
        assert periods[0] == (start, end)

    def test_create_period_groups_monthly(self, simple_fitted_mmm):
        """Test period grouping for monthly frequency."""
        incr = simple_fitted_mmm.incrementality
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-03-31")

        periods = incr._create_period_groups(start, end, "monthly")

        assert len(periods) == 3  # Jan, Feb, Mar
        assert periods[0][1] == pd.Timestamp("2024-01-31")
        assert periods[1][1] == pd.Timestamp("2024-02-29")  # 2024 is leap year
        assert periods[2][1] == pd.Timestamp("2024-03-31")

    def test_subsample_posterior_draws_returns_all_when_none(self, simple_fitted_mmm):
        """Test subsampling returns all draws when num_samples=None."""
        incr = simple_fitted_mmm.incrementality
        draw_indices = incr._subsample_posterior_draws(
            num_samples=None, random_state=None
        )

        total_draws = simple_fitted_mmm.idata.posterior.dims["draw"]
        assert len(draw_indices) == total_draws or draw_indices is None

    def test_aggregate_spend_matches_data_wrapper(self, simple_fitted_mmm):
        """Test that _aggregate_spend delegates correctly to data wrapper."""
        incr = simple_fitted_mmm.incrementality

        spend_incr = incr._aggregate_spend(frequency="monthly")

        data_monthly = simple_fitted_mmm.data.aggregate_time(
            period="monthly", method="sum"
        )
        spend_data = data_monthly.get_channel_spend()

        xr.testing.assert_allclose(spend_incr, spend_data)


# ==================== Ground Truth Testing ====================


def evaluate_channel_contribution(mmm, channel_data_values, original_scale=False):
    """Evaluate channel_contribution for given channel_data using sample_posterior_predictive.

    Uses the standard PyMC evaluation path (completely independent from
    extract_response_distribution + vectorize_graph) as an oracle.
    """
    var_name = (
        "channel_contribution_original_scale"
        if original_scale
        else "channel_contribution"
    )
    model = cm(mmm.model)
    with model:
        pm.set_data({"channel_data": channel_data_values})
        result = pm.sample_posterior_predictive(
            mmm.idata,
            var_names=[var_name],
        )
    return result.posterior_predictive[var_name]


def compute_ground_truth_incremental(
    mmm,
    counterfactual_spend_factor=0.0,
    target_channel_idx=None,
    target_period_mask=None,
):
    """Compute ground truth incremental contribution using manual counterfactual.

    Incremental = baseline_contribution - counterfactual_contribution
    """
    actual_data = mmm.model["channel_data"].get_value()

    cf_data = actual_data.copy()
    if target_period_mask is None:
        target_period_mask = np.ones(actual_data.shape[0], dtype=bool)

    if target_channel_idx is not None:
        cf_data[target_period_mask, target_channel_idx] = (
            actual_data[target_period_mask, target_channel_idx]
            * counterfactual_spend_factor
        )
    else:
        cf_data[target_period_mask] = (
            actual_data[target_period_mask] * counterfactual_spend_factor
        )

    baseline_contrib = evaluate_channel_contribution(mmm, actual_data)
    cf_contrib = evaluate_channel_contribution(mmm, cf_data)

    return baseline_contrib - cf_contrib


def compute_ground_truth_incremental_by_period(
    mmm,
    frequency="all_time",
    counterfactual_spend_factor=0.0,
    include_carryover=True,
    original_scale=True,
):
    """Compute ground truth incremental contribution per period using the oracle.

    For each period defined by *frequency*, creates a **separate** counterfactual
    where only that period's spend is modified (all other periods keep actual
    spend), evaluates using ``sample_posterior_predictive`` (the oracle), and
    sums the difference over the appropriate evaluation window.

    This mirrors the logic of ``compute_incremental_contribution()`` which
    processes each period independently with its own counterfactual, and serves
    as a reference implementation that is completely independent of the
    vectorized graph path.

    Parameters
    ----------
    mmm : MMM
        Fitted MMM model.
    frequency : str
        One of ``"original"``, ``"monthly"``, ``"all_time"``, etc.
    counterfactual_spend_factor : float
        Factor applied to the target period's spend (``0.0`` = zero-out).
    include_carryover : bool
        Whether to include adstock carryover effects (both carry-in and
        carry-out).
    original_scale : bool, default=True
        Return contributions in original scale of target variable.

    Returns
    -------
    xr.DataArray
        Ground truth incremental contribution with dimensions matching
        ``compute_incremental_contribution`` output.
    """
    actual_data = mmm.model["channel_data"].get_value()
    dates = pd.to_datetime(mmm.idata.fit_data.date.values)

    incr = mmm.incrementality
    periods = incr._create_period_groups(dates[0], dates[-1], frequency)
    l_max = mmm.adstock.l_max
    inferred_freq = pd.infer_freq(dates)

    # Evaluate baseline once (reused for all periods)
    baseline_contrib = evaluate_channel_contribution(
        mmm, actual_data, original_scale=original_scale
    )

    period_results = []
    for t0, t1 in periods:
        # Create counterfactual: only modify spend in [t0, t1]
        target_mask = (dates >= t0) & (dates <= t1)
        cf_data = actual_data.copy()
        cf_data[target_mask] = actual_data[target_mask] * counterfactual_spend_factor

        cf_contrib = evaluate_channel_contribution(
            mmm, cf_data, original_scale=original_scale
        )

        # Sign convention
        if counterfactual_spend_factor > 1.0:
            diff = cf_contrib - baseline_contrib
        else:
            diff = baseline_contrib - cf_contrib

        # Determine evaluation window for summing
        if include_carryover:
            carryout_end = t1 + _convert_frequency_to_timedelta(l_max, inferred_freq)
            eval_mask = (dates >= t0) & (dates <= carryout_end)
        else:
            eval_mask = (dates >= t0) & (dates <= t1)

        # Sum over evaluation window
        period_incr = diff.sel(date=dates[eval_mask]).sum(dim="date")
        # Shape: (chain, draw, channel, *custom_dims)

        # Stack chain x draw into sample, assign period label
        stacked_raw = period_incr.stack(sample=("chain", "draw"))
        # Replace MultiIndex with plain integer coords to match implementation
        stacked = (
            stacked_raw.reset_index("sample", drop=True)
            .assign_coords(
                sample=np.arange(stacked_raw.sizes["sample"]),
                date=t1,
            )
            .expand_dims("date")
        )
        period_results.append(stacked)

    # Concatenate and format
    if frequency == "all_time":
        result = period_results[0].squeeze("date", drop=True)
    else:
        result = xr.concat(period_results, dim="date")

    # Standard dimension order
    core_dims = ["sample", "channel"]
    extra_dims = [d for d in result.dims if d not in [*core_dims, "date"]]

    if frequency == "all_time":
        dim_order = ["sample", "channel", *extra_dims]
    else:
        dim_order = ["sample", "date", "channel", *extra_dims]

    return result.transpose(*dim_order)


class TestGroundTruthValidation:
    """Validate vectorized incrementality against sample_posterior_predictive oracle."""

    @pytest.mark.parametrize(
        "frequency, include_carryover, original_scale",
        [
            ("original", True, True),
            ("monthly", True, True),
            ("all_time", True, True),
            ("monthly", False, True),
            ("monthly", True, False),
        ],
    )
    def test_factor_0_matches_ground_truth(
        self, simple_fitted_mmm, frequency, include_carryover, original_scale
    ):
        """Validate factor=0 incrementality against ground truth.

        For each period defined by *frequency*, the ground truth creates a
        **separate** counterfactual where only that period's channel spend is
        zeroed out (all other periods retain actual spend).  This mirrors
        ``compute_incremental_contribution()`` which processes periods
        independently.
        """
        gt = compute_ground_truth_incremental_by_period(
            simple_fitted_mmm,
            frequency=frequency,
            counterfactual_spend_factor=0.0,
            include_carryover=include_carryover,
            original_scale=original_scale,
        )

        result = simple_fitted_mmm.incrementality.compute_incremental_contribution(
            frequency=frequency,
            counterfactual_spend_factor=0.0,
            include_carryover=include_carryover,
            original_scale=original_scale,
        )

        xr.testing.assert_allclose(result, gt, rtol=1e-4)

    def test_marginal_factor_ground_truth(self, simple_fitted_mmm):
        """Validate marginal incrementality (factor=1.01) against ground truth."""
        factor = 1.01
        actual_data = simple_fitted_mmm.model["channel_data"].get_value()

        baseline_contrib = evaluate_channel_contribution(simple_fitted_mmm, actual_data)
        perturbed_contrib = evaluate_channel_contribution(
            simple_fitted_mmm, actual_data * factor
        )

        ground_truth_marginal = (
            (perturbed_contrib - baseline_contrib)
            .sum(dim="date")
            .stack(sample=("chain", "draw"))
            .transpose("sample", "channel")
        )

        result = simple_fitted_mmm.incrementality.compute_incremental_contribution(
            frequency="all_time", counterfactual_spend_factor=factor
        )

        xr.testing.assert_allclose(result, ground_truth_marginal, rtol=1e-4)

    def test_panel_model_ground_truth(self, panel_fitted_mmm):
        """Validate ground truth for panel model with country dimension."""
        ground_truth = compute_ground_truth_incremental(
            panel_fitted_mmm, counterfactual_spend_factor=0.0
        )
        ground_truth_all_time = (
            ground_truth.sum(dim="date")
            .stack(sample=("chain", "draw"))
            .transpose("sample", "channel", "country")
        )

        result = panel_fitted_mmm.incrementality.compute_incremental_contribution(
            frequency="all_time", counterfactual_spend_factor=0.0
        )

        assert "country" in result.dims
        xr.testing.assert_allclose(result, ground_truth_all_time, rtol=1e-4)
