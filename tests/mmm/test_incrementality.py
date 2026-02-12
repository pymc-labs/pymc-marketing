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

from pymc_marketing.mmm.utils import _convert_frequency_to_timedelta


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
        pm.set_data(
            {
                "channel_data": channel_data_values.astype(
                    mmm.model["channel_data"].type.dtype
                )
            }
        )
        result = pm.sample_posterior_predictive(
            mmm.idata,
            var_names=[var_name],
        )
    return result.posterior_predictive[var_name]


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


class TestIncrementality:
    """Tests for compute_incremental_contribution and supporting methods."""

    @pytest.mark.parametrize(
        "model_fixture, frequency, include_carryover, original_scale, counterfactual_spend_factor",
        [
            ("simple_fitted_mmm", "original", True, True, 0.0),
            ("simple_fitted_mmm", "monthly", True, True, 0.0),
            ("simple_fitted_mmm", "all_time", True, True, 0.0),
            ("simple_fitted_mmm", "monthly", False, True, 0.0),
            ("simple_fitted_mmm", "monthly", True, False, 0.0),
            ("simple_fitted_mmm", "monthly", True, True, 1.01),
            ("panel_fitted_mmm", "monthly", True, True, 0.0),
            ("panel_fitted_mmm", "all_time", True, True, 1.01),
        ],
    )
    def test_compute_incremental_contribution_matches_ground_truth(
        self,
        request,
        model_fixture,
        frequency,
        include_carryover,
        original_scale,
        counterfactual_spend_factor,
    ):
        """Validate incrementality against ground truth.

        For each period defined by *frequency*, the ground truth creates a
        **separate** counterfactual where only that period's spend is modified
        (all other periods retain actual spend).  This mirrors
        ``compute_incremental_contribution()`` which processes periods
        independently.
        """
        mmm = request.getfixturevalue(model_fixture)
        gt = compute_ground_truth_incremental_by_period(
            mmm,
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            include_carryover=include_carryover,
            original_scale=original_scale,
        )

        result = mmm.incrementality.compute_incremental_contribution(
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            include_carryover=include_carryover,
            original_scale=original_scale,
        )

        xr.testing.assert_allclose(result, gt, rtol=1e-4)

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

        result_sub = incr.compute_incremental_contribution(
            frequency="all_time",
            num_samples=10,
            random_state=42,
        )

        assert result_sub.sizes["sample"] == 10

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

    def test_aggregate_spend_matches_data_wrapper(self, simple_fitted_mmm):
        """Test that _aggregate_spend delegates correctly to data wrapper."""
        incr = simple_fitted_mmm.incrementality

        spend_incr = incr._aggregate_spend(frequency="monthly")

        data_monthly = simple_fitted_mmm.data.aggregate_time(
            period="monthly", method="sum"
        )
        spend_data = data_monthly.get_channel_spend()

        xr.testing.assert_allclose(spend_incr, spend_data)


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    @pytest.mark.parametrize("frequency", ["all_time", "monthly"])
    def test_contribution_over_spend_ground_truth(self, simple_fitted_mmm, frequency):
        """Test that contribution_over_spend matches ground truth."""
        incr = simple_fitted_mmm.incrementality
        roas = incr.contribution_over_spend(frequency=frequency)

        ground_truth = incr.compute_incremental_contribution(
            frequency=frequency, counterfactual_spend_factor=0.0
        )
        spend = simple_fitted_mmm.data.aggregate_time(
            period=frequency, method="sum"
        ).get_channel_spend()
        roas_ground_truth = ground_truth / spend
        xr.testing.assert_allclose(roas, roas_ground_truth)

    @pytest.mark.parametrize("frequency", ["all_time", "monthly"])
    def test_marginal_contribution_over_spend_ground_truth(
        self, simple_fitted_mmm, frequency
    ):
        """Test that marginal_contribution_over_spend matches ground truth."""
        incr = simple_fitted_mmm.incrementality
        spend_increase_pct = 0.01

        mroas = incr.marginal_contribution_over_spend(
            frequency=frequency, spend_increase_pct=spend_increase_pct
        )

        ground_truth = incr.compute_incremental_contribution(
            frequency=frequency, counterfactual_spend_factor=1.0 + spend_increase_pct
        )
        spend = simple_fitted_mmm.data.aggregate_time(
            period=frequency, method="sum"
        ).get_channel_spend()
        incremental_spend = spend_increase_pct * spend
        incremental_spend_safe = xr.where(
            incremental_spend == 0, np.nan, incremental_spend
        )
        mroas_ground_truth = ground_truth / incremental_spend_safe
        xr.testing.assert_allclose(mroas, mroas_ground_truth)

    def test_contribution_over_spend_handles_zero_spend(self, simple_fitted_mmm):
        """Test that zero spend results in NaN ROAS."""
        from unittest.mock import patch

        incr = simple_fitted_mmm.incrementality

        # Get actual spend and inject a zero entry
        spend = incr._aggregate_spend("original")
        zero_spend = spend.copy(deep=True)
        zero_date = zero_spend.date.values[0]
        zero_channel = zero_spend.channel.values[0]
        zero_spend.loc[dict(date=zero_date, channel=zero_channel)] = 0.0

        with patch.object(incr, "_aggregate_spend", return_value=zero_spend):
            roas = incr.contribution_over_spend(frequency="original")

        # Where spend is zero, ROAS should be NaN
        zero_roas = roas.sel(date=zero_date, channel=zero_channel)
        assert np.isnan(zero_roas).all(), "Expected NaN ROAS where spend is zero"

        # Where spend is non-zero, ROAS should be finite
        nonzero_roas = roas.sel(channel=zero_spend.channel.values[1])
        assert np.isfinite(nonzero_roas).all(), (
            "Expected finite ROAS where spend is non-zero"
        )

    def test_spend_over_contribution_is_reciprocal(self, simple_fitted_mmm):
        """Test that CAC is reciprocal of ROAS."""
        incr = simple_fitted_mmm.incrementality
        roas = incr.contribution_over_spend(frequency="all_time")
        cac = incr.spend_over_contribution(frequency="all_time")

        expected_cac = 1.0 / roas
        xr.testing.assert_allclose(cac, expected_cac)

    def test_marginal_less_than_total_for_saturation(self, simple_fitted_mmm):
        """Test that marginal ROAS < total ROAS for saturating response."""
        incr = simple_fitted_mmm.incrementality

        total_roas = incr.contribution_over_spend(frequency="all_time")
        marginal_roas = incr.marginal_contribution_over_spend(
            frequency="all_time",
            spend_increase_pct=0.01,
        )

        mean_total = total_roas.mean(dim="sample")
        mean_marginal = marginal_roas.mean(dim="sample")

        # For saturating curves, marginal should be less than total
        assert (mean_marginal < mean_total).any()

    @pytest.mark.parametrize(
        "method_name",
        [
            "contribution_over_spend",
            "spend_over_contribution",
            "marginal_contribution_over_spend",
        ],
    )
    def test_convenience_functions_support_period_filtering(
        self, simple_fitted_mmm, method_name
    ):
        """Test that convenience functions accept period_start/period_end."""
        incr = simple_fitted_mmm.incrementality
        all_dates = pd.to_datetime(simple_fitted_mmm.idata.fit_data.date.values)
        mid_date = all_dates[len(all_dates) // 2]

        method = getattr(incr, method_name)
        result = method(
            frequency="original",
            period_start=all_dates[0],
            period_end=mid_date,
        )

        assert len(result.date) < len(all_dates)

    @pytest.mark.parametrize(
        "method_name",
        [
            "contribution_over_spend",
            "spend_over_contribution",
            "marginal_contribution_over_spend",
        ],
    )
    def test_convenience_functions_support_num_samples(
        self, simple_fitted_mmm, method_name
    ):
        """Test that num_samples subsamples the posterior."""
        incr = simple_fitted_mmm.incrementality

        method = getattr(incr, method_name)
        result = method(
            frequency="all_time",
            num_samples=10,
            random_state=42,
        )
        assert result.sizes["sample"] == 10
