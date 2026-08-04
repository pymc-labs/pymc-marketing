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

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pydantic import ValidationError

from pymc_marketing.mmm.incrementality import (
    IdentityLinkReducer,
    Incrementality,
    IncrementalReducer,
    LogLinkReducer,
)
from pymc_marketing.model_graph import deterministics_to_flat


def evaluate_under_spend(mmm, channel_data_values, var_name):
    """Evaluate ``var_name`` for given channel_data using sample_posterior_predictive.

    Uses the standard PyMC evaluation path (completely independent from
    extract_response_distribution + vectorize_graph) as an oracle.
    """
    names = mmm.frozen_deterministics
    if names:
        model = deterministics_to_flat(mmm.model, names=names)
    else:
        model = mmm.model.copy()
    with model:
        pm.set_data(
            {
                "channel_data": channel_data_values.astype(
                    model["channel_data"].type.dtype
                )
            }
        )
        result = pm.sample_posterior_predictive(
            mmm.idata,
            var_names=[var_name],
        )
    return result.posterior_predictive[var_name]


def evaluate_channel_contribution(mmm, channel_data_values, original_scale=False):
    """Evaluate channel_contribution for given channel_data (identity-link oracle)."""
    var_name = (
        "channel_contribution_original_scale"
        if original_scale
        else "channel_contribution"
    )
    return evaluate_under_spend(mmm, channel_data_values, var_name)


def compute_ground_truth_incremental_by_period(
    mmm,
    frequency="all_time",
    counterfactual_spend_factor=0.0,
    include_carryover=True,
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

    Returns
    -------
    xr.DataArray
        Ground truth incremental contribution with dimensions matching
        ``compute_incremental_contribution`` output:
        ``(chain, draw, date, channel, *custom_dims)`` or
        ``(chain, draw, channel, *custom_dims)`` for ``"all_time"``.
    """
    actual_data = mmm.model["channel_data"].get_value()
    dates = pd.to_datetime(mmm.idata.fit_data.date.values)

    # pm.set_data cannot accept float when the model's channel_data is integer.
    # Fractional factors (e.g. 1.01) produce float values that would truncate.
    if counterfactual_spend_factor not in (0.0, 1.0) and np.issubdtype(
        actual_data.dtype, np.integer
    ):
        raise ValueError(
            f"counterfactual_spend_factor={counterfactual_spend_factor} produces "
            "fractional values, but the model's channel_data has integer dtype. "
            "pm.set_data rejects float for integer shared variables. Use a model "
            "fit with float channel_data (e.g. simple_fitted_mmm) for "
            "marginal incrementality ground truth."
        )

    incr = mmm.incrementality
    periods = incr._create_period_groups(dates[0], dates[-1], frequency)
    l_max = mmm.adstock.l_max
    inferred_freq = pd.infer_freq(dates)

    # Evaluate baseline once (reused for all periods), always in original scale
    baseline_contrib = evaluate_channel_contribution(
        mmm, actual_data, original_scale=True
    )

    period_results = []
    for t0, t1 in periods:
        # Create counterfactual: only modify spend in [t0, t1]
        target_mask = (dates >= t0) & (dates <= t1)
        cf_data = actual_data.copy()
        cf_data[target_mask] = actual_data[target_mask] * counterfactual_spend_factor

        cf_contrib = evaluate_channel_contribution(mmm, cf_data, original_scale=True)

        # Sign convention
        if counterfactual_spend_factor > 1.0:
            diff = cf_contrib - baseline_contrib
        else:
            diff = baseline_contrib - cf_contrib

        # Determine evaluation window for summing
        if include_carryover:
            carryout_end = t1 + l_max * pd.tseries.frequencies.to_offset(inferred_freq)
            eval_mask = (dates >= t0) & (dates <= carryout_end)
        else:
            eval_mask = (dates >= t0) & (dates <= t1)

        # Sum over evaluation window
        period_incr = diff.sel(date=dates[eval_mask]).sum(dim="date")
        # Shape: (chain, draw, channel, *custom_dims)

        # Assign period label and expand date dim
        period_incr = period_incr.assign_coords(date=t1).expand_dims("date")
        period_results.append(period_incr)

    return _format_ground_truth(period_results, frequency)


def _format_ground_truth(period_results, frequency):
    """Concatenate per-period ground truth and match the module's dimension order."""
    if frequency == "all_time":
        result = period_results[0].squeeze("date", drop=True)
    else:
        result = xr.concat(period_results, dim="date")

    # Standard dimension order: (chain, draw, [date,] channel, *custom_dims)
    core_dims = ["chain", "draw", "channel"]
    extra_dims = [d for d in result.dims if d not in [*core_dims, "date"]]

    if frequency == "all_time":
        dim_order = ["chain", "draw", "channel", *extra_dims]
    else:
        dim_order = ["chain", "draw", "date", "channel", *extra_dims]

    return result.transpose(*dim_order)


def compute_log_link_ground_truth_by_period(
    mmm,
    frequency="all_time",
    counterfactual_spend_factor=0.0,
    include_carryover=True,
):
    """Ground truth incremental contribution for a multiplicative (log-link) model.

    Deliberately brute force, and deliberately **per channel**: under a log
    link the response is multiplicative, so channel *m*'s increment cannot be
    read off a single all-channels counterfactual the way it can under an
    identity link.  Each ``(period, channel)`` pair therefore gets its own
    counterfactual, evaluated on the model's response-scale prediction
    ``{output_var}_original_scale`` through ``sample_posterior_predictive``.

    Shares no code with the incrementality module -- in particular it never
    forms a linear-predictor difference -- so it is an independent check of
    both the algebra and the carryover window.

    Parameters
    ----------
    mmm : MMM
        Fitted MMM built with ``link="log"``.
    frequency : str
        One of ``"original"``, ``"monthly"``, ``"all_time"``, etc.
    counterfactual_spend_factor : float
        Factor applied to the target period's spend (``0.0`` = zero-out).
    include_carryover : bool
        Whether to include adstock carryover effects.

    Returns
    -------
    xr.DataArray
        Ground truth with dimensions matching
        ``compute_incremental_contribution`` output.
    """
    actual_data = mmm.model["channel_data"].get_value()
    dates = pd.to_datetime(mmm.idata.fit_data.date.values)
    channels = list(mmm.channel_columns)
    response_var = f"{mmm.output_var}_original_scale"

    # Do not assume channel sits on axis 1: panel models lay channel_data out as
    # (date, *custom_dims, channel), so the axis has to be looked up.
    channel_axis = list(mmm.data.get_channel_data().dims).index("channel")

    periods = mmm.incrementality._create_period_groups(dates[0], dates[-1], frequency)
    l_max = mmm.adstock.l_max
    freq_offset = pd.tseries.frequencies.to_offset(pd.infer_freq(dates))

    baseline = evaluate_under_spend(mmm, actual_data, response_var)

    period_results = []
    for t0, t1 in periods:
        target_mask = (dates >= t0) & (dates <= t1)
        if include_carryover:
            eval_mask = (dates >= t0) & (dates <= t1 + l_max * freq_offset)
        else:
            eval_mask = (dates >= t0) & (dates <= t1)

        channel_results = []
        for i, channel in enumerate(channels):
            # Perturb ONE channel; every other channel keeps actual spend.
            selector = [slice(None)] * actual_data.ndim
            selector[0] = target_mask
            selector[channel_axis] = i
            selector = tuple(selector)

            cf_data = actual_data.copy()
            cf_data[selector] = actual_data[selector] * counterfactual_spend_factor
            cf = evaluate_under_spend(mmm, cf_data, response_var)

            # Sign convention matches compute_incremental_contribution.
            if counterfactual_spend_factor > 1.0:
                diff = cf - baseline
            else:
                diff = baseline - cf

            channel_results.append(
                diff.sel(date=dates[eval_mask])
                .sum(dim="date")
                .assign_coords(channel=channel)
            )

        period_incr = xr.concat(channel_results, dim="channel")
        period_results.append(period_incr.assign_coords(date=t1).expand_dims("date"))

    return _format_ground_truth(period_results, frequency)


@pytest.fixture
def incrementality_lite():
    """Lightweight Incrementality stub for testing helpers without model fitting.

    Provides an ``Incrementality`` instance whose ``data`` and ``idata``
    attributes are thin stubs — just enough for ``_validate_input``,
    ``_create_period_groups``, and the static helper methods.  Avoids the
    expensive model-build + prior-sampling that ``simple_fitted_mmm`` requires.

    Returns ``(incr, l_max)`` tuple.
    """
    dates = pd.date_range("2023-01-01", periods=14, freq="W")
    spend_values = (
        np.random.default_rng(42).integers(100, 500, size=(14, 2)).astype(float)
    )
    spend = xr.DataArray(
        spend_values,
        dims=("date", "channel"),
        coords={"date": dates, "channel": ["ch_A", "ch_B"]},
    )

    incr = object.__new__(Incrementality)
    incr.data = SimpleNamespace(
        dates=dates,
        get_channel_spend=lambda: spend,
    )
    incr.idata = SimpleNamespace(
        fit_data=SimpleNamespace(
            date=SimpleNamespace(values=dates.values),
        ),
    )

    return incr, 4


class TestIncrementality:
    """Tests for compute_incremental_contribution and supporting methods."""

    @pytest.mark.parametrize(
        "model_fixture, frequency, include_carryover, counterfactual_spend_factor",
        [
            ("simple_fitted_mmm", "original", True, 0.0),
            ("simple_fitted_mmm", "monthly", True, 0.0),
            ("simple_fitted_mmm", "all_time", True, 0.0),
            ("simple_fitted_mmm", "monthly", False, 0.0),
            ("simple_fitted_mmm", "monthly", True, 1.01),
            ("panel_fitted_mmm", "monthly", True, 0.0),
            ("panel_fitted_mmm", "all_time", True, 1.01),
            ("monthly_fitted_mmm", "original", True, 0.0),
            ("time_varying_media_fitted_mmm", "original", True, 0.0),
            ("time_varying_media_fitted_mmm", "monthly", True, 0.0),
            ("time_varying_media_fitted_mmm", "all_time", True, 0.0),
            ("time_varying_intercept_fitted_mmm", "original", True, 0.0),
            ("time_varying_intercept_fitted_mmm", "monthly", True, 0.0),
            ("time_varying_intercept_fitted_mmm", "all_time", True, 0.0),
        ],
    )
    def test_compute_incremental_contribution_matches_ground_truth(
        self,
        request,
        model_fixture,
        frequency,
        include_carryover,
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
        )

        result = mmm.incrementality.compute_incremental_contribution(
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            include_carryover=include_carryover,
        )
        # check no Nans in ground truth
        assert not np.isnan(gt).any()  # sanity check
        xr.testing.assert_allclose(result, gt, rtol=1e-4)

    def test_negative_counterfactual_factor_raises_error(self, incrementality_lite):
        """Test that negative counterfactual factor raises ValueError."""
        incr, _ = incrementality_lite
        with pytest.raises(
            ValueError, match="counterfactual_spend_factor must be >= 0"
        ):
            incr.compute_incremental_contribution(
                frequency="all_time",
                counterfactual_spend_factor=-0.5,
            )

    def test_start_date_end_date_filters_dates(self, simple_fitted_mmm):
        """Test that start_date and end_date filter date range."""
        incr = simple_fitted_mmm.incrementality
        all_dates = pd.to_datetime(simple_fitted_mmm.idata.fit_data.date.values)
        mid_date = all_dates[len(all_dates) // 2]

        result = incr.compute_incremental_contribution(
            frequency="original",
            start_date=all_dates[0],
            end_date=mid_date,
        )

        result_dates = pd.to_datetime(result.date.values)
        assert result_dates[-1] <= mid_date
        assert len(result_dates) < len(all_dates)

    def test_num_samples_subsamples_posterior(self, simple_fitted_mmm):
        """Test that num_samples reduces draw dimension."""
        incr = simple_fitted_mmm.incrementality

        result_sub = incr.compute_incremental_contribution(
            frequency="all_time",
            num_samples=10,
            random_state=42,
        )

        assert result_sub.sizes["chain"] == 1
        assert result_sub.sizes["draw"] == 10

    def test_num_samples_respects_total_across_chains(self, simple_fitted_mmm):
        """Test that num_samples returns exactly num_samples with multi-chain posteriors."""
        import copy

        mmm = simple_fitted_mmm

        # Duplicate posterior along chain dim to simulate 2 chains
        original_posterior = mmm.idata.posterior
        if hasattr(original_posterior, "dataset"):
            original_posterior = original_posterior.dataset
        chain2 = original_posterior.assign_coords(
            chain=[original_posterior.chain.values[0] + 1]
        )
        multi_chain_posterior = xr.concat([original_posterior, chain2], dim="chain")

        # Replace posterior in idata with multi-chain version
        mmm_idata_backup = copy.copy(mmm.idata)
        mmm.idata["posterior"] = multi_chain_posterior

        # Sanity: verify we now have 2 chains
        n_chains = mmm.idata.posterior.dims["chain"]
        assert n_chains == 2, f"Expected 2 chains, got {n_chains}"

        num_samples = 10
        incr = mmm.incrementality
        result = incr.compute_incremental_contribution(
            frequency="all_time",
            num_samples=num_samples,
            random_state=42,
        )

        assert result.sizes["chain"] == 1
        assert result.sizes["draw"] == num_samples

        # Restore original idata
        mmm.idata = mmm_idata_backup

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

    def test_time_varying_intercept_no_unused_input_error(
        self, time_varying_intercept_fitted_mmm
    ):
        """Regression: time_varying_intercept without time_varying_media must not raise.

        When the model has time_varying_intercept=True but
        time_varying_media=False, ``time_index`` exists in the model but is
        not part of the ``channel_contribution`` graph.  Previously the code
        unconditionally added it as a compiled function input, causing
        ``UnusedInputError``.
        """
        incr = time_varying_intercept_fitted_mmm.incrementality
        result = incr.contribution_over_spend(frequency="all_time")
        assert result.sizes["chain"] >= 1
        assert result.sizes["draw"] >= 1

    def test_aggregate_spend_matches_data_wrapper(self, simple_fitted_mmm):
        """Test that _aggregate_spend delegates correctly to data wrapper."""
        incr = simple_fitted_mmm.incrementality

        spend_incr = incr._aggregate_spend(frequency="monthly")

        data_monthly = simple_fitted_mmm.data.aggregate_time(
            period="monthly", method="sum"
        )
        spend_data = data_monthly.get_channel_spend()

        xr.testing.assert_allclose(spend_incr, spend_data)

    def test_create_period_groups_rejects_mid_end_date(self, incrementality_lite):
        """_create_period_groups raises ValueError for mid-period end dates."""
        incr, _ = incrementality_lite

        with pytest.raises(ValueError, match="falls in the middle of a"):
            incr._create_period_groups(
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-02-15"),
                "monthly",
            )


class TestHelperMethods:
    """Tests for extracted helper methods."""

    def test_validate_input(self, incrementality_lite):
        """None dates default to data bounds; invalid ranges are rejected."""
        incr, _ = incrementality_lite
        dates = incr.data.dates

        # None dates default to data bounds
        start, end = incr._validate_input(None, None)
        assert start == dates[0]
        assert end == dates[-1]

        # Start before data raises
        with pytest.raises(ValueError, match="before fitted data"):
            incr._validate_input("1900-01-01", None)

        # End after data raises
        with pytest.raises(ValueError, match="after fitted data"):
            incr._validate_input(None, "2099-12-31")

        # Reversed range raises
        with pytest.raises(ValueError, match="is after"):
            incr._validate_input(dates[-1], dates[0])

    def test_compute_window_metadata_no_padding_interior(self, incrementality_lite):
        """Interior periods have no left/right padding."""
        incr, l_max = incrementality_lite
        dates = incr.data.dates
        freq = pd.infer_freq(dates)
        freq_offset = pd.tseries.frequencies.to_offset(freq)

        # Pick a period well within the data
        mid_idx = len(dates) // 2

        # Only run test if the period is far enough from boundaries
        if mid_idx >= l_max and (len(dates) - 1 - mid_idx) >= l_max:
            period = [(dates[mid_idx], dates[mid_idx])]
            window_infos, _max_window = Incrementality._compute_window_metadata(
                period, dates, l_max, freq_offset, freq
            )
            assert window_infos[0]["left_pad"] == 0
            assert window_infos[0]["right_pad"] == 0

    def test_compute_window_metadata_boundary_padding(self, incrementality_lite):
        """Boundary periods get non-zero padding."""
        incr, l_max = incrementality_lite
        dates = incr.data.dates
        freq = pd.infer_freq(dates)
        freq_offset = pd.tseries.frequencies.to_offset(freq)

        # First date should need left padding (ideal_start < data_start)
        first_period = [(dates[0], dates[0])]
        window_infos, _ = Incrementality._compute_window_metadata(
            first_period, dates, l_max, freq_offset, freq
        )
        assert window_infos[0]["left_pad"] > 0

    def test_build_counterfactual_scenarios_shape(self, incrementality_lite):
        """Counterfactual array has correct shape."""
        incr, l_max = incrementality_lite
        dates = incr.data.dates
        freq = pd.infer_freq(dates)
        freq_offset = pd.tseries.frequencies.to_offset(freq)
        baseline_array = incr.data.get_channel_spend().values
        extra_shape = baseline_array.shape[1:]

        periods = incr._create_period_groups(dates[0], dates[-1], "monthly")
        window_infos, max_window = Incrementality._compute_window_metadata(
            periods, dates, l_max, freq_offset, freq
        )

        cf_array, cf_eval_masks, period_labels = (
            Incrementality._build_counterfactual_scenarios(
                periods=periods,
                window_infos=window_infos,
                max_window=max_window,
                baseline_array=baseline_array,
                counterfactual_spend_factor=0.0,
                include_carryover=True,
                l_max=l_max,
                freq_offset=freq_offset,
                extra_shape=extra_shape,
                dtype="float64",
            )
        )

        assert cf_array.shape == (len(periods), max_window, *extra_shape)
        assert len(cf_eval_masks) == len(periods)
        assert len(period_labels) == len(periods)

    def test_build_counterfactual_scenarios_factor_applied(self, incrementality_lite):
        """Counterfactual factor is only applied to target period dates."""
        incr, l_max = incrementality_lite
        dates = incr.data.dates
        freq = pd.infer_freq(dates)
        freq_offset = pd.tseries.frequencies.to_offset(freq)
        baseline_array = incr.data.get_channel_spend().values
        extra_shape = baseline_array.shape[1:]

        # Use "original" frequency: each period is a single date
        periods = incr._create_period_groups(dates[0], dates[-1], "original")
        window_infos, max_window = Incrementality._compute_window_metadata(
            periods, dates, l_max, freq_offset, freq
        )

        # Factor = 0 should zero out only the target date
        cf_array_zero, _, _ = Incrementality._build_counterfactual_scenarios(
            periods=periods,
            window_infos=window_infos,
            max_window=max_window,
            baseline_array=baseline_array,
            counterfactual_spend_factor=0.0,
            include_carryover=True,
            l_max=l_max,
            freq_offset=freq_offset,
            extra_shape=extra_shape,
            dtype="float64",
        )

        # Factor = 1 (no change) should preserve all data
        cf_array_one, _, _ = Incrementality._build_counterfactual_scenarios(
            periods=periods,
            window_infos=window_infos,
            max_window=max_window,
            baseline_array=baseline_array,
            counterfactual_spend_factor=1.0,
            include_carryover=True,
            l_max=l_max,
            freq_offset=freq_offset,
            extra_shape=extra_shape,
            dtype="float64",
        )

        # With factor=0, some values should be zero (the target period dates)
        assert (cf_array_zero == 0).any()
        # With factor=1, counterfactual should equal baseline (no modification)
        # within padded windows
        expected_baseline = np.zeros(
            (len(periods), max_window, *extra_shape), dtype="float64"
        )
        for i, info in enumerate(window_infos):
            start_pos = info["left_pad"]
            end_pos = start_pos + info["n_actual"]
            if info["n_actual"] > 0:
                expected_baseline[i, start_pos:end_pos] = baseline_array[
                    info["in_window"]
                ].astype("float64")
        np.testing.assert_allclose(cf_array_one, expected_baseline)

    def test_invalid_frequency_raises_validation_error(self, incrementality_lite):
        """Invalid frequency raises pydantic ValidationError."""
        incr, _ = incrementality_lite

        with pytest.raises(ValidationError):
            incr._create_period_groups(
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-03-31"),
                "biweekly",  # invalid
            )


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
        # check that we didn't drop any dates (sanity check)
        if frequency != "all_time":
            assert len(roas.date) == len(spend.date)
        # check no Nans in ground truth (sanity check)
        assert not np.isnan(roas_ground_truth).any()
        # check that roas is not empty (sanity check)
        assert roas.size > 0
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

        mean_total = total_roas.mean(dim=["chain", "draw"])
        mean_marginal = marginal_roas.mean(dim=["chain", "draw"])

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
        """Test that convenience functions accept start_date/end_date."""
        incr = simple_fitted_mmm.incrementality
        all_dates = pd.to_datetime(simple_fitted_mmm.idata.fit_data.date.values)
        mid_date = all_dates[len(all_dates) // 2]

        method = getattr(incr, method_name)
        result = method(
            frequency="original",
            start_date=all_dates[0],
            end_date=mid_date,
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
        assert result.sizes["draw"] == 10


class TestIncrementalReducers:
    """Unit tests for the link-specific reduction, without a fitted model.

    These exercise the one piece of the calculation that depends on the link
    function, in isolation from the counterfactual machinery.
    """

    @staticmethod
    def _delta(values, dates):
        return xr.DataArray(
            np.asarray(values, dtype=float),
            dims=("sample", "date", "channel"),
            coords={"date": dates},
        )

    def test_identity_reducer_is_scaled_sum_over_date(self):
        """Identity link: the base term cancels, leaving s * sum_t delta."""
        dates = pd.date_range("2024-01-01", periods=3, freq="W")
        delta = self._delta([[[1.0, -2.0], [3.0, 4.0], [0.5, 0.5]]], dates)

        result = IdentityLinkReducer(scale=10.0).counterfactual_minus_baseline(delta)

        np.testing.assert_allclose(result.values, [[45.0, 25.0]])
        assert result.dims == ("sample", "channel")

    def test_log_reducer_weights_by_baseline_response(self):
        """Log link: sum_t yhat_t * (exp(delta) - 1)."""
        dates = pd.date_range("2024-01-01", periods=2, freq="W")
        delta = self._delta([[[0.1, -0.2], [0.3, -0.4]]], dates)
        baseline = xr.DataArray(
            [[100.0, 200.0]],
            dims=("sample", "date"),
            coords={"date": dates},
        )

        result = LogLinkReducer(baseline_response=baseline)
        result = result.counterfactual_minus_baseline(delta)

        expected = [
            [
                100.0 * np.expm1(0.1) + 200.0 * np.expm1(0.3),
                100.0 * np.expm1(-0.2) + 200.0 * np.expm1(-0.4),
            ]
        ]
        np.testing.assert_allclose(result.values, expected)

    def test_log_reducer_selects_baseline_by_date_label(self):
        """The baseline spans the fitted data; only the eval window is used."""
        all_dates = pd.date_range("2024-01-01", periods=5, freq="W")
        baseline = xr.DataArray(
            [[1.0, 2.0, 4.0, 8.0, 16.0]],
            dims=("sample", "date"),
            coords={"date": all_dates},
        )
        window = all_dates[2:4]
        delta = self._delta([[[0.5], [0.25]]], window)

        result = LogLinkReducer(
            baseline_response=baseline
        ).counterfactual_minus_baseline(delta)

        expected = 4.0 * np.expm1(0.5) + 8.0 * np.expm1(0.25)
        np.testing.assert_allclose(result.values, [[expected]])

    def test_log_reducer_zero_perturbation_gives_zero(self):
        """No change in the linear predictor means no incremental response."""
        dates = pd.date_range("2024-01-01", periods=3, freq="W")
        delta = self._delta(np.zeros((2, 3, 2)), dates)
        baseline = xr.DataArray(
            np.full((2, 3), 50.0),
            dims=("sample", "date"),
            coords={"date": dates},
        )

        result = LogLinkReducer(
            baseline_response=baseline
        ).counterfactual_minus_baseline(delta)

        np.testing.assert_allclose(result.values, np.zeros((2, 2)))

    def test_log_reducer_beats_naive_exp_minus_one_on_tiny_perturbation(self):
        """expm1 keeps precision where exp(x) - 1 loses it.

        Marginal incrementality perturbs spend by 1 %, so delta is small and
        ``exp(delta) - 1`` cancels leading digits.
        """
        dates = pd.date_range("2024-01-01", periods=1, freq="W")
        tiny = 1e-17
        delta = self._delta([[[tiny]]], dates)
        baseline = xr.DataArray(
            [[1.0]], dims=("sample", "date"), coords={"date": dates}
        )

        result = LogLinkReducer(
            baseline_response=baseline
        ).counterfactual_minus_baseline(delta)

        # exp(1e-17) - 1 rounds to exactly 0 in float64; expm1 does not.
        assert np.exp(tiny) - 1.0 == 0.0
        np.testing.assert_allclose(result.values, [[tiny]])

    def test_reducer_is_abstract(self):
        """The base class cannot be instantiated without a reduction."""
        with pytest.raises(TypeError):
            IncrementalReducer()


class TestLogLinkIncrementality:
    """Incrementality for multiplicative (log-link) models."""

    @pytest.mark.parametrize(
        "model_fixture, frequency, include_carryover, counterfactual_spend_factor",
        [
            ("log_link_fitted_mmm", "all_time", True, 0.0),
            ("log_link_fitted_mmm", "monthly", True, 0.0),
            ("log_link_fitted_mmm", "all_time", False, 0.0),
            ("log_link_fitted_mmm", "all_time", True, 1.01),
            ("log_link_panel_fitted_mmm", "all_time", True, 0.0),
        ],
    )
    def test_matches_per_channel_ground_truth(
        self,
        request,
        model_fixture,
        frequency,
        include_carryover,
        counterfactual_spend_factor,
    ):
        """Validate against brute-force per-channel response-scale counterfactuals.

        The ground truth runs one ``sample_posterior_predictive`` per
        ``(period, channel)`` on ``y_original_scale``.  Matching it confirms
        that reading per-channel increments off a single all-channels
        perturbation stays valid under a log link, provided the baseline
        response is used as the weight.
        """
        mmm = request.getfixturevalue(model_fixture)

        gt = compute_log_link_ground_truth_by_period(
            mmm,
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            include_carryover=include_carryover,
        )
        result = mmm.incrementality.compute_incremental_contribution(
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            include_carryover=include_carryover,
        )

        assert not np.isnan(gt).any()  # sanity check
        xr.testing.assert_allclose(result, gt, rtol=1e-6)

    def test_single_channel_equals_total_media_contribution(
        self, log_link_single_channel_fitted_mmm
    ):
        """With one channel, incrementality must equal the model's own counterfactual.

        ``LogLinkSpec.create_media_contribution_deterministic`` registers
        ``total_media_contribution_original_scale`` as
        ``sum_t [exp(mu) - exp(mu - mu_media)] * s``.  With a single channel
        ``mu_media`` is that channel's contribution, and ``LogSaturation`` maps
        zero spend to zero contribution, so a full-range zero-out counterfactual
        must reproduce it draw for draw.
        """
        mmm = log_link_single_channel_fitted_mmm

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time",
            counterfactual_spend_factor=0.0,
        ).squeeze("channel", drop=True)
        expected = mmm.idata.posterior["total_media_contribution_original_scale"]

        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_result_is_not_the_linear_predictor_difference(self, log_link_fitted_mmm):
        """Guard against a regression to summing linear-predictor contributions.

        The pre-fix behaviour summed ``channel_contribution`` differences and
        multiplied by ``target_scale``, which is not incremental response under
        a log link.  Assert the two disagree, so an accidental revert fails
        loudly rather than returning plausible-looking numbers.
        """
        mmm = log_link_fitted_mmm

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time",
            counterfactual_spend_factor=0.0,
        )

        actual_data = mmm.model["channel_data"].get_value()
        baseline = evaluate_under_spend(mmm, actual_data, "channel_contribution")
        cf = evaluate_under_spend(
            mmm, np.zeros_like(actual_data), "channel_contribution"
        )
        linear_predictor_version = (baseline - cf).sum(
            dim="date"
        ) * mmm.data.get_target_scale()

        assert not np.allclose(
            result.values, linear_predictor_version.transpose(*result.dims).values
        )

    def test_mean_central_tendency_applies_lognormal_correction(
        self, log_link_fitted_mmm
    ):
        """``central_tendency="mean"`` rescales by exp(sigma**2 / 2) per draw.

        The correction multiplies the baseline response, which enters the
        increment linearly, so it factors straight through the sum.
        """
        incr = log_link_fitted_mmm.incrementality
        kwargs = {"frequency": "all_time", "counterfactual_spend_factor": 0.0}

        median = incr.compute_incremental_contribution(
            central_tendency="median", **kwargs
        )
        mean = incr.compute_incremental_contribution(central_tendency="mean", **kwargs)

        sigma = log_link_fitted_mmm.idata.posterior["y_sigma"]
        expected_ratio = np.exp(sigma**2 / 2)

        ratio = (mean / median).transpose("chain", "draw", "channel")
        np.testing.assert_allclose(
            ratio.values,
            np.broadcast_to(
                expected_ratio.values[..., None],
                ratio.shape,
            ),
            rtol=1e-6,
        )

    def test_convenience_methods_run_under_log_link(self, log_link_fitted_mmm):
        """ROAS, CAC and mROAS all reduce through the log-link path."""
        incr = log_link_fitted_mmm.incrementality

        roas = incr.contribution_over_spend(frequency="all_time")
        cac = incr.spend_over_contribution(frequency="all_time")
        mroas = incr.marginal_contribution_over_spend(frequency="all_time")

        for result in (roas, cac, mroas):
            assert set(result.dims) == {"chain", "draw", "channel"}
            assert np.isfinite(result.values).all()
        np.testing.assert_allclose(cac.values, 1.0 / roas.values, rtol=1e-10)


class TestLinkDispatch:
    """Selection of the link-specific reducer."""

    def test_identity_link_uses_identity_reducer(self, simple_fitted_mmm):
        incr = simple_fitted_mmm.incrementality
        reducer = incr._build_reducer(incr.idata.posterior.dataset, "median")
        assert isinstance(reducer, IdentityLinkReducer)

    def test_log_link_uses_log_reducer(self, log_link_fitted_mmm):
        incr = log_link_fitted_mmm.incrementality
        reducer = incr._build_reducer(incr.idata.posterior.dataset, "median")
        assert isinstance(reducer, LogLinkReducer)

    def test_unknown_link_raises_not_implemented(self, simple_fitted_mmm):
        """A new link must add a reducer rather than silently reuse the additive one."""
        incr = simple_fitted_mmm.incrementality
        incr.model.link = "sqrt"

        with pytest.raises(NotImplementedError, match="link='sqrt'"):
            incr.compute_incremental_contribution(frequency="all_time")

    def test_invalid_central_tendency_raises(self, simple_fitted_mmm):
        incr = simple_fitted_mmm.incrementality

        with pytest.raises(ValueError, match="central_tendency must be"):
            incr.compute_incremental_contribution(
                frequency="all_time", central_tendency="mode"
            )

    def test_central_tendency_is_a_no_op_for_identity_link(self, simple_fitted_mmm):
        """The Normal likelihood's mean equals its median, so both agree."""
        incr = simple_fitted_mmm.incrementality

        median = incr.compute_incremental_contribution(
            frequency="all_time", central_tendency="median"
        )
        mean = incr.compute_incremental_contribution(
            frequency="all_time", central_tendency="mean"
        )

        xr.testing.assert_allclose(median, mean)

    def test_missing_baseline_response_raises(self, log_link_fitted_mmm):
        """A posterior stripped of the response-scale prediction fails clearly."""
        incr = log_link_fitted_mmm.incrementality
        posterior = incr.idata.posterior.dataset.drop_vars("y_original_scale")

        with pytest.raises(ValueError, match="y_original_scale"):
            incr._build_reducer(posterior, "median")
