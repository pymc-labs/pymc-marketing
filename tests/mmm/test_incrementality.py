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
import pytest
import xarray as xr
from pydantic import ValidationError

from pymc_marketing.mmm import incrementality as incrementality_module
from pymc_marketing.mmm.additive_effect import IncrementalitySpec
from pymc_marketing.mmm.counterfactual import (
    CounterfactualEvaluator,
    EvaluationWindows,
)
from pymc_marketing.mmm.incrementality import (
    IdentityLinkReducer,
    Incrementality,
    IncrementalReducer,
    LogLinkReducer,
)
from pymc_marketing.mmm.spend_reach import (
    SpendProbe,
    TemporalReach,
    resolve_channel_dependent_effects,
)
from tests.mmm.incrementality_utils import (
    compute_ground_truth_incremental_by_period,
    compute_log_link_ground_truth_by_period,
    effective_l_max,
    evaluate_under_spend,
    measure_reach,
    measure_spend_reach,
    mediated_identity_oracle,
    source_spend,
)


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
        # The identity-link reduction is a sum of differences against the
        # oracle's difference of sums, so the two agree to float rounding and
        # nothing about the algebra is approximate.  Measured across this whole
        # parametrisation the worst relative error is 2.2e-11, so the tolerance
        # keeps three orders of margin rather than the seven 1e-4 allowed.
        xr.testing.assert_allclose(result, gt, rtol=1e-8)

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

    def test_num_samples_selects_whole_draws_across_chains(self, simple_fitted_mmm):
        """``num_samples`` counts across chains and carries whole draws through.

        The second chain gets a rescaled ``saturation_beta`` so every
        ``(chain, draw)`` pair yields a distinct increment.  Duplicating the
        posterior instead would let a chain-major/draw-major mix-up pass
        unnoticed, since both orderings would then select equal values.
        """
        mmm = simple_fitted_mmm
        posterior = mmm.idata.posterior
        if hasattr(posterior, "dataset"):
            posterior = posterior.dataset

        chain2 = posterior.assign_coords(chain=[posterior.chain.values[0] + 1])
        chain2["saturation_beta"] = chain2["saturation_beta"] * 0.5
        multi_chain = xr.concat([posterior, chain2], dim="chain")
        mmm.idata["posterior"] = multi_chain

        n_chains = multi_chain.sizes["chain"]
        n_draws = multi_chain.sizes["draw"]
        num_samples = 10
        assert n_chains == 2

        full = mmm.incrementality.compute_incremental_contribution(frequency="all_time")
        subset = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time",
            num_samples=num_samples,
            random_state=42,
        )

        # num_samples is a total, not a per-chain count.
        assert subset.sizes["chain"] == 1
        assert subset.sizes["draw"] == num_samples

        # The premise: the chains genuinely disagree, so draw identity matters.
        assert not np.allclose(full.isel(chain=0).values, full.isel(chain=1).values)

        # Same chain-major flat selection that subsample_draws makes.
        flat_indices = np.random.default_rng(42).choice(
            n_chains * n_draws, size=num_samples, replace=False
        )
        expected = (
            full.stack(sample=("chain", "draw"))
            .isel(sample=flat_indices)
            .transpose("sample", ...)
        )

        np.testing.assert_allclose(
            subset.squeeze("chain", drop=True).transpose("draw", ...).values,
            expected.values,
            rtol=1e-10,
        )

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

    def test_an_interior_window_reaches_l_max_either_side(self, incrementality_lite):
        """An interior period reaches l_max fitted dates either side."""
        incr, l_max = incrementality_lite
        dates = incr.data.dates
        freq_offset = pd.tseries.frequencies.to_offset(pd.infer_freq(dates))

        # Pick a period well within the data
        mid_idx = len(dates) // 2

        # Only run test if the period is far enough from boundaries
        if mid_idx >= l_max and (len(dates) - 1 - mid_idx) >= l_max:
            windows = EvaluationWindows.build(
                periods=[(dates[mid_idx], dates[mid_idx])],
                dates=dates,
                l_max=l_max,
                freq_offset=freq_offset,
            )
            assert windows.windows[0].n_actual == 2 * l_max + 1

    def test_a_boundary_window_is_clamped_to_the_data(self, incrementality_lite):
        """A boundary period's window stops at the data instead of padding past it.

        There is no history before the first fitted date to supply, and the
        graph's own adstock already pads with zeros there.  Synthesising rows
        instead would be inert for spend but not for an effect whose contribution
        at zero spend is its own intercept.
        """
        incr, l_max = incrementality_lite
        dates = incr.data.dates
        freq_offset = pd.tseries.frequencies.to_offset(pd.infer_freq(dates))

        windows = EvaluationWindows.build(
            periods=[(dates[0], dates[0])],
            dates=dates,
            l_max=l_max,
            freq_offset=freq_offset,
        )
        # No dates exist before dates[0], so the window starts there and only
        # carries the carry-out side.
        assert windows.windows[0].actual_dates[0] == dates[0]
        assert windows.windows[0].n_actual == l_max + 1

    def test_full_axis_windows_span_every_fitted_date(self, incrementality_lite):
        """Full-axis mode gives every period the complete date axis."""
        incr, l_max = incrementality_lite
        dates = incr.data.dates
        freq_offset = pd.tseries.frequencies.to_offset(pd.infer_freq(dates))
        periods = incr._create_period_groups(dates[0], dates[-1], "monthly")

        windows = EvaluationWindows.build(
            periods=periods,
            dates=dates,
            l_max=l_max,
            freq_offset=freq_offset,
            full_axis=True,
        )

        assert windows.max_window == len(dates)
        assert all(window.n_actual == len(dates) for window in windows.windows)
        # Each period still knows its own bounds, which is what keeps the
        # perturbation and the aggregation period-specific.
        assert [(w.start, w.end) for w in windows.windows] == list(periods)

    def test_build_counterfactual_scenarios_shape(self, incrementality_lite):
        """Counterfactual array has correct shape."""
        incr, l_max = incrementality_lite
        dates = incr.data.dates
        freq_offset = pd.tseries.frequencies.to_offset(pd.infer_freq(dates))
        baseline_array = incr.data.get_channel_spend().values
        extra_shape = baseline_array.shape[1:]

        periods = incr._create_period_groups(dates[0], dates[-1], "monthly")
        windows = EvaluationWindows.build(
            periods=periods, dates=dates, l_max=l_max, freq_offset=freq_offset
        )

        scenarios = windows.build_scenarios(
            baseline_array=baseline_array,
            counterfactual_spend_factor=0.0,
            dtype="float64",
            channel_axis=None,
            n_channels=baseline_array.shape[-1],
            estimand="per_channel",
        )

        # Separable mode: one all-channels scenario per period.
        assert scenarios.spend.shape == (
            len(periods),
            windows.max_window,
            *extra_shape,
        )
        assert [window.end for window in windows.windows] == [end for _, end in periods]
        # Every channel of a period reads that period's single scenario.
        assert {
            scenarios.rows[(0, channel)] for channel in range(baseline_array.shape[-1])
        } == {0}

    def test_build_counterfactual_scenarios_factor_applied(self, incrementality_lite):
        """Counterfactual factor is only applied to target period dates."""
        incr, l_max = incrementality_lite
        dates = incr.data.dates
        freq_offset = pd.tseries.frequencies.to_offset(pd.infer_freq(dates))
        baseline_array = incr.data.get_channel_spend().values
        extra_shape = baseline_array.shape[1:]

        # Use "original" frequency: each period is a single date
        periods = incr._create_period_groups(dates[0], dates[-1], "original")
        windows = EvaluationWindows.build(
            periods=periods, dates=dates, l_max=l_max, freq_offset=freq_offset
        )

        def build(factor):
            return windows.build_scenarios(
                baseline_array=baseline_array,
                counterfactual_spend_factor=factor,
                dtype="float64",
                channel_axis=None,
                n_channels=baseline_array.shape[-1],
                estimand="per_channel",
            ).spend

        # Factor = 0 should zero out only the target date
        cf_array_zero = build(0.0)
        # Factor = 1 (no change) should preserve all data
        cf_array_one = build(1.0)

        # With factor=0, some values should be zero (the target period dates)
        assert (cf_array_zero == 0).any()
        # With factor=1, counterfactual should equal baseline (no modification)
        # within padded windows
        expected_baseline = np.zeros(
            (len(periods), windows.max_window, *extra_shape), dtype="float64"
        )
        for i, window in enumerate(windows.windows):
            expected_baseline[i, : window.n_actual] = baseline_array[
                window.in_window
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
    def test_contribution_over_spend_matches_the_oracle(
        self, simple_fitted_mmm, simple_mmm_data, frequency
    ):
        """ROAS against a numerator and a denominator the module did not compute.

        Both halves are independent: the increment comes from the
        ``sample_posterior_predictive`` oracle, the spend from the fixture's own
        frame.  Dividing the module's own increment by the module's own spend --
        as this test used to -- checks the division and would pass with an
        arbitrarily wrong increment, which is the whole of what ROAS is.
        """
        roas = simple_fitted_mmm.incrementality.contribution_over_spend(
            frequency=frequency
        )
        increment = compute_ground_truth_incremental_by_period(
            simple_fitted_mmm, frequency=frequency
        )
        spend = source_spend(
            simple_mmm_data["X"], frequency, simple_fitted_mmm.channel_columns
        )

        assert roas.size > 0  # sanity check
        assert not np.isnan(roas).any()  # sanity check
        xr.testing.assert_allclose(roas, increment / spend, rtol=1e-6)

    @pytest.mark.parametrize("frequency", ["all_time", "monthly"])
    def test_marginal_contribution_over_spend_matches_the_oracle(
        self, simple_fitted_mmm, simple_mmm_data, frequency
    ):
        """mROAS is equation (11): a 1 % perturbation over 1 % of the spend.

        Same independence as the total: the numerator is the oracle's increment at
        ``alpha = 1.01`` and the denominator is one percent of the source frame's
        spend, so the factor and the divisor are checked rather than cancelled.
        """
        spend_increase_pct = 0.01

        mroas = simple_fitted_mmm.incrementality.marginal_contribution_over_spend(
            frequency=frequency, spend_increase_pct=spend_increase_pct
        )
        increment = compute_ground_truth_incremental_by_period(
            simple_fitted_mmm,
            frequency=frequency,
            counterfactual_spend_factor=1.0 + spend_increase_pct,
        )
        spend = source_spend(
            simple_mmm_data["X"], frequency, simple_fitted_mmm.channel_columns
        )

        assert not np.isnan(mroas).any()  # sanity check
        xr.testing.assert_allclose(
            mroas, increment / (spend_increase_pct * spend), rtol=1e-6
        )

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


class TestStackSamples:
    """Flattening ``(chain, draw)`` ahead of the link-specific reduction."""

    def test_preserves_custom_dims_and_their_coords(self):
        """Panel quantities keep every non-sample dimension, labels included.

        ``y_sigma`` and ``y_original_scale`` both carry the model's custom
        dims, and the reduction broadcasts them against a ``(date, channel,
        *custom_dims)`` perturbation by name, so the labels have to survive.
        """
        da = xr.DataArray(
            np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2),
            dims=("chain", "draw", "country"),
            coords={"country": ["US", "UK"]},
        )

        stacked = Incrementality._stack_samples(da)

        assert stacked.dims == ("sample", "country")
        assert stacked.coords["country"].values.tolist() == ["US", "UK"]

    def test_flattening_is_chain_major(self):
        """Positions along ``sample`` must line up with the batched evaluation.

        ``extract_response_distribution`` stacks the posterior chain-major, so
        anything multiplied into its output has to use the same order.
        """
        da = xr.DataArray(
            np.arange(2 * 3, dtype=float).reshape(2, 3),
            dims=("chain", "draw"),
        )

        stacked = Incrementality._stack_samples(da)

        assert stacked.dims == ("sample",)
        np.testing.assert_allclose(stacked.values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


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
            # At all_time the carryover mask cannot differ (the last period
            # already ends at the last fitted date), so a monthly case is what
            # actually exercises include_carryover=False under a log link.
            ("log_link_fitted_mmm", "monthly", False, 0.0),
            ("log_link_fitted_mmm", "all_time", True, 1.01),
            # Factors strictly between 0 and 1 measure the cost of *less*
            # spend, and 1.0 is the no-op boundary of the sign convention.
            ("log_link_fitted_mmm", "all_time", True, 0.5),
            ("log_link_fitted_mmm", "all_time", True, 1.0),
            # Controls enter the baseline weight rather than cancelling.
            ("log_link_control_fitted_mmm", "all_time", True, 0.0),
            ("log_link_control_fitted_mmm", "all_time", True, 1.01),
            # HSGP media multiplier on the date-dependent evaluation path.
            ("log_link_time_varying_media_fitted_mmm", "all_time", True, 0.0),
            ("log_link_time_varying_media_fitted_mmm", "monthly", True, 0.0),
            ("log_link_panel_fitted_mmm", "all_time", True, 0.0),
            ("log_link_panel_fitted_mmm", "monthly", True, 0.0),
            ("log_link_panel_fitted_mmm", "all_time", False, 0.0),
            ("log_link_panel_fitted_mmm", "all_time", True, 1.01),
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

    def test_channel_increments_exceed_total_media_by_the_overlap(
        self, log_link_fitted_mmm
    ):
        """Per-channel increments must not sum to the total media increment.

        Under a log link, removing both channels at once is not the sum of
        removing each alone.  Writing :math:`e_m = \\exp(-v_m)`, the per-channel
        increments sum to :math:`\\sum_t \\hat{Y}_t (2 - e_1 - e_2)` while the
        joint knock-out is :math:`\\sum_t \\hat{Y}_t (1 - e_1 e_2)`, leaving an
        overlap of :math:`\\sum_t \\hat{Y}_t (1 - e_1)(1 - e_2) > 0` whenever
        both contributions are positive.  So the per-channel sum *overstates*
        the joint effect: media effects double count where they coincide.

        This is a property of the model, not of the estimator.  Asserting the
        direction pins the interpretation down, so nobody "fixes" the module
        into forcing additivity.
        """
        mmm = log_link_fitted_mmm
        kwargs = {"frequency": "all_time", "counterfactual_spend_factor": 0.0}

        per_channel = mmm.incrementality.compute_incremental_contribution(**kwargs)
        joint = mmm.incrementality.compute_joint_incremental_contribution(**kwargs)

        # LogSaturation with a HalfNormal beta keeps every contribution
        # positive, which is what fixes the sign of the overlap term.
        assert (mmm.idata.posterior["channel_contribution"] > 0).all()

        summed = per_channel.sum(dim="channel")
        assert (summed > joint).all()
        assert not np.allclose(summed.values, joint.values)

        # The joint estimand is a spend knock-out, and LogSaturation sends zero
        # spend to zero contribution, so it has to land on the model's own
        # media counterfactual deterministic.
        xr.testing.assert_allclose(
            joint,
            mmm.idata.posterior["total_media_contribution_original_scale"],
            rtol=1e-6,
        )

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

    def test_panel_mean_central_tendency_corrects_each_custom_dim_cell(
        self, log_link_panel_fitted_mmm
    ):
        """Panel ``central_tendency="mean"`` applies each cell's own sigma.

        A panel ``y_sigma`` keeps the model's custom dims, so the LogNormal
        correction differs per country.  Collapsing it to a single factor --
        or dropping the dim while stacking samples -- has to fail here.
        """
        incr = log_link_panel_fitted_mmm.incrementality
        kwargs = {"frequency": "all_time", "counterfactual_spend_factor": 0.0}

        median = incr.compute_incremental_contribution(
            central_tendency="median", **kwargs
        )
        mean = incr.compute_incremental_contribution(central_tendency="mean", **kwargs)

        sigma = log_link_panel_fitted_mmm.idata.posterior["y_sigma"]
        assert "country" in sigma.dims  # guards the premise of this test

        ratio = mean / median
        xr.testing.assert_allclose(
            ratio, np.exp(sigma**2 / 2).broadcast_like(ratio), rtol=1e-6
        )

    def test_convenience_methods_match_the_log_link_oracle(
        self, log_link_fitted_mmm, log_link_mmm_data
    ):
        """ROAS and mROAS under a log link, against response-scale counterfactuals.

        This is the headline of the change and it needs a number, not a shape.
        Asserting only that the results are finite and reciprocal -- as this used
        to -- passes for any increment whatsoever, so the correctness of the
        log-link ROAS rested entirely on the increment's own tests.
        """
        mmm = log_link_fitted_mmm
        incr = mmm.incrementality
        spend = source_spend(log_link_mmm_data["X"], "all_time", mmm.channel_columns)

        roas = incr.contribution_over_spend(frequency="all_time")
        mroas = incr.marginal_contribution_over_spend(frequency="all_time")

        xr.testing.assert_allclose(
            roas,
            compute_log_link_ground_truth_by_period(mmm, frequency="all_time") / spend,
            rtol=1e-6,
        )
        xr.testing.assert_allclose(
            mroas,
            compute_log_link_ground_truth_by_period(
                mmm, frequency="all_time", counterfactual_spend_factor=1.01
            )
            / (0.01 * spend),
            rtol=1e-6,
        )

    def test_cac_is_the_reciprocal_of_roas_under_log_link(self, log_link_fitted_mmm):
        """CAC adds nothing but a reciprocal, and the dimensions survive it."""
        incr = log_link_fitted_mmm.incrementality

        roas = incr.contribution_over_spend(frequency="all_time")
        cac = incr.spend_over_contribution(frequency="all_time")

        for result in (roas, cac):
            assert set(result.dims) == {"chain", "draw", "channel"}
            assert np.isfinite(result.values).all()
        np.testing.assert_allclose(cac.values, 1.0 / roas.values, rtol=1e-10)

    def test_a_two_chain_posterior_matches_the_oracle(
        self, log_link_two_chain_fitted_mmm
    ):
        """Draws are flattened chain-major, and the flattening has to be right.

        The reduction lines a posterior array up against a batched graph
        evaluation positionally, so its ``(chain, draw)`` flattening has to agree
        with the one ``extract_response_distribution`` used.  With a single chain
        both orders coincide and a mismatch is invisible; with two it is a wrong
        number rather than an error, which is why every log-link fixture having
        ``chain=1`` was worth fixing.
        """
        mmm = log_link_two_chain_fitted_mmm
        assert mmm.idata.posterior.dataset.sizes["chain"] == 2  # guards the premise

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time"
        )
        expected = compute_log_link_ground_truth_by_period(mmm, frequency="all_time")

        xr.testing.assert_allclose(result, expected, rtol=1e-6)

        # The mean correction is also per draw, so it has to survive the same
        # flattening rather than being broadcast against the wrong chain.
        mean = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time", central_tendency="mean"
        )
        sigma = mmm.idata.posterior["y_sigma"]
        xr.testing.assert_allclose(
            mean / result, np.exp(sigma**2 / 2).broadcast_like(result), rtol=1e-6
        )

    def test_num_samples_matches_the_oracle_on_the_selected_draws(
        self, log_link_two_chain_fitted_mmm
    ):
        """Subsampling keeps the baseline weights aligned with the evaluated draws.

        ``LogLinkReducer`` weights each draw's linear-predictor change by the
        *same* draw's baseline response, taken from the subsampled posterior,
        while the change itself comes from a graph conditioned on that same
        subsample.  A draw misalignment between the two would weight deltas by
        the wrong draws' baselines -- a silently wrong number, not an error --
        and a self-consistency check cannot see it.  So the subsampled result is
        compared against the full oracle restricted to the very draws
        ``subsample_draws`` selects, on a two-chain posterior so the chain-major
        flattening is observable too.
        """
        mmm = log_link_two_chain_fitted_mmm
        posterior = mmm.idata.posterior.dataset
        n_chains, n_draws = posterior.sizes["chain"], posterior.sizes["draw"]
        num_samples = 10

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time", num_samples=num_samples, random_state=123
        )

        full = compute_log_link_ground_truth_by_period(mmm, frequency="all_time")
        # Same chain-major flat selection that subsample_draws makes.
        flat_indices = np.random.default_rng(123).choice(
            n_chains * n_draws, size=num_samples, replace=False
        )
        expected = (
            full.stack(sample=("chain", "draw"))
            .isel(sample=flat_indices)
            .transpose("sample", ...)
        )

        np.testing.assert_allclose(
            result.squeeze("chain", drop=True).transpose("draw", ...).values,
            expected.values,
            rtol=1e-6,
        )

    def test_mean_central_tendency_matches_the_corrected_oracle(
        self, log_link_fitted_mmm
    ):
        """End to end, the mean scale is the median-scale oracle times exp(sigma^2/2).

        The ratio tests above cannot catch a mean-scale result that is wrong on
        both scales by the same factor.  The correction multiplies the baseline
        weight, which enters the increment linearly, so the corrected oracle is
        exact rather than approximate -- and a monthly frequency runs it through
        the windowing rather than only the full range.
        """
        mmm = log_link_fitted_mmm

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="monthly", central_tendency="mean"
        )

        sigma = mmm.idata.posterior["y_sigma"]
        expected = compute_log_link_ground_truth_by_period(
            mmm, frequency="monthly"
        ) * np.exp(sigma**2 / 2)

        xr.testing.assert_allclose(result, expected.transpose(*result.dims), rtol=1e-6)

    def test_contribution_over_spend_matches_the_monthly_log_link_oracle(
        self, log_link_fitted_mmm, log_link_mmm_data
    ):
        """Monthly ROAS under a log link: windowed increments over monthly spend.

        The all-time case divides one number by one number; the monthly one has
        to line period labels up between the increment and the spend, which is
        where a date-alignment bug would hide.
        """
        mmm = log_link_fitted_mmm

        roas = mmm.incrementality.contribution_over_spend(frequency="monthly")

        spend = source_spend(log_link_mmm_data["X"], "monthly", mmm.channel_columns)
        expected = (
            compute_log_link_ground_truth_by_period(mmm, frequency="monthly") / spend
        )

        assert not np.isnan(roas).any()
        xr.testing.assert_allclose(roas, expected, rtol=1e-6)

    def test_panel_roas_matches_the_log_link_oracle(
        self, log_link_panel_fitted_mmm, panel_mmm_data
    ):
        """Panel ROAS divides each country's increment by that country's spend."""
        mmm = log_link_panel_fitted_mmm

        roas = mmm.incrementality.contribution_over_spend(frequency="all_time")

        totals = panel_mmm_data["X"].groupby("country")[list(mmm.channel_columns)].sum()
        spend = xr.DataArray(
            totals.to_numpy(),
            dims=("country", "channel"),
            coords={
                "country": totals.index.to_list(),
                "channel": list(mmm.channel_columns),
            },
        )
        expected = (
            compute_log_link_ground_truth_by_period(mmm, frequency="all_time") / spend
        )

        assert set(roas.dims) == {"chain", "draw", "channel", "country"}
        xr.testing.assert_allclose(roas, expected.transpose(*roas.dims), rtol=1e-6)

    def test_date_filtering_matches_the_log_link_oracle(self, log_link_fitted_mmm):
        """``start_date``/``end_date`` restrict the periods, not just the output.

        The identity-link filtering test only checks the returned date labels;
        under a log link the filtered increment also has to carry the right
        baseline weights for the shorter window, so it is compared against the
        oracle built over the same restricted range.
        """
        mmm = log_link_fitted_mmm
        dates = pd.to_datetime(mmm.idata.fit_data.date.values)
        start, end = dates[3], dates[9]

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time", start_date=start, end_date=end
        )
        expected = compute_log_link_ground_truth_by_period(
            mmm, frequency="all_time", start_date=start, end_date=end
        )

        xr.testing.assert_allclose(result, expected, rtol=1e-6)
        # And the restriction genuinely changed the answer.
        full = mmm.incrementality.compute_incremental_contribution(frequency="all_time")
        assert not np.allclose(result.values, full.values)


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

    def test_fixed_sigma_lognormal_refuses_mean_central_tendency(
        self, log_link_fixed_sigma_fitted_mmm
    ):
        """A fixed LogNormal sigma leaves no ``y_sigma`` to build the correction from.

        The median scale needs no correction, so it still works; asking for the
        mean has to fail with a message naming the missing variable rather than
        silently returning median-scale numbers.
        """
        incr = log_link_fixed_sigma_fitted_mmm.incrementality

        median = incr.compute_incremental_contribution(frequency="all_time")
        assert np.isfinite(median.values).all()

        with pytest.raises(ValueError, match="y_sigma"):
            incr.compute_incremental_contribution(
                frequency="all_time", central_tendency="mean"
            )


class TestMediatedMuEffects:
    """Incrementality when a ``mu_effect`` stands between spend and the response.

    The module's original assumption was that a spend counterfactual reaches the
    response only through ``channel_contribution``.  A funnel effect breaks that:
    it reads ``channel_data`` itself, so part of the incremental response travels
    through the effect rather than through ``channel_contribution``.  These tests
    pin down that the mediated part is now included, that it is *material* (so a
    regression would be caught rather than shrugged at), and that an effect which
    has not opted in is refused instead of silently dropped.
    """

    @pytest.mark.parametrize("frequency", ["all_time", "monthly"])
    @pytest.mark.parametrize("counterfactual_spend_factor", [0.0, 1.01])
    def test_matches_oracle_under_log_link(
        self, funnel_log_link_fitted_mmm, frequency, counterfactual_spend_factor
    ):
        """Mediated increments match a per-channel posterior-predictive oracle.

        The oracle re-runs ``sample_posterior_predictive`` on the response scale
        for every (period, channel) counterfactual and never forms a
        linear-predictor difference, so agreement checks the mediated algebra,
        the widened carryover window and the windowing of the effect's own
        ``lf_budget`` data all at once.
        """
        mmm = funnel_log_link_fitted_mmm

        result = mmm.incrementality.compute_incremental_contribution(
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
        )
        expected = compute_log_link_ground_truth_by_period(
            mmm,
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            l_max=effective_l_max(mmm),
        )

        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_matches_oracle_under_identity_link(self, funnel_identity_fitted_mmm):
        """The mediated path is included under an additive link too.

        Under ``link="identity"`` the increment is ``target_scale`` times the
        summed change in the linear predictor, so the oracle can read the
        mediated part straight off the effect's own contribution deterministic --
        an independent route to the same number.
        """
        mmm = funnel_identity_fitted_mmm

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time"
        )
        expected = mediated_identity_oracle(
            mmm, "funnel_effect_contribution", l_max=effective_l_max(mmm)
        )

        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_joint_matches_oracle(self, funnel_log_link_fitted_mmm):
        """The joint estimand matches an all-channels-at-once oracle."""
        mmm = funnel_log_link_fitted_mmm

        result = mmm.incrementality.compute_joint_incremental_contribution(
            frequency="all_time"
        )
        expected = compute_log_link_ground_truth_by_period(
            mmm,
            frequency="all_time",
            l_max=effective_l_max(mmm),
            joint=True,
        )

        assert "channel" not in result.dims
        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_leave_one_out_overcounts_the_joint(self, funnel_log_link_fitted_mmm):
        """Per-channel increments do not sum to the joint, and are labelled so.

        Two sources of non-additivity compound here: the log link, and a funnel
        that mixes channels inside a saturating transform.  The per-channel
        numbers are leave-one-out, so interaction mass is counted by every
        channel that touches it and the sum exceeds the joint.  Asserting the
        direction rather than a value keeps this a statement about the estimand.
        """
        incr = funnel_log_link_fitted_mmm.incrementality

        loo = incr.compute_incremental_contribution(frequency="all_time")
        joint = incr.compute_joint_incremental_contribution(frequency="all_time")

        loo_total = float(loo.sum("channel").mean(("chain", "draw")))
        joint_total = float(joint.mean(("chain", "draw")))

        assert joint_total > 0
        assert loo_total > joint_total

    def test_joint_equals_channel_sum_for_an_additive_model(self, simple_fitted_mmm):
        """With an additive link and no mediation the two estimands coincide.

        This is the boundary case the paper's :math:`\\text{ROAS}_m` derivation
        assumes, and the reason the joint estimand is not interesting there.
        """
        incr = simple_fitted_mmm.incrementality

        loo = incr.compute_incremental_contribution(frequency="all_time")
        joint = incr.compute_joint_incremental_contribution(frequency="all_time")

        xr.testing.assert_allclose(loo.sum("channel"), joint, rtol=1e-10)

    def test_ignoring_the_mediated_path_understates_the_increment(
        self, funnel_log_link_fitted_mmm, monkeypatch
    ):
        """The mediated part is material, not a rounding correction.

        Reproduces the pre-fix behaviour by resolving no effects at all, which is
        exactly what the module used to do, and checks the resulting number is
        materially smaller.  Without a magnitude assertion, a regression that
        quietly dropped the effect again would still pass every other test here.

        Dropping the effect also has to be *caught*, so the completeness guard
        is checked to fire before being disabled to take the measurement.
        """
        incr = funnel_log_link_fitted_mmm.incrementality
        full = incr.compute_incremental_contribution(frequency="all_time")

        monkeypatch.setattr(
            incrementality_module, "resolve_channel_dependent_effects", lambda model: ()
        )
        with pytest.raises(NotImplementedError, match="Some path from spend"):
            incr.compute_incremental_contribution(frequency="all_time")

        monkeypatch.setattr(
            SpendProbe,
            "assert_increment_is_complete",
            lambda self, **kwargs: None,
        )
        direct_only = incr.compute_incremental_contribution(frequency="all_time")

        full_total = float(full.sum("channel").mean(("chain", "draw")))
        direct_total = float(direct_only.sum("channel").mean(("chain", "draw")))

        assert full_total > 0
        # The mediated path carries a double-digit share for one of the channels
        # in this fixture.  How large a share is a property of the priors, so the
        # bound is deliberately loose; what it rules out is the path vanishing.
        assert direct_total < 0.95 * full_total

    def test_effect_that_ignores_spend_is_left_alone(self, trend_effect_fitted_mmm):
        """A ``mu_effect`` the counterfactual cannot reach needs no opt-in.

        A linear trend does not read ``channel_data``, so it cancels in the
        difference.  It must be skipped without ever consulting its spec --
        otherwise every effect that already exists would suddenly have to opt in.
        """
        incr = trend_effect_fitted_mmm.incrementality

        assert resolve_channel_dependent_effects(incr.model) == ()
        # And the computation still runs, on the separable path.
        result = incr.compute_incremental_contribution(frequency="all_time")
        assert set(result.dims) == {"chain", "draw", "channel"}

    def test_effect_without_a_spec_raises(self, funnel_not_opted_in_fitted_mmm):
        """A spend-dependent effect that has not opted in is refused."""
        incr = funnel_not_opted_in_fitted_mmm.incrementality

        with pytest.raises(NotImplementedError, match="incrementality_spec"):
            incr.compute_incremental_contribution(frequency="all_time")

    def test_a_duck_typed_baseline_effect_still_works(
        self, duck_typed_baseline_effect_fitted_mmm
    ):
        """An effect that is not a ``MuEffect`` does not break incrementality.

        The additive-effect module documents duck typing as a supported way to
        write an effect, so such an effect has no ``contribution_var_name`` to
        read.  It is also not reached by a spend counterfactual here, which
        makes asking it for one pointless.  Demanding the attribute up front
        would turn every model owning such an effect into a crash.
        """
        incr = duck_typed_baseline_effect_fitted_mmm.incrementality

        assert resolve_channel_dependent_effects(incr.model) == ()
        result = incr.compute_incremental_contribution(frequency="all_time")
        assert set(result.dims) == {"chain", "draw", "channel"}

    def test_a_duck_typed_spend_effect_is_refused(
        self, duck_typed_spend_effect_fitted_mmm
    ):
        """An unattributable effect that spend reaches is caught, not ignored.

        The other side of the previous test: being unable to name the effect's
        contribution is only harmless while the counterfactual cannot reach it.
        Here it can, so the increment would be missing a real path -- and the
        completeness guard is what notices, since effect resolution deliberately
        said nothing.
        """
        incr = duck_typed_spend_effect_fitted_mmm.incrementality

        with pytest.raises(NotImplementedError, match="Some path from spend"):
            incr.compute_incremental_contribution(frequency="all_time")

    def test_a_misreported_contribution_is_refused(self, funnel_misreported_fitted_mmm):
        """Naming the wrong contribution variable does not pass unnoticed.

        The effect names ``funnel_demand``: a real variable, spend-dependent,
        and not the term it adds to the linear predictor.  Evaluating it in
        place of the true contribution would leave the mediated path out of the
        increment while looking entirely well-formed.
        """
        incr = funnel_misreported_fitted_mmm.incrementality

        with pytest.raises(NotImplementedError, match="Some path from spend"):
            incr.compute_incremental_contribution(frequency="all_time")

    def test_a_global_effect_selects_full_axis_evaluation(
        self, global_normalization_fitted_mmm
    ):
        """An effect reading the whole series is evaluated on the whole series.

        ``spend / spend.mean("date")`` has no bounded reach to widen a window
        by: perturbing one date moves *every* date, including earlier ones,
        because the denominator is a reduction over the axis.  Evaluating it on
        a window would silently compute a different function -- the mean of the
        window rather than of the series -- so the probe has to notice and
        switch to the full axis.

        A monthly frequency is what gives the test teeth: at ``all_time`` the
        window is already the whole axis and the distinction cannot fail.
        """
        mmm = global_normalization_fitted_mmm

        assert measure_reach(mmm)["global_effect_contribution"].requires_full_axis

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="monthly"
        )
        expected = compute_log_link_ground_truth_by_period(mmm, frequency="monthly")

        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_a_global_effect_is_exactly_zero_at_unit_factor(
        self, global_normalization_fitted_mmm
    ):
        """Not perturbing spend produces no increment, to the last bit.

        Under full-axis evaluation the counterfactual array at ``alpha = 1.0``
        is the baseline array, so the two evaluations are the same computation
        on the same inputs and the difference is exactly zero -- no tolerance
        needed.  A windowing bug would show up here as a small non-zero number,
        which is the shape a truncation error takes.
        """
        result = global_normalization_fitted_mmm.incrementality.compute_incremental_contribution(
            frequency="monthly", counterfactual_spend_factor=1.0
        )

        np.testing.assert_array_equal(result.values, np.zeros_like(result.values))

    def test_under_declared_carryover_is_refused(
        self, funnel_under_declared_fitted_mmm
    ):
        """Declaring less reach than the effect has raises, with both numbers.

        This is the quiet failure the measurement exists for: the window would
        be sized for the direct path, the mediated tail would fall outside it,
        and the increment would come back low with nothing to indicate anything
        had been cut.
        """
        incr = funnel_under_declared_fitted_mmm.incrementality

        with pytest.raises(ValueError, match="additional_carryover_lags=0"):
            incr.compute_incremental_contribution(frequency="all_time")

    def test_measured_reach_is_covered_by_the_declaration(
        self, funnel_log_link_fitted_mmm
    ):
        """The fixture declares at least what the probe measures.

        The oracle these tests compare against builds its window from the
        declaration, so a measurement exceeding it would make the module and the
        oracle disagree about the window rather than about the algebra.
        """
        mmm = funnel_log_link_fitted_mmm
        effects = resolve_channel_dependent_effects(mmm)
        measured = measure_reach(mmm)["funnel_effect_contribution"]

        assert not measured.requires_full_axis
        # Two chained adstocks of l_max each reach 2 * (l_max - 1) lags, of
        # which the model's own window already covers l_max.
        assert measured.additional_carryover_lags == mmm.adstock.l_max - 2
        assert measured.additional_carryover_lags <= effects[0].declared_carryover_lags

    def test_effective_l_max_widens_the_window(self, funnel_log_link_fitted_mmm):
        """A mediated path that chains a second adstock lengthens the window."""
        mmm = funnel_log_link_fitted_mmm
        effects = resolve_channel_dependent_effects(mmm)

        assert len(effects) == 1
        assert effects[0].contribution_var == "funnel_effect_contribution"
        extra = mmm.mu_effects[0].demand_transform.adstock.l_max
        assert effective_l_max(mmm) == mmm.adstock.l_max + extra

    def test_the_window_length_is_a_known_number(self, funnel_log_link_fitted_mmm):
        """The window the oracle sums over, written down rather than derived.

        Every other mediated test takes its window from ``effective_l_max``, which
        reads it back off the module.  Narrow the module's own calculation and both
        sides narrow together, so a dozen oracle comparisons keep passing while the
        mediated tail is quietly cut off.  One literal is what breaks that tie.
        """
        mmm = funnel_log_link_fitted_mmm

        assert mmm.adstock.l_max == 3
        assert mmm.mu_effects[0].demand_transform.adstock.l_max == 3
        assert effective_l_max(mmm) == 6

    def test_a_declared_full_axis_mode_is_honoured(self, funnel_full_axis_fitted_mmm):
        """``evaluation_mode="full"`` overrides a measurement that says otherwise.

        The effect's reach is bounded and the probe says so, so ``"auto"`` would
        give it a window.  An author may still know of a dependence on the whole
        series that one perturbation cannot reveal, and honouring the declaration
        costs compute rather than correctness -- which the oracle then confirms.
        """
        mmm = funnel_full_axis_fitted_mmm

        reach = measure_spend_reach(mmm)
        assert not reach.measured["funnel_effect_contribution"].requires_full_axis
        assert reach.requires_full_axis

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="monthly"
        )
        expected = compute_log_link_ground_truth_by_period(
            mmm, frequency="monthly", l_max=effective_l_max(mmm)
        )
        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_declaring_a_window_for_a_date_reduction_is_refused(
        self, windowed_global_normalization_fitted_mmm
    ):
        """``evaluation_mode="window"`` against a measured full-axis need raises.

        The mirror image of an under-declared carryover: the declaration is
        falsifiable, the probe falsifies it, and honouring it would evaluate the
        mean of the window in place of the mean of the series.
        """
        incr = windowed_global_normalization_fitted_mmm.incrementality

        with pytest.raises(ValueError, match="evaluation_mode='window'"):
            incr.compute_incremental_contribution(frequency="monthly")

    def test_a_dark_start_still_measures_the_mediated_reach(
        self, dark_start_funnel_fitted_mmm
    ):
        """A flighted campaign does not disable the probe.

        The end-to-end form of :class:`TestSpendProbe`'s blocker: with no spend
        before the tenth date, a probe placed a quarter of the way along the axis
        perturbs an all-zero row, measures nothing, and sizes the window from the
        model's own adstock alone.  The mediated tail then falls outside it and the
        increment comes back low with nothing to indicate anything was cut, so the
        oracle -- which sums over the *right* window -- is what catches it.
        """
        mmm = dark_start_funnel_fitted_mmm
        spend = mmm.data.get_channel_data().values
        assert not spend[len(spend) // 4].any()  # guards the premise

        assert effective_l_max(mmm) == 6
        result = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time"
        )
        expected = compute_log_link_ground_truth_by_period(
            mmm, frequency="all_time", l_max=6
        )

        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_a_mediator_reading_a_dark_channel_is_probed_on_its_own_dates(
        self, dark_probe_funnel_fitted_mmm, partial_channel_funnel_fitted_mmm
    ):
        """A mediated channel dark at the probed date still gets its tail.

        An A/B on one graph: the mediator reads ``channel_1`` alone and declares
        nothing, so the window has only the measurement to go on.  In the control
        every date carries spend on both channels; in the treatment the channels'
        spend supports are disjoint, and a probe placed by total spend lands
        where ``channel_1`` is dark -- the mediator then came back bit-identical,
        its measured reach collapsed to nothing, and the window was sized from
        the model's own adstock alone, silently understating the increment.  The
        two arms have to agree: probing per channel when no date covers them all
        is what makes them.
        """
        control = partial_channel_funnel_fitted_mmm
        treatment = dark_probe_funnel_fitted_mmm

        # Guards the premise: no date moves both channels, and the head of the
        # axis -- where a single probe would land -- is dark on the mediated one.
        spend = treatment.data.get_channel_data().values
        assert not (spend[:, 0] * spend[:, 1]).any()
        assert not spend[: len(spend) // 8, 0].any()

        reach_control = measure_spend_reach(control)
        reach_treatment = measure_spend_reach(treatment)
        mediator = "funnel_effect_contribution"

        assert reach_control.measured[mediator] != TemporalReach.none()
        assert reach_treatment.measured[mediator] == reach_control.measured[mediator]
        assert reach_treatment.effective_l_max == reach_control.effective_l_max
        assert not reach_treatment.requires_full_axis

        result = treatment.incrementality.compute_incremental_contribution(
            frequency="all_time"
        )
        expected = compute_log_link_ground_truth_by_period(
            treatment, frequency="all_time", l_max=reach_treatment.effective_l_max
        )

        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_evaluator_discovers_the_effect_s_own_data(
        self, funnel_log_link_fitted_mmm
    ):
        """Date-indexed inputs an effect reads are found in the graph, not declared.

        ``lf_budget`` has to be cut to the same window as spend or the graph is
        handed inputs of two different lengths.  Discovery is by traversal so an
        effect cannot forget to mention one; ``channel_data`` and ``time_index``
        are handled separately and must not be double-counted.
        """
        mmm = funnel_log_link_fitted_mmm
        incr = mmm.incrementality
        effects = resolve_channel_dependent_effects(incr.model)

        evaluator = CounterfactualEvaluator(
            pymc_model=mmm.model,
            posterior=incr.idata.posterior.dataset,
            response_vars=["channel_contribution", effects[0].contribution_var],
            frozen_deterministics=mmm.frozen_deterministics,
            dates=incr.data.dates,
        )

        assert evaluator.windowed_data_vars == ("lf_budget",)
        assert evaluator.non_date_dims["channel_contribution"] == ("channel",)
        assert evaluator.non_date_dims["funnel_effect_contribution"] == ()

    def test_cutting_an_input_matches_the_spend_window(self, incrementality_lite):
        """An auxiliary input is cut position for position like spend is."""
        incr, l_max = incrementality_lite
        dates = incr.data.dates
        freq_offset = pd.tseries.frequencies.to_offset(pd.infer_freq(dates))
        periods = incr._create_period_groups(dates[0], dates[-1], "original")
        windows = EvaluationWindows.build(
            periods=periods, dates=dates, l_max=l_max, freq_offset=freq_offset
        )
        values = np.arange(len(dates), dtype="float64") + 1.0

        cut = windows.cut(values, dtype="float64")

        assert cut.shape == (len(periods), windows.max_window)
        # Fitted values start at position zero, in order.
        window = windows.windows[0]
        np.testing.assert_allclose(cut[0, : window.n_actual], values[window.in_window])
        # Anything past the window's own length is padding, and is zero.
        assert (cut[0, window.n_actual :] == 0).all()

    @staticmethod
    def _scenarios(incrementality_lite, *, channel_axis, estimand, frequency):
        """Build scenarios directly, bypassing the compiled evaluation."""
        incr, l_max = incrementality_lite
        dates = incr.data.dates
        freq_offset = pd.tseries.frequencies.to_offset(pd.infer_freq(dates))
        baseline_array = incr.data.get_channel_spend().values
        periods = incr._create_period_groups(dates[0], dates[-1], frequency)
        windows = EvaluationWindows.build(
            periods=periods, dates=dates, l_max=l_max, freq_offset=freq_offset
        )
        return periods, windows.build_scenarios(
            baseline_array=baseline_array,
            counterfactual_spend_factor=0.0,
            dtype="float64",
            channel_axis=channel_axis,
            n_channels=baseline_array.shape[-1],
            estimand=estimand,
        )

    def test_per_channel_scenarios_perturb_one_channel_each(self, incrementality_lite):
        """In per-channel mode each scenario zeroes exactly one channel."""
        n_channels = incrementality_lite[0].data.get_channel_spend().shape[-1]
        _, scenarios = self._scenarios(
            incrementality_lite,
            channel_axis=0,
            estimand="per_channel",
            frequency="all_time",
        )

        assert scenarios.spend.shape[0] == n_channels
        for channel in range(n_channels):
            spend = scenarios.spend[scenarios.rows[(0, channel)]]
            assert (spend[:, channel] == 0).all()
            others = [c for c in range(n_channels) if c != channel]
            assert (spend[:, others] != 0).any()

    @pytest.mark.parametrize("frequency", ["all_time", "monthly"])
    @pytest.mark.parametrize(
        ("channel_axis", "estimand", "per_period"),
        [
            (None, "per_channel", 1),
            (None, "joint", 1),
            (0, "per_channel", 2),
            (0, "joint", 1),
        ],
    )
    def test_only_the_scenarios_that_are_read_are_built(
        self, incrementality_lite, frequency, channel_axis, estimand, per_period
    ):
        """Scenario counts are pinned, per estimand and per mode.

        Everything downstream is proportional to this count -- the perturbed
        spend arrays, the compiled evaluation, the memory it holds -- so a
        mediated model asked for the joint number must not build the
        per-channel rows it will never read.  Separable models need one
        all-channels row either way, since each channel's column can be read
        straight off it.
        """
        periods, scenarios = self._scenarios(
            incrementality_lite,
            channel_axis=channel_axis,
            estimand=estimand,
            frequency=frequency,
        )

        assert scenarios.spend.shape[0] == per_period * len(periods)
        # And whatever the estimand asks to read is addressable.
        keys = (
            range(2)
            if estimand == "per_channel" and channel_axis is not None
            else [None]
        )
        for period_idx in range(len(periods)):
            for key in keys:
                assert (period_idx, key) in scenarios.rows

    def test_the_joint_scenario_zeroes_every_channel(self, incrementality_lite):
        """The all-channels perturbation is exactly that."""
        _, scenarios = self._scenarios(
            incrementality_lite,
            channel_axis=0,
            estimand="joint",
            frequency="all_time",
        )

        assert (scenarios.spend[scenarios.rows[(0, None)]] == 0).all()


class TestMediatedEdgeCases:
    """Mediation crossed with the settings that change how a window is built.

    Every test in :class:`TestMediatedMuEffects` uses the same funnel fixture at
    its default settings, so the mediated path and the ordinary machinery are
    only ever exercised in one combination.  These are the combinations where
    the two interact: the window's length, its position on the date axis, the
    layout of the array being perturbed, and how many effects have to be
    collected.
    """

    def test_the_window_comes_from_the_mediator_when_the_model_carries_nothing(
        self, funnel_instantaneous_adstock_fitted_mmm
    ):
        """With an instantaneous adstock, the mediated path sets the window alone.

        Every other mediated test has the model's own adstock to fall back on,
        so a bug that widened the window by the model's ``l_max`` instead of the
        mediator's would still land inside the right ballpark.  Here the model's
        adstock reaches lag zero only and the mediator has to supply the rest.
        """
        mmm = funnel_instantaneous_adstock_fitted_mmm

        assert mmm.adstock.l_max == 1
        l_max = effective_l_max(mmm)
        assert l_max > mmm.adstock.l_max

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="monthly"
        )
        expected = compute_log_link_ground_truth_by_period(
            mmm, frequency="monthly", l_max=l_max
        )

        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_carryover_can_be_excluded_under_mediation(
        self, funnel_log_link_fitted_mmm
    ):
        """``include_carryover=False`` still evaluates on the widened window.

        The two lengths do different jobs and must not be conflated: the window
        is sized for the longest path spend can take, so the graph sees the
        right inputs, while the *evaluation mask* is what ``include_carryover``
        narrows to the period itself.  Collapsing the window along with the mask
        would change the numbers inside the period too.
        """
        mmm = funnel_log_link_fitted_mmm

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="monthly", include_carryover=False
        )
        expected = compute_log_link_ground_truth_by_period(
            mmm,
            frequency="monthly",
            include_carryover=False,
            l_max=effective_l_max(mmm),
        )

        xr.testing.assert_allclose(result, expected, rtol=1e-6)
        # And excluding the tail really does leave something out.
        with_carryover = mmm.incrementality.compute_incremental_contribution(
            frequency="monthly"
        )
        assert float(result.sum()) < float(with_carryover.sum())

    def test_a_time_varying_multiplier_survives_the_widened_window(
        self, funnel_time_varying_media_fitted_mmm
    ):
        """Mediation and ``time_index`` are consistent about where a window sits.

        The HSGP multiplier evaluates its basis at the window's absolute
        positions on the date axis, and mediation is what makes that window
        wider than the model's own adstock implies.  Get the positions from the
        un-widened window and the multiplier is read at the wrong dates.
        """
        mmm = funnel_time_varying_media_fitted_mmm

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="monthly"
        )
        expected = compute_log_link_ground_truth_by_period(
            mmm, frequency="monthly", l_max=effective_l_max(mmm)
        )

        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_two_mediators_are_both_collected_and_the_longer_sets_the_window(
        self, two_mediators_fitted_mmm
    ):
        """Two effects of different reach: both included, the wider one governs.

        Taking the first effect's reach, or averaging the two, would truncate
        the slower mediator's tail; summing only one contribution would drop a
        real path.  A single-effect fixture cannot tell any of these apart.
        """
        mmm = two_mediators_fitted_mmm
        effects = resolve_channel_dependent_effects(mmm)

        assert {effect.contribution_var for effect in effects} == {
            "funnel_effect_contribution",
            "slow_effect_contribution",
        }
        declared = [effect.declared_carryover_lags for effect in effects]
        assert effective_l_max(mmm) == mmm.adstock.l_max + max(declared)

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time"
        )
        expected = compute_log_link_ground_truth_by_period(
            mmm, frequency="all_time", l_max=effective_l_max(mmm)
        )

        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_a_panel_model_perturbs_the_channel_axis_not_the_first_one(
        self, panel_mediated_fitted_mmm
    ):
        """Mediated panel models index ``channel`` where it actually is.

        ``channel_data`` is ``(date, country, channel)`` here, so the
        per-channel scenarios have to perturb the last axis.  Perturbing axis
        zero of the non-date axes would silently zero a *country* instead, and
        with two of each the shapes would agree throughout.

        The effect's contribution also drops the panel dimension, which the
        increment has to broadcast back rather than refuse.
        """
        mmm = panel_mediated_fitted_mmm
        incr = mmm.incrementality

        (effect,) = resolve_channel_dependent_effects(incr.model)
        assert effect.contribution_var == "global_effect_contribution"

        result = incr.compute_incremental_contribution(frequency="all_time")
        expected = mediated_identity_oracle(mmm, effect.contribution_var)

        assert set(result.dims) == {"chain", "draw", "channel", "country"}
        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_a_panel_log_link_model_with_a_mediator_matches_the_oracle(
        self, panel_mediated_log_link_fitted_mmm
    ):
        """Panel layout, log link and mediation combine without shortcuts.

        ``channel_data`` is ``(date, country, channel)``, the effect's
        contribution drops the panel dimension, and the log-link reducer weights
        the perturbation with a ``(date, country)`` baseline response.  The
        identity-link panel test and the non-mediated log-link panel test each
        pass with one of those axes confused in a way the other cancels; the
        combination is what pins all three down at once.
        """
        mmm = panel_mediated_log_link_fitted_mmm

        result = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time"
        )
        expected = compute_log_link_ground_truth_by_period(
            mmm, frequency="all_time", l_max=effective_l_max(mmm)
        )

        assert set(result.dims) == {"chain", "draw", "channel", "country"}
        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_subsampling_the_posterior_keeps_the_mediated_path(
        self, funnel_log_link_fitted_mmm
    ):
        """``num_samples`` selects draws for every evaluated node alike.

        The mediated increment reads several nodes out of one compiled
        evaluation, so a subsample that applied to ``channel_contribution`` but
        not to the effect would misalign the two and produce a difference of
        unrelated draws -- a wrong number, not an error.
        """
        incr = funnel_log_link_fitted_mmm.incrementality

        result = incr.compute_incremental_contribution(
            frequency="all_time", num_samples=10, random_state=20260804
        )

        assert result.sizes["draw"] * result.sizes["chain"] == 10
        # And it is still the mediated number, not the direct one: rerunning on
        # the same subsample is deterministic, so the two agree exactly.
        again = incr.compute_incremental_contribution(
            frequency="all_time", num_samples=10, random_state=20260804
        )
        xr.testing.assert_allclose(result, again, rtol=0)

    @pytest.mark.parametrize("counterfactual_spend_factor", [0.0, 0.5, 1.0, 1.01])
    @pytest.mark.parametrize("estimand", ["per_channel", "joint"])
    def test_every_factor_and_estimand_matches_the_oracle(
        self, funnel_log_link_fitted_mmm, counterfactual_spend_factor, estimand
    ):
        """The whole factor grid, on both estimands, under mediation.

        ``0.0`` is the leave-one-out case, ``0.5`` a halving, ``1.01`` the
        marginal perturbation, and ``1.0`` no intervention at all -- which must
        come back as exactly zero rather than as accumulated float noise, since
        the counterfactual array is then the baseline array.
        """
        mmm = funnel_log_link_fitted_mmm
        incr = mmm.incrementality
        compute = (
            incr.compute_incremental_contribution
            if estimand == "per_channel"
            else incr.compute_joint_incremental_contribution
        )

        result = compute(
            frequency="all_time",
            counterfactual_spend_factor=counterfactual_spend_factor,
        )

        if counterfactual_spend_factor == 1.0:
            np.testing.assert_array_equal(result.values, np.zeros_like(result.values))
            return

        expected = compute_log_link_ground_truth_by_period(
            mmm,
            frequency="all_time",
            counterfactual_spend_factor=counterfactual_spend_factor,
            l_max=effective_l_max(mmm),
            joint=estimand == "joint",
        )
        xr.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_the_gap_between_the_estimands_has_no_fixed_direction(
        self, negative_effect_fitted_mmm
    ):
        """Summed per-channel increments can fall *below* the joint one.

        With strictly positive contributions the per-channel numbers overcount,
        because interaction mass is claimed by every channel that touches it --
        which is the case the module docstring used to state as though it were
        general.  A mediated term that subtracts from the response reverses the
        interaction, and the sum lands under the joint number instead.  The
        estimands differ by an interaction, not by a bias with a known sign.
        """
        incr = negative_effect_fitted_mmm.incrementality

        per_channel = incr.compute_incremental_contribution(frequency="all_time")
        joint = incr.compute_joint_incremental_contribution(frequency="all_time")

        summed = float(per_channel.sum("channel").mean(("chain", "draw")))
        joint_total = float(joint.mean(("chain", "draw")))

        assert summed < joint_total


class TestIncrementalitySpecValidation:
    """What the extension point accepts, and what it refuses.

    ``IncrementalitySpec`` is the only thing a third-party effect has to write
    to take part in incrementality, and its ``additional_carryover_lags`` sizes
    an evaluation window.  Accepting ``True`` or ``1.5`` there would produce a
    window of a nonsensical length rather than a message saying so.
    """

    @pytest.mark.parametrize("bad", [-1, -10])
    def test_a_negative_window_is_refused(self, bad):
        """A negative carryover length is a mistake, not a shorter window."""
        with pytest.raises(ValidationError, match="additional_carryover_lags"):
            IncrementalitySpec(additional_carryover_lags=bad)

    @pytest.mark.parametrize("bad", [1.5, 2.0, True, "3"])
    def test_a_non_integer_window_is_refused(self, bad):
        """Only integers size a window; a float or a bool is a typo."""
        with pytest.raises(ValidationError, match="additional_carryover_lags"):
            IncrementalitySpec(additional_carryover_lags=bad)

    @pytest.mark.parametrize("good", [0, 7, np.int64(4)])
    def test_an_integer_window_is_accepted(self, good):
        """Zero is meaningful, and a NumPy integer is still an integer.

        ``0`` says "this effect adds no carryover of its own", which the
        reconciliation then checks against the probe.  NumPy integers arrive
        wherever a caller writes ``spec.l_max`` of something array-shaped.
        """
        spec = IncrementalitySpec(additional_carryover_lags=good)

        assert spec.additional_carryover_lags == int(good)

    def test_the_default_leaves_everything_to_be_measured(self):
        """An empty spec opts in and declares nothing."""
        spec = IncrementalitySpec()

        assert spec.additional_carryover_lags is None
        assert spec.evaluation_mode == "auto"

    def test_an_unknown_field_is_refused(self):
        """A misspelled field is a silent no-op unless the model forbids it."""
        with pytest.raises(ValidationError, match="carryover_lags"):
            IncrementalitySpec(carryover_lags=3)

    @pytest.mark.parametrize("bad", ["windowed", "partial", ""])
    def test_an_unknown_evaluation_mode_is_refused(self, bad):
        """The mode is a closed set; anything else would be read as ``auto``."""
        with pytest.raises(ValidationError, match="evaluation_mode"):
            IncrementalitySpec(evaluation_mode=bad)

    def test_the_spec_is_immutable(self):
        """An effect's declaration cannot be edited after it is read."""
        spec = IncrementalitySpec(additional_carryover_lags=2)

        with pytest.raises(ValidationError):
            spec.additional_carryover_lags = 5
