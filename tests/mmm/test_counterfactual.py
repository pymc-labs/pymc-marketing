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
"""Tests for the counterfactual evaluation machinery: windows, scenarios, inputs."""

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytest
import xarray as xr

from pymc_marketing.mmm.counterfactual import (
    CounterfactualEvaluator,
    CounterfactualScenarios,
    EvaluationWindows,
    PeriodWindow,
)


def build_aux_dims_model(aux_dims, n_dates, n_country):
    """A minimal model whose auxiliary date-indexed data has *aux_dims*.

    Hand-built rather than fitted, because what is under test is which axis of
    an input gets cut, and an MMM cannot be asked to lay its mediator's data out
    ``("country", "date")`` on request.  Small enough that the whole permutation
    matrix is cheap.
    """
    sizes = {"date": n_dates, "country": n_country}
    coords = {
        "date": list(range(n_dates)),
        "country": [f"country_{i}" for i in range(n_country)],
        "channel": ["channel_1"],
    }
    # One set of values, laid out as asked.  Distinct everywhere, so cutting the
    # wrong axis cannot coincide with cutting the right one, and a transpose of
    # the same numbers rather than different numbers, so two layouts are
    # genuinely comparable.
    canonical = xr.DataArray(
        np.arange(sizes["date"] * sizes["country"], dtype="float64").reshape(
            sizes["date"], sizes["country"]
        ),
        dims=("date", "country"),
    )
    if "country" not in aux_dims:
        canonical = canonical.isel(country=0)
    aux_values = canonical.transpose(*aux_dims).values

    with pm.Model(coords=coords) as model:
        channel_data = pmd.Data(
            "channel_data",
            np.ones((n_dates, n_country, 1)),
            dims=("date", "country", "channel"),
        )
        aux = pmd.Data("aux_input", aux_values, dims=aux_dims)
        beta = pmd.Normal("beta", mu=0.0, sigma=1.0)
        alpha = pmd.Normal("alpha", mu=0.0, sigma=1.0)
        pmd.Deterministic("channel_contribution", beta * channel_data + aux + alpha)

    posterior = xr.Dataset(
        {
            "beta": (("chain", "draw"), np.array([[1.0, 2.0]])),
            "alpha": (("chain", "draw"), np.array([[0.5, -0.5]])),
        },
        coords={"chain": [0], "draw": [0, 1]},
    )
    return model, posterior, aux_values


class TestDateIndexedInputs:
    """Auxiliary inputs are cut along the axis they call ``date``.

    Discovery accepts any ``pm.Data`` whose dimensions include ``date``, and an
    effect is free to declare its data ``("country", "date")``.  Cutting axis
    zero regardless either raises a misleading length error or -- when the panel
    happens to be as wide as the calendar is long -- quietly windows the wrong
    axis and returns numbers that look entirely reasonable.
    """

    dates = pd.date_range("2023-01-02", freq="W-MON", periods=6)

    def _windows(self, *slices, max_window):
        """Build windows over ``self.dates`` directly, one per given slice."""
        windows = []
        for window_slice in slices:
            in_window = np.zeros(len(self.dates), dtype=bool)
            in_window[window_slice] = True
            actual_dates = self.dates[in_window]
            windows.append(
                PeriodWindow.build(
                    start=actual_dates[0],
                    end=actual_dates[-1],
                    dates=self.dates,
                    in_window=in_window,
                    eval_end=actual_dates[-1],
                )
            )
        return EvaluationWindows(
            windows=windows, max_window=max_window, dates=self.dates
        )

    def _evaluator(self, aux_dims, n_country, dates=None):
        model, posterior, aux_values = build_aux_dims_model(
            aux_dims, n_dates=len(self.dates), n_country=n_country
        )
        evaluator = CounterfactualEvaluator(
            pymc_model=model,
            posterior=posterior,
            response_vars=["channel_contribution"],
            frozen_deterministics=[],
            dates=self.dates if dates is None else dates,
        )
        return evaluator, aux_values

    @pytest.mark.parametrize(
        "aux_dims", [("date",), ("date", "country"), ("country", "date")]
    )
    def test_the_date_axis_is_located_by_name(self, aux_dims):
        """Whatever the layout, the discovered input knows where ``date`` is."""
        evaluator, _ = self._evaluator(aux_dims, n_country=2)

        (aux,) = evaluator._aux
        assert aux.dims == aux_dims
        assert aux.date_axis == aux_dims.index("date")

    @pytest.mark.parametrize("n_country", [2, 6])
    def test_cutting_takes_the_date_axis_and_leaves_the_layout(self, n_country):
        """A ``("country", "date")`` input is cut along ``date``, not ``country``.

        ``n_country=6`` is the trap: the panel is exactly as wide as the
        calendar is long, so cutting axis zero raises nothing and returns an
        array of the right shape filled with the wrong numbers.
        """
        evaluator, aux_values = self._evaluator(("country", "date"), n_country)
        windows = self._windows(slice(0, 3), max_window=4)

        (aux,) = evaluator._aux
        cut = aux.cut(windows)

        # (period, country, date), with date cut to max_window and padded.
        assert cut.shape == (1, n_country, 4)
        np.testing.assert_allclose(cut[0, :, :3], aux_values[:, :3])
        assert (cut[0, :, 3:] == 0).all()

    def test_the_evaluation_does_not_depend_on_the_layout(self):
        """The same data in two layouts evaluates to the same counterfactual.

        The end-to-end statement the axis bookkeeping exists for: transposing an
        effect's input is a change of notation, and notation must not move the
        numbers.
        """
        windows = self._windows(slice(0, 4), max_window=4)
        scenarios = CounterfactualScenarios(
            spend=np.ones((1, 4, 2, 1)),
            period_index=np.array([0]),
            rows={(0, None): 0},
        )

        results = []
        for aux_dims in [("date", "country"), ("country", "date")]:
            evaluator, _ = self._evaluator(aux_dims, n_country=2)
            results.append(
                evaluator.evaluate_counterfactual(scenarios, windows=windows)[
                    "channel_contribution"
                ]
            )

        np.testing.assert_allclose(*results)

    def test_batching_does_not_change_the_result(self):
        """Splitting the scenarios across evaluations returns the same numbers.

        Batching exists to bound the working set of a single evaluation, so it
        has to be invisible in the output -- including the ordering, since
        :attr:`CounterfactualScenarios.rows` addresses results by position.
        Exact equality, not a tolerance: the same graph on the same inputs.
        """
        evaluator, _ = self._evaluator(("date", "country"), n_country=2)
        windows = self._windows(slice(0, 4), slice(2, 6), max_window=4)
        rng = np.random.default_rng(20260804)
        scenarios = CounterfactualScenarios(
            spend=rng.normal(size=(5, 4, 2, 1)),
            period_index=np.array([0, 0, 1, 1, 1]),
            rows={(0, None): 1, (1, None): 4},
        )
        one_shot = evaluator.evaluate_counterfactual(
            scenarios, windows=windows, batch_size=len(scenarios.spend)
        )
        # A batch size that does not divide the scenario count, so the last
        # batch is short and any off-by-one in the bookkeeping shows.
        batched = evaluator.evaluate_counterfactual(
            scenarios, windows=windows, batch_size=2
        )

        assert one_shot.keys() == batched.keys()
        for name, values in one_shot.items():
            np.testing.assert_array_equal(values, batched[name])

    def test_the_batch_size_falls_back_to_one_scenario(self):
        """An output too large for the budget still gets evaluated, alone."""
        evaluator, _ = self._evaluator(("date", "country"), n_country=2)

        assert evaluator._batch_size(n_scenarios=10, max_window=10**9) == 1
        # And a small problem is never split for no reason.
        assert evaluator._batch_size(n_scenarios=10, max_window=4) == 10

    def test_an_input_that_does_not_span_the_dates_is_refused(self):
        """A date dimension of the wrong length is an error, and says which.

        The length is checked on the ``date`` axis rather than on axis zero, so
        the message names the real problem instead of reporting a ``country``
        count as a row count.
        """
        with pytest.raises(ValueError, match="entries along its 'date' dimension"):
            self._evaluator(("country", "date"), n_country=2, dates=self.dates[:4])


class TestPeriodEdgeCases:
    """Periods that are degenerate, and the windows they produce.

    A period is a request, not a fact about the data: it can name dates the
    model never saw, or name exactly one.  Both have to survive the window
    machinery, which stacks periods into a rectangular array and would divide by
    zero or index an empty axis if a degenerate period were left unhandled.
    """

    dates = pd.date_range("2023-01-02", freq="W-MON", periods=8)
    freq_offset = pd.tseries.frequencies.to_offset("W-MON")

    def test_a_period_outside_the_data_yields_an_empty_window(self):
        """A period the fitted data does not reach produces no evaluated dates.

        Nothing about it is an error -- an ``all_time`` request over a padded
        calendar can produce one -- so it has to come out as an empty window and
        an all-false evaluation mask rather than as an exception or a window
        borrowed from a neighbour.
        """
        far_future = pd.Timestamp("2030-01-07")
        windows = EvaluationWindows.build(
            periods=[(self.dates[0], self.dates[2]), (far_future, far_future)],
            dates=self.dates,
            l_max=1,
            freq_offset=self.freq_offset,
        )

        empty = windows.windows[1]
        assert empty.n_actual == 0
        assert len(empty.actual_dates) == 0
        assert not empty.in_window.any()

        scenarios = windows.build_scenarios(
            baseline_array=np.ones((len(self.dates), 2)),
            counterfactual_spend_factor=0.0,
            dtype="float64",
            channel_axis=None,
            n_channels=2,
            estimand="per_channel",
        )

        # The empty period contributes a row of padding and sums to nothing.
        assert not empty.eval_mask(windows.max_window).any()
        assert len(empty.eval_dates) == 0
        assert (scenarios.spend[scenarios.rows[(1, None)]] == 0).all()

    def test_a_single_date_period_keeps_its_carryover(self):
        """One date wide, and still evaluated over the carryover that follows it.

        The narrowest period there is, and the one where the difference between
        the *window* and the *evaluation mask* is most visible: the window
        reaches back for history it will not sum, and forward for carryover it
        will.
        """
        windows = EvaluationWindows.build(
            periods=[(self.dates[4], self.dates[4])],
            dates=self.dates,
            l_max=2,
            freq_offset=self.freq_offset,
        )
        (window,) = windows.windows

        assert window.n_actual == 5
        assert window.actual_dates[0] == self.dates[2]

        scenarios = windows.build_scenarios(
            baseline_array=np.ones((len(self.dates), 2)),
            counterfactual_spend_factor=0.0,
            dtype="float64",
            channel_axis=None,
            n_channels=2,
            estimand="per_channel",
        )

        # Perturbed on its own date; summed over that date and the two after.
        np.testing.assert_array_equal(
            window.eval_mask(windows.max_window), [False, False, True, True, True]
        )
        np.testing.assert_array_equal(window.eval_dates, self.dates[4:7])
        spend = scenarios.spend[scenarios.rows[(0, None)]]
        np.testing.assert_array_equal(spend[:, 0], [1.0, 1.0, 0.0, 1.0, 1.0])

    def test_carryover_is_dropped_from_the_mask_but_not_from_the_window(self):
        """``include_carryover=False`` narrows the mask and leaves the window.

        The window still has to carry the surrounding dates, because the graph
        needs them to compute the period's own dates correctly; only the sum is
        restricted.
        """
        windows = EvaluationWindows.build(
            periods=[(self.dates[4], self.dates[4])],
            dates=self.dates,
            l_max=2,
            freq_offset=self.freq_offset,
            include_carryover=False,
        )

        (window,) = windows.windows
        assert window.n_actual == 5
        np.testing.assert_array_equal(
            window.eval_mask(windows.max_window), [False, False, True, False, False]
        )
        np.testing.assert_array_equal(window.eval_dates, self.dates[4:5])
