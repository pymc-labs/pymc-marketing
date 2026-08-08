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

import warnings

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
from pymc_marketing.mmm.spend_reach import linear_predictor


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


def build_observed_model(n_dates=6):
    """A minimal model with a likelihood, for exercising target validation.

    ``build_aux_dims_model`` has no observed variable, and the refusals worth
    testing are exactly the ones about what a target *is* -- an observed
    variable, a free parameter, an integer input -- so this model carries one
    of each.
    """
    coords = {"date": list(range(n_dates))}
    with pm.Model(coords=coords) as model:
        channel_data = pmd.Data("channel_data", np.ones(n_dates), dims=("date",))
        pmd.Data("int_data", np.ones(n_dates, dtype="int64"), dims=("date",))
        pmd.Data("unrelated", np.ones(n_dates), dims=("date",))
        beta = pmd.Normal("beta", mu=0.0, sigma=1.0)
        # A second free parameter, because a single-variable posterior collapses
        # to a DataArray inside az.extract and never reaches the guard under test.
        alpha = pmd.Normal("alpha", mu=0.0, sigma=1.0)
        contribution = pmd.Deterministic(
            "channel_contribution", beta * channel_data + alpha
        )
        pmd.Normal(
            "y",
            mu=contribution,
            sigma=1.0,
            observed=xr.DataArray(np.zeros(n_dates), dims="date"),
            dims=("date",),
        )

    posterior = xr.Dataset(
        {
            "beta": (("chain", "draw"), np.array([[1.0, 2.0]])),
            "alpha": (("chain", "draw"), np.array([[0.5, -0.5]])),
        },
        coords={"chain": [0], "draw": [0, 1]},
    )
    return model, posterior


class TestInterventionValidation:
    """Targets the intervention machinery must refuse, and how it says so.

    Every refusal happens at construction, before any graph work: a bad target
    silently producing a number is the failure mode all of these exist to
    prevent.
    """

    dates = pd.date_range("2023-01-02", freq="W-MON", periods=6)

    def _construct(self, model, posterior, **kwargs):
        kwargs.setdefault("response_vars", ["channel_contribution"])
        kwargs.setdefault("frozen_deterministics", [])
        return CounterfactualEvaluator(
            pymc_model=model,
            posterior=posterior,
            dates=self.dates,
            **kwargs,
        )

    def test_an_unknown_target_is_refused_by_name(self):
        model, posterior = build_observed_model()
        with pytest.raises(ValueError, match="'nope' is not a variable"):
            self._construct(model, posterior, intervention_target="nope")

    def test_the_observed_variable_is_refused(self):
        """``do`` on an observed variable silently deletes the likelihood."""
        model, posterior = build_observed_model()
        with pytest.raises(ValueError, match="remove the likelihood"):
            self._construct(model, posterior, intervention_target="y")

    def test_a_free_parameter_is_refused(self):
        model, posterior = build_observed_model()
        with pytest.raises(ValueError, match="is a random variable"):
            self._construct(model, posterior, intervention_target="beta")

    def test_a_frozen_target_is_refused(self):
        """A node cannot be held at its posterior values and intervened on."""
        model, posterior = build_observed_model()
        with pytest.raises(ValueError, match="both frozen and intervened"):
            self._construct(
                model,
                posterior,
                intervention_target="channel_contribution",
                frozen_deterministics=["channel_contribution"],
            )

    def test_an_integer_target_is_refused(self):
        model, posterior = build_observed_model()
        with pytest.raises(
            ValueError, match="'int_data' requires values of float type"
        ):
            self._construct(model, posterior, intervention_target="int_data")

    def test_a_target_without_a_leading_date_dimension_is_refused(self):
        model, posterior, _ = build_aux_dims_model(
            ("country", "date"), n_dates=len(self.dates), n_country=2
        )
        with pytest.raises(ValueError, match="leading 'date' dimension"):
            self._construct(model, posterior, intervention_target="aux_input")

    def test_an_unknown_mode_is_refused(self):
        model, posterior = build_observed_model()
        with pytest.raises(ValueError, match="Unknown intervention_mode"):
            self._construct(model, posterior, intervention_mode="clamp")

    def test_a_response_independent_of_the_target_is_refused(self):
        """Intervening on a node the responses never read is a diagnosis, not a number.

        Without the guard the compiled function would raise an opaque
        ``UnusedInputError`` -- or worse, evaluate: every counterfactual would
        equal the baseline and every increment would be zero.
        """
        model, posterior = build_observed_model()
        with pytest.raises(ValueError, match="depend on the intervention target"):
            self._construct(model, posterior, intervention_target="unrelated")


class TestEndogenousIntervention:
    """Interventions on a node the model computes, on the toy model.

    ``channel_contribution = beta * channel_data + aux + alpha`` maps zero
    spend to ``aux + alpha``, not to zero -- deliberately the case where
    zeroing spend and zeroing the contribution are different questions, so the
    two paths must agree where the math says so and differ where it says not.
    """

    dates = pd.date_range("2023-01-02", freq="W-MON", periods=6)

    def _evaluators(self, mode):
        model, posterior, aux_values = build_aux_dims_model(
            ("date", "country"), n_dates=len(self.dates), n_country=2
        )
        default = CounterfactualEvaluator(
            pymc_model=model,
            posterior=posterior,
            response_vars=["channel_contribution"],
            frozen_deterministics=[],
            dates=self.dates,
        )
        endogenous = CounterfactualEvaluator(
            pymc_model=model,
            posterior=posterior,
            response_vars=["channel_contribution"],
            frozen_deterministics=[],
            dates=self.dates,
            intervention_target="channel_contribution",
            intervention_mode=mode,
        )
        return default, endogenous, aux_values

    def test_replace_mode_returns_the_intervention_itself(self):
        """The target's parents are cut and the evaluator's input is the value.

        Also the case where the response *is* the intervention: no random
        variable is left in its graph, so the result has to be broadcast over
        the posterior samples rather than crash on a missing ``sample`` axis.
        """
        _, evaluator, _ = self._evaluators("replace")

        assert evaluator.windowed_data_vars == ()

        values = np.arange(12, dtype="float64").reshape(6, 2, 1)
        out = evaluator.evaluate_baseline(values)["channel_contribution"]

        assert out.shape == (2, 6, 2, 1)
        np.testing.assert_array_equal(out, np.broadcast_to(values, out.shape))

    def test_a_mask_of_ones_reproduces_the_baseline(self):
        """Scaling by one is the factual model, and spend becomes a held input."""
        default, scaled, _ = self._evaluators("scale")

        # The spend data is no longer the intervention target, so discovery
        # picks it up as an auxiliary input held at its factual values.  A set
        # comparison, because the clone's registration order is its own.
        assert set(scaled.windowed_data_vars) == {"channel_data", "aux_input"}

        baseline = default.evaluate_baseline(np.ones((6, 2, 1)))
        masked = scaled.evaluate_baseline(np.ones((6, 2, 1)))

        np.testing.assert_allclose(
            masked["channel_contribution"],
            baseline["channel_contribution"],
            rtol=1e-12,
        )

    def test_a_mask_of_zeros_returns_exactly_zero(self):
        """The intervened node, not its factual twin, is what gets evaluated.

        The intervened model registers the factual computation under the
        target's own name alongside the intervened ``do_<target>`` node, and
        resolving the wrong one would return the baseline: every element here
        would be nonzero.
        """
        _, scaled, _ = self._evaluators("scale")

        out = scaled.evaluate_baseline(np.zeros((6, 2, 1)))["channel_contribution"]

        assert (out == 0.0).all()

    def test_the_rename_warning_stays_internal(self):
        """The scale-mode self-reference is by design; the user hears nothing."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            self._evaluators("scale")

        messages = [str(entry.message) for entry in record]
        assert not any("Intervention expression" in message for message in messages)

    def test_zeroing_spend_and_zeroing_the_contribution_differ_as_the_algebra_says(
        self,
    ):
        """The two estimands, computed exactly, on a model where they differ.

        Zeroing spend removes ``beta * channel_data`` and leaves ``aux + alpha``
        standing; zeroing the contribution removes all of it.  The gap between
        the two increments is exactly ``aux + alpha``, everywhere nonzero for
        this posterior.
        """
        default, scaled, aux_values = self._evaluators("scale")
        beta = np.array([1.0, 2.0]).reshape(2, 1, 1, 1)
        alpha = np.array([0.5, -0.5]).reshape(2, 1, 1, 1)
        aux = aux_values[np.newaxis, :, :, np.newaxis]

        spend_increment = (
            default.evaluate_baseline(np.ones((6, 2, 1)))["channel_contribution"]
            - default.evaluate_baseline(np.zeros((6, 2, 1)))["channel_contribution"]
        )
        mask_increment = (
            scaled.evaluate_baseline(np.ones((6, 2, 1)))["channel_contribution"]
            - scaled.evaluate_baseline(np.zeros((6, 2, 1)))["channel_contribution"]
        )

        # Spend path: f(x) - f(0) = beta * channel_data, with channel_data = 1.
        np.testing.assert_allclose(
            spend_increment, np.broadcast_to(beta, spend_increment.shape), rtol=1e-12
        )
        # Mask path: the whole factual contribution.
        np.testing.assert_allclose(
            mask_increment,
            np.broadcast_to(beta + aux + alpha, mask_increment.shape),
            rtol=1e-12,
        )
        # And the gap is aux + alpha: nonzero in every element for this posterior.
        assert (np.abs(mask_increment - spend_increment) > 0).all()


class TestSpendMaskEquivalence:
    """Where the two intervention paths must agree on a real MMM, and where not.

    ``simple_fitted_mmm`` is the equivalence regime by construction: identity
    link, logistic saturation with ``f(0) = 0``, no time-varying media.  There,
    zeroing a channel's spend and zeroing its contribution are the same
    estimand; at a fractional factor they provably part ways, with a pinned
    direction.
    """

    @staticmethod
    def _evaluators(mmm, response_vars):
        incrementality = mmm.incrementality
        common = {
            "pymc_model": mmm.model,
            "posterior": incrementality.idata.posterior.dataset,
            "response_vars": response_vars,
            "frozen_deterministics": mmm.frozen_deterministics,
            "dates": incrementality.data.dates,
        }
        default = CounterfactualEvaluator(**common)
        masked = CounterfactualEvaluator(
            **common,
            intervention_target="channel_contribution",
            intervention_mode="scale",
        )
        spend = incrementality.data.get_channel_data().values.astype("float64")
        return default, masked, spend

    def test_factor_zero_is_the_same_estimand(self, simple_fitted_mmm):
        """Per channel: spend zeroing equals contribution zeroing, node by node.

        The linear predictor is passed as a raw variable of the *live* model's
        graph -- the way ``Incrementality`` passes it -- so this also pins its
        re-resolution against the intervened clone.
        """
        predictor = linear_predictor(simple_fitted_mmm)
        default, masked, spend = self._evaluators(
            simple_fitted_mmm, ["channel_contribution", predictor]
        )
        ones = np.ones_like(spend)

        spend_base = default.evaluate_baseline(spend)
        mask_base = masked.evaluate_baseline(ones)
        for name in ("channel_contribution", "mu"):
            np.testing.assert_allclose(
                mask_base[name], spend_base[name], rtol=1e-6, atol=1e-10
            )

        for channel in range(spend.shape[1]):
            spend_zero = spend.copy()
            spend_zero[:, channel] = 0.0
            mask_zero = ones.copy()
            mask_zero[:, channel] = 0.0

            spend_cf = default.evaluate_baseline(spend_zero)
            mask_cf = masked.evaluate_baseline(mask_zero)
            for name in ("channel_contribution", "mu"):
                np.testing.assert_allclose(
                    spend_base[name] - spend_cf[name],
                    mask_base[name] - mask_cf[name],
                    rtol=1e-6,
                    atol=1e-10,
                )

    def test_a_fractional_factor_separates_the_estimands(self, simple_fitted_mmm):
        """Halving spend removes less than half the effect; the mask removes exactly half.

        Concavity with ``f(0) = 0`` gives ``f(x) - f(x/2) <= f(x) / 2``
        pointwise, and adstock is linear so the argument survives it.  The
        spend-path increment is therefore bounded by the mask-path increment,
        strictly wherever the channel contributes at all.
        """
        default, masked, spend = self._evaluators(
            simple_fitted_mmm, ["channel_contribution"]
        )
        ones = np.ones_like(spend)
        name = "channel_contribution"

        spend_base = default.evaluate_baseline(spend)[name]
        mask_base = masked.evaluate_baseline(ones)[name]

        for channel in range(spend.shape[1]):
            spend_half = spend.copy()
            spend_half[:, channel] *= 0.5
            mask_half = ones.copy()
            mask_half[:, channel] = 0.5

            spend_increment = (
                spend_base - default.evaluate_baseline(spend_half)[name]
            ).sum(axis=1)[:, channel]
            mask_increment = (
                mask_base - masked.evaluate_baseline(mask_half)[name]
            ).sum(axis=1)[:, channel]

            gap = mask_increment - spend_increment
            assert (gap >= -1e-9).all()
            # Strict wherever the channel's factual contribution is not
            # numerically nothing.
            contributes = mask_increment > 1e-8
            assert contributes.any()
            assert (gap[contributes] > 0).all()

    def test_scale_mode_survives_time_varying_media(
        self, time_varying_media_fitted_mmm
    ):
        """`do` composes with frozen HSGP deterministics, warning-free.

        The time-varying multiplier is held at its posterior values on both
        paths, so a mask of ones still reproduces the default baseline exactly.
        """
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            default, masked, spend = self._evaluators(
                time_varying_media_fitted_mmm, ["channel_contribution"]
            )

        messages = [str(entry.message) for entry in record]
        assert not any("Intervention expression" in message for message in messages)

        baseline = default.evaluate_baseline(spend)["channel_contribution"]
        masked_baseline = masked.evaluate_baseline(np.ones_like(spend))[
            "channel_contribution"
        ]
        np.testing.assert_allclose(masked_baseline, baseline, rtol=1e-10)

    def test_scale_mode_keeps_the_panel_layout(self, panel_fitted_mmm):
        """A panel model's extra dimension flows through the mask path unchanged."""
        default, masked, spend = self._evaluators(
            panel_fitted_mmm, ["channel_contribution"]
        )

        assert (
            masked.non_date_dims["channel_contribution"]
            == default.non_date_dims["channel_contribution"]
        )

        baseline = default.evaluate_baseline(spend)["channel_contribution"]
        masked_baseline = masked.evaluate_baseline(np.ones_like(spend))[
            "channel_contribution"
        ]
        np.testing.assert_allclose(masked_baseline, baseline, rtol=1e-10)
