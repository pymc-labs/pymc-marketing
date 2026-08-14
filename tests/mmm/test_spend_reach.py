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
"""Tests for the spend-reach probe: window length and increment completeness."""

import numpy as np
import pytest

from pymc_marketing.mmm.additive_effect import IncrementalitySpec
from pymc_marketing.mmm.counterfactual import CounterfactualEvaluator
from pymc_marketing.mmm.spend_reach import (
    CHANNEL_CONTRIBUTION,
    LINEAR_PREDICTOR,
    ChannelDependentEffect,
    SpendProbe,
    TemporalReach,
    linear_predictor,
    resolve_channel_dependent_effects,
)


class FakeEvaluator:
    """Evaluator stand-in whose nodes are known functions of spend.

    :class:`SpendProbe` asks two things of an evaluator: the dtype the spend array
    has to be in, and an evaluation of every node under a given spend array.
    Supplying analytic nodes -- a causal filter of a chosen length, a reduction
    over ``date`` -- makes the measurement checkable against arithmetic rather
    than against a fitted model, and makes the degenerate spend arrays the probe
    has to survive cheap to construct.
    """

    channel_dtype = "float64"

    def __init__(self, **nodes):
        self.nodes = nodes

    def evaluate_baseline(self, channel_data):
        """Evaluate every node under *channel_data*."""
        return {name: node(channel_data) for name, node in self.nodes.items()}


def causal_filter(reach):
    """A ``channel_contribution``-shaped node moving for *reach* dates after spend.

    An impulse at date *i* therefore moves dates ``i .. i + reach - 1`` and no
    others, which is what an adstock of ``l_max = reach`` does.
    """

    def node(spend):
        out = np.zeros_like(spend, dtype="float64")
        for lag in range(reach):
            out[lag:] += spend[: len(spend) - lag] / (lag + 1)
        return out[np.newaxis]

    return node


def single_channel_filter(reach, channel):
    """A mediator-shaped node reading one channel and ignoring the others.

    An effect is free to select channels before transforming, so its node moves
    only when *that* channel's spend does -- the case a probe placed by total
    spend alone can miss entirely.
    """

    def node(spend):
        out = np.zeros(len(spend))
        column = spend[:, channel]
        for lag in range(reach):
            out[lag:] += column[: len(column) - lag] / (lag + 1)
        return out[np.newaxis]

    return node


def date_normalized(spend):
    """A node whose value at every date depends on every date.

    ``x / x.mean("date")`` and its relatives: not a causal filter, so no window
    reproduces it.  Written here as a ``channel_contribution``, because a custom
    saturation is free to do this and no ``mu_effect`` need be involved.
    """
    total = spend.reshape(len(spend), -1).sum(axis=1)
    return (total / total.mean())[np.newaxis]


def constant_node(spend):
    """A node a change in spend does not move at all."""
    return np.ones((1, len(spend)))


def float32_cancellation_values(
    n_dates,
    *,
    n_channels=6,
    node_magnitude=12_000.0,
    delta=2e-3,
    changed=2,
    seed=27,
):
    """Build float32 ``channel_contribution``/predictor arrays with rounding noise.

    ``channel_totals`` and ``probed_totals`` agree exactly in float64: only
    ``changed`` moves, by exactly *delta*.  Rounding each side to float32
    independently -- the per-channel totals on one side, their sum on the other
    -- does not preserve that identity, because the sum's float32 grid (spacing
    ``eps32 * node_magnitude``) is coarser than the grid a single small channel
    rounds on.  The resulting mismatch is on the order of ``eps32 *
    node_magnitude``, not of *delta*, which is what makes it noise rather than a
    real unaccounted contribution: it does not shrink as more decimal digits of
    *delta* are supplied, and it does not grow if the channels are re-scaled
    without changing ``node_magnitude``.

    Parameters
    ----------
    n_dates : int
        Number of dates to broadcast the (date-independent) values over.
    n_channels : int, default=6
        Number of synthetic ``channel_contribution`` components.
    node_magnitude : float, default=12_000.0
        Target magnitude of the predictor and of the channel totals combined --
        the scale the float32 rounding step is taken relative to.
    delta : float, default=2e-3
        Size of the move applied to channel *changed*, small relative to
        *node_magnitude*.
    changed : int, default=2
        Index of the channel that moves.
    seed : int, default=27
        Seed for the channel split.  Fixed rather than randomized per test run:
        the magnitude of the float32 cancellation this produces depends on
        exactly where the rounding boundaries fall, and a seed found to produce
        a representative amount of it is kept rather than re-drawn.

    Returns
    -------
    tuple of np.ndarray
        ``(channel_base, channel_probed, predictor_base, predictor_probed)``,
        each float32, shaped ``(n_dates, n_channels)`` for the channel pair and
        ``(n_dates,)`` for the predictor pair.
    """
    rng = np.random.default_rng(seed)
    channel_totals = rng.uniform(10.0, 3000.0, n_channels)
    channel_totals *= node_magnitude / channel_totals.sum()
    probed_totals = channel_totals.copy()
    probed_totals[changed] += delta

    channel_base32 = channel_totals.astype("float32")
    channel_probed32 = probed_totals.astype("float32")
    # The predictor is rounded from the float64 *sum*, not from the already
    # float32-rounded channel totals: that is what a real graph does, since the
    # predictor and channel_contribution are separate evaluations rather than
    # one derived from the other's rounded output.
    predictor_base32 = np.float32(channel_base32.sum(dtype=np.float32))
    predictor_probed32 = np.float32(channel_probed32.sum(dtype=np.float32))

    channel_base = np.broadcast_to(channel_base32, (n_dates, n_channels)).astype(
        "float32"
    )
    channel_probed = np.broadcast_to(channel_probed32, (n_dates, n_channels)).astype(
        "float32"
    )
    predictor_base = np.full(n_dates, predictor_base32, dtype="float32")
    predictor_probed = np.full(n_dates, predictor_probed32, dtype="float32")
    return channel_base, channel_probed, predictor_base, predictor_probed


def dtype_probe(*, spend, channel_dtype, **paired_arrays):
    """Build a :class:`SpendProbe` whose nodes return arrays chosen by hand.

    :class:`FakeEvaluator` normally derives a node's output from the spend array
    it is handed; here the desired -- already dtype-rounded -- baseline and
    probed values are supplied directly, because the cancellation this module
    tests for is the product of a real graph's many chained float32 operations,
    which a two-line node function cannot reproduce from arithmetic on *spend*
    alone.

    Each entry of *paired_arrays* maps a node name to a ``(baseline, probed)``
    pair.  :class:`SpendProbe` evaluates a node exactly twice: once outside
    construction, building the baseline dict from the untouched *spend* array,
    and once inside ``__init__``, evaluating the perturbed array.  The node
    function returns the baseline member of the pair when handed an array equal
    to *spend*, and the probed member otherwise.

    Parameters
    ----------
    spend : np.ndarray
        Actual spend, ``(n_dates, n_channels)``.  Only its shape and the dates
        it makes probeable matter; the constructed nodes ignore its values.
    channel_dtype : str
        Dtype the evaluator reports, and so the dtype the perturbed spend array
        is cast to before a node ever sees it.
    **paired_arrays
        Node name to ``(baseline, probed)`` array pair.

    Returns
    -------
    SpendProbe
        A probe built from the paired arrays.
    """

    def make_node(baseline_value, probed_value):
        def node(candidate):
            return baseline_value if np.array_equal(candidate, spend) else probed_value

        return node

    nodes = {
        name: make_node(baseline_value, probed_value)
        for name, (baseline_value, probed_value) in paired_arrays.items()
    }
    evaluator = FakeEvaluator(**nodes)
    evaluator.channel_dtype = channel_dtype
    baseline = evaluator.evaluate_baseline(spend)
    return SpendProbe(
        evaluator=evaluator,
        baseline=baseline,
        baseline_array=spend,
        counterfactual_spend_factor=0.0,
    )


def measure_spend_reach(mmm, counterfactual_spend_factor=0.0):
    """Return the :class:`SpendReach` the module's probe arrives at for *mmm*.

    Reassembles the pieces :meth:`Incrementality._compute_increments` puts
    together, so a test can ask what the probe concluded without inferring it from
    the numbers that come out the other end.
    """
    incr = mmm.incrementality
    effects = resolve_channel_dependent_effects(mmm)
    evaluator = CounterfactualEvaluator(
        pymc_model=mmm.model,
        posterior=incr.idata.posterior.dataset,
        response_vars=[
            CHANNEL_CONTRIBUTION,
            *(effect.contribution_var for effect in effects),
        ],
        frozen_deterministics=mmm.frozen_deterministics,
        dates=incr.data.dates,
    )
    baseline_array = incr.data.get_channel_data().values
    probe = SpendProbe(
        evaluator=evaluator,
        baseline=evaluator.evaluate_baseline(baseline_array),
        baseline_array=baseline_array,
        counterfactual_spend_factor=counterfactual_spend_factor,
    )
    return probe.measure(effects=effects, l_max=mmm.adstock.l_max)


def measure_reach(mmm, counterfactual_spend_factor=0.0):
    """Return the per-node :class:`TemporalReach` the probe measured for *mmm*."""
    return measure_spend_reach(mmm, counterfactual_spend_factor).measured


def effective_l_max(mmm):
    """Evaluation-window half-length the module will use for *mmm*.

    The oracle the incrementality tests compare against sums over the same
    window, so it has to agree with the module about the length.  Derived rather
    than hard-coded because the fixtures differ in both the model's own
    ``l_max`` and the mediator's, and a literal here would be a literal repeated
    eleven times; the two headline fixtures pin the value itself in
    ``test_the_window_length_is_a_known_number``.

    Cached on the model instance, because deriving it compiles a fresh evaluator
    and runs a full probe -- and most tests ask twice, once for the oracle's
    window and once inside the module call under test.
    """
    cached = getattr(mmm, "_tests_effective_l_max", None)
    if cached is None:
        cached = measure_spend_reach(mmm).effective_l_max
        mmm._tests_effective_l_max = cached
    return cached


class TestResolveChannelDependentEffects:
    """What ``resolve_channel_dependent_effects`` does with an unusual effect.

    ``AttributeError`` is overloaded: Python raises it both when an attribute is
    genuinely absent (the documented duck-typing escape hatch) and when a
    property that *is* present fails while computing its value.  The two must
    be told apart, because collapsing them means a bug inside
    ``contribution_var_name`` gets reclassified as "this effect has none" and
    silently dropped instead of surfacing as the programming error it is.
    """

    def test_attribute_error_inside_the_property_propagates(self, simple_fitted_mmm):
        """A property that raises ``AttributeError`` internally must not be caught.

        On the pre-fix code the broad ``except (AttributeError, ...)`` swallows
        this exactly like a genuinely missing attribute, and the effect is
        silently skipped instead of the bug surfacing.
        """

        class BuggyContributionVarNameEffect:
            """An effect whose property touches a misspelled instance attribute."""

            @property
            def contribution_var_name(self) -> str:
                """Read a name that was never set, raising ``AttributeError``."""
                return self._typo_never_set

        mmm = simple_fitted_mmm
        mmm.mu_effects = [*mmm.mu_effects, BuggyContributionVarNameEffect()]

        with pytest.raises(AttributeError):
            resolve_channel_dependent_effects(mmm)

    def test_a_duck_typed_instance_attribute_is_resolved(self, simple_fitted_mmm):
        """An instance-level ``contribution_var_name`` is not mistaken for absent.

        Duck-typed effects that set ``contribution_var_name`` on the instance
        rather than as a class-level property are a documented pattern; the fix
        must keep resolving them rather than special-casing them away as
        "class has no such attribute".
        """

        class DuckTypedInstanceAttributeEffect:
            """An effect that sets ``contribution_var_name`` in ``__init__``."""

            def __init__(self, contribution_var_name: str) -> None:
                self.contribution_var_name = contribution_var_name

            def incrementality_spec(self):
                """Opt in and declare nothing: the reach is measured."""
                return IncrementalitySpec()

        # A real, already-registered, channel-dependent deterministic (created
        # by ``mock_fit``'s ``add_original_scale_contribution_variable`` call)
        # with no declared dims, so it is trivially within the allowed dims of
        # any model regardless of how many extra dims that model carries.
        contribution_var = "total_media_contribution_original_scale"
        mmm = simple_fitted_mmm
        mmm.mu_effects = [
            *mmm.mu_effects,
            DuckTypedInstanceAttributeEffect(contribution_var),
        ]

        resolved = resolve_channel_dependent_effects(mmm)

        assert any(effect.contribution_var == contribution_var for effect in resolved)


class TestSpendProbe:
    """What the probe establishes, and what happens when it cannot.

    Two facts about the fitted graph are read off one single-date perturbation:
    how far in time it moves each evaluated node, and whether those nodes account
    for the whole move in the linear predictor.  Both are only as good as the date
    the probe picks, so the choice is tested here directly rather than inferred
    from the numbers a fitted model happens to produce.
    """

    n_dates = 24
    l_max = 3
    dark = 9
    """Dates before this carry no spend in the flighted-spend cases below."""

    @staticmethod
    def _spend(n_dates=24, n_channels=2, dark=0, spike=None):
        """Flat spend, less a dark run at the front and plus one spike."""
        spend = np.full((n_dates, n_channels), 2.0)
        spend[:dark] = 0.0
        if spike is not None:
            spend[spike] = 10.0
        return spend

    def _probe(self, spend, **nodes):
        """A probe over analytic nodes, built the way the module builds one."""
        evaluator = FakeEvaluator(**nodes)
        return SpendProbe(
            evaluator=evaluator,
            baseline=evaluator.evaluate_baseline(spend),
            baseline_array=spend,
            counterfactual_spend_factor=0.0,
        )

    # ---------- reach ----------

    def test_a_causal_filter_is_measured_at_its_own_length(self):
        """A node reaching past the model's adstock widens the window by the excess."""
        probe = self._probe(
            self._spend(), **{CHANNEL_CONTRIBUTION: causal_filter(self.l_max + 3)}
        )

        reach = probe.measure(effects=(), l_max=self.l_max)

        assert not reach.requires_full_axis
        # Reaching l_max + 3 dates is l_max + 2 lags, of which l_max is covered.
        assert reach.effective_l_max == self.l_max + 2

    def test_a_filter_inside_the_model_s_own_window_widens_nothing(self):
        """The ordinary case has to cost nothing: the same window as before."""
        probe = self._probe(
            self._spend(), **{CHANNEL_CONTRIBUTION: causal_filter(self.l_max)}
        )

        assert probe.measure(effects=(), l_max=self.l_max).effective_l_max == self.l_max

    def test_a_reduction_over_date_cannot_be_windowed(self):
        """A direct path that is not causal in ``date`` selects the full axis.

        No ``mu_effect`` is involved: a custom adstock or saturation that
        normalises over the date axis puts a reduction inside
        ``channel_contribution`` itself.  Measuring only the effects would leave
        such a model windowed silently wrong -- the window's mean standing in for
        the series' mean -- which is a different function, not a truncation.
        """
        probe = self._probe(self._spend(), **{CHANNEL_CONTRIBUTION: date_normalized})

        reach = probe.measure(effects=(), l_max=self.l_max)

        assert reach.measured[CHANNEL_CONTRIBUTION].requires_full_axis
        assert reach.requires_full_axis

    def test_a_full_axis_reach_still_carries_the_lags_it_demonstrated(self):
        """A tail that runs off the axis is unbounded, not unmeasured.

        The probe cannot say how long such a tail is, but it can say it had not
        ended by the last fitted date, which is a lower bound on the lags: from
        an impulse at ``probe_index`` the node was still moving at
        ``n_dates - 1``.  Reporting zero there would throw that away, and with it
        the only thing that lets a false declaration be caught for a full-axis
        node.
        """
        # A filter spanning the whole axis: from a probe near the front it
        # reaches past the last date, so the reach cannot be bounded.
        probe = self._probe(
            self._spend(n_dates=self.n_dates),
            **{CHANNEL_CONTRIBUTION: causal_filter(self.n_dates)},
        )
        assert probe.probe_indices == [1]

        reach = probe.measure(effects=(), l_max=self.l_max)
        measured = reach.measured[CHANNEL_CONTRIBUTION]

        assert measured.requires_full_axis
        # Moving from date 1 to date 23 is 22 lags, of which l_max is covered.
        assert measured.additional_carryover_lags == (
            self.n_dates - 1 - probe.probe_indices[0] - self.l_max
        )
        assert measured.additional_carryover_lags == 19

    @pytest.mark.parametrize("mode", ["auto", "full"])
    def test_a_declaration_under_a_full_axis_measurement_is_refused(self, mode):
        """A declaration the probe contradicted is refused on the full axis too.

        The reach here is unbounded, so the evaluation will run to the axis end
        and no window will cut anything.  The declaration is still false, and an
        author who believes the effect settles after two periods may well have
        built it around that, so it is refused rather than silently widened.

        The message has to say *that*, though.  Repeating the windowed case's
        "the increment would be understated" would send the reader looking for a
        truncation that is not happening.  Neither evaluation mode changes the
        answer: the measurement is what the declaration is judged against.
        """
        effect = ChannelDependentEffect(
            contribution_var="mediator",
            label="Mediator",
            declared_carryover_lags=2,
            declared_evaluation_mode=mode,
        )
        probe = self._probe(
            self._spend(n_dates=self.n_dates),
            **{
                CHANNEL_CONTRIBUTION: causal_filter(self.l_max),
                "mediator": single_channel_filter(self.n_dates, channel=0),
            },
        )
        assert probe.measure(effects=(), l_max=self.l_max).effective_l_max == self.l_max

        with pytest.raises(ValueError, match="at least 19 periods further") as excinfo:
            probe.measure(effects=(effect,), l_max=self.l_max)

        message = str(excinfo.value)
        assert "nothing is cut" in message
        assert "declared window" not in message

    def test_the_probe_avoids_a_date_that_carries_no_spend(self):
        """With a dark run at the front, the probe lands past it and still measures.

        Scaling an all-zero row by anything leaves it alone, so a probe placed by
        position measures nothing and reports a bounded reach it never observed.
        """
        spend = self._spend(dark=self.dark, spike=self.dark)
        probe = self._probe(
            spend, **{CHANNEL_CONTRIBUTION: causal_filter(self.l_max + 3)}
        )

        # The date a fixed positional heuristic picks is inside the dark run.
        assert not spend[len(spend) // 4].any()
        # Every channel spends past the dark run, so one probe covers them all.
        assert len(probe.probe_indices) == 1
        assert spend[probe.probe_indices[0]].all()
        assert probe.measure(effects=(), l_max=self.l_max).effective_l_max == (
            self.l_max + 2
        )

    def test_the_probe_avoids_the_last_date_too(self):
        """Flighted spend peaking on the final date must not park the probe there.

        An impulse at the last date has nowhere later to move, so
        :meth:`_reach_of` reads ``last_moved == n_dates - 1`` and reports every
        node full-axis regardless of its true reach.  A dark head followed by
        spend that only grows produces exactly this: the argmax fallback used
        to land on the last date because nothing excluded it, the same way the
        unexcluded first date used to swallow a reduction over ``date``.
        """
        spend = np.zeros((self.n_dates, 1))
        spend[self.dark :, 0] = np.arange(1, self.n_dates - self.dark + 1, dtype=float)
        assert spend[-1, 0] == spend.max()  # spend peaks on the final date

        probe = self._probe(spend, **{CHANNEL_CONTRIBUTION: causal_filter(1)})

        assert probe.probe_indices != []
        assert probe.probe_indices[0] != self.n_dates - 1
        reach = probe.measure(effects=(), l_max=0)
        assert not reach.requires_full_axis

    def test_perturbing_a_dark_date_moves_nothing_at_all(self):
        """The failure mode the choice of date exists to avoid, stated directly.

        Without this, the previous test could pass for the wrong reason -- a probe
        anywhere in the dark run happening to work.  It does not: every node comes
        back bit-identical, so the reach measurement sees a node spend does not
        move and the completeness check compares zero against zero.
        """
        spend = self._spend(dark=self.dark, spike=self.dark)
        evaluator = FakeEvaluator(
            **{CHANNEL_CONTRIBUTION: causal_filter(self.l_max + 3)}
        )

        probed = evaluator.evaluate_baseline(
            SpendProbe._perturb(
                baseline_array=spend,
                probe_index=len(spend) // 4,
                counterfactual_spend_factor=0.0,
                dtype="float64",
            )
        )

        np.testing.assert_array_equal(
            probed[CHANNEL_CONTRIBUTION],
            evaluator.evaluate_baseline(spend)[CHANNEL_CONTRIBUTION],
        )

    @pytest.mark.parametrize("spend_on_first_date", [False, True])
    def test_no_interior_spend_falls_back_to_the_full_axis(self, spend_on_first_date):
        """Spend nowhere, or only on the first date, leaves nothing to probe.

        The first date is excluded deliberately: an impulse there has no earlier
        date to move, so a reduction over ``date`` could not be told apart from a
        causal filter.  With nothing measured, the window falls back to the only
        mode that is correct without a measurement, and says so.
        """
        spend = self._spend(dark=self.n_dates)
        if spend_on_first_date:
            spend[0] = 2.0
        probe = self._probe(spend, **{CHANNEL_CONTRIBUTION: causal_filter(2)})

        assert probe.probe_indices == []
        with pytest.warns(UserWarning, match="could not be measured"):
            reach = probe.measure(effects=(), l_max=self.l_max)

        assert reach.requires_full_axis
        assert reach.effective_l_max == self.l_max
        assert reach.measured == {}

    def test_an_unprobeable_axis_also_skips_the_completeness_check(self):
        """With no probe, the completeness guard has nothing to compare -- and says so.

        The predictor here moves twice as much as the accounted nodes, which a
        probe would refuse loudly.  With nothing probeable both guards are
        disarmed at once, so the fallback warning has to admit the second one
        too rather than presenting full-axis evaluation as the whole cost.
        """
        spend = self._spend(dark=self.n_dates)
        node = causal_filter(2)
        probe = self._probe(
            spend,
            **{
                CHANNEL_CONTRIBUTION: node,
                LINEAR_PREDICTOR: lambda s: 2 * node(s).sum(axis=-1),
            },
        )

        assert probe.probe_indices == []
        # The misattribution goes unnoticed: nothing was probed, so nothing is
        # compared.  This is the disarmed state the warning below has to name.
        probe.assert_increment_is_complete(
            effects=(),
            non_date_dims={CHANNEL_CONTRIBUTION: ("channel",), LINEAR_PREDICTOR: ()},
        )

        with pytest.warns(UserWarning, match="completeness .* goes unverified"):
            probe.measure(effects=(), l_max=self.l_max)

    def test_a_mediator_reading_a_dark_channel_gets_its_own_probe(self):
        """Channels with no common spend date are probed one by one.

        The probe date used to be chosen by *total* spend, so a mediator reading
        only a channel that is dark at that date came back bit-identical: reach
        ``none()``, window sized from the model's own adstock, mediated tail cut
        off, increment silently understated.  With disjoint spend supports there
        is no single date that moves every channel, so every spending channel has
        to get a probe of its own.
        """
        spend = np.zeros((self.n_dates, 2))
        spend[1:3, 0] = 5.0  # channel 0: the head, where one probe would land
        spend[12, 1] = 2.0  # channel 1: dark until far past the head
        effect = ChannelDependentEffect(
            contribution_var="mediator",
            label="Mediator",
            declared_carryover_lags=None,
            declared_evaluation_mode="auto",
        )
        probe = self._probe(
            spend,
            **{
                CHANNEL_CONTRIBUTION: causal_filter(self.l_max),
                "mediator": single_channel_filter(self.l_max + 3, channel=1),
            },
        )

        # One probe per channel, each at a date its channel spends.
        assert len(probe.probe_indices) == 2
        assert spend[probe.probe_indices].any(axis=0).all()

        reach = probe.measure(effects=(effect,), l_max=self.l_max)

        assert reach.measured["mediator"] != TemporalReach.none()
        assert not reach.requires_full_axis
        # Reaching l_max + 3 dates is l_max + 2 lags, of which l_max is covered.
        assert reach.effective_l_max == self.l_max + 2

    def test_one_date_every_channel_spends_keeps_the_probe_single(self):
        """A covering date is preferred: one probe moves every possible subset.

        Both channels are dark over the head, so a covering probe exists only
        further along the axis -- and one evaluation is all the measurement then
        costs, exactly as before.
        """
        spend = self._spend(dark=self.dark)
        spend[: self.dark + 2, 0] = 0.0  # channel 0 stays dark a little longer

        indices = SpendProbe._select_probe_indices(spend)

        assert len(indices) == 1
        assert spend[indices[0]].all()

    def test_an_all_zero_channel_does_not_force_extra_probes(self):
        """A channel that never spends cannot be moved, so it needs no probe."""
        spend = self._spend()
        spend[:, 1] = 0.0

        assert len(SpendProbe._select_probe_indices(spend)) == 1

    def test_a_channel_spending_only_on_the_first_date_disables_the_probe(self):
        """A channel no interior impulse can reach leaves the window untrusted.

        The counterfactual will move that channel's first-date spend, but no
        probe can, so nothing measured on the other channels vouches for it.
        """
        spend = self._spend()
        spend[:, 1] = 0.0
        spend[0, 1] = 2.0
        probe = self._probe(spend, **{CHANNEL_CONTRIBUTION: causal_filter(2)})

        assert probe.probe_indices == []
        with pytest.warns(UserWarning, match="could not be measured"):
            assert probe.measure(effects=(), l_max=self.l_max).requires_full_axis

    def test_a_channel_spending_only_on_the_last_date_disables_the_probe(self):
        """The last-date counterpart of the first-date-only case.

        Spend sitting only at the very last date is exactly as unreachable by an
        interior impulse as spend sitting only at the very first, so it has to
        trip the same guard.
        """
        spend = self._spend()
        spend[:, 1] = 0.0
        spend[-1, 1] = 2.0

        assert SpendProbe._select_probe_indices(spend) == []

    def test_a_channel_spending_only_at_the_first_and_last_dates_disables_the_probe(
        self,
    ):
        """A channel reachable only at the axis ends is unprobeable either way.

        No interior impulse can move spend that sits only at the first or last
        date, so nothing measured on the other channels can vouch for this one
        either; the honest fallback has to fire.
        """
        spend = self._spend()
        spend[:, 1] = 0.0
        spend[0, 1] = 2.0
        spend[-1, 1] = 2.0

        assert SpendProbe._select_probe_indices(spend) == []

    def test_the_widest_reach_wins_on_both_counts(self):
        """Combining per-node reaches takes the longest tail and any full axis.

        Per node rather than aggregated, because a declaration is judged against
        its own node's measurement -- a slow mediator beside a fast one must not
        make the fast one's honest declaration look like an under-declaration.
        """
        combined = TemporalReach.widest(
            [
                TemporalReach(additional_carryover_lags=2, requires_full_axis=False),
                TemporalReach.full_axis(),
            ]
        )

        assert combined.additional_carryover_lags == 2
        assert combined.requires_full_axis
        assert TemporalReach.widest([]) == TemporalReach.none()

    def test_the_predictor_is_recoverable_under_an_identity_link(
        self, simple_fitted_mmm
    ):
        """An identity-link MMM only *names* its linear predictor.

        ``mmm.py`` wraps it in a ``Deterministic`` under a log link and assigns
        ``.name`` under an identity one, so it cannot be looked up in
        ``named_vars`` and has to be recovered from the observed variable's
        ancestors instead.
        """
        assert LINEAR_PREDICTOR not in simple_fitted_mmm.model.named_vars

        node = linear_predictor(simple_fitted_mmm)

        assert node is not None
        assert node.name == LINEAR_PREDICTOR

    def test_a_second_node_named_mu_is_refused_rather_than_guessed(
        self, simple_fitted_mmm, shadow_named_node
    ):
        """Two nodes named ``mu`` make the recovery a coin toss, so it refuses.

        Recovering the predictor by name is only sound while the name picks out
        one node.  A custom effect whose intermediate happens to be named ``mu``
        breaks that, and graph traversal is unordered, so the completeness check
        would bind to whichever of the two turned up first: a spurious refusal
        on a correct model, or a vacuous pass on a broken one.  Neither is worth
        having over an error that names the fix.
        """
        shadow_named_node(simple_fitted_mmm, LINEAR_PREDICTOR)

        with pytest.raises(ValueError, match="nodes named 'mu'"):
            linear_predictor(simple_fitted_mmm)

    def test_declared_carryover_survives_an_unprobeable_axis(self):
        """With nothing to probe, a declaration is all there is to go on.

        ``requires_full_axis`` widens the *window*, but the carry-out that enters
        each period's sum is sized by ``effective_l_max``, so falling back to the
        model's own ``l_max`` would silently shorten the sum by the declared
        mediated tail -- a narrower failure than the one the fallback exists to
        avoid, but the same kind.  Nothing was measured, so nothing can contradict
        the declaration and it is taken at face value.
        """
        probe = self._probe(
            self._spend(dark=self.n_dates),
            **{CHANNEL_CONTRIBUTION: causal_filter(2), "mediator": causal_filter(2)},
        )
        effect = ChannelDependentEffect(
            contribution_var="mediator",
            label="Mediator",
            declared_carryover_lags=4,
            declared_evaluation_mode="auto",
        )

        with pytest.warns(UserWarning, match="could not be measured"):
            reach = probe.measure(effects=(effect,), l_max=self.l_max)

        assert reach.requires_full_axis
        assert reach.effective_l_max == self.l_max + 4

    def test_a_plain_mmm_has_its_direct_path_measured(self, simple_fitted_mmm):
        """The probe runs on a model with no ``mu_effects`` in sight.

        Which is the point: the hazards above need no effect to exist, and gating
        the probe on ``mu_effects`` would leave every plain MMM unmeasured.  For an
        ordinary adstock the answer is that nothing changes.
        """
        reach = measure_spend_reach(simple_fitted_mmm)

        assert set(reach.measured) == {CHANNEL_CONTRIBUTION}
        assert not reach.requires_full_axis
        assert reach.effective_l_max == simple_fitted_mmm.adstock.l_max

    # ---------- non-finite predictions ----------

    def test_a_non_finite_probed_node_fails_construction_loudly(self):
        """A NaN produced only under the perturbation is caught at construction.

        ``spend / spend`` is finite everywhere on the baseline, which spends a
        flat amount at every date, but the probe zeroes one date's spend to
        perturb it -- so that date's evaluation divides zero by zero, the
        shape of a real custom-transform bug.  Before this check, that NaN
        reached :meth:`SpendProbe.measure`, where it poisoned ``largest`` in
        ``_reach_of`` and made every date compare as unmoved, so
        ``np.flatnonzero(moved)[-1]`` raised a bare ``IndexError`` instead of
        naming the actual problem.
        """
        spend = self._spend()

        def divide_by_self(spend: np.ndarray) -> np.ndarray:
            column = spend[:, 0]
            return (column / column)[np.newaxis]

        with pytest.raises(ValueError, match=CHANNEL_CONTRIBUTION) as excinfo:
            self._probe(spend, **{CHANNEL_CONTRIBUTION: divide_by_self})

        assert "non-finite" in str(excinfo.value)

    def test_a_non_finite_baseline_only_fails_construction_loudly(self):
        """A NaN present only in the baseline's linear-predictor entry is also caught.

        The baseline dict, including the linear predictor entry, is never run
        back through the reach measurement, so a NaN sitting there is only
        ever read by :meth:`SpendProbe.assert_increment_is_complete`.  Before
        this check, that method would compare a finite probed predictor
        against a NaN baseline, find the accounted and expected moves do not
        match within tolerance, and raise the misleading unattributed-spend
        ``NotImplementedError`` -- blaming attribution for what is a
        numerical problem in the model's own prediction.
        """
        spend = self._spend()
        evaluator = FakeEvaluator(
            **{
                CHANNEL_CONTRIBUTION: causal_filter(self.l_max),
                LINEAR_PREDICTOR: lambda spend: causal_filter(self.l_max)(spend).sum(
                    axis=-1
                ),
            }
        )
        baseline = evaluator.evaluate_baseline(spend)
        baseline[LINEAR_PREDICTOR][0, 5] = np.nan

        with pytest.raises(ValueError, match=LINEAR_PREDICTOR) as excinfo:
            SpendProbe(
                evaluator=evaluator,
                baseline=baseline,
                baseline_array=spend,
                counterfactual_spend_factor=0.0,
            )

        assert "non-finite" in str(excinfo.value)

    # ---------- completeness ----------

    def _assert_complete(self, **nodes):
        """Run the completeness check over analytic nodes."""
        probe = self._probe(self._spend(), **nodes)
        probe.assert_increment_is_complete(
            effects=(),
            non_date_dims={CHANNEL_CONTRIBUTION: ("channel",), LINEAR_PREDICTOR: ()},
        )

    def test_a_predictor_the_accounted_nodes_reproduce_is_accepted(self):
        """The identity the whole increment rests on, satisfied."""
        node = causal_filter(self.l_max)

        self._assert_complete(
            **{
                CHANNEL_CONTRIBUTION: node,
                LINEAR_PREDICTOR: lambda spend: node(spend).sum(axis=-1),
            }
        )

    def test_a_predictor_that_moves_more_than_the_accounted_nodes_is_refused(self):
        """Spend reaching the predictor by an unattributed route is caught."""
        node = causal_filter(self.l_max)

        with pytest.raises(NotImplementedError, match="Some path from spend"):
            self._assert_complete(
                **{
                    CHANNEL_CONTRIBUTION: node,
                    LINEAR_PREDICTOR: lambda spend: 2 * node(spend).sum(axis=-1),
                }
            )

    def test_accounted_nodes_moving_while_the_predictor_does_not_is_refused(self):
        """A predictor that stands still cannot excuse nodes that move.

        Scaling the comparison by the *predictor's* own move alone makes this pass
        vacuously: the tolerance collapses to zero and the guard returns early
        rather than comparing anything.  A frozen or misidentified predictor is
        exactly the case where that matters, since the increment would then be
        assembled from nodes nothing has vouched for.
        """
        with pytest.raises(NotImplementedError, match="Some path from spend"):
            self._assert_complete(
                **{
                    CHANNEL_CONTRIBUTION: causal_filter(self.l_max),
                    LINEAR_PREDICTOR: constant_node,
                }
            )

    def test_nothing_moving_anywhere_is_not_an_error(self):
        """Two nodes spend does not reach agree, and there is nothing to report."""
        self._assert_complete(
            **{
                CHANNEL_CONTRIBUTION: lambda spend: np.ones(
                    (1, len(spend), spend.shape[-1])
                ),
                LINEAR_PREDICTOR: constant_node,
            }
        )

    # ---------- dtype cancellation noise ----------

    def test_float32_cancellation_noise_does_not_trip_the_completeness_check(self):
        """A float32 model's own rounding must not read as unattributed spend.

        ``channel_base``/``channel_probed`` and ``predictor_base``/
        ``predictor_probed`` are built so the accounted identity -- the
        predictor's move equals the sum of the channel moves -- holds exactly
        in float64; only independently rounding each side to float32 disturbs
        it (see :func:`float32_cancellation_values`).  The resulting move is
        small (a few thousandths) while the nodes themselves are not (order
        ``1e4``), which is exactly the shape a real float32 model produces: the
        old ``atol = COMPLETENESS_TOLERANCE * scale`` floor is proportional to
        the *move*, so it cannot absorb rounding proportional to the *node*.
        Before the fix this raises the misleading unattributed-spend
        ``NotImplementedError``; after it, the check passes.
        """
        spend = self._spend()
        channel_base, channel_probed, predictor_base, predictor_probed = (
            float32_cancellation_values(len(spend))
        )
        probe = dtype_probe(
            spend=spend,
            channel_dtype="float32",
            **{
                CHANNEL_CONTRIBUTION: (
                    channel_base[np.newaxis],
                    channel_probed[np.newaxis],
                ),
                LINEAR_PREDICTOR: (
                    predictor_base[np.newaxis],
                    predictor_probed[np.newaxis],
                ),
            },
        )

        probe.assert_increment_is_complete(
            effects=(),
            non_date_dims={CHANNEL_CONTRIBUTION: ("channel",), LINEAR_PREDICTOR: ()},
        )

    def test_a_genuinely_broken_float32_attribution_is_still_refused(self):
        """The loosened tolerance does not stop catching a real unaccounted path.

        Mirrors the case above at the same node magnitude (``1e4``) and dtype,
        but here the channels genuinely account for only 95% of the predictor's
        100-unit move -- a real missing contribution, not rounding noise. Its
        size (5 units) is far above both the old tolerance and the new
        ``K * eps * node_magnitude`` floor, so this must keep raising after the
        fix exactly as it does before it.
        """
        spend = self._spend()
        n_dates = len(spend)
        predictor_base = np.full(n_dates, 12_000.0, dtype="float32")
        predictor_probed = np.full(n_dates, 12_100.0, dtype="float32")
        channel_base = np.broadcast_to(
            np.array([6_000.0, 6_000.0], dtype="float32"), (n_dates, 2)
        ).astype("float32")
        # Only 95 of the predictor's 100-unit move is attributed.
        channel_probed = np.broadcast_to(
            np.array([6_047.5, 6_047.5], dtype="float32"), (n_dates, 2)
        ).astype("float32")

        probe = dtype_probe(
            spend=spend,
            channel_dtype="float32",
            **{
                CHANNEL_CONTRIBUTION: (
                    channel_base[np.newaxis],
                    channel_probed[np.newaxis],
                ),
                LINEAR_PREDICTOR: (
                    predictor_base[np.newaxis],
                    predictor_probed[np.newaxis],
                ),
            },
        )

        with pytest.raises(NotImplementedError, match="Some path from spend"):
            probe.assert_increment_is_complete(
                effects=(),
                non_date_dims={
                    CHANNEL_CONTRIBUTION: ("channel",),
                    LINEAR_PREDICTOR: (),
                },
            )
