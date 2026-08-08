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
