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
r"""How far a change in spend reaches, and whether the accounting closes.

:mod:`~pymc_marketing.mmm.incrementality` computes a difference of predictions
and divides it by spend.  Before it can, two facts about the fitted graph have to
be established, and neither can be assumed:

1. **How far in time a change in spend moves the evaluated nodes.**  The
   counterfactual is evaluated on a *window* around each period rather than on
   the whole date axis.  A window is only valid if the perturbation's effect dies
   inside it, and if each evaluated node is a causal function of ``date`` at all.
   Guess low and the tail falls outside the window, which returns a smaller
   increment with nothing to indicate anything was cut.
2. **That the evaluated nodes account for the whole move in the linear
   predictor.**  The increment is assembled from ``channel_contribution`` plus
   the resolved ``mu_effects``.  If spend reaches :math:`\mu` by some other
   route, the increment reports one part of the response as though it were all of
   it.

Both are read off *one* extra evaluation: perturb spend at a single interior date
on the untruncated axis and compare against the baseline.  That shared
measurement is why the two questions live in the same module -- and why they live
apart from :class:`~pymc_marketing.mmm.incrementality.Incrementality`, which owns
periods, windows, spend and the link-specific reduction, and has nothing to say
about either.

The entry point is :meth:`SpendProbe.measure`, which returns a
:class:`SpendReach`: an evaluation-window length and whether a window is usable
at all.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors

from pymc_marketing.mmm.counterfactual import CounterfactualEvaluator

if TYPE_CHECKING:
    from pymc_marketing.mmm.mmm import MMM

__all__ = [
    "CHANNEL_CONTRIBUTION",
    "LINEAR_PREDICTOR",
    "ChannelDependentEffect",
    "SpendProbe",
    "SpendReach",
    "TemporalReach",
    "linear_predictor",
    "resolve_channel_dependent_effects",
]

CHANNEL_CONTRIBUTION = "channel_contribution"
"""Response variable holding the per-channel linear-predictor contribution."""

LINEAR_PREDICTOR = "mu"
"""Name the MMM gives its linear predictor, registered or not."""


@dataclass(frozen=True)
class ChannelDependentEffect:
    """A ``mu_effect`` whose contribution a spend counterfactual can reach.

    Produced by :func:`resolve_channel_dependent_effects` for every effect that
    (a) has ``channel_data`` among the ancestors of its contribution variable and
    (b) opted in via
    :meth:`~pymc_marketing.mmm.additive_effect.MuEffect.incrementality_spec`.

    Parameters
    ----------
    contribution_var : str
        Name of the deterministic holding the term this effect adds to the
        linear predictor.
    label : str
        Class name of the originating effect, for error messages.
    declared_carryover_lags : int or None
        Extra evaluation-window length declared by the effect's
        :class:`~pymc_marketing.mmm.additive_effect.IncrementalitySpec`, or
        ``None`` to have it measured.
    declared_evaluation_mode : {"auto", "window", "full"}
        The spec's ``evaluation_mode``.
    """

    contribution_var: str
    label: str
    declared_carryover_lags: int | None
    declared_evaluation_mode: Literal["auto", "window", "full"]


def resolve_channel_dependent_effects(
    model: MMM,
) -> tuple[ChannelDependentEffect, ...]:
    """Find the ``mu_effects`` a spend counterfactual reaches.

    The counterfactual perturbs ``channel_data``, so any effect with
    ``channel_data`` among its ancestors carries part of the resulting change in
    the linear predictor and has to be evaluated alongside
    ``channel_contribution``.  A funnel mediator is the motivating case:
    upper-funnel spend moves latent demand, demand moves lower-funnel spend, and
    only then does the target respond.

    Effects that do not depend on ``channel_data`` -- a linear trend, an event
    window, seasonality -- are part of the baseline.  They cancel in the
    difference and are skipped without consulting their spec.

    An effect whose contribution cannot be located at all is *not* an error here.
    Duck-typed effects are a documented pattern (see the module docstring of
    :mod:`~pymc_marketing.mmm.additive_effect`) and need not carry a
    ``contribution_var_name``; a ``MuEffect`` built without a ``prefix`` raises
    rather than returning one.  Neither says anything about whether spend reaches
    the effect, so the failure is deferred: such an effect is left unaccounted,
    and :meth:`SpendProbe.assert_increment_is_complete` raises only if a spend
    path really does escape through it.  Raising eagerly instead would break
    every model that merely *owns* such an effect.

    Parameters
    ----------
    model : MMM
        The fitted model whose ``mu_effects`` are being resolved.

    Returns
    -------
    tuple of ChannelDependentEffect
        One entry per effect to include, in ``mu_effects`` order.  Empty for a
        model with no channel-dependent effects, which is the separable case the
        incrementality module was originally written for.

    Raises
    ------
    NotImplementedError
        If an effect is known to depend on ``channel_data`` but has not opted in
        via
        :meth:`~pymc_marketing.mmm.additive_effect.MuEffect.incrementality_spec`.
        A refusal to guess: the alternative is dropping a real part of the
        increment and reporting the remainder as if it were the whole.
    ValueError
        If an included effect's contribution carries dimensions outside
        ``("date", *model.dims)``.
    """
    pymc_model = model.model
    channel_data = pymc_model[CounterfactualEvaluator.CHANNEL_DATA]
    allowed_dims = {"date", *model.dims}
    resolved: list[ChannelDependentEffect] = []

    for effect in model.mu_effects:
        label = type(effect).__name__
        try:
            name = effect.contribution_var_name
        except (AttributeError, NotImplementedError):
            continue
        if name not in pymc_model.named_vars:
            continue

        node = pymc_model[name]
        if channel_data not in set(ancestors([node])):
            continue

        spec = getattr(effect, "incrementality_spec", lambda: None)()
        if spec is None:
            raise NotImplementedError(
                f"The mu_effect {label!r} contributes {name!r}, which "
                "depends on channel spend, so a spend counterfactual moves "
                "it and it forms part of the incremental response.  It has "
                "not opted in to incrementality: implement "
                "'incrementality_spec' returning an IncrementalitySpec.  "
                "Ignoring the effect would report the direct path alone as "
                "if it were the total."
            )

        effect_dims = set(pymc_model.named_vars_to_dims.get(name, ()))
        if not effect_dims <= allowed_dims:
            raise ValueError(
                f"The contribution {name!r} of mu_effect {label!r} has "
                f"dimensions {tuple(sorted(effect_dims))}, which is not a "
                f"subset of {tuple(sorted(allowed_dims))}.  A term added to "
                "the linear predictor cannot carry dimensions the linear "
                "predictor does not have."
            )

        resolved.append(
            ChannelDependentEffect(
                contribution_var=name,
                label=label,
                declared_carryover_lags=spec.additional_carryover_lags,
                declared_evaluation_mode=spec.evaluation_mode,
            )
        )

    return tuple(resolved)


def linear_predictor(model: MMM) -> Variable | None:
    """Find the node the increment is assembled to reproduce.

    Under a log link the MMM registers its linear predictor as the
    ``Deterministic`` ``mu``.  Under an identity link the same quantity is an
    anonymous intermediate that only carries the *name* ``mu``, so it has to be
    recovered from the graph of the observed variable.

    Parameters
    ----------
    model : MMM
        The fitted model to search.

    Returns
    -------
    Variable or None
        The linear predictor, or ``None`` if this model does not expose one -- in
        which case :meth:`SpendProbe.assert_increment_is_complete` has nothing to
        check against and says so.
    """
    pymc_model = model.model
    if LINEAR_PREDICTOR in pymc_model.named_vars:
        if LINEAR_PREDICTOR in model.frozen_deterministics:
            # Frozen at its posterior value, so it would not respond to the
            # probe and the completeness check would compare zero to zero.
            return None
        return pymc_model[LINEAR_PREDICTOR]

    output_var = model.output_var
    if output_var not in pymc_model.named_vars:
        return None
    observed = pymc_model[output_var]
    return next(
        (
            node
            for node in ancestors([observed])
            if getattr(node, "name", None) == LINEAR_PREDICTOR and node is not observed
        ),
        None,
    )


@dataclass(frozen=True)
class TemporalReach:
    """How much of the date axis a change in spend moves one node over.

    Measured by :meth:`SpendProbe.measure` rather than taken on trust, because a
    node that reaches further than the evaluation window would have its tail cut
    off and the increment would come back quietly low.  Measured *per node*, so
    that one effect's long tail cannot be held against another effect's honest
    declaration of a short one.

    Parameters
    ----------
    additional_carryover_lags : int
        Periods beyond the model's own ``adstock.l_max`` over which the node
        still moves after a change in spend at a single date.
    requires_full_axis : bool
        Whether the node has to be evaluated on the complete fitted date axis.
        True when a perturbation still moves the far end of the axis (reach
        longer than the axis can show), or moves dates *before* the perturbed one
        -- the signature of a reduction over ``date``, which takes a different
        value on a truncated axis and so cannot be windowed at all.
    """

    additional_carryover_lags: int
    requires_full_axis: bool

    @classmethod
    def none(cls) -> TemporalReach:
        """Return the reach of a node a change in spend does not move."""
        return cls(additional_carryover_lags=0, requires_full_axis=False)

    @classmethod
    def full_axis(cls) -> TemporalReach:
        """Return the reach of a node no window can reproduce."""
        return cls(additional_carryover_lags=0, requires_full_axis=True)

    @classmethod
    def widest(cls, reaches: Iterable[TemporalReach]) -> TemporalReach:
        """Combine per-node reaches into the one an evaluation has to satisfy.

        Parameters
        ----------
        reaches : iterable of TemporalReach
            The individual reaches.  Empty means nothing to accommodate.

        Returns
        -------
        TemporalReach
            Long enough for the longest, and full-axis if any one of them is.
        """
        reaches = list(reaches)
        return cls(
            additional_carryover_lags=max(
                (reach.additional_carryover_lags for reach in reaches), default=0
            ),
            requires_full_axis=any(reach.requires_full_axis for reach in reaches),
        )


@dataclass(frozen=True)
class SpendReach:
    """What the evaluation is allowed to assume about its window.

    The whole contract between this module and
    :class:`~pymc_marketing.mmm.incrementality.Incrementality`: two numbers, plus
    the per-node measurements they were derived from for anyone who wants to see
    the working.

    Parameters
    ----------
    effective_l_max : int
        Evaluation-window half-length covering every path spend can take -- the
        model's own ``adstock.l_max`` plus the widest extra carryover measured.  A
        mediated path that chains a second adstock outlives the direct one, and a
        window sized for the direct path alone would cut the tail off.
    requires_full_axis : bool
        Whether every period has to be evaluated on the complete fitted date
        axis, because some node's value depends on the whole series.
    measured : mapping
        Per evaluated node, its :class:`TemporalReach`.  Empty when no probe was
        possible.
    """

    effective_l_max: int
    requires_full_axis: bool
    measured: Mapping[str, TemporalReach] = field(default_factory=dict)


class SpendProbe:
    """One single-date spend perturbation, and the two facts read off it.

    A probe is an *impulse*: spend at one interior date is scaled, every other
    date left alone, and the whole graph re-evaluated on the untruncated axis.
    Comparing that against the baseline makes observable what the evaluation
    otherwise has to assume -- how far forward a change in spend still moves each
    node, whether any node moves *backwards* in time, and whether the nodes being
    evaluated account for the whole move in the linear predictor.

    Parameters
    ----------
    evaluator : CounterfactualEvaluator
        Compiled evaluator over the accounted nodes.  Reused rather than
        recompiled: the probe is one more call to a function that already exists.
    baseline : dict
        Unperturbed evaluation on the full date axis, per node.
    baseline_array : np.ndarray
        Actual spend, ``(n_dates, *extra_shape)``.
    counterfactual_spend_factor : float
        The factor the caller will apply, so the measurement is taken where the
        analysis will be run.  A factor of exactly ``1.0`` perturbs nothing, so
        the probe falls back to zeroing the date out.

    Attributes
    ----------
    probe_index : int or None
        Index of the perturbed date, or ``None`` if no date could be perturbed --
        see :meth:`_select_probe_index`.
    """

    REACH_TOLERANCE = 1e-9
    """Relative size below which a probed move counts as no move at all.

    :meth:`measure` compares each date's move against the largest move the same
    node makes, so the threshold is scale-free.  The tails it is applied to decay
    geometrically and are usually cut to exactly zero by an adstock's own
    truncation, which puts the real signal many orders of magnitude above float
    noise.
    """

    COMPLETENESS_TOLERANCE = 1e-6
    """Relative slack allowed between the predictor's move and the accounted one.

    :meth:`assert_increment_is_complete` compares two float sums of the same
    terms in different orders, so they agree to rounding and not beyond.  An
    unaccounted path, by contrast, leaves a discrepancy of the size of a real
    contribution -- there is nothing in between for the threshold to have to
    discriminate.
    """

    def __init__(
        self,
        *,
        evaluator: CounterfactualEvaluator,
        baseline: dict[str, np.ndarray],
        baseline_array: np.ndarray,
        counterfactual_spend_factor: float,
    ) -> None:
        self.baseline = baseline
        self.probe_index = self._select_probe_index(baseline_array)
        self.probed: dict[str, np.ndarray] | None = None
        if self.probe_index is not None:
            self.probed = evaluator.evaluate_baseline(
                self._perturb(
                    baseline_array=baseline_array,
                    probe_index=self.probe_index,
                    counterfactual_spend_factor=counterfactual_spend_factor,
                    dtype=evaluator.channel_dtype,
                )
            )

    # ==================== The perturbation ====================

    @staticmethod
    def _select_probe_index(baseline_array: np.ndarray) -> int | None:
        """Choose a date whose spend the probe can actually move.

        A multiplicative perturbation of an all-zero row is a no-op, and a no-op
        probe is worse than no probe: every node comes back unmoved, so the reach
        looks bounded and the completeness check compares zero against zero.
        Both guards would pass by failing to see anything.  So the date is chosen
        by *spend* rather than by position, and a probe that could not move
        anything is reported as no probe at all.

        Among the dates that carry spend, an early one is preferred.  The two
        things :meth:`measure` reads off the probe both need room on the axis: a
        forward tail has to be able to end before the last date, or its length
        cannot be bounded, and a backward move has to have at least one earlier
        date to show up in.  Placing the probe near the front maximises the
        former and costs nothing for the latter, since a reduction over ``date``
        moves *every* date rather than only nearby ones.

        Parameters
        ----------
        baseline_array : np.ndarray
            Actual spend, ``(n_dates, *extra_shape)``.

        Returns
        -------
        int or None
            Index of the date to perturb, or ``None`` when no date other than the
            first carries any spend -- an all-zero spend array, or one with spend
            only at the very first date, where no interior impulse exists.
        """
        n_dates = baseline_array.shape[0]
        per_date = np.abs(baseline_array.reshape(n_dates, -1)).sum(axis=1)
        # The first date is excluded: an impulse there has no earlier date to
        # move, so a reduction over "date" would be indistinguishable from a
        # causal filter.
        if n_dates < 2 or not per_date[1:].any():
            return None

        head = slice(1, max(2, n_dates // 8))
        candidates = per_date[head]
        if candidates.any():
            return head.start + int(np.argmax(candidates))
        return 1 + int(np.argmax(per_date[1:]))

    @staticmethod
    def _perturb(
        *,
        baseline_array: np.ndarray,
        probe_index: int,
        counterfactual_spend_factor: float,
        dtype: str,
    ) -> np.ndarray:
        """Return spend with one date scaled and every other date untouched.

        Parameters
        ----------
        baseline_array : np.ndarray
            Actual spend, ``(n_dates, *extra_shape)``.
        probe_index : int
            Date to perturb.
        counterfactual_spend_factor : float
            Factor to scale it by; ``1.0`` would perturb nothing, so it falls
            back to zeroing the date out.
        dtype : str
            Dtype the compiled evaluator expects.

        Returns
        -------
        np.ndarray
            A copy of *baseline_array*, one row scaled.
        """
        factor = (
            counterfactual_spend_factor if counterfactual_spend_factor != 1.0 else 0.0
        )
        probe_array = baseline_array.astype(dtype, copy=True)
        probe_array[probe_index] = probe_array[probe_index] * factor
        return probe_array

    # ==================== Reach ====================

    def measure(
        self,
        *,
        effects: Sequence[ChannelDependentEffect],
        l_max: int,
    ) -> SpendReach:
        r"""Measure how far in time a change in spend moves the evaluated nodes.

        The evaluation window is sized by this number, so taking it from an
        effect's own declaration alone is a hazard: declare too little and the
        mediated tail falls outside the window, which returns a smaller increment
        with no indication anything was cut.  So it is measured.  Each node's
        contribution under the probe is compared against the baseline, and the
        last date that moves by more than :attr:`REACH_TOLERANCE` of that node's
        largest move fixes its reach.

        ``channel_contribution`` is measured alongside the effects and not
        assumed to inherit the model's own ``adstock.l_max``.  A custom adstock or
        saturation that reduces over ``date`` -- anything of the shape
        ``x / x.mean("date")`` -- is not a causal filter, and a plain MMM carrying
        one would otherwise be windowed silently wrong.

        Two outcomes select full-axis evaluation instead of a window: a
        perturbation that still moves the far end of the axis (reach longer than
        the axis can show), and one that moves dates *before* the perturbed one.
        The latter cannot happen through a causal filter and identifies exactly
        the reduction above, whose value depends on the whole series, so no
        window reproduces it.

        Parameters
        ----------
        effects : sequence of ChannelDependentEffect
            The effects being evaluated alongside ``channel_contribution``.
            Empty for a separable model, which still has
            ``channel_contribution`` measured.
        l_max : int
            The model's own ``adstock.l_max``, subtracted from each measured
            reach because the window already carries it.

        Returns
        -------
        SpendReach
            The window length and mode the evaluation has to use.
        """
        if self.probed is None or self.probe_index is None:
            # No date could be perturbed, so nothing about the window was
            # established.  Fall back to the only mode that is correct without a
            # measurement rather than trusting an unmeasured one.
            warnings.warn(
                "Channel spend carries no non-zero date after the first, so the "
                "reach of a spend counterfactual could not be measured.  Every "
                "period will be evaluated on the full date axis, which is "
                "correct but slower than a window.",
                UserWarning,
                stacklevel=3,
            )
            return SpendReach(effective_l_max=l_max, requires_full_axis=True)

        probed, probe_index = self.probed, self.probe_index
        measured = {
            name: self._reach_of(
                name, probed=probed, probe_index=probe_index, l_max=l_max
            )
            for name in (
                CHANNEL_CONTRIBUTION,
                *(effect.contribution_var for effect in effects),
            )
        }
        combined = TemporalReach.widest(
            [
                measured[CHANNEL_CONTRIBUTION],
                *self._reconcile_declarations(effects, measured),
            ]
        )
        return SpendReach(
            effective_l_max=l_max + combined.additional_carryover_lags,
            requires_full_axis=combined.requires_full_axis,
            measured=measured,
        )

    def _reach_of(
        self,
        name: str,
        *,
        probed: dict[str, np.ndarray],
        probe_index: int,
        l_max: int,
    ) -> TemporalReach:
        """Measure one node's reach from the probe.

        Parameters
        ----------
        name : str
            Node to measure.
        probed : dict
            The probe evaluation, per node.
        probe_index : int
            Index of the perturbed date.
        l_max : int
            The model's own ``adstock.l_max``.

        Returns
        -------
        TemporalReach
            The node's measured reach.
        """
        # Collapse samples and every non-date dim: the question is which dates
        # moved, not which cells.
        per_date = np.abs(probed[name] - self.baseline[name])
        n_dates = per_date.shape[1]
        per_date = per_date.reshape(per_date.shape[0], n_dates, -1).max(axis=(0, 2))
        largest = per_date.max()
        if largest == 0.0:
            return TemporalReach.none()

        moved = per_date > self.REACH_TOLERANCE * largest
        last_moved = int(np.flatnonzero(moved)[-1])
        if moved[:probe_index].any() or last_moved == n_dates - 1:
            return TemporalReach.full_axis()
        return TemporalReach(
            additional_carryover_lags=max(last_moved - probe_index - l_max, 0),
            requires_full_axis=False,
        )

    @staticmethod
    def _reconcile_declarations(
        effects: Sequence[ChannelDependentEffect],
        measured: Mapping[str, TemporalReach],
    ) -> list[TemporalReach]:
        """Combine each effect's measured reach with what it declared.

        A declaration wider than the measurement is honoured -- a wider window
        only costs compute, and a caller may know of a tail the probe's single
        perturbation left below tolerance.  A narrower one is rejected rather
        than quietly overridden, because it is evidence that the effect's author
        believes something false about it.  Each declaration is judged against
        that effect's own measurement.

        Parameters
        ----------
        effects : sequence of ChannelDependentEffect
            Effects whose specs are being reconciled.
        measured : mapping
            Per-node measured reach.  An effect missing from it was not measured
            to move at all.

        Returns
        -------
        list of TemporalReach
            One reconciled reach per effect, in the order given.

        Raises
        ------
        ValueError
            If an effect declares fewer carryover lags than were measured for it,
            or declares ``evaluation_mode="window"`` while being measured to need
            the full axis.
        """
        reconciled: list[TemporalReach] = []
        for effect in effects:
            own = measured.get(effect.contribution_var, TemporalReach.none())
            lags = own.additional_carryover_lags
            requires_full_axis = own.requires_full_axis

            declared = effect.declared_carryover_lags
            if declared is not None:
                if declared < lags:
                    raise ValueError(
                        f"The mu_effect {effect.label!r} declares "
                        f"additional_carryover_lags={declared}, but a change in "
                        "spend was measured still moving "
                        f"{effect.contribution_var!r} {lags} periods further "
                        "than the model's own adstock.  Evaluating on the "
                        "declared window would cut that tail off and understate "
                        f"the increment.  Raise the declaration to at least "
                        f"{lags}, or drop it and have it measured."
                    )
                lags = max(lags, declared)

            if effect.declared_evaluation_mode == "full":
                requires_full_axis = True
            elif effect.declared_evaluation_mode == "window" and requires_full_axis:
                raise ValueError(
                    f"The mu_effect {effect.label!r} declares "
                    "evaluation_mode='window', but a change in spend at one "
                    f"date was measured moving {effect.contribution_var!r} "
                    "either before that date or all the way to the end of the "
                    "date axis.  Neither is reproducible on a window, so the "
                    "declaration would produce a truncated evaluation.  Use "
                    "'auto' or 'full'."
                )

            reconciled.append(
                TemporalReach(
                    additional_carryover_lags=lags,
                    requires_full_axis=requires_full_axis,
                )
            )

        return reconciled

    # ==================== Completeness ====================

    def assert_increment_is_complete(
        self,
        *,
        effects: Sequence[ChannelDependentEffect],
        non_date_dims: Mapping[str, tuple[str, ...]],
    ) -> None:
        r"""Check the evaluated nodes account for the whole move in the predictor.

        Everything downstream rests on one identity:

        .. math::

            \Delta \mu_t = \sum_c \Delta v_{t,c} + \sum_j \Delta e_{t,j}

        -- spend moves the linear predictor through ``channel_contribution`` and
        the resolved effects, and through nothing else.  Up to here that is an
        assumption, and three separate mistakes break it silently: an effect that
        reports the wrong contribution variable, an effect that cannot be
        attributed at all, and a model-level node that reads ``channel_data``
        outside any effect.  Each drops a real part of the increment and reports
        the remainder as if it were the whole.

        So it is checked rather than assumed, against the probe evaluation
        :meth:`measure` already paid for.  A structural check -- does spend reach
        :math:`\mu` other than through the accounted nodes -- would miss the
        misreporting case, because a variable *upstream* of the true contribution
        blocks the same paths while entering :math:`\mu` through a nonlinearity
        that makes the sum above false.

        Parameters
        ----------
        effects : sequence of ChannelDependentEffect
            The effects being evaluated alongside ``channel_contribution``.
        non_date_dims : mapping
            Per node, the dimensions its evaluation carries after ``sample`` and
            ``date``.  The terms are added by name: an effect may legitimately
            drop a dimension the predictor has, and a panel model's predictor
            need not order the ones it keeps the way spend does.

        Raises
        ------
        NotImplementedError
            If the accounted nodes do not reproduce the predictor's move.
        """
        if self.probed is None or LINEAR_PREDICTOR not in self.baseline:
            return

        probed = self.probed

        def moved(name: str) -> xr.DataArray:
            return xr.DataArray(
                probed[name] - self.baseline[name],
                dims=("sample", "date", *non_date_dims[name]),
            )

        # channel_contribution carries a channel dimension the predictor does
        # not: it enters mu summed over channels.
        accounted = moved(CHANNEL_CONTRIBUTION).sum("channel")
        for effect in effects:
            accounted = accounted + moved(effect.contribution_var)

        expected = moved(LINEAR_PREDICTOR)
        expected, accounted = xr.broadcast(expected, accounted)
        # Scaled by whichever side moved more, so that a predictor which did not
        # move cannot excuse accounted nodes that did.  Taking the predictor's
        # scale alone would pass a misattribution vacuously.
        scale = max(float(np.abs(expected).max()), float(np.abs(accounted).max()))
        if scale == 0.0 or np.allclose(
            accounted.values,
            expected.values,
            rtol=0.0,
            atol=self.COMPLETENESS_TOLERANCE * scale,
        ):
            return

        accounted_names = ", ".join(
            [CHANNEL_CONTRIBUTION, *(effect.contribution_var for effect in effects)]
        )
        largest = float(np.abs(accounted - expected).max()) / scale
        raise NotImplementedError(
            "Perturbing channel spend moved the linear predictor by more than "
            f"the variables incrementality is evaluating ({accounted_names}) "
            f"account for -- by {largest:.1%} of the largest move either side "
            "makes.  Some path from spend to the response is unattributed, so "
            "the increment would report part of it as the whole.  Every "
            "mu_effect that spend reaches has to register the term it adds to "
            "the linear predictor as a Deterministic, return that name from "
            "'contribution_var_name', and opt in through 'incrementality_spec'."
        )
