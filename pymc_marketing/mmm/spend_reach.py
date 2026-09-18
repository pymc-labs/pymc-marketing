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
and divides it by spend.  Before it can, three facts about the fitted graph have
to be established, and none of them can be assumed:

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
3. **Whether one channel's spend moves another channel's contribution.**  A
   per-channel increment read off a single all-channels perturbation takes column
   *c* of ``channel_contribution`` to be a function of channel *c*'s spend alone.
   A media transform with a shared denominator makes that false for every channel
   at once, and neither of the other two measurements can see it.

The first two are read off *one* extra evaluation: perturb spend at a single
interior date on the untruncated axis and compare against the baseline.  The
third is the same perturbation restricted to one channel, taken per channel and
only when a per-channel column is going to be read.  That shared measurement is
why the questions live in the same module -- and why they live apart from
:class:`~pymc_marketing.mmm.incrementality.Incrementality`, which owns periods,
windows, spend and the link-specific reduction, and has nothing to say about any
of them.

The entry points are :meth:`SpendProbe.measure`, which returns a
:class:`SpendReach`: an evaluation-window length and whether a window is usable
at all, and :meth:`SpendProbe.mixes_channels`, which answers the third question.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors

from pymc_marketing.mmm.counterfactual import CounterfactualEvaluator, find_named_node

if TYPE_CHECKING:
    from pymc_marketing.mmm.mmm import MMM

_PKG_PREFIX = str(Path(__file__).resolve().parent.parent)
"""The ``pymc_marketing`` package directory, computed once.

Passed to ``warnings.warn(..., skip_file_prefixes=(_PKG_PREFIX,))`` so a
warning raised deep in the incrementality machinery is attributed to the
first frame outside the package rather than to a fixed number of frames up
the stack.  Call depth from here to user code differs by entry point --
``Incrementality.contribution_over_spend`` is a few frames shallower than
``mmm.summary.roas(method="incremental")`` -- so no static ``stacklevel``
integer is right for all of them at once; Python >= 3.12 is required by
``pyproject.toml``, so ``skip_file_prefixes`` is always available.
"""

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

COMPLETENESS_ACCUMULATION_SAFETY_FACTOR = 100
"""Safety factor for the float rounding accumulated across a compiled graph.

:meth:`SpendProbe.assert_increment_is_complete` compares two sums of the same
terms assembled through different chains of operations (adstock, saturation,
one reduction per ``mu_effect``), each of which rounds.  A single operation
rounds to within about one unit in the last place, but the accumulation across
many chained operations can be several orders of magnitude larger than that
one step; 100 is a generous margin for that accumulation without being so
large that it could hide a real, small unattributed contribution.
"""

_ABSENT = object()
"""Sentinel for ``inspect.getattr_static``: tells "missing" apart from any real value."""


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
    ``NotImplementedError`` rather than returning one.  Neither says anything
    about whether spend reaches the effect, so the failure is deferred: such an
    effect is left unaccounted, and :meth:`SpendProbe.assert_increment_is_complete`
    raises only if a spend path really does escape through it.  Raising eagerly
    instead would break every model that merely *owns* such an effect.  This is
    distinct from an ``AttributeError`` raised *inside* a present
    ``contribution_var_name`` (for example a property that touches a missing
    ``self`` attribute): that is a bug in the effect, not an absent attribute,
    and propagates instead of being swallowed.

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
    AttributeError
        Propagated unchanged if an effect's ``contribution_var_name`` property
        is present but raises while computing its value, rather than being
        mistaken for an absent attribute.
    """
    pymc_model = model.model
    channel_data = pymc_model[CounterfactualEvaluator.CHANNEL_DATA]
    allowed_dims = {"date", *model.dims}
    resolved: list[ChannelDependentEffect] = []

    for effect in model.mu_effects:
        label = type(effect).__name__
        # ``getattr_static`` reports whether the attribute exists at all --
        # class-level or, for a duck-typed effect, set directly on the
        # instance -- without invoking a property, so it cannot itself raise.
        # It is the only way to tell "no such attribute" (skip, the documented
        # duck-typing escape hatch) apart from "attribute present but its
        # getter raised" (a bug in the effect, which must propagate rather
        # than being reclassified as absent).
        if inspect.getattr_static(effect, "contribution_var_name", _ABSENT) is _ABSENT:
            continue
        try:
            name = effect.contribution_var_name
        except NotImplementedError:
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

    The MMM registers its linear predictor as the ``Deterministic`` ``mu``
    under both links, so the ``named_vars`` lookup below finds it.  The graph
    search is kept for models that do not register it, such as anything fitted
    before ``mu`` was registered under the identity link.

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

    Raises
    ------
    ValueError
        If the graph carries more than one node named ``mu``, which leaves the
        predictor impossible to identify by name.  This only applies to models
        that reach the graph search: where ``mu`` is registered, the
        ``named_vars`` lookup below settles it first and PyMC has already
        enforced that the name is unique.
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
    # Rooted at the deterministics too, and not only at the observed, so that
    # this search and the one CounterfactualEvaluator runs on the intervened
    # clone cover the same graph: an identity-link ``mu`` is upstream of the
    # observed either way, so the wider roots cannot lose it.
    return find_named_node(
        [observed, *pymc_model.deterministics],
        LINEAR_PREDICTOR,
        exclude=[observed],
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
        still moves after a change in spend at a single date.  Exact for a reach
        the axis could bound; a lower bound alongside ``requires_full_axis``,
        where all the probe established is that the tail had not ended by the
        last fitted date.
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
    def full_axis(cls, additional_carryover_lags: int = 0) -> TemporalReach:
        """Return the reach of a node no window can reproduce.

        Parameters
        ----------
        additional_carryover_lags : int, default=0
            The lag lower bound the probe still established, if any.  Full-axis
            evaluation sums to the axis end regardless, so this does not size
            anything; it is kept so that a declaration narrower than what was
            plainly measured can still be refused rather than accepted.

        Returns
        -------
        TemporalReach
            Full-axis, carrying whatever lower bound was measured.
        """
        return cls(
            additional_carryover_lags=additional_carryover_lags,
            requires_full_axis=True,
        )

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
    """Single-date spend perturbations, and the two facts read off them.

    A probe is an *impulse*: spend at one interior date is scaled, every other
    date left alone, and the whole graph re-evaluated on the untruncated axis.
    Comparing that against the baseline makes observable what the evaluation
    otherwise has to assume -- how far forward a change in spend still moves each
    node, whether any node moves *backwards* in time, whether the nodes being
    evaluated account for the whole move in the linear predictor, and (on
    request, see :meth:`mixes_channels`) whether one channel's spend moves
    another channel's contribution.

    One probe suffices only if every channel spends at the probed date, because
    an effect is free to read a subset of channels.  When no single date covers
    every spending channel, one probe per channel is taken instead -- see
    :meth:`_select_probe_indices` -- and each fact is read off all of them.

    Parameters
    ----------
    evaluator : CounterfactualEvaluator
        Compiled evaluator over the accounted nodes.  Reused rather than
        recompiled: each probe is one more call to a function that already
        exists.  Presumed to be a default-configuration evaluator -- spend as
        the intervention target, in ``"replace"`` mode -- since the probe's
        perturbations are spend arrays.
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
    probe_indices : list of int
        Indices of the perturbed dates, empty if no date could be perturbed --
        see :meth:`_select_probe_indices`.
    probes : dict
        Per probed date, the evaluation of the perturbed spend.

    Raises
    ------
    ValueError
        If *baseline* or any probe evaluation contains a non-finite value
        (NaN or infinity) in any node.  Checked eagerly, in ``__init__``,
        because a non-finite cell otherwise reaches :meth:`measure` or
        :meth:`assert_increment_is_complete` and misfires their guards with an
        unrelated ``IndexError`` or a misleading ``NotImplementedError`` that
        blames unattributed spend for what is really a non-finite prediction.
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
        for name, array in baseline.items():
            self._assert_finite(name, array)
        # Kept so the mixing probes (:meth:`mixes_channels`) can be taken later
        # and only if some caller needs them: they cost one more call to this
        # same compiled function per spending channel, which is worth paying
        # only where a per-channel column is actually going to be read.
        self._evaluator = evaluator
        self._baseline_array = baseline_array
        self._counterfactual_spend_factor = counterfactual_spend_factor
        self._mixes_channels: bool | None = None
        self.probe_indices = self._select_probe_indices(baseline_array)
        self.probes: dict[int, dict[str, np.ndarray]] = {
            probe_index: evaluator.evaluate_baseline(
                self._perturb(
                    baseline_array=baseline_array,
                    probe_index=probe_index,
                    counterfactual_spend_factor=counterfactual_spend_factor,
                    dtype=evaluator.channel_dtype,
                )
            )
            for probe_index in self.probe_indices
        }
        for probed in self.probes.values():
            for name, array in probed.items():
                self._assert_finite(name, array)

    @staticmethod
    def _assert_finite(name: str, array: np.ndarray) -> None:
        """Raise loudly if *array* holds a non-finite prediction.

        Checked here, in ``__init__``, rather than in :meth:`measure` or only
        on probe evaluations: the completeness check
        (:meth:`assert_increment_is_complete`) runs before :meth:`measure` at
        the call site in
        :mod:`~pymc_marketing.mmm.incrementality`, so a check placed at the
        top of :meth:`measure` would leave that misleading path in place; and
        the baseline dict (including the linear-predictor entry) and the
        no-probe path currently let a non-finite value flow silently into a
        non-finite increment without ever being evaluated as a probe.  A
        single non-finite cell is enough to misfire two different guards
        downstream: ``_reach_of`` reads a NaN ``largest`` as "every date
        compares as unmoved" and raises a bare ``IndexError``, and
        :meth:`assert_increment_is_complete` reads the mismatch as an
        unattributed spend path and raises ``NotImplementedError``.  Neither
        error names the real problem, which is why it is caught here first.

        Parameters
        ----------
        name : str
            Node the array belongs to, named in the error so the offending
            part of the graph is identifiable.
        array : np.ndarray
            Evaluation to check, ``(n_samples, n_dates, ...)``.

        Raises
        ------
        ValueError
            If any cell of *array* is not finite (NaN or infinite).
        """
        # Fast path: the overwhelming majority of calls are all-finite, and
        # `.all()` short-circuits on the first non-finite cell it finds.
        finite = np.isfinite(array)
        if finite.all():
            return

        non_finite = ~finite
        count = int(non_finite.sum())
        date_axis = 1
        reduce_axes = tuple(
            axis for axis in range(non_finite.ndim) if axis != date_axis
        )
        moved_on_date = non_finite.any(axis=reduce_axes) if reduce_axes else non_finite
        first_date_index = int(np.flatnonzero(moved_on_date)[0])
        raise ValueError(
            f"The model's evaluation of {name!r} produced {count} non-finite "
            "value" + ("s" if count != 1 else "") + " (NaN or infinity), the "
            f"first at date index {first_date_index}.  This means the model "
            "itself produced non-finite predictions -- not that spend is "
            "unattributed -- so SpendProbe cannot compare it against anything. "
            "A common cause is a custom transform (e.g. a link function or "
            "saturation) dividing zero by zero at that date for some "
            "posterior draw.  Check the model's transforms for "
            f"{name!r} before re-running incrementality."
        )

    # ==================== The perturbation ====================

    @classmethod
    def _select_probe_indices(cls, baseline_array: np.ndarray) -> list[int]:
        """Choose dates whose spend the probes can actually move -- per channel.

        A multiplicative perturbation of an all-zero cell is a no-op, and a no-op
        probe is worse than no probe: every node comes back unmoved, so the reach
        looks bounded and the completeness check compares zero against zero.
        Both guards would pass by failing to see anything.  Nor is it enough for
        the chosen date to carry spend on *some* channel: an effect is free to
        read a subset of channels, and one whose channels are all dark at the
        probed date is exactly as invisible as under an all-zero probe.  So the
        dates are chosen per channel, and a probe that could not move anything is
        reported as no probe at all.

        One date at which every spending channel is active covers every possible
        subset, so it is preferred: a single probe, exactly as cheap as before.
        Only when no such date exists does each channel get its own probe, at one
        extra evaluation per further date.

        Among the dates that carry spend, an early one is preferred.  The two
        things :meth:`measure` reads off a probe both need room on the axis: a
        forward tail has to be able to end before the last date, or its length
        cannot be bounded, and a backward move has to have at least one earlier
        date to show up in.  Placing the probe near the front maximises the
        former and costs nothing for the latter, since a reduction over ``date``
        moves *every* date rather than only nearby ones.

        Parameters
        ----------
        baseline_array : np.ndarray
            Actual spend, ``(n_dates, *extra_shape)``.  The trailing axis is the
            channel; any axes in between are summed over, so a panel channel is
            probeable wherever any of its cells spends.

        Returns
        -------
        list of int
            Indices of the dates to perturb.  Empty when nothing can be probed:
            an all-zero spend array, one with spend only at the very first or
            very last date, or one where some channel spends *only* at the
            first or last date -- no interior impulse could reach that
            channel, so no window can be trusted.
        """
        n_dates = baseline_array.shape[0]
        n_channels = baseline_array.shape[-1] if baseline_array.ndim > 1 else 1
        per_date = np.abs(baseline_array.reshape(n_dates, -1, n_channels)).sum(axis=1)
        if n_dates < 3:
            return []

        # The first and last dates are both excluded.  An impulse at the first
        # date has no earlier date to move, so a reduction over "date" would be
        # indistinguishable from a causal filter.  An impulse at the last date
        # has no later date to move either: its own reach would read as bounded
        # at zero, and the "moved past the axis end" check in _reach_of would
        # fire on every node regardless of its true reach.
        eligible = per_date[1:-1]
        probeable = eligible.any(axis=0)
        # A channel with no spend anywhere cannot be moved by the counterfactual
        # either, so nothing about it needs measuring.  One whose only spend
        # sits at an excluded edge date is different: the counterfactual will
        # move it, but no probe can.
        only_first = ~probeable & per_date[0].astype(bool)
        only_last = ~probeable & per_date[-1].astype(bool)
        if only_first.any() or only_last.any() or not probeable.any():
            return []

        covering = (eligible[:, probeable] > 0).all(axis=1)
        if covering.any():
            return [
                cls._pick_probe_date(
                    np.where(covering, eligible.sum(axis=1), 0.0), n_dates
                )
            ]
        return sorted(
            {
                cls._pick_probe_date(eligible[:, channel], n_dates)
                for channel in np.flatnonzero(probeable)
            }
        )

    @staticmethod
    def _pick_probe_date(magnitude: np.ndarray, n_dates: int) -> int:
        """Choose which interior date to probe, given a per-date magnitude.

        Prefers an early date, for the reasons :meth:`_select_probe_indices`
        gives: a forward tail needs room on the axis to end in, and a backward
        move shows up from anywhere.  Among the head dates, the largest magnitude
        wins, so the perturbation is as far above float noise as the data allows.

        Parameters
        ----------
        magnitude : np.ndarray
            Per *interior* date magnitude, shape ``(n_dates - 2,)``, whose index
            zero is date index one.  At least one entry must be non-zero.
        n_dates : int
            Length of the full date axis, which fixes how much of it counts as
            the head.

        Returns
        -------
        int
            Index into the full date axis.
        """
        head_stop = max(2, n_dates // 8) - 1
        in_head = np.flatnonzero(magnitude[:head_stop])
        chosen = in_head if in_head.size else np.flatnonzero(magnitude)
        return 1 + int(chosen[np.argmax(magnitude[chosen])])

    @classmethod
    def _select_channel_probe_indices(
        cls, baseline_array: np.ndarray
    ) -> dict[int, int] | None:
        """Choose one probe date per channel, for the mixing measurement.

        Distinct from :meth:`_select_probe_indices`, which is free to answer with
        a single date that covers every channel: a mixing probe has to perturb
        *one* channel and watch the others, so each channel needs a date of its
        own at which it spends.

        Parameters
        ----------
        baseline_array : np.ndarray
            Actual spend, ``(n_dates, *extra_shape)``, channel last.

        Returns
        -------
        dict or None
            Channel index to date index, over the channels that spend at some
            interior date.  ``None`` when some channel spends but cannot be
            probed, which leaves its separability unmeasurable rather than
            measured to hold.
        """
        n_dates = baseline_array.shape[0]
        n_channels = baseline_array.shape[-1] if baseline_array.ndim > 1 else 1
        per_date = np.abs(baseline_array.reshape(n_dates, -1, n_channels)).sum(axis=1)
        if n_dates < 3:
            return None

        eligible = per_date[1:-1]
        probeable = eligible.any(axis=0)
        # A channel that never spends cannot be moved by the counterfactual
        # either, so it has no separability to establish.  One that spends only
        # at an excluded edge date does, and no probe can establish it.
        if (per_date.any(axis=0) & ~probeable).any():
            return None
        return {
            int(channel): cls._pick_probe_date(eligible[:, channel], n_dates)
            for channel in np.flatnonzero(probeable)
        }

    @staticmethod
    def _perturb(
        *,
        baseline_array: np.ndarray,
        probe_index: int,
        counterfactual_spend_factor: float,
        dtype: str,
        channel: int | None = None,
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
        channel : int, optional
            Perturb this channel alone rather than the whole date, which is what
            makes the other channels' response to it observable.  Indexes the
            trailing axis, as ``channel_data`` lays it out.

        Returns
        -------
        np.ndarray
            A copy of *baseline_array*, one row -- or one cell of one row --
            scaled.
        """
        factor = (
            counterfactual_spend_factor if counterfactual_spend_factor != 1.0 else 0.0
        )
        probe_array = baseline_array.astype(dtype, copy=True)
        if channel is None:
            probe_array[probe_index] = probe_array[probe_index] * factor
        else:
            probe_array[probe_index, ..., channel] = (
                probe_array[probe_index, ..., channel] * factor
            )
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
        if not self.probes:
            # No date could be perturbed, so nothing about the window was
            # established.  Fall back to the only mode that is correct without a
            # measurement rather than trusting an unmeasured one.
            warnings.warn(
                "Some channel's spend carries no non-zero date after the first or "
                "before the last, so the reach of a spend counterfactual could "
                "not be measured.  "
                "Every period will be evaluated on the full date axis, which is "
                "correct but slower than a window.  With no probe, the "
                "completeness of the increment -- that the evaluated nodes "
                "account for the whole move in the linear predictor -- also "
                "goes unverified.",
                UserWarning,
                skip_file_prefixes=(_PKG_PREFIX,),
            )
            # The declarations are all there is to go on, and they are still
            # reported: full-axis evaluation sums to the axis end and so covers
            # any declared mediated tail on its own, but effective_l_max is read
            # by callers as the reach the evaluation accommodates, and answering
            # with the model's own l_max would understate what was asked for.
            # Nothing was measured, so no declaration can be contradicted and
            # neither check in _reconcile_declarations fires.
            declared = TemporalReach.widest(self._reconcile_declarations(effects, {}))
            return SpendReach(
                effective_l_max=l_max + declared.additional_carryover_lags,
                requires_full_axis=True,
            )

        # Per node, the widest reach any probe demonstrates: a probe at a date
        # where some channel is dark understates the nodes downstream of that
        # channel, and the probe that does cover it is the one that saw truth.
        measured = {
            name: TemporalReach.widest(
                self._reach_of(
                    name, probed=probed, probe_index=probe_index, l_max=l_max
                )
                for probe_index, probed in self.probes.items()
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
        lags = max(last_moved - probe_index - l_max, 0)
        if moved[:probe_index].any() or last_moved == n_dates - 1:
            # The axis could not bound the tail, but it did establish that the
            # node still moved that far, and a declaration claiming less than
            # that is still falsified.  Carrying the bound is what keeps
            # _reconcile_declarations able to say so.
            return TemporalReach.full_axis(additional_carryover_lags=lags)
        return TemporalReach(
            additional_carryover_lags=lags,
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
        that effect's own measurement, including a full-axis one, where the
        measured lags are a lower bound: nothing would be truncated there, since
        the sum runs to the axis end either way, but a declaration the probe has
        already contradicted is no more believable for that.

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
                    # Under full-axis evaluation the measurement is a lower
                    # bound and the sum runs to the axis end, so no window
                    # exists to cut the tail off.  The declaration is refused
                    # for the other reason the check has: it is false, and an
                    # author who believes it may have built the effect around
                    # it.  Saying "the increment would be understated" there
                    # would send the reader looking for a truncation that is
                    # not happening.
                    if requires_full_axis:
                        measured_as = f"at least {lags} periods further"
                        consequence = (
                            "The date axis could not bound that tail, so the "
                            "evaluation runs to the axis end and nothing is "
                            "cut; the declaration is refused because it is "
                            "false, not because the increment would come back "
                            "short."
                        )
                    else:
                        measured_as = f"{lags} periods further"
                        consequence = (
                            "Evaluating on the declared window would cut that "
                            "tail off and understate the increment."
                        )
                    raise ValueError(
                        f"The mu_effect {effect.label!r} declares "
                        f"additional_carryover_lags={declared}, but a change in "
                        "spend was measured still moving "
                        f"{effect.contribution_var!r} {measured_as} "
                        f"than the model's own adstock.  {consequence}  "
                        f"Raise the declaration to at least {lags}, or drop it "
                        "and have it measured."
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

    # ==================== Channel separability ====================

    def mixes_channels(self, *, non_date_dims: Mapping[str, tuple[str, ...]]) -> bool:
        r"""Measure whether one channel's spend moves another channel's column.

        The separable readout takes column *m* of a single all-channels
        counterfactual to be channel *m*'s unilateral counterfactual.  That holds
        only while :math:`v_{t,c}` is a function of channel *c*'s spend alone,
        and nothing in the MMM enforces it: ``forward_pass`` hands the saturation
        the whole ``(date, channel)`` tensor, so a transform with a shared
        denominator -- :math:`x_c / (1 + \sum_{c'} x_{c'})`, channels competing
        for one pool of attention -- is expressible and makes that readout wrong
        for every channel at once.

        Neither of the other guards can see it.  :meth:`measure` collapses every
        non-date dimension before comparing, so a move that stays within a date
        is invisible to it, and
        :meth:`assert_increment_is_complete` sums over channels, which is exactly
        the operation the cross-column movement is conserved under.  So it is
        measured here, and separately: one probe per spending channel, perturbing
        that channel alone at a date it spends, checking that no other channel's
        column moves.  Per channel rather than once, because mixing need not be
        symmetric -- a single probe on the channel nothing leaks into would come
        back clean.

        The cost is one call to the already-compiled evaluator per spending
        channel, so callers should ask only when a per-channel column is going to
        be read.

        Parameters
        ----------
        non_date_dims : mapping
            Per node, the dimensions its evaluation carries after ``sample`` and
            ``date``.  Only ``channel_contribution``'s entry is read, to find
            which axis the channels lie along: it is not always the last one, and
            an assumption there would silently compare the wrong slices.

        Returns
        -------
        bool
            Whether the channels have to be perturbed one at a time.  ``True``
            also when some spending channel could not be probed: correctness
            beats speed, and per-channel scenarios are right either way while
            assuming separability is silently wrong.
        """
        if self._mixes_channels is None:
            self._mixes_channels = self._measure_channel_mixing(non_date_dims)
        return self._mixes_channels

    def _measure_channel_mixing(
        self, non_date_dims: Mapping[str, tuple[str, ...]]
    ) -> bool:
        """Take the per-channel probes and read the answer off them.

        Parameters
        ----------
        non_date_dims : mapping
            Per node, the dimensions after ``sample`` and ``date``.

        Returns
        -------
        bool
            Whether any channel's perturbation moved another channel's column.
        """
        baseline_array = self._baseline_array
        n_channels = baseline_array.shape[-1] if baseline_array.ndim > 1 else 1
        if n_channels < 2:
            # One column cannot be moved by another channel's spend, and the
            # per-channel and joint perturbations coincide anyway.
            return False

        placements = self._select_channel_probe_indices(baseline_array)
        if placements is None:
            # Some channel spends only where no interior impulse can reach it,
            # so its separability is unmeasurable.  Answer conservatively: the
            # per-channel scenarios this selects are correct under mixing and
            # merely more expensive without it.
            return True

        # `+ 2` for the sample and date axes the evaluations carry in front.
        channel_axis = non_date_dims[CHANNEL_CONTRIBUTION].index("channel") + 2
        baseline = self.baseline[CHANNEL_CONTRIBUTION]
        for channel, probe_index in placements.items():
            probed = self._evaluator.evaluate_baseline(
                self._perturb(
                    baseline_array=baseline_array,
                    probe_index=probe_index,
                    counterfactual_spend_factor=self._counterfactual_spend_factor,
                    dtype=self._evaluator.channel_dtype,
                    channel=channel,
                )
            )[CHANNEL_CONTRIBUTION]
            self._assert_finite(CHANNEL_CONTRIBUTION, probed)

            moved = np.abs(probed - baseline)
            # Scaled by the largest move this probe made anywhere, mirroring
            # _reach_of: the threshold is then free of the units the
            # contribution happens to be in.
            largest = float(moved.max())
            if largest == 0.0:
                continue
            others = float(np.delete(moved, channel, axis=channel_axis).max())
            if others > self.REACH_TOLERANCE * largest:
                return True
        return False

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
        if LINEAR_PREDICTOR not in self.baseline:
            return

        # Every probe has to satisfy the identity: an unattributed path from one
        # channel is visible only under a probe at a date that channel spends.
        for probed in self.probes.values():

            def moved(
                name: str, probed: dict[str, np.ndarray] = probed
            ) -> xr.DataArray:
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
            # Scaled by whichever side moved more, so that a predictor which did
            # not move cannot excuse accounted nodes that did.  Taking the
            # predictor's scale alone would pass a misattribution vacuously.
            scale = max(float(np.abs(expected).max()), float(np.abs(accounted).max()))

            # `expected` and `accounted` are each a difference of two
            # full-magnitude evaluations, so their rounding floor is
            # eps * |node|, not eps * |move|: two sums of the same terms
            # assembled through different chains of operations (adstock,
            # saturation, one reduction per mu_effect) round independently, and
            # nothing keeps their errors from surviving the subtraction.  A
            # tolerance proportional only to `scale` -- the size of the move --
            # cannot absorb that, and under float32 (eps around 1.2e-7, a
            # supported pytensor config) it does not have to be small before
            # the accumulated rounding exceeds it.  So the floor also accounts
            # for the magnitude of the nodes themselves: the largest baseline
            # or probed value among every node the identity compares, which is
            # where cancellation inside the accounted sum sets the true floor
            # when large terms offset -- the predictor alone would understate
            # it.
            compared_names = (
                LINEAR_PREDICTOR,
                CHANNEL_CONTRIBUTION,
                *(effect.contribution_var for effect in effects),
            )
            compared_arrays = [
                array
                for name in compared_names
                for array in (self.baseline[name], probed[name])
            ]
            node_magnitude = max(
                float(np.abs(array).max()) for array in compared_arrays
            )
            # The dtype actually produced by the compared arrays, not an
            # assumption: nothing here gates on the model's floatX, and a
            # float64 model must not be loosened by a float32-sized floor.
            eps = np.finfo(np.result_type(*compared_arrays)).eps
            # Unavoidable tradeoff: under float32 a real unattributed path
            # smaller than roughly
            # COMPLETENESS_ACCUMULATION_SAFETY_FACTOR * eps * |node| / |move|
            # of the move now passes undetected.  Under float64 the floor is
            # on the order of 1e-11 relative and never binds in practice, so
            # float64 behaviour is unchanged.
            atol = max(
                self.COMPLETENESS_TOLERANCE * scale,
                COMPLETENESS_ACCUMULATION_SAFETY_FACTOR * eps * node_magnitude,
            )
            if scale == 0.0 or np.allclose(
                accounted.values,
                expected.values,
                rtol=0.0,
                atol=atol,
            ):
                continue

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
