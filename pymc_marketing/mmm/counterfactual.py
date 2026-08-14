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
"""Evaluating a fitted MMM under counterfactual spend.

:mod:`~pymc_marketing.mmm.incrementality` asks what would have happened at a
different level of spend.  Answering it means running the fitted graph again on
perturbed inputs, many times over, and this module is the part that runs it.  It
knows about windows, scenarios and compiled graphs; it knows nothing about
estimands, link functions or ROAS, which is where the dependency between the two
modules stops.

The unit of work is a **scenario**: one period's spend, perturbed for one
channel or for all of them at once, evaluated over a stretch of dates wide
enough to carry the perturbation's whole effect.  Three pieces make that up:

* :class:`EvaluationWindows` decides which dates each period is evaluated over
  and cuts every date-indexed array to match.  Once a counterfactual is
  evaluated on a window rather than on the full axis, *every* date-indexed input
  of the graph has to be cut in lockstep, or the graph is handed a window-length
  spend array and a full-length mediator array.
* :class:`CounterfactualScenarios` holds the perturbed spend together with the
  bookkeeping that says which row answers which question.
* :class:`CounterfactualEvaluator` compiles the model graph once, conditioned on
  the posterior, and evaluates the scenarios in bounded batches.

The intervention itself is expressed through :func:`pymc.do`.  The evaluator
grafts a symbolic input onto the intervened node, and only then conditions the
graph on the posterior and vectorizes it over scenarios::

    do(model, {target: intervention})            # the intervention
      -> extract_response_distribution(...)      # condition on posterior draws
      -> vectorize_graph(...)                    # batch over scenarios

``do`` states *what* is intervened on; the batching states *how many* values it
takes.  The split matters because ``do`` requires the intervention to have the
target's own dimensions, so the scenario axis can only be introduced afterwards.

Which node is intervened on decides which question the result answers.
Intervening on ``channel_data`` -- the default -- is the **spend**
counterfactual: the perturbation propagates through adstock, saturation and any
mediated effect, which is the total effect of moving spend.  Scaling
``channel_contribution`` by a zero mask instead answers **effect removal**: what
would have happened without this channel's contribution, whatever its spend was.
The two coincide only when the media transform maps zero spend to zero
contribution and no time-varying multiplier scales it, and only for a factor of
zero: for fractional factors they must differ, because saturation is nonlinear
in spend while the mask is linear in contribution.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import pytensor.xtensor as ptx
import xarray as xr
from pandas.tseries.offsets import BaseOffset
from pymc import do
from pytensor import function
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors
from pytensor.xtensor.vectorization import vectorize_graph

from pymc_marketing.pytensor_utils import extract_response_distribution

__all__ = [
    "CounterfactualEvaluator",
    "CounterfactualScenarios",
    "DateIndexedInput",
    "Estimand",
    "EvaluationWindows",
    "InterventionMode",
    "PeriodWindow",
    "find_named_node",
]

# Private: it is the key type of an internal mapping rather than a user-facing
# name, and autodoc cannot format a bare generic-tuple alias.  The underscore is
# what keeps it out of the recursive api-reference autosummary, which collects
# public module attributes by name rather than from ``__all__``.
_ScenarioKey = tuple[int, int | None]
"""Which scenario a row of :attr:`CounterfactualScenarios.spend` answers for.

``(period_idx, channel_idx)``, with ``channel_idx=None`` for the scenario that
perturbs every channel at once.  ``None`` rather than a negative sentinel because
the entry is also a channel *position*: ``-1`` is a perfectly good index into a
channel axis, meaning the last channel, and the two readings cannot be told apart
at the point of use.
"""

Estimand = Literal["per_channel", "joint"]
"""Which counterfactual an increment answers for.

``"per_channel"`` intervenes on one channel at a time and keeps a ``channel``
dimension; ``"joint"`` intervenes on every channel together and returns a single
number per period.
"""

InterventionMode = Literal["replace", "scale"]
"""How an intervention grafts onto the target node.

``"replace"`` substitutes the evaluator's input for the target outright, cutting
the target's parents.  It is the spend path: the scenario array *is* the
intervention.  ``"scale"`` multiplies the target's factual value by the
evaluator's input, so a mask of ones reproduces the baseline and a mask of zeros
removes the target's effect while everything upstream keeps its posterior value.
Scaling has to stay symbolic because an endogenous target's factual values
differ per posterior sample; there is no array a caller could hand over.
"""


def find_named_node(
    roots: Sequence[Variable],
    name: str,
    *,
    exclude: Iterable[Variable] = (),
) -> Variable | None:
    """Find the one node of a graph that carries a given name.

    Not every interesting quantity is a registered variable of a PyMC model: an
    MMM's linear predictor is a ``Deterministic`` under a log link but only an
    *anonymous intermediate carrying the name* ``mu`` under an identity link, so
    it has to be recovered from the graph instead of looked up.  Recovering it
    that way is sound only while the name picks out a single node, and graph
    traversal is unordered, so taking the first match would bind to whichever
    node happened to come up first.  Downstream that decides whether the
    increment's completeness check compares the right two quantities, and a
    wrong binding is invisible: it either refuses a correct model or passes a
    broken one.  So an ambiguous name is refused rather than guessed at.

    Parameters
    ----------
    roots : sequence of Variable
        The nodes whose ancestors are searched.  The roots themselves are part
        of the search, as :func:`~pytensor.graph.traversal.ancestors` yields
        them.
    name : str
        The name to look for.
    exclude : iterable of Variable, optional
        Nodes that may not be returned, compared by identity.  The observed
        variable belongs here whenever it could carry the requested name
        itself, as a target column named after the quantity being searched for
        would otherwise resolve to the observation.

    Returns
    -------
    Variable or None
        The single node carrying *name*, or ``None`` if the graph carries no
        such node.

    Raises
    ------
    ValueError
        If more than one distinct node carries *name*, since no choice between
        them can be made reliably.

    Notes
    -----
    One variant is out of reach here: a user who *registers* a ``Deterministic``
    named ``mu`` on an identity-link model shadows the predictor in the model's
    ``named_vars``, which callers consult before any graph search runs.  No scan
    of the graph is performed in that case, so this function never sees the
    collision.
    """
    excluded = [node for node in exclude]
    found: list[Variable] = []
    for node in ancestors(roots):
        if getattr(node, "name", None) != name:
            continue
        if any(node is other for other in excluded):
            continue
        if not any(node is other for other in found):
            found.append(node)

    if len(found) > 1:
        raise ValueError(
            f"The graph carries {len(found)} distinct nodes named {name!r}, so "
            f"{name!r} cannot be bound to a single node reliably and the "
            "increment's completeness check would have no way to tell which "
            "one it is comparing against.  Rename the intermediate node so "
            f"that only the model's own {name!r} carries the name."
        )
    return found[0] if found else None


@dataclass(frozen=True)
class PeriodWindow:
    """The stretch of fitted dates one period is evaluated over.

    Two nested date ranges, and the distinction between them is load-bearing.
    The *window* is what the graph is handed, wide enough on both sides that the
    perturbation's whole effect is computed correctly.  The *evaluation dates* are
    the subset whose differences are summed into the increment.  Both are recorded
    here, in the two forms the rest of the code needs them in -- a mask over the
    full date axis, for slicing a full-axis baseline, and the dates themselves,
    for labelling coordinates -- so that no caller has to recompute either from
    ``l_max`` and a frequency offset and hope it lands on the same answer.

    Parameters
    ----------
    start, end : pd.Timestamp
        Bounds of the period itself -- the dates the counterfactual factor is
        applied to, as opposed to the wider window it is evaluated over.
    in_window : np.ndarray
        Boolean mask over the full date axis selecting the window's dates.
    actual_dates : pd.DatetimeIndex
        Those dates, in order.
    in_eval : np.ndarray
        Boolean mask over the full date axis selecting the dates that enter the
        sum.  A subset of :attr:`in_window`, by construction.
    eval_dates : pd.DatetimeIndex
        Those dates, in order.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    in_window: np.ndarray
    actual_dates: pd.DatetimeIndex
    in_eval: np.ndarray
    eval_dates: pd.DatetimeIndex

    @classmethod
    def build(
        cls,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        dates: pd.DatetimeIndex,
        in_window: np.ndarray,
        eval_end: pd.Timestamp,
    ) -> PeriodWindow:
        """Derive both date ranges from the window mask and the carry-out end.

        Parameters
        ----------
        start, end : pd.Timestamp
            Bounds of the period itself.
        dates : pd.DatetimeIndex
            The full fitted date axis.
        in_window : np.ndarray
            Boolean mask over *dates* selecting the window.
        eval_end : pd.Timestamp
            Last date to sum: *end* itself when carryover is excluded, and
            otherwise as far as the carry-out reaches, which is *end* plus the
            window's ``l_max`` for a bounded reach and the last fitted date for a
            reach the axis could not bound.

        Returns
        -------
        PeriodWindow
            The window, with its evaluation subset intersected against it so the
            two cannot disagree.
        """
        in_eval = in_window & (dates >= start) & (dates <= eval_end)
        return cls(
            start=start,
            end=end,
            in_window=in_window,
            actual_dates=dates[in_window],
            in_eval=in_eval,
            eval_dates=dates[in_eval],
        )

    @property
    def n_actual(self) -> int:
        """How many fitted dates fall in the window."""
        return len(self.actual_dates)

    def eval_mask(self, max_window: int) -> np.ndarray:
        """Positions in the padded window that enter the sum.

        Parameters
        ----------
        max_window : int
            Padded window length every period is stacked to.

        Returns
        -------
        np.ndarray
            Boolean mask of length *max_window*.
        """
        mask = np.zeros(max_window, dtype=bool)
        mask[np.flatnonzero(self.in_eval[self.in_window])] = True
        return mask

    def offsets_within(self, first: pd.Timestamp, last: pd.Timestamp) -> np.ndarray:
        """Positions inside the window of the dates in ``[first, last]``.

        Parameters
        ----------
        first, last : pd.Timestamp
            Inclusive bounds to select.

        Returns
        -------
        np.ndarray
            Integer offsets into the padded window.  Empty when the window holds
            no fitted dates at all, which happens for a period that falls
            entirely outside the fitted range.
        """
        if self.n_actual == 0:
            return np.array([], dtype=int)
        selected = (self.actual_dates >= first) & (self.actual_dates <= last)
        return np.where(selected)[0]


@dataclass(frozen=True)
class EvaluationWindows:
    """Per-period evaluation windows, and the cutting of arrays to fit them.

    A counterfactual is evaluated on a window around each period rather than on
    the whole date axis, because the alternative is re-evaluating the entire
    series once per (period, channel) pair.  The window reaches back ``l_max``
    for carry-in history and forward ``l_max`` for carry-out, and is *clamped* to
    the fitted dates rather than padded out to those ideal bounds.  Clamping is
    what makes a windowed evaluation agree with a full-axis one on every
    evaluated date: at the start of the data there is no history to supply, the
    graph's own adstock already pads with zeros there, and padding instead would
    inject synthetic history -- inert for spend, but not for an effect whose
    contribution at zero spend is its own intercept.

    The dates that are *summed* run from the period's own start to its carry-out
    end, and the estimand that fixes is a forward-looking one: a period's
    increment is what moving its spend does to that period and to the dates after
    it.  Under full-axis evaluation, selected for a node whose value depends on
    the whole series, the perturbation also moves dates *before* the period, and
    those moves are deliberately left out of the sum.  Summing them would make
    the periods overlap in a quantity every caller reads as a decomposition, and
    it is not what "this period's increment" is taken to mean.  A date-reducing
    node is therefore evaluated on the whole series, as its value requires, and
    still attributed forwards.

    Parameters
    ----------
    windows : list of PeriodWindow
        One per period, in period order.
    max_window : int
        Longest window.  Shorter ones are right-padded to it so they stack for
        batched evaluation; a causal filter cannot reach back into that padding
        from any evaluated date.
    dates : pd.DatetimeIndex
        The full fitted date axis the windows index into.
    """

    windows: list[PeriodWindow]
    max_window: int
    dates: pd.DatetimeIndex

    @classmethod
    def build(
        cls,
        *,
        periods: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
        dates: pd.DatetimeIndex,
        l_max: int,
        freq_offset: BaseOffset,
        full_axis: bool = False,
        include_carryover: bool = True,
    ) -> EvaluationWindows:
        """Work out the window of each period, and which of its dates are summed.

        Parameters
        ----------
        periods : sequence of (pd.Timestamp, pd.Timestamp)
            Period ``(start, end)`` pairs.
        dates : pd.DatetimeIndex
            All dates from the fitted data.
        l_max : int
            Evaluation-window half-length, already widened for any mediated
            path.
        freq_offset : pd.DateOffset
            Calendar-aware frequency offset.
        include_carryover : bool, default=True
            Whether the *evaluation* range reaches past the period to pick up its
            carry-out.  Independent of the window, which always does: the two
            lengths do different jobs and collapsing them would change the
            numbers inside the period too.
        full_axis : bool, default=False
            Evaluate every period on the complete fitted date axis, for an
            effect whose value depends on the whole series.  Expressed as a
            window spanning every date rather than as a separate code path, so
            scenario building, input cutting and period aggregation stay
            identical: each period still perturbs only its own dates and still
            sums its increment over its own evaluation mask.

        Returns
        -------
        EvaluationWindows
            The windows, and the length they stack to.
        """
        windows: list[PeriodWindow] = []
        for start, end in periods:
            if full_axis:
                in_window = np.ones(len(dates), dtype=bool)
                # Full-axis mode is selected precisely when no number could be
                # put on some node's reach: the probe saw it still moving the
                # last fitted date, or moving dates before the perturbed one.
                # Stopping the sum at end + l_max would then cut a tail that was
                # measured *not* to have ended, which is the silent under-count
                # the measurement exists to prevent.  Summing to the axis end
                # instead costs nothing in correctness: for a date the
                # perturbation does not reach, the counterfactual and the
                # baseline are the same computation on the same inputs, so the
                # extra summands are exactly zero.
                eval_end = dates[-1]
            else:
                # Reach back l_max for carry-in context and forward l_max so
                # carryover is captured; the eval mask decides what is summed.
                in_window = (dates >= start - l_max * freq_offset) & (
                    dates <= end + l_max * freq_offset
                )
                eval_end = end + l_max * freq_offset
            windows.append(
                PeriodWindow.build(
                    start=start,
                    end=end,
                    dates=dates,
                    in_window=in_window,
                    eval_end=eval_end if include_carryover else end,
                )
            )

        return cls(
            windows=windows,
            max_window=max(window.n_actual for window in windows),
            dates=dates,
        )

    def __len__(self) -> int:
        """Count the periods."""
        return len(self.windows)

    def cut(self, values: np.ndarray, dtype: str) -> np.ndarray:
        """Cut a date-first array into one padded window per period.

        Parameters
        ----------
        values : np.ndarray
            Array with ``date`` as its first axis, shape ``(n_dates, *rest)``.
        dtype : str
            NumPy dtype of the output.

        Returns
        -------
        np.ndarray
            Shape ``(n_periods, max_window, *rest)``.  Positions beyond a
            window's own length are zero; they exist only so windows of
            different lengths stack, and no evaluated date reaches them, since
            the filters involved look backwards.
        """
        cut = np.zeros((len(self), self.max_window, *values.shape[1:]), dtype=dtype)
        for period_idx, window in enumerate(self.windows):
            cut[period_idx, : window.n_actual] = values[window.in_window].astype(dtype)
        return cut

    def time_index(self, dtype: str) -> np.ndarray:
        """Build the positions along the fitted axis that each window covers.

        HSGP-based latent variables such as ``media_temporal_latent_multiplier``
        evaluate their basis functions at these positions, so a window has to say
        where on the axis it sits rather than counting from zero.

        Parameters
        ----------
        dtype : str
            NumPy dtype for the output.

        Returns
        -------
        np.ndarray
            Shape ``(n_periods, max_window)``.  Indices may run past
            ``n_dates`` where a short window is padded out.
        """
        rows: list[np.ndarray] = []
        for window in self.windows:
            start_idx = (
                int(self.dates.searchsorted(window.actual_dates[0]))
                if window.n_actual > 0
                else 0
            )
            rows.append(np.arange(start_idx, start_idx + self.max_window, dtype=dtype))
        return np.stack(rows, axis=0)

    def build_scenarios(
        self,
        *,
        baseline_array: np.ndarray,
        counterfactual_spend_factor: float,
        dtype: str,
        channel_axis: int | None,
        n_channels: int,
        estimand: Estimand,
    ) -> CounterfactualScenarios:
        """Build the perturbed spend arrays that have to be evaluated.

        Two scenario layouts are produced, and which one applies is the whole
        difference between a plain MMM and a mediated one:

        * ``channel_axis is None`` -- **separable**.  One all-channels
          perturbation per period.  Because :math:`v_{t,c}` depends on channel
          *c*'s spend alone, column *m* of that single scenario already *is*
          channel *m*'s counterfactual, so every channel key points at the same
          row and the joint scenario costs nothing extra.
        * ``channel_axis`` given -- **per channel**.  One perturbation per
          (period, channel), because a mediated effect mixes channels before
          reaching the response and no per-channel column survives to be read
          off.  Asked for the joint estimand this collapses back to one
          perturbation per period: the per-channel rows would otherwise be
          built, evaluated and never read.

        Parameters
        ----------
        baseline_array : np.ndarray
            Actual channel spend, shape ``(n_dates, *extra_shape)``.
        counterfactual_spend_factor : float
            Multiplicative factor for counterfactual spend.
        dtype : str
            NumPy dtype for the output array.
        channel_axis : int or None
            Axis of ``channel`` within ``baseline_array``'s non-date axes, or
            ``None`` to perturb all channels at once.  Panel models lay
            ``channel_data`` out as ``(date, *custom_dims, channel)``, so this is
            not always zero.
        n_channels : int
            Number of channels.
        estimand : {"per_channel", "joint"}
            Which scenarios are going to be read.  Only those get built: the
            difference is a factor of ``n_channels`` in every downstream cost for
            a mediated model asked for the joint number.

        Returns
        -------
        CounterfactualScenarios
            Perturbed spend plus the bookkeeping needed to find the row for a
            given ``_ScenarioKey`` and to broadcast per-period arrays over
            scenarios.
        """
        separable = channel_axis is None
        # Index prefix selecting a single channel out of a padded window: dates
        # come first, then any custom dims that sit ahead of channel.
        channel_prefix: tuple = (
            () if channel_axis is None else (slice(None),) * channel_axis
        )
        # Actual spend, cut into one window per period.  The counterfactual
        # factor is applied on top, per scenario.
        windowed = self.cut(baseline_array, dtype)

        spend: list[np.ndarray] = []
        period_index: list[int] = []
        rows: dict[_ScenarioKey, int] = {}

        for period_idx, window in enumerate(self.windows):
            target_offsets = window.offsets_within(window.start, window.end)

            perturbed: list[int | None] = (
                [None] if separable or estimand == "joint" else list(range(n_channels))
            )
            for channel in perturbed:
                padded = windowed[period_idx].copy()
                if channel is None:
                    padded[target_offsets] *= counterfactual_spend_factor
                else:
                    padded[(target_offsets, *channel_prefix, channel)] *= (
                        counterfactual_spend_factor
                    )
                rows[(period_idx, channel)] = len(spend)
                spend.append(padded)
                period_index.append(period_idx)

            if separable:
                # One row serves every channel, and it is the joint row too.
                joint_row = rows[(period_idx, None)]
                for channel in range(n_channels):
                    rows[(period_idx, channel)] = joint_row

        return CounterfactualScenarios(
            spend=np.stack(spend, axis=0),
            period_index=np.asarray(period_index, dtype=int),
            rows=rows,
        )


class DateIndexedInput(NamedTuple):
    """A date-indexed ``pm.Data`` replaced by a batched evaluator input.

    Parameters
    ----------
    name : str
        Name of the data variable in the model.
    values : np.ndarray
        Its fitted values, in the model's own dimension order.  Read from the
        model's own shared variable rather than from ``fit_data``, because an
        effect's data need not appear there -- the funnel example builds the
        model from an ``xr.Dataset`` carrying the mediator's inputs, then fits
        from a frame that does not.
    dtype : str
        Dtype the compiled evaluator expects.
    dims : tuple of str
        Its dimensions, in the model's order.
    date_axis : int
        Position of ``date`` within :attr:`dims`.  Stored rather than assumed to
        be zero: an effect is free to declare its data as ``("country", "date")``,
        and cutting the wrong axis would either raise a misleading length error
        or, when the two lengths happen to coincide, quietly cut the panel up
        instead of the calendar.
    """

    name: str
    values: np.ndarray
    dtype: str
    dims: tuple[str, ...]
    date_axis: int

    def cut(self, windows: EvaluationWindows) -> np.ndarray:
        """Cut this input into one padded window per period.

        Parameters
        ----------
        windows : EvaluationWindows
            The windows to cut to.

        Returns
        -------
        np.ndarray
            Shape ``(n_periods, *dims)`` with ``date`` cut to
            ``windows.max_window`` and left where the model expects it, since
            the compiled graph reads the input by dimension order.
        """
        cut = windows.cut(np.moveaxis(self.values, self.date_axis, 0), self.dtype)
        # (period, date, *rest) back to (period, *dims).
        return np.moveaxis(cut, 1, self.date_axis + 1)


@dataclass(frozen=True)
class CounterfactualScenarios:
    """Perturbed spend arrays for every scenario that has to be evaluated.

    A *scenario* is a (period, perturbed channel) pair.  When no ``mu_effect``
    stands between spend and the response, a single all-channels perturbation
    per period suffices and every channel reads its own column out of it, so all
    of a period's keys point at the same row -- see
    :meth:`EvaluationWindows.build_scenarios`.

    Parameters
    ----------
    spend : np.ndarray
        Counterfactual ``channel_data``, shape
        ``(n_scenarios, max_window, *extra_shape)``.
    period_index : np.ndarray
        Period each scenario belongs to, shape ``(n_scenarios,)``.  Indexes any
        per-period array (a windowed data variable, a ``time_index`` row) up to
        the scenario axis.
    rows : dict
        Maps a ``_ScenarioKey`` to a row of :attr:`spend`.

    See Also
    --------
    PeriodWindow : Which dates each period is evaluated over, and which are summed.
    """

    spend: np.ndarray
    period_index: np.ndarray
    rows: dict[_ScenarioKey, int]


class CounterfactualEvaluator:
    """Compiled batched evaluator for the nodes a counterfactual intervention reaches.

    Applies the intervention with :func:`pymc.do`, conditions the intervened
    model's graph on posterior draws, swaps every date-indexed ``pm.Data`` the
    evaluation needs for a batched input, and compiles *one* function returning
    all requested nodes.  Extracting the nodes together matters:
    ``channel_contribution`` and a mediated effect read the same spend data
    through the same adstock, and a single extraction keeps that subgraph shared
    instead of computing it once per node.

    The intervention is a single ``(target, mode)`` pair for now.  The
    direct-effect estimand -- holding a mediator at its factual value while
    spend moves -- needs *simultaneous* interventions and will generalize this
    to a mapping; the pair is threaded through the private helpers as one unit
    so that change stays local.

    The batched inputs are what make a *window* evaluation possible.  Every
    date-indexed input in the graph has to be cut to the same window in lockstep,
    or the graph is handed a ``max_window``-long spend array and an
    ``n_dates``-long mediator array.  This class discovers those inputs from the
    graph and cuts them itself.

    Every result is transposed to ``(sample, date, *non_date_dims)`` before it
    is returned.  A node's own axis order is whatever the operations that built
    it happened to produce -- a panel model's linear predictor comes out
    ``(country, date)`` while its ``channel_contribution`` comes out
    ``(date, country, channel)`` -- and callers that add those two together
    cannot each rediscover that.

    Parameters
    ----------
    pymc_model : pm.Model
        The fitted model whose graph is evaluated.
    posterior : xr.Dataset
        Posterior samples (already subsampled).  Draws are flattened into a
        single ``sample`` axis in chain-major order.
    response_vars : sequence of str or Variable
        Nodes to evaluate, in the order the results are keyed by.  A node is
        accepted directly for a quantity the model does not register, such as
        the linear predictor under an identity link.
    frozen_deterministics : list of str
        Deterministics to hold at their posterior values instead of recomputing.
    dates : pd.DatetimeIndex
        Dates of the fitted data, used to validate the discovered inputs.
    intervention_target : str, default ``"channel_data"``
        Name of the model variable the counterfactual intervenes on: a data
        variable or a deterministic carrying a leading ``date`` dimension.
        Random variables are refused -- observed ones because intervening would
        silently delete the likelihood, free ones because a parameter is
        conditioned on, not intervened on.
    intervention_mode : InterventionMode, default ``"replace"``
        How the intervention grafts onto the target; see
        :data:`InterventionMode`.  Under ``"scale"``, asking for the target
        itself as a response variable returns the *intervened* value, symmetric
        with ``"replace"`` where the evaluator's input is that value.

    Attributes
    ----------
    non_date_dims : dict
        Per response variable, the dimensions of its result after ``sample`` and
        ``date``, which is the node's own dimension order with ``date`` removed.
    windowed_data_vars : tuple of str
        Date-indexed ``pm.Data`` variables discovered in the graph and cut
        alongside the intervention values, excluding the intervention target and
        ``time_index``.  Under an endogenous target this includes
        ``channel_data`` itself, held at its factual values.
    target_dtype : dtype
        Dtype the intervention values are cast to; :attr:`channel_dtype` is its
        alias for the default spend target.

    Raises
    ------
    ValueError
        If the intervention target does not exist, is a random variable, is
        frozen, is not of a floating dtype, or lacks a leading ``date``
        dimension; if no response variable depends on the target; if a
        discovered date-indexed input does not span the fitted date axis; if
        a response variable has no name (an anonymous node such as a raw
        arithmetic expression, unless ``.name`` was set on it); or if two
        response variables share a name.
    """

    CHANNEL_DATA = "channel_data"
    TIME_INDEX = "time_index"
    BATCH_DIM = "__batch__"
    """Dimension the scenario axis is carried on through the vectorized graph."""

    MAX_BATCH_ELEMENTS = 32_000_000
    """Output elements one evaluation may produce before it is split in two.

    Roughly a quarter of a gigabyte in float64, which leaves room for the
    intermediates a batch of that size implies while keeping the number of
    evaluations -- and so the per-call overhead -- small for everything but the
    largest runs.  Not a hard memory bound, which cannot be had without knowing
    the graph: a bound on the term that grows with the scenario count.
    """

    def __init__(
        self,
        *,
        pymc_model,
        posterior: xr.Dataset,
        response_vars: Sequence[str | Variable],
        frozen_deterministics: list[str],
        dates: pd.DatetimeIndex,
        intervention_target: str = CHANNEL_DATA,
        intervention_mode: InterventionMode = "replace",
    ) -> None:
        # Results are keyed by name whether the caller asked by name or handed
        # over the node: the linear predictor is only a registered variable
        # under a log link, but it is named "mu" either way.
        self.response_vars = tuple(
            var if isinstance(var, str) else var.name for var in response_vars
        )
        self._validate_response_names(self.response_vars)
        self.intervention_target = intervention_target
        self.intervention_mode: InterventionMode = intervention_mode
        target_var = self._validate_target(
            pymc_model,
            target=intervention_target,
            mode=intervention_mode,
            frozen_deterministics=frozen_deterministics,
        )
        # ``do`` clones the model, shared variables included, so from here on
        # every lookup goes through ``do_model``: a variable taken from
        # ``pymc_model`` is not part of the intervened graph.  That includes
        # response variables handed over as raw nodes, which are re-resolved by
        # name against the clone.
        do_model, placeholder = self._intervened_model(
            pymc_model, target_var=target_var, mode=intervention_mode
        )
        graphs: list = extract_response_distribution(
            pymc_model=do_model,
            idata=xr.DataTree.from_dict({"/posterior": posterior}),
            response_variable=[
                self._resolve_response(
                    do_model,
                    var,
                    target=intervention_target,
                    mode=intervention_mode,
                )
                for var in response_vars
            ],
            frozen_deterministics=frozen_deterministics,
        )
        n_samples = posterior.sizes["chain"] * posterior.sizes["draw"]
        # A response the intervention has cut every random variable out of --
        # the target itself under "replace", say -- comes back without a
        # "sample" axis.  Broadcasting it keeps the promised result layout.
        graphs = [
            graph
            if "sample" in graph.type.dims
            else graph.expand_dims(sample=n_samples)
            for graph in graphs
        ]
        graph_ancestors = set(ancestors(graphs))
        if placeholder not in graph_ancestors:
            raise ValueError(
                f"None of the response variables {list(self.response_vars)} "
                f"depend on the intervention target {intervention_target!r}; "
                "every counterfactual would equal the baseline."
            )

        self.non_date_dims = {
            name: self._non_date_dims(graph)
            for name, graph in zip(self.response_vars, graphs, strict=True)
        }
        # Output elements one scenario costs per date of its window, summed over
        # the evaluated nodes.  Sizes the evaluation batches.
        self._elements_per_date = n_samples * sum(
            int(np.prod([len(do_model.coords[dim]) for dim in dims], dtype=int))
            for dims in self.non_date_dims.values()
        )

        # The intervention values are the evaluator's first input, whatever the
        # target: perturbed spend under the default, a mask under "scale".
        self.target_dtype = placeholder.dtype
        replace: dict = {placeholder: self._batched(placeholder, intervention_target)}
        func_inputs: list = [replace[placeholder]]

        # time_index: only replaced when the graph actually reads it (with
        # time_varying_intercept but not time_varying_media it is unused, and
        # passing it would raise UnusedInputError).
        self.time_dtype: str | None = None
        if (
            self.TIME_INDEX in do_model.named_vars
            and do_model[self.TIME_INDEX] in graph_ancestors
        ):
            time_index = do_model[self.TIME_INDEX]
            self.time_dtype = time_index.dtype
            replace[time_index] = self._batched(time_index, "time_index")
            func_inputs.append(replace[time_index])

        # Any other date-indexed pm.Data the graph reads.  For a plain MMM there
        # are none; a mediated effect brings its own (an exogenous budget, a
        # category-demand series), and an endogenous target turns spend itself
        # into one, held at its factual values.  Discovery is by graph traversal
        # rather than by declaration so an effect cannot forget to mention one.
        self._aux: list[DateIndexedInput] = []
        for data in self._date_indexed_data(
            do_model,
            graph_ancestors,
            handled={intervention_target, self.TIME_INDEX},
        ):
            values = np.asarray(data.eval())
            dims = tuple(do_model.named_vars_to_dims[data.name])
            date_axis = dims.index("date")
            if values.shape[date_axis] != len(dates):
                raise ValueError(
                    f"Date-indexed data variable {data.name!r} has "
                    f"{values.shape[date_axis]} entries along its 'date' "
                    f"dimension but the fitted data has {len(dates)} dates.  "
                    "Incrementality needs it to span the fitted date axis so it "
                    "can be cut to a window alongside spend."
                )
            batched = self._batched(data, data.name)
            replace[data] = batched
            func_inputs.append(batched)
            self._aux.append(
                DateIndexedInput(
                    name=data.name,
                    values=values,
                    dtype=data.dtype,
                    dims=dims,
                    date_axis=date_axis,
                )
            )

        # Canonical output layout, so that a caller adding two nodes together
        # does not have to reconcile the axis order each of them happens to
        # carry.  Transposing inside the graph rather than on the results keeps
        # it a view where the backend can make it one.
        outputs = [
            out.transpose(self.BATCH_DIM, "sample", "date", *self.non_date_dims[name])
            for name, out in zip(
                self.response_vars,
                vectorize_graph(graphs, replace=replace),
                strict=True,
            )
        ]
        self._evaluator = function(func_inputs, outputs)

    @property
    def windowed_data_vars(self) -> tuple[str, ...]:
        """Names of the auxiliary date-indexed inputs discovered in the graph."""
        return tuple(aux.name for aux in self._aux)

    @property
    def channel_dtype(self) -> str:
        """Alias of :attr:`target_dtype` under the default spend target."""
        return self.target_dtype

    @staticmethod
    def _validate_response_names(response_names: tuple[str | None, ...]) -> None:
        """Refuse response names that would resolve or key results wrongly.

        Runs on the materialized names, before any graph work: an unnamed node
        would otherwise fall through to the graph scan in
        :meth:`_resolve_response` and match whichever anonymous intermediate
        the traversal happens to reach first, and two responses sharing a name
        would collapse last-wins in the name-keyed result dicts, silently
        dropping one computed output.

        Parameters
        ----------
        response_names : tuple of str or None
            The materialized response names, one per requested response
            variable, in the order they were requested.

        Raises
        ------
        ValueError
            If any name is ``None``, or if two response variables share a
            name.
        """
        first_seen: dict[str, int] = {}
        for position, name in enumerate(response_names):
            if name is None:
                raise ValueError(
                    f"Response variable at position {position} has no name; "
                    "set '.name' on the variable or pass a registered "
                    "variable name instead."
                )
            if name in first_seen:
                raise ValueError(
                    f"Response variable name {name!r} is used more than "
                    f"once, at positions {first_seen[name]} and {position}; "
                    "give each response variable a distinct name."
                )
            first_seen[name] = position

    @staticmethod
    def _validate_target(
        pymc_model,
        *,
        target: str,
        mode: str,
        frozen_deterministics: list[str],
    ) -> Variable:
        """Refuse intervention targets the machinery cannot honestly evaluate.

        Parameters
        ----------
        pymc_model : pm.Model
            The model the target is looked up in.
        target : str
            Name of the variable to intervene on.
        mode : str
            The requested intervention mode, checked against
            :data:`InterventionMode`.
        frozen_deterministics : list of str
            Deterministics held at their posterior values; the target must not
            be among them.

        Returns
        -------
        Variable
            The target variable.

        Raises
        ------
        ValueError
            See the class docstring; every refusal names the target.
        """
        if mode not in ("replace", "scale"):
            raise ValueError(
                f"Unknown intervention_mode {mode!r}; expected 'replace' or 'scale'."
            )
        if target not in pymc_model.named_vars:
            raise ValueError(
                f"Intervention target {target!r} is not a variable of the "
                f"model, which has {sorted(pymc_model.named_vars)}."
            )
        target_var = pymc_model[target]
        if target_var in pymc_model.observed_RVs:
            raise ValueError(
                f"Intervention target {target!r} is the model's observed "
                "variable; intervening on it would silently remove the "
                "likelihood.  Intervene on the quantity it observes instead."
            )
        if target_var in pymc_model.free_RVs:
            raise ValueError(
                f"Intervention target {target!r} is a random variable.  "
                "Interventions apply to data variables and deterministics; a "
                "parameter is conditioned on its posterior, not intervened on."
            )
        if target in frozen_deterministics:
            raise ValueError(
                f"Cannot intervene on {target!r}: it is held at its posterior "
                "values through frozen_deterministics, and a node cannot be "
                "both frozen and intervened on."
            )
        if np.dtype(target_var.dtype).kind != "f":
            raise ValueError(
                f"A counterfactual intervention on {target!r} requires values "
                f"of float type, got {target_var.dtype}"
            )
        dims = getattr(target_var.type, "dims", None)
        if not dims or dims[0] != "date":
            raise ValueError(
                f"Intervention target {target!r} must carry a leading 'date' "
                f"dimension, got dims {dims}.  Windowed evaluation cuts the "
                "intervention values along their first axis."
            )
        return target_var

    @staticmethod
    def _intervened_model(pymc_model, *, target_var: Variable, mode: InterventionMode):
        """Apply the intervention and return the intervened model with its input.

        Parameters
        ----------
        pymc_model : pm.Model
            The model to intervene on.  Never modified: ``do`` clones it.
        target_var : Variable
            The validated target variable.
        mode : InterventionMode
            How the placeholder grafts onto the target.

        Returns
        -------
        tuple of (pm.Model, XTensorVariable)
            The intervened model and the symbolic input the intervention values
            flow through.  The placeholder carries the target's own dimensions;
            the scenario axis is only introduced by the later batch
            vectorization, which is why the intervention and the batching are
            separate steps.
        """
        placeholder = ptx.xtensor(
            name=f"{target_var.name}_intervention",
            dtype=target_var.dtype,
            shape=target_var.type.shape,
            dims=target_var.type.dims,
        )
        value = placeholder if mode == "replace" else target_var * placeholder
        with warnings.catch_warnings():
            # The scale-mode intervention references the target on purpose:
            # ``target * mask`` is what holds the factual value in the graph.
            # pymc warns about the self-reference and registers the intervened
            # node as ``do_<target>``.
            warnings.filterwarnings(
                "ignore",
                message="Intervention expression references the variable "
                "that is being intervened",
            )
            do_model = do(pymc_model, {target_var.name: value})
        # ``do`` renames a replace-mode placeholder to the target's own name,
        # in place, so nothing may key on ``placeholder.name`` afterwards.
        return do_model, placeholder

    @staticmethod
    def _resolve_response(
        do_model,
        var: str | Variable,
        *,
        target: str,
        mode: InterventionMode,
    ) -> Variable:
        """Resolve one requested response variable against the intervened model.

        Parameters
        ----------
        do_model : pm.Model
            The intervened model.
        var : str or Variable
            The requested response.  A raw Variable is a node of the *original*
            model's graph and only its name carries over to the clone.
        target : str
            The intervention target's name.
        mode : InterventionMode
            The intervention mode; decides what the target's own name means.

        Returns
        -------
        Variable
            The corresponding node of the intervened model.

        Raises
        ------
        ValueError
            If the requested response is neither registered in the intervened
            model nor recoverable from its graph, or if its name is carried by
            more than one node of that graph.
        """
        name = var if isinstance(var, str) else var.name
        if name is None:
            # Defense in depth: __init__ already refuses an unnamed response
            # before any graph work happens, but were this guard skipped, the
            # scan below would match ``getattr(node, "name", None) == None``
            # against the first anonymous node the traversal happens to reach,
            # not against the requested variable at all.
            raise ValueError(
                "Response variable has no name; set '.name' on the variable "
                "or pass a registered variable name instead."
            )
        if mode == "scale" and name == target:
            # Under "scale" the intervened model registers *both* the factual
            # computation, still under the target's name, and the intervened
            # node, renamed "do_<target>".  A caller asking for the target of
            # an intervention wants the intervened value; the plain lookup
            # below would silently return the factual one, and every increment
            # would be zero.
            return do_model[f"do_{target}"]
        if name in do_model.named_vars:
            return do_model[name]
        # An anonymous node, such as the linear predictor under an identity
        # link: the clone carries a node of the same name, recovered the same
        # way spend_reach.linear_predictor found the original.
        observed = list(do_model.observed_RVs)
        resolved = find_named_node(
            observed + list(do_model.deterministics),
            name,
            exclude=observed,
        )
        if resolved is None:
            raise ValueError(
                f"Response variable {name!r} was not found in the intervened "
                "model's graph."
            )
        return resolved

    @staticmethod
    def _non_date_dims(graph) -> tuple[str, ...]:
        """Return the dimensions of one extracted response graph, less the two known ones.

        Read from the graph rather than from ``named_vars_to_dims`` because the
        graph is what gets evaluated: a node the model does not register has no
        entry there, and a registered one can still be laid out differently from
        its declaration once :func:`extract_response_distribution` has added the
        posterior's ``sample`` axis.

        Parameters
        ----------
        graph : Variable
            One extracted response graph, carrying ``sample`` and ``date``.

        Returns
        -------
        tuple of str
            The remaining dimensions, in the graph's own order.
        """
        return tuple(d for d in graph.type.dims if d not in ("sample", "date"))

    @staticmethod
    def _batched(variable, name: str):
        """Return a batched xtensor standing in for a date-indexed input."""
        return ptx.xtensor(
            name=f"{name}_batched",
            dtype=variable.dtype,
            shape=(None, *variable.type.shape),
            dims=(CounterfactualEvaluator.BATCH_DIM, *variable.type.dims),
        )

    @staticmethod
    def _date_indexed_data(
        pymc_model, graph_ancestors: set, *, handled: set[str]
    ) -> list:
        """Find the date-indexed ``pm.Data`` the graph reads, beyond those handled above.

        Parameters
        ----------
        pymc_model : pm.Model
            The model being evaluated.
        graph_ancestors : set
            Ancestors of the extracted response graphs.
        handled : set of str
            Names already covered by dedicated inputs: the intervention target
            and ``time_index``.

        Returns
        -------
        list
            The data variables, in the model's declaration order so that the
            compiled signature is deterministic.
        """
        return [
            data
            for data in pymc_model.data_vars
            if data in graph_ancestors
            and data.name not in handled
            and "date" in pymc_model.named_vars_to_dims.get(data.name, ())
        ]

    def evaluate_baseline(self, target_values: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate every node on one set of intervention values, over the full date axis.

        Parameters
        ----------
        target_values : np.ndarray
            Values for the intervention target, shape ``(n_dates, *extra_shape)``.
            Under the default spend target this is the actual spend; under
            ``"scale"`` it is the mask, and ones reproduce the factual baseline.

        Returns
        -------
        dict
            Per response variable, an array of shape
            ``(n_samples, n_dates, *non_date_dims)``.
        """
        args: list[np.ndarray] = [target_values[np.newaxis].astype(self.target_dtype)]
        if self.time_dtype is not None:
            args.append(
                np.arange(len(target_values))[np.newaxis].astype(self.time_dtype)
            )
        args.extend(aux.values[np.newaxis].astype(aux.dtype) for aux in self._aux)
        return {
            name: out[0]
            for name, out in zip(
                self.response_vars, self._evaluator(*args), strict=True
            )
        }

    def evaluate_counterfactual(
        self,
        scenarios: CounterfactualScenarios,
        *,
        windows: EvaluationWindows,
        batch_size: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluate every node on all counterfactual scenarios.

        Auxiliary date-indexed inputs are cut here rather than by the caller,
        then broadcast from per-period to per-scenario through
        :attr:`CounterfactualScenarios.period_index`.

        Scenarios are evaluated in batches.  The graph's intermediates are what
        makes the difference: an adstock of length ``l_max`` materializes a
        lagged copy per lag, so a batch's working set is a multiple of its
        output, and a mediated per-channel run over many periods can ask for far
        more of that at once than it has to.  Batches are concatenated in
        scenario order, so :attr:`CounterfactualScenarios.rows` keeps addressing
        the same results.

        Parameters
        ----------
        scenarios : CounterfactualScenarios
            Perturbed spend and its scenario bookkeeping.
        windows : EvaluationWindows
            The windows the scenarios were built for.
        batch_size : int, optional
            Scenarios per evaluation.  Defaults to whatever keeps a batch's
            outputs under :attr:`MAX_BATCH_ELEMENTS`, and is exposed so a test
            can pin one batch against many.

        Returns
        -------
        dict
            Per response variable, an array of shape
            ``(n_scenarios, n_samples, max_window, *non_date_dims)``.
        """
        args: list[np.ndarray] = [scenarios.spend]
        if self.time_dtype is not None:
            args.append(windows.time_index(self.time_dtype)[scenarios.period_index])
        args.extend(aux.cut(windows)[scenarios.period_index] for aux in self._aux)

        n_scenarios = len(scenarios.spend)
        if batch_size is None:
            batch_size = self._batch_size(n_scenarios, windows.max_window)
        batches = [
            self._evaluator(*(arg[start : start + batch_size] for arg in args))
            for start in range(0, n_scenarios, batch_size)
        ]
        return {
            name: np.concatenate([batch[idx] for batch in batches], axis=0)
            for idx, name in enumerate(self.response_vars)
        }

    def _batch_size(self, n_scenarios: int, max_window: int) -> int:
        """Choose how many scenarios one evaluation may cover.

        Parameters
        ----------
        n_scenarios : int
            Total scenarios to evaluate; a batch never needs to exceed it.
        max_window : int
            Padded window length, one of the factors in a scenario's size.

        Returns
        -------
        int
            At least one, so a single scenario larger than the budget is still
            attempted rather than refused.
        """
        per_scenario = max(self._elements_per_date * max_window, 1)
        return max(1, min(n_scenarios, self.MAX_BATCH_ELEMENTS // per_scenario))
