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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import pytensor.xtensor as ptx
import xarray as xr
from pandas.tseries.offsets import BaseOffset
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
    "PeriodWindow",
    "ScenarioKey",
]

type ScenarioKey = tuple[int, int | None]
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
            Last date to sum, which is *end* plus the carry-out length when
            carryover is included and *end* itself when it is not.

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
            Whether the *evaluation* range reaches ``l_max`` past the period to
            pick up its carry-out.  Independent of the window, which always does:
            the two lengths do different jobs and collapsing them would change
            the numbers inside the period too.
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
            else:
                # Reach back l_max for carry-in context and forward l_max so
                # carryover is captured; the eval mask decides what is summed.
                in_window = (dates >= start - l_max * freq_offset) & (
                    dates <= end + l_max * freq_offset
                )
            windows.append(
                PeriodWindow.build(
                    start=start,
                    end=end,
                    dates=dates,
                    in_window=in_window,
                    eval_end=end + l_max * freq_offset if include_carryover else end,
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
            given :data:`ScenarioKey` and to broadcast per-period arrays over
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
        rows: dict[ScenarioKey, int] = {}

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
        Maps a :data:`ScenarioKey` to a row of :attr:`spend`.

    See Also
    --------
    PeriodWindow : Which dates each period is evaluated over, and which are summed.
    """

    spend: np.ndarray
    period_index: np.ndarray
    rows: dict[ScenarioKey, int]


class CounterfactualEvaluator:
    """Compiled batched evaluator for the nodes a spend counterfactual reaches.

    Conditions the model graph on posterior draws, swaps every date-indexed
    ``pm.Data`` the evaluation needs for a batched input, and compiles *one*
    function returning all requested nodes.  Extracting the nodes together
    matters: ``channel_contribution`` and a mediated effect read the same spend
    data through the same adstock, and a single extraction keeps that subgraph
    shared instead of computing it once per node.

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

    Attributes
    ----------
    non_date_dims : dict
        Per response variable, the dimensions of its result after ``sample`` and
        ``date``, which is the node's own dimension order with ``date`` removed.
    windowed_data_vars : tuple of str
        Date-indexed ``pm.Data`` variables discovered in the graph and cut
        alongside spend, excluding ``channel_data`` and ``time_index``.

    Raises
    ------
    ValueError
        If ``channel_data`` is not of a floating dtype, or a discovered
        date-indexed input does not span the fitted date axis.
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
    ) -> None:
        # Results are keyed by name whether the caller asked by name or handed
        # over the node: the linear predictor is only a registered variable
        # under a log link, but it is named "mu" either way.
        self.response_vars = tuple(
            var if isinstance(var, str) else var.name for var in response_vars
        )
        graphs: list = extract_response_distribution(
            pymc_model=pymc_model,
            idata=xr.DataTree.from_dict({"/posterior": posterior}),
            response_variable=list(response_vars),
            frozen_deterministics=frozen_deterministics,
        )
        graph_ancestors = set(ancestors(graphs))

        self.non_date_dims = {
            name: self._non_date_dims(graph)
            for name, graph in zip(self.response_vars, graphs, strict=True)
        }
        # Output elements one scenario costs per date of its window, summed over
        # the evaluated nodes.  Sizes the evaluation batches.
        n_samples = posterior.sizes["chain"] * posterior.sizes["draw"]
        self._elements_per_date = n_samples * sum(
            int(np.prod([len(pymc_model.coords[dim]) for dim in dims], dtype=int))
            for dims in self.non_date_dims.values()
        )

        # Spend: the input the counterfactual actually perturbs.  A float dtype
        # is required so that a fractional counterfactual_spend_factor (1.01,
        # say) is not truncated.
        channel_data = pymc_model[self.CHANNEL_DATA]
        if np.dtype(channel_data.dtype).kind != "f":
            raise ValueError(
                "Incrementality requires channel data of float type, got "
                f"{channel_data.dtype}"
            )
        self.channel_dtype = channel_data.dtype
        replace: dict = {channel_data: self._batched(channel_data, "channel_data")}
        func_inputs: list = [replace[channel_data]]

        # time_index: only replaced when the graph actually reads it (with
        # time_varying_intercept but not time_varying_media it is unused, and
        # passing it would raise UnusedInputError).
        self.time_dtype: str | None = None
        if (
            self.TIME_INDEX in pymc_model.named_vars
            and pymc_model[self.TIME_INDEX] in graph_ancestors
        ):
            time_index = pymc_model[self.TIME_INDEX]
            self.time_dtype = time_index.dtype
            replace[time_index] = self._batched(time_index, "time_index")
            func_inputs.append(replace[time_index])

        # Any other date-indexed pm.Data the graph reads.  For a plain MMM there
        # are none; a mediated effect brings its own (an exogenous budget, a
        # category-demand series).  Discovery is by graph traversal rather than
        # by declaration so an effect cannot forget to mention one.
        self._aux: list[DateIndexedInput] = []
        for data in self._date_indexed_data(pymc_model, graph_ancestors):
            values = np.asarray(data.eval())
            dims = tuple(pymc_model.named_vars_to_dims[data.name])
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

    @classmethod
    def _date_indexed_data(cls, pymc_model, graph_ancestors: set) -> list:
        """Find the date-indexed ``pm.Data`` the graph reads, beyond the two handled above.

        Parameters
        ----------
        pymc_model : pm.Model
            The model being evaluated.
        graph_ancestors : set
            Ancestors of the extracted response graphs.

        Returns
        -------
        list
            The data variables, in the model's declaration order so that the
            compiled signature is deterministic.
        """
        handled = {cls.CHANNEL_DATA, cls.TIME_INDEX}
        return [
            data
            for data in pymc_model.data_vars
            if data in graph_ancestors
            and data.name not in handled
            and "date" in pymc_model.named_vars_to_dims.get(data.name, ())
        ]

    def evaluate_baseline(self, channel_data: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate every node on the actual data, over the full date axis.

        Parameters
        ----------
        channel_data : np.ndarray
            Actual spend, shape ``(n_dates, *extra_shape)``.

        Returns
        -------
        dict
            Per response variable, an array of shape
            ``(n_samples, n_dates, *non_date_dims)``.
        """
        args: list[np.ndarray] = [channel_data[np.newaxis].astype(self.channel_dtype)]
        if self.time_dtype is not None:
            args.append(
                np.arange(len(channel_data))[np.newaxis].astype(self.time_dtype)
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
