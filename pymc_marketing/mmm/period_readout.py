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
"""Reading one period's increment out of its evaluated scenarios.

:class:`~pymc_marketing.mmm.incrementality.Incrementality` evaluates every
scenario on a window of dates around its period, one block of posterior samples
at a time (:mod:`~pymc_marketing.mmm.counterfactual`).  This module reads those
evaluations out, one scenario at a time:

* :class:`LinearPredictorDelta` turns one scenario of one block into the
  period's change in the linear predictor, differenced against the baseline
  before anything is summed.
* A :class:`PeriodReadout` reduces that change to what one public output keeps
  of the period: its total (:class:`PeriodTotal`), its increment on each date
  it lands on (:class:`PeriodBand`), or its current and future value
  (:class:`PeriodCurrentFuture`).  Each reduces as soon as its output allows,
  so nothing larger than that is held across blocks of samples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from pymc_marketing.mmm.spend_reach import CHANNEL_CONTRIBUTION

if TYPE_CHECKING:
    from pandas.tseries.offsets import BaseOffset

    from pymc_marketing.mmm.counterfactual import PeriodWindow
    from pymc_marketing.mmm.incrementality import IncrementalReducer

__all__ = [
    "LinearPredictorDelta",
    "PeriodBand",
    "PeriodCurrentFuture",
    "PeriodReadout",
    "PeriodTotal",
    "ReadoutContext",
]


@dataclass(frozen=True)
class LinearPredictorDelta:
    r"""Change in the linear predictor over one period's evaluation dates.

    The counterfactual reaches :math:`\mu` through ``channel_contribution``
    and through every included effect, and :math:`\mu` is their *sum*, so
    the perturbation is the sum of their perturbations:

    .. math::

        \Delta \mu_t = \Delta v_{t,m} + \sum_j \Delta e_{t,j}

    This is what keeps the
    :class:`~pymc_marketing.mmm.incrementality.IncrementalReducer` hierarchy
    out of the mediation problem entirely.  A reducer converts a change in the
    linear predictor into a change in the response; it does not care how many
    nodes that change was collected from.

    Each node is differenced against the baseline before anything is summed,
    and both sides are read through the window's slices, so only the
    difference is a new array.  Where a scenario's other channel columns answer
    for other channels, only the column read is differenced.  Summing
    ``channel_contribution`` over channels before differencing would hold less,
    but it would move the rounding error to the scale of the whole
    contribution: differencing first keeps an unperturbed column's change at
    exactly zero.

    Parameters
    ----------
    effect_names : tuple of str
        The included effects' contribution variables.  Given rather than read
        off the outputs' keys, which also carry nodes evaluated for other
        reasons: the linear predictor is evaluated to check the increment is
        complete and must not be added into it.
    non_date_dims : mapping
        Per response variable, its dimensions with ``date`` removed.
    coords : mapping
        The labels of the non-date dimensions in the order the evaluator
        computed them, the model's.  The change carries them so that xarray
        pairs it with each reducer input by label: the target scale,
        ``mu_original_scale`` and the mean-scale correction are read from
        ``idata`` in whatever order it stores them.  A dimension without an
        entry stays unlabeled.
    """

    effect_names: tuple[str, ...]
    non_date_dims: Mapping[str, tuple[str, ...]]
    coords: Mapping[str, np.ndarray]

    def __call__(
        self,
        counterfactual: Mapping[str, np.ndarray],
        baseline: Mapping[str, np.ndarray],
        *,
        row: int,
        window: PeriodWindow,
        channel: int | None,
    ) -> xr.DataArray:
        """Return one scenario's change in the linear predictor.

        Parameters
        ----------
        counterfactual : mapping
            Per response variable, predictions per scenario, shape
            ``(n_scenarios, n_samples, max_window, *non_date_dims)``.
        baseline : mapping
            Per response variable, unperturbed predictions on the full date
            axis for the same samples, shape
            ``(n_samples, n_dates, *non_date_dims)``.
        row : int
            Row of *counterfactual* to read, from
            :attr:`~pymc_marketing.mmm.counterfactual.CounterfactualScenarios.rows`.
        window : PeriodWindow
            The period's window.  Its
            :attr:`~pymc_marketing.mmm.counterfactual.PeriodWindow.window_eval_slice`
            reads the evaluation dates out of the padded counterfactual window,
            its :attr:`~pymc_marketing.mmm.counterfactual.PeriodWindow.eval_slice`
            the same dates out of the full-axis baseline, and its
            :attr:`~pymc_marketing.mmm.counterfactual.PeriodWindow.eval_dates`
            label them, so the reducer can align a baseline response by label.
            Coming from one place is what makes them the same dates.
        channel : int or None
            Column of ``channel_contribution`` to read, or ``None`` to sum the
            channel dimension away.  A column is read only when *row* is a
            **shared** all-channels perturbation whose other columns answer for
            other channels; that is the separable case, where the untouched
            columns of a channel's own counterfactual are exactly unmoved and
            summing would return the joint delta instead.  When *row* is a
            perturbation of one channel alone -- the mediated case, and the
            measured-mixing one -- the whole delta belongs to that channel,
            including the movement it caused in the other columns, so the sum
            is what the unilateral estimand asks for.

        Returns
        -------
        xr.DataArray
            Dimensions ``("sample", "date", *custom_dims)``, labeled with
            :attr:`coords`.
        """
        delta_mu = self._delta(
            CHANNEL_CONTRIBUTION, counterfactual, baseline, row, window, channel
        )
        if channel is None:
            delta_mu = delta_mu.sum(dim="channel")
        for name in self.effect_names:
            # xarray aligns by name, so an effect carrying only a subset of
            # the model's dimensions broadcasts without any reshaping here.
            delta_mu = delta_mu + self._delta(
                name, counterfactual, baseline, row, window, None
            )
        return delta_mu

    def _delta(
        self,
        name: str,
        counterfactual: Mapping[str, np.ndarray],
        baseline: Mapping[str, np.ndarray],
        row: int,
        window: PeriodWindow,
        column: int | None,
    ) -> xr.DataArray:
        """Difference one node of one scenario against the baseline.

        Basic indexing throughout, so both operands are views and the
        difference is the only array made.  With *column*, only that
        ``channel`` column is read, and the dimension is dropped.
        """
        dims = self.non_date_dims[name]
        scenario_index: tuple = (row, slice(None), window.window_eval_slice)
        baseline_index: tuple = (slice(None), window.eval_slice)
        if column is not None:
            to_channel = (slice(None),) * dims.index("channel")
            scenario_index = (*scenario_index, *to_channel, column)
            baseline_index = (*baseline_index, *to_channel, column)
            dims = tuple(dim for dim in dims if dim != "channel")
        return xr.DataArray(
            counterfactual[name][scenario_index] - baseline[name][baseline_index],
            dims=("sample", "date", *dims),
            coords={
                "date": window.eval_dates,
                **{dim: self.coords[dim] for dim in dims if dim in self.coords},
            },
        )


def current_window_end(
    end: pd.Timestamp, *, lags: int, freq_offset: BaseOffset
) -> pd.Timestamp:
    """Return the last date a period's current window covers.

    Parameters
    ----------
    end : pd.Timestamp
        The period's end.
    lags : int
        Data periods after *end* that still count as current.
    freq_offset : BaseOffset
        The data's date frequency, which *lags* is counted in.

    Returns
    -------
    pd.Timestamp
        *end* plus *lags* data periods.
    """
    # Adding zero periods of an anchored offset rolls an off-anchor date
    # forward (a month end plus 0 weeks is the next Sunday), so a zero lag
    # leaves the period end where it is.
    return end + lags * freq_offset if lags else end


def split_band(
    band: xr.DataArray,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lags: int,
    freq_offset: BaseOffset,
) -> tuple[xr.DataArray, xr.DataArray, bool]:
    """Reduce one period's band to its current and future value.

    Current is every realization date from the period's start through its
    end plus ``lags`` data periods, future every date after that.

    Parameters
    ----------
    band : xr.DataArray
        The period's per-date increment on ``realization_date``, with a
        boolean ``observed`` coordinate on the same dimension.  Unobserved
        dates hold ``NaN``.
    start, end : pd.Timestamp
        The period's bounds.
    lags : int
        Data periods after ``end`` that still count as current.
    freq_offset : BaseOffset
        The data's date frequency, which ``lags`` is counted in.

    Returns
    -------
    current, future : xr.DataArray
        *band* summed over each window, without ``realization_date``.
        Neither sum skips ``NaN``: an unobserved date is carryover the model
        has and the data cannot show, not a zero.
    complete : bool
        Whether every date in the band is observed.
    """
    current_end = current_window_end(end, lags=lags, freq_offset=freq_offset)
    realization = band.indexes["realization_date"]
    in_current = (realization >= start) & (realization <= current_end)

    current = band.isel(realization_date=in_current).sum(
        "realization_date", skipna=False
    )
    future = band.isel(realization_date=realization > current_end).sum(
        "realization_date", skipna=False
    )
    complete = bool(band.coords["observed"].all())

    return current, future, complete


def tail_dates(
    *,
    window: PeriodWindow,
    dates: pd.DatetimeIndex,
    tail_lags: int,
    freq_offset: BaseOffset,
) -> pd.DatetimeIndex:
    """List the dates past the fitted axis that a period's carryover lands on.

    Parameters
    ----------
    window : PeriodWindow
        The period's window.
    dates : pd.DatetimeIndex
        The full fitted date axis.
    tail_lags : int
        Lags after the period's last fitted spend date that its carryover
        is known to reach, in data periods.
    freq_offset : BaseOffset
        The data's date frequency.

    Returns
    -------
    pd.DatetimeIndex
        From the first date after the last fitted date to the period's last
        fitted spend date plus *tail_lags*; empty for a period with no fitted
        spend date.
    """
    spend = dates[(dates >= window.start) & (dates <= window.end)]
    tail = pd.DatetimeIndex([])
    if len(spend):
        reach_end = spend[-1] + tail_lags * freq_offset
        tail = pd.date_range(dates[-1], reach_end, freq=freq_offset)
        tail = tail[tail > dates[-1]]
    return tail


def append_unobserved_tail(
    band: xr.DataArray,
    *,
    window: PeriodWindow,
    dates: pd.DatetimeIndex,
    tail_lags: int,
    freq_offset: BaseOffset,
) -> xr.DataArray:
    """Extend a period's band past the fitted axis, as far as it reaches.

    Windows are clamped to the fitted dates, so a period near the end of the
    data is evaluated on fewer dates than its carryover lands on.  The
    missing dates are appended as ``NaN`` with ``observed=False``: carryover
    the model has but the data ends before, which summing as zero would
    silently truncate.

    They start at the first date after the last fitted date and run to the
    period's last fitted spend date plus *tail_lags*.  Not to the window's
    own bound, which sits one date past a plain adstock's kernel, and not to
    the period's calendar end, which for a last period the data ends
    partway through lies past any spend there is.  With a windowed
    evaluation *tail_lags* is the reach the probe measured.  Under full-axis
    evaluation no horizon was measured, and *tail_lags* is the kernel's own
    trailing lags, so the appended dates are a lower bound on what the data
    cannot show: a node that reduces over ``date``, or a mediated path, can
    reach further.

    Only the trailing side is extended.  A kernel with leading mass also
    loses the mass that falls before the first fitted date, which no date
    on the axis can label, so ``observed`` does not mark it.  Such a kernel
    moves dates before the probed one, which sends the model to full-axis
    evaluation; one whose leading weights the probe cannot see stays
    windowed, and
    :meth:`~pymc_marketing.mmm.incrementality.Incrementality.split_incremental_contribution_current_future`
    refuses every mode other than ``ConvMode.After`` regardless.

    Parameters
    ----------
    band : xr.DataArray
        The period's increment over its evaluated dates, dims
        ``(chain, draw, realization_date, *out_dims)``.
    window : PeriodWindow
        The period's window.
    dates : pd.DatetimeIndex
        The full fitted date axis.
    tail_lags : int
        Lags after the period's last fitted spend date that its carryover
        is known to reach, in data periods.
    freq_offset : BaseOffset
        The data's date frequency.

    Returns
    -------
    xr.DataArray
        *band* with the unobserved dates appended and a boolean
        ``observed`` coordinate on ``realization_date``.
    """
    tail = tail_dates(
        window=window, dates=dates, tail_lags=tail_lags, freq_offset=freq_offset
    )
    realization = band.indexes["realization_date"].append(tail)
    return band.reindex(realization_date=realization).assign_coords(
        observed=("realization_date", ~realization.isin(tail))
    )


@dataclass(frozen=True)
class ReadoutContext:
    """What every period of one call is labelled with.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The full fitted date axis.
    reported_order : mapping
        Custom-dim coordinates in the order of ``constant_data``, which is the
        order callers see.
    tail_lags : int
        Lags after a period's last fitted spend date that its carryover is
        known to reach: the measured
        :attr:`~pymc_marketing.mmm.spend_reach.SpendReach.max_lag`, or under
        full-axis evaluation the kernel's own trailing lags.  Required, so
        that no readout gets a band without its unobserved tail by leaving it
        out; a total does not read it.
    freq_offset : BaseOffset
        The data's date frequency, to label the carryover dates that fall past
        the fitted axis.
    """

    dates: pd.DatetimeIndex
    reported_order: Mapping[str, np.ndarray]
    tail_lags: int
    freq_offset: BaseOffset


class PeriodReadout[B, R](ABC):
    """What one public output keeps of each period's increment.

    :class:`~pymc_marketing.mmm.incrementality.Incrementality` reduces the
    counterfactual one block of samples at a time, and the readout decides
    what is kept at each step:

    1. :meth:`reduce_row` reduces one scenario's change in the linear
       predictor to a response-scale increment, before the channels are
       stacked.
    2. :meth:`reduce_block` reduces one period of one block, its scenarios
       stacked over ``channel`` and its samples laid out as
       ``(chain, draw)``, to what is held until every block is in.  The
       blocks are joined along ``chain`` and ``draw``, so it keeps both.
    3. :meth:`finish_period` turns the joined blocks into the period as the
       output reads it.  It runs once per period rather than once per block.

    One subclass per output, so that each states what it keeps, rather than a
    flag that switches between code paths.  ``B`` is what one block of a
    period is held as, ``R`` what a period is read out as.
    """

    @abstractmethod
    def reduce_row(
        self, delta_mu: xr.DataArray, reducer: IncrementalReducer
    ) -> xr.DataArray:
        """Reduce one scenario's change in the linear predictor.

        Parameters
        ----------
        delta_mu : xr.DataArray
            Dims ``("sample", "date", *custom_dims)``, from
            :class:`LinearPredictorDelta`.
        reducer : IncrementalReducer
            The link's reduction, restricted to the block's samples.

        Returns
        -------
        xr.DataArray
            The scenario's response-scale increment: summed over ``date``, or
            per date with ``date`` renamed ``realization_date``.
        """

    @abstractmethod
    def reduce_block(
        self,
        increment: xr.DataArray,
        *,
        window: PeriodWindow,
        context: ReadoutContext,
    ) -> B:
        """Reduce one period of one block of samples to what is held.

        Parameters
        ----------
        increment : xr.DataArray
            The period's scenarios after :meth:`reduce_row`, stacked over
            ``channel`` for the per-channel estimand, with the sign convention
            applied.  Dims ``("chain", "draw", *dims)``, where *dims* are what
            :meth:`reduce_row` kept, the block's labels on ``chain`` and
            ``draw``, and the custom dims in the model's label order.
        window : PeriodWindow
            The period's window.
        context : ReadoutContext
            What every period of the call is labelled with.

        Returns
        -------
        xr.DataArray or xr.Dataset
            What is held of the block until every block is in.
        """

    @abstractmethod
    def finish_period(
        self, increment: B, *, window: PeriodWindow, context: ReadoutContext
    ) -> R:
        """Read one period out of its blocks, joined.

        Parameters
        ----------
        increment : xr.DataArray or xr.Dataset
            What :meth:`reduce_block` returned for each block, joined along
            ``chain`` and ``draw``, so labelled with the posterior's.
        window : PeriodWindow
            The period's window.
        context : ReadoutContext
            What every period of the call is labelled with.

        Returns
        -------
        object
            The period as the output reads it, with the custom dims in the
            order callers see.
        """


class PeriodTotal(PeriodReadout[xr.DataArray, xr.DataArray]):
    """Each period's increment summed over its evaluation dates.

    What
    :meth:`~pymc_marketing.mmm.incrementality.Incrementality.compute_incremental_contribution`
    and
    :meth:`~pymc_marketing.mmm.incrementality.Incrementality.compute_joint_incremental_contribution`
    report.  A period gets dims ``(date, chain, draw, *out_dims)``, with
    ``date`` of length one, the period end.
    """

    def reduce_row(
        self, delta_mu: xr.DataArray, reducer: IncrementalReducer
    ) -> xr.DataArray:
        """See :meth:`PeriodReadout.reduce_row`."""
        return reducer.counterfactual_minus_baseline(delta_mu)

    def reduce_block(
        self,
        increment: xr.DataArray,
        *,
        window: PeriodWindow,
        context: ReadoutContext,
    ) -> xr.DataArray:
        """See :meth:`PeriodReadout.reduce_block`; a total is held as is."""
        return increment

    def finish_period(
        self,
        increment: xr.DataArray,
        *,
        window: PeriodWindow,
        context: ReadoutContext,
    ) -> xr.DataArray:
        """See :meth:`PeriodReadout.finish_period`."""
        return (
            increment.assign_coords(date=window.end)
            .expand_dims("date")
            .sel(context.reported_order)
        )


class PeriodBand(PeriodReadout[xr.DataArray, xr.DataArray]):
    """Each period's increment on every date it lands on.

    What
    :meth:`~pymc_marketing.mmm.incrementality.Incrementality.split_incremental_contribution_over_time`
    lays out as a matrix, for either estimand.  A period gets dims
    ``(chain, draw, realization_date, *out_dims)``: its evaluated dates,
    extended past the fitted axis as far as the carryover reaches, with a
    boolean ``observed`` coordinate on ``realization_date``; see
    :func:`append_unobserved_tail`.
    """

    def reduce_row(
        self, delta_mu: xr.DataArray, reducer: IncrementalReducer
    ) -> xr.DataArray:
        """See :meth:`PeriodReadout.reduce_row`."""
        return _per_date_increment(delta_mu, reducer)

    def reduce_block(
        self,
        increment: xr.DataArray,
        *,
        window: PeriodWindow,
        context: ReadoutContext,
    ) -> xr.DataArray:
        """See :meth:`PeriodReadout.reduce_block`; a band is held as is."""
        return increment

    def finish_period(
        self,
        increment: xr.DataArray,
        *,
        window: PeriodWindow,
        context: ReadoutContext,
    ) -> xr.DataArray:
        """See :meth:`PeriodReadout.finish_period`."""
        return _band(increment, window=window, context=context)


@dataclass(frozen=True)
class PeriodCurrentFuture(
    PeriodReadout[xr.Dataset, tuple[xr.DataArray, xr.DataArray, bool]]
):
    """Each period's increment split into its current and future value.

    What
    :meth:`~pymc_marketing.mmm.incrementality.Incrementality.split_incremental_contribution_current_future`
    reports.  Each block's band is split as soon as its scenarios are
    reduced, so only the two sums are held across blocks, never a band.  Each
    sum runs along ``realization_date`` alone, so splitting block by block
    adds the same numbers in the same order as splitting the joined band.

    Parameters
    ----------
    lags : int
        Data periods after a period's end that still count as current.
    """

    lags: int

    def reduce_row(
        self, delta_mu: xr.DataArray, reducer: IncrementalReducer
    ) -> xr.DataArray:
        """See :meth:`PeriodReadout.reduce_row`."""
        return _per_date_increment(delta_mu, reducer)

    def reduce_block(
        self,
        increment: xr.DataArray,
        *,
        window: PeriodWindow,
        context: ReadoutContext,
    ) -> xr.Dataset:
        """See :meth:`PeriodReadout.reduce_block`.

        Returns
        -------
        xr.Dataset
            ``current`` and ``future``, as :func:`split_band` sums them.
        """
        current, future, _ = split_band(
            _band(increment, window=window, context=context),
            start=window.start,
            end=window.end,
            lags=self.lags,
            freq_offset=context.freq_offset,
        )
        return xr.Dataset({"current": current, "future": future})

    def finish_period(
        self,
        increment: xr.Dataset,
        *,
        window: PeriodWindow,
        context: ReadoutContext,
    ) -> tuple[xr.DataArray, xr.DataArray, bool]:
        """See :meth:`PeriodReadout.finish_period`.

        Returns
        -------
        current, future : xr.DataArray
            As :func:`split_band` returns them.
        complete : bool
            Whether every date of the band is observed: whether none of the
            period's carryover falls past the fitted axis.
        """
        tail = tail_dates(
            window=window,
            dates=context.dates,
            tail_lags=context.tail_lags,
            freq_offset=context.freq_offset,
        )
        return increment["current"], increment["future"], tail.empty


def _per_date_increment(
    delta_mu: xr.DataArray, reducer: IncrementalReducer
) -> xr.DataArray:
    """Reduce one scenario per date, labelled by ``realization_date``."""
    return reducer.per_date_increment(delta_mu).rename(date="realization_date")


def _band(
    increment: xr.DataArray, *, window: PeriodWindow, context: ReadoutContext
) -> xr.DataArray:
    """Append a band's unobserved tail and put it in the order callers see."""
    return append_unobserved_tail(
        increment,
        window=window,
        dates=context.dates,
        tail_lags=context.tail_lags,
        freq_offset=context.freq_offset,
    ).sel(context.reported_order)
