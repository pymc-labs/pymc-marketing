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
at a time (:mod:`~pymc_marketing.mmm.counterfactual`).  This module turns one
scenario of one block into the period's change in the linear predictor,
differenced against the baseline before anything is summed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from pymc_marketing.mmm.spend_reach import CHANNEL_CONTRIBUTION

if TYPE_CHECKING:
    from pymc_marketing.mmm.counterfactual import PeriodWindow

__all__ = ["LinearPredictorDelta"]


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
