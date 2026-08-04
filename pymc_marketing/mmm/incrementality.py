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
r"""Incrementality and counterfactual analysis for Marketing Mix Models.

This module provides functionality to compute **incremental channel
contributions** using counterfactual analysis, properly accounting for
adstock carryover effects.

Concept
-------
Incrementality measures the *causal* impact of a marketing channel by
comparing two scenarios:

1. **Actual**: the model prediction with real spend data.
2. **Counterfactual**: the model prediction with spend removed or perturbed.

The difference between these two predictions is the **incremental
contribution** of that channel.  Because MMMs include adstock
transformations, spend at time *t* affects outcomes at
*t, t + 1, ..., t + l_max*.  A naïve element-wise comparison ignores this
temporal attribution; this module handles it correctly by extending the
evaluation window to capture both carry-in and carry-out effects.

**Total incrementality** (zero-out counterfactual):

.. math::

    \Delta Y_m = \sum_{t=t_0}^{t_1 + L - 1}
        \bigl[\hat{Y}_t(x;\,\Omega)
            - \hat{Y}_t(x^{\text{cf}};\,\Omega)\bigr]

where the counterfactual spend zeroes out only the evaluation period:

.. math::

    x^{\text{cf}}_{s,m} =
    \begin{cases}
        0        & s \in [t_0,\, t_1] \\
        x_{s,m}  & \text{otherwise}
    \end{cases}

**Marginal incrementality** (small perturbation):

.. math::

    \delta Y_m = \sum_{t=t_0}^{t_1 + L - 1}
        \bigl[\hat{Y}_t(\tilde{x};\,\Omega)
            - \hat{Y}_t(x;\,\Omega)\bigr]

where the perturbed spend scales only the evaluation period:

.. math::

    \tilde{x}_{s,m} =
    \begin{cases}
        \alpha\, x_{s,m}  & s \in [t_0,\, t_1] \\
        x_{s,m}           & \text{otherwise}
    \end{cases}

Here *m* is the channel, *x* the spend vector, *L* the adstock window
length (``l_max``), *Ω* the posterior parameter samples, and
*α* the ``counterfactual_spend_factor``.  Spend **outside**
:math:`[t_0, t_1]` is always kept at its actual value so that adstock
carry-in is correctly accounted for.

The intervention is on **spend**, not on a channel's effect.  With
:math:`\alpha = 0` the two agree only if the saturation sends zero spend to
zero contribution; otherwise (or with ``time_varying_media``) a residual
effect survives the counterfactual.  To remove a component's effect
directly, see
:meth:`~pymc_marketing.mmm.mmm.MMM.compute_counterfactual_contributions_dataset`.

Incrementality is a **general-purpose building block**.  Dividing
incremental contribution by spend gives **ROAS** (Return on Ad Spend) when
the model's target variable is revenue; taking the reciprocal
(spend / contribution) gives **CAC** (Customer Acquisition Cost) when the
target is customer count.  The same logic applies to any target variable.

Link functions
--------------
The counterfactual is applied to spend and evaluated on
``channel_contribution``, which lives in the **linear predictor**
:math:`\mu_t = \text{base}_t + \sum_c v_{t,c}` -- not on the response
scale.  Turning a change in :math:`v_{t,m}` into a change in
:math:`\hat{Y}_t = \text{inv}(\mu_t)\,s` is link-dependent, and is the job
of an :class:`IncrementalReducer`:

* ``link="identity"`` (:class:`IdentityLinkReducer`) -- the response is
  additive in the media contributions, the base term cancels, and the
  increment is :math:`s \sum_t \Delta_{t,m}`.  Per-channel increments are
  independent of the baseline and of each other, and they sum to the total
  media increment.
* ``link="log"`` (:class:`LogLinkReducer`) -- the response is
  *multiplicative*, so the base term does **not** cancel and the increment
  is :math:`\sum_t \hat{Y}_t [\exp(\Delta_{t,m}) - 1]`.  Per-channel
  increments depend on the baseline, the controls and the other channels,
  and they do **not** sum to the total media increment.  This is a property
  of the model, not of the estimator: the paper's derivation of
  :math:`\text{ROAS}_m` assumes an additive response, and that assumption
  does not survive a non-linear link.

Because the increment is formed per posterior draw and only then
aggregated, credible intervals are correct under both links.

Mediated effects
----------------
Spend does not always reach the response through ``channel_contribution``
alone.  A ``mu_effect`` can read ``channel_data`` itself -- a funnel
mediator, where upper-funnel spend creates demand, demand drives
lower-funnel spend, and only that converts -- and then part of the
incremental response travels through the effect.  Such an effect is
included in the increment, additively in the linear predictor:

.. math::

    \Delta \mu_t = \Delta v_{t,m} + \sum_j \Delta e_{t,j}

which is then handed to the same :class:`IncrementalReducer` as before.
The reducers are untouched by mediation: they convert a change in the
linear predictor into a change in the response, and do not care how many
nodes that change was collected from.

Three things follow, and they are why mediation is not free:

* **Effects must opt in.**  An effect whose contribution depends on
  ``channel_data`` and has not implemented
  :meth:`~pymc_marketing.mmm.additive_effect.MuEffect.incrementality_spec`
  raises ``NotImplementedError``.  Ignoring it would report the direct path
  as if it were the total.  Effects that do *not* depend on spend -- trends,
  events, seasonality -- are part of the baseline, cancel in the difference,
  and are skipped without being asked anything.
* **One counterfactual per channel.**  Without mediation a single
  all-channels perturbation is enough, because :math:`v_{t,c}` depends on
  channel *c*'s spend alone and column *m* of that one evaluation *is*
  channel *m*'s counterfactual.  A funnel sums over channels *inside* a
  nonlinear transform, so no per-channel column survives and each channel
  needs its own perturbation.
* **The window gets longer.**  A mediated path that chains a second adstock
  behind the model's own outlives it, so the effect declares how much
  further it reaches and the evaluation window is sized for the longest
  path spend can take.

Estimands
---------
:meth:`Incrementality.compute_incremental_contribution` is *leave-one-out*:
each channel's number answers "what would we lose without this channel,
holding the others at their actual spend".
:meth:`Incrementality.compute_joint_incremental_contribution` perturbs every
channel at once and answers "how much does media drive in total".

The two agree only when the response is additive in the channels, that is
under ``link="identity"`` with no channel-dependent effect.  Otherwise
interaction mass is counted by every channel that touches it and the
leave-one-out numbers sum to *more* than the joint.  Summing per-channel
increments is therefore not a way to get a total, and the gap is a property
of the model rather than an error in either number.

Examples
--------
Compute quarterly incremental contributions:

.. code-block:: python

    incremental = mmm.incrementality.compute_incremental_contribution(
        frequency="quarterly",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

Compute quarterly ROAS (when target variable is revenue):

.. code-block:: python

    roas = mmm.incrementality.contribution_over_spend(
        frequency="quarterly",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

Compute monthly CAC (when target variable is customer count):

.. code-block:: python

    cac = mmm.incrementality.spend_over_contribution(
        frequency="monthly",
    )

Compute marginal ROAS (return on next dollar):

.. code-block:: python

    mroas = mmm.incrementality.marginal_contribution_over_spend(
        frequency="quarterly",
    )

References
----------
Google MMM Paper: https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import pandas as pd
import pytensor.xtensor as ptx
import xarray as xr
from pandas.tseries.offsets import BaseOffset
from pydantic import ConfigDict, validate_call
from pytensor import function
from pytensor.graph.traversal import ancestors
from pytensor.xtensor.vectorization import vectorize_graph

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
from pymc_marketing.data.idata.schema import Frequency
from pymc_marketing.data.idata.utils import subsample_draws
from pymc_marketing.mmm.link import LinkFunction
from pymc_marketing.pytensor_utils import extract_response_distribution

if TYPE_CHECKING:
    from numpy.random import Generator, RandomState

    from pymc_marketing.mmm.mmm import MMM

CentralTendency = Literal["median", "mean"]


class IncrementalReducer(ABC):
    r"""Map a linear-predictor perturbation to a response-scale increment.

    :class:`Incrementality` perturbs spend and evaluates
    ``channel_contribution``, which lives in the **linear predictor**

    .. math::

        \mu_t = \text{base}_t + \sum_c v_{t,c}

    where :math:`v_{t,c}` is channel *c*'s contribution and
    :math:`\text{base}_t` collects the intercept, controls and seasonality.
    The response is :math:`\hat{Y}_t = \text{inv}(\mu_t)\,s` for inverse link
    :math:`\text{inv}` and target scale :math:`s`.

    Translating :math:`\Delta_{t,m} = v^{\text{cf}}_{t,m} - v_{t,m}` into a
    change in :math:`\hat{Y}` is the *only* link-dependent step of the
    calculation, so it is isolated here.  Subclasses correspond one-to-one to
    the :class:`~pymc_marketing.mmm.link.LinkSpec` implementations; see
    :meth:`Incrementality._build_reducer` for the dispatch.

    Notes
    -----
    Every subclass assumes the counterfactual reaches the response *only*
    through ``channel_contribution``.  That assumption is what lets a single
    all-channels evaluation be read per channel: :math:`v_{t,c}` depends on
    channel *c*'s spend alone, so column *m* of an all-channels counterfactual
    equals column *m* of a channel-*m*-only counterfactual.

    See Also
    --------
    IdentityLinkReducer : Additive response (``link="identity"``).
    LogLinkReducer : Multiplicative response (``link="log"``).
    """

    @abstractmethod
    def counterfactual_minus_baseline(self, delta: xr.DataArray) -> xr.DataArray:
        r"""Return :math:`\sum_t [\hat{Y}^{\text{cf}}_t - \hat{Y}_t]`.

        Parameters
        ----------
        delta : xr.DataArray
            :math:`\Delta_{t,m}`, the counterfactual-minus-baseline change in
            the linear-predictor contribution.  Its first dimension is
            ``"sample"``, followed by ``"date"`` and then the model's own
            non-date dimensions -- panel models order these
            ``(*custom_dims, "channel")``, so do not assume ``"channel"``
            comes first.  Reductions here are by dimension *name* for exactly
            that reason.  The ``date`` coordinates span the evaluation window
            of a single period.

        Returns
        -------
        xr.DataArray
            Response-scale difference summed over ``date``.  The ``date``
            dimension is dropped; every other dimension is preserved.
        """


class IdentityLinkReducer(IncrementalReducer):
    r"""Increment reducer for an additive response (``link="identity"``).

    With :math:`\text{inv} = \text{id}` the base term cancels exactly:

    .. math::

        \sum_t [\hat{Y}^{\text{cf}}_t - \hat{Y}_t] = s \sum_t \Delta_{t,m}

    Per-channel increments are therefore independent of the baseline, of the
    control variables and of the other channels, and they sum to the total
    media increment -- the setting in which the paper's :math:`\text{ROAS}_m`
    is derived.

    Parameters
    ----------
    scale : xr.DataArray or float
        Target scale :math:`s` mapping the linear predictor back to the
        response scale.  Scalar, or dimensioned for panel models
        (e.g. ``("country",)``).
    """

    def __init__(self, scale: xr.DataArray | float) -> None:
        self.scale = scale

    def counterfactual_minus_baseline(self, delta: xr.DataArray) -> xr.DataArray:
        """See :meth:`IncrementalReducer.counterfactual_minus_baseline`."""
        return delta.sum(dim="date") * self.scale


class LogLinkReducer(IncrementalReducer):
    r"""Increment reducer for a multiplicative response (``link="log"``).

    With :math:`\text{inv} = \exp` the base term does **not** cancel.  Because
    the linear predictor is additive in the channel contributions, perturbing
    channel *m* alone gives
    :math:`\exp(\mu^{\text{cf}}_t) = \exp(\mu_t)\exp(\Delta_{t,m})`, hence

    .. math::

        \sum_t [\hat{Y}^{\text{cf}}_t - \hat{Y}_t]
            = \sum_t \hat{Y}_t \bigl[\exp(\Delta_{t,m}) - 1\bigr]

    so the baseline response :math:`\hat{Y}_t` enters as a weight.  This is
    the same estimand as
    :meth:`~pymc_marketing.mmm.mmm.MMM.compute_counterfactual_contributions_dataset`,
    evaluated per posterior draw, but with the spend counterfactual and
    carryover window of the incrementality module rather than a whole-component
    knock-out.

    Two consequences follow, and both are properties of the *model* rather
    than artefacts of this implementation: per-channel increments depend on
    the baseline, the controls and the other channels; and they do not sum to
    the total media increment.

    ``expm1`` is used instead of ``exp(x) - 1`` because marginal
    incrementality perturbs spend by only 1%, which makes :math:`\Delta` small
    and the subtraction cancellation-prone.

    Parameters
    ----------
    baseline_response : xr.DataArray
        Baseline prediction on the response scale -- the model's
        ``{output_var}_original_scale`` deterministic -- with dimensions
        ``("sample", "date", *custom_dims)`` and ``date`` coordinates spanning
        the fitted data.  It already carries ``target_scale``, so the
        increment needs no further rescaling.
    """

    def __init__(self, baseline_response: xr.DataArray) -> None:
        self.baseline_response = baseline_response

    def counterfactual_minus_baseline(self, delta: xr.DataArray) -> xr.DataArray:
        """See :meth:`IncrementalReducer.counterfactual_minus_baseline`."""
        baseline = self.baseline_response.sel(date=delta.coords["date"])
        return (baseline * np.expm1(delta)).sum(dim="date")


JOINT_CHANNEL = -1
"""Sentinel channel index for the scenario that perturbs every channel at once."""

CHANNEL_CONTRIBUTION = "channel_contribution"
"""Response variable holding the per-channel linear-predictor contribution."""


@dataclass(frozen=True)
class ChannelDependentEffect:
    """A ``mu_effect`` whose contribution a spend counterfactual can reach.

    Produced by :meth:`Incrementality._resolve_channel_dependent_effects` for
    every effect that (a) has ``channel_data`` among the ancestors of its
    contribution variable and (b) opted in via
    :meth:`~pymc_marketing.mmm.additive_effect.MuEffect.incrementality_spec`.

    Parameters
    ----------
    contribution_var : str
        Name of the deterministic holding the term this effect adds to the
        linear predictor.
    additional_carryover_lags : int
        Extra evaluation-window length this effect needs, from its
        :class:`~pymc_marketing.mmm.additive_effect.IncrementalitySpec`.
    """

    contribution_var: str
    additional_carryover_lags: int


class DateIndexedInput(NamedTuple):
    """A date-indexed ``pm.Data`` replaced by a batched evaluator input.

    Parameters
    ----------
    name : str
        Name of the data variable in the model.
    values : np.ndarray
        Its fitted values, with ``date`` first.  Read from the model's own shared
        variable rather than from ``fit_data``, because an effect's data need not
        appear there -- the funnel example builds the model from an ``xr.Dataset``
        carrying the mediator's inputs, then fits from a frame that does not.
    dtype : str
        Dtype the compiled evaluator expects.
    """

    name: str
    values: np.ndarray
    dtype: str


@dataclass(frozen=True)
class CounterfactualScenarios:
    """Perturbed spend arrays for every scenario that has to be evaluated.

    A *scenario* is a (period, perturbed channel) pair.  When no ``mu_effect``
    stands between spend and the response, a single all-channels perturbation
    per period suffices and every channel reads its own column out of it, so all
    of a period's keys point at the same row -- see
    :meth:`Incrementality._build_counterfactual_scenarios`.

    Parameters
    ----------
    spend : np.ndarray
        Counterfactual ``channel_data``, shape
        ``(n_scenarios, max_window, *extra_shape)``.
    period_index : np.ndarray
        Period each scenario belongs to, shape ``(n_scenarios,)``.  Indexes any
        per-period array (a windowed data variable, a ``time_index`` row) up to
        the scenario axis.
    channel_index : np.ndarray
        Channel each scenario perturbs, shape ``(n_scenarios,)``, or
        :data:`JOINT_CHANNEL` for the all-channels scenario.
    rows : dict
        Maps ``(period_idx, channel_idx)`` to a row of :attr:`spend`, with
        :data:`JOINT_CHANNEL` as the channel of the all-channels scenario.
    eval_masks : list of np.ndarray
        Per-period boolean mask over the padded window selecting the dates that
        enter the sum.
    period_labels : list of pd.Timestamp
        End date of each period.
    """

    spend: np.ndarray
    period_index: np.ndarray
    channel_index: np.ndarray
    rows: dict[tuple[int, int], int]
    eval_masks: list[np.ndarray]
    period_labels: list[pd.Timestamp]


class CounterfactualEvaluator:
    """Compiled batched evaluator for the nodes a spend counterfactual reaches.

    Conditions the model graph on posterior draws, swaps every date-indexed
    ``pm.Data`` the evaluation needs for a batched input, and compiles *one*
    function returning all requested nodes.  Extracting the nodes together
    matters: ``channel_contribution`` and a mediated effect read the same spend
    data through the same adstock, and a single extraction keeps that subgraph
    shared instead of computing it once per node.

    The batched inputs are what make a *window* evaluation possible.  The module
    evaluates counterfactuals on windows of length ``max_window`` rather than on
    the full date axis, so every date-indexed input in the graph has to be cut
    to the same window in lockstep -- otherwise the graph is handed a
    ``max_window``-long spend array and an ``n_dates``-long mediator array.
    This class discovers those inputs from the graph and windows them itself.

    Parameters
    ----------
    pymc_model : pm.Model
        The fitted model whose graph is evaluated.
    posterior : xr.Dataset
        Posterior samples (already subsampled).  Draws are flattened into a
        single ``sample`` axis in chain-major order.
    response_vars : sequence of str
        Nodes to evaluate, in the order the results are keyed by.
    frozen_deterministics : list of str
        Deterministics to hold at their posterior values instead of recomputing.
    dates : pd.DatetimeIndex
        Dates of the fitted data, used to validate the discovered inputs.

    Attributes
    ----------
    non_date_dims : dict
        Per response variable, its dimensions with ``date`` removed, in the
        model's own order.
    windowed_data_vars : tuple of str
        Date-indexed ``pm.Data`` variables discovered in the graph and windowed
        alongside spend, excluding ``channel_data`` and ``time_index``.

    Raises
    ------
    ValueError
        If ``channel_data`` is not of a floating dtype, or a discovered
        date-indexed input does not span the fitted date axis.
    """

    CHANNEL_DATA = "channel_data"
    TIME_INDEX = "time_index"

    def __init__(
        self,
        *,
        pymc_model,
        posterior: xr.Dataset,
        response_vars: Sequence[str],
        frozen_deterministics: list[str],
        dates: pd.DatetimeIndex,
    ) -> None:
        self.response_vars = tuple(response_vars)
        graphs: list = extract_response_distribution(
            pymc_model=pymc_model,
            idata=xr.DataTree.from_dict({"/posterior": posterior}),
            response_variable=list(self.response_vars),
            frozen_deterministics=frozen_deterministics,
        )
        graph_ancestors = set(ancestors(graphs))

        self.non_date_dims = {
            name: tuple(
                d for d in pymc_model.named_vars_to_dims.get(name, ()) if d != "date"
            )
            for name in self.response_vars
        }

        # Spend: the input the counterfactual actually perturbs.  float64 is
        # required so that a fractional counterfactual_spend_factor (1.01, say)
        # is not truncated.
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
            if values.shape[0] != len(dates):
                raise ValueError(
                    f"Date-indexed data variable {data.name!r} has "
                    f"{values.shape[0]} rows but the fitted data has "
                    f"{len(dates)} dates.  Incrementality needs it to span the "
                    "fitted date axis so it can be windowed alongside spend."
                )
            batched = self._batched(data, data.name)
            replace[data] = batched
            func_inputs.append(batched)
            self._aux.append(
                DateIndexedInput(name=data.name, values=values, dtype=data.dtype)
            )

        self._evaluator = function(
            func_inputs, vectorize_graph(graphs, replace=replace)
        )

    @property
    def windowed_data_vars(self) -> tuple[str, ...]:
        """Names of the auxiliary date-indexed inputs discovered in the graph."""
        return tuple(aux.name for aux in self._aux)

    @staticmethod
    def _batched(variable, name: str):
        """Return a batched xtensor standing in for a date-indexed input."""
        return ptx.xtensor(
            name=f"{name}_batched",
            dtype=variable.dtype,
            shape=(None, *variable.type.shape),
            dims=("__batch__", *variable.type.dims),
        )

    @classmethod
    def _date_indexed_data(cls, pymc_model, graph_ancestors: set) -> list:
        """Date-indexed ``pm.Data`` the graph reads, other than the two handled above.

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
        window_infos: list[dict],
        max_window: int,
        dates: pd.DatetimeIndex,
    ) -> dict[str, np.ndarray]:
        """Evaluate every node on all counterfactual scenarios in one call.

        Auxiliary date-indexed inputs are windowed here rather than by the
        caller, then broadcast from per-period to per-scenario via
        :attr:`CounterfactualScenarios.period_index`.

        Parameters
        ----------
        scenarios : CounterfactualScenarios
            Perturbed spend and its scenario bookkeeping.
        window_infos : list of dict
            Per-period window metadata from
            :meth:`Incrementality._compute_window_metadata`.
        max_window : int
            Padded window length.
        dates : pd.DatetimeIndex
            Dates of the fitted data.

        Returns
        -------
        dict
            Per response variable, an array of shape
            ``(n_scenarios, n_samples, max_window, *non_date_dims)``.
        """
        args: list[np.ndarray] = [scenarios.spend]
        if self.time_dtype is not None:
            args.append(
                Incrementality._build_time_index_array(
                    window_infos=window_infos,
                    dates=dates,
                    max_window=max_window,
                    dtype=self.time_dtype,
                )[scenarios.period_index]
            )
        for aux in self._aux:
            windowed = Incrementality._window_date_axis(
                values=aux.values,
                window_infos=window_infos,
                max_window=max_window,
                dtype=aux.dtype,
            )
            args.append(windowed[scenarios.period_index])
        return dict(zip(self.response_vars, self._evaluator(*args), strict=True))


class Incrementality:
    """Incrementality and counterfactual analysis for MMM models.

    Computes incremental channel contributions by comparing predictions with
    actual spend vs. counterfactual (perturbed) spend, accounting for
    adstock carryover effects.  See the :mod:`module docstring
    <pymc_marketing.mmm.incrementality>` for the full mathematical
    formulation and design rationale.

    Parameters
    ----------
    model : MMM
        Fitted MMM model instance.
    idata : xr.DataTree, optional
        DataTree containing posterior samples and fit data.
        If not provided, uses the incrementality test result data. Default is None.
    frozen_deterministics : dict[str, str], optional
        Mapping of deterministic variable names to group names for freezing.
        Variables in this dict will have their values frozen during
        counterfactual simulations. Default is empty dict.

    Attributes
    ----------
    idata : xr.DataTree
        Posterior samples and fit data.
    data : MMMIDataWrapper
        Data wrapper for accessing model data.

    Raises
    ------
    ValueError
        If both ``idata`` and ``data`` are provided, or neither is.

    Examples
    --------
    >>> incr = mmm.incrementality
    >>> roas = incr.contribution_over_spend(frequency="quarterly")
    >>> cac = incr.spend_over_contribution(frequency="monthly")
    """

    def __init__(
        self,
        model: MMM,
        idata: xr.DataTree | None = None,
        data: MMMIDataWrapper | None = None,
    ):
        if idata is not None and data is not None:
            raise ValueError("Provide either 'idata' or 'data', not both.")
        if idata is None and data is None:
            raise ValueError("Provide either 'idata' or 'data'.")

        self.model = model
        if data is not None:
            self.data = data
            self.idata = data.idata
        else:
            self.idata = idata
            self.data = MMMIDataWrapper.from_mmm(model, idata)

        in_model_not_idata, in_idata_not_model = self.data.compare_coords(model)
        if in_idata_not_model or in_model_not_idata:
            raise ValueError(
                "idata coordinates don't match the fitted model. "
                "Compute incrementality on the original (unfiltered) data "
                "first, then aggregate the results."
            )

    # ==================== Link Dispatch ====================

    @staticmethod
    def _stack_samples(da: xr.DataArray) -> xr.DataArray:
        """Flatten ``(chain, draw)`` into a single ``sample`` dimension.

        Uses the same C-order (chain-major) flattening as the batched graph
        evaluation, so positions along ``sample`` line up with the evaluated
        predictions.

        Every dimension other than ``chain`` and ``draw`` is kept, along with
        its coordinates: the reduction broadcasts these arrays against the
        perturbation by dimension *name*, so a panel model's custom dims have
        to survive the flattening.

        Parameters
        ----------
        da : xr.DataArray
            Array with ``chain`` and ``draw`` dimensions.

        Returns
        -------
        xr.DataArray
            Array with dimensions ``("sample", *other_dims)``, where
            ``other_dims`` are *da*'s remaining dimensions in their original
            order.
        """
        other_dims = tuple(d for d in da.dims if d not in ("chain", "draw"))
        stacked = da.transpose("chain", "draw", *other_dims)
        return xr.DataArray(
            stacked.values.reshape(-1, *stacked.shape[2:]),
            dims=("sample", *other_dims),
            coords={
                dim: stacked.coords[dim] for dim in other_dims if dim in stacked.coords
            },
        )

    def _mean_correction(
        self,
        posterior: xr.Dataset,
        central_tendency: CentralTendency,
    ) -> xr.DataArray:
        """Per-draw factor rescaling a median-scale prediction to the mean scale.

        Parameters
        ----------
        posterior : xr.Dataset
            Posterior samples (already subsampled).
        central_tendency : {"median", "mean"}
            Requested central tendency of the counterfactual predictions.

        Returns
        -------
        xr.DataArray
            Scalar ``1.0`` when no correction applies, otherwise a factor over
            ``sample`` and any dimensions the likelihood scale carries -- for a
            panel model the correction differs per custom-dim cell.
        """
        if central_tendency == "median":
            return xr.DataArray(1.0)

        correction = self.model._link_spec.mean_correction(
            posterior, self.model.output_var
        )
        if "chain" not in correction.dims:
            return correction
        return self._stack_samples(correction)

    def _baseline_response(self, posterior: xr.Dataset) -> xr.DataArray:
        """Baseline prediction on the response scale, flattened over samples.

        Parameters
        ----------
        posterior : xr.Dataset
            Posterior samples (already subsampled).

        Returns
        -------
        xr.DataArray
            ``{output_var}_original_scale`` with dimensions
            ``("sample", "date", *custom_dims)``.

        Raises
        ------
        ValueError
            If the deterministic is absent from the posterior.
        """
        name = f"{self.model.output_var}_original_scale"
        if name not in posterior:
            raise ValueError(
                f"Incrementality under link='{self.model.link}' needs the baseline "
                f"response-scale prediction '{name}', which is not in the posterior. "
                "It is registered automatically when a log-link model is built, so "
                "this posterior was most likely sampled with a restricted "
                "'var_names'. Refit without filtering it out."
            )
        return self._stack_samples(posterior[name])

    def _build_reducer(
        self,
        posterior: xr.Dataset,
        central_tendency: CentralTendency,
    ) -> IncrementalReducer:
        """Select the :class:`IncrementalReducer` matching the model's link.

        Parameters
        ----------
        posterior : xr.Dataset
            Posterior samples (already subsampled).
        central_tendency : {"median", "mean"}
            Requested central tendency of the counterfactual predictions.

        Returns
        -------
        IncrementalReducer
            Reducer that maps linear-predictor perturbations to response-scale
            increments.

        Raises
        ------
        NotImplementedError
            If the model's link function has no reducer.  Failing here is
            deliberate: silently reusing the additive reduction under a
            non-additive link returns numbers that are not incremental
            response.
        """
        correction = self._mean_correction(posterior, central_tendency)

        if self.model.link == LinkFunction.IDENTITY:
            return IdentityLinkReducer(
                scale=self.data.get_target_scale() * correction,
            )
        if self.model.link == LinkFunction.LOG:
            return LogLinkReducer(
                baseline_response=self._baseline_response(posterior) * correction,
            )
        raise NotImplementedError(
            f"Incrementality is not implemented for link='{self.model.link}'. "
            "Add an IncrementalReducer subclass describing how a change in the "
            "linear predictor maps to the response scale under this link."
        )

    # ==================== Effect Resolution ====================

    def _resolve_channel_dependent_effects(self) -> tuple[ChannelDependentEffect, ...]:
        """Find the ``mu_effects`` a spend counterfactual reaches.

        The counterfactual perturbs ``channel_data``, so any effect with
        ``channel_data`` among its ancestors carries part of the resulting change
        in the linear predictor and has to be evaluated alongside
        ``channel_contribution``.  A funnel mediator is the motivating case:
        upper-funnel spend moves latent demand, demand moves lower-funnel spend,
        and only then does the target respond.

        Effects that do not depend on ``channel_data`` -- a linear trend, an
        event window, seasonality -- are part of the baseline.  They cancel in
        the difference and are skipped without consulting their spec.

        Returns
        -------
        tuple of ChannelDependentEffect
            One entry per effect to include, in ``mu_effects`` order.  Empty for
            a model with no channel-dependent effects, which is the separable
            case the module was originally written for.

        Raises
        ------
        NotImplementedError
            If an effect depends on ``channel_data`` but has not opted in via
            :meth:`~pymc_marketing.mmm.additive_effect.MuEffect.incrementality_spec`,
            or if its contribution cannot be located in the model at all.  Both
            are refusals to guess: the alternative is dropping a real part of the
            increment and reporting the remainder as if it were the whole.
        ValueError
            If an included effect's contribution carries dimensions outside
            ``("date", *model.dims)``.
        """
        model = self.model.model
        channel_data = model[CounterfactualEvaluator.CHANNEL_DATA]
        allowed_dims = {"date", *self.model.dims}
        resolved: list[ChannelDependentEffect] = []

        for effect in self.model.mu_effects:
            label = type(effect).__name__
            try:
                name = effect.contribution_var_name
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"Incrementality cannot analyse the mu_effect {label!r} "
                    "because it does not expose 'contribution_var_name', so "
                    "there is no way to tell whether a spend counterfactual "
                    "reaches it.  Define that property (see MuEffect) to make "
                    "the effect analysable."
                ) from exc
            if name not in model.named_vars:
                raise NotImplementedError(
                    f"The mu_effect {label!r} declares its contribution as "
                    f"{name!r}, which is not a variable of the model, so "
                    "incrementality cannot tell whether a spend counterfactual "
                    "reaches it.  Register the contribution as a Deterministic "
                    "under that name."
                )

            node = model[name]
            if channel_data not in set(ancestors([node])):
                continue

            spec = effect.incrementality_spec()
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

            effect_dims = set(model.named_vars_to_dims.get(name, ()))
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
                    additional_carryover_lags=spec.additional_carryover_lags,
                )
            )

        return tuple(resolved)

    @staticmethod
    def _effective_l_max(l_max: int, effects: Sequence[ChannelDependentEffect]) -> int:
        """Evaluation-window half-length covering every path spend can take.

        Parameters
        ----------
        l_max : int
            The model's own ``adstock.l_max``, which bounds the direct path.
        effects : sequence of ChannelDependentEffect
            Included effects, each declaring how much further it propagates a
            change in spend.

        Returns
        -------
        int
            ``l_max`` plus the largest declared extra carryover.  A mediated
            path that chains a second adstock outlives the direct one, and a
            window sized for the direct path alone would cut the tail off.
        """
        return l_max + max(
            (effect.additional_carryover_lags for effect in effects), default=0
        )

    # ==================== Core Computation ====================

    def compute_incremental_contribution(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        include_carryover: bool = True,
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
        counterfactual_spend_factor: float = 0.0,
        central_tendency: CentralTendency = "median",
    ) -> xr.DataArray:
        r"""Compute incremental channel contributions using counterfactual analysis.

        Core incrementality function.  Compares the model's prediction under
        actual spend with its prediction under a counterfactual spend
        scenario, properly accounting for adstock carryover.  Results are
        always returned in the original scale of the target variable, with the
        model's link function applied -- see the :mod:`module docstring
        <pymc_marketing.mmm.incrementality>` for the full mathematical
        formulation and for what per-channel increments do and do not mean
        under a multiplicative (log-link) model.

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time aggregation frequency. ``"original"`` uses data's native
            frequency. ``"all_time"`` returns a single value across the entire
            period.
        start_date : str or pd.Timestamp, optional
            Start date for evaluation window. If None, uses start of fitted data.
        end_date : str or pd.Timestamp, optional
            End date for evaluation window. If None, uses end of fitted data.
        include_carryover : bool, default=True
            Include adstock carryover effects.  When True, prepends ``l_max``
            observations before the period to capture historical effects
            carrying into the evaluation period, and extends the evaluation
            window by ``l_max`` periods to capture trailing adstock effects
            from spend during the period.
        num_samples : int or None, optional
            Number of posterior samples to use. If None, all samples are used.
            If less than total available (chain × draw), a random subset is
            drawn.
        random_state : RandomState or Generator or None, optional
            Random state for reproducible subsampling.
            Only used when ``num_samples`` is not None.
        counterfactual_spend_factor : float, default=0.0
            Multiplicative factor *α* applied to channel spend in the
            counterfactual scenario.

            - ``0.0`` (default): Zeroes out channel spend → **total**
              incremental contribution (classic on/off counterfactual).
            - ``1.01``: Scales spend to 101% of actual → **marginal**
              incremental contribution (response to a 1 % spend increase).
            - Any value ≥ 0: Supported.  Values > 1 measure the upside of
              *more* spend; values in (0, 1) measure the cost of *less* spend.

            Note that *α* intervenes on **spend**, which is not the same as
            removing a channel's *effect*.  The two coincide only when the
            saturation maps zero spend to zero contribution (as
            :class:`~pymc_marketing.mmm.components.saturation.LogSaturation`
            does).  With a saturation whose value at zero spend is non-zero,
            or with ``time_varying_media`` scaling the contribution, ``0.0``
            still leaves a residual channel effect in the response.  To remove
            a component's effect outright, use
            :meth:`~pymc_marketing.mmm.mmm.MMM.compute_counterfactual_contributions_dataset`.
        central_tendency : {"median", "mean"}, default="median"
            Central tendency of the predictions being differenced.  Only
            meaningful for non-linear links: under ``link="log"`` the model's
            response-scale prediction :math:`\exp(\mu)\,s` is the *median* of
            the ``LogNormal`` likelihood, and ``"mean"`` rescales it by
            :math:`\exp(\sigma^2 / 2)` to give an increment on the
            conditional-mean scale.  Ignored under ``link="identity"``, where
            the Normal mean and median coincide.

        Returns
        -------
        xr.DataArray
            Incremental contributions in original scale with dimensions:

            - ``(chain, draw, date, channel, *custom_dims)`` when
              ``frequency != "all_time"``
            - ``(chain, draw, channel, *custom_dims)`` when
              ``frequency == "all_time"``

            For models with hierarchical dimensions like ``dims=("country",)``,
            output has shape ``(chain, draw, date, channel, country)``.

            **Sign convention**: The result is always
            ``Y(perturbed) − Y(actual)`` when *α > 1* and
            ``Y(actual) − Y(counterfactual)`` when *α < 1* (including 0).
            Both total and marginal incrementality are therefore positive for
            channels with a positive effect.

            **Estimand**: each channel's number is *leave-one-out* -- what would
            be lost without that channel, with the others at actual spend.  They
            sum to the total only when the response is additive in the channels;
            see :meth:`compute_joint_incremental_contribution`.

        Raises
        ------
        ValueError
            If frequency is invalid, period dates are outside fitted data
            range, ``counterfactual_spend_factor`` is negative, or
            ``central_tendency`` is not one of ``{"median", "mean"}``.
        NotImplementedError
            If the model's link function has no :class:`IncrementalReducer`, or a
            ``mu_effect`` that depends on channel spend has not opted in via
            :meth:`~pymc_marketing.mmm.additive_effect.MuEffect.incrementality_spec`.

        See Also
        --------
        compute_joint_incremental_contribution :
            All channels perturbed together, for a total rather than a split.

        References
        ----------
        Google MMM Paper:
        https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf


        Examples
        --------
        Compute quarterly incremental contributions:

        .. code-block:: python

            incremental = mmm.incrementality.compute_incremental_contribution(
                frequency="quarterly",
                start_date="2024-01-01",
                end_date="2024-12-31",
            )

        Mean contribution per channel per quarter:

        .. code-block:: python

            incremental.mean(dim=["chain", "draw"])

        Total annual contribution (all_time):

        .. code-block:: python

            annual = mmm.incrementality.compute_incremental_contribution(
                frequency="all_time",
                start_date="2024-01-01",
                end_date="2024-12-31",
            )

        Quarterly marginal incrementality (1 % spend increase):

        .. code-block:: python

            marginal = mmm.incrementality.compute_incremental_contribution(
                frequency="quarterly",
                counterfactual_spend_factor=1.01,
            )

        """
        return self._compute_increments(
            estimand="per_channel",
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=include_carryover,
            num_samples=num_samples,
            random_state=random_state,
            counterfactual_spend_factor=counterfactual_spend_factor,
            central_tendency=central_tendency,
        )

    def compute_joint_incremental_contribution(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        include_carryover: bool = True,
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
        counterfactual_spend_factor: float = 0.0,
        central_tendency: CentralTendency = "median",
    ) -> xr.DataArray:
        r"""Compute the incremental contribution of *all* channels together.

        Perturbs every channel in the same counterfactual and returns one number
        per period, rather than perturbing channels one at a time.

        This is a different estimand from summing
        :meth:`compute_incremental_contribution` over ``channel``, and the
        difference is not an error in either of them.  Per-channel increments are
        *leave-one-out*: each answers "what would we lose without this channel,
        holding the others at their actual spend".  Whenever the response is not
        additive in the channels -- under ``link="log"``, or when a ``mu_effect``
        mixes channels before they reach the response -- the two disagree,
        because interaction mass is counted by every channel that touches it.
        Under ``link="identity"`` with no channel-dependent effects they
        coincide exactly.

        Report this number when the question is "how much of the target does
        media drive in total", and the per-channel ones when the question is
        "which channel should I cut".  Adding the per-channel numbers up answers
        neither.

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time aggregation frequency, as in
            :meth:`compute_incremental_contribution`.
        start_date : str or pd.Timestamp, optional
            Start date for evaluation window.  If None, uses start of fitted data.
        end_date : str or pd.Timestamp, optional
            End date for evaluation window.  If None, uses end of fitted data.
        include_carryover : bool, default=True
            Include adstock carryover effects.
        num_samples : int or None, optional
            Number of posterior samples to use.
        random_state : RandomState or Generator or None, optional
            Random state for reproducible subsampling.
        counterfactual_spend_factor : float, default=0.0
            Multiplicative factor applied to *every* channel's spend.
        central_tendency : {"median", "mean"}, default="median"
            Central tendency of the predictions being differenced.

        Returns
        -------
        xr.DataArray
            Joint incremental contribution in original scale, with dimensions
            ``(chain, draw, date, *custom_dims)``, or without ``date`` when
            ``frequency == "all_time"``.  There is no ``channel`` dimension: the
            number is not attributable to a single channel.

        Examples
        --------
        Total media incrementality, and how much per-channel numbers overcount it:

        .. code-block:: python

            joint = mmm.incrementality.compute_joint_incremental_contribution(
                frequency="all_time"
            )
            loo = mmm.incrementality.compute_incremental_contribution(
                frequency="all_time"
            )
            overlap = (
                loo.sum("channel").mean(("chain", "draw"))
                / joint.mean(("chain", "draw"))
                - 1
            )

        See Also
        --------
        compute_incremental_contribution : Per-channel, leave-one-out increments.
        """
        return self._compute_increments(
            estimand="joint",
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=include_carryover,
            num_samples=num_samples,
            random_state=random_state,
            counterfactual_spend_factor=counterfactual_spend_factor,
            central_tendency=central_tendency,
        )

    def _compute_increments(
        self,
        *,
        estimand: Literal["per_channel", "joint"],
        frequency: Frequency,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
        include_carryover: bool,
        num_samples: int | None,
        random_state: RandomState | Generator | None,
        counterfactual_spend_factor: float,
        central_tendency: CentralTendency,
    ) -> xr.DataArray:
        """Shared machinery behind the per-channel and joint estimands.

        Resolves which nodes the counterfactual reaches, compiles one batched
        evaluator for them, builds the scenarios, evaluates, and reduces.  The
        two public entry points differ only in ``estimand``.

        Parameters
        ----------
        estimand : {"per_channel", "joint"}
            Whether to perturb channels one at a time or all together.
        frequency : Frequency
            Time aggregation frequency.
        start_date : str or pd.Timestamp, optional
            Start of the evaluation range.
        end_date : str or pd.Timestamp, optional
            End of the evaluation range.
        include_carryover : bool
            Whether to extend the evaluation window by the carryover length.
        num_samples : int or None
            Posterior subsample size.
        random_state : RandomState or Generator or None
            Seed for the subsample.
        counterfactual_spend_factor : float
            Multiplicative factor applied to spend in the counterfactual.
        central_tendency : {"median", "mean"}
            Central tendency of the differenced predictions.

        Returns
        -------
        xr.DataArray
            Incremental contributions in the original scale of the target.

        Raises
        ------
        ValueError
            If an input is out of range, or the date frequency cannot be inferred.
        NotImplementedError
            If the link has no reducer, or a channel-dependent ``mu_effect`` has
            not opted in.
        """
        # Validate inputs
        if counterfactual_spend_factor < 0:
            raise ValueError(
                f"counterfactual_spend_factor must be >= 0, got {counterfactual_spend_factor}"
            )
        if central_tendency not in ("median", "mean"):
            raise ValueError(
                f"central_tendency must be 'median' or 'mean', got {central_tendency!r}"
            )

        # Validate and parse dates
        start_date_ts, end_date_ts = self._validate_input(start_date, end_date)

        # Subsample posterior if needed (correctly across chain x draw)
        posterior_sub = subsample_draws(
            self.idata.posterior.dataset,
            num_samples=num_samples,
            random_state=random_state,
        )
        n_chains = posterior_sub.sizes["chain"]
        n_draws = posterior_sub.sizes["draw"]

        # Resolve the link-specific reduction and the effects to include before
        # compiling anything, so an unsupported link or an effect that has not
        # opted in fails fast rather than after the expensive work.
        reducer = self._build_reducer(posterior_sub, central_tendency)
        effects = self._resolve_channel_dependent_effects()

        # Create period groups based on frequency
        dates = self.data.dates
        periods = self._create_period_groups(start_date_ts, end_date_ts, frequency)

        # A mediated path outlives the direct one, so the window is sized for the
        # longest path spend can take, not just for the model's own adstock.
        l_max = self._effective_l_max(self.model.adstock.l_max, effects)
        inferred_freq: str | None = pd.infer_freq(dates)
        if inferred_freq is None:
            raise ValueError(
                "Could not infer frequency from the date index. "
                "Ensure the fitted data has a regular date frequency."
            )
        freq: str = inferred_freq
        freq_offset = pd.tseries.frequencies.to_offset(freq)

        # Compile one batched evaluator over every node the counterfactual
        # reaches: channel_contribution plus each included effect's contribution.
        posterior_predictive_model = self.model.model
        evaluator = CounterfactualEvaluator(
            pymc_model=posterior_predictive_model,
            posterior=posterior_sub,
            response_vars=[
                CHANNEL_CONTRIBUTION,
                *(effect.contribution_var for effect in effects),
            ],
            frozen_deterministics=self.model.frozen_deterministics,
            dates=dates,
        )

        # Evaluate baseline on full dataset (once).  Comparable to the windowed
        # counterfactuals because a window is clamped to the fitted dates and
        # carries l_max of history, so every evaluated date sees the same inputs
        # it would on the full axis.
        baseline_array = self.data.get_channel_data().values
        baseline = evaluator.evaluate_baseline(baseline_array)
        # Per node: (n_samples, n_dates, *non_date_dims)

        # Compute, for each period, the required window metadata including
        # the start/end indices into `dates` and any necessary left/right padding,
        # and determine the maximum window length across all periods.
        window_infos, max_window = self._compute_window_metadata(
            periods, dates, l_max, freq_offset, freq
        )

        # Where a channel sits among channel_data's non-date axes.  Needed only
        # in per-channel mode, and it is not always axis 0: panel models lay
        # channel_data out as (date, *custom_dims, channel).
        channel_data_dims = list(
            posterior_predictive_model.named_vars_to_dims.get(
                CounterfactualEvaluator.CHANNEL_DATA, ()
            )
        )
        channel_axis = None
        if effects:
            # An effect can mix channels before reaching the response, so no
            # per-channel column survives to be read off a single all-channels
            # perturbation.  One counterfactual per channel is then unavoidable.
            channel_axis = [d for d in channel_data_dims if d != "date"].index(
                "channel"
            )

        scenarios = self._build_counterfactual_scenarios(
            periods=periods,
            window_infos=window_infos,
            max_window=max_window,
            baseline_array=baseline_array,
            counterfactual_spend_factor=counterfactual_spend_factor,
            include_carryover=include_carryover,
            l_max=l_max,
            freq_offset=freq_offset,
            dtype=evaluator.channel_dtype,
            channel_axis=channel_axis,
            n_channels=len(self.model.channel_columns),
            include_joint=estimand == "joint",
        )

        # Evaluate all counterfactuals at once
        counterfactual = evaluator.evaluate_counterfactual(
            scenarios,
            window_infos=window_infos,
            max_window=max_window,
            dates=dates,
        )
        # Per node: (n_scenarios, n_samples, max_window, *non_date_dims)

        # Assemble results
        return self._compute_period_increments(
            periods=periods,
            scenarios=scenarios,
            baseline=baseline,
            counterfactual=counterfactual,
            non_date_dims=evaluator.non_date_dims,
            dates=dates,
            include_carryover=include_carryover,
            l_max=l_max,
            freq_offset=freq_offset,
            counterfactual_spend_factor=counterfactual_spend_factor,
            frequency=frequency,
            n_chains=n_chains,
            n_draws=n_draws,
            reducer=reducer,
            estimand=estimand,
        )

    def _validate_input(
        self,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Parse and validate input dates against the fitted data range.

        Parameters
        ----------
        start_date : str or pd.Timestamp or None
            Start date. If None, uses start of fitted data.
        end_date : str or pd.Timestamp or None
            End date. If None, uses end of fitted data.

        Returns
        -------
        tuple of (pd.Timestamp, pd.Timestamp)
            Validated ``(start_date, end_date)``.

        Raises
        ------
        ValueError
            If dates are outside fitted data range or start > end.
        """
        dates = self.data.dates
        data_start = dates[0]
        data_end = dates[-1]

        start_date_ts: pd.Timestamp = (
            data_start if start_date is None else pd.to_datetime(start_date)
        )
        end_date_ts: pd.Timestamp = (
            data_end if end_date is None else pd.to_datetime(end_date)
        )

        if start_date_ts < data_start:
            raise ValueError(
                f"start_date '{start_date_ts.date()}' is before fitted data "
                f"start '{data_start.date()}'."
            )
        if end_date_ts > data_end:
            raise ValueError(
                f"end_date '{end_date_ts.date()}' is after fitted data "
                f"end '{data_end.date()}'."
            )
        if start_date_ts > end_date_ts:
            raise ValueError(
                f"start_date '{start_date_ts.date()}' is after "
                f"end_date '{end_date_ts.date()}'."
            )

        return start_date_ts, end_date_ts

    @staticmethod
    def _compute_window_metadata(
        periods: list[tuple[pd.Timestamp, pd.Timestamp]],
        dates: pd.DatetimeIndex,
        l_max: int,
        freq_offset: BaseOffset,
        freq: str,
    ) -> tuple[list[dict], int]:
        """Compute per-period window metadata for counterfactual evaluation.

        For each period, determines the ideal window
        ``[t0 - l_max, t1 + l_max]`` and finds the fitted dates inside it.  The
        window is *clamped* to the fitted range rather than padded out to the
        ideal bounds, which is what makes a windowed evaluation agree with a
        full-axis one on every evaluated date.

        Parameters
        ----------
        periods : list of (pd.Timestamp, pd.Timestamp)
            Period ``(start, end)`` pairs.
        dates : pd.DatetimeIndex
            All dates from the fitted data.
        l_max : int
            Adstock maximum lag.
        freq_offset : pd.DateOffset
            Calendar-aware frequency offset.
        freq : str
            Pandas frequency string for ``date_range``.

        Returns
        -------
        tuple of (list[dict], int)
            ``(window_infos, max_window)`` where each dict has keys
            ``n_actual``, ``in_window`` and ``actual_dates``, and ``max_window``
            is the longest window, to which shorter ones are right-padded.
        """
        window_infos: list[dict] = []
        for t0, t1 in periods:
            # Reach back l_max for carry-in context and forward l_max so
            # carryover is captured; the eval mask decides what is summed.
            ideal_start = t0 - l_max * freq_offset
            ideal_end = t1 + l_max * freq_offset

            # Clamp to the fitted dates rather than padding out to the ideal
            # window.  At the start of the data there is no history to supply,
            # and the graph's own adstock already pads with zeros there, so a
            # window starting at the first fitted date reproduces the full-axis
            # evaluation exactly.  Padding it instead would inject l_max rows of
            # synthetic history, which is inert for spend (zero in, zero out) but
            # not for an effect whose contribution at zero spend is its own
            # intercept.
            in_window = (dates >= ideal_start) & (dates <= ideal_end)
            actual_dates = dates[in_window]

            window_infos.append(
                {
                    "n_actual": int(in_window.sum()),
                    "in_window": in_window,
                    "actual_dates": actual_dates,
                }
            )

        # Uniform length so the windows can be stacked for batched evaluation.
        # Short windows are padded on the right, where a causal filter cannot
        # reach back from any evaluated date.
        max_window = max(w["n_actual"] for w in window_infos)

        return window_infos, max_window

    @staticmethod
    def _window_date_axis(
        values: np.ndarray,
        window_infos: list[dict],
        max_window: int,
        dtype: str,
    ) -> np.ndarray:
        """Cut a date-indexed array into one padded window per period.

        Every date-indexed input of the graph has to be cut the same way, or the
        window evaluation would mix a windowed spend array with a full-length
        mediator array.  Trailing positions beyond the window's own length are
        zero; they exist only so windows of different lengths stack, and no
        evaluated date reaches them, since the filters involved look backwards.

        Parameters
        ----------
        values : np.ndarray
            Array with ``date`` as its first axis, shape ``(n_dates, *rest)``.
        window_infos : list of dict
            Per-period metadata from :meth:`_compute_window_metadata`.
        max_window : int
            Padded window length, uniform across periods so windows stack.
        dtype : str
            NumPy dtype of the output.

        Returns
        -------
        np.ndarray
            Shape ``(n_periods, max_window, *rest)``.
        """
        windowed = np.zeros(
            (len(window_infos), max_window, *values.shape[1:]), dtype=dtype
        )
        for period_idx, info in enumerate(window_infos):
            windowed[period_idx, : info["n_actual"]] = values[info["in_window"]].astype(
                dtype
            )
        return windowed

    @classmethod
    def _build_counterfactual_scenarios(
        cls,
        periods: list[tuple[pd.Timestamp, pd.Timestamp]],
        window_infos: list[dict],
        max_window: int,
        baseline_array: np.ndarray,
        counterfactual_spend_factor: float,
        include_carryover: bool,
        l_max: int,
        freq_offset: BaseOffset,
        dtype: str,
        channel_axis: int | None,
        n_channels: int,
        include_joint: bool,
    ) -> CounterfactualScenarios:
        """Build zero-padded counterfactual arrays for batched evaluation.

        Each counterfactual window covers the fitted dates in
        ``[t0 - l_max, t1 + l_max]``, giving every evaluated date its full
        carry-in history and capturing carry-out past the period.  Windows shorter
        than ``max_window`` are right-padded with zeros so they stack for batched
        evaluation; a causal filter cannot reach back from an evaluated date into
        that padding.

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
          off.  The joint scenario, when requested, is a further row per period.

        Parameters
        ----------
        periods : list of (pd.Timestamp, pd.Timestamp)
            Period ``(start, end)`` pairs.
        window_infos : list of dict
            Per-period metadata from :meth:`_compute_window_metadata`.
        max_window : int
            Maximum padded window size across all periods.
        baseline_array : np.ndarray
            Actual channel spend data, shape ``(n_dates, *extra_shape)``.
        counterfactual_spend_factor : float
            Multiplicative factor for counterfactual spend.
        include_carryover : bool
            Whether to include carryover effects in eval mask.
        l_max : int
            Evaluation-window half-length, already widened for any mediated
            path by :meth:`_effective_l_max`.
        freq_offset : pd.DateOffset
            Calendar-aware frequency offset.
        dtype : str
            NumPy dtype for the output array.
        channel_axis : int or None
            Axis of ``channel`` within ``baseline_array``'s non-date axes, or
            ``None`` to perturb all channels at once.  Panel models lay
            ``channel_data`` out as ``(date, *custom_dims, channel)``, so this is
            not always zero.
        n_channels : int
            Number of channels.
        include_joint : bool
            Whether to emit the all-channels scenario in per-channel mode.  It is
            always present in separable mode, where it is the only scenario.

        Returns
        -------
        CounterfactualScenarios
            Perturbed spend plus the bookkeeping needed to find the row for a
            given (period, channel) and to broadcast per-period arrays over
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
        windowed = cls._window_date_axis(
            values=baseline_array,
            window_infos=window_infos,
            max_window=max_window,
            dtype=dtype,
        )

        spend: list[np.ndarray] = []
        period_index: list[int] = []
        channel_index: list[int] = []
        rows: dict[tuple[int, int], int] = {}
        eval_masks: list[np.ndarray] = []
        period_labels: list[pd.Timestamp] = []

        for period_idx, ((t0, t1), info) in enumerate(
            zip(periods, window_infos, strict=True)
        ):
            period_labels.append(t1)

            # Offsets of [t0, t1] within the padded window: the dates the
            # counterfactual factor applies to.
            if info["n_actual"] > 0:
                in_target = (info["actual_dates"] >= t0) & (info["actual_dates"] <= t1)
                target_offsets = np.where(in_target)[0]
            else:
                target_offsets = np.array([], dtype=int)

            perturbed = [JOINT_CHANNEL] if separable else list(range(n_channels))
            if not separable and include_joint:
                perturbed.append(JOINT_CHANNEL)

            for channel in perturbed:
                padded = windowed[period_idx].copy()
                if channel == JOINT_CHANNEL:
                    padded[target_offsets] *= counterfactual_spend_factor
                else:
                    padded[(target_offsets, *channel_prefix, channel)] *= (
                        counterfactual_spend_factor
                    )
                rows[(period_idx, channel)] = len(spend)
                spend.append(padded)
                period_index.append(period_idx)
                channel_index.append(channel)

            if separable:
                # One row serves every channel, and it is the joint row too.
                joint_row = rows[(period_idx, JOINT_CHANNEL)]
                for channel in range(n_channels):
                    rows[(period_idx, channel)] = joint_row

            # Eval mask: actual-data positions in [t0, carryout_end].  Only
            # actual positions, so it stays consistent with the baseline
            # evaluation, which covers actual dates alone.
            cf_mask = np.zeros(max_window, dtype=bool)
            if info["n_actual"] > 0:
                eval_end = t1 + l_max * freq_offset if include_carryover else t1
                in_eval = (info["actual_dates"] >= t0) & (
                    info["actual_dates"] <= eval_end
                )
                cf_mask[np.where(in_eval)[0]] = True
            eval_masks.append(cf_mask)

        return CounterfactualScenarios(
            spend=np.stack(spend, axis=0),
            period_index=np.asarray(period_index, dtype=int),
            channel_index=np.asarray(channel_index, dtype=int),
            rows=rows,
            eval_masks=eval_masks,
            period_labels=period_labels,
        )

    @staticmethod
    def _build_time_index_array(
        window_infos: list[dict],
        dates: pd.DatetimeIndex,
        max_window: int,
        dtype: str,
    ) -> np.ndarray:
        """Build batched time_index arrays for counterfactual windows.

        Each window needs a corresponding ``time_index`` array so that
        HSGP-based latent variables (e.g. ``media_temporal_latent_multiplier``)
        evaluate their basis functions at the correct temporal positions.

        Parameters
        ----------
        window_infos : list of dict
            Per-period metadata from :meth:`_compute_window_metadata`.
        dates : pd.DatetimeIndex
            All dates from the fitted data.
        max_window : int
            Maximum padded window size across all periods.
        dtype : str
            NumPy dtype for the output array.

        Returns
        -------
        np.ndarray
            Time index array of shape ``(n_periods, max_window)``, where
            each row contains sequential integer indices corresponding to
            the temporal positions in the window.  Indices may extend
            beyond ``[0, n_dates)`` for boundary padding.
        """
        time_index_scenarios: list[np.ndarray] = []
        for info in window_infos:
            if info["n_actual"] > 0:
                start_idx = int(np.searchsorted(dates, info["actual_dates"][0]))
            else:
                start_idx = 0
            window_time_index = np.arange(
                start_idx, start_idx + max_window, dtype=dtype
            )
            time_index_scenarios.append(window_time_index)
        return np.stack(time_index_scenarios, axis=0)

    @staticmethod
    def _delta_mu(
        row: int,
        cf_mask: np.ndarray,
        window_dates: pd.DatetimeIndex,
        baseline: dict[str, np.ndarray],
        counterfactual: dict[str, np.ndarray],
        non_date_dims: dict[str, tuple[str, ...]],
        channel: int | None,
    ) -> xr.DataArray:
        r"""Change in the linear predictor over one period's evaluation window.

        The counterfactual reaches :math:`\mu` through ``channel_contribution``
        and through every included effect, and :math:`\mu` is their *sum*, so
        the perturbation is the sum of their perturbations:

        .. math::

            \Delta \mu_t = \Delta v_{t,m} + \sum_j \Delta e_{t,j}

        This is what keeps the :class:`IncrementalReducer` hierarchy out of the
        mediation problem entirely.  A reducer converts a change in the linear
        predictor into a change in the response; it does not care how many nodes
        that change was collected from.

        Parameters
        ----------
        row : int
            Row of the counterfactual predictions to read, from
            :attr:`CounterfactualScenarios.rows`.
        cf_mask : np.ndarray
            Boolean mask over the padded window selecting the evaluation dates.
        window_dates : pd.DatetimeIndex
            The evaluation dates themselves, used as coordinates so the reducer
            can align a baseline response by label.
        baseline : dict
            Per response variable, unperturbed predictions already restricted to
            the evaluation dates, shape ``(n_samples, n_eval_dates, *non_date_dims)``.
        counterfactual : dict
            Per response variable, predictions per scenario, shape
            ``(n_scenarios, n_samples, max_window, *non_date_dims)``.
        non_date_dims : dict
            Per response variable, its dimensions with ``date`` removed.
        channel : int or None
            Channel to read out of ``channel_contribution``, or ``None`` to sum
            the channel dimension away for the joint estimand.

        Returns
        -------
        xr.DataArray
            Dimensions ``("sample", "date", *custom_dims)``.
        """

        def delta(name: str) -> xr.DataArray:
            return xr.DataArray(
                counterfactual[name][row][:, cf_mask] - baseline[name],
                dims=("sample", "date", *non_date_dims[name]),
                coords={"date": window_dates},
            )

        channel_delta = delta(CHANNEL_CONTRIBUTION)
        delta_mu = (
            channel_delta.sum(dim="channel")
            if channel is None
            else channel_delta.isel(channel=channel, drop=True)
        )
        for name in non_date_dims:
            if name != CHANNEL_CONTRIBUTION:
                # xarray aligns by name, so an effect carrying only a subset of
                # the model's dimensions broadcasts without any reshaping here.
                delta_mu = delta_mu + delta(name)
        return delta_mu

    def _compute_period_increments(
        self,
        periods: list[tuple[pd.Timestamp, pd.Timestamp]],
        scenarios: CounterfactualScenarios,
        baseline: dict[str, np.ndarray],
        counterfactual: dict[str, np.ndarray],
        non_date_dims: dict[str, tuple[str, ...]],
        dates: pd.DatetimeIndex,
        include_carryover: bool,
        l_max: int,
        freq_offset: BaseOffset,
        counterfactual_spend_factor: float,
        frequency: Frequency,
        n_chains: int,
        n_draws: int,
        reducer: IncrementalReducer,
        estimand: Literal["per_channel", "joint"],
    ) -> xr.DataArray:
        """Assemble per-period incremental results into a single DataArray.

        For each period, forms the per-date change in the linear predictor over
        the evaluation window, hands it to *reducer* to be converted into a
        response-scale increment, applies the sign convention, reshapes the
        flattened sample dimension back to ``(chain, draw)``, and concatenates
        all periods into a single ``xr.DataArray``.

        Parameters
        ----------
        periods : list of (pd.Timestamp, pd.Timestamp)
            Period ``(start, end)`` pairs.
        scenarios : CounterfactualScenarios
            Scenario bookkeeping, used to find the row belonging to each
            (period, perturbation) pair.
        baseline : dict
            Per response variable, predictions on actual data over the full date
            axis.
        counterfactual : dict
            Per response variable, predictions per scenario.
        non_date_dims : dict
            Per response variable, its dimensions with ``date`` removed.
        dates : pd.DatetimeIndex
            All dates from the fitted data.
        include_carryover : bool
            Whether to include carryover effects in eval mask.
        l_max : int
            Evaluation-window half-length from :meth:`_effective_l_max`.
        freq_offset : pd.DateOffset
            Calendar-aware frequency offset.
        counterfactual_spend_factor : float
            Multiplicative factor used for sign convention.
        frequency : Frequency
            Time aggregation frequency.
        n_chains : int
            Number of MCMC chains in the posterior.
        n_draws : int
            Number of draws per chain.
        reducer : IncrementalReducer
            Link-specific reduction from a linear-predictor perturbation to a
            response-scale increment.  It owns the rescaling to original
            units, so no further ``target_scale`` multiplication happens here.
        estimand : {"per_channel", "joint"}
            ``"per_channel"`` perturbs one channel at a time and keeps a
            ``channel`` dimension; ``"joint"`` perturbs all channels together and
            returns a single number per period.

        Returns
        -------
        xr.DataArray
            Incremental contributions in original scale.  Dimensions
            ``(chain, draw, date, channel, *custom_dims)`` for
            ``"per_channel"``, without ``channel`` for ``"joint"``, and without
            ``date`` when ``frequency == "all_time"``.
        """
        fit_data = self.idata.fit_data
        channels = list(self.model.channel_columns)
        custom_dims = list(self.model.dims)
        out_dims = (
            ["channel", *custom_dims] if estimand == "per_channel" else custom_dims
        )
        results = []

        for period_idx, (t0, t1) in enumerate(periods):
            # Baseline: the same eval dates, taken from the full-dataset
            # prediction.  bl_mask picks [t0, eval_end] out of the full date
            # index; cf_mask picks the actual-data positions of that same range
            # out of the padded window, in the same order.
            eval_end = t1 + l_max * freq_offset if include_carryover else t1
            bl_mask = (dates >= t0) & (dates <= eval_end)
            cf_mask = scenarios.eval_masks[period_idx]

            # Unperturbed prediction over the same evaluation dates, in the same
            # order: bl_mask picks [t0, eval_end] out of the full date index,
            # cf_mask picks those same dates out of the padded window.
            period_baseline = {
                name: values[:, bl_mask] for name, values in baseline.items()
            }

            # (scenario key, channel to read) per output slice.  The joint
            # estimand reads a single scenario and sums the channel dimension
            # away; the per-channel one reads its own scenario per channel.
            slices: list[tuple[int, int | None]] = (
                [(idx, idx) for idx in range(len(channels))]
                if estimand == "per_channel"
                else [(JOINT_CHANNEL, None)]
            )
            deltas = [
                self._delta_mu(
                    row=scenarios.rows[(period_idx, key)],
                    cf_mask=cf_mask,
                    window_dates=dates[bl_mask],
                    baseline=period_baseline,
                    counterfactual=counterfactual,
                    non_date_dims=non_date_dims,
                    channel=channel,
                )
                for key, channel in slices
            ]
            delta_mu = (
                xr.concat(deltas, dim="channel").assign_coords(channel=channels)
                if estimand == "per_channel"
                else deltas[0]
            )

            # Link-specific reduction to a response-scale difference, then the
            # sign convention:
            # factor > 1 → Y(perturbed) - Y(actual)    (marginal)
            # factor < 1 → Y(actual) - Y(counterfactual) (total)
            total_incremental = reducer.counterfactual_minus_baseline(
                delta_mu
            ).transpose("sample", *out_dims)
            if counterfactual_spend_factor <= 1.0:
                total_incremental = -total_incremental
            # Shape: (n_samples, *out_dims) where n_samples = n_chains * n_draws

            # Reshape flattened sample → (chain, draw) to preserve MCMC structure
            reshaped = total_incremental.values.reshape(
                n_chains, n_draws, *total_incremental.shape[1:]
            )

            coords: dict = {
                "chain": np.arange(n_chains),
                "draw": np.arange(n_draws),
                **{dim: fit_data.coords[dim].values for dim in custom_dims},
            }
            if estimand == "per_channel":
                coords["channel"] = channels
            results.append(
                xr.DataArray(reshaped, dims=("chain", "draw", *out_dims), coords=coords)
                .assign_coords(date=scenarios.period_labels[period_idx])
                .expand_dims("date")
            )

        # Concatenate all periods
        if frequency == "all_time":
            # Single period, no date dimension
            result = results[0].squeeze("date", drop=True)
        else:
            result = xr.concat(results, dim="date")

        # Ensure standard dimension order
        dim_order = ["chain", "draw", "date", *out_dims]
        if frequency == "all_time":
            dim_order.remove("date")
        # Already on the original (response) scale: the reducer applied the
        # link's inverse transform and target_scale per draw, before summing.
        return result.transpose(*dim_order)

    # ==================== Convenience Methods ====================

    def contribution_over_spend(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        include_carryover: bool = True,
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
        central_tendency: CentralTendency = "median",
    ) -> xr.DataArray:
        """Compute incremental contribution per unit of spend.

        Wraps :meth:`compute_incremental_contribution` (with
        ``counterfactual_spend_factor=0``) and divides by total spend.
        The interpretation depends on the model's target variable --
        e.g. **ROAS** when the target is revenue, **customers per dollar**
        when the target is acquisitions.

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time aggregation frequency.
        start_date, end_date : str or pd.Timestamp, optional
            Date range for computation.
        include_carryover : bool, default=True
            Include adstock carryover effects.
        num_samples : int or None, optional
            Number of posterior samples to use. If None, all samples are used.
        random_state : RandomState or Generator or None, optional
            Random state for reproducible subsampling.
        central_tendency : {"median", "mean"}, default="median"
            Central tendency of the counterfactual predictions.  Only
            meaningful for non-linear links; see
            :meth:`compute_incremental_contribution`.

        Returns
        -------
        xr.DataArray
            Contribution per unit spend with dimensions
            ``(chain, draw, date, channel, *custom_dims)``.
            Zero spend results in NaN for that channel/period.

        Examples
        --------
        >>> roas = mmm.incrementality.contribution_over_spend(
        ...     frequency="quarterly",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31",
        ... )
        """
        incremental = self.compute_incremental_contribution(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=include_carryover,
            num_samples=num_samples,
            random_state=random_state,
            counterfactual_spend_factor=0.0,
            central_tendency=central_tendency,
        )

        spend = self._aggregate_spend(frequency, start_date, end_date)
        spend_safe = xr.where(spend == 0, np.nan, spend)

        return incremental / spend_safe

    def spend_over_contribution(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        include_carryover: bool = True,
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
        central_tendency: CentralTendency = "median",
    ) -> xr.DataArray:
        """Compute spend per unit of incremental contribution.

        Reciprocal of :meth:`contribution_over_spend`.  The interpretation
        depends on the model's target variable -- e.g. **CAC** (Customer
        Acquisition Cost) when the target is customer count

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time aggregation frequency.
        start_date, end_date : str or pd.Timestamp, optional
            Date range for computation.
        include_carryover : bool, default=True
            Include adstock carryover effects.
        num_samples : int or None, optional
            Number of posterior samples to use. If None, all samples are used.
        random_state : RandomState or Generator or None, optional
            Random state for reproducible subsampling.
        central_tendency : {"median", "mean"}, default="median"
            Central tendency of the counterfactual predictions.  Only
            meaningful for non-linear links; see
            :meth:`compute_incremental_contribution`.

        Returns
        -------
        xr.DataArray
            Spend per unit contribution with dimensions
            ``(chain, draw, date, channel, *custom_dims)``.
            Zero contribution results in Inf; zero spend results in NaN.

        Examples
        --------
        >>> cac = mmm.incrementality.spend_over_contribution(
        ...     frequency="monthly",
        ... )
        """
        ratio = self.contribution_over_spend(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=include_carryover,
            num_samples=num_samples,
            random_state=random_state,
            central_tendency=central_tendency,
        )

        return 1.0 / ratio

    def marginal_contribution_over_spend(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        include_carryover: bool = True,
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
        spend_increase_pct: float = 0.01,
        central_tendency: CentralTendency = "median",
    ) -> xr.DataArray:
        """Compute marginal contribution per additional unit of spend.

        Unlike :meth:`contribution_over_spend` which measures **total**
        efficiency (zero-out counterfactual), this method measures the
        **marginal** efficiency at the current spend level -- i.e. the slope
        of the response curve at the current operating point.  This captures
        diminishing returns: a heavily invested channel may have a low
        marginal efficiency even if its total efficiency is high.  See the
        :mod:`module docstring <pymc_marketing.mmm.incrementality>` for the
        marginal incrementality formula.

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time aggregation frequency.
        start_date, end_date : str or pd.Timestamp, optional
            Date range for computation.
        include_carryover : bool, default=True
            Include adstock carryover effects.
        num_samples : int or None, optional
            Number of posterior samples to use. If None, all samples are used.
        random_state : RandomState or Generator or None, optional
            Random state for reproducible subsampling.
        spend_increase_pct : float, default=0.01
            Fractional spend increase for the perturbation (default 1 %).
            Must be > 0.  Smaller values give a closer approximation to the
            true derivative but may suffer from numerical noise.
        central_tendency : {"median", "mean"}, default="median"
            Central tendency of the counterfactual predictions.  Only
            meaningful for non-linear links; see
            :meth:`compute_incremental_contribution`.

        Returns
        -------
        xr.DataArray
            Marginal contribution per unit spend with dimensions
            ``(chain, draw, date, channel, *custom_dims)``.
            Zero spend results in NaN for that channel/period.

        Raises
        ------
        ValueError
            If ``spend_increase_pct <= 0``.

        Examples
        --------
        >>> mroas = mmm.incrementality.marginal_contribution_over_spend(
        ...     frequency="quarterly",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31",
        ... )
        """
        if spend_increase_pct <= 0:
            raise ValueError(
                f"spend_increase_pct must be > 0, got {spend_increase_pct}"
            )

        factor = 1.0 + spend_increase_pct

        marginal_contribution = self.compute_incremental_contribution(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=include_carryover,
            num_samples=num_samples,
            random_state=random_state,
            counterfactual_spend_factor=factor,
            central_tendency=central_tendency,
        )

        spend = self._aggregate_spend(frequency, start_date, end_date)

        # Denominator is the *incremental* spend: pct * total_spend
        incremental_spend = spend_increase_pct * spend
        incremental_spend_safe = xr.where(
            incremental_spend == 0, np.nan, incremental_spend
        )

        return marginal_contribution / incremental_spend_safe

    # ==================== Period & Subsampling Helpers ====================

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def _create_period_groups(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        frequency: Frequency,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Create list of (period_start, period_end) tuples for given frequency.

        Parameters
        ----------
        start : pd.Timestamp
            Start of overall date range
        end : pd.Timestamp
            End of overall date range
        frequency : Frequency
            Time aggregation frequency

        Returns
        -------
        list of tuple
            List of (period_start, period_end) pairs. For "all_time", returns
            single tuple. For "original", returns one tuple per date. For other
            frequencies, returns tuples aligned to period boundaries (week-end,
            month-end, etc.).
        """
        if frequency == "all_time":
            return [(start, end)]

        if frequency == "original":
            # One tuple per date in the data's native frequency
            dates = pd.date_range(
                start,
                end,
                freq=pd.infer_freq(self.data.dates),
            )
            return [(d, d) for d in dates]

        # Map frequency to pandas period code
        freq_map = {
            "weekly": "W",
            "monthly": "M",
            "quarterly": "Q",
            "yearly": "Y",
        }

        dates = pd.date_range(start, end, freq="D")
        periods = dates.to_period(freq_map[frequency])
        unique_periods = periods.unique()

        # Validate that end aligns with a period boundary.
        last_period_boundary = unique_periods[-1].to_timestamp(how="end").normalize()
        if end < last_period_boundary:
            data_last_date = self.data.dates[-1]
            if end != data_last_date:
                raise ValueError(
                    f"end_date ({end.strftime('%Y-%m-%d')}) falls in the "
                    f"middle of a {frequency} period that ends on "
                    f"{last_period_boundary.strftime('%Y-%m-%d')}. "
                    f"Use an end_date that aligns with a {frequency} "
                    f"boundary, or omit end_date to use the last date "
                    f"of the fitted data "
                    f"({data_last_date.strftime('%Y-%m-%d')})."
                )

        period_ranges = []
        for period in unique_periods:
            period_start = period.to_timestamp()
            period_end = period.to_timestamp(how="end").normalize()

            # Clip start to the requested range (needed when the user
            # passes a start date inside a period)
            period_start = max(period_start, start)

            period_ranges.append((period_start, period_end))

        return period_ranges

    def _aggregate_spend(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> xr.DataArray:
        """Aggregate channel spend by frequency over a date range.

        Delegates to self.data (MMMIDataWrapper) for date filtering and time
        aggregation.

        Parameters
        ----------
        frequency : Frequency
            Time aggregation frequency
        start_date, end_date : str or pd.Timestamp, optional
            Date range. If None, uses full fitted data range.

        Returns
        -------
        xr.DataArray
            Aggregated spend with dims (date, channel, *custom_dims) or
            (channel, *custom_dims) for "all_time"
        """
        # 1. Filter to date range
        data = self.data.filter_dates(start_date, end_date)

        # 2. Aggregate over time (no-op for "original")
        if frequency != "original":
            data = data.aggregate_time(period=frequency, method="sum")

        # 3. Return spend with channel dimension
        return data.get_channel_spend()
