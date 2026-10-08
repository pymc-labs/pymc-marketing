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
  behind the model's own outlives it, so the evaluation window is sized for
  the longest path spend can take -- measured on the graph, not declared.

Both the measurement and the check that the increment is complete live in
:mod:`~pymc_marketing.mmm.spend_reach`, which reads them off single-date spend
perturbations.  This module asks it for a window length and a
mode and otherwise knows nothing about either.

Estimands
---------
:meth:`Incrementality.compute_incremental_contribution` is a *unilateral
intervention*: each channel's number answers "what changes if this channel's
spend is scaled by ``counterfactual_spend_factor``, holding the others at
their actual spend".  At the default ``factor = 0`` that is the familiar
leave-one-out question, "what would we lose without this channel"; at
``0.5`` it is a halving and at ``1.01`` a one-percent increase, and neither
is a leave-one-out.
:meth:`Incrementality.compute_joint_incremental_contribution` applies the
same factor to every channel at once and answers "how much does media drive
in total".

Both intervene on *spend*, which is not the same as removing a channel's
term from the model: at ``factor = 0`` they coincide only where zero spend
produces zero contribution, as it does for a saturation through the origin
but not for one with an intercept.

The two estimands agree only when the response is additive in the channels,
that is under ``link="identity"`` with no channel-dependent effect.
Otherwise they differ by the interaction between the channels, and the sign
of the gap is not fixed: at ``factor = 0`` with strictly positive
contributions the unilateral numbers sum to *more* than the joint, because
interaction mass is counted by every channel that touches it, but with
contributions of mixed sign, or with ``factor > 1``, the gap can go the
other way.  Summing per-channel increments is not a way to get a total
either way, and the gap is a property of the model rather than an error in
either number.

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

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import ConfigDict, validate_call

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
from pymc_marketing.data.idata.schema import Frequency
from pymc_marketing.data.idata.utils import subsample_draws
from pymc_marketing.mmm.counterfactual import (
    CounterfactualEvaluator,
    CounterfactualScenarios,
    Estimand,
    EvaluationWindows,
)
from pymc_marketing.mmm.link import LinkFunction
from pymc_marketing.mmm.spend_reach import (
    _PKG_PREFIX,
    CHANNEL_CONTRIBUTION,
    SpendProbe,
    linear_predictor,
    resolve_channel_dependent_effects,
)
from pymc_marketing.mmm.transformers import ConvMode

if TYPE_CHECKING:
    from numpy.random import Generator, RandomState
    from pandas.tseries.offsets import BaseOffset

    from pymc_marketing.mmm.counterfactual import PeriodWindow
    from pymc_marketing.mmm.mmm import MMM
    from pymc_marketing.mmm.spend_reach import SpendReach

__all__ = [
    "CentralTendency",
    "CounterfactualEvaluator",
    "CounterfactualScenarios",
    "Estimand",
    "EvaluationWindows",
    "IdentityLinkReducer",
    "IncrementalReducer",
    "Incrementality",
    "LogLinkReducer",
    "PeriodIncrements",
    "kernel_trailing_lags",
]

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
    Every subclass assumes *delta* is the **complete** change in the linear
    predictor: ``channel_contribution`` plus every channel-dependent
    ``mu_effect`` the counterfactual reaches.  Collecting those terms is the
    caller's job (:meth:`Incrementality._delta_mu`), and that the collected
    nodes account for the whole move is checked rather than assumed, by
    :meth:`~pymc_marketing.mmm.spend_reach.SpendProbe.assert_increment_is_complete`.
    A reducer only converts that change into a response-scale increment; it
    does not care how many nodes the change was collected from.

    See Also
    --------
    IdentityLinkReducer : Additive response (``link="identity"``).
    LogLinkReducer : Multiplicative response (``link="log"``).
    """

    @abstractmethod
    def per_date_increment(self, delta: xr.DataArray) -> xr.DataArray:
        r"""Return :math:`\hat{Y}^{\text{cf}}_t - \hat{Y}_t` for every date.

        The per-date term every reduction is built from.  Keeping the ``date``
        axis is what lets
        :meth:`Incrementality.split_incremental_contribution_over_time` say
        *when* a period's increment lands, rather than only how large it is in
        total.

        Parameters
        ----------
        delta : xr.DataArray
            As in :meth:`counterfactual_minus_baseline`.

        Returns
        -------
        xr.DataArray
            Response-scale difference per date.  Every dimension of *delta*,
            ``date`` included, is preserved.
        """

    def counterfactual_minus_baseline(self, delta: xr.DataArray) -> xr.DataArray:
        r"""Return :math:`\sum_t [\hat{Y}^{\text{cf}}_t - \hat{Y}_t]`.

        Parameters
        ----------
        delta : xr.DataArray
            :math:`\Delta_{t,m}`, the counterfactual-minus-baseline change in
            the linear-predictor contribution.  It carries ``"sample"``,
            ``"date"``, the model's own non-date dimensions and -- for the
            per-channel estimand -- ``"channel"``, and **no dimension order is
            guaranteed**: the per-channel path concatenates along ``"channel"``
            last, which puts it first, while a panel model's own dims arrive in
            the layout the graph produced.  Reductions here are by dimension
            *name* for exactly that reason.  The ``date`` coordinates span the
            evaluation window of a single period.

        Returns
        -------
        xr.DataArray
            Response-scale difference summed over ``date``.  The ``date``
            dimension is dropped; every other dimension is preserved.
        """
        return self.per_date_increment(delta).sum(dim="date")


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

    def per_date_increment(self, delta: xr.DataArray) -> xr.DataArray:
        """See :meth:`IncrementalReducer.per_date_increment`."""
        return delta * self.scale


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

    def per_date_increment(self, delta: xr.DataArray) -> xr.DataArray:
        """See :meth:`IncrementalReducer.per_date_increment`."""
        baseline = self.baseline_response.sel(date=delta.coords["date"])
        return baseline * np.expm1(delta)


def kernel_trailing_lags(l_max: int, mode: ConvMode) -> int:
    """Lags after a spend date that the adstock kernel itself reaches.

    Follows the padding in
    :func:`~pymc_marketing.mmm.transformers.batched_convolution`: ``After``
    places the kernel's ``l_max`` weights on lags ``0 .. l_max - 1``, ``Overlap``
    centres them so ``l_max // 2`` fall after the spend date, and ``Before``
    places them all on or before it.
    :meth:`~pymc_marketing.mmm.counterfactual.EvaluationWindows.build` ends the
    ``Overlap`` and ``Before`` evaluation ranges the same way; its windowed
    ``After`` range keeps one date of slack past the last lag.

    The one definition of how far a plain kernel carries.  :class:`Incrementality`
    passes it to :meth:`~pymc_marketing.mmm.spend_reach.SpendProbe.measure` as
    the floor under the measured horizon, and under full-axis evaluation, where
    no horizon was measured, uses it as a lower bound on how far a period's
    carryover runs.

    Parameters
    ----------
    l_max : int
        The adstock's number of kernel weights.
    mode : ConvMode
        The adstock's convolution mode.

    Returns
    -------
    int
        The last lag the kernel places weight on, ``0`` for ``Before``.

    Raises
    ------
    ValueError
        For an unknown mode.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.mmm.incrementality import kernel_trailing_lags
        from pymc_marketing.mmm.transformers import ConvMode

        kernel_trailing_lags(8, ConvMode.After)  # 7
        kernel_trailing_lags(8, ConvMode.Overlap)  # 4
        kernel_trailing_lags(8, ConvMode.Before)  # 0
    """
    if mode == ConvMode.After:
        return int(l_max - 1)
    if mode == ConvMode.Overlap:
        return int(l_max // 2)
    if mode == ConvMode.Before:
        return 0
    raise ValueError(f"Wrong Mode: {mode}, expected one of {', '.join(ConvMode)}")


@dataclass(frozen=True)
class PeriodIncrements:
    """Every period's increment, before the periods are laid out together.

    What the evaluation shared by every :class:`Incrementality` method returns,
    so that each public method chooses its own layout without re-deriving what
    the evaluation established: stacked totals for
    :meth:`~Incrementality.compute_incremental_contribution`, a dense matrix for
    :meth:`~Incrementality.split_incremental_contribution_over_time`, and one
    band at a time for
    :meth:`~Incrementality.split_incremental_contribution_current_future`.

    Parameters
    ----------
    periods : list of xr.DataArray
        One per period, in period order.  Either the period's total, with a
        length-one ``date`` dimension holding the period end, or its band: the
        per-date increment on ``realization_date`` over the period's evaluated
        dates plus its unobserved tail, with a boolean ``observed`` coordinate.
    windows : EvaluationWindows
        The windows the periods were evaluated over, in the same order.
    reach : SpendReach
        What the probe measured about how far spend moves the evaluated nodes.
    freq_offset : BaseOffset
        The data's date frequency, as resolved and validated for the evaluation.
    """

    periods: list[xr.DataArray]
    windows: EvaluationWindows
    reach: SpendReach
    freq_offset: BaseOffset


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
        Fitted MMM model instance.  Its ``frozen_deterministics`` property
        decides which deterministics are held at their posterior values during
        counterfactual evaluation.
    idata : xr.DataTree, optional
        DataTree containing posterior samples and fit data.  Exactly one of
        ``idata`` and ``data`` must be provided.
    data : MMMIDataWrapper, optional
        Existing data wrapper to reuse instead of building one from ``idata``.

    Attributes
    ----------
    model : MMM
        The fitted model whose graph the counterfactuals are evaluated on.
    idata : xr.DataTree
        Posterior samples and fit data.
    data : MMMIDataWrapper
        Data wrapper for accessing model data.

    Raises
    ------
    ValueError
        If both ``idata`` and ``data`` are provided, or neither is; or if the
        idata coordinates do not match the fitted model's.

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

    def _mean_scale_factor(
        self,
        posterior: xr.Dataset,
        central_tendency: CentralTendency,
    ) -> xr.DataArray:
        """Per-draw factor rescaling a median-scale prediction to the mean scale.

        An incrementality reducer folds this into a scale, so only a
        *multiplicative* correction can be used here.  Under the identity link
        with a ``TruncatedNormal`` likelihood the correction is an offset that
        is nonlinear in ``mu``, so it does not cancel in a difference of two
        predictions and cannot be expressed as a factor:
        :meth:`~pymc_marketing.mmm.link.LinkSpec.mean_scale_factor` raises
        there.  That is why
        :meth:`~pymc_marketing.mmm.mmm.MMM.compute_counterfactual_contributions_dataset`
        can return mean-scale numbers for a model on which
        ``central_tendency="mean"`` fails here: it applies the offset to a
        level, while this applies a factor to a difference.

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

        Raises
        ------
        ValueError
            If ``E[y]`` is undefined for the likelihood, or if its correction
            is an offset rather than a factor.
        """
        if central_tendency == "median":
            return xr.DataArray(1.0)

        correction = self.model._link_spec.mean_scale_factor(
            posterior,
            self.model.model_config["likelihood"],
            self.model.output_var,
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
        correction = self._mean_scale_factor(posterior, central_tendency)

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
            carrying into the evaluation period. The dates summed outside the
            period follow ``adstock.mode``: after the period for ``After``,
            before it for ``Before``, and on both sides for ``Overlap``.
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
            conditional-mean scale.  Under ``link="identity"`` it is a no-op
            for the likelihoods whose mean is ``mu``, but it *raises* for
            ``TruncatedNormal``: the truncation correction is an offset that
            is nonlinear in ``mu``, so it neither cancels in the difference
            nor folds into the reducer's scale.  Use ``"median"`` there, or
            :meth:`~pymc_marketing.mmm.mmm.MMM.compute_counterfactual_contributions_dataset`,
            which corrects a level rather than a difference.

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

            **Estimand**: each channel's number is a *unilateral intervention*
            -- what changes when that channel's spend is scaled by *α*, with the
            others at actual spend.  At *α = 0* that is the leave-one-out
            question.  The numbers sum to the total only when the response is
            additive in the channels; see
            :meth:`compute_joint_incremental_contribution`.

        Raises
        ------
        ValueError
            If frequency is invalid, period dates are outside fitted data
            range, ``counterfactual_spend_factor`` is negative, or
            ``central_tendency`` is not one of ``{"median", "mean"}``.  Also
            raised if a ``mu_effect`` declares fewer carryover lags, or a
            narrower ``evaluation_mode``, than a spend counterfactual was
            measured to need (a declaration narrower than what was measured is
            refused rather than silently overridden); if the model produces
            non-finite predictions (NaN or infinity) for some posterior draw,
            which usually means a transform is dividing zero by zero; or if a
            post-fit mutation of an auxiliary date-indexed input is detected
            (``MMM.sample_posterior_predictive(..., clone_model=False)`` or a
            direct ``pm.set_data(...)`` call after fitting).  See
            :mod:`~pymc_marketing.mmm.spend_reach` for the full story on each.
        NotImplementedError
            If the model's link function has no :class:`IncrementalReducer`, or a
            ``mu_effect`` that depends on channel spend has not opted in via
            :meth:`~pymc_marketing.mmm.additive_effect.MuEffect.incrementality_spec`.
            Also raised if the accounted nodes (``channel_contribution`` plus
            the resolved effects) do not reproduce the full move in the linear
            predictor, which means some path from spend to the response is
            unattributed; see
            :meth:`~pymc_marketing.mmm.spend_reach.SpendProbe.assert_increment_is_complete`.

        Warns
        -----
        UserWarning
            If a spend counterfactual's reach could not be measured because no
            interior date could be probed, the evaluation falls back to
            evaluating every period on the full date axis instead of a window,
            which is correct but slower, and the completeness check above is
            skipped for lack of anything to compare it against.  See
            :meth:`~pymc_marketing.mmm.spend_reach.SpendProbe.measure`.

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
        increments = self._compute_increments(
            scope="per_channel",
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=include_carryover,
            num_samples=num_samples,
            random_state=random_state,
            counterfactual_spend_factor=counterfactual_spend_factor,
            central_tendency=central_tendency,
        )
        return self._stack_periods(increments, frequency)

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
        *unilateral*: each answers "what changes if this channel's spend is
        scaled by *α*, holding the others at their actual spend" -- at the
        default *α = 0*, "what would we lose without this channel".  Whenever the
        response is not additive in the channels -- under ``link="log"``, or when
        a ``mu_effect`` mixes channels before they reach the response -- the two
        disagree by the interaction between the channels.  Under
        ``link="identity"`` with no channel-dependent effects they coincide
        exactly.

        The direction of the disagreement is not fixed.  With strictly positive
        contributions at *α = 0* the unilateral numbers sum to more than the
        joint, since interaction mass is counted by every channel that touches
        it; with contributions of mixed sign, or with *α > 1*, the sum can fall
        short instead.  Either way it is not a total.

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
        Total media incrementality, and how far the per-channel numbers are from
        it:

        .. code-block:: python

            joint = mmm.incrementality.compute_joint_incremental_contribution(
                frequency="all_time"
            )
            unilateral = mmm.incrementality.compute_incremental_contribution(
                frequency="all_time"
            )
            interaction = (
                unilateral.sum("channel").mean(("chain", "draw"))
                / joint.mean(("chain", "draw"))
                - 1
            )

        See Also
        --------
        compute_incremental_contribution : Per-channel, unilateral increments.
        """
        increments = self._compute_increments(
            scope="joint",
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=include_carryover,
            num_samples=num_samples,
            random_state=random_state,
            counterfactual_spend_factor=counterfactual_spend_factor,
            central_tendency=central_tendency,
        )
        return self._stack_periods(increments, frequency)

    def split_incremental_contribution_over_time(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        estimand: Literal["counterfactual", "allocation"] = "counterfactual",
        method: Literal["pipeline", "closed_form"] = "pipeline",
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
        counterfactual_spend_factor: float = 0.0,
        central_tendency: CentralTendency = "median",
    ) -> xr.DataArray:
        r"""Split each period's incremental contribution by when it lands.

        :meth:`compute_incremental_contribution` evaluates the response to each
        period's spend on every date the carryover reaches, then sums those
        dates.  This method keeps them.  Entry :math:`A[s, u]` is the increment
        that spend in period :math:`s` produces on date :math:`u`, so

        .. math::

            \sum_u A[s, u] = \text{compute\_incremental\_contribution}[s]

        holds to floating-point tolerance, for both links and either
        ``adstock_first``.  Rows are the carryover-inclusive value of a period's
        spend; reading a column instead (what landed on a date, from all earlier
        spend) is a different estimand when the response mixes cohorts, which is
        what ``estimand="allocation"`` will provide.

        In particular, the entries do not add up to ``channel_contribution``
        when saturation follows adstock (``adstock_first=True``, the default).
        Each row zeroes one period's spend against all the others, so with a
        concave response the period increments summed over ``spend_date``
        come in below the channel's total contribution, by how much depends on
        how saturated the channel is.  Only the all-time counterfactual, which
        zeroes every period at once, matches ``channel_contribution``.  Use
        these numbers to value a period's spend, not to decompose the reported
        total; ``estimand="allocation"`` is the one that will reconcile with
        it.  With ``adstock_first=False`` the response is separable by cohort
        and the two agree.

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Aggregation of the spend periods, as in
            :meth:`compute_incremental_contribution`.  At ``"original"`` a row is
            one date's spend and the realization axis is a lag axis; at an
            aggregated frequency a row is a period cohort spanning the period's
            dates plus the carryover after them.
        start_date, end_date : str or pd.Timestamp, optional
            Range of spend periods.  Defaults to the fitted data.
        estimand : {"counterfactual", "allocation"}, default="counterfactual"
            ``"counterfactual"`` zeroes (or scales) one period's spend and records
            the per-date response, which reconciles with today's incrementality
            row by row.  ``"allocation"`` (Aumann-Shapley, reconciling with
            ``channel_contribution`` column by column) is not implemented yet.
        method : {"pipeline", "closed_form"}, default="pipeline"
            ``"pipeline"`` evaluates the model graph and is exact for every
            supported model.  ``"closed_form"`` (cumulative adstock weights,
            valid only for cohort-separable models) is not implemented yet.
        num_samples : int or None, optional
            Number of posterior samples to use; all of them when None.
        random_state : RandomState or Generator or None, optional
            Seed for the subsample.
        counterfactual_spend_factor : float, default=0.0
            As in :meth:`compute_incremental_contribution`.
        central_tendency : {"median", "mean"}, default="median"
            As in :meth:`compute_incremental_contribution`.

        Returns
        -------
        xr.DataArray
            Dims ``(chain, draw, spend_date, realization_date, channel,
            *custom_dims)``; for ``"all_time"`` ``spend_date`` is a scalar
            coordinate rather than a dimension.  The
            posterior is returned draw by draw: any summary belongs after
            whatever reduction the caller applies.  Coordinates:

            - ``spend_date``: the period end, matching ``date`` in
              :meth:`compute_incremental_contribution`; ``period_start`` on the
              same dimension gives the other bound.
            - ``observed`` on ``(spend_date, realization_date)``: ``False``
              where the period's carryover runs past the end of the fitted data.
              Those entries are ``NaN``, and they run from the first date after
              the last fitted date to the period's last spend date plus
              ``effective_horizon``.

            Entries outside a period's evaluated dates are ``0``.  With a
            windowed evaluation (the usual case) the probe measured every date
            the period's spend moves, so these are zero by construction.  When
            the probe could not bound the reach, every period is evaluated on
            the full date axis (``assumptions["evaluation"] == "full_axis"``),
            and two things change.

            First, which dates a row covers follows ``adstock.mode``.  Under
            ``ConvMode.After`` a row runs from the period's start to the end of
            the axis, and the dates before the period are left out by
            convention: a node that reduces over ``date`` can move them, and
            keeping them would make rows overlap.  A leading kernel places real
            mass before the period, so those dates are kept: from the start of
            the axis to its end under ``ConvMode.Overlap``, and to the period's
            end under ``ConvMode.Before``.

            Second, no horizon was measured, so the unobserved tail runs to the
            period's last spend date plus the kernel's own trailing lags
            (``l_max - 1`` under ``After``, ``l_max // 2`` under ``Overlap``,
            none under ``Before``).  ``observed=False`` is then a lower bound
            on what the data cannot show, because a node that reduces over
            ``date``, or a mediated path, can carry the effect further.
            ``observed`` only ever marks the trailing side: a leading kernel's
            mass before the first fitted date has no date on the axis to mark.

            Its ``attrs``:

            - ``estimand`` and ``method``, as passed.
            - ``effective_horizon``: the longest carryover lag, in data periods
              (:attr:`~pymc_marketing.mmm.spend_reach.SpendReach.max_lag`).
              The kernel's own trailing lags for a plain adstock, ``l_max - 1``
              under ``ConvMode.After``, and longer where the probe measured a
              mediated path that outlives them.  Absent under full-axis
              evaluation, where there is no measured horizon;
              ``attrs.get("effective_horizon")`` then gives ``None``.
            - ``assumptions``: a JSON string (``json.loads`` it) recording the
              model settings the matrix depends on and ``"evaluation"``,
              ``"window"`` or ``"full_axis"``.  A string, like every attribute
              here, so the result can be written with ``to_netcdf``.
            - ``warnings``: notes on how to read the result, as a JSON string
              holding a list (``json.loads`` it), for the same reason.

        Raises
        ------
        NotImplementedError
            For ``estimand="allocation"`` or ``method="closed_form"``.
        ValueError
            For an unknown ``estimand`` or ``method``, and for everything
            :meth:`compute_incremental_contribution` raises on.

        Warns
        -----
        UserWarning
            If ``adstock.mode`` is not ``ConvMode.After``.  The entries remain
            correct, and dates before the spend period carry the kernel's
            leading mass, but "realized now vs later" does not describe them.

        Notes
        -----
        Nothing past ``l_max`` is modelled.  With ``l_max=13`` on weekly data the
        future value reported here is future-within-a-quarter, whatever the
        channel's real carryover; raising ``l_max`` to lengthen the tail buys a
        number driven by the adstock prior, not by the data.

        The matrix is dense in ``realization_date``, so that columns line up
        across rows and ``sel`` works directly.  Reductions need care on two
        counts.  With xarray's default ``skipna`` they count the unobserved cells
        as zero: ``A.sum("spend_date")`` is ``0`` on every date past the last
        fitted one instead of a gap, and
        ``A.resample(realization_date="MS").sum()`` gives a finite total for a
        month the data only partly shows.  Pass ``skipna=False`` to any reduction
        over the matrix to keep those cells ``NaN``.  And because ``observed`` is
        defined on both ``spend_date`` and ``realization_date``, a reduction or
        ``resample`` over either one drops it, so reduce it alongside if you
        need it, for example ``A.observed.all("spend_date")``.  Row sums are the
        one deliberate exception: ``A.sum("realization_date")`` with the default
        ``skipna`` adds up the observed part of each period's effect, which is
        what :meth:`compute_incremental_contribution` reports.

        At ``frequency="original"`` the matrix's size is
        ``n_samples x n_dates x (n_dates + effective_horizon) x n_channels``;
        use ``num_samples`` or an aggregated frequency on long series.
        :meth:`split_incremental_contribution_current_future` works on each
        period's band instead and never builds this matrix.

        See Also
        --------
        split_incremental_contribution_current_future : The matrix reduced to
            realized vs future value.

        Examples
        --------
        .. code-block:: python

            A = mmm.incrementality.split_incremental_contribution_over_time(
                frequency="monthly", num_samples=500, random_state=0
            )
            # Reconciles with today's per-period incrementality:
            A.sum("realization_date")
            # Increments landing on each date, NaN where the data cannot show
            # it.  With adstock_first=True (the default) this is not the
            # channel_contribution on that date: see the summary above.
            A.sum("spend_date", skipna=False)
        """
        if estimand not in ("counterfactual", "allocation"):
            raise ValueError(
                f"estimand must be 'counterfactual' or 'allocation', got {estimand!r}"
            )

        if method not in ("pipeline", "closed_form"):
            raise ValueError(
                f"method must be 'pipeline' or 'closed_form', got {method!r}"
            )

        if estimand == "allocation":
            raise NotImplementedError(
                "estimand='allocation' (Aumann-Shapley shares that reconcile with "
                "channel_contribution) is not implemented yet. See "
                "https://github.com/pymc-labs/pymc-marketing/issues/2941."
            )

        if method == "closed_form":
            raise NotImplementedError(
                "method='closed_form' is not implemented yet; use "
                "method='pipeline', which is exact for every supported model. See "
                "https://github.com/pymc-labs/pymc-marketing/issues/2941."
            )

        mode = self.model.adstock.mode
        notes: list[str] = []

        if mode != ConvMode.After:
            message = (
                f"adstock.mode is {mode!s}, so spend moves dates before its own "
                "period.  The carryover matrix records those dates, but they are "
                "leading kernel mass, not value realized after the spend, and a "
                "current-vs-future reading of the matrix does not apply."
            )
            notes.append(message)
            warnings.warn(message, UserWarning, skip_file_prefixes=(_PKG_PREFIX,))

        increments = self._compute_increments(
            scope="per_channel",
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=True,
            num_samples=num_samples,
            random_state=random_state,
            counterfactual_spend_factor=counterfactual_spend_factor,
            central_tendency=central_tendency,
            keep_date_axis=True,
        )

        matrix = self._carryover_band_to_matrix(increments, frequency=frequency)

        if not bool(matrix.coords["observed"].all()):
            notes.append(
                "Some periods' carryover runs past the end of the fitted data; "
                "those entries are NaN with observed=False, and the row sums "
                "count only the part the data covers."
            )

        dim_order = ["chain", "draw", "spend_date", "realization_date", "channel"]
        if frequency == "all_time":
            dim_order.remove("spend_date")

        matrix = matrix.transpose(*dim_order, *self.model.dims)

        matrix.attrs = self._carryover_attrs(
            increments,
            estimand=estimand,
            method=method,
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            notes=notes,
        )

        return matrix

    def split_incremental_contribution_current_future(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        horizon: int | None = None,
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
        counterfactual_spend_factor: float = 0.0,
        central_tendency: CentralTendency = "median",
    ) -> xr.Dataset:
        """Split each period's incremental contribution into current and future.

        *Current* is what a period's spend produced on the dates of that same
        reporting period, *future* is the carryover that lands afterwards.  The
        two add up to :meth:`compute_incremental_contribution` wherever the
        carryover window is fully observed.  The numbers are those of
        :meth:`split_incremental_contribution_over_time` reduced over
        ``realization_date``, but they are reduced one period's band at a time,
        so the dense matrix is never built.

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Reporting period.  At an aggregated frequency, *current* is the
            whole period, so spend late in a month that lands early the next
            month is future value.  This is a period-level cohort; computing at
            ``"original"`` and aggregating afterwards answers a different
            question.
        start_date, end_date : str or pd.Timestamp, optional
            Range of spend periods.  Defaults to the fitted data.
        horizon : int, optional
            Count lags ``0 .. horizon`` after each spend date as current, for a
            fixed near-term window instead of the same-period default.  ``0``
            equals the default.  A positive ``horizon`` needs
            ``frequency="original"``: at an aggregated frequency, "the period
            plus ``horizon`` data periods" would give spend early in a month a
            longer current window than spend late in it, which is neither
            definition of current.  Must lie in ``0 .. effective_horizon``.
        num_samples : int or None, optional
            Number of posterior samples to use; all of them when None.
        random_state : RandomState or Generator or None, optional
            Seed for the subsample.
        counterfactual_spend_factor : float, default=0.0
            As in :meth:`compute_incremental_contribution`.
        central_tendency : {"median", "mean"}, default="median"
            As in :meth:`compute_incremental_contribution`.

        Returns
        -------
        xr.Dataset
            Variables ``current``, ``future`` and ``future_share`` (future over
            current plus future), each ``(chain, draw, spend_date, channel,
            *custom_dims)`` without ``spend_date`` for ``"all_time"``.  A boolean
            ``complete`` coordinate on ``spend_date`` marks periods whose
            carryover is fully observed.  Neither sum skips the unobserved
            carryover past the last fitted date: ``future`` is ``NaN`` whenever
            ``complete`` is ``False``, and so is ``current`` when some of the
            unobserved carryover falls inside its own window (a positive
            ``horizon`` near the end of the data, or a last period the data ends
            partway through).  ``future_share`` is ``NaN`` there too, and also
            wherever current plus future is zero, as for a channel with no spend
            in the period; ``complete`` tells the two apart.  Attributes are
            those of :meth:`split_incremental_contribution_over_time`, plus
            ``horizon``.

        Raises
        ------
        TypeError
            If ``horizon`` is not an integer.
        ValueError
            If ``adstock.mode`` is not ``ConvMode.After``: spend then moves dates
            before its period, which neither *current* nor *future* describes.
            If the reach probe forced full-axis evaluation: there is then no
            measured horizon for *future* to be a share of.  If ``horizon`` is
            negative, positive at an aggregated frequency, or past
            ``effective_horizon``.  All of these are raised before any period is
            evaluated.

        Notes
        -----
        The shares are shares of *modelled* value within the truncated kernel.
        Nothing past ``l_max`` is attributed, so with ``l_max=13`` on weekly data
        "future" means future within a quarter.

        At an aggregated frequency ``future_share`` is not a property of the
        channel.  It depends on when the spend fell within the period: the same
        channel and model give a month whose spend is front-loaded a smaller
        future share than one whose spend comes in the last week, exactly as the
        adstock weights predict.  To combine periods, divide the sums,
        ``future.sum() / (current + future).sum()``, rather than averaging the
        shares.

        ``current + future`` is each period's incremental contribution, so the
        caveat of :meth:`split_incremental_contribution_over_time` applies:
        summed across periods it falls short of ``channel_contribution`` when
        saturation follows adstock.

        With the default dates, ``frequency="all_time"`` always reports
        ``future`` as ``NaN``: the single period contains the last fitted date,
        whose carryover the data cannot show.  Pass an ``end_date`` at least
        ``effective_horizon`` periods before the last fitted date to get a
        complete all-time split.

        See Also
        --------
        split_incremental_contribution_over_time : The per-date increments this
            reduces.

        Examples
        --------
        .. code-block:: python

            split = mmm.incrementality.split_incremental_contribution_current_future(
                frequency="monthly", num_samples=500, random_state=0
            )
            # Share of each month's value that lands after the month, on the
            # months whose carryover the data fully shows:
            split["future_share"].where(split["complete"])
        """
        mode = self.model.adstock.mode
        if mode != ConvMode.After:
            raise ValueError(
                "split_incremental_contribution_current_future needs "
                f"adstock.mode=ConvMode.After, got {mode!s}.  Under a leading "
                "kernel spend moves dates before its own period, which is "
                "neither current nor future value.  Use "
                "split_incremental_contribution_over_time to inspect where the "
                "effect lands."
            )

        lags = self._validate_horizon(horizon, frequency)

        increments = self._compute_increments(
            scope="per_channel",
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=True,
            num_samples=num_samples,
            random_state=random_state,
            counterfactual_spend_factor=counterfactual_spend_factor,
            central_tendency=central_tendency,
            keep_date_axis=True,
            # Checked as soon as the probe has measured the reach, before the
            # periods themselves are evaluated.
            validate_reach=partial(self._check_reach_supports_split, lags=lags),
        )

        windows = increments.windows.windows
        current, future, complete = [], [], []
        for window, band in zip(windows, increments.periods, strict=True):
            period_current, period_future, period_complete = self._split_band(
                band,
                start=window.start,
                end=window.end,
                lags=lags,
                freq_offset=increments.freq_offset,
            )
            current.append(period_current)
            future.append(period_future)
            complete.append(period_complete)

        result = self._stack_current_future(
            current,
            future,
            complete,
            spend_dates=pd.DatetimeIndex([window.end for window in windows]),
            period_starts=pd.DatetimeIndex([window.start for window in windows]),
            frequency=frequency,
            custom_dims=list(self.model.dims),
        )

        notes: list[str] = []
        if not all(complete):
            notes.append(
                "Some periods' carryover runs past the end of the fitted data; "
                "future is NaN there, and so is current where that carryover "
                "falls inside the current window."
            )

        result.attrs = self._carryover_attrs(
            increments,
            estimand="counterfactual",
            method="pipeline",
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            notes=notes,
        )
        result.attrs["horizon"] = lags

        return result

    @staticmethod
    def _validate_horizon(horizon: int | None, frequency: Frequency) -> int:
        """Check a requested ``horizon`` and turn it into a number of lags.

        Parameters
        ----------
        horizon : int or None
            As passed to :meth:`split_incremental_contribution_current_future`.
        frequency : Frequency
            Reporting period of the split.

        Returns
        -------
        int
            Lags after each spend date that count as current, ``0`` for
            ``None``.

        Raises
        ------
        TypeError
            If ``horizon`` is not an integer; a ``bool`` is refused too.
        ValueError
            If ``horizon`` is negative, or positive at an aggregated frequency.
        """
        if horizon is None:
            return 0

        if isinstance(horizon, bool) or not isinstance(horizon, int | np.integer):
            raise TypeError(f"horizon must be an integer, got {horizon!r}")

        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")

        if horizon > 0 and frequency != "original":
            raise ValueError(
                "A positive horizon needs frequency='original', where it "
                "counts lags 0..horizon after each spend date; got "
                f"frequency={frequency!r}.  At an aggregated frequency, "
                "current is the spend's own reporting period."
            )

        return int(horizon)

    @staticmethod
    def _check_reach_supports_split(reach: SpendReach, *, lags: int) -> None:
        """Refuse a current/future split the measured reach cannot support.

        Parameters
        ----------
        reach : SpendReach
            What the probe measured about how far spend moves the evaluated
            nodes.
        lags : int
            Lags after each spend date that count as current.

        Raises
        ------
        ValueError
            If the evaluation needs the full date axis, which leaves no
            measured horizon for *future* to be a share of, or if ``lags`` is
            past the measured horizon.
        """
        if reach.requires_full_axis or reach.max_lag is None:
            raise ValueError(
                "The spend probe could not bound this model's reach, so "
                "periods would be evaluated on the full date axis.  A future "
                "share against a horizon the model was measured not to "
                "respect is not a number; use "
                "split_incremental_contribution_over_time and its observed "
                "coordinate instead."
            )

        if lags > reach.max_lag:
            raise ValueError(
                f"horizon must be in 0..{reach.max_lag} (the measured "
                f"carryover lags), got {lags}"
            )

    @staticmethod
    def _split_band(
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
        # Adding zero periods of an anchored offset rolls an off-anchor date
        # forward (a month end plus 0 weeks is the next Sunday), so a zero lag
        # leaves the period end where it is.
        current_end = end + lags * freq_offset if lags else end
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

    @staticmethod
    def _stack_current_future(
        current: Sequence[xr.DataArray],
        future: Sequence[xr.DataArray],
        complete: Sequence[bool],
        *,
        spend_dates: pd.DatetimeIndex,
        period_starts: pd.DatetimeIndex,
        frequency: Frequency,
        custom_dims: Sequence[str],
    ) -> xr.Dataset:
        """Stack each period's current and future value into one dataset.

        Parameters
        ----------
        current, future : sequence of xr.DataArray
            One per period, in period order, each with dims
            ``(chain, draw, channel, *custom_dims)`` in any order.
        complete : sequence of bool
            Per period, whether its carryover is fully observed.
        spend_dates, period_starts : pd.DatetimeIndex
            Per period, its end, which labels ``spend_date``, and its start.
        frequency : Frequency
            Aggregation of the spend periods.  ``"all_time"`` keeps
            ``spend_date`` as a scalar coordinate instead of a dimension.
        custom_dims : sequence of str
            The model's own dimensions, placed last.

        Returns
        -------
        xr.Dataset
            ``current``, ``future`` and ``future_share``, the future value over
            current plus future, ``NaN`` where that total is zero.  Coordinates
            ``period_start`` and ``complete`` sit on ``spend_date``.
        """
        index = pd.Index(spend_dates, name="spend_date")
        split = {
            "current": xr.concat(current, dim=index),
            "future": xr.concat(future, dim=index),
        }
        total = split["current"] + split["future"]
        split["future_share"] = (split["future"] / total).where(total != 0)

        result = xr.Dataset(split).assign_coords(
            period_start=("spend_date", period_starts),
            complete=("spend_date", list(complete)),
        )

        dim_order = ["chain", "draw", "spend_date", "channel", *custom_dims]
        if frequency == "all_time":
            result = result.squeeze("spend_date", drop=False)
            dim_order.remove("spend_date")

        return result.transpose(*dim_order)

    def _carryover_attrs(
        self,
        increments: PeriodIncrements,
        *,
        estimand: str,
        method: str,
        frequency: Frequency,
        counterfactual_spend_factor: float,
        notes: list[str],
    ) -> dict:
        """Describe a carryover result in attributes ``to_netcdf`` can write.

        Parameters
        ----------
        increments : PeriodIncrements
            The evaluation the result was reduced from.
        estimand, method : str
            As passed to :meth:`split_incremental_contribution_over_time`.
        frequency : Frequency
            Aggregation of the spend periods.
        counterfactual_spend_factor : float
            Multiplicative factor applied to spend in the counterfactual.
        notes : list of str
            Caller-specific notes on how to read the result; the ones every
            carryover result shares are appended here.

        Returns
        -------
        dict
            ``estimand``, ``method``, ``effective_horizon`` (omitted under
            full-axis evaluation), and ``assumptions`` and ``warnings`` as JSON
            strings.  A list attribute would not survive ``to_netcdf``: the
            scipy engine, the one a plain install has, refuses it, and h5netcdf
            reads an empty list back as a float array and a single note as a
            bare string.
        """
        reach = increments.reach
        notes = list(notes)
        if reach.requires_full_axis:
            note = (
                "The spend probe could not bound the perturbation's reach, so each "
                "period was evaluated on the full date axis.  There is no "
                "measured carryover horizon: observed=False marks only the "
                "adstock kernel's own trailing lags past the fitted axis, a lower "
                "bound on what the data cannot show."
            )
            if self.model.adstock.mode == ConvMode.After:
                # Under a leading kernel the dates before a period are kept, and
                # the mode warning already says so.
                note += (
                    "  Dates before a period are excluded by convention rather "
                    "than measured to be zero."
                )
            notes.append(note)
        if self.model.time_varying_media:
            notes.append(
                "time_varying_media scales the contribution on each realization "
                "date by that date's latent multiplier, held at its posterior "
                "values; the split reflects those draws, not the adstock alone."
            )
        assumptions = {
            "adstock_first": bool(self.model.adstock_first),
            "normalize": bool(getattr(self.model.adstock, "normalize", False)),
            "mode": str(self.model.adstock.mode),
            "l_max": int(self.model.adstock.l_max),
            "link": str(self.model.link),
            "time_varying_media": bool(self.model.time_varying_media),
            "frequency": frequency,
            "counterfactual_spend_factor": float(counterfactual_spend_factor),
            "evaluation": "full_axis" if reach.requires_full_axis else "window",
        }
        attrs: dict = {"estimand": estimand, "method": method}
        if reach.max_lag is not None:
            attrs["effective_horizon"] = int(reach.max_lag)
        attrs["assumptions"] = json.dumps(assumptions)
        attrs["warnings"] = json.dumps(notes)
        return attrs

    def _compute_increments(
        self,
        *,
        scope: Estimand,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
        include_carryover: bool,
        num_samples: int | None,
        random_state: RandomState | Generator | None,
        counterfactual_spend_factor: float,
        central_tendency: CentralTendency,
        keep_date_axis: bool = False,
        validate_reach: Callable[[SpendReach], None] | None = None,
    ) -> PeriodIncrements:
        """Shared machinery behind the per-channel and joint increments.

        Resolves which nodes the counterfactual reaches, compiles one batched
        evaluator for them, builds the scenarios, evaluates, and reduces each
        period.  The callers differ in ``scope`` and in how they lay the
        periods out.

        Parameters
        ----------
        scope : {"per_channel", "joint"}
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
        keep_date_axis : bool, default=False
            Return each period's increment per realization date instead of
            summed over its evaluation window; see
            :meth:`_compute_period_increments`.
        validate_reach : callable, optional
            Called with the measured :class:`~pymc_marketing.mmm.spend_reach.SpendReach`
            before any period is evaluated, so that a request the reach rules
            out fails before the expensive part rather than after it.

        Returns
        -------
        PeriodIncrements
            Each period's increment in the original scale of the target, with
            the windows, measured reach and date frequency they came from.

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
        effects = resolve_channel_dependent_effects(self.model)

        # Create period groups based on frequency
        dates = self.data.dates
        periods = self._create_period_groups(start_date_ts, end_date_ts, frequency)

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
        # A model carrying mu_effects also evaluates the linear predictor, whose
        # only use is the completeness check below -- it costs a few additions
        # on top of a subgraph already being computed, and it is the difference
        # between assuming the increment is complete and knowing it.  A model
        # without mu_effects does not need it: an MMM assembles its predictor as
        # intercept plus channel_contribution plus controls, seasonality and the
        # effects, so with no effects there is no route by which spend could
        # reach mu other than the one already being evaluated.
        posterior_predictive_model = self.model.model
        effect_names = tuple(effect.contribution_var for effect in effects)
        predictor = linear_predictor(self.model) if self.model.mu_effects else None
        if self.model.mu_effects and predictor is None:
            # The one case the check exists for is a model with mu_effects, so
            # losing it here is worth a word: an effect that reports the wrong
            # contribution variable, or a model-level node reading spend outside
            # any effect, would drop a real part of the increment and the result
            # would report the remainder as if it were the whole.
            warnings.warn(
                "This model's linear predictor could not be recovered, either "
                "because it is frozen at its posterior values or because it is "
                "not exposed by the model's graph.  The increment is still "
                "computed, but the completeness of the increment -- that the "
                "evaluated nodes account for the whole move in the linear "
                "predictor -- goes unverified.",
                UserWarning,
                skip_file_prefixes=(_PKG_PREFIX,),
            )
        # The evaluator applies the spend intervention through pm.do, which
        # clones the model; response variables handed over as raw nodes, like
        # the predictor above, are re-resolved by name against that clone.
        # expected_aux_values lets the evaluator catch a post-fit mutation of
        # its auxiliary inputs -- MMM.sample_posterior_predictive(...,
        # clone_model=False) or a direct pm.set_data() call -- rather than
        # silently evaluating on whatever the live model happens to hold.
        # constant_data is missing only for idata built before this group was
        # recorded, and the evaluator falls back to the live snapshot then.
        evaluator = CounterfactualEvaluator(
            pymc_model=posterior_predictive_model,
            posterior=posterior_sub,
            response_vars=[
                CHANNEL_CONTRIBUTION,
                *effect_names,
                *([predictor] if predictor is not None else []),
            ],
            frozen_deterministics=self.model.frozen_deterministics,
            dates=dates,
            expected_aux_values=(
                self.idata.constant_data
                if hasattr(self.idata, "constant_data")
                else None
            ),
        )

        # channel_data's declared axis order, which the baseline array below and
        # the per-channel scenarios further down are both read against.  Taken
        # from the model rather than assumed: it is not always (date, channel),
        # since panel models lay channel_data out as (date, *custom_dims,
        # channel).
        channel_data_dims = list(
            posterior_predictive_model.named_vars_to_dims.get(
                CounterfactualEvaluator.CHANNEL_DATA, ()
            )
        )

        # Evaluate baseline on full dataset (once).  Comparable to the windowed
        # counterfactuals because a window is clamped to the fitted dates and
        # carries l_max of history, so every evaluated date sees the same inputs
        # it would on the full axis.  Transposed into the declared order rather
        # than trusting constant_data to already hold it, so date-first and the
        # position of ``channel`` are both facts about the model here, not
        # assumptions about the idata group.
        channel_data = self.data.get_channel_data()
        if channel_data_dims:
            channel_data = channel_data.transpose(*channel_data_dims)
        baseline_array = channel_data.values
        baseline = evaluator.evaluate_baseline(baseline_array)
        # Per node: (n_samples, n_dates, *non_date_dims)

        # What the window is allowed to assume, measured on the compiled graph
        # rather than declared.  Unconditional: a mediated path that outlives the
        # direct one is the obvious reason a window has to be widened, but it is
        # not the only one -- a custom adstock or saturation that reduces over
        # ``date`` is not a causal filter, and a plain MMM carrying one cannot be
        # windowed at all.  Costs one more call to an already-compiled function.
        probe = SpendProbe(
            evaluator=evaluator,
            baseline=baseline,
            baseline_array=baseline_array,
            counterfactual_spend_factor=counterfactual_spend_factor,
        )
        probe.assert_increment_is_complete(
            effects=effects, non_date_dims=evaluator.non_date_dims
        )
        # The adstock kernel's own trailing lags: the floor under the measured
        # horizon, and the lower bound on the unobserved tail when the probe
        # sends every period to the full date axis.
        trailing_lags = kernel_trailing_lags(
            self.model.adstock.l_max, self.model.adstock.mode
        )
        reach = probe.measure(
            effects=effects,
            l_max=self.model.adstock.l_max,
            trailing_lags=trailing_lags,
        )
        if validate_reach is not None:
            validate_reach(reach)
        l_max = reach.effective_l_max

        # The stretch of dates each period is evaluated over, which of them enter
        # the sum, and the length they all stack to.
        windows = EvaluationWindows.build(
            periods=periods,
            dates=dates,
            l_max=l_max,
            freq_offset=freq_offset,
            full_axis=reach.requires_full_axis,
            include_carryover=include_carryover,
            mode=self.model.adstock.mode,
        )

        # Two ways a per-channel column can stop being channel m's unilateral
        # counterfactual.  An effect can mix channels before reaching the
        # response, so no per-channel column survives at all.  Failing that, the
        # media transform itself can mix them -- ``forward_pass`` hands the
        # saturation the whole (date, channel) tensor, so a shared denominator is
        # expressible -- which is measured rather than assumed away, and only
        # where a per-channel column is going to be read: the joint estimand sums
        # over channels either way, and with effects the per-channel scenarios
        # are already being built.
        needs_dedicated_rows = bool(effects) or (
            scope == "per_channel"
            and probe.mixes_channels(non_date_dims=evaluator.non_date_dims)
        )
        # Where a channel sits among channel_data's non-date axes.  Needed only
        # in per-channel mode, and it is not always axis 0.
        channel_axis = None
        if needs_dedicated_rows:
            channel_axis = [d for d in channel_data_dims if d != "date"].index(
                "channel"
            )

        scenarios = windows.build_scenarios(
            baseline_array=baseline_array,
            counterfactual_spend_factor=counterfactual_spend_factor,
            dtype=evaluator.channel_dtype,
            channel_axis=channel_axis,
            n_channels=len(self.model.channel_columns),
            estimand=scope,
        )

        counterfactual = evaluator.evaluate_counterfactual(scenarios, windows=windows)
        # Per node: (n_scenarios, n_samples, max_window, *non_date_dims)

        # Assemble results
        period_results = self._compute_period_increments(
            windows=windows,
            scenarios=scenarios,
            baseline=baseline,
            counterfactual=counterfactual,
            non_date_dims=evaluator.non_date_dims,
            effect_names=effect_names,
            counterfactual_spend_factor=counterfactual_spend_factor,
            n_chains=n_chains,
            n_draws=n_draws,
            reducer=reducer,
            scope=scope,
            dedicated_channel_rows=channel_axis is not None,
            freq_offset=freq_offset,
            # How far past a period's last spend date its carryover is known to
            # land: the measured reach, or under full-axis evaluation the
            # kernel's own trailing lags, which bound it from below.
            tail_lags=reach.max_lag if reach.max_lag is not None else trailing_lags,
            keep_date_axis=keep_date_axis,
        )
        return PeriodIncrements(
            periods=period_results,
            windows=windows,
            reach=reach,
            freq_offset=freq_offset,
        )

    @staticmethod
    def _stack_periods(
        increments: PeriodIncrements, frequency: Frequency
    ) -> xr.DataArray:
        """Stack per-period totals into one array, one ``date`` per period.

        Parameters
        ----------
        increments : PeriodIncrements
            Per-period totals, as returned by :meth:`_compute_increments`
            without ``keep_date_axis``.
        frequency : Frequency
            Time aggregation frequency; ``"all_time"`` drops ``date``.

        Returns
        -------
        xr.DataArray
            Dims ``(chain, draw, date, *out_dims)``, without ``date`` for
            ``"all_time"``.
        """
        if frequency == "all_time":
            # Single period, no date dimension
            result = increments.periods[0].squeeze("date", drop=True)
        else:
            result = xr.concat(increments.periods, dim="date")
        # Already on the original (response) scale: the reducer applied the
        # link's inverse transform and target_scale per draw, before summing.
        return result.transpose("chain", "draw", ...)

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
    def _delta_mu(
        row: int,
        cf_mask: np.ndarray,
        window_dates: pd.DatetimeIndex,
        baseline: dict[str, np.ndarray],
        counterfactual: dict[str, np.ndarray],
        non_date_dims: dict[str, tuple[str, ...]],
        effect_names: Sequence[str],
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
            Boolean mask over the padded window selecting the evaluation dates,
            from :meth:`~pymc_marketing.mmm.counterfactual.PeriodWindow.eval_mask`.
        window_dates : pd.DatetimeIndex
            The same dates as labels, from
            :attr:`~pymc_marketing.mmm.counterfactual.PeriodWindow.eval_dates`, so
            the reducer can align a baseline response by label.  Coming from one
            place is what makes them the same dates: derived separately, their
            agreement would rest on two expressions being kept in step.
        baseline : dict
            Per response variable, unperturbed predictions already restricted to
            the evaluation dates, shape ``(n_samples, n_eval_dates, *non_date_dims)``.
        counterfactual : dict
            Per response variable, predictions per scenario, shape
            ``(n_scenarios, n_samples, max_window, *non_date_dims)``.
        non_date_dims : dict
            Per response variable, its dimensions with ``date`` removed.
        effect_names : sequence of str
            The included effects' contribution variables.  Passed in rather than
            read off the other arguments' keys, which also carry nodes evaluated
            for other reasons -- the linear predictor is evaluated to check the
            increment is complete and must not be added into it.
        channel : int or None
            Column of ``channel_contribution`` to read, or ``None`` to sum the
            channel dimension away.  A column is read only when *row* is a
            **shared** all-channels perturbation whose other columns answer for
            other channels; that is the separable case, where the untouched
            columns of a channel's own counterfactual are exactly unmoved and
            summing would return the joint delta instead.  When *row* is a
            perturbation of one channel alone -- the mediated case, and the
            measured-mixing one -- the whole delta belongs to that channel,
            including the movement it caused in the other columns, so the sum is
            what the unilateral estimand asks for.

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
        for name in effect_names:
            # xarray aligns by name, so an effect carrying only a subset of
            # the model's dimensions broadcasts without any reshaping here.
            delta_mu = delta_mu + delta(name)
        return delta_mu

    def _compute_period_increments(
        self,
        windows: EvaluationWindows,
        scenarios: CounterfactualScenarios,
        baseline: dict[str, np.ndarray],
        counterfactual: dict[str, np.ndarray],
        non_date_dims: dict[str, tuple[str, ...]],
        effect_names: Sequence[str],
        counterfactual_spend_factor: float,
        n_chains: int,
        n_draws: int,
        reducer: IncrementalReducer,
        scope: Estimand,
        dedicated_channel_rows: bool,
        freq_offset: BaseOffset,
        tail_lags: int,
        keep_date_axis: bool = False,
    ) -> list[xr.DataArray]:
        """Compute each period's incremental result.

        For each period, forms the per-date change in the linear predictor over
        the evaluation window, hands it to *reducer* to be converted into a
        response-scale increment, applies the sign convention, and reshapes the
        flattened sample dimension back to ``(chain, draw)``.  Laying the
        periods out together is left to the caller.

        Parameters
        ----------
        windows : EvaluationWindows
            The per-period windows the scenarios were built for.  They own the
            evaluation dates, so this method neither knows nor recomputes the
            carry-out length.
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
        effect_names : sequence of str
            Contribution variables of the included effects, which is a subset of
            the evaluated nodes.
        counterfactual_spend_factor : float
            Multiplicative factor used for sign convention.
        n_chains : int
            Number of MCMC chains in the posterior.
        n_draws : int
            Number of draws per chain.
        reducer : IncrementalReducer
            Link-specific reduction from a linear-predictor perturbation to a
            response-scale increment.  It owns the rescaling to original
            units, so no further ``target_scale`` multiplication happens here.
        scope : {"per_channel", "joint"}
            ``"per_channel"`` perturbs one channel at a time and keeps a
            ``channel`` dimension; ``"joint"`` perturbs all channels together and
            returns a single number per period.
        dedicated_channel_rows : bool
            Whether *scenarios* holds a row per (period, channel), each
            perturbing that channel alone.  This is what decides how a
            per-channel row is read: a dedicated row's whole
            ``channel_contribution`` delta belongs to the perturbed channel,
            while a shared all-channels row has to be read column by column,
            since summing it would hand every channel the joint delta.
        freq_offset : BaseOffset
            The data's date frequency, to label the carryover dates that fall
            past the fitted axis when *keep_date_axis* is set.
        tail_lags : int
            Lags after a period's last fitted spend date that its carryover is
            known to reach: the measured
            :attr:`~pymc_marketing.mmm.spend_reach.SpendReach.max_lag`, or under
            full-axis evaluation the kernel's own trailing lags.  Required, so
            that no caller gets a band without its unobserved tail by leaving it
            out; it goes unused when *keep_date_axis* is not set.
        keep_date_axis : bool, default=False
            Keep the per-date increment instead of summing it over the window,
            and extend it past the fitted axis as far as the carryover reaches;
            see :meth:`_append_unobserved_tail`.

        Returns
        -------
        list of xr.DataArray
            One per period, in period order, in original scale.  Dims
            ``(date, chain, draw, *out_dims)`` with ``date`` of length one, the
            period end; ``out_dims`` is ``(channel, *custom_dims)`` for
            ``"per_channel"`` and ``custom_dims`` for ``"joint"``.  With
            *keep_date_axis*, ``(chain, draw, realization_date, *out_dims)``
            over the period's evaluated dates and its unobserved tail, with a
            boolean ``observed`` coordinate on ``realization_date``.
        """
        fit_data = self.idata.fit_data
        dates = self.data.dates
        channels = list(self.model.channel_columns)
        custom_dims = list(self.model.dims)
        out_dims = ["channel", *custom_dims] if scope == "per_channel" else custom_dims
        results = []

        for period_idx, window in enumerate(windows.windows):
            # The same evaluation dates on both sides of the difference, in the
            # same order: ``in_eval`` picks them out of the full-axis baseline and
            # ``eval_mask`` picks them out of the padded counterfactual window.
            # Both come from the window itself, so there is no second expression
            # to keep in step with the first.
            period_baseline = {
                name: values[:, window.in_eval] for name, values in baseline.items()
            }

            # (scenario key, channel column to read) per output slice.  The joint
            # estimand reads a single scenario and sums the channel dimension
            # away; the per-channel one reads its own scenario per channel, and
            # sums that too whenever the scenario perturbed that channel alone --
            # see ``_delta_mu``'s ``channel`` for why the two cases differ.
            slices: list[tuple[int | None, int | None]] = (
                [
                    (idx, None if dedicated_channel_rows else idx)
                    for idx in range(len(channels))
                ]
                if scope == "per_channel"
                else [(None, None)]
            )
            deltas = [
                self._delta_mu(
                    row=scenarios.rows[(period_idx, key)],
                    cf_mask=window.eval_mask(windows.max_window),
                    window_dates=window.eval_dates,
                    baseline=period_baseline,
                    counterfactual=counterfactual,
                    non_date_dims=non_date_dims,
                    effect_names=effect_names,
                    channel=channel,
                )
                for key, channel in slices
            ]
            delta_mu = (
                xr.concat(deltas, dim="channel").assign_coords(channel=channels)
                if scope == "per_channel"
                else deltas[0]
            )

            # Link-specific reduction to a response-scale difference, then the
            # sign convention:
            # factor > 1 → Y(perturbed) - Y(actual)    (marginal)
            # factor < 1 → Y(actual) - Y(counterfactual) (total)
            # With keep_date_axis the per-date term is kept and ``date`` rides
            # along by name, so the reshape below threads it like any other dim.
            if keep_date_axis:
                increment = reducer.per_date_increment(delta_mu).transpose(
                    "sample", "date", *out_dims
                )
            else:
                increment = reducer.counterfactual_minus_baseline(delta_mu).transpose(
                    "sample", *out_dims
                )
            if counterfactual_spend_factor <= 1.0:
                increment = -increment
            # Shape: (n_samples, [date,] *out_dims), n_samples = n_chains * n_draws

            # Reshape flattened sample → (chain, draw) to preserve MCMC structure
            reshaped = increment.values.reshape(n_chains, n_draws, *increment.shape[1:])

            coords: dict = {
                "chain": np.arange(n_chains),
                "draw": np.arange(n_draws),
                **{dim: fit_data.coords[dim].values for dim in custom_dims},
            }
            if scope == "per_channel":
                coords["channel"] = channels

            if keep_date_axis:
                band = xr.DataArray(
                    reshaped,
                    dims=("chain", "draw", "realization_date", *out_dims),
                    coords={**coords, "realization_date": window.eval_dates},
                )
                period_result = self._append_unobserved_tail(
                    band,
                    window=window,
                    dates=dates,
                    tail_lags=tail_lags,
                    freq_offset=freq_offset,
                )
            else:
                total = xr.DataArray(
                    reshaped, dims=("chain", "draw", *out_dims), coords=coords
                )
                period_result = total.assign_coords(date=window.end).expand_dims("date")
            results.append(period_result)

        return results

    @staticmethod
    def _append_unobserved_tail(
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
        windowed, and :meth:`split_incremental_contribution_current_future`
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
        spend = dates[(dates >= window.start) & (dates <= window.end)]
        tail = pd.DatetimeIndex([])
        if len(spend):
            reach_end = spend[-1] + tail_lags * freq_offset
            tail = pd.date_range(dates[-1], reach_end, freq=freq_offset)
            tail = tail[tail > dates[-1]]
        realization = band.indexes["realization_date"].append(tail)
        return band.reindex(realization_date=realization).assign_coords(
            observed=("realization_date", ~realization.isin(tail))
        )

    @staticmethod
    def _carryover_band_to_matrix(
        increments: PeriodIncrements, frequency: Frequency
    ) -> xr.DataArray:
        """Lay per-period increment bands out as a spend x realization matrix.

        Each period contributes one row, labelled ``spend_date`` by the period's
        end (the same label :meth:`compute_incremental_contribution` stamps on
        its ``date``).  The columns are the union of every row's realization
        dates.  Three kinds of entry result:

        - on the row's band and on the fitted axis: the model's per-date
          increment;
        - on the row's band but past the fitted axis: ``NaN``, with
          ``observed=False`` (see :meth:`_append_unobserved_tail`);
        - off the row's band: ``0``.  With a windowed evaluation the probe
          (:class:`~pymc_marketing.mmm.spend_reach.SpendProbe`) measured the
          window to contain every date the perturbation moves, so these entries
          are zero by construction.  Under full-axis evaluation the band
          follows ``adstock.mode``.  With ``ConvMode.After`` it starts at the
          period itself and runs to the axis end, and the dates before the
          period are excluded by convention, not measured: a node that reduces
          over ``date`` can move them, and keeping them would make rows
          overlap.  A leading kernel keeps the dates before the period, where
          it places real mass: the band starts at the axis start and ends at
          the axis end under ``ConvMode.Overlap``, at the period's end under
          ``ConvMode.Before``.

        A sum over ``realization_date`` with xarray's default ``skipna`` therefore
        reproduces :meth:`compute_incremental_contribution` exactly.

        Parameters
        ----------
        increments : PeriodIncrements
            Per-period bands, from :meth:`_compute_increments` with
            ``keep_date_axis``.
        frequency : Frequency
            Aggregation frequency; ``"all_time"`` drops ``spend_date``.

        Returns
        -------
        xr.DataArray
            Dims ``(chain, draw, spend_date, realization_date, *out_dims)``,
            with a ``period_start`` coordinate on ``spend_date`` and a boolean
            ``observed`` coordinate on ``(spend_date, realization_date)``.

        Raises
        ------
        ValueError
            If a band's dates are missing from the realization axis, which the
            union makes impossible.
        """
        bands = increments.periods
        windows = increments.windows.windows
        realization = bands[0].indexes["realization_date"]
        for band in bands[1:]:
            realization = realization.union(band.indexes["realization_date"])

        template = bands[0]
        out_dims = template.dims[3:]
        n_rows, n_cols = len(bands), len(realization)
        values = np.zeros(
            (
                template.sizes["chain"],
                template.sizes["draw"],
                n_rows,
                n_cols,
                *(template.sizes[dim] for dim in out_dims),
            ),
            dtype=template.dtype,
        )
        observed = np.ones((n_rows, n_cols), dtype=bool)
        for row, band in enumerate(bands):
            cols = realization.get_indexer(band.indexes["realization_date"])
            if (cols < 0).any():  # pragma: no cover - the union contains them
                raise ValueError("A band's dates are missing from the matrix axis.")
            values[:, :, row, cols] = band.values
            observed[row, cols] = band.coords["observed"].values

        coords = {
            name: coord
            for name, coord in template.coords.items()
            if name in ("chain", "draw", *out_dims)
        }
        matrix = xr.DataArray(
            values,
            dims=("chain", "draw", "spend_date", "realization_date", *out_dims),
            coords={
                **coords,
                "spend_date": [window.end for window in windows],
                "realization_date": realization,
            },
        ).assign_coords(
            period_start=("spend_date", [window.start for window in windows]),
            observed=(("spend_date", "realization_date"), observed),
        )
        if frequency == "all_time":
            # Keep the period's bounds as scalar coordinates, so a reduction can
            # still tell current from future dates.
            matrix = matrix.squeeze("spend_date", drop=False)
        return matrix

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

        Raises
        ------
        ValueError
            If ``frequency`` is invalid, the requested dates fall outside the
            fitted data range, or ``central_tendency`` is not ``"median"`` or
            ``"mean"``; or, at compute time, if a ``mu_effect``'s declared
            reach is narrower than what was measured, or a post-fit mutation
            of an auxiliary input (e.g.
            ``MMM.sample_posterior_predictive(..., clone_model=False)``) is
            detected; or if the model produces non-finite predictions.  See
            :mod:`~pymc_marketing.mmm.spend_reach` for the full story on each.
        NotImplementedError
            If the model's link function has no :class:`IncrementalReducer`,
            if a channel-dependent ``mu_effect`` has not opted in via
            :meth:`~pymc_marketing.mmm.additive_effect.MuEffect.incrementality_spec`,
            or if the accounted nodes do not reproduce the full move in the
            linear predictor.

        Warns
        -----
        UserWarning
            If a spend counterfactual's reach could not be measured (no
            interior date could be probed), the evaluation falls back to the
            full date axis, which is correct but slower than a window, and
            the completeness check above is skipped for lack of anything to
            compare.  See
            :meth:`~pymc_marketing.mmm.spend_reach.SpendProbe.measure`.

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
        Acquisition Cost) when the target is customer count.

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

        Raises
        ------
        ValueError, NotImplementedError
            Delegates to :meth:`contribution_over_spend`; see there for the
            full list.

        Warns
        -----
        UserWarning
            Delegates to :meth:`contribution_over_spend`; see there for the
            full-axis fallback warning.

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
            If ``spend_increase_pct <= 0``; or, at compute time, the same
            declaration-reconciliation and mutation-guard ``ValueError``
            cases documented on :meth:`compute_incremental_contribution`.
        NotImplementedError
            If the model's link function has no :class:`IncrementalReducer`,
            a channel-dependent ``mu_effect`` has not opted in, or the
            accounted nodes do not reproduce the full move in the linear
            predictor; see :meth:`compute_incremental_contribution`.

        Warns
        -----
        UserWarning
            If a spend counterfactual's reach could not be measured, the
            evaluation falls back to the full date axis and the completeness
            check is skipped; see
            :meth:`~pymc_marketing.mmm.spend_reach.SpendProbe.measure`.

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
