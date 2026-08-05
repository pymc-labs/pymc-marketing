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

from abc import ABC, abstractmethod
from collections.abc import Sequence
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
    CHANNEL_CONTRIBUTION,
    SpendProbe,
    linear_predictor,
    resolve_channel_dependent_effects,
)

if TYPE_CHECKING:
    from numpy.random import Generator, RandomState

    from pymc_marketing.mmm.mmm import MMM

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
        )

        # Evaluate baseline on full dataset (once).  Comparable to the windowed
        # counterfactuals because a window is clamped to the fitted dates and
        # carries l_max of history, so every evaluated date sees the same inputs
        # it would on the full axis.
        baseline_array = self.data.get_channel_data().values
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
        reach = probe.measure(effects=effects, l_max=self.model.adstock.l_max)
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

        scenarios = windows.build_scenarios(
            baseline_array=baseline_array,
            counterfactual_spend_factor=counterfactual_spend_factor,
            dtype=evaluator.channel_dtype,
            channel_axis=channel_axis,
            n_channels=len(self.model.channel_columns),
            estimand=estimand,
        )

        counterfactual = evaluator.evaluate_counterfactual(scenarios, windows=windows)
        # Per node: (n_scenarios, n_samples, max_window, *non_date_dims)

        # Assemble results
        return self._compute_period_increments(
            windows=windows,
            scenarios=scenarios,
            baseline=baseline,
            counterfactual=counterfactual,
            non_date_dims=evaluator.non_date_dims,
            effect_names=effect_names,
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

        for period_idx, window in enumerate(windows.windows):
            # The same evaluation dates on both sides of the difference, in the
            # same order: ``in_eval`` picks them out of the full-axis baseline and
            # ``eval_mask`` picks them out of the padded counterfactual window.
            # Both come from the window itself, so there is no second expression
            # to keep in step with the first.
            period_baseline = {
                name: values[:, window.in_eval] for name, values in baseline.items()
            }

            # (scenario key, channel to read) per output slice.  The joint
            # estimand reads a single scenario and sums the channel dimension
            # away; the per-channel one reads its own scenario per channel.
            slices: list[tuple[int | None, int | None]] = (
                [(idx, idx) for idx in range(len(channels))]
                if estimand == "per_channel"
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
                .assign_coords(date=window.end)
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
