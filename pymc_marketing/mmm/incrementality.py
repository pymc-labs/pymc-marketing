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
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import ConfigDict, validate_call

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
from pymc_marketing.data.idata.schema import Frequency
from pymc_marketing.data.idata.utils import subsample_draws
from pymc_marketing.mmm.components.adstock import (
    BinomialAdstock,
    DelayedAdstock,
    GeometricAdstock,
    NoAdstock,
    WeibullCDFAdstock,
    WeibullPDFAdstock,
)
from pymc_marketing.mmm.counterfactual import (
    CounterfactualEvaluator,
    CounterfactualScenarios,
    Estimand,
    EvaluationWindows,
)
from pymc_marketing.mmm.evaluator_cache import (
    EvaluatorCacheKey,
    EvaluatorLease,
    constant_data_fingerprint,
)
from pymc_marketing.mmm.link import MU_ORIGINAL_SCALE, LinkFunction
from pymc_marketing.mmm.spend_reach import (
    _PKG_PREFIX,
    CHANNEL_CONTRIBUTION,
    SpendProbe,
    linear_predictor,
    resolve_channel_dependent_effects,
)
from pymc_marketing.mmm.transformers import ConvMode
from pymc_marketing.mmm.utils import infer_frequency
from pymc_marketing.pytensor_utils import (
    needed_posterior_names,
    reorder_to_model_coords,
    transpose_to_model_dims,
)

if TYPE_CHECKING:
    from numpy.random import Generator, RandomState
    from pandas.tseries.offsets import BaseOffset
    from pytensor.graph.basic import Variable

    from pymc_marketing.mmm.components.adstock import AdstockTransformation
    from pymc_marketing.mmm.components.saturation import SaturationTransformation
    from pymc_marketing.mmm.counterfactual import PeriodWindow
    from pymc_marketing.mmm.mmm import MMM
    from pymc_marketing.mmm.spend_reach import ChannelDependentEffect, SpendReach

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

LAG_DIM = "time since exposure"

ALLOCATION_REQUIREMENT = "estimand='allocation'"
CLOSED_FORM_REQUIREMENT = "method='closed_form'"

BUILTIN_ADSTOCKS = (
    BinomialAdstock,
    DelayedAdstock,
    GeometricAdstock,
    NoAdstock,
    WeibullCDFAdstock,
    WeibullPDFAdstock,
)

INCOMPLETE_SPLIT_NOTE = (
    "Some periods' carryover runs past the end of the fitted data; future is "
    "NaN there, and so is current where that carryover falls inside the "
    "current window."
)

CUSTOM_ADSTOCK_NOTE = (
    "This model's adstock is not one of the built-in ones.  The allocation reads "
    "it as a linear convolution, and it passed a homogeneity check (doubling "
    "the spend doubles the kernel), which does not prove linearity.  If it is "
    "not a linear convolution, the allocation is wrong even if the columns add "
    "up."
)


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
        ``mu_original_scale`` deterministic (the LogNormal median
        ``exp(mu) * target_scale``) -- with dimensions
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


def _refuse_unbounded_reach(
    reach: SpendReach, *, kernel_lags: int, requirement: str
) -> None:
    """Refuse a reach that runs further than the adstock kernel.

    Passed as ``validate_reach`` with *kernel_lags* and *requirement* bound.
    The allocation and the closed form both read each date's response as the
    sum of one cohort's kernel weights, so the probe has to find that a change
    in spend moves the response no further than the kernel does.

    Parameters
    ----------
    reach : SpendReach
        What the probe measured.
    kernel_lags : int
        The last lag the adstock's own kernel reaches, from
        :func:`kernel_trailing_lags`.
    requirement : str
        What reads the kernel, for the message: :data:`ALLOCATION_REQUIREMENT`
        or :data:`CLOSED_FORM_REQUIREMENT`.

    Raises
    ------
    NotImplementedError
        If the reach requires the full date axis, or runs past the kernel.
    """
    if reach.requires_full_axis:
        raise NotImplementedError(
            f"{requirement} reads the adstock as a fixed causal kernel, "
            "and the spend probe could not verify that the effect of a change "
            "in spend dies out within the fitted dates.  A transform that "
            "reduces over date, carryover that outlasts the fitted dates, or a "
            "channel that spends only where the probe cannot perturb it (on the "
            "first or last date) each leave that unverified."
        )

    if reach.max_lag is not None and reach.max_lag > kernel_lags:
        raise NotImplementedError(
            f"{requirement} reads each date's response as the adstock "
            f"kernel's, which reaches {kernel_lags} periods after the spend, "
            f"but the spend probe found that spend moves the response up to "
            f"{reach.max_lag} periods later.  Something else in the media "
            "transform, such as a saturation that carries spend over a window, "
            "spreads it across dates, which the kernel does not describe."
        )


def _is_builtin_adstock(adstock: object) -> bool:
    """Whether *adstock* is exactly one of the library's own adstocks.

    The type is compared, not its ancestry: a subclass of a built-in adstock can
    override ``function`` with anything, so it is not known to be linear.  The
    classes are listed in :data:`BUILTIN_ADSTOCKS`, not told apart by where
    they are defined, so moving one between modules cannot demote it.

    Parameters
    ----------
    adstock : object
        The model's adstock transformation.

    Returns
    -------
    bool
        ``True`` if its class is one of :data:`BUILTIN_ADSTOCKS`.
    """
    return type(adstock) in BUILTIN_ADSTOCKS


def _require_homogeneous_adstock(
    adstock: AdstockTransformation, weights: xr.DataArray, posterior: xr.Dataset
) -> None:
    """Refuse a custom adstock whose response to a unit impulse does not scale.

    The allocation reads the response to a unit impulse as a convolution kernel.
    That holds only if the adstock is linear, and a nonlinear one gives an
    allocation that is wrong, with the columns still reconciling under
    ``adstock_first=True``.  Doubling the impulse
    must double the response; this catches an adstock with a power or a
    saturation inside, but a pass is necessary, not sufficient, for linearity.

    Parameters
    ----------
    adstock : AdstockTransformation
        The model's adstock, not a built-in one.
    weights : xr.DataArray
        Its response to a unit impulse at *posterior*.
    posterior : xr.Dataset
        The draws *weights* were sampled at.

    Raises
    ------
    NotImplementedError
        If the response to a double impulse is not twice *weights*.
    """
    with _quiet_posterior_predictive_log():
        doubled = adstock.sample_curve(
            parameters=posterior, amount=2.0, progressbar=False
        )
    if not np.allclose(doubled.values, 2 * weights.values, rtol=1e-5, atol=1e-12):
        raise NotImplementedError(
            "estimand='allocation' needs the adstock to be a linear convolution "
            f"(spend x kernel), but {type(adstock).__name__} does not scale: "
            "doubling the spend does not double its response, so the "
            "allocation would be wrong."
        )
    warnings.warn(CUSTOM_ADSTOCK_NOTE, UserWarning, skip_file_prefixes=(_PKG_PREFIX,))


def _refuse_dedicated_rows(needs_dedicated_rows: bool, *, requirement: str) -> None:
    """Refuse a result whose per-channel columns are not each channel's own.

    Passed as ``validate_dedicated_rows`` with *requirement* bound.  A
    channel-dependent ``mu_effect`` is refused before this, from the model
    alone, so what is left is what the probe measured about the media
    transform.

    Parameters
    ----------
    needs_dedicated_rows : bool
        Whether the channels have to be perturbed one at a time.
    requirement : str
        What needs the columns, for the message: :data:`ALLOCATION_REQUIREMENT`
        or :data:`CLOSED_FORM_REQUIREMENT`.

    Raises
    ------
    NotImplementedError
        If they do.
    """
    if needs_dedicated_rows:
        raise NotImplementedError(
            f"{requirement} needs each channel's column to depend on "
            "that channel's spend alone, and the spend probe could not "
            "establish that: either the media transform mixes channels, or some "
            "channel spends only where the probe cannot perturb it."
        )


class _DropSamplingLine(logging.Filter):
    """Drop pymc's ``Sampling: [...]`` record and keep every other one."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("Sampling:")


@contextmanager
def _quiet_posterior_predictive_log() -> Iterator[None]:
    """Hold back pymc's ``Sampling: [...]`` line around an internal call.

    :meth:`~pymc_marketing.mmm.components.base.Transformation.sample_curve`
    goes through ``pm.sample_posterior_predictive``, which logs the variables it
    samples at ``INFO``.  Inside a method that never asked to sample, the line
    reads as if it had.  Only that record is dropped, and the logger's level is
    left alone, so any other message still gets through.

    Yields
    ------
    None
    """
    logger = logging.getLogger("pymc.sampling.forward")
    dropper = _DropSamplingLine()
    logger.addFilter(dropper)
    try:
        yield
    finally:
        logger.removeFilter(dropper)


def _transformation_parameters(
    posterior: xr.Dataset,
    transformation: AdstockTransformation | SaturationTransformation,
) -> xr.Dataset:
    """Keep a posterior's draws, and only the variables a transformation reads.

    ``sample_curve`` and a graph evaluation turn every variable they are given
    into per-draw dictionaries, and a kernel is ``l_max`` long, so handing them
    ``channel_contribution``, ``mu`` and the rest is memory spent for nothing.
    The variables of the transformation's own priors are kept, the
    hyperparameters of a hierarchical one among them.  A transformation with
    fixed parameters has no variable in the posterior, so it keeps the smallest
    one there is, which is where its draws come from.

    Parameters
    ----------
    posterior : xr.Dataset
        The draws the transformation is evaluated at.
    transformation : AdstockTransformation or SaturationTransformation
        The transformation whose variables to keep.

    Returns
    -------
    xr.Dataset
        *posterior* without the variables the transformation does not read.
    """
    roots = tuple(transformation.variable_mapping.values())
    # A prior with parameters that are priors names them ``<name>_<parameter>``.
    needed = {
        name
        for name in posterior.data_vars
        if any(name == root or str(name).startswith(f"{root}_") for root in roots)
    }
    if not needed:
        # pymc takes the number of draws from a variable, so a transformation
        # with fixed parameters keeps the smallest one the posterior has.
        needed = {min(posterior.data_vars, key=lambda name: posterior[name].size)}
    return posterior.drop_vars(
        [name for name in posterior.data_vars if name not in needed]
    )


def _missing_variables(
    transformation: AdstockTransformation | SaturationTransformation,
    posterior: xr.Dataset,
) -> list[str]:
    """List the random parameters of a transformation that the posterior lacks.

    Fixed parameters never appear in a posterior and are skipped.

    Parameters
    ----------
    transformation : AdstockTransformation or SaturationTransformation
        The transformation to check.
    posterior : xr.Dataset
        The posterior it is evaluated at.

    Returns
    -------
    list of str
        The names of the variables that are missing.
    """
    return [
        name
        for parameter, name in transformation.variable_mapping.items()
        if hasattr(transformation.function_priors[parameter], "create_variable")
        and name not in posterior
    ]


def _relabel_draws(sampled: xr.DataArray, labels: Mapping[str, Any]) -> xr.DataArray:
    """Label a posterior-predictive sample with the draws it was computed at.

    ``pm.sample_posterior_predictive`` returns the draws in the order of the
    posterior it is given, under that posterior's ``chain`` and ``draw``
    labels.  Those are not the pipeline's: the pipeline labels its results
    with ``np.arange``, while :func:`~pymc_marketing.data.idata.utils.subsample_draws`
    drops the coordinates of a subsample and a full posterior keeps whatever
    labels it was given.  Relabeling by position is right only while the order
    holds: nothing between the subsample and the sampling may reorder or drop
    draws.  The sizes are checked here: a draw lost or added on the way would
    pair draw *i* of one array with draw *j* of another, so it raises.  The
    parity tests pin the order itself.

    Parameters
    ----------
    sampled : xr.DataArray
        The sample, with ``chain`` and ``draw`` dimensions.
    labels : mapping
        The ``chain`` and ``draw`` coordinates to give it.

    Returns
    -------
    xr.DataArray
        *sampled* under *labels*.

    Raises
    ------
    ValueError
        If the sample does not have as many chains and draws as *labels*.
    """
    for dim in ("chain", "draw"):
        if sampled.sizes[dim] != len(labels[dim]):
            raise ValueError(
                f"The sample has {sampled.sizes[dim]} {dim}s where the posterior "
                f"it was computed at has {len(labels[dim])}."
            )
    return sampled.assign_coords(chain=labels["chain"], draw=labels["draw"])


def _adstock_terms(
    weights: xr.DataArray, spend: xr.DataArray
) -> Iterator[xr.DataArray]:
    """Yield each lag's term of the adstocked spend, lag by lag.

    Term ``l`` is :math:`w_l x_{t-l}` on every date :math:`t`, with spend before
    the first date taken as zero, as
    :func:`~pymc_marketing.mmm.transformers.batched_convolution` pads it under
    ``ConvMode.After``.  Their sum is the adstocked spend.

    Parameters
    ----------
    weights : xr.DataArray
        The adstock kernel, with a ``"time since exposure"`` lag dimension.
    spend : xr.DataArray
        The channel spend, with a ``date`` dimension.

    Yields
    ------
    xr.DataArray
        One lag's term, ``weights``'s dims less the lag, broadcast with
        ``spend``'s.
    """
    for lag in range(weights.sizes[LAG_DIM]):
        yield weights.isel({LAG_DIM: lag}, drop=True) * spend.shift(
            date=lag, fill_value=0.0
        )


def _row_of_dates(
    dates: pd.DatetimeIndex, periods: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> np.ndarray:
    """Map each date to the spend period it falls in.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The fitted date axis.
    periods : list of tuple
        ``(start, end)`` of each spend period, in row order.

    Returns
    -------
    np.ndarray
        Per date, the index of its period, or ``-1`` for a date in none of them.
    """
    row_of = np.full(len(dates), -1)
    for row, (start, end) in enumerate(periods):
        row_of[(dates >= start) & (dates <= end)] = row
    return row_of


def _scatter_by_cohort(
    values: np.ndarray,
    landed: np.ndarray,
    *,
    row_of: np.ndarray,
    lag: int,
    columns: np.ndarray | None = None,
) -> None:
    """Add one lag's landed contribution to its spend periods' rows, in place.

    What lands on date :math:`t` at lag ``lag`` was spent on :math:`t - lag`, so
    it goes to that date's row, in column :math:`t`.  Spend dates outside every
    period are dropped.

    Parameters
    ----------
    values : np.ndarray
        The matrix, ``(chain, draw, row, realization_date, *rest)``.
    landed : np.ndarray
        What this lag puts on each date, ``(chain, draw, date, *rest)``, on the
        fitted dates.
    row_of : np.ndarray
        Per date, its row, or ``-1``; see :func:`_row_of_dates`.
    lag : int
        The lag *landed* was taken at.
    columns : np.ndarray, optional
        Per fitted date, its column in *values*, or ``-1`` for a date that is not
        a column.  By default the columns are the fitted dates themselves.
    """
    spent = np.arange(len(row_of) - lag)
    rows = row_of[spent]
    in_rows = rows >= 0
    realized = spent[in_rows] + lag
    rows = rows[in_rows]
    if columns is not None:
        placed = columns[realized] >= 0
        rows, realized, target = (
            rows[placed],
            realized[placed],
            columns[realized][placed],
        )
    else:
        target = realized
    values[:, :, rows, target] += landed[:, :, realized]


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


@dataclass(frozen=True)
class ProbedReach:
    """Leased evaluator, factual baseline and measured reach of one evaluation.

    Parameters
    ----------
    lease : EvaluatorLease
        The compiled batched evaluator, held for the duration of the request.
    baseline : dict
        The factual evaluation of every node, per node
        ``(n_samples, n_dates, *non_date_dims)``.
    baseline_array : np.ndarray
        The factual spend the baseline was evaluated on.
    probe : SpendProbe
        The probe, for what is measured on demand (channel mixing).
    reach : SpendReach
        How far in time a change in spend moves the evaluated nodes.
    effect_names : tuple of str
        The contribution variables of the channel-dependent effects evaluated.
    trailing_lags : int
        The adstock kernel's own trailing lags.
    channel_data_dims : list of str
        ``channel_data``'s declared axis order.
    """

    lease: EvaluatorLease
    baseline: dict
    baseline_array: np.ndarray
    probe: SpendProbe
    reach: SpendReach
    effect_names: tuple[str, ...]
    trailing_lags: int
    channel_data_dims: list[str]

    @property
    def evaluator(self) -> CounterfactualEvaluator:
        """The compiled batched evaluator this request holds."""
        return self.lease.evaluator

    def release(self) -> None:
        """Return the evaluator to the cache, also when the request raised."""
        self.lease.release()


@dataclass(frozen=True)
class PreparedEvaluation:
    """Validated request, posterior, reducer, probe and windows of one evaluation.

    Parameters
    ----------
    start_ts, end_ts : pd.Timestamp
        The validated range of spend periods.
    periods : list of tuple
        ``(start, end)`` of each spend period.
    posterior : xr.Dataset
        The subsampled posterior everything is evaluated at.
    reducer : IncrementalReducer
        Maps a linear-predictor change to a response-scale increment.
    effects : tuple of ChannelDependentEffect
        The channel-dependent effects evaluated with ``channel_contribution``.
    probed : ProbedReach
        The leased evaluator, the baseline and the measured reach.
    freq_offset : BaseOffset
        The data's date frequency.
    windows : EvaluationWindows
        Each period's evaluation window.
    """

    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    periods: list[tuple[pd.Timestamp, pd.Timestamp]]
    posterior: xr.Dataset
    reducer: IncrementalReducer
    effects: tuple[ChannelDependentEffect, ...]
    probed: ProbedReach
    freq_offset: BaseOffset
    windows: EvaluationWindows

    def release(self) -> None:
        """Return the leased evaluator to the model's cache."""
        self.probed.release()


@dataclass(frozen=True)
class ClosedFormInputs:
    r"""A prepared evaluation plus the saturated spend the closed form reads.

    Parameters
    ----------
    prepared : PreparedEvaluation
        The validated request, posterior, probe and windows.
    saturated : xr.DataArray
        :math:`\beta g(x_t) - \beta g(f x_t)` in original scale, per date and
        draw, signed like the pipeline's increments.
    labels : dict
        ``chain`` and ``draw`` coordinates of the pipeline's results.
    """

    prepared: PreparedEvaluation
    saturated: xr.DataArray
    labels: dict


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

    Notes
    -----
    Computation reuses compiled evaluators cached on the fitted model, so repeated calls do not recompile.  A call
    leases the evaluator it uses and returns it when it finishes, so concurrent calls on one model each hold their own
    evaluator -- a call that finds the cache empty compiles one -- rather than sharing the draws it is rebound to.
    An in-place edit of ``idata.constant_data`` while an instance is held is not detected by that instance; a fresh
    ``mmm.incrementality`` sees it.

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
        if data is None:
            data = MMMIDataWrapper.from_mmm(model, idata)
        self.data = data
        self.idata = data.idata

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
            ``mu_original_scale`` with dimensions
            ``("sample", "date", *custom_dims)``.

        Raises
        ------
        ValueError
            If the deterministic is absent from the posterior.
        """
        if MU_ORIGINAL_SCALE not in posterior:
            raise ValueError(
                f"Incrementality under link='{self.model.link}' needs the baseline "
                f"response-scale prediction '{MU_ORIGINAL_SCALE}', which is not in "
                "the posterior. It is registered automatically when a log-link "
                "model is built, so this posterior was most likely sampled with a "
                "restricted 'var_names'. Refit without filtering it out."
            )

        return self._stack_samples(posterior[MU_ORIGINAL_SCALE])

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
            which corrects a level rather than a difference.  A wrapped
            likelihood (``Censored``, ``Scaled``) is refused under ``"mean"``
            on either link and by both methods, because the wrapper moves the
            observed mean off any ``mu``-based quantity; use ``"median"``.

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

            **End of the data**: with ``include_carryover=True`` and
            ``ConvMode.After``, a period's increment is summed only up to the
            last fitted date, which leaves out the carryover the data does not
            show: that of the last periods, and with the default ``end_date``
            that of the ``"all_time"`` period.
            Their numbers are then lower bounds on the periods' effects, not
            totals, provided more spend never lowers the response on a later
            date (as with the library's adstocks and saturations and
            non-negative coefficients, under either link) and the periods were
            evaluated on a window, the usual case (see ``assumptions`` in
            :meth:`split_incremental_contribution_over_time`).  The shortfall
            grows with the spend late in the data;
            :meth:`split_incremental_contribution_current_future` flags those
            periods with ``complete=False``.

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

        holds to floating-point tolerance for ``estimand="counterfactual"``, for
        both links and either ``adstock_first``.  Rows are the
        carryover-inclusive value of a period's spend, up to the last fitted
        date (see Notes); reading a column instead (what landed on a date, from
        all earlier spend) is a different estimand when the response mixes
        cohorts, which is what ``estimand="allocation"`` provides.  Its rows are
        Aumann-Shapley values, so with ``adstock_first=True`` and a concave
        saturation they do not equal
        :meth:`compute_incremental_contribution`.

        In particular, the entries do not add up to ``channel_contribution``
        when saturation follows adstock (``adstock_first=True``, the default).
        Each row zeroes one period's spend against all the others, so with a
        concave response the period increments summed over ``spend_date``
        come in below the channel's total contribution, by how much depends on
        how saturated the channel is.  Use these numbers to value a period's
        spend, not to decompose the reported total; ``estimand="allocation"``
        is the one that reconciles with it.  On an identity-link model
        without mediated effects, the all-time counterfactual, which zeroes
        every period at once, matches ``channel_contribution``; the
        period increments match it only with ``adstock_first=False``, where the
        response is separable by cohort.  Under the log link
        ``channel_contribution`` lives in the linear predictor and the
        increments on the response scale, so neither matches it.

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
            row by row.  ``"allocation"`` splits each date's contribution among
            the spend periods that produced it, so that it reconciles with
            ``channel_contribution`` column by column; see the Notes.  It needs
            an identity link, a forward-looking adstock (``ConvMode.After``) and
            ``counterfactual_spend_factor=0``, and it covers the fitted dates
            only.
            It also needs a linear convolution for an adstock: the built-in ones
            are, a custom one is checked for homogeneity, warned about if it
            passes and refused if it fails.  With ``adstock_first=True`` it
            needs non-negative channel data and kernel weights too.
        method : {"pipeline", "closed_form"}, default="pipeline"
            ``"pipeline"`` evaluates the model once per spend period and is
            exact for every supported model.  ``"closed_form"`` computes the
            same matrix from the adstock kernel and the saturated spend, with
            no evaluation per period, for models where saturation runs first;
            see the Notes.  It raises on any other model:

            - ``adstock_first=True``, or a link other than identity;
            - a custom adstock, or one that is not ``ConvMode.After``;
            - a channel-dependent ``mu_effect``, or ``time_varying_media``;
            - a media transform the spend probe finds to carry spend over a
              window of dates or to mix channels;
            - an effect the spend probe cannot bound within the fitted dates,
              such as carryover longer than the data, which the pipeline still
              evaluates on the full date axis;
            - ``estimand="allocation"`` with a ``counterfactual_spend_factor``
              other than ``0``.
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
              ``effective_horizon``.  The allocation never has such entries:
              its ``realization_date`` is the fitted dates, and ``observed`` is
              ``True`` throughout.
            - ``complete`` on ``spend_date``, for ``estimand="allocation"``
              only: ``False`` for a period whose carryover runs past the last
              fitted date, so that its row is the part of its effect the data
              can show.  A scalar for ``"all_time"``.

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
              ``"window"`` or ``"full_axis"``.  For ``estimand="allocation"``
              also ``"allocation_rule"``: ``"aumann_shapley_share"`` with
              ``adstock_first=True``, ``"separable_cohorts"`` with
              ``adstock_first=False``, where the counterfactual matrix is the
              allocation.  A string, like every attribute here, so the result
              can be written with ``to_netcdf``.
            - ``warnings``: notes on how to read the result, as a JSON string
              holding a list (``json.loads`` it), for the same reason.

        Raises
        ------
        NotImplementedError
            For ``method="closed_form"`` on a model outside the class it is
            exact for (see ``method``).  For ``estimand="allocation"`` under a
            log link, an adstock that is not ``ConvMode.After`` or a
            channel-dependent ``mu_effect``, and when the spend probe cannot
            verify that the effect of a change in spend dies out within the
            fitted dates, or cannot establish that each channel's column
            depends on that channel's spend alone.  The probe falls short on a
            transform that reduces over ``date`` or mixes channels, but also on
            carryover that outlasts the fitted dates and on a channel that
            spends only on the first or last date, which a plain adstock can
            have.  Also for ``estimand="allocation"`` when the probe finds that
            spend moves the response further than the adstock's kernel reaches
            (a saturation that carries spend over a window), and, with
            ``adstock_first=True``, for negative channel data and for a negative
            kernel weight, and for a custom adstock that is not homogeneous
            (doubling the spend does not double the kernel).
        ValueError
            For an unknown ``estimand`` or ``method``, for
            ``estimand="allocation"`` with a ``counterfactual_spend_factor``
            other than ``0`` or, with ``adstock_first=True`` or a custom
            adstock, an adstock parameter that is missing from the posterior.
            The same for ``method="closed_form"``, which also needs the
            saturation's parameters in the posterior.  And for everything
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

        ``estimand="allocation"`` answers the column question: how much of the
        contribution on date :math:`t` does the spend of period :math:`s`
        account for?  With ``adstock_first=True`` one date's contribution is
        :math:`\beta\,g(z_t)` for the adstocked spend
        :math:`z_t = \sum_l w_l x_{t-l}`, and saturation mixes the cohorts into
        :math:`z_t`, so no cohort has a contribution of its own.  The
        Aumann-Shapley value, the continuous counterpart of the Shapley value
        (integrated gradients along the ray from zero spend to actual spend), is
        symmetric between cohorts and adds up.  It is one choice among several
        such splits -- the discrete Shapley value has the same properties and
        gives different numbers for a concave response -- and for this response
        it collapses to a share split,

        .. math::

            A[s, t] = \left(c_t - c_t^{0}\right)
            \frac{w_{t-s}\,x_s}{z_t},

        where :math:`c_t^{0}` is the contribution at zero spend.  Hence
        ``A.sum("spend_date")`` is ``channel_contribution`` (less its zero-spend
        value, which is nil for a saturation through the origin) on every date.
        With a linear response it equals the ``"counterfactual"`` matrix; with a
        concave one the counterfactual rows are smaller, which is the gap
        described above.  With ``adstock_first=False`` the response is already
        separable by cohort, the counterfactual matrix reconciles by column,
        and the allocation is that matrix; its columns too are
        ``channel_contribution`` less its zero-spend value.  The allocation
        covers the fitted
        dates only: attributing a date past the data would need spend that has
        not happened, so there is no ``observed=False`` tail.  Rows are the part
        of each period's effect the data can show, and the ``complete``
        coordinate on ``spend_date`` marks the ones that are the whole of it:
        ``False`` for the last periods, whose carryover runs past the last
        fitted date.  Reduce only ``complete`` rows to a current-vs-future
        reading.  The columns add up to the contribution only when every spend
        period is a row; a narrower ``start_date`` / ``end_date`` leaves out the
        other periods' shares.

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
        what :meth:`compute_incremental_contribution` reports.  For a row with
        ``observed=False`` cells, that sum is a lower bound on the period's
        effect, not its total, under the conditions given in
        :meth:`compute_incremental_contribution`: ``ConvMode.After``, a
        windowed evaluation (``assumptions["evaluation"] == "window"``), and a
        response that more spend never lowers on a later date.  Pass
        ``skipna=False`` to get ``NaN`` for those rows instead.

        At ``frequency="original"`` the matrix's size is
        ``n_samples x n_dates x (n_dates + effective_horizon) x n_channels``;
        use ``num_samples`` or an aggregated frequency on long series.
        :meth:`split_incremental_contribution_current_future` works on each
        period's band instead and never builds this matrix.

        ``method="closed_form"`` applies when saturation runs before the
        adstock.  The model is then additively separable across spend dates:
        spend on date :math:`s` puts :math:`w_l V_s` on date :math:`s + l`,
        with :math:`w_l` the adstock's kernel and
        :math:`V_s = \beta g(x_s) - \beta g(f x_s)` the change in the saturated
        spend when the spend is scaled by the counterfactual factor :math:`f`.
        Both are read off the posterior, the second in one evaluation of the
        saturation for all dates, and the matrix is the pipeline's entry for
        entry.  The counterfactual and the allocation coincide on such a
        model, so either estimand can use it.  Its cost does not grow with the
        number of periods; the spend probe, which both methods run, does not
        depend on them either, so at coarse frequencies the two cost about the
        same.

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
            # Reconciles with today's per-period incrementality, and like it is
            # a lower bound for the last periods (see Notes):
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

        # One route per combination, each refusing what it does not cover before
        # anything is evaluated.  The closed form takes both estimands, which
        # coincide on the models it accepts.
        if method == "closed_form":
            return self._split_closed_form(
                frequency=frequency,
                start_date=start_date,
                end_date=end_date,
                estimand=estimand,
                method=method,
                num_samples=num_samples,
                random_state=random_state,
                counterfactual_spend_factor=counterfactual_spend_factor,
                central_tendency=central_tendency,
            )
        elif estimand == "allocation":
            return self._split_allocation(
                frequency=frequency,
                start_date=start_date,
                end_date=end_date,
                method=method,
                num_samples=num_samples,
                random_state=random_state,
                counterfactual_spend_factor=counterfactual_spend_factor,
                central_tendency=central_tendency,
            )
        elif estimand == "counterfactual":
            return self._split_counterfactual(
                frequency=frequency,
                start_date=start_date,
                end_date=end_date,
                method=method,
                num_samples=num_samples,
                random_state=random_state,
                counterfactual_spend_factor=counterfactual_spend_factor,
                central_tendency=central_tendency,
            )
        else:  # pragma: no cover - estimand was validated above
            raise ValueError(f"An unknown estimand {estimand!r} reached the dispatch.")

    def _split_counterfactual(
        self,
        *,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
        method: str,
        num_samples: int | None,
        random_state: RandomState | Generator | None,
        counterfactual_spend_factor: float,
        central_tendency: CentralTendency,
    ) -> xr.DataArray:
        """Build the ``estimand="counterfactual"`` matrix with ``method="pipeline"``.

        The body of :meth:`split_incremental_contribution_over_time` for that
        combination; see there for the definition and the layout.  Each period
        is evaluated with its spend scaled, so none of the structural guards of
        the other routes apply: an adstock that is not ``ConvMode.After`` is
        only warned about.

        Parameters
        ----------
        frequency, start_date, end_date, num_samples, random_state
            As in :meth:`split_incremental_contribution_over_time`.
        counterfactual_spend_factor, central_tendency
            As in :meth:`split_incremental_contribution_over_time`.
        method : str
            Echoed into the result's attributes.

        Returns
        -------
        xr.DataArray
            As :meth:`split_incremental_contribution_over_time`.

        Raises
        ------
        ValueError
            If an input is out of range, or the date frequency cannot be inferred.
        NotImplementedError
            If the link has no reducer, or a channel-dependent ``mu_effect`` has
            not opted in.

        Warns
        -----
        UserWarning
            If ``adstock.mode`` is not ``ConvMode.After``.
        """
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

        return self._finish_counterfactual_matrix(
            matrix,
            increments.reach,
            notes=notes,
            estimand="counterfactual",
            method=method,
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
        )

    def _finish_counterfactual_matrix(
        self,
        matrix: xr.DataArray,
        reach: SpendReach,
        *,
        notes: list[str],
        estimand: str,
        method: str,
        frequency: Frequency,
        counterfactual_spend_factor: float,
    ) -> xr.DataArray:
        """Order the dims of a counterfactual matrix and describe it.

        The pipeline and the closed form both call this once their matrix is
        laid out, so the two return the same dimension order, notes and
        attributes.

        Parameters
        ----------
        matrix : xr.DataArray
            As :meth:`_carryover_band_to_matrix` returns it.
        reach : SpendReach
            What the spend probe measured for the evaluation behind *matrix*.
        notes : list of str
            Notes already collected on how to read the result.
        estimand, method : str
            As passed to :meth:`split_incremental_contribution_over_time`.
        frequency : Frequency
            Aggregation of the spend periods.
        counterfactual_spend_factor : float
            Multiplicative factor applied to spend in the counterfactual.

        Returns
        -------
        xr.DataArray
            *matrix*, transposed and with its attributes set.
        """
        mode = self.model.adstock.mode
        if not bool(matrix.coords["observed"].all()):
            note = (
                "Some periods' carryover runs past the end of the fitted data; "
                "those entries are NaN with observed=False, and the row sums "
                "count only the part the data covers"
            )
            # Only a forward-only kernel on a measured window keeps the observed
            # cells independent of what comes after the data, so that the
            # skipped tail is all a row sum misses.
            if mode == ConvMode.After and not reach.requires_full_axis:
                note += (
                    ", a lower bound on those periods' effect whenever more "
                    "spend never lowers the response on a later date"
                )
            notes.append(note + ".")

        dim_order = ["chain", "draw", "spend_date", "realization_date", "channel"]
        if frequency == "all_time":
            dim_order.remove("spend_date")

        matrix = matrix.transpose(*dim_order, *self.model.dims)

        matrix.attrs = self._carryover_attrs(
            reach,
            estimand=estimand,
            method=method,
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            notes=notes,
        )

        return matrix

    def _split_allocation(
        self,
        *,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
        method: str,
        num_samples: int | None,
        random_state: RandomState | Generator | None,
        counterfactual_spend_factor: float,
        central_tendency: CentralTendency,
    ) -> xr.DataArray:
        """Build the ``estimand="allocation"`` matrix.

        The body of :meth:`split_incremental_contribution_over_time` for the
        allocation estimand; see there for the definition and the layout.  What
        the model's structure rules out is refused before anything is
        evaluated; what only the spend probe can establish (an effect that dies out within the fitted dates,
        channels that do not mix) is refused after the probe and before any
        period is evaluated.

        Parameters
        ----------
        frequency, start_date, end_date, num_samples, random_state, central_tendency
            As in :meth:`split_incremental_contribution_over_time`.
        method : str
            Echoed into the result's attributes.
        counterfactual_spend_factor : float
            Must be ``0``: the allocation is measured against zero spend.

        Returns
        -------
        xr.DataArray
            As :meth:`split_incremental_contribution_over_time`, with
            ``realization_date`` restricted to the fitted dates and a ``complete``
            coordinate on ``spend_date``.

        Raises
        ------
        ValueError
            If ``counterfactual_spend_factor`` is not ``0``, or a random adstock
            parameter is missing from the posterior.
        NotImplementedError
            Under a log link, a non-``After`` adstock or a channel-dependent
            ``mu_effect``; for a custom adstock that is not linear; for negative
            channel data or kernel weights with ``adstock_first=True``; when the
            probe could
            not verify that the effect of a change in spend dies out within the
            fitted dates, or finds that it outlasts the kernel; and when it
            could not establish that each channel's column depends on that
            channel's spend alone.
        """
        self._refuse_unsupported_allocation(counterfactual_spend_factor)

        start_ts, end_ts = self._validate_input(start_date, end_date)
        periods = self._create_period_groups(start_ts, end_ts, frequency)

        # One subset for both computations below: drawing twice from a
        # Generator would give each a different one.
        posterior_sub = self._posterior_in_data_order(
            subsample_draws(
                self.idata.posterior.dataset,
                num_samples=num_samples,
                random_state=random_state,
            )
        )

        # The shares need c - c0 on every date, so adstock-first evaluates the
        # whole axis whatever rows were asked for.
        whole_axis = self.model.adstock_first
        # What the kernel alone can refuse is refused before the evaluation.
        weights = None
        if whole_axis:
            weights = self._adstock_kernel(posterior_sub, divides=True)
        elif not _is_builtin_adstock(self.model.adstock):
            # Nothing divides by the adstocked spend here, but the cohorts only
            # separate if the adstock is a linear convolution.
            self._adstock_kernel(posterior_sub, divides=False)
        increments = self._allocation_increments(
            frequency="all_time" if whole_axis else frequency,
            start_date=None if whole_axis else start_date,
            end_date=None if whole_axis else end_date,
            posterior=posterior_sub,
            central_tendency=central_tendency,
        )
        max_lag = increments.reach.max_lag
        if max_lag is None:  # pragma: no cover - _refuse_unbounded_reach ran
            raise ValueError("An unbounded reach reached the allocation.")

        if weights is not None:
            rule = "aumann_shapley_share"
            matrix = self._allocate_by_shares(
                increments.periods[0],
                weights=weights,
                periods=periods,
                frequency=frequency,
            )
            complete = self._rows_complete(periods, self.data.dates, max_lag)
        else:
            rule = "separable_cohorts"
            # Saturation runs before the adstock, so each cohort's response is
            # its own and the counterfactual already adds up by column.
            matrix = self._carryover_band_to_matrix(increments, frequency=frequency)
            # A row is complete when no part of its carryover was left past the
            # fitted dates, which is what the band's unobserved tail records.
            complete = [
                bool(done)
                for done in np.atleast_1d(
                    matrix.coords["observed"].all("realization_date").values
                )
            ]
            matrix = self._on_fitted_dates(matrix)

        return self._finish_allocation_matrix(
            matrix,
            complete,
            increments.reach,
            rule=rule,
            method=method,
            frequency=frequency,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    def _finish_allocation_matrix(
        self,
        matrix: xr.DataArray,
        complete: list[bool],
        reach: SpendReach,
        *,
        rule: str,
        method: str,
        frequency: Frequency,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> xr.DataArray:
        """Mark each row's completeness, order the dims and describe the allocation.

        Called by the pipeline's and the closed form's allocation once the
        matrix holds the fitted dates only.

        Parameters
        ----------
        matrix : xr.DataArray
            The allocation on the fitted dates.
        complete : list of bool
            Per row, whether its carryover lands within the fitted dates.
        reach : SpendReach
            What the spend probe measured for the evaluation behind *matrix*.
        rule : str
            The allocation rule, echoed into ``assumptions``.
        method : str
            Echoed into the result's attributes.
        frequency : Frequency
            Aggregation of the spend periods.
        start_ts, end_ts : pd.Timestamp
            The validated range of spend periods.

        Returns
        -------
        xr.DataArray
            *matrix* with a ``complete`` coordinate, transposed, with attributes.
        """
        matrix = matrix.assign_coords(
            complete=complete[0]
            if frequency == "all_time"
            else ("spend_date", complete)
        )

        notes = [
            "The allocation covers the fitted dates only: attributing a date "
            "past the data would need spend that has not happened, so there is "
            "no observed=False tail and each row is the part of the period's "
            "effect the data can show.",
        ]
        dates = self.data.dates
        if start_ts <= dates[0] and end_ts >= dates[-1]:
            notes.append(
                "Columns sum to channel_contribution less its value at zero "
                "spend, which is nil for a saturation through the origin."
            )
        else:
            notes.append(
                "start_date and end_date keep only some spend periods, so each "
                "column sums to the part of channel_contribution (less its "
                "value at zero spend) that those periods produced, not to all "
                "of it."
            )
        if not _is_builtin_adstock(self.model.adstock):
            notes.append(CUSTOM_ADSTOCK_NOTE)
        if not all(complete):
            notes.append(
                "Rows with complete=False are periods whose carryover runs past "
                "the end of the fitted data.  Their entries are exact, but the "
                "part landing after the last fitted date is not in the matrix, "
                "so reducing them to current vs future value understates future "
                "value."
            )

        dim_order = ["chain", "draw", "spend_date", "realization_date", "channel"]

        if frequency == "all_time":
            dim_order.remove("spend_date")

        matrix = matrix.transpose(*dim_order, *self.model.dims)

        matrix.attrs = self._carryover_attrs(
            reach,
            estimand="allocation",
            method=method,
            frequency=frequency,
            counterfactual_spend_factor=0.0,
            notes=notes,
            extra_assumptions={"allocation_rule": rule},
        )
        return matrix

    def _data_freq_offset(self) -> BaseOffset:
        """Resolve the fitted data's date frequency.

        Returns
        -------
        BaseOffset
            The offset between consecutive fitted dates.

        Raises
        ------
        ValueError
            If the frequency cannot be inferred from the date index.
        """
        inferred_freq = infer_frequency(self.data.dates)
        if inferred_freq is None:
            raise ValueError(
                "Could not infer frequency from the date index. "
                "Ensure the fitted data has a regular date frequency."
            )
        return pd.tseries.frequencies.to_offset(inferred_freq)

    def _closed_form_inputs(
        self,
        *,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
        num_samples: int | None,
        random_state: RandomState | Generator | None,
        central_tendency: CentralTendency,
        counterfactual_spend_factor: float,
        lags: int | None = None,
    ) -> ClosedFormInputs:
        r"""Probe the model and evaluate the saturation stage, for the closed form.

        Runs :meth:`_prepare_evaluation`, refuses a media transform the spend
        probe finds to mix channels, and evaluates the change in saturated
        spend :math:`V_t` once for all dates.

        Parameters
        ----------
        frequency, start_date, end_date, num_samples, random_state, central_tendency
            As in :meth:`split_incremental_contribution_over_time`.
        counterfactual_spend_factor : float
            Multiplicative factor applied to spend in the counterfactual.
        lags : int, optional
            For a current/future split, the lags after each spend date that count
            as current, checked against the measured reach.

        Returns
        -------
        ClosedFormInputs
            The probe, the windows and the saturated spend.

        Raises
        ------
        ValueError
            If ``central_tendency`` is unknown, ``counterfactual_spend_factor`` is
            negative, the date range is out of range, or *lags* is past the
            measured reach.
        NotImplementedError
            When the spend probe cannot verify that the effect of a change in
            spend dies out within the fitted dates and no further than the
            kernel reaches, or cannot establish that each channel's column
            depends on that channel's spend alone.
        """
        adstock = self.model.adstock
        prepared = self._prepare_evaluation(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            num_samples=num_samples,
            random_state=random_state,
            central_tendency=central_tendency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            include_carryover=True,
            validate_reach=partial(
                self._validate_closed_form_reach,
                kernel_lags=kernel_trailing_lags(adstock.l_max, adstock.mode),
                lags=lags,
            ),
        )
        # The probe is the closed form's only use of the evaluator, so the lease
        # ends with it rather than with the matrix the caller builds afterwards.
        try:
            # No channel-dependent effects reach this point: the guards refused them.
            probed = prepared.probed
            _refuse_dedicated_rows(
                probed.probe.mixes_channels(
                    non_date_dims=probed.evaluator.non_date_dims
                ),
                requirement=CLOSED_FORM_REQUIREMENT,
            )

            # The identity link is one of the guards, so the reducer only rescales.
            reducer = cast(IdentityLinkReducer, prepared.reducer)
            posterior = prepared.posterior
            labels = {
                "chain": np.arange(posterior.sizes["chain"]),
                "draw": np.arange(posterior.sizes["draw"]),
            }
            saturated = _relabel_draws(
                self._saturated_media(posterior, counterfactual_spend_factor)
                * reducer.scale,
                labels,
            )
        finally:
            prepared.release()
        return ClosedFormInputs(prepared=prepared, saturated=saturated, labels=labels)

    @staticmethod
    def _validate_closed_form_reach(
        reach: SpendReach, *, kernel_lags: int, lags: int | None
    ) -> None:
        """Refuse a reach the closed form cannot be read from.

        Parameters
        ----------
        reach : SpendReach
            What the probe measured.
        kernel_lags : int
            The last lag the adstock's own kernel reaches.
        lags : int, optional
            For a current/future split, the lags that count as current.

        Raises
        ------
        NotImplementedError
            If the reach is unbounded or runs past the kernel.
        ValueError
            If *lags* is past the measured reach.
        """
        _refuse_unbounded_reach(
            reach, kernel_lags=kernel_lags, requirement=CLOSED_FORM_REQUIREMENT
        )
        if lags is not None:
            Incrementality._check_reach_supports_split(reach, lags=lags)

    def _split_closed_form(
        self,
        *,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
        estimand: str,
        method: str,
        num_samples: int | None,
        random_state: RandomState | Generator | None,
        counterfactual_spend_factor: float,
        central_tendency: CentralTendency,
    ) -> xr.DataArray:
        r"""Build the matrix of ``method="closed_form"`` without evaluating any period.

        With saturation first, a cohort's increment on lag :math:`l` is
        :math:`w_l V_s`: the adstock's kernel times
        :math:`V_s = \beta g(x_s) - \beta g(f x_s)`, the change in the
        saturated spend.  Both are read off the posterior, the second by
        evaluating the saturation stage alone, and no period is evaluated.  What
        the model's structure rules out is refused first; what only the spend
        probe can establish (a saturation that carries spend over a window or
        mixes channels) is refused after the probe, with the checks the
        allocation makes.  The layout is that of the pipeline route, entry for entry, and
        both estimands give the same numbers here: the model is additively
        separable across cohorts, where a counterfactual and an allocation
        coincide.

        Parameters
        ----------
        frequency, start_date, end_date, num_samples, random_state
            As in :meth:`split_incremental_contribution_over_time`.
        counterfactual_spend_factor, central_tendency
            As in :meth:`split_incremental_contribution_over_time`.
        estimand : {"counterfactual", "allocation"}
            Which layout to return.
        method : str
            Echoed into the result's attributes.

        Returns
        -------
        xr.DataArray
            As :meth:`split_incremental_contribution_over_time`.

        Raises
        ------
        ValueError
            For ``estimand="allocation"`` with a ``counterfactual_spend_factor``
            other than ``0``, for a random adstock or saturation parameter
            missing from the posterior, if ``central_tendency`` is unknown, or
            if the date range is out of range.
        NotImplementedError
            For a model outside the class the closed form is exact for (see
            :meth:`_refuse_unsupported_closed_form`); when the spend probe
            cannot verify that the effect of a change in spend dies out within
            the fitted dates and no further than the kernel reaches, or cannot
            establish that each channel's column depends on that channel's
            spend alone.
        """
        if estimand == "allocation":
            self._refuse_allocation_spend_factor(counterfactual_spend_factor)
        self._refuse_unsupported_closed_form()

        inputs = self._closed_form_inputs(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            num_samples=num_samples,
            random_state=random_state,
            central_tendency=central_tendency,
            counterfactual_spend_factor=counterfactual_spend_factor,
        )
        prepared = inputs.prepared
        probed, windows = prepared.probed, prepared.windows
        freq_offset = prepared.freq_offset
        weights = self._adstock_kernel(prepared.posterior, divides=False)
        matrix = self._closed_form_matrix(
            inputs.saturated,
            _relabel_draws(weights, inputs.labels),
            periods=prepared.periods,
            windows=windows,
            tail_lags=probed.reach.max_lag,
            freq_offset=freq_offset,
            frequency=frequency,
        )
        start_ts, end_ts = prepared.start_ts, prepared.end_ts

        if estimand == "counterfactual":
            return self._finish_counterfactual_matrix(
                matrix,
                probed.reach,
                notes=[],
                estimand=estimand,
                method=method,
                frequency=frequency,
                counterfactual_spend_factor=counterfactual_spend_factor,
            )

        complete = [
            bool(done)
            for done in np.atleast_1d(
                matrix.coords["observed"].all("realization_date").values
            )
        ]
        return self._finish_allocation_matrix(
            self._on_fitted_dates(matrix),
            complete,
            probed.reach,
            rule="separable_cohorts",
            method=method,
            frequency=frequency,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    def _saturated_media(
        self, posterior: xr.Dataset, counterfactual_spend_factor: float
    ) -> xr.DataArray:
        r"""Evaluate how much the saturation stage moves when spend is scaled.

        Per date and draw, :math:`\beta g(x_t) - \beta g(f x_t)` with
        :math:`f` the counterfactual spend factor, signed as the pipeline signs
        its increments: actual minus counterfactual for :math:`f \le 1`, the
        other way round above.  With :math:`f = 0` this is the saturated spend
        less its value at zero spend, so a saturation that does not pass
        through the origin is handled too.  When saturation runs before the
        adstock, ``channel_contribution`` is the adstock of these values.

        The spend goes in as the model sees it, divided by ``channel_scale``
        with a non-finite ratio taken as zero, mirroring the input scaling in
        :meth:`~pymc_marketing.mmm.mmm.MMM.build_model`.  Both spend levels are
        evaluated in one posterior-predictive run.

        Parameters
        ----------
        posterior : xr.Dataset
            The already subsampled posterior.
        counterfactual_spend_factor : float
            Multiplicative factor applied to spend in the counterfactual.

        Returns
        -------
        xr.DataArray
            The increment in the model's own units (the target scaled),
            ``(chain, draw, ...)`` with ``date`` among the other dimensions.
        """
        saturation = self.model.saturation
        # The model scales its input the same way when it builds the graph
        # (``channel_data / channel_scale`` with NaN and inf switched to 0, in
        # MMM.build_model).  That one is symbolic, so the two cannot share code;
        # keep them in step.
        spend = self.data.get_channel_data() / self.data.get_channel_scale()
        spend = spend.where(np.isfinite(spend), 0.0)
        parameters = _transformation_parameters(posterior, saturation)

        # Both spend levels go through one application of the saturation: a
        # second application would create its parameters a second time.
        coords = {
            dim: np.asarray(coord)
            for dim, coord in spend.coords.items()
            if dim in spend.dims
        }
        coords["spend_level"] = np.asarray(["actual", "counterfactual"])
        with _quiet_posterior_predictive_log():
            sampled = saturation._sample_response(
                "saturated_media",
                parameters,
                np.stack([spend.values, counterfactual_spend_factor * spend.values]),
                dims=("spend_level", *map(str, spend.dims)),
                coords=coords,
                core_dim="date",
                progressbar=False,
            )

        increment = sampled.sel(spend_level="actual", drop=True) - sampled.sel(
            spend_level="counterfactual", drop=True
        )
        return increment if counterfactual_spend_factor <= 1.0 else -increment

    def _closed_form_matrix(
        self,
        saturated: xr.DataArray,
        weights: xr.DataArray,
        *,
        periods: list[tuple[pd.Timestamp, pd.Timestamp]],
        windows: EvaluationWindows,
        tail_lags: int | None,
        freq_offset: BaseOffset,
        frequency: Frequency,
    ) -> xr.DataArray:
        r"""Lay out :math:`A[s, u] = w_{u-s} V_s` as the pipeline route would.

        One pass over the kernel's lags: what lag :math:`l` lands on each date
        is :math:`w_l V_{t-l}`, and it
        goes to the period its spend date belongs to, so an aggregated period's
        row is the sum of its dates' cohorts.  The columns are those of the
        pipeline route, the evaluated dates of every period plus the dates past
        the fitted axis that a period's carryover reaches, and these are ``NaN``
        with ``observed=False``: carryover the model has but the data ends
        before.

        Parameters
        ----------
        saturated : xr.DataArray
            :math:`V_t` in original scale, ``(chain, draw, ...)`` with ``date``.
        weights : xr.DataArray
            The adstock kernel, with a ``"time since exposure"`` dimension.
        periods : list of tuple
            ``(start, end)`` of each spend period, the rows.
        windows : EvaluationWindows
            The periods' evaluation windows, for the rows' labels and columns.
        tail_lags : int
            Lags past a period's last fitted spend date that its carryover
            reaches.
        freq_offset : BaseOffset
            The data's date frequency.
        frequency : Frequency
            Aggregation of the spend periods.

        Returns
        -------
        xr.DataArray
            As :meth:`_carryover_band_to_matrix` returns it.
        """
        if tail_lags is None:  # pragma: no cover - _refuse_unbounded_reach ran
            raise ValueError("An unbounded reach reached the closed form.")

        dates = self.data.dates
        out_dims = tuple(
            str(dim) for dim in saturated.dims if dim not in ("chain", "draw", "date")
        )
        n_chains, n_draws = saturated.sizes["chain"], saturated.sizes["draw"]
        sizes = tuple(saturated.sizes[dim] for dim in out_dims)

        tails = [
            self._tail_dates(
                window=window,
                dates=dates,
                tail_lags=tail_lags,
                freq_offset=freq_offset,
            )
            for window in windows.windows
        ]
        realization = windows.windows[0].eval_dates
        for window, tail in zip(windows.windows, tails, strict=True):
            realization = realization.union(window.eval_dates).union(tail)

        # Straight into the matrix: it is the size of the result, and a second
        # array of that size to copy from would cost more than the kernel pass.
        values = np.zeros((n_chains, n_draws, len(periods), len(realization), *sizes))
        observed = np.ones((len(periods), len(realization)), dtype=bool)
        row_of = _row_of_dates(dates, periods)
        columns = realization.get_indexer(dates)
        for lag, term in enumerate(_adstock_terms(weights, saturated)):
            _scatter_by_cohort(
                values,
                term.transpose("chain", "draw", "date", *out_dims).values,
                row_of=row_of,
                lag=lag,
                columns=columns,
            )
        for row, tail in enumerate(tails):
            tail_columns = realization.get_indexer(tail)
            values[:, :, row, tail_columns] = np.nan
            observed[row, tail_columns] = False

        return self._matrix_from_arrays(
            values,
            observed,
            realization=realization,
            windows=windows.windows,
            coords={
                "chain": saturated["chain"].values,
                "draw": saturated["draw"].values,
                **{dim: saturated.coords[dim].values for dim in out_dims},
            },
            out_dims=out_dims,
            frequency=frequency,
        )

    def _refuse_unsupported_closed_form(self) -> None:
        r"""Refuse what the closed form is not exact for, from the model alone.

        ``method="closed_form"`` reads each cohort's response as
        :math:`w_l V_s` on lag :math:`l`: the adstock's kernel times a change in
        saturated spend that depends on that date's spend alone.  That holds
        only when saturation runs first, the kernel is a known linear
        convolution that places its first weight on the spend period, and
        ``channel_contribution`` is the only route from spend to the response.
        Outside that class the closed form is not exact, so it raises;
        ``method="pipeline"`` handles those models.

        Every check reads the model's structure or its posterior, so none of
        them needs an evaluation.  What only the spend probe can establish (a
        saturation that carries spend over a window or mixes channels) is
        refused after the probe, with the same checks as the allocation.

        Raises
        ------
        ValueError
            If a random adstock or saturation parameter is missing from the
            posterior.
        NotImplementedError
            For ``adstock_first=True``, a non-identity link, a custom adstock, a
            channel-dependent ``mu_effect``, an adstock that is not
            ``ConvMode.After`` or ``time_varying_media``.
        """
        if self.model.adstock_first:
            raise NotImplementedError(
                "method='closed_form' needs adstock_first=False.  With "
                "saturation after the adstock it acts on the sum of every "
                "cohort's carryover, so a cohort's response is not its kernel "
                "times its own spend.  Use method='pipeline'."
            )

        self._refuse_non_separable_route(
            CLOSED_FORM_REQUIREMENT, hint="  Use method='pipeline'."
        )

        if not _is_builtin_adstock(self.model.adstock):
            raise NotImplementedError(
                "method='closed_form' needs one of the library's adstocks, whose "
                "kernels are known to be linear convolutions, got "
                f"{type(self.model.adstock).__name__}.  Use method='pipeline', "
                "which evaluates any adstock."
            )

        # TODO: support time_varying_media.  The pipeline already does, by
        # scaling the contribution on each realization date.  Here a cohort's
        # entry on lag l would be beta * m_{s+l} * w_l * g(x_s): the multiplier
        # has to enter the scatter per lag, and the split can no longer use one
        # cumulative weight per draw.
        if self.model.time_varying_media:
            raise NotImplementedError(
                "method='closed_form' does not support time_varying_media.  The "
                "temporal multiplier acts on each realization date, so a "
                "cohort's entries are no longer the kernel's weights times one "
                "number.  Use method='pipeline'."
            )

        self._refuse_missing_parameters("adstock", CLOSED_FORM_REQUIREMENT)
        self._refuse_missing_parameters("saturation", CLOSED_FORM_REQUIREMENT)

    def _refuse_non_separable_route(self, requirement: str, hint: str = "") -> None:
        """Refuse the model structures that neither the allocation nor the closed form reads.

        Both read a date's contribution as cohorts of one channel's own spend,
        each spread over the dates after it by the adstock kernel.  That needs
        an identity link, a kernel placed on and after the spend date, and
        ``channel_contribution`` as the only route from spend to the response.

        Parameters
        ----------
        requirement : str
            Who refuses, for the message: :data:`ALLOCATION_REQUIREMENT` or
            :data:`CLOSED_FORM_REQUIREMENT`.
        hint : str, optional
            A sentence appended to each message, such as where to go instead.

        Raises
        ------
        NotImplementedError
            Under a non-identity link, an adstock that is not
            ``ConvMode.After`` or a channel-dependent ``mu_effect``.
        """
        if self.model.link != LinkFunction.IDENTITY:
            raise NotImplementedError(
                f"{requirement} needs link='identity'.  Under "
                f"link='{self.model.link}' channel_contribution lives in the "
                "linear predictor, so a response-scale increment is not a sum "
                f"of the cohorts' contributions.{hint}"
            )

        mode = self.model.adstock.mode
        if mode != ConvMode.After:
            raise NotImplementedError(
                f"{requirement} needs a forward-looking adstock "
                f"(ConvMode.After), got {mode!s}.  A leading kernel puts weight "
                "on dates before the spend, which the kernel's lags do not "
                f"describe.{hint}"
            )

        if resolve_channel_dependent_effects(self.model):
            raise NotImplementedError(
                f"{requirement} needs channel_contribution to be the only route "
                "from spend to the response.  This model has a "
                "channel-dependent mu_effect, whose contribution would be left "
                f"out.{hint}"
            )

    @staticmethod
    def _refuse_allocation_spend_factor(counterfactual_spend_factor: float) -> None:
        """Refuse an allocation against anything but zero spend.

        Parameters
        ----------
        counterfactual_spend_factor : float
            Must be ``0``: the allocation splits each date's contribution less
            its value at zero spend.

        Raises
        ------
        ValueError
            If ``counterfactual_spend_factor`` is not ``0``.
        """
        if counterfactual_spend_factor != 0.0:
            raise ValueError(
                "estimand='allocation' splits the contribution against zero "
                "spend, so counterfactual_spend_factor must be 0, got "
                f"{counterfactual_spend_factor}."
            )

    def _refuse_unsupported_allocation(
        self, counterfactual_spend_factor: float
    ) -> None:
        """Refuse what the allocation is not defined for, from the model alone.

        Every check here reads the model's structure, so none of them needs an
        evaluation.  The ones that need the spend probe run inside
        :meth:`_compute_increments`, through :func:`_refuse_unbounded_reach` and
        :func:`_refuse_dedicated_rows`.

        Parameters
        ----------
        counterfactual_spend_factor : float
            Must be ``0``: the allocation is measured against zero spend.

        Raises
        ------
        ValueError
            If ``counterfactual_spend_factor`` is not ``0``, or a random adstock
            parameter is missing from the posterior.
        NotImplementedError
            Under a log link, a non-``After`` adstock or a channel-dependent
            ``mu_effect``; and, with ``adstock_first=True``, for negative
            channel data.
        """
        self._refuse_allocation_spend_factor(counterfactual_spend_factor)
        self._refuse_non_separable_route(ALLOCATION_REQUIREMENT)

        if self.model.adstock_first:
            self._refuse_signed_spend()

        # The kernel is sampled whenever the shares need it or a custom adstock
        # has to be checked for linearity.
        if self.model.adstock_first or not _is_builtin_adstock(self.model.adstock):
            self._refuse_missing_parameters("adstock")

    def _refuse_signed_spend(self) -> None:
        """Refuse negative spend, which the adstock-first shares cannot split.

        The shares divide each date's contribution by its adstocked spend, and
        with spend of both signs that can sit near or at zero while the
        contribution does not, so the columns would not reconcile.

        Raises
        ------
        NotImplementedError
            If any channel has negative spend.
        """
        if (self.data.get_channel_data() < 0).any():
            raise NotImplementedError(
                "estimand='allocation' with adstock_first=True splits each "
                "date's contribution in proportion to the adstocked spend, "
                "which is not defined for negative channel data.  Remove the "
                "negative values, or fit with adstock_first=False, where the "
                "allocation does not divide by it."
            )

    def _refuse_missing_parameters(
        self,
        component: Literal["adstock", "saturation"],
        requirement: str = ALLOCATION_REQUIREMENT,
    ) -> None:
        """Refuse a posterior that lacks a parameter *component* is drawn from.

        Evaluating the transformation draws a parameter that is missing from the
        posterior from its prior, so the kernel or the saturated spend would no
        longer be the one the contribution was computed at.  Fixed parameters
        never appear in the posterior and are skipped.

        Parameters
        ----------
        component : {"adstock", "saturation"}
            Which of the model's transformations is evaluated.
        requirement : str, default=ALLOCATION_REQUIREMENT
            What needs it, for the message.

        Raises
        ------
        ValueError
            If a random parameter of *component* is not in the posterior.
        """
        missing = _missing_variables(
            getattr(self.model, component), self.idata.posterior.dataset
        )
        if missing:
            raise ValueError(
                f"{requirement} needs the {component} parameters {missing} in the "
                "posterior; without them they would be drawn from the prior."
            )

    def _allocation_increments(
        self,
        *,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
        posterior: xr.Dataset,
        central_tendency: CentralTendency,
    ) -> PeriodIncrements:
        """Evaluate the zero-spend counterfactual the allocation is built from.

        Per channel, carryover included, kept per realization date, on the
        caller's posterior subset, and refused when the probe cannot verify that the
        effect of a spend change dies out within the fitted dates or cannot show the channels to be separable.

        Parameters
        ----------
        frequency, start_date, end_date
            The spend periods to evaluate.
        posterior : xr.Dataset
            The already subsampled posterior, shared with the shares.
        central_tendency : {"median", "mean"}
            As in :meth:`split_incremental_contribution_over_time`.

        Returns
        -------
        PeriodIncrements
            Each period's per-date band, as :meth:`_compute_increments` with
            ``keep_date_axis``.
        """
        return self._compute_increments(
            scope="per_channel",
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=True,
            num_samples=None,
            random_state=None,
            counterfactual_spend_factor=0.0,
            central_tendency=central_tendency,
            keep_date_axis=True,
            validate_reach=partial(
                _refuse_unbounded_reach,
                requirement=ALLOCATION_REQUIREMENT,
                kernel_lags=kernel_trailing_lags(
                    self.model.adstock.l_max, self.model.adstock.mode
                ),
            ),
            posterior=posterior,
            validate_dedicated_rows=partial(
                _refuse_dedicated_rows, requirement=ALLOCATION_REQUIREMENT
            ),
        )

    @staticmethod
    def _rows_complete(
        periods: list[tuple[pd.Timestamp, pd.Timestamp]],
        dates: pd.DatetimeIndex,
        tail_lags: int,
    ) -> list[bool]:
        """Whether each period's carryover lands within the fitted dates.

        Parameters
        ----------
        periods : list of tuple
            ``(start, end)`` of each spend period.
        dates : pd.DatetimeIndex
            The fitted date axis.
        tail_lags : int
            Lags after a period's last spend date that its carryover reaches,
            in data periods.

        Returns
        -------
        list of bool
            ``True`` where the period's last fitted spend date plus *tail_lags*
            is still a fitted date.  A period with no fitted date has no spend,
            so nothing of it lands past the data and it is ``True``, as the
            counterfactual's ``observed`` coordinate has it.
        """
        complete = []
        for start, end in periods:
            spent = np.flatnonzero((dates >= start) & (dates <= end))
            complete.append(
                not len(spent) or bool(spent[-1] + tail_lags <= len(dates) - 1)
            )
        return complete

    def _on_fitted_dates(self, matrix: xr.DataArray) -> xr.DataArray:
        """Give a matrix of per-period bands the fitted dates as its columns.

        The bands only cover the dates the requested periods move, and any past
        the last fitted date are not data.  Every other fitted date is a cell no
        requested period reaches, so it is ``0``, and an observed zero.

        Parameters
        ----------
        matrix : xr.DataArray
            As :meth:`_carryover_band_to_matrix` returns it.

        Returns
        -------
        xr.DataArray
            *matrix* with ``realization_date`` equal to the fitted dates.
        """
        dates = self.data.dates
        within = matrix.realization_date.values <= dates[-1]
        matrix = matrix.isel(realization_date=within)
        # reindex would fill the non-index ``observed`` coordinate with NaN.
        observed = matrix.coords["observed"].reindex(
            realization_date=dates, fill_value=True
        )
        return (
            matrix.drop_vars("observed")
            .reindex(realization_date=dates, fill_value=0.0)
            .assign_coords(observed=observed)
        )

    def _allocate_by_shares(
        self,
        contribution: xr.DataArray,
        *,
        weights: xr.DataArray,
        periods: list[tuple[pd.Timestamp, pd.Timestamp]],
        frequency: Frequency,
    ) -> xr.DataArray:
        r"""Split each date's contribution among the cohorts that produced it.

        The Aumann-Shapley share split for ``adstock_first=True``,

        .. math::

            A[s, t] = (c_t - c_t^0)\,\frac{w_{t-s}\,x_s}{z_t},

        with :math:`z_t = \sum_l w_l x_{t-l}`.  The weights are the adstock's
        impulse response at the posterior draws, which is its kernel for any
        causal convolution; a date with no adstocked spend gets ``0``.  The
        shares do not depend on the channel's scale, so the spend is taken as
        recorded.

        Two passes over the lags keep memory at a few arrays the size of
        *contribution* rather than ``l_max`` of them: the first sums the
        adstocked spend :math:`z_t`, the second recomputes each lag's term and
        places its share of :math:`(c_t - c_t^0) / z_t` in the matrix.

        Parameters
        ----------
        contribution : xr.DataArray
            :math:`c_t - c_t^0` in original scale, as the per-date band of the
            all-time counterfactual.  Dims ``(chain, draw, realization_date,
            channel, *custom_dims)`` with an ``observed`` coordinate.
        weights : xr.DataArray
            The adstock's kernel at the draws *contribution* was computed at,
            from :meth:`_adstock_kernel`.
        periods : list of tuple
            ``(start, end)`` of each spend period, the rows of the result.
        frequency : Frequency
            Aggregation of the periods; ``"all_time"`` drops ``spend_date``.

        Returns
        -------
        xr.DataArray
            Dims ``(chain, draw, spend_date, realization_date, channel,
            *custom_dims)``, ``realization_date`` being the fitted dates, with
            ``period_start`` and ``observed`` coordinates as
            :meth:`_carryover_band_to_matrix`.  ``spend_date`` is a scalar
            coordinate for a single ``"all_time"`` period.
        """
        out_dims = tuple(
            dim
            for dim in contribution.dims
            if dim not in ("chain", "draw", "realization_date")
        )
        total = (
            contribution.isel(realization_date=contribution["observed"].values)
            .drop_vars("observed")
            .rename(realization_date="date")
        )
        dates = pd.DatetimeIndex(total["date"].values)
        # The share tests pin the allocation's columns to the contribution draw
        # by draw, on a subsample too, so a change in the order fails them.
        weights = _relabel_draws(
            weights, {"chain": total["chain"].values, "draw": total["draw"].values}
        )
        # The kernel and the spend go into the contribution's label order, the
        # one its values and the matrix's coordinates are laid out in: a labeled
        # product keeps its first operand's order, and the masks below pair
        # cells exactly.  The posterior and ``constant_data`` may store others.
        in_total_order = {dim: total[dim].values for dim in out_dims}
        weights = weights.sel(
            {
                dim: labels
                for dim, labels in in_total_order.items()
                if dim in weights.dims
            }
        )
        spend = self.data.get_channel_data().sel(in_total_order)

        adstocked = sum(_adstock_terms(weights, spend))
        positive = adstocked > 0
        # The contribution per unit of adstocked spend: each cohort's entry is
        # this times its own term.  A NaN contribution on a date with spend
        # still comes through as NaN.
        per_unit = (total / adstocked.where(positive)).where(positive, 0.0)

        row_of = _row_of_dates(dates, periods)
        values = np.zeros(
            (
                total.sizes["chain"],
                total.sizes["draw"],
                len(periods),
                len(dates),
                *(total.sizes[dim] for dim in out_dims),
            ),
            dtype=total.dtype,
        )
        for lag, term in enumerate(_adstock_terms(weights, spend)):
            landed = per_unit * term
            _scatter_by_cohort(
                values,
                landed.transpose("chain", "draw", "date", *out_dims).values,
                row_of=row_of,
                lag=lag,
            )

        matrix = xr.DataArray(
            values,
            dims=("chain", "draw", "spend_date", "realization_date", *out_dims),
            coords={
                **{
                    name: coord
                    for name, coord in total.coords.items()
                    if name in ("chain", "draw", *out_dims)
                },
                "spend_date": [end for _, end in periods],
                "realization_date": dates,
            },
        ).assign_coords(
            period_start=("spend_date", [start for start, _ in periods]),
            observed=(
                ("spend_date", "realization_date"),
                np.ones((len(periods), len(dates)), dtype=bool),
            ),
        )

        if frequency == "all_time":
            matrix = matrix.squeeze("spend_date", drop=False)

        return matrix

    def _adstock_parameters(self, posterior: xr.Dataset) -> xr.Dataset:
        """Keep the posterior's draws, and only the adstock's own variables.

        Parameters
        ----------
        posterior : xr.Dataset
            The draws the kernel is sampled at.

        Returns
        -------
        xr.Dataset
            *posterior* without the variables the adstock does not read; see
            :func:`_transformation_parameters`.
        """
        return _transformation_parameters(posterior, self.model.adstock)

    def _adstock_kernel(self, posterior: xr.Dataset, *, divides: bool) -> xr.DataArray:
        """Sample the adstock's kernel at each posterior draw, and vet it.

        Everything the allocation needs to know about the kernel is checked
        here, from the posterior alone, so a kernel it cannot use is refused
        before any period is evaluated.

        Parameters
        ----------
        posterior : xr.Dataset
            The draws to evaluate the kernel at.
        divides : bool
            Whether the allocation divides by the adstocked spend, as the
            adstock-first shares do.  Only then must the weights be
            non-negative.

        Returns
        -------
        xr.DataArray
            The adstock of a unit impulse, with a ``"time since exposure"`` lag
            dimension of length ``l_max`` and the parameters' other dimensions.
            The draws are in the posterior's order and under its labels, which
            need not be the pipeline's; see :func:`_relabel_draws`.

        Raises
        ------
        NotImplementedError
            If *divides* and a kernel weight is negative, or the adstock is a
            custom one that fails the homogeneity check.
        """
        adstock = self.model.adstock
        parameters = self._adstock_parameters(posterior)
        with _quiet_posterior_predictive_log():
            weights = adstock.sample_curve(parameters=parameters, progressbar=False)
        if divides and bool((weights < 0).any()):
            raise NotImplementedError(
                "estimand='allocation' needs the adstock's kernel weights to be "
                "non-negative: with a negative weight the adstocked spend can "
                "be zero or negative on a date where the contribution is not, "
                "and that date would be left out of the matrix."
            )
        if not _is_builtin_adstock(adstock):
            _require_homogeneous_adstock(adstock, weights, parameters)
        return weights

    def split_incremental_contribution_current_future(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        horizon: int | None = None,
        method: Literal["pipeline", "closed_form"] = "pipeline",
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
        counterfactual_spend_factor: float = 0.0,
        central_tendency: CentralTendency = "median",
    ) -> xr.Dataset:
        r"""Split each period's incremental contribution into current and future.

        *Current* is what a period's spend produced on the dates of that same
        reporting period, *future* is the carryover that lands afterwards.  The
        two add up to :meth:`compute_incremental_contribution` wherever the
        carryover window is fully observed.  Elsewhere ``future`` is ``NaN``,
        while :meth:`compute_incremental_contribution` counts only the observed
        part, a lower bound on the period's effect when more spend never lowers
        the response on a later date.  The numbers are those of
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
        method : {"pipeline", "closed_form"}, default="pipeline"
            ``"pipeline"`` evaluates the model once per spend period and is
            exact for every supported model.  ``"closed_form"`` computes the
            same split, ``NaN`` for ``NaN``, from the adstock kernel and the
            saturated spend, with no evaluation per period and no matrix; see
            the Notes.  It raises on the models listed under ``method`` in
            :meth:`split_incremental_contribution_over_time`, except the last:
            the split has no estimand, so a ``counterfactual_spend_factor``
            other than ``0`` works here.
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
            ``effective_horizon``, or if ``method`` is unknown.  All of these
            are raised before any period is evaluated.
        NotImplementedError
            For ``method="closed_form"`` on a model outside the class it is
            exact for, or when the spend probe finds that spend moves the
            response further than the kernel reaches.

        Notes
        -----
        The shares are shares of *modelled* value within the truncated kernel.
        Nothing past ``l_max`` is attributed, so with ``l_max=13`` on weekly data
        "future" means future within a quarter.

        At an aggregated frequency ``future_share`` is not a property of the
        channel.  It depends on when the spend fell within the period: the same
        channel and model give a month whose spend is front-loaded a smaller
        future share than one whose spend comes in the last week, exactly as the
        adstock weights predict.  To combine periods, divide the sums over
        ``spend_date`` rather than averaging the shares:
        ``split["future"].sum("spend_date")`` over
        ``(split["current"] + split["future"]).sum("spend_date")``.  With the
        default ``skipna`` both sums run over the complete periods only, because
        ``future`` is ``NaN`` on the others.

        ``current + future`` is each period's incremental contribution, so the
        caveat of :meth:`split_incremental_contribution_over_time` applies:
        summed across periods it falls short of ``channel_contribution`` when
        saturation follows adstock.

        With the default dates, ``frequency="all_time"`` always reports
        ``future`` as ``NaN``: the single period contains the last fitted date,
        whose carryover the data cannot show.  Pass an ``end_date`` at least
        ``effective_horizon`` periods before the last fitted date to get a
        complete all-time split.

        With ``method="closed_form"``, spend on date :math:`s` puts
        :math:`V_s W\,C(m)` into current and the rest of :math:`V_s W` into
        future.  :math:`V_s` is the change in the saturated spend,
        :math:`W` the kernel's total weight and :math:`C(m)` the share of the
        kernel within the :math:`m` lags that count, as
        :meth:`~pymc_marketing.mmm.components.adstock.AdstockTransformation.cumulative_weights`
        returns it.  At ``frequency="original"`` :math:`m` is ``horizon`` for
        every date; at an aggregated frequency it is the number of data periods
        left in the reporting period.  A period's value is the sum over its
        dates.

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
        if method not in ("pipeline", "closed_form"):
            raise ValueError(
                f"method must be 'pipeline' or 'closed_form', got {method!r}"
            )

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

        if method == "closed_form":
            self._refuse_unsupported_closed_form()

            return self._current_future_closed_form(
                frequency=frequency,
                start_date=start_date,
                end_date=end_date,
                lags=lags,
                num_samples=num_samples,
                random_state=random_state,
                counterfactual_spend_factor=counterfactual_spend_factor,
                central_tendency=central_tendency,
            )

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
            notes.append(INCOMPLETE_SPLIT_NOTE)

        result.attrs = self._carryover_attrs(
            increments.reach,
            estimand="counterfactual",
            method="pipeline",
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            notes=notes,
        )
        result.attrs["horizon"] = lags

        return result

    def _current_future_closed_form(
        self,
        *,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
        lags: int,
        num_samples: int | None,
        random_state: RandomState | Generator | None,
        counterfactual_spend_factor: float,
        central_tendency: CentralTendency,
    ) -> xr.Dataset:
        r"""Build the current/future split of ``method="closed_form"`` without a matrix.

        Date :math:`s`, spent in period :math:`P`, puts :math:`V_s \sum_{l \le m}
        w_l` on the dates up to the period's current end, :math:`m` data periods
        after it.  With :math:`W = \sum_l w_l` and :math:`C(m)` the share of the
        kernel within :math:`m` lags, that is :math:`V_s W\,C(m)`, and the rest
        of :math:`V_s W` is future.  A period's value is the sum over its dates.
        At ``frequency="original"`` :math:`m` is the one constant ``horizon`` and
        :math:`C` is the share
        :meth:`~pymc_marketing.mmm.components.adstock.AdstockTransformation.cumulative_weights`
        returns; at an aggregated frequency :math:`m` is the number of data
        periods left in the reporting period, which depends on the date.  Both
        read the cumulative kernel from one sampling, and no period-by-date
        matrix is built.

        ``NaN`` follows the pipeline: ``future`` where a period's carryover runs
        past the data, ``current`` too when that carryover falls inside the
        current window.

        Parameters
        ----------
        frequency, start_date, end_date, num_samples, random_state
            As in :meth:`split_incremental_contribution_current_future`.
        counterfactual_spend_factor, central_tendency
            As in :meth:`split_incremental_contribution_current_future`.
        lags : int
            Lags after each spend date that count as current, already validated.

        Returns
        -------
        xr.Dataset
            As :meth:`split_incremental_contribution_current_future`.
        """
        inputs = self._closed_form_inputs(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            num_samples=num_samples,
            random_state=random_state,
            central_tendency=central_tendency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            lags=lags,
        )
        prepared = inputs.prepared
        adstock = self.model.adstock
        reach, windows = prepared.probed.reach, prepared.windows.windows
        max_lag = reach.max_lag
        if max_lag is None:  # pragma: no cover - _refuse_unbounded_reach ran
            raise ValueError("An unbounded reach reached the closed form.")

        # The kernel's weight within each lag, W C(m), read off one sampling of
        # the kernel.  Its last entry is the whole kernel, so the future part is
        # exactly zero for a date whose window covers it.
        cumulative = _relabel_draws(
            self._adstock_kernel(prepared.posterior, divides=False), inputs.labels
        ).cumsum(LAG_DIM)

        dates = self.data.dates
        offset = prepared.freq_offset
        saturated = inputs.saturated
        out_dims = tuple(
            str(dim) for dim in saturated.dims if dim not in ("chain", "draw", "date")
        )
        row_of = _row_of_dates(dates, prepared.periods)
        current_end = pd.DatetimeIndex(
            [
                self._current_window_end(window.end, lags=lags, freq_offset=offset)
                for window in windows
            ]
        )
        spent, counted = self._lags_within_window(
            dates,
            row_of,
            current_end,
            l_max=adstock.l_max,
            max_lag=max_lag,
            freq_offset=offset,
        )

        cohorts = saturated.isel(date=spent)
        within = cumulative.isel({LAG_DIM: xr.DataArray(counted, dims="date")})
        within = within.assign_coords(date=cohorts["date"].values)
        whole = cumulative.isel({LAG_DIM: -1}, drop=True)
        order = ("chain", "draw", "date", *out_dims)
        rows = row_of[spent]

        current = self._sum_by_period(
            (cohorts * within).transpose(*order).values, rows, len(windows)
        )
        future = self._sum_by_period(
            (cohorts * (whole - within)).transpose(*order).values, rows, len(windows)
        )

        cut, current_unobserved, future_unobserved = self._cut_periods(
            dates, row_of, current_end, max_lag=max_lag, freq_offset=offset
        )
        current[:, :, current_unobserved] = np.nan
        future[:, :, future_unobserved] = np.nan

        coords = {
            **inputs.labels,
            "spend_date": pd.Index(
                [window.end for window in windows], name="spend_date"
            ),
            **{dim: saturated.coords[dim].values for dim in out_dims},
        }
        dims = ("chain", "draw", "spend_date", *out_dims)
        result = self._current_future_dataset(
            xr.DataArray(current, dims=dims, coords=coords),
            xr.DataArray(future, dims=dims, coords=coords),
            list(~cut),
            period_starts=pd.DatetimeIndex([window.start for window in windows]),
            frequency=frequency,
            custom_dims=list(self.model.dims),
        )

        notes: list[str] = []
        if cut.any():
            notes.append(INCOMPLETE_SPLIT_NOTE)
        result.attrs = self._carryover_attrs(
            reach,
            estimand="counterfactual",
            method="closed_form",
            frequency=frequency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            notes=notes,
        )
        result.attrs["horizon"] = lags

        return result

    @staticmethod
    def _current_window_end(
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

    @staticmethod
    def _dates_past_the_data(
        dates: pd.DatetimeIndex, *, periods: int, freq_offset: BaseOffset
    ) -> pd.DatetimeIndex:
        """Extend the fitted dates by *periods* data periods.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            The fitted date axis.
        periods : int
            How many dates to add after the last one.
        freq_offset : BaseOffset
            The data's date frequency.

        Returns
        -------
        pd.DatetimeIndex
            *dates* followed by the *periods* dates after them.
        """
        return pd.DatetimeIndex(
            dates.append(
                pd.date_range(dates[-1], periods=periods + 1, freq=freq_offset)[1:]
            )
        )

    @staticmethod
    def _lags_within_window(
        dates: pd.DatetimeIndex,
        row_of: np.ndarray,
        current_end: pd.DatetimeIndex,
        *,
        l_max: int,
        max_lag: int,
        freq_offset: BaseOffset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Count, for each spend date, the kernel lags inside its current window.

        Lag :math:`l` of spend on date :math:`s` lands :math:`l` data periods
        later, and it is current if that date is on or before the window's end.
        Lags past the kernel carry no weight, so the count stops at
        ``l_max - 1``.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            The fitted date axis.
        row_of : np.ndarray
            Per date, its period, or ``-1``; see :func:`_row_of_dates`.
        current_end : pd.DatetimeIndex
            Per period, the last date of its current window.
        l_max : int
            The adstock's number of kernel weights.
        max_lag : int
            The measured reach, how far past the data a window can need dates.
        freq_offset : BaseOffset
            The data's date frequency.

        Returns
        -------
        spent : np.ndarray
            Indices of the dates that belong to a period.
        counted : np.ndarray
            For each of them, the last lag that is current, ``0 .. l_max - 1``.
        """
        spent = np.flatnonzero(row_of >= 0)
        extended = Incrementality._dates_past_the_data(
            dates, periods=max_lag, freq_offset=freq_offset
        )
        position = np.searchsorted(
            extended.values, current_end.values[row_of[spent]], "right"
        )
        return spent, np.clip(position - 1 - spent, 0, l_max - 1)

    @staticmethod
    def _cut_periods(
        dates: pd.DatetimeIndex,
        row_of: np.ndarray,
        current_end: pd.DatetimeIndex,
        *,
        max_lag: int,
        freq_offset: BaseOffset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the periods whose carryover the data cuts, and where that shows.

        A period is cut when its last spend date plus the reach lies past the
        last fitted date.  Its current value is then unobserved if the current
        window reaches the first date after the data, and its future value if
        the carryover runs past the window.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            The fitted date axis.
        row_of : np.ndarray
            Per date, its period, or ``-1``; see :func:`_row_of_dates`.
        current_end : pd.DatetimeIndex
            Per period, the last date of its current window.
        max_lag : int
            The measured reach, in data periods.
        freq_offset : BaseOffset
            The data's date frequency.

        Returns
        -------
        cut, current_unobserved, future_unobserved : np.ndarray
            Booleans per period.
        """
        n_dates = len(dates)
        last_spent = np.full(len(current_end), -1)
        spent = np.flatnonzero(row_of >= 0)
        np.maximum.at(last_spent, row_of[spent], spent)
        cut = (last_spent >= 0) & (last_spent + max_lag > n_dates - 1)

        extended = Incrementality._dates_past_the_data(
            dates, periods=max_lag, freq_offset=freq_offset
        )
        reached = extended.values[np.where(cut, last_spent + max_lag, 0)]
        # The first unobserved date comes from the last fitted one, since the
        # extended axis is empty past the data when the reach is zero.
        first_unobserved = dates[-1] + freq_offset
        current_unobserved = cut & np.asarray(first_unobserved <= current_end)
        future_unobserved = cut & (reached > current_end.values)
        return cut, current_unobserved, future_unobserved

    @staticmethod
    def _sum_by_period(values: np.ndarray, rows: np.ndarray, n_rows: int) -> np.ndarray:
        """Sum per-date values into their periods.

        Parameters
        ----------
        values : np.ndarray
            ``(chain, draw, date, *rest)`` on the dates that belong to a period.
        rows : np.ndarray
            Each of those dates' period, non-decreasing.
        n_rows : int
            The number of periods; a period without dates sums to ``0``.

        Returns
        -------
        np.ndarray
            ``(chain, draw, period, *rest)``.
        """
        summed = np.zeros((*values.shape[:2], n_rows, *values.shape[3:]))
        if len(rows):
            # Each period's dates are contiguous, so reduceat sums each run.
            starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
            summed[:, :, rows[starts]] = np.add.reduceat(values, starts, axis=2)
        return summed

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
        current_end = Incrementality._current_window_end(
            end, lags=lags, freq_offset=freq_offset
        )
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

        return Incrementality._current_future_dataset(
            xr.concat(current, dim=index),
            xr.concat(future, dim=index),
            complete,
            period_starts=period_starts,
            frequency=frequency,
            custom_dims=custom_dims,
        )

    @staticmethod
    def _current_future_dataset(
        current: xr.DataArray,
        future: xr.DataArray,
        complete: Sequence[bool],
        *,
        period_starts: pd.DatetimeIndex,
        frequency: Frequency,
        custom_dims: Sequence[str],
    ) -> xr.Dataset:
        """Assemble current, future and future share into the split's dataset.

        The pipeline stacks one array per period into it, the closed form passes
        every period at once.

        Parameters
        ----------
        current, future : xr.DataArray
            Dims ``(chain, draw, spend_date, channel, *custom_dims)`` in any
            order, with ``spend_date`` labelled by each period's end.
        complete : sequence of bool
            Per period, whether its carryover is fully observed.
        period_starts : pd.DatetimeIndex
            Per period, its start.
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
        total = current + future
        result = xr.Dataset(
            {
                "current": current,
                "future": future,
                "future_share": (future / total).where(total != 0),
            }
        ).assign_coords(
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
        reach: SpendReach,
        *,
        estimand: str,
        method: str,
        frequency: Frequency,
        counterfactual_spend_factor: float,
        notes: list[str],
        extra_assumptions: dict | None = None,
    ) -> dict:
        """Describe a carryover result in attributes ``to_netcdf`` can write.

        Parameters
        ----------
        reach : SpendReach
            What the spend probe measured for the evaluation behind the result.
        estimand, method : str
            As passed to :meth:`split_incremental_contribution_over_time`.
        frequency : Frequency
            Aggregation of the spend periods.
        counterfactual_spend_factor : float
            Multiplicative factor applied to spend in the counterfactual.
        notes : list of str
            Caller-specific notes on how to read the result; the ones every
            carryover result shares are appended here.
        extra_assumptions : dict, optional
            Further JSON-serializable entries for ``assumptions``.

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
            **(extra_assumptions or {}),
        }
        attrs: dict = {"estimand": estimand, "method": method}
        if reach.max_lag is not None:
            attrs["effective_horizon"] = int(reach.max_lag)
        attrs["assumptions"] = json.dumps(assumptions)
        attrs["warnings"] = json.dumps(notes)
        return attrs

    @cached_property
    def _data_fingerprint(self) -> tuple | None:
        """Content fingerprint of :attr:`idata`'s ``constant_data``, computed once.

        ``idata`` is fixed for the life of an ``Incrementality`` object.
        Hashing every variable on every request would only repeat the same work.
        """
        return constant_data_fingerprint(self.idata)

    def _evaluator_cache_key(
        self, response_vars: Sequence[str | Variable]
    ) -> EvaluatorCacheKey:
        """Return the cache identity of the evaluator :meth:`_compute_increments` needs.

        The key mirrors what the evaluator bakes into its compiled graph (responses, target, mode, frozen
        deterministics) and identifies the data by its dates and a ``constant_data`` fingerprint.

        Parameters
        ----------
        response_vars : sequence of str or Variable
            Responses in order.  A node is keyed by its name, which is how the
            evaluator resolves it against its intervened clone.

        Returns
        -------
        EvaluatorCacheKey
            Identity the evaluator is cached under.

        Raises
        ------
        ValueError
            If a response node has no name, which the evaluator could not
            resolve either.
        """
        names = tuple(
            var if isinstance(var, str) else var.name for var in response_vars
        )
        unnamed = [position for position, name in enumerate(names) if name is None]
        if unnamed:
            raise ValueError(
                f"Response variables at positions {unnamed} have no name; the "
                "evaluator resolves responses by name."
            )
        return EvaluatorCacheKey(
            response_vars=cast(tuple[str, ...], names),
            intervention_target=CounterfactualEvaluator.CHANNEL_DATA,
            intervention_mode="replace",
            frozen_deterministics=tuple(self.model.frozen_deterministics),
            dates=tuple(self.data.dates),
            data_fingerprint=self._data_fingerprint,
        )

    def _prepare_evaluation(
        self,
        *,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
        num_samples: int | None,
        random_state: RandomState | Generator | None,
        central_tendency: CentralTendency,
        counterfactual_spend_factor: float,
        include_carryover: bool,
        validate_reach: Callable[[SpendReach], None] | None,
        posterior: xr.Dataset | None = None,
    ) -> PreparedEvaluation:
        """Validate a request, subsample the posterior, probe the reach and build the windows.

        Everything that comes before a period is evaluated or a closed form is
        read, in one order for every route, so that the pipeline and the closed
        form validate, subsample and window a request the same way.

        Parameters
        ----------
        frequency, start_date, end_date, num_samples, random_state, central_tendency
            As in :meth:`_compute_increments`.
        counterfactual_spend_factor : float
            Multiplicative factor applied to spend in the counterfactual.
        include_carryover : bool
            Whether to extend the evaluation window by the carryover length.
        validate_reach : callable, optional
            Called with the measured reach before the windows are built.
        posterior : xr.Dataset, optional
            An already subsampled posterior, used as is.

        Returns
        -------
        PreparedEvaluation
            The periods, the posterior, the reducer, the effects, the probe, the
            date frequency and the windows.

        Raises
        ------
        ValueError
            If an input is out of range, or the date frequency cannot be inferred.
        NotImplementedError
            If the link has no reducer, or a channel-dependent ``mu_effect`` has
            not opted in.
        """
        if counterfactual_spend_factor < 0:
            raise ValueError(
                f"counterfactual_spend_factor must be >= 0, got {counterfactual_spend_factor}"
            )
        if central_tendency not in ("median", "mean"):
            raise ValueError(
                f"central_tendency must be 'median' or 'mean', got {central_tendency!r}"
            )

        start_ts, end_ts = self._validate_input(start_date, end_date)

        # Subsample the posterior across chain x draw, unless the caller did.
        posterior_sub = self._posterior_in_data_order(
            posterior
            if posterior is not None
            else subsample_draws(
                self.idata.posterior.dataset,
                num_samples=num_samples,
                random_state=random_state,
            )
        )

        # Resolve the link-specific reduction and the effects to include before
        # compiling anything, so an unsupported link or an effect that has not
        # opted in fails fast rather than after the expensive work.
        reducer = self._build_reducer(posterior_sub, central_tendency)
        effects = resolve_channel_dependent_effects(self.model)

        periods = self._create_period_groups(start_ts, end_ts, frequency)
        freq_offset = self._data_freq_offset()

        probed = self._probe_reach(
            posterior=posterior_sub,
            effects=effects,
            counterfactual_spend_factor=counterfactual_spend_factor,
            validate_reach=validate_reach,
        )

        # The stretch of dates each period is evaluated over, which of them enter
        # the sum, and the length they all stack to.  A failure here still has to
        # return the leased evaluator; the caller only releases once it has a
        # prepared evaluation.
        try:
            windows = EvaluationWindows.build(
                periods=periods,
                dates=self.data.dates,
                l_max=probed.reach.effective_l_max,
                freq_offset=freq_offset,
                full_axis=probed.reach.requires_full_axis,
                include_carryover=include_carryover,
                mode=self.model.adstock.mode,
            )
        except BaseException:
            probed.release()
            raise

        return PreparedEvaluation(
            start_ts=start_ts,
            end_ts=end_ts,
            periods=periods,
            posterior=posterior_sub,
            reducer=reducer,
            effects=tuple(effects),
            probed=probed,
            freq_offset=freq_offset,
            windows=windows,
        )

    def _evaluator_posterior(
        self, posterior: xr.Dataset, response_vars: Sequence[str | Variable]
    ) -> xr.Dataset:
        """Return the posterior variables the evaluator's graphs read.

        A compiled evaluator keeps the draws it was conditioned on for as long as
        it lies in the cache, so handing it the whole subsample would pin every
        large deterministic in the posterior -- ``channel_contribution`` among
        them -- although the graphs read only the free RVs and the frozen
        deterministics they are built over.  Narrowing here leaves the reducer,
        which does want the whole subsample, untouched.

        Parameters
        ----------
        posterior : xr.Dataset
            The already subsampled posterior.
        response_vars : sequence of str or Variable
            Responses the evaluator is built for.  A response given as a raw node
            is resolved against the model for the traversal only; the evaluator
            still receives the node.

        Returns
        -------
        xr.Dataset
            The variables those responses read, in ``posterior``'s own order.  A
            variable the posterior lacks is left out, so the evaluator raises its
            usual error when the graph asks for it.  When none of the variables
            they read is in ``posterior`` -- in particular when they read none,
            every channel parameter being fixed in the model config -- the result
            holds no variable, only the ``chain`` and ``draw`` coordinates the
            evaluator still broadcasts over; an empty selection would drop them.
        """
        model = self.model.model
        resolved = [
            model[var] if isinstance(var, str) else var for var in response_vars
        ]
        read = set(
            needed_posterior_names(model, resolved, self.model.frozen_deterministics)
        )
        names = [name for name in posterior.data_vars if name in read]
        if names:
            return posterior[names]
        return xr.Dataset(
            coords={"chain": posterior["chain"], "draw": posterior["draw"]}
        )

    def _probe_reach(
        self,
        *,
        posterior: xr.Dataset,
        effects: Sequence[ChannelDependentEffect],
        counterfactual_spend_factor: float,
        validate_reach: Callable[[SpendReach], None] | None,
    ) -> ProbedReach:
        """Compile the batched evaluator, evaluate the baseline and probe the reach.

        Called from :meth:`_prepare_evaluation` for every route.  The closed
        form evaluates no period, but it still needs the reach and the channel
        mixing that the probe measures on the evaluator.

        Parameters
        ----------
        posterior : xr.Dataset
            The already subsampled posterior the evaluator conditions on.
        effects : sequence of ChannelDependentEffect
            The channel-dependent effects to evaluate with ``channel_contribution``.
        counterfactual_spend_factor : float
            Multiplicative factor the probe applies to spend.
        validate_reach : callable, optional
            Called with the measured
            :class:`~pymc_marketing.mmm.spend_reach.SpendReach`, so that a request
            the reach rules out fails before anything else is evaluated.

        Returns
        -------
        ProbedReach
            The evaluator, the baseline and the probe with what it measured.
        """
        dates = self.data.dates

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
        response_vars: list[str | Variable] = [
            CHANNEL_CONTRIBUTION,
            *effect_names,
            *([predictor] if predictor is not None else []),
        ]
        # Reuse the compiled evaluator for this graph shape and dataset.  The
        # evaluator is leased for the whole request: it leaves the cache here and
        # comes back below, so a concurrent call misses and compiles its own
        # rather than rebinding the draws under this one.
        posterior_for_evaluator = self._evaluator_posterior(posterior, response_vars)
        build_evaluator = partial(
            CounterfactualEvaluator,
            pymc_model=posterior_predictive_model,
            posterior=posterior_for_evaluator,
            response_vars=response_vars,
            frozen_deterministics=self.model.frozen_deterministics,
            dates=dates,
            expected_aux_values=(
                self.idata.constant_data
                if hasattr(self.idata, "constant_data")
                else None
            ),
        )
        cache = self.model._incrementality_evaluator_cache()
        lease = cache.lease(self._evaluator_cache_key(response_vars), build_evaluator)
        evaluator = lease.evaluator
        # A hit carries the draws of the request that built it; a miss was just
        # conditioned on this request's posterior, so rebinding it would copy the
        # same draws a second time.
        try:
            if lease.cached:
                evaluator.set_posterior(posterior_for_evaluator)

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
            channel_data = self._channel_data_in_model_order()
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
        except BaseException:
            # The request never got a prepared evaluation to release from, so the
            # evaluator goes back here rather than being dropped with the frame.
            lease.release()
            raise

        return ProbedReach(
            lease=lease,
            baseline=baseline,
            baseline_array=baseline_array,
            probe=probe,
            reach=reach,
            effect_names=effect_names,
            trailing_lags=trailing_lags,
            channel_data_dims=channel_data_dims,
        )

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
        posterior: xr.Dataset | None = None,
        validate_dedicated_rows: Callable[[bool], None] | None = None,
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
        posterior : xr.Dataset, optional
            A posterior the caller has already subsampled, used as is:
            ``num_samples`` and ``random_state`` are then ignored.  For a caller
            that needs the very same draws for a second computation, since
            drawing twice from a ``Generator`` gives two different subsets.
        validate_dedicated_rows : callable, optional
            Called with whether the model needs a dedicated counterfactual row
            per channel (a channel-dependent ``mu_effect``, or a media transform
            the probe measured to mix channels or could not measure), before
            any period is evaluated.  For results that only make sense when each
            channel's column depends on that channel's own spend alone.

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
        prepared = self._prepare_evaluation(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            num_samples=num_samples,
            random_state=random_state,
            central_tendency=central_tendency,
            counterfactual_spend_factor=counterfactual_spend_factor,
            include_carryover=include_carryover,
            validate_reach=validate_reach,
            posterior=posterior,
        )
        # The evaluator is held until the increments it was leased for are
        # computed, and returned also when that raises.
        try:
            posterior_sub = prepared.posterior
            n_chains = posterior_sub.sizes["chain"]
            n_draws = posterior_sub.sizes["draw"]
            reducer = prepared.reducer
            effects = prepared.effects
            freq_offset = prepared.freq_offset
            windows = prepared.windows
            probed = prepared.probed
            evaluator = probed.evaluator
            baseline = probed.baseline
            baseline_array = probed.baseline_array
            probe = probed.probe
            reach = probed.reach
            effect_names = probed.effect_names
            trailing_lags = probed.trailing_lags
            channel_data_dims = probed.channel_data_dims

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
            if validate_dedicated_rows is not None:
                validate_dedicated_rows(needs_dedicated_rows)

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

            counterfactual = evaluator.evaluate_counterfactual(
                scenarios, windows=windows
            )
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
        finally:
            prepared.release()

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
        coords: Mapping[str, np.ndarray],
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
        coords : mapping
            The labels of the non-date dimensions in the order the evaluator
            computed them, the model's.  The change carries them so that xarray
            pairs it with each reducer input by label: the target scale,
            ``mu_original_scale`` and the mean-scale correction are read from
            ``idata`` in whatever order it stores them.  A dimension without an
            entry stays unlabeled.
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
            Dimensions ``("sample", "date", *custom_dims)``, labeled with
            *coords*.
        """

        def delta(name: str) -> xr.DataArray:
            dims = non_date_dims[name]
            return xr.DataArray(
                counterfactual[name][row][:, cf_mask] - baseline[name],
                dims=("sample", "date", *dims),
                coords={
                    "date": window_dates,
                    **{dim: coords[dim] for dim in dims if dim in coords},
                },
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

    def _channel_data_in_model_order(self) -> xr.DataArray:
        """Return the channel data with each labeled dim in the model's coord order.

        The draws are read in the model's order, and a reloaded model's coords
        can be sorted while ``constant_data`` keeps the order of the ``xr.Dataset``
        it was built from.  The baseline is evaluated on this array, so its
        positions pair with the draws'.
        """
        channel_data = self.data.get_channel_data()
        name = channel_data.name or "channel_data"
        ordered = reorder_to_model_coords(
            channel_data.to_dataset(name=name), self.model.model.coords
        )
        return ordered[name]

    def _posterior_in_data_order(self, posterior: xr.Dataset) -> xr.Dataset:
        """Return ``posterior`` laid out as PyMC reads it next to ``constant_data``.

        The closed form and the allocation shares evaluate the saturation and
        the adstock kernel through PyMC, which pairs the draws with the channel
        data by position, so the two must share one label order.  PyMC also
        reads each parameter's values in storage order against the dims the
        transformation's priors declare, which are the dims the model was built
        with, so each variable is put in the model's dim order too.  A
        posterior stored otherwise, such as a reordered, transposed or reloaded
        ``idata``, is laid out here.  The evaluator binds its own copy by dim
        name in the model's order, and the reducer pairs by label.
        """
        channel_data = self.data.get_channel_data()
        in_data_order = reorder_to_model_coords(
            posterior,
            {
                str(dim): channel_data.coords[dim].values
                for dim in channel_data.dims
                if dim != "date" and dim in channel_data.coords
            },
        )
        return transpose_to_model_dims(
            in_data_order, self.model.model.named_vars_to_dims
        )

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
        dates = self.data.dates
        custom_dims = list(self.model.dims)
        # The evaluator computed every node in the model's label order, so the
        # change in the linear predictor is labeled with the model's coords:
        # the channel data the baseline was evaluated on is in that order.  The
        # reducer's inputs come from ``idata`` in whatever order it stores them,
        # and xarray pairs them with the change by label.  The result then takes
        # its labels from the reduced increment itself, never by position, and
        # the custom dims go back to the order of ``constant_data`` below, which
        # is the order callers see.
        channel_data = self._channel_data_in_model_order()
        model_coords = {
            dim: channel_data.coords[dim].values for dim in ["channel", *custom_dims]
        }
        channels = model_coords["channel"].tolist()
        data_order = self.data.get_channel_data()
        reported_order = {dim: data_order.coords[dim].values for dim in custom_dims}
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
                    coords=model_coords,
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
                # A labeled product keeps its first operand's order, which may
                # be the reducer input's, so the labels are read off the result.
                **{dim: increment.coords[dim].values for dim in out_dims},
            }

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
            results.append(period_result.sel(reported_order))

        return results

    @staticmethod
    def _tail_dates(
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
        tail = Incrementality._tail_dates(
            window=window, dates=dates, tail_lags=tail_lags, freq_offset=freq_offset
        )
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
        out_dims = tuple(map(str, template.dims[3:]))
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
        return Incrementality._matrix_from_arrays(
            values,
            observed,
            realization=realization,
            windows=windows,
            coords=coords,
            out_dims=out_dims,
            frequency=frequency,
        )

    @staticmethod
    def _matrix_from_arrays(
        values: np.ndarray,
        observed: np.ndarray,
        *,
        realization: pd.DatetimeIndex,
        windows: Sequence[PeriodWindow],
        coords: dict,
        out_dims: Sequence[str],
        frequency: Frequency,
    ) -> xr.DataArray:
        """Wrap a spend x realization array in the carryover matrix's coordinates.

        Both routes build the matrix through this, so their layouts match: the
        pipeline fills the array from per-period bands, the closed form from the
        adstock kernel.

        Parameters
        ----------
        values : np.ndarray
            The entries, ``(chain, draw, spend_date, realization_date, *out_dims)``.
        observed : np.ndarray
            ``(spend_date, realization_date)``; ``False`` where the data ends
            before the entry's date.
        realization : pd.DatetimeIndex
            The columns.
        windows : sequence of PeriodWindow
            One per row, in row order, for the rows' labels.
        coords : dict
            Coordinates of ``chain``, ``draw`` and *out_dims*.
        out_dims : sequence of str
            Dimensions after ``realization_date``.
        frequency : Frequency
            Aggregation of the spend periods; ``"all_time"`` drops ``spend_date``.

        Returns
        -------
        xr.DataArray
            Dims ``(chain, draw, spend_date, realization_date, *out_dims)`` with
            ``period_start`` and ``observed`` coordinates.
        """
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
                freq=infer_frequency(self.data.dates),
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
