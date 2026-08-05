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
"""Shared oracles and probe helpers for the incrementality test modules.

The tests for :mod:`pymc_marketing.mmm.incrementality`,
:mod:`pymc_marketing.mmm.counterfactual` and
:mod:`pymc_marketing.mmm.spend_reach` all compare the modules' vectorized
evaluation against brute-force ``sample_posterior_predictive`` counterfactuals
that share no code with them.  Those oracles live here so each test module can
import the same reference implementation instead of repeating it.
"""

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pymc_marketing.mmm.counterfactual import CounterfactualEvaluator
from pymc_marketing.mmm.spend_reach import (
    CHANNEL_CONTRIBUTION,
    SpendProbe,
    resolve_channel_dependent_effects,
)
from pymc_marketing.model_graph import deterministics_to_flat


def evaluate_under_spend(mmm, channel_data_values, var_name):
    """Evaluate ``var_name`` for given channel_data using sample_posterior_predictive.

    Uses the standard PyMC evaluation path (completely independent from
    extract_response_distribution + vectorize_graph) as an oracle.
    """
    names = mmm.frozen_deterministics
    if names:
        model = deterministics_to_flat(mmm.model, names=names)
    else:
        model = mmm.model.copy()
    with model:
        pm.set_data(
            {
                "channel_data": channel_data_values.astype(
                    model["channel_data"].type.dtype
                )
            }
        )
        result = pm.sample_posterior_predictive(
            mmm.idata,
            var_names=[var_name],
        )
    return result.posterior_predictive[var_name]


def evaluate_channel_contribution(mmm, channel_data_values, original_scale=False):
    """Evaluate channel_contribution for given channel_data (identity-link oracle)."""
    var_name = (
        "channel_contribution_original_scale"
        if original_scale
        else "channel_contribution"
    )
    return evaluate_under_spend(mmm, channel_data_values, var_name)


def compute_ground_truth_incremental_by_period(
    mmm,
    frequency="all_time",
    counterfactual_spend_factor=0.0,
    include_carryover=True,
):
    """Compute ground truth incremental contribution per period using the oracle.

    For each period defined by *frequency*, creates a **separate** counterfactual
    where only that period's spend is modified (all other periods keep actual
    spend), evaluates using ``sample_posterior_predictive`` (the oracle), and
    sums the difference over the appropriate evaluation window.

    This mirrors the logic of ``compute_incremental_contribution()`` which
    processes each period independently with its own counterfactual, and serves
    as a reference implementation that is completely independent of the
    vectorized graph path.

    Parameters
    ----------
    mmm : MMM
        Fitted MMM model.
    frequency : str
        One of ``"original"``, ``"monthly"``, ``"all_time"``, etc.
    counterfactual_spend_factor : float
        Factor applied to the target period's spend (``0.0`` = zero-out).
    include_carryover : bool
        Whether to include adstock carryover effects (both carry-in and
        carry-out).

    Returns
    -------
    xr.DataArray
        Ground truth incremental contribution with dimensions matching
        ``compute_incremental_contribution`` output:
        ``(chain, draw, date, channel, *custom_dims)`` or
        ``(chain, draw, channel, *custom_dims)`` for ``"all_time"``.
    """
    actual_data = mmm.model["channel_data"].get_value()
    dates = pd.to_datetime(mmm.idata.fit_data.date.values)

    # pm.set_data cannot accept float when the model's channel_data is integer.
    # Fractional factors (e.g. 1.01) produce float values that would truncate.
    if counterfactual_spend_factor not in (0.0, 1.0) and np.issubdtype(
        actual_data.dtype, np.integer
    ):
        raise ValueError(
            f"counterfactual_spend_factor={counterfactual_spend_factor} produces "
            "fractional values, but the model's channel_data has integer dtype. "
            "pm.set_data rejects float for integer shared variables. Use a model "
            "fit with float channel_data (e.g. simple_fitted_mmm) for "
            "marginal incrementality ground truth."
        )

    incr = mmm.incrementality
    periods = incr._create_period_groups(dates[0], dates[-1], frequency)
    l_max = mmm.adstock.l_max
    inferred_freq = pd.infer_freq(dates)

    # Evaluate baseline once (reused for all periods), always in original scale
    baseline_contrib = evaluate_channel_contribution(
        mmm, actual_data, original_scale=True
    )

    period_results = []
    for t0, t1 in periods:
        # Create counterfactual: only modify spend in [t0, t1]
        target_mask = (dates >= t0) & (dates <= t1)
        cf_data = actual_data.copy()
        cf_data[target_mask] = actual_data[target_mask] * counterfactual_spend_factor

        cf_contrib = evaluate_channel_contribution(mmm, cf_data, original_scale=True)

        # Sign convention
        if counterfactual_spend_factor > 1.0:
            diff = cf_contrib - baseline_contrib
        else:
            diff = baseline_contrib - cf_contrib

        # Determine evaluation window for summing
        if include_carryover:
            carryout_end = t1 + l_max * pd.tseries.frequencies.to_offset(inferred_freq)
            eval_mask = (dates >= t0) & (dates <= carryout_end)
        else:
            eval_mask = (dates >= t0) & (dates <= t1)

        # Sum over evaluation window
        period_incr = diff.sel(date=dates[eval_mask]).sum(dim="date")
        # Shape: (chain, draw, channel, *custom_dims)

        # Assign period label and expand date dim
        period_incr = period_incr.assign_coords(date=t1).expand_dims("date")
        period_results.append(period_incr)

    return _format_ground_truth(period_results, frequency)


def _format_ground_truth(period_results, frequency, joint=False):
    """Concatenate per-period ground truth and match the module's dimension order."""
    if frequency == "all_time":
        result = period_results[0].squeeze("date", drop=True)
    else:
        result = xr.concat(period_results, dim="date")

    # Standard dimension order: (chain, draw, [date,] [channel,] *custom_dims)
    channel_dims = [] if joint else ["channel"]
    core_dims = ["chain", "draw", *channel_dims]
    extra_dims = [d for d in result.dims if d not in [*core_dims, "date"]]

    date_dims = [] if frequency == "all_time" else ["date"]
    return result.transpose("chain", "draw", *date_dims, *channel_dims, *extra_dims)


def compute_log_link_ground_truth_by_period(
    mmm,
    frequency="all_time",
    counterfactual_spend_factor=0.0,
    include_carryover=True,
    l_max=None,
    joint=False,
    start_date=None,
    end_date=None,
):
    """Ground truth incremental contribution for a multiplicative (log-link) model.

    Deliberately brute force, and deliberately **per channel**: under a log
    link the response is multiplicative, so channel *m*'s increment cannot be
    read off a single all-channels counterfactual the way it can under an
    identity link.  Each ``(period, channel)`` pair therefore gets its own
    counterfactual, evaluated on the model's response-scale prediction
    ``{output_var}_original_scale`` through ``sample_posterior_predictive``.

    Shares no code with the incrementality module -- in particular it never
    forms a linear-predictor difference -- so it is an independent check of
    both the algebra and the carryover window.

    Parameters
    ----------
    mmm : MMM
        Fitted MMM built with ``link="log"``.
    frequency : str
        One of ``"original"``, ``"monthly"``, ``"all_time"``, etc.
    counterfactual_spend_factor : float
        Factor applied to the target period's spend (``0.0`` = zero-out).
    include_carryover : bool
        Whether to include adstock carryover effects.
    l_max : int, optional
        Carry-out length of the evaluation window.  Defaults to the model's own
        ``adstock.l_max``; a mediated effect that chains a second adstock needs a
        longer window, and the module's
        :attr:`~pymc_marketing.mmm.spend_reach.SpendReach.effective_l_max` is
        what it has to agree with.
    joint : bool, default False
        Perturb every channel in one counterfactual and return a single number
        per period, matching
        :meth:`Incrementality.compute_joint_incremental_contribution`, instead of
        one leave-one-out counterfactual per channel.
    start_date, end_date : str or pd.Timestamp, optional
        Date range the periods are built over, matching the module's
        ``start_date``/``end_date`` parameters.  Default to the fitted range.

    Returns
    -------
    xr.DataArray
        Ground truth with dimensions matching
        ``compute_incremental_contribution`` output.
    """
    actual_data = mmm.model["channel_data"].get_value()
    dates = pd.to_datetime(mmm.idata.fit_data.date.values)
    channels = list(mmm.channel_columns)
    response_var = f"{mmm.output_var}_original_scale"

    # Do not assume channel sits on axis 1: panel models lay channel_data out as
    # (date, *custom_dims, channel), so the axis has to be looked up.
    channel_axis = list(mmm.data.get_channel_data().dims).index("channel")

    start = dates[0] if start_date is None else pd.to_datetime(start_date)
    end = dates[-1] if end_date is None else pd.to_datetime(end_date)
    periods = mmm.incrementality._create_period_groups(start, end, frequency)
    if l_max is None:
        l_max = mmm.adstock.l_max
    freq_offset = pd.tseries.frequencies.to_offset(pd.infer_freq(dates))

    baseline = evaluate_under_spend(mmm, actual_data, response_var)

    period_results = []
    for t0, t1 in periods:
        target_mask = (dates >= t0) & (dates <= t1)
        if include_carryover:
            eval_mask = (dates >= t0) & (dates <= t1 + l_max * freq_offset)
        else:
            eval_mask = (dates >= t0) & (dates <= t1)

        channel_results = []
        for i, channel in enumerate(channels):
            # Perturb ONE channel; every other channel keeps actual spend.  For
            # the joint estimand, perturb them all in a single counterfactual.
            selector: list = [slice(None)] * actual_data.ndim
            selector[0] = target_mask
            if not joint:
                selector[channel_axis] = i

            cf_data = actual_data.copy()
            cf_data[tuple(selector)] = (
                actual_data[tuple(selector)] * counterfactual_spend_factor
            )
            cf = evaluate_under_spend(mmm, cf_data, response_var)

            # Sign convention matches compute_incremental_contribution.
            if counterfactual_spend_factor > 1.0:
                diff = cf - baseline
            else:
                diff = baseline - cf

            summed = diff.sel(date=dates[eval_mask]).sum(dim="date")
            if joint:
                channel_results.append(summed)
                break
            channel_results.append(summed.assign_coords(channel=channel))

        if joint:
            period_incr = channel_results[0]
        else:
            period_incr = xr.concat(channel_results, dim="channel")
        period_results.append(period_incr.assign_coords(date=t1).expand_dims("date"))

    return _format_ground_truth(period_results, frequency, joint=joint)


def source_spend(X, frequency, channels):
    """Total spend per channel per period, computed from the source frame.

    Independent of both the module and ``MMMIDataWrapper``.  A ROAS test whose
    denominator comes from ``_aggregate_spend`` -- the very call the method under
    test makes -- verifies a division and nothing else, and would pass with an
    arbitrarily wrong numerator.

    Parameters
    ----------
    X : pd.DataFrame
        The fixture's own source frame, with a ``date`` column.
    frequency : str
        ``"all_time"`` or ``"monthly"``.
    channels : sequence of str
        Channel columns, in the model's order.

    Returns
    -------
    xr.DataArray
        Dimensions ``("date", "channel")``, or ``("channel",)`` for
        ``"all_time"``.  Period labels are the period *ends*, matching
        :meth:`Incrementality._create_period_groups`.
    """
    frame = X.set_index("date")[list(channels)]
    if frequency == "all_time":
        return xr.DataArray(
            frame.to_numpy().sum(axis=0),
            dims="channel",
            coords={"channel": list(channels)},
        )
    totals = frame.groupby(frame.index.to_period("M")).sum()
    return xr.DataArray(
        totals.to_numpy(),
        dims=("date", "channel"),
        coords={
            "date": [
                period.to_timestamp(how="end").normalize() for period in totals.index
            ],
            "channel": list(channels),
        },
    )


def measure_spend_reach(mmm, counterfactual_spend_factor=0.0):
    """Return the :class:`SpendReach` the module's probe arrives at for *mmm*.

    Reassembles the pieces :meth:`Incrementality._compute_increments` puts
    together, so a test can ask what the probe concluded without inferring it from
    the numbers that come out the other end.
    """
    incr = mmm.incrementality
    effects = resolve_channel_dependent_effects(mmm)
    evaluator = CounterfactualEvaluator(
        pymc_model=mmm.model,
        posterior=incr.idata.posterior.dataset,
        response_vars=[
            CHANNEL_CONTRIBUTION,
            *(effect.contribution_var for effect in effects),
        ],
        frozen_deterministics=mmm.frozen_deterministics,
        dates=incr.data.dates,
    )
    baseline_array = incr.data.get_channel_data().values
    probe = SpendProbe(
        evaluator=evaluator,
        baseline=evaluator.evaluate_baseline(baseline_array),
        baseline_array=baseline_array,
        counterfactual_spend_factor=counterfactual_spend_factor,
    )
    return probe.measure(effects=effects, l_max=mmm.adstock.l_max)


def measure_reach(mmm, counterfactual_spend_factor=0.0):
    """Return the per-node :class:`TemporalReach` the probe measured for *mmm*."""
    return measure_spend_reach(mmm, counterfactual_spend_factor).measured


def effective_l_max(mmm):
    """Evaluation-window half-length the module will use for *mmm*.

    The oracle these tests compare against sums over the same window, so it has
    to agree with the module about the length.  Derived rather than hard-coded
    because the fixtures differ in both the model's own ``l_max`` and the
    mediator's, and a literal here would be a literal repeated eleven times; the
    two headline fixtures pin the value itself in
    ``test_the_window_length_is_a_known_number``.

    Cached on the model instance, because deriving it compiles a fresh evaluator
    and runs a full probe -- and most tests ask twice, once for the oracle's
    window and once inside the module call under test.
    """
    cached = getattr(mmm, "_tests_effective_l_max", None)
    if cached is None:
        cached = measure_spend_reach(mmm).effective_l_max
        mmm._tests_effective_l_max = cached
    return cached


def mediated_identity_oracle(mmm, *effect_vars, l_max=None):
    """Per-channel mediated increment for an additive model, read off the graph.

    Under ``link="identity"`` the increment is ``target_scale`` times the summed
    change in the linear predictor, so the mediated part is available directly
    from each effect's own contribution deterministic -- no response-scale
    counterfactual needed.  Every evaluation goes through
    ``sample_posterior_predictive``, so this shares no code with the module.

    Parameters
    ----------
    mmm : MMM
        Fitted MMM built with ``link="identity"``.
    *effect_vars : str
        Contribution variables of the mediated effects to include.
    l_max : int, optional
        Carry-out length; defaults to the reconciled reach the module uses.

    Returns
    -------
    xr.DataArray
        All-time increment with dimensions ``(chain, draw, channel, *dims)``.
    """
    dates = pd.to_datetime(mmm.idata.fit_data.date.values)
    actual = mmm.model["channel_data"].get_value()
    target_scale = mmm.idata.constant_data["target_scale"].squeeze(drop=True)
    if l_max is None:
        l_max = effective_l_max(mmm)
    freq_offset = pd.tseries.frequencies.to_offset(pd.infer_freq(dates))
    eval_dates = dates[dates <= dates[-1] + l_max * freq_offset]
    # Panel models lay channel_data out as (date, *custom_dims, channel), so the
    # axis a per-channel counterfactual zeroes has to be looked up, not assumed.
    channel_axis = list(mmm.data.get_channel_data().dims).index("channel")

    def predictor(channel_data):
        return (
            evaluate_under_spend(mmm, channel_data, "channel_contribution"),
            [evaluate_under_spend(mmm, channel_data, var) for var in effect_vars],
        )

    base_channel, base_effects = predictor(actual)

    expected = []
    for idx, channel in enumerate(mmm.channel_columns):
        cf_data = actual.copy()
        selector: list = [slice(None)] * actual.ndim
        selector[channel_axis] = idx
        cf_data[tuple(selector)] = 0.0
        cf_channel, cf_effects = predictor(cf_data)

        delta = base_channel.isel(channel=idx, drop=True) - cf_channel.isel(
            channel=idx, drop=True
        )
        for base, counterfactual in zip(base_effects, cf_effects, strict=True):
            delta = delta + (base - counterfactual)

        expected.append(
            (delta.sel(date=eval_dates).sum(dim="date") * target_scale).assign_coords(
                channel=channel
            )
        )

    return xr.concat(expected, dim="channel").transpose("chain", "draw", "channel", ...)
