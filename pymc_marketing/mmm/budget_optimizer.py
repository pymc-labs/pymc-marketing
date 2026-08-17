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

"""Budget optimization module.

Overview
--------

Optimize how to allocate a total budget across channels (and optional extra dims) to
maximize an expected response derived from a fitted MMM posterior.

Quickstart (multi‑dimensional MMM)
----------------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    import xarray as xr
    from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
    from pymc_marketing.mmm.mmm import (
        MMM,
        BudgetOptimizerWrapper,
    )

    # 1) Fit a model (toy example)
    X = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=30, freq="W-MON"),
            "geo": np.random.choice(["A", "B"], size=30),
            "C1": np.random.rand(30),
            "C2": np.random.rand(30),
        }
    )
    y = pd.Series(np.random.rand(30), name="y")

    mmm = MMM(
        date_column="date",
        dims=("geo",),
        channel_columns=["C1", "C2"],
        target_column="y",
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
    )
    mmm.fit(X, y)

    # 2) Wrap the fitted model for allocation over a future window
    wrapper = BudgetOptimizerWrapper(
        model=mmm,
        start_date=X["date"].max() + pd.Timedelta(weeks=1),
        end_date=X["date"].max() + pd.Timedelta(weeks=8),
    )

    # Optional: choose which (channel, geo) cells to optimize
    budgets_to_optimize = xr.DataArray(
        np.array([[True, False], [True, True]]),
        dims=["channel", "geo"],
        coords={"channel": ["C1", "C2"], "geo": ["A", "B"]},
    )

    # Optional: distribute each cell's budget over the time window (must sum to 1 along date)
    dates = pd.date_range(wrapper.start_date, wrapper.end_date, freq="W-MON")
    factors = xr.DataArray(
        np.vstack(
            [
                np.full(len(dates), 1 / len(dates)),  # C1: uniform
                np.linspace(0.7, 0.3, len(dates)),  # C2: front‑to‑back taper
            ]
        ),
        dims=["channel", "date"],
        coords={"channel": ["C1", "C2"], "date": np.arange(len(dates))},
    )

    # 3) Optimize
    optimal, res = wrapper.optimize_budget(
        budget=100.0,
        budgets_to_optimize=budgets_to_optimize,
        budget_distribution_over_period=factors,
        response_variable="total_media_contribution_original_scale",
    )
    # `optimal` is an xr.DataArray with dims (channel, geo)

Using cost_per_unit (non-monetary channels)
-------------------------------------------

When channels are measured in non-monetary units (impressions, clicks, GRPs),
pass ``cost_per_unit`` so the optimizer converts dollar budgets into the
model's native units internally.  All user-facing inputs and outputs remain
in monetary units.

.. code-block:: python

    # cost_per_unit DataFrame (xr.DataArray) for the optimisation window.
    # Rows = dates in the future window; columns = channels with $/unit rates.
    # Channels absent from the DataFrame default to 1.0 (already in spend units).
    cpu_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-03-03", periods=8, freq="W-MON"),
            "C1": [0.05] * 8,  # $0.05 per impression
            "C2": [1.20] * 8,  # $1.20 per click
        }
    )

    optimal, res = wrapper.optimize_budget(
        budget=100.0,
        cost_per_unit=cpu_df,
    )
    # `optimal` budgets are in dollars.  The optimizer divided by
    # cost_per_unit internally before feeding into the model.

Use a custom pymc model with any dimensionality
-----------------------------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    import pymc as pm
    import xarray as xr
    from pymc.model.fgraph import clone_model
    from pymc_marketing.mmm.budget_optimizer import (
        BudgetOptimizer,
        optimizer_xarray_builder,
    )

    # 1) Build and fit any PyMC model that exposes:
    #    - a variable named 'channel_data' with dims ("date", "channel", ...)
    #    - a deterministic named 'total_media_contribution_original_scale' (scalar)
    #    - optionally a deterministic named 'channel_contribution' with dims ("date", "channel", ...)
    #      so the optimizer can auto-detect optimizable cells; otherwise pass budgets_to_optimize.

    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=30, freq="W-MON")
    channels = ["C1", "C2", "C3"]
    X = rng.uniform(0.0, 1.0, size=(len(dates), len(channels)))
    true_beta = np.array([0.8, 0.4, 0.2])
    y = (X @ true_beta) + rng.normal(0.0, 0.1, size=len(dates))

    coords = {"date": dates, "channel": channels}
    with pm.Model(coords=coords) as train_model:
        pm.Data("channel_data", X, dims=("date", "channel"))
        beta = pm.Normal("beta", 0.0, 1.0, dims="channel")
        channel_contrib = train_model["channel_data"] * beta
        mu = channel_contrib.sum(axis=-1)  # sum over channel axis
        # Per-period contribution
        pm.Deterministic("total_contribution_per_period", mu, dims="date")
        # For optimization: sum over all dimensions to get a scalar
        pm.Deterministic("total_media_contribution_original_scale", mu.sum(), dims=())
        pm.Deterministic(
            "channel_contribution",
            channel_contrib,
            dims=("date", "channel"),
        )
        sigma = pm.HalfNormal("sigma", 0.2)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="date")

        idata = pm.sample(100, tune=100, chains=2, random_seed=1)


    # 2) Create a minimal wrapper satisfying OptimizerCompatibleModelWrapper
    wrapper = CustomModelWrapper(base_model=train_model, idata=idata, channels=channels)

    # 3) Optimize N future periods with optional bounds and/or masks
    optimizer = BudgetOptimizer(model=wrapper, num_periods=8)

    # Optional: bounds per channel (single budget dim, using dict)
    bounds = {"C1": (0.0, 50.0), "C2": (0.0, 40.0), "C3": (0.0, 60.0)}

    # Or as an xarray when you have multiple budget dims, e.g. (channel, geo):
    # bounds = optimizer_xarray_builder(
    #     value=np.array([[0.0, 50.0], [0.0, 40.0], [0.0, 60.0]]),
    #     channel=channels,
    #     bound=["lower", "upper"],
    # )

    allocation, result = optimizer.allocate_budget(
        total_budget=100.0, budget_bounds=bounds
    )
    # allocation is an xr.DataArray with dims inferred from your model's channel_data dims (excluding date)

Requirements
------------

- Pass a ``pm.Model`` and ``xr.DataTree`` directly to :class:`BudgetOptimizer`.
  The model must expose a ``channel_data`` variable (or whatever ``channel_data_var`` names)
  with dims including ``"date"`` and all budget dims (e.g., ``("channel", "geo")``).
  The optimizer replaces ``channel_data`` with the optimization variable under the hood.
- Posterior must contain a response variable (default: ``"total_media_contribution_original_scale"``)
  or any custom ``response_variable`` you pass, and the required MMM deterministics
  (e.g. ``channel_contribution``).
- For time distribution: pass a DataArray with dims ``("date", *budget_dims)`` and values along
  ``date`` summing to 1 for each budget cell.
- Bounds can be a dict only for single‑dimensional budgets; otherwise use an
  xarray.DataArray (use ``optimizer_xarray_builder(...)``).
- For backward compatibility, pass a legacy wrapper (implementing
  ``OptimizerCompatibleModel``) as ``model=``, the optimizer will unpack it
  automatically.

Notes
-----
- If ``budgets_to_optimize`` is not provided, the optimizer auto‑detects cells with
  historical information using
  ``idata.posterior.channel_contribution.mean(("chain","draw","date")).astype(bool)``.
- Default bounds are ``[0, total_budget]`` on each optimized cell.
- ``allocate_budget(...)`` returns a :class:`BudgetOptimizationResult`. It unpacks as
  ``optimal, res = optimizer.allocate_budget(...)`` for backward compatibility; set
  ``callback=True`` to additionally populate ``result.callback_info`` with per‑iteration
  diagnostics (objective, gradient, constraints) for monitoring.
"""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
    runtime_checkable,
)

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import xarray as xr
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    PrivateAttr,
    model_validator,
)
from pymc import Model, do
from pymc.model.fgraph import clone_model
from pymc.model.transform.optimization import freeze_dims_and_data
from pytensor import function
from pytensor.compile.sharedvalue import SharedVariable, shared
from pytensor.graph import rewrite_graph
from pytensor.graph.traversal import ancestors
from pytensor.xtensor.type import XTensorVariable
from scipy.optimize import OptimizeResult, minimize
from xarray import DataArray, DataTree

from pymc_marketing.mmm.constraints import (
    Constraint,
    build_default_sum_constraint,
    compile_constraints_for_scipy,
)
from pymc_marketing.mmm.optimization_variables import (
    FLAT_DIM,
    LeverVariable,
    MediaVariable,
    OptimizationVariables,
    align_to_model_coords,
)
from pymc_marketing.mmm.utility import UtilityFunctionType, average_response
from pymc_marketing.pytensor_utils import merge_models
from pymc_marketing.version import __version__

# Delayed import inside methods to avoid circular dependency on pytensor_utils


class ConstraintIterationInfo(TypedDict):
    """Constraint state at one solver iteration."""

    type: str
    value: float | np.ndarray
    jac: np.ndarray | None


class OptimizationIterationInfo(TypedDict):
    """Solver state at one iteration, recorded when ``callback=True``."""

    x: np.ndarray
    fun: float
    jac: np.ndarray
    constraint_info: NotRequired[list[ConstraintIterationInfo]]


@dataclass
class BudgetOptimizationResult:
    """Result of :meth:`BudgetOptimizer.allocate_budget`.

    Iterating the result yields ``(budgets, scipy_result)``, so the
    long-standing two-element unpacking keeps working unchanged::

        optimal_budgets, scipy_result = optimizer.allocate_budget(...)

    Attribute access is the recommended interface going forward.

    Attributes
    ----------
    budgets : xarray.DataArray
        The optimized budget allocation in monetary units, labelled with the
        model's budget dims and coords.
    scipy_result : scipy.optimize.OptimizeResult
        The raw scipy optimization result (solver diagnostics, ``x``, ``fun``,
        convergence status).
    optimized_vars : dict[str, xarray.DataArray]
        Optimal values of any non-media decision variables co-optimized
        alongside the media budgets. Empty until such variables are declared
        (see https://github.com/pymc-labs/pymc-marketing/pull/2621).
    callback_info : list[OptimizationIterationInfo] or None
        Per-iteration diagnostics (``x``, ``fun``, ``jac``, constraint values)
        when ``allocate_budget(callback=True)``; ``None`` otherwise.
    """

    budgets: DataArray
    scipy_result: OptimizeResult
    optimized_vars: dict[str, DataArray] = field(default_factory=dict)
    callback_info: list[OptimizationIterationInfo] | None = None

    def __iter__(self):
        """Yield ``(budgets, scipy_result)`` for two-element unpacking."""
        yield self.budgets
        yield self.scipy_result


def optimizer_xarray_builder(value, **kwargs):
    """Create an xarray.DataArray with flexible dimensions and coordinates.

    Parameters
    ----------
    value : array-like
        The data values for the DataArray. Shape must match the dimensions
        implied by the kwargs.
    **kwargs
        Key-value pairs representing dimension names and their corresponding
        coordinates.

    Returns
    -------
    xarray.DataArray
        The resulting DataArray with the specified dimensions and values.

    Raises
    ------
    ValueError
        If the shape of `value` doesn't match the lengths of the specified
        coordinates.

    Examples
    --------
    Create a DataArray for budget bounds with channels and bound types:

    .. code-block:: python

        bounds = optimizer_xarray_builder(
            value=np.array([[0.0, 50.0], [0.0, 40.0], [0.0, 60.0]]),
            channel=["C1", "C2", "C3"],
            bound=["lower", "upper"],
        )

    Create a DataArray for budget allocation with channels and regions:

    .. code-block:: python

        allocation = optimizer_xarray_builder(
            value=np.array([[10.0, 20.0], [15.0, 25.0], [30.0, 45.0]]),
            channel=["C1", "C2", "C3"],
            region=["North", "South"],
        )
    """
    # Extract the dimensions and coordinates
    dims = list(kwargs.keys())
    coords = {dim: kwargs[dim] for dim in dims}

    # Validate the shape of `value`
    expected_shape = tuple(len(coords[dim]) for dim in dims)
    if np.shape(value) != expected_shape:
        raise ValueError(
            f"""The shape of 'value' {np.shape(value)} does not match the expected shape {expected_shape},
            based on the provided dimensions.
            """
        )

    return xr.DataArray(value, coords=coords, dims=dims)


class MinimizeException(Exception):
    """Custom exception for optimization failure."""

    def __init__(self, message: str):
        super().__init__(message)


def _extract_dataset(node: Any, group: str) -> xr.Dataset:
    """Extract an xr.Dataset for a group from either DataTree or InferenceData."""
    child = node[group]
    if isinstance(child, xr.Dataset):
        return child
    if hasattr(child, "to_dataset"):
        return child.to_dataset()
    return child


def _to_datatree(idata: Any) -> DataTree:
    """Convert InferenceData to DataTree, returning DataTree as-is."""
    if isinstance(idata, DataTree):
        return idata
    groups = {}
    for group in idata.groups():
        groups[group] = getattr(idata, group)
    return DataTree.from_dict(groups)


def merge_inference_data(
    idatas: Sequence[DataTree],
    prefixes: Sequence[str] | None = None,
    *,
    merge_on: str | None = "channel_data",
    use_every_n_draw: int = 1,
) -> DataTree:
    """Merge multiple :class:`xarray.DataTree` objects with per-model prefixes.

    This is the companion to :func:`~pymc_marketing.pytensor_utils.merge_models`
    for the inference-data side of a multi-model budget optimization.  After
    calling both functions you have a merged ``pm.Model`` and a merged
    ``DataTree`` that can be passed directly to
    :class:`BudgetOptimizer`.

    Parameters
    ----------
    idatas : list[xarray.DataTree]
        Posterior samples from each fitted model.  All objects must have
        a ``posterior`` group.
    prefixes : list[str] or None, optional
        Per-model prefix applied to every variable and dimension name that
        is not in ``merge_on``.  If ``None`` (default), prefixes are
        auto-generated as ``["model1", "model2", ...]``.
    merge_on : str or None, optional
        Variable name (and its associated dimensions) that is *shared*
        across all models and therefore **not** prefixed.  Typically
        ``"channel_data"`` so the shared budget variable remains
        unprefixed.  Pass ``None`` to prefix every variable.
    use_every_n_draw : int, optional
        Thinning factor, keeps every *n*-th posterior draw before
        merging.  Useful when merging many models to keep memory usage
        manageable.  Defaults to ``1`` (no thinning).

    Returns
    -------
    xarray.DataTree
        A single merged ``DataTree`` with prefixed variables and
        dimensions ready for use as the ``idata`` argument to
        :class:`BudgetOptimizer`.

    Raises
    ------
    ValueError
        If ``prefixes`` is provided but its length does not match
        ``len(idatas)``.

    Examples
    --------
    Merge two fitted MMMs for a multi-region optimization:

    .. code-block:: python

        from pymc_marketing.pytensor_utils import merge_models
        from pymc_marketing.mmm.budget_optimizer import (
            merge_inference_data,
            BudgetOptimizer,
        )

        # Step 1 – build per-model optimization models
        m1 = mmm_north.create_optimization_model("2025-01-01", "2025-03-31")
        m2 = mmm_south.create_optimization_model("2025-01-01", "2025-03-31")

        # Step 2 – merge PyMC models (from pytensor_utils)
        merged_model = merge_models(
            [m1, m2], prefixes=["north", "south"], merge_on="channel_data"
        )

        # Step 3 – merge inference data
        merged_idata = merge_inference_data(
            idatas=[mmm_north.idata, mmm_south.idata],
            prefixes=["north", "south"],
            merge_on="channel_data",
            use_every_n_draw=2,
        )

        # Step 4 – optimize
        optimizer = BudgetOptimizer(
            model=merged_model,
            idata=merged_idata,
            num_periods=13,
            adstock_periods=mmm_north.adstock.l_max,
            response_variable="north_total_media_contribution_original_scale",
        )
        optimal, result = optimizer.allocate_budget(total_budget=100_000)
    """
    n = len(idatas)
    if n < 1:
        raise ValueError("Need at least 1 InferenceData object to merge.")

    if prefixes is None:
        prefixes = [f"model{i + 1}" for i in range(n)]
    elif len(prefixes) != n:
        raise ValueError(
            f"Number of prefixes ({len(prefixes)}) must match number of idatas ({n})."
        )

    def _prefix_single(idata: DataTree, prefix: str) -> DataTree:
        """Rename variables and dims in *idata* by prepending *prefix*."""
        shared_vars: set[str] = {"chain", "draw", "__obs__"}
        if merge_on:
            shared_vars.add(merge_on)

        shared_dims: set[str] = set(shared_vars)
        if merge_on:
            # The shared variable does not have to live in constant_data, so look
            # through every group we prefix. Resolving its dims only from
            # constant_data would leave them prefixed while the variable name
            # stayed shared, silently changing the merge result.
            for group in ("constant_data", "posterior", "observed_data"):
                if group not in idata:
                    continue
                merge_ds = _extract_dataset(idata, group)
                if merge_on in merge_ds:
                    shared_dims.update(str(dim) for dim in merge_ds[merge_on].dims)
                    break
            else:
                # merge_on is absent everywhere, so its dims cannot be identified
                # and will all be prefixed. Make that visible rather than silent.
                warnings.warn(
                    f"merge_on={merge_on!r} was not found in the 'constant_data', "
                    f"'posterior' or 'observed_data' groups of the idata being "
                    f"prefixed with {prefix!r}. Its dimensions cannot be identified, "
                    "so they will be prefixed even though the variable name is kept "
                    "shared. Pass merge_on=None to prefix everything, or include the "
                    "shared variable in the idata.",
                    UserWarning,
                    stacklevel=3,
                )

        # Build a new DataTree with renamed groups
        new_groups: dict[str, xr.Dataset] = {}
        for group in ("posterior", "constant_data", "observed_data"):
            if group not in idata:
                continue
            ds = _extract_dataset(idata, group)
            rename_dict: dict[str, str] = {}
            for var in ds.data_vars:
                if var not in shared_vars and not str(var).startswith(f"{prefix}_"):
                    rename_dict[str(var)] = f"{prefix}_{var}"
            for dim in ds.dims:
                if dim not in shared_dims and not str(dim).startswith(f"{prefix}_"):
                    rename_dict[str(dim)] = f"{prefix}_{dim}"
            if rename_dict:
                ds = ds.rename(rename_dict)
            new_groups[group] = ds

        return DataTree.from_dict(new_groups)

    # Thin and optionally prefix each idata
    thinned: list[DataTree] = []
    for idata_i, prefix in zip(idatas, prefixes, strict=True):
        # Thin draws from the posterior group
        posterior_ds = _extract_dataset(idata_i, "posterior")
        thinned_posterior = posterior_ds.isel(draw=slice(None, None, use_every_n_draw))

        # Build a new DataTree with thinned posterior
        new_groups: dict[str, xr.Dataset] = {"posterior": thinned_posterior}
        for group in ("constant_data", "observed_data"):
            if group in idata_i:
                new_groups[group] = _extract_dataset(idata_i, group)

        thinned_i = DataTree.from_dict(new_groups)
        if prefix:
            thinned_i = _prefix_single(thinned_i, prefix)
        thinned.append(thinned_i)

    if n == 1:
        return thinned[0]

    # Merge all thinned/prefixed objects
    merged_groups: dict[str, xr.Dataset] = {}
    for idata_i in thinned:
        for group in ("posterior", "constant_data", "observed_data"):
            if group not in idata_i:
                continue
            ds = _extract_dataset(idata_i, group)
            if group in merged_groups:
                merged_groups[group] = xr.merge([merged_groups[group], ds])
            else:
                merged_groups[group] = ds

    return DataTree.from_dict(merged_groups)


def merge_models_and_idata(
    models: Sequence[Model],
    idatas: Sequence[DataTree],
    *,
    prefixes: Sequence[str] | None = None,
    merge_on: str | None = "channel_data",
    use_every_n_draw: int = 1,
) -> tuple[Model, DataTree]:
    """Merge multiple PyMC models and their DataTree objects in one call.

    Convenience wrapper that calls
    :func:`~pymc_marketing.pytensor_utils.merge_models` and
    :func:`merge_inference_data` with the same ``prefixes`` and ``merge_on``
    arguments, returning both results as a ``(model, idata)`` tuple ready for
    :class:`BudgetOptimizer`.

    Parameters
    ----------
    models : list of pm.Model
        Optimization models, one per fitted MMM (e.g. from
        ``mmm.create_optimization_model(...)``).
    idatas : list of xarray.DataTree
        Posterior samples corresponding to each model (e.g. ``mmm.idata``).
        Must be the same length as *models*.
    prefixes : list of str or None, optional
        Per-model prefix applied to every variable and dimension name that is
        not in ``merge_on``.  If ``None`` (default), prefixes are
        auto-generated as ``["model1", "model2", ...]``.
    merge_on : str or None, optional
        Variable name that is *shared* across all models and therefore **not**
        prefixed in either the PyMC graph or the posterior.  Defaults to
        ``"channel_data"`` which is the standard budget input node used by
        :class:`BudgetOptimizer`.  Pass ``None`` to prefix every variable.
    use_every_n_draw : int, optional
        Thinning factor applied to each ``DataTree`` before merging.
        Keeps every *n*-th posterior draw.  Defaults to ``1`` (no thinning).

    Returns
    -------
    tuple of (pm.Model, xarray.DataTree)
        ``(merged_model, merged_idata)`` ready to be passed directly to
        :class:`BudgetOptimizer`.

    Raises
    ------
    ValueError
        If ``len(models) != len(idatas)``, or if ``models`` has fewer than 2
        elements, or if ``prefixes`` length does not match.

    Examples
    --------
    Merge two fitted MMMs for a multi-region optimization:

    .. code-block:: python

        from pymc_marketing.mmm import merge_models_and_idata, BudgetOptimizer

        m1 = mmm_north.create_optimization_model("2025-01-01", "2025-03-31")
        m2 = mmm_south.create_optimization_model("2025-01-01", "2025-03-31")

        merged_model, merged_idata = merge_models_and_idata(
            models=[m1, m2],
            idatas=[mmm_north.idata, mmm_south.idata],
            prefixes=["north", "south"],
            merge_on="channel_data",
            use_every_n_draw=2,
        )

        optimizer = BudgetOptimizer(
            model=merged_model,
            idata=merged_idata,
            num_periods=13,
            adstock_periods=mmm_north.adstock.l_max,
            response_variable="north_total_media_contribution_original_scale",
        )
        optimal, result = optimizer.allocate_budget(total_budget=100_000)
    """
    if len(models) != len(idatas):
        raise ValueError(
            f"len(models) ({len(models)}) must equal len(idatas) ({len(idatas)})."
        )

    merged_model = merge_models(
        list(models),
        prefixes=list(prefixes) if prefixes is not None else None,
        merge_on=merge_on,
    )
    merged_idata = merge_inference_data(
        idatas,
        prefixes=prefixes,
        merge_on=merge_on,
        use_every_n_draw=use_every_n_draw,
    )
    return merged_model, merged_idata


@runtime_checkable
class OptimizerCompatibleModel(Protocol):
    """Protocol for marketing mix model wrappers compatible with the BudgetOptimizer.

    Any object satisfying this Protocol can be passed as the ``model`` argument
    to :class:`BudgetOptimizer` for backward compatibility.

    Attributes
    ----------
    idata : xarray.DataTree or arviz.InferenceData
        Fitted posterior inference data.  Must contain a ``posterior`` group with a
        ``channel_contribution`` variable (or whatever ``response_variable`` you pass).
    """

    adstock: Any
    _channel_scales: Any
    idata: Any  # xr.DataTree or arviz.InferenceData

    def optimization_model(self, num_periods: int) -> Model:
        """Return a PyMC model configured for ``num_periods`` optimization steps.

        Parameters
        ----------
        num_periods : int
            Number of time periods to allocate budget for (excluding any
            adstock warm-up; the implementation is responsible for adding
            ``adstock_periods`` extra steps when needed).

        Returns
        -------
        pymc.Model
            A (possibly cloned) PyMC model ready for use by
            :class:`BudgetOptimizer`.
        """
        ...

    def _set_predictors_for_optimization(self, num_periods: int) -> Model:
        """Return a PyMC model (deprecated; use :meth:`optimization_model` instead)."""
        ...


# Backward-compatible alias, will be removed in the next major version.
OptimizerCompatibleModelWrapper = OptimizerCompatibleModel


def _concrete_method(model: Any, name: str) -> Callable[..., Model] | None:
    """Return ``model``'s real implementation of *name*, if it has one.

    ``OptimizerCompatibleModel`` is a Protocol whose method bodies are just ``...``,
    so a wrapper inheriting from it always *has* the attribute even when it never
    implemented one.  Checking ``type(model).__dict__`` would exclude
    inherited-but-concrete implementations, which is exactly the case for
    third-party subclasses of a wrapper.  Compare against the protocol stub
    instead: any implementation that is not the stub is a real one, wherever it
    lives in the MRO.

    Parameters
    ----------
    model : Any
        Candidate optimizer-compatible wrapper.
    name : str
        Method name to resolve, e.g. ``"optimization_model"``.

    Returns
    -------
    Callable or None
        The bound method, or ``None`` when the object only carries the inert
        protocol stub (or nothing at all).
    """
    impl = getattr(type(model), name, None)
    if impl is None or impl is getattr(OptimizerCompatibleModel, name, None):
        return None
    return getattr(model, name)


class BuildMergedModel(OptimizerCompatibleModel):
    """Merge multiple optimizer-compatible models into a single model.

    This wrapper combines several optimizer-compatible MMM wrappers by:

    - Merging their posterior ``DataTree`` with per-model prefixes
    - Optionally thinning posterior draws via ``use_every_n_draw``
    - Exposing a persistent merged PyMC ``Model`` for optimization through
      ``_set_predictors_for_optimization`` and a dynamic ``model`` property for
      inspection when needed

    Parameters
    ----------
    models : list[OptimizerCompatibleModel]
        A list of wrappers that each expose ``idata`` and
        ``optimization_model(num_periods: int) -> Model``.
    prefixes : list[str] | None, optional
        Per-model prefixes used when merging. If ``None``, defaults to
        ``["model1", "model2", ...]`` with one prefix per model.
    merge_on : str | None, optional, default "channel_data"
        Name of a variable expected to be present in all models and that should
        remain unprefixed and be used for aligning/merging dims (e.g.,
        ``"channel_data"``). If ``None``, no variable is treated as shared and
        all variables/dims are prefixed.
    use_every_n_draw : int, optional, default 1
        Thinning factor applied when merging idatas. Keeps every n-th draw.

    Attributes
    ----------
    prefixes : list[str]
        The final list of prefixes used for each model.
    models : list[OptimizerCompatibleModel]
        The provided list of wrappers.
    num_models : int
        Number of models being merged.
    num_periods : int | None
        Number of forecast periods inferred from the primary model (if available).
    idata : xr.DataTree
        The merged and prefixed posterior (and data) container.
    adstock : Any
        Carried over from the primary model when available.
    model : pymc.Model
        Property returning a merged PyMC model; see Notes.

    Examples
    --------
    Merge three multidimensional MMMs into a single optimizer model:

    .. code-block:: python

        from pymc_marketing.mmm.mmm import (
            MMM,
            BudgetOptimizerWrapper,
        )
        from pymc_marketing.mmm.budget_optimizer import (
            BuildMergedModel,
            BudgetOptimizer,
        )

        # Assume m1, m2, m3 are already fitted MMM instances
        w1 = BudgetOptimizerWrapper(model=m1, start_date=start, end_date=end)
        w2 = BudgetOptimizerWrapper(model=m2, start_date=start, end_date=end)
        w3 = BudgetOptimizerWrapper(model=m3, start_date=start, end_date=end)

        merged = BuildMergedModel(
            models=[w1, w2, w3],
            prefixes=["north", "south", "west"],
            merge_on="channel_data",
            use_every_n_draw=2,
        )

        optimizer = BudgetOptimizer(
            model=merged,
            num_periods=merged.num_periods,
            response_variable="north_total_media_contribution_original_scale",
        )

    Single model: auto-prefix and thin draws:

    .. code-block:: python

        merged_single = BuildMergedModel(
            models=[w1],
            prefixes=None,  # auto -> ["model1"]
            merge_on="channel_data",
            use_every_n_draw=5,
        )
        m_opt = merged_single.optimization_model(num_periods=merged_single.num_periods)

    Merge everything with prefixes (no shared variable retained):

    .. code-block:: python

        merged_all_prefixed = BuildMergedModel(
            models=[w1, w2],
            prefixes=["a", "b"],
            merge_on=None,  # do not keep any unprefixed variable
        )
    """

    def __init__(
        self,
        models: list[OptimizerCompatibleModel],
        prefixes: list[str] | None = None,
        merge_on: str | None = "channel_data",
        use_every_n_draw: int = 1,
    ) -> None:
        warnings.warn(
            "BuildMergedModel is deprecated and will be removed in a future version. "
            "Use merge_models_and_idata() from pymc_marketing.mmm instead, then pass "
            "the returned (model, idata) directly to BudgetOptimizer. "
            "See the BudgetOptimizer docstring for the recommended multi-model workflow.",
            DeprecationWarning,
            stacklevel=2,
        )
        if len(models) < 1:
            raise ValueError("Need at least 1 model")

        self._channel_scales = 1.0
        self.models = models
        self.num_models = len(models)

        # Auto-generate prefixes if not provided - ALL models get prefixes
        if prefixes is None:
            self.prefixes = [f"model{i + 1}" for i in range(self.num_models)]
        else:
            if len(prefixes) != len(models):
                raise ValueError(
                    f"Number of prefixes ({len(prefixes)}) must match number of models ({len(models)})"
                )
            self.prefixes = prefixes

        self.merge_on = merge_on
        self.use_every_n_draw = use_every_n_draw

        # Use first model as primary for attributes
        self.primary_model = models[0]
        self.num_periods = getattr(self.primary_model, "num_periods", None)

        # Merge idata from all models with appropriate prefixes
        self._merge_idata()

        if hasattr(self.primary_model, "adstock_periods"):
            self.adstock_periods = self.primary_model.adstock_periods
        elif hasattr(self.primary_model, "adstock"):
            self.adstock_periods = self.primary_model.adstock.l_max
        else:
            self.adstock_periods = 0

        # Persistent merged model used for optimization
        self._persistent_merged_model: Model | None = None
        self._persistent_num_periods: int | None = None

    def _merge_idata(self) -> None:
        if self.num_models == 1:
            idata = self.models[0].idata.isel(
                draw=slice(None, None, self.use_every_n_draw)
            )
            if self.prefixes[0]:
                idata = self._prefix_idata(idata, self.prefixes[0])
            self.idata = idata
            return

        merged_idata = None
        for i, model in enumerate(self.models):
            prefix = self.prefixes[i]
            idata_i = model.idata.isel(
                draw=slice(None, None, self.use_every_n_draw)
            ).copy()
            if prefix:
                idata_i = self._prefix_idata(idata_i, prefix)

            if merged_idata is None:
                merged_idata = idata_i
            else:
                for group in ("posterior", "constant_data", "observed_data"):
                    if group in idata_i:
                        if group in merged_idata:
                            merged_idata[group] = xr.merge(
                                [merged_idata[group], idata_i[group]]
                            )
                        else:
                            merged_idata[group] = idata_i[group]

        self.idata = merged_idata

    def _prefix_idata(self, idata, prefix: str):
        shared_vars = {"chain", "draw", "__obs__"}
        if self.merge_on:
            shared_vars.add(self.merge_on)

        shared_dims = set(shared_vars)
        if (
            self.merge_on
            and "constant_data" in idata
            and self.merge_on in idata.constant_data
        ):
            merge_dims = list(idata.constant_data[self.merge_on].dims)
            shared_dims.update(merge_dims)

        prefixed_idata = idata.copy()
        for group in ("posterior", "constant_data", "observed_data"):
            if group in prefixed_idata:
                rename_dict = {}
                for var in prefixed_idata[group].data_vars:
                    if var not in shared_vars and not var.startswith(f"{prefix}_"):
                        rename_dict[var] = f"{prefix}_{var}"
                for dim in prefixed_idata[group].dims:
                    if dim not in shared_dims and not dim.startswith(f"{prefix}_"):
                        rename_dict[dim] = f"{prefix}_{dim}"
                if rename_dict:
                    prefixed_idata[group] = prefixed_idata[group].dataset.rename(
                        rename_dict
                    )

        return prefixed_idata

    def optimization_model(self, num_periods: int) -> Model:
        """Return a merged PyMC model configured for ``num_periods`` optimization steps.

        Builds (or reuses a cached) merged model from all child models. Each
        child model's :meth:`optimization_model` is called; the results are
        combined with :func:`merge_models`.

        Parameters
        ----------
        num_periods : int
            Number of optimization periods (excluding adstock warm-up).

        Returns
        -------
        pymc.Model
            A persistent merged model ready for :class:`BudgetOptimizer`.
        """
        # If we already built a persistent model for this horizon, reuse it
        if (
            self._persistent_merged_model is not None
            and self._persistent_num_periods == int(num_periods)
        ):
            return self._persistent_merged_model

        # Build per-model optimization models.
        # Prefer a concrete optimization_model implementation; fall back to
        # _set_predictors_for_optimization if only the protocol stub exists.
        pymc_models = []
        for m in self.models:
            build = _concrete_method(m, "optimization_model") or _concrete_method(
                m, "_set_predictors_for_optimization"
            )
            if build is not None:
                pymc_models.append(build(num_periods=num_periods))
            else:
                raise ValueError(
                    f"Model wrapper {type(m).__name__!r} has no "
                    "optimization_model or _set_predictors_for_optimization method."
                )
        if self.num_models == 1:
            self._persistent_merged_model = freeze_dims_and_data(pymc_models[0])
        else:
            self._persistent_merged_model = merge_models(
                models=pymc_models, prefixes=self.prefixes, merge_on=self.merge_on
            )

        self._persistent_num_periods = int(num_periods)
        return self._persistent_merged_model

    def _set_predictors_for_optimization(self, num_periods: int) -> Model:
        """Return a merged PyMC model (deprecated; use :meth:`optimization_model` instead).

        .. deprecated::
            ``_set_predictors_for_optimization`` will be removed in a future
            version. :class:`BudgetOptimizer` now calls
            :meth:`optimization_model` directly.
        """
        warnings.warn(
            "_set_predictors_for_optimization is deprecated and will be removed in a "
            "future version. Use optimization_model() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.optimization_model(num_periods)

    @property
    def model(self) -> Model:
        """Return the merged PyMC model.

        If a persistent optimization model exists, return it. Otherwise, try to lazily
        construct it using the known number of periods. As a fallback, merge the
        underlying training models from each wrapper (non-persistent).
        """
        # If a persistent optimization model exists, expose it for mutation
        if self._persistent_merged_model is not None:
            return self._persistent_merged_model

        # If we know the number of periods, lazily build the persistent model now
        if self.num_periods is not None:
            return self.optimization_model(int(self.num_periods))

        # Fallback: dynamic merged training models (non-persistent)
        # Obtain each wrapper's training model dynamically; not all wrappers statically expose `.model`.
        # Cast to Any first to avoid mypy attr-defined errors for Protocol wrappers.
        pymc_models = [cast(Any, model).model for model in self.models]
        if self.num_models == 1:
            return pymc_models[0]
        return merge_models(
            models=pymc_models, prefixes=self.prefixes, merge_on=self.merge_on
        )


class BudgetOptimizer(BaseModel):
    """A class for optimizing budget allocation in a marketing mix model.

    The goal of this optimization is to maximize the total expected response
    by allocating the given budget across different marketing channels. The
    optimization is performed using the Sequential Least Squares Quadratic
    Programming (SLSQP) method, which is a gradient-based optimization algorithm
    suitable for solving constrained optimization problems.

    For more information on the SLSQP algorithm, refer to the documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Parameters
    ----------
    model : pm.Model
        The PyMC model configured for the optimization horizon. The model must
        contain a ``pm.Data`` variable named ``channel_data_var`` (default
        ``"channel_data"``) whose dims include the channel and, optionally,
        additional dimensions (e.g. geo).
    idata : xarray.DataTree or arviz.InferenceData
        Fitted posterior inference data from the model.
    num_periods : int
        Number of time units at the desired time granularity to allocate budget for.
    adstock_periods : int, optional
        Number of extra warm-up periods prepended for adstock carryover.
        Equivalent to ``adstock.l_max`` on the built-in MMM. Defaults to 0.
    channel_scales : float or array-like, optional
        Per-channel scale factors used to convert monetary budgets into the
        model's native units. A scalar ``1.0`` means no scaling. Defaults to 1.0.
    mu_effects : Sequence, optional
        Additional ``mu`` effects carried over from the fitted model. Effects
        exposing ``replace_for_optimization`` are not supported yet and raise
        ``NotImplementedError``. Defaults to an empty list.
    frozen_deterministics : list of str, optional
        Names of ``Deterministic`` variables to freeze at their posterior values
        instead of recomputing them from the graph. Required for models with HSGP
        or time-varying components. Defaults to ``None``.
    response_variable : str, optional
        The response variable to optimize. Default is
        ``"total_media_contribution_original_scale"``.
    utility_function : UtilityFunctionType, optional
        The utility function to maximize. Default is the mean of the response distribution.
    budgets_to_optimize : xarray.DataArray, optional
        Mask defining a subset of budgets to optimize. Non-optimized budgets remain fixed at 0.
    constraints : Sequence[Constraint], optional
        Constraints for the optimizer. If empty, a default sum-equals-total-budget
        constraint is added automatically. If non-empty, the caller is in charge:
        no default is added. Pass ``build_default_sum_constraint()`` explicitly
        to keep the sum constraint alongside custom ones.
    budget_distribution_over_period : xarray.DataArray, optional
        Fixed temporal distribution of each budget cell across periods.
        Must have dims ``("date", *budget_dims)`` where the ``"date"``
        dim has length ``num_periods``. Values must sum to 1 along the
        ``"date"`` dim for every combination of the remaining dims
        (i.e., ``budget_distribution_over_period.sum(dim="date")`` must
        be all ones). Each value is the fraction of that cell's total
        budget assigned to the corresponding period, e.g. fractions
        ``[0.4, 0.3, 0.2, 0.1]`` along ``"date"`` mean 40 % of the
        budget in period 0, 30 % in period 1, and so on.
        If None, budget is distributed uniformly
        (``1 / num_periods`` per period).
    channel_data_var : str, optional
        Name of the ``pm.Data`` variable inside ``model`` that holds channel
        spend / media inputs. Defaults to ``"channel_data"``.
    channel_contribution_var : str, optional
        Name of the per-channel contribution variable in the posterior used
        to auto-detect non-zero channels. Defaults to ``"channel_contribution"``.
    date_dim : str, optional
        Name of the date dimension in the model. Defaults to ``"date"``.
    cost_per_unit : xarray.DataArray, optional
        Cost-per-unit conversion factors for translating monetary budgets into
        the model's native units. Must have dims ``("date", *budget_dims)``
        where ``"date"`` has length ``num_periods``. If ``None``, budgets are
        assumed to already be in the model's native units.
    compile_kwargs : dict, optional
        Extra keyword arguments forwarded to PyTensor's ``function()`` during
        compilation. Useful for setting ``mode``.

    Notes
    -----
    For backward compatibility, pass a legacy wrapper (implementing
    ``OptimizerCompatibleModel``) as ``model=``, the optimizer will
    unpack it automatically via a ``model_validator``.

    Examples
    --------
    Basic usage, pass a PyMC model and its posterior inference data directly:

    .. code-block:: python

        import pymc_marketing as pmm

        # mmm is a fitted multidimensional MMM
        pymc_model = mmm.create_optimization_model(
            start_date="2025-01-01",
            end_date="2025-03-31",
        )
        optimizer = pmm.mmm.BudgetOptimizer(
            model=pymc_model,
            idata=mmm.idata,
            num_periods=13,
            adstock_periods=mmm.adstock.l_max,
            response_variable="total_media_contribution_original_scale",
        )
        optimal, result = optimizer.allocate_budget(total_budget=100_000)
    """

    num_periods: int = Field(
        ...,
        gt=0,
        description="Number of time units at the desired time granularity to allocate budget for.",
    )

    model: InstanceOf[Model] = Field(
        ...,
        description="The PyMC model configured for the optimization horizon.",
    )

    idata: DataTree = Field(
        ...,
        description="Posterior samples from the fitted model.",
    )

    adstock_periods: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of extra carry-over periods appended to the optimization horizon. "
            "For built-in MMM this equals adstock.l_max; defaults to 0 (no carry-over)."
        ),
    )

    channel_scales: Any = Field(
        default=1.0,
        description=(
            "Per-channel scale factors used to convert monetary budgets into the model's "
            "native units. A scalar 1.0 means no scaling. A 1-D array of length n_channels "
            "applies a per-channel scale."
        ),
    )

    optimizable_vars: dict[str, Sequence[tuple[float | None, float | None]] | None] = (
        Field(
            default_factory=dict,
            description=(
                "Additional pm.Data variables to co-optimize alongside "
                "`channel_data_var`, keyed by variable name. Each value gives the "
                "native (low, high) bounds per entry in the variable's coordinate "
                "order, or None for unbounded. Each variable must have exactly one "
                "dim, which must not be the date dim, since date-varying variables "
                "are not supported. Optimal values are returned in "
                "`result.optimized_vars`. These variables are optimized in their "
                "own units and do not participate in the default budget-sum "
                "constraint, which sums the media budgets alone."
            ),
        )
    )

    response_variable: str = Field(
        default="total_media_contribution_original_scale",
        description="The response variable to optimize.",
    )

    utility_function: UtilityFunctionType = Field(
        default=average_response,
        description="Utility function to maximize.",
    )

    budgets_to_optimize: DataArray | None = Field(
        default=None,
        description="Mask defining a subset of budgets to optimize. Non-optimized budgets remain fixed at 0.",
    )

    constraints: Sequence[Constraint] = Field(
        default=(),
        description=(
            "Constraints for the optimizer. Empty means the default sum "
            "constraint is added automatically; non-empty means the caller "
            "is in charge (no default is added). Pass "
            "`build_default_sum_constraint()` explicitly to keep the sum "
            "constraint alongside custom ones."
        ),
    )

    budget_distribution_over_period: DataArray | None = Field(
        default=None,
        description=(
            "Fixed temporal distribution of each budget cell across periods. "
            "Must have dims ('date', *budget_dims) where 'date' has length num_periods. "
            "Values must sum to 1 along 'date' for every combination of the remaining dims. "
            "If None, budget is distributed uniformly (1/num_periods per period)."
        ),
    )

    cost_per_unit: DataArray | None = Field(
        default=None,
        description=(
            "Cost per unit conversion factors for converting budgets from "
            "monetary units (dollars) to original units (impressions, clicks). "
            "Must have dims (date, *budget_dims) where date has length "
            "num_periods. If None, budgets are assumed to already be in "
            "the model's native units (no conversion applied)."
        ),
    )

    compile_kwargs: dict | None = Field(
        default=None,
        description="Keyword arguments for the model compilation. Especially useful to pass compilation mode",
    )

    frozen_deterministics: list[str] | None = Field(
        default=None,
        description=(
            "Names of Deterministic variables to freeze at posterior values "
            "instead of recomputing from the graph. Required for models with "
            "HSGP or time-varying components."
        ),
    )

    channel_data_var: str = Field(
        default="channel_data",
        description=(
            "Name of the PyMC Data variable in the model that holds channel spend/media inputs. "
            "Defaults to 'channel_data'. Override if your model uses a different variable name "
            "(e.g. 'media_spend', 'x_media')."
        ),
    )

    channel_contribution_var: str = Field(
        default="channel_contribution",
        description=(
            "Name of the per-channel contribution variable in the model's posterior. "
            "Used to automatically detect which channels are non-zero when "
            "'budgets_to_optimize' is not provided. "
            "Defaults to 'channel_contribution'."
        ),
    )

    date_dim: str = Field(
        default="date",
        description=(
            "Name of the date dimension in the model. "
            "Defaults to 'date'. Override if your model uses a different dimension name."
        ),
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _total_budget: SharedVariable = PrivateAttr()
    _budget_dims: list[str] = PrivateAttr()
    _budget_coords: dict[str, list] = PrivateAttr()
    _budget_shape: tuple[int, ...] = PrivateAttr()
    _budgets_flat: XTensorVariable = PrivateAttr()
    _budgets: XTensorVariable = PrivateAttr()
    _budget_distribution_over_period_tensor: XTensorVariable | None = PrivateAttr()
    _cost_per_unit_tensor: XTensorVariable | None = PrivateAttr()
    _pymc_model: Model = PrivateAttr()
    _variables: OptimizationVariables = PrivateAttr()
    _objective_and_grad: Callable = PrivateAttr()
    _constraints: dict = PrivateAttr()
    _compiled_constraints: list[dict] = PrivateAttr()

    @model_validator(mode="before")
    @classmethod
    def _handle_legacy_model_arg(cls, data: Any) -> Any:
        """Backward compat: extract ``pm.Model`` and ``idata`` from legacy wrapper.

        If ``model=`` is passed as an OptimizerCompatibleModel wrapper,
        unpack the underlying ``pm.Model`` and ``idata`` from it.

        ``mmm.py`` still calls ``BudgetOptimizer(model=self, num_periods=..., ...)`` passing
        the MMM instance itself.  A plain ``pm.Model`` is now the canonical value for ``model``,
        so this validator checks whether the value looks like a wrapper (has
        ``optimization_model()``) and, if so, unpacks it before Pydantic validates the field.
        """
        if not isinstance(data, dict):
            return data
        idata = data.get("idata")
        if idata is not None and not isinstance(idata, DataTree):
            data["idata"] = _to_datatree(idata)
        model = data.get("model")
        if model is None:
            return data
        # If it's already a plain pm.Model, nothing to do.
        if isinstance(model, Model):
            return data
        # Legacy path: model is an OptimizerCompatibleModel wrapper.
        if not hasattr(model, "optimization_model") and not hasattr(
            model, "_set_predictors_for_optimization"
        ):
            return data
        num_periods = data.get("num_periods")
        if num_periods is None:
            raise ValueError(
                "num_periods must be provided when using the legacy model= argument"
            )
        # Prefer a concrete optimization_model implementation over the
        # protocol stub.  A real implementation lives on the class itself,
        # not inherited from the Protocol.
        build = _concrete_method(model, "optimization_model") or _concrete_method(
            model, "_set_predictors_for_optimization"
        )
        if build is not None:
            data["model"] = build(int(num_periods))
        else:
            raise ValueError(
                f"Model wrapper {type(model).__name__!r} implements neither "
                "optimization_model() nor _set_predictors_for_optimization(); "
                "it only carries the inert OptimizerCompatibleModel protocol stub. "
                "Implement optimization_model(num_periods) to return the pm.Model "
                "for the optimization horizon, or pass a pm.Model directly."
            )
        data["idata"] = _to_datatree(model.idata)
        # Infer adstock_periods from wrapper if not already set
        if "adstock_periods" not in data:
            if hasattr(model, "adstock") and hasattr(model.adstock, "l_max"):
                data["adstock_periods"] = model.adstock.l_max
            elif hasattr(model, "adstock_periods"):
                data["adstock_periods"] = model.adstock_periods
        # Infer channel_scales from wrapper if not already set
        if "channel_scales" not in data:
            if hasattr(model, "_channel_scales"):
                data["channel_scales"] = model._channel_scales
            elif hasattr(model, "channel_scales"):
                data["channel_scales"] = model.channel_scales
        # Drop mu_effects if a caller still passes it (removed field): the
        # optimizer never consumed it; non-optimizable effects are already
        # baked into the model graph by the time it is built.
        if "mu_effects" in data:
            warnings.warn(
                "BudgetOptimizer no longer accepts mu_effects; the argument is "
                "ignored. Effects are baked into the model graph at build time.",
                DeprecationWarning,
                stacklevel=2,
            )
            data.pop("mu_effects")
        if "frozen_deterministics" not in data and hasattr(
            model, "frozen_deterministics"
        ):
            data["frozen_deterministics"] = model.frozen_deterministics
        return data

    DEFAULT_MINIMIZE_KWARGS: ClassVar[dict] = {
        "method": "SLSQP",
        "options": {"ftol": 1e-9, "maxiter": 1_000},
    }

    def model_post_init(self, context: Any, /) -> None:
        """Build optimization tensors and compile objective after Field validation."""
        # 1. The model is passed in already built (no optimization_model() call needed)

        # 2. Shared variable for total_budget
        self._total_budget = shared(np.array(0.0, dtype="float64"), name="total_budget")

        # 3. Identify budget dimensions and shapes
        self._budget_dims = [
            dim
            for dim in self.model.named_vars_to_dims[self.channel_data_var]
            if dim != self.date_dim
        ]
        self._budget_coords = {
            dim: list(self.model.coords[dim]) for dim in self._budget_dims
        }
        self._budget_shape = tuple(len(coord) for coord in self._budget_coords.values())

        # 3b. Validate channel_scales here rather than letting a wrong length surface
        #     as a broadcasting error deep inside graph construction.
        self._validate_channel_scales()

        # 4. Resolve the budget mask. This is the single owner of the auto-detect
        #    rule: when no mask is given, optimize the cells the posterior says the
        #    model has information about (non-zero mean channel contribution), and
        #    fall back to all cells when that variable is absent. Adaptor layers such
        #    as MMM.budget_optimizer deliberately do not narrow the mask themselves.
        if self.budgets_to_optimize is None:
            try:
                posterior_ds = _extract_dataset(self.idata, "posterior")
            except (KeyError, TypeError):
                posterior_ds = None

            if (
                posterior_ds is not None
                and self.channel_contribution_var in posterior_ds.data_vars
            ):
                # Auto-detect non-zero channels from posterior
                self.budgets_to_optimize = (
                    posterior_ds[self.channel_contribution_var]
                    .mean(("chain", "draw", self.date_dim))
                    .astype(bool)
                )
            else:
                ones = np.ones(self._budget_shape, dtype=bool)
                self.budgets_to_optimize = xr.DataArray(
                    ones, coords=self._budget_coords, dims=self._budget_dims
                )
        else:
            # Validate user-supplied mask against posterior channel contributions
            try:
                posterior_ds = _extract_dataset(self.idata, "posterior")
            except (KeyError, TypeError):
                posterior_ds = None

            if (
                posterior_ds is not None
                and self.channel_contribution_var in posterior_ds.data_vars
            ):
                expected_mask = (
                    posterior_ds[self.channel_contribution_var]
                    .mean(("chain", "draw", self.date_dim))
                    .astype(bool)
                )
                if np.any((self.budgets_to_optimize > expected_mask).values):
                    raise ValueError(
                        "budgets_to_optimize mask contains True values at coordinates where the model has no "
                        "information."
                    )

        # Align the mask with the model's coordinate order, not just its dim
        # order: every coordinate-bearing input is consumed positionally
        # against the model's tensor layout, so all of them go through the same
        # alignment (see `align_to_model_coords`).
        self.budgets_to_optimize = (
            align_to_model_coords(
                self.budgets_to_optimize,
                self._budget_coords,
                label="budgets_to_optimize",
            )
            .transpose(*self._budget_dims)
            .astype(bool)
        )

        # 6. Validate and process budget_distribution_over_period
        self._budget_distribution_over_period_tensor = (
            self._validate_and_process_budget_distribution(
                budget_distribution_over_period=self.budget_distribution_over_period,
                num_periods=self.num_periods,
                budget_dims=self._budget_dims,
                budgets_to_optimize=self.budgets_to_optimize,
                date_dim=self.date_dim,
            )
        )

        # 6b. Validate and process cost_per_unit
        self._cost_per_unit_tensor = self._validate_and_process_cost_per_unit(
            cost_per_unit=self.cost_per_unit,
            num_periods=self.num_periods,
            budget_dims=self._budget_dims,
            budget_coords=self._budget_coords,
            date_dim=self.date_dim,
        )

        # 7. Build the optimization variables and substitute them into the
        # model graph. One do() call over every variable keeps gradients joint.
        media_variable = MediaVariable(
            name=self.channel_data_var,
            mask=self.budgets_to_optimize,
            num_periods=self.num_periods,
            adstock_periods=self.adstock_periods,
            channel_scales=self.channel_scales,
            dtype=self.model[self.channel_data_var].dtype,
            date_dim=self.date_dim,
            budget_distribution_over_period_tensor=self._budget_distribution_over_period_tensor,
            cost_per_unit_tensor=self._cost_per_unit_tensor,
        )
        # Optimizable vars become lever variables appended after media. They
        # are optimized in their own units and stay out of the budget-sum
        # constraint, which sums the media tensor alone.
        levers = [
            LeverVariable.from_model(
                self.model, lever_name, lever_bounds, date_dim=self.date_dim
            )
            for lever_name, lever_bounds in self.optimizable_vars.items()
        ]

        self._variables = OptimizationVariables([media_variable, *levers])
        self._budgets_flat = self._variables.flat
        self._budgets = media_variable.scattered(
            self._variables.variable_slice(self.channel_data_var)
        )

        self._pymc_model = do(
            freeze_dims_and_data(self.model, data=[]),
            self._variables.substitutions(),
        )

        # 8. Validate that the requested response variable actually exists in
        # the underlying PyMC model.
        if self.response_variable not in self._pymc_model.named_vars:
            available = sorted(self._pymc_model.named_vars)
            raise ValueError(
                f"response_variable={self.response_variable!r} is not in the "
                f"PyMC model. Available variables: {available}. "
                "Pass an explicit response_variable to BudgetOptimizer."
            )

        # 8b. Every lever must be able to move the response variable. A lever
        # the objective cannot reach has an identically zero gradient, so the
        # solver would return its warm-start value and report it as optimal.
        # Graph reachability answers this exactly, before anything is compiled.
        if self.optimizable_vars:
            reachable = set(ancestors([self._pymc_model[self.response_variable]]))
            unreachable = [
                name
                for name in self.optimizable_vars
                if self._pymc_model[name] not in reachable
            ]
            if unreachable:
                raise ValueError(
                    f"response_variable={self.response_variable!r} does not "
                    f"depend on optimizable_vars {unreachable}, so their "
                    "gradient is identically zero and they cannot be "
                    "optimized. Pass a response variable that includes their "
                    "contribution."
                )

        # 9. Compile objective & gradient
        self._compile_objective_and_grad()

        # 10. Build constraints
        self._constraints = {}
        self.set_constraints(constraints=self.constraints)

    @property
    def optimization_variables(self) -> OptimizationVariables:
        """The decision vector's variables: media, plus any ``optimizable_vars``.

        Exposed so a custom
        :class:`~pymc_marketing.mmm.constraints.Constraint` can reach a
        variable's segment of the flat vector, which is what constraining
        anything other than the media budgets requires. For example, capping
        the total of a lever named ``promo_data``::

            Constraint(
                key="max_total_discount",
                constraint_type="ineq",
                constraint_fun=lambda budgets, total, opt: (
                    1.0 - opt.optimization_variables.variable_slice("promo_data").sum()
                ),
            )
        """
        return self._variables

    def set_constraints(self, constraints: Sequence[Constraint]) -> None:
        """Set constraints for the optimizer.

        An empty ``constraints`` auto-adds the default sum constraint; a
        non-empty one means the caller is in charge.
        """
        add_default = not constraints

        self._constraints = {}
        for c in constraints:
            if c.key in self._constraints:
                raise ValueError(
                    f"Duplicate constraint key {c.key!r}. Constraint keys must be unique."
                )
            self._constraints[c.key] = c

        if add_default:
            self._constraints["default"] = build_default_sum_constraint("default")

        # Compile constraints to be used by SciPy
        self._compiled_constraints = compile_constraints_for_scipy(
            constraints=self._constraints, optimizer=self
        )

    def _validate_channel_scales(self) -> None:
        """Check that ``channel_scales`` is a scalar or a 1-D array over channels.

        Raises
        ------
        ValueError
            If ``channel_scales`` has more than one dimension, or is 1-D with a
            length that does not match the model's ``channel`` coordinate.
        """
        scales = np.asarray(self.channel_scales)
        if scales.ndim == 0:
            return
        if scales.ndim > 1:
            raise ValueError(
                "channel_scales must be a scalar or a 1-D array over channels, got "
                f"an array with shape {scales.shape}."
            )
        n_channels = len(self.model.coords.get("channel", ()))
        if n_channels and scales.shape[0] != n_channels:
            raise ValueError(
                f"channel_scales has length {scales.shape[0]} but the model has "
                f"{n_channels} channels."
            )

    def _validate_and_process_budget_distribution(
        self,
        budget_distribution_over_period: DataArray | None,
        num_periods: int,
        budget_dims: list[str],
        budgets_to_optimize: DataArray,
        date_dim: str = "date",
    ) -> XTensorVariable | None:
        """Validate and process budget distribution over periods.

        Parameters
        ----------
        budget_distribution_over_period : DataArray | None
            Distribution factors for budget allocation over time.
        num_periods : int
            Number of time periods to allocate budget for.
        budget_dims : list[str]
            List of budget dimensions (excluding the date dimension).
        budgets_to_optimize : DataArray
            Mask defining which budgets to optimize.
        date_dim : str, optional
            Name of the date dimension. Default is ``"date"``.

        Returns
        -------
        XTensorVariable | None
            Processed tensor containing masked time factors, or None if no distribution provided.
        """
        if budget_distribution_over_period is None:
            return None

        # Validate dimensions - date should be first
        expected_dims = (date_dim, *budget_dims)
        if set(budget_distribution_over_period.dims) != set(expected_dims):
            raise ValueError(
                f"budget_distribution_over_period must have dims {expected_dims}, "
                f"but got {budget_distribution_over_period.dims}"
            )

        # Validate date dimension length
        if len(budget_distribution_over_period.coords[date_dim]) != num_periods:
            raise ValueError(
                f"budget_distribution_over_period {date_dim!r} dimension must have length {num_periods}, "
                f"but got {len(budget_distribution_over_period.coords[date_dim])}"
            )

        # Validate that factors sum to 1 along date dimension
        sums = budget_distribution_over_period.sum(dim=date_dim)
        if not np.allclose(sums.values, 1.0, rtol=1e-5):
            raise ValueError(
                f"budget_distribution_over_period must sum to 1 along the {date_dim!r} dimension "
                "for each combination of other dimensions"
            )

        # Align to the model's coordinate order before flattening: the factors
        # are combined positionally with the (model-ordered) mask below, so an
        # input in a different coordinate order would hand each cell another
        # cell's temporal profile.
        budget_distribution_over_period = align_to_model_coords(
            budget_distribution_over_period,
            {dim: list(budgets_to_optimize.coords[dim].values) for dim in budget_dims},
            label="budget_distribution_over_period",
        )

        # Pre-process: Apply the mask to get only factors for optimized budgets
        # This avoids shape mismatches during gradient computation
        time_factors_full = budget_distribution_over_period.transpose(
            *expected_dims
        ).values

        # Reshape to (num_periods, flat_budget_dims) and apply mask
        time_factors_flat = time_factors_full.reshape((num_periods, -1))
        bool_mask = budgets_to_optimize.values.flatten()
        time_factors_masked = time_factors_flat[:, bool_mask]

        # Store only the masked tensor
        return ptx.xtensor_constant(
            time_factors_masked,
            name="budget_distribution_over_period",
            dims=(date_dim, FLAT_DIM),
        )

    @staticmethod
    def _validate_and_process_cost_per_unit(
        cost_per_unit: DataArray | None,
        num_periods: int,
        budget_dims: list[str],
        budget_coords: dict[str, list] | None = None,
        date_dim: str = "date",
    ) -> XTensorVariable | None:
        """Validate and convert cost_per_unit to a PyTensor constant.

        Parameters
        ----------
        cost_per_unit : DataArray or None
            Cost per unit with dims (date, *budget_dims).
        num_periods : int
            Number of optimization periods.
        budget_dims : list[str]
            Budget dimension names (excluding the date dimension).
        budget_coords : dict[str, list] or None
            Model coordinate order for each budget dimension, used to reindex
            the cost_per_unit DataArray before extracting values.
        date_dim : str, optional
            Name of the date dimension. Default is ``"date"``.

        Returns
        -------
        XTensorVariable or None
            Constant tensor with shape (num_periods, *budget_shape), or
            None if no cost_per_unit provided.

        Raises
        ------
        ValueError
            If dimensions or date length don't match expectations.
        """
        if cost_per_unit is None:
            return None

        expected_dims = (date_dim, *budget_dims)
        if set(cost_per_unit.dims) != set(expected_dims):
            raise ValueError(
                f"cost_per_unit must have exactly the dims {set(expected_dims)}, "
                f"but got {set(cost_per_unit.dims)}"
            )

        if len(cost_per_unit.coords[date_dim]) != num_periods:
            raise ValueError(
                f"cost_per_unit {date_dim!r} dimension must have length {num_periods}, "
                f"but got {len(cost_per_unit.coords[date_dim])}"
            )

        if budget_coords is not None:
            # Align before validating: a coordinate the model has but
            # cost_per_unit lacks becomes NaN during alignment, and validating
            # first would let it through into the graph.
            cost_per_unit = align_to_model_coords(
                cost_per_unit, budget_coords, label="cost_per_unit"
            )

        if cost_per_unit.isnull().any() or (cost_per_unit <= 0).any():
            raise ValueError(
                "cost_per_unit values must be positive "
                "(no NaN, zero, or negative values)."
            )
        values = cost_per_unit.transpose(*expected_dims)
        return ptx.as_xtensor(values, name="cost_per_unit")

    def extract_response_distribution(self, response_variable: str) -> XTensorVariable:
        """Extract the response distribution graph, conditioned on posterior parameters.

        Examples
        --------
        ``BudgetOptimizer(...).extract_response_distribution("channel_contribution")``
        returns a graph that computes ``"channel_contribution"`` as a function of both
        the newly introduced budgets and the posterior of model parameters.
        """
        # Local import to avoid circular import at module load time
        from pymc_marketing.pytensor_utils import extract_response_distribution

        return extract_response_distribution(
            pymc_model=self._pymc_model,
            idata=_extract_dataset(self.idata, "posterior"),
            response_variable=response_variable,
            frozen_deterministics=self.frozen_deterministics,
        )

    def _compile_objective_and_grad(self):
        """Compile the objective function and its gradient, both referencing `self._budgets_flat`."""
        budgets_flat = self._budgets_flat
        budgets = self._budgets
        response_distribution = self.extract_response_distribution(
            self.response_variable
        )

        objective = -self.utility_function(
            samples=response_distribution, budgets=budgets
        )
        # xtensor Ops don't have gradients implemented yet, so we need to include `lower_xtensor`
        objective_tensor = rewrite_graph(
            objective.values, include=("lower_xtensor", "canonicalize", "stabilize")
        )
        objective_grad = pt.grad(objective_tensor, budgets_flat)

        objective_and_grad_func = function(
            [budgets_flat], [objective, objective_grad], **self.compile_kwargs or {}
        )

        # Avoid repeated input validation for performance
        if hasattr(objective_and_grad_func, "trust_input"):
            objective_and_grad_func.trust_input = True

        self._objective_and_grad = objective_and_grad_func

    def allocate_budget(
        self,
        total_budget: float,
        budget_bounds: DataArray | dict[str, tuple[float, float]] | None = None,
        x0: np.ndarray | DataArray | dict[str, DataArray] | None = None,
        minimize_kwargs: dict[str, Any] | None = None,
        return_if_fail: bool = False,
        callback: bool = False,
    ) -> BudgetOptimizationResult:
        """
        Allocate the budget based on `total_budget`, optional `budget_bounds`, and custom constraints.

        The default sum constraint ensures that the sum of the optimized budget
        equals `total_budget`. If `budget_bounds` are not provided, each channel
        will be constrained to lie in [0, total_budget].

        Parameters
        ----------
        total_budget : float
            The total budget to allocate.
        budget_bounds : DataArray or dict, optional
            - If None, default bounds of [0, total_budget] per channel are assumed.
            - If a dict, must map each channel to (low, high) budget pairs (only valid if there's one dimension).
            - If an xarray.DataArray, must have dims ``(*budget_dims, "bound")``,
              specifying [low, high] per channel cell.
        x0 : np.ndarray, DataArray or dict, optional
            Initial guess. A labelled ``DataArray`` over the budget dims (e.g. a
            previous solution or a current spend plan) is packed into the flat
            decision vector automatically; a dict maps decision-variable names
            to labelled ``DataArray`` objects. A flat ``np.ndarray`` of size (n,) is
            also accepted. If None, the total budget is spread uniformly across
            all drivers to be optimized.
        minimize_kwargs : dict, optional
            Extra kwargs for `scipy.optimize.minimize`. Defaults to method="SLSQP",
            ftol=1e-9, maxiter=1_000.
        return_if_fail : bool, optional
            Return output even if optimization fails. Default is False.
        callback : bool, optional
            Whether to track optimization progress. When True, ``result.callback_info`` is
            populated with a list of dictionaries with optimization information at
            each iteration including 'x' (parameter values), 'fun' (objective value),
            'jac' (gradient), and constraint information. Default is False.

        Returns
        -------
        BudgetOptimizationResult
            Result object with ``budgets`` (the optimized allocation across
            channels, monetary units), ``scipy_result`` (the raw scipy
            optimization result), ``optimized_vars`` (empty for now), and
            ``callback_info`` (per-iteration diagnostics when ``callback=True``, else
            ``None``). Iterating the result yields ``(budgets, scipy_result)``,
            so ``optimal, res = optimizer.allocate_budget(...)`` keeps working.

        Raises
        ------
        MinimizeException
            If the optimization fails for any reason, the exception message will contain the details.

        Notes
        -----
        **Units and cost_per_unit**:

        - All budget inputs (total_budget, budget_bounds) are in monetary units.
        - If cost_per_unit is provided, the optimizer converts internally:
          ``budget_in_original_units[t] = budget_in_dollars[t] / cost_per_unit[t]``
        - Each time period uses its own cost_per_unit value (no averaging).
        - Output optimal_budgets are in monetary units for user convenience.
        """
        # set total budget
        self._total_budget.set_value(np.asarray(total_budget, dtype="float64"))

        # coordinate user-provided and default minimize_kwargs
        if minimize_kwargs is None:
            minimize_kwargs = self.DEFAULT_MINIMIZE_KWARGS
        else:
            # Merge with defaults (preferring user-supplied keys)
            minimize_kwargs = {**self.DEFAULT_MINIMIZE_KWARGS, **minimize_kwargs}

        # 1. Process budget bounds
        bounds_overrides: dict[str, list[tuple[float | None, float | None]]] = {}
        if budget_bounds is None:
            warnings.warn(
                "No budget bounds provided. Using default bounds (0, total_budget) for each channel.",
                UserWarning,
                stacklevel=2,
            )
            budget_bounds_array = None
        elif isinstance(budget_bounds, dict):
            if len(self._budget_dims) > 1:
                raise ValueError(
                    f"Dict approach to budget_bounds is not supported for multi-dimensional budgets. "
                    f"budget_dims = {self._budget_dims}. Pass an xarray.DataArray instead."
                )

            # Flatten each channel's bounds into an array

            budget_bounds_array = np.broadcast_to(
                [
                    np.asarray(budget_bounds[channel])
                    for channel in self._budget_coords[self._budget_dims[0]]
                ],
                (*self._budget_shape, 2),
            )
        elif isinstance(budget_bounds, DataArray):
            # Must have dims (*self._budget_dims, "bound")
            if set(budget_bounds.dims) != set([*self._budget_dims, "bound"]):
                raise ValueError(
                    f"budget_bounds must be a DataArray with dims {(*self._budget_dims, 'bound')}"
                )
            budget_bounds_array = (
                align_to_model_coords(
                    budget_bounds, self._budget_coords, label="budget_bounds"
                )
                .transpose(*self._budget_dims, "bound")
                .values
            )
        else:
            raise ValueError(
                "budget_bounds must be a dictionary or an xarray.DataArray"
            )

        # 2. Build the final bounds list, variable by variable. With no
        # user-supplied bounds the media variable's own defaults apply.
        if budget_bounds_array is not None:
            bounds_overrides[self.channel_data_var] = [
                (low, high)
                for (low, high) in budget_bounds_array[self.budgets_to_optimize.values]  # type: ignore
            ]
        bounds = self._variables.bounds(total_budget, overrides=bounds_overrides)

        # 3. Construct the initial guess (x0) if not provided; labelled values
        # are packed into flat order by the optimization variables.
        if x0 is None:
            x0 = self._variables.x0(total_budget)
        elif isinstance(x0, DataArray | dict):
            x0 = self._variables.pack(x0)
        x0 = np.asarray(x0).astype(self._budgets_flat.type.dtype)

        # filter x0 based on shape/type of self._budgets_flat
        # will raise a TypeError if x0 does not have acceptable shape and/or type
        x0 = self._budgets_flat.type.filter(x0)

        # 5. Set up callback tracking if requested
        callback_info: list[OptimizationIterationInfo] = []

        def track_progress(xk):
            """Track optimization progress at each iteration."""
            # Evaluate objective and gradient
            obj_val, grad_val = self._objective_and_grad(xk)

            # Store iteration info
            iter_info: OptimizationIterationInfo = {
                "x": np.array(xk),  # Current parameter values
                "fun": float(obj_val),  # Objective function value (scalar)
                "jac": np.array(grad_val),  # Gradient values
            }

            # Evaluate constraint values and gradients if available
            if self._compiled_constraints:
                constraint_info: list[ConstraintIterationInfo] = []

                for _, constraint in enumerate(self._compiled_constraints):
                    # Evaluate constraint function
                    c_val = constraint["fun"](xk)
                    # Evaluate constraint gradient
                    c_jac = constraint["jac"](xk)

                    constraint_info.append(
                        {
                            "type": constraint["type"],
                            "value": float(c_val)
                            if np.ndim(c_val) == 0
                            else np.array(c_val),
                            "jac": np.array(c_jac) if c_jac is not None else None,
                        }
                    )

                iter_info["constraint_info"] = constraint_info

            callback_info.append(iter_info)

        # 5. Run the SciPy optimizer
        scipy_callback = track_progress if callback else None

        result = minimize(
            fun=self._objective_and_grad,
            x0=x0,
            jac=True,
            bounds=bounds,
            constraints=self._compiled_constraints,
            callback=scipy_callback,
            **minimize_kwargs,
        )

        # 6. Process results
        if result.success or return_if_fail:
            # Media is the named output; every other optimization variable
            # travels in `optimized_vars`. This is already the general form, so
            # registering further variables needs no change here: with media
            # alone the remainder is simply empty.
            unpacked = self._variables.unpack(result.x)
            optimal_budgets = unpacked.pop(self.channel_data_var)
            optimal_budgets.attrs["pymc_marketing_version"] = __version__

            return BudgetOptimizationResult(
                budgets=optimal_budgets,
                scipy_result=result,
                optimized_vars=unpacked,
                callback_info=callback_info if callback else None,
            )

        else:
            raise MinimizeException(f"Optimization failed: {result.message}")


class CustomModelWrapper(BaseModel):
    """Wrapper for custom PyMC models to be used with :class:`BudgetOptimizer`.

    This wrapper lets you plug any fitted PyMC model into the budget optimizer
    without subclassing the built-in :class:`~pymc_marketing.mmm.MMM`.  It
    clones your ``base_model`` for each optimization run and resets the
    channel-data variable to zeros of the appropriate shape.

    Parameters
    ----------
    base_model : pymc.Model
        A PyMC model that contains a shared ``channel_data`` variable
        (or the variable named by ``channel_data_var``). The fitted
        posterior is passed separately via ``idata``.
    idata : arviz.InferenceData or xarray.DataTree
        Posterior samples from the fitted model.
    channels : Sequence[str]
        Names of the channel dimensions, in the same order as the last axis
        of the channel-data variable.
    adstock_periods : int, optional
        Number of extra periods to prepend for adstock warm-up.  The model is
        built with ``num_periods + adstock_periods`` date steps, and only the
        last ``num_periods`` are used when computing the response.  Defaults
        to ``0`` (no warm-up).
    channel_data_var : str, optional
        Name of the shared channel-data variable inside ``base_model``.
        Defaults to ``"channel_data"``.
    adstock : Any, optional
        Deprecated. Pass ``adstock_periods`` instead.

    Examples
    --------
    Build a simple custom model and wrap it for budget optimization:

    .. code-block:: python

        import pymc as pm
        import numpy as np
        from pymc_marketing.mmm.budget_optimizer import (
            CustomModelWrapper,
            BudgetOptimizer,
        )

        channels = ["tv", "search", "social"]
        n_obs, n_channels = 52, len(channels)

        with pm.Model(coords={"date": range(n_obs), "channel": channels}) as base_model:
            channel_data = pm.Data(
                "channel_data", np.zeros((n_obs, n_channels)), dims=("date", "channel")
            )
            beta = pm.Normal("beta", mu=0, sigma=1, dims="channel")
            mu = (channel_data * beta).sum(axis=-1)
            pm.Normal("y", mu=mu, sigma=1, observed=np.zeros(n_obs))

        # After sampling, wrap for optimization:
        wrapper = CustomModelWrapper(
            base_model=base_model,
            idata=idata,  # your posterior samples
            channels=channels,
        )

        optimizer = BudgetOptimizer(
            model=wrapper.optimization_model(num_periods=13),
            idata=wrapper.idata,
            adstock_periods=wrapper.adstock_periods,
            channel_scales=wrapper.channel_scales,
            num_periods=13,
        )
        optimal, result = optimizer.allocate_budget(total_budget=100_000)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    base_model: Model = Field(
        ...,
        description="Underlying PyMC model to be cloned for optimization.",
    )
    idata: Any = Field(
        ...,
        description="Posterior samples from the fitted model (xarray.DataTree or arviz.InferenceData).",
    )
    channel_columns: list[str] = Field(
        ...,
        description="Channel labels used for budget optimization.",
    )
    adstock_periods: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of extra warm-up periods prepended for adstock carryover. "
            "Equivalent to ``adstock.l_max`` on the built-in MMM. Defaults to 0."
        ),
    )
    channel_data_var: str = Field(
        default="channel_data",
        description=(
            "Name of the shared channel-data variable inside ``base_model``. "
            "Defaults to ``'channel_data'``."
        ),
    )

    _channel_scales: Any = PrivateAttr(default=1.0)
    _adstock_arg: Any = PrivateAttr(default=None)

    @property
    def channel_scales(self) -> float | np.ndarray:
        """Per-channel scale factors used by the budget optimizer.

        Returns
        -------
        float or np.ndarray
            Scale factor(s) applied to channel budgets. Defaults to 1.0 (no scaling).
        """
        return self._channel_scales

    @channel_scales.setter
    def channel_scales(self, value: float | np.ndarray) -> None:
        self._channel_scales = value

    def __init__(
        self,
        base_model: Model,
        idata: Any,
        channels: Sequence[str],
        adstock_periods: int = 0,
        channel_data_var: str = "channel_data",
        adstock: Any = None,
        **kwargs,
    ) -> None:
        warnings.warn(
            "CustomModelWrapper is deprecated and will be removed in a future version. "
            "Build the optimization model directly instead:\n\n"
            "    from pymc.model.fgraph import clone_model\n"
            "    cloned = clone_model(base_model)\n"
            "    pm.set_data(\n"
            '        {"channel_data": np.zeros((num_periods + adstock_periods, n_channels))},\n'
            "        model=cloned,\n"
            "    )\n"
            "    optimizer = BudgetOptimizer(\n"
            "        model=cloned, idata=idata, num_periods=num_periods, ...\n"
            "    )",
            DeprecationWarning,
            stacklevel=2,
        )

        if adstock is not None:
            warnings.warn(
                "The 'adstock' argument is deprecated and will be removed in a future version. "
                "Pass 'adstock_periods=adstock.l_max' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if adstock_periods == 0 and hasattr(adstock, "l_max"):
                adstock_periods = adstock.l_max

        super().__init__(
            base_model=base_model,
            idata=idata,
            channel_columns=list(channels),
            adstock_periods=adstock_periods,
            channel_data_var=channel_data_var,
            **kwargs,
        )
        self._adstock_arg = adstock

    def optimization_model(self, num_periods: int) -> pm.Model:
        """Clone ``base_model`` and resize the channel-data variable for optimization.

        Parameters
        ----------
        num_periods : int
            Number of optimization periods (excluding any adstock warm-up).
            The model will be built with ``num_periods + adstock_periods`` date steps.

        Returns
        -------
        pymc.Model
            A cloned model ready for use by :class:`BudgetOptimizer`.
        """
        total_periods = num_periods + self.adstock_periods
        coords = {
            "date": np.arange(total_periods),
            "channel": self.channel_columns,
        }
        model_clone = clone_model(self.base_model)
        pm.set_data(
            {
                self.channel_data_var: np.zeros(
                    (total_periods, len(self.channel_columns))
                )
            },
            model=model_clone,
            coords=coords,
        )
        return model_clone

    def _set_predictors_for_optimization(self, num_periods: int) -> pm.Model:
        """Return an optimization model (deprecated; use :meth:`optimization_model` instead).

        .. deprecated::
            ``_set_predictors_for_optimization`` is deprecated and will be removed in a
            future version.  :class:`BudgetOptimizer` now calls
            :meth:`optimization_model` directly.
        """
        warnings.warn(
            "_set_predictors_for_optimization is deprecated and will be removed in a "
            "future version. The BudgetOptimizer now calls optimization_model "
            "directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.optimization_model(num_periods)


OptimizerCompatibleModel.register(CustomModelWrapper)
