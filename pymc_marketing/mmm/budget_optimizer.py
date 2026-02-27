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
---------------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    import xarray as xr
    from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
    from pymc_marketing.mmm.multidimensional import (
        MMM,
        MultiDimensionalBudgetOptimizerWrapper,
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
    wrapper = MultiDimensionalBudgetOptimizerWrapper(
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

Use a custom pymc model with any dimensionality
----------------------------------------------

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
    #    - a deterministic named 'total_contribution' with dim "date"
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
        pm.Deterministic("total_contribution", mu.sum(), dims=())
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

- The optimizer works on any wrapper that satisfies `OptimizerCompatibleModelWrapper`:
  - Attributes: `adstock`, `_channel_scales`, `idata` (arviz.InferenceData with posterior)
  - Method: `_set_predictors_for_optimization(num_periods) -> pm.Model` that returns a PyMC
    model where a variable named `channel_data` exists with dims including `"date"` and all
    budget dims (e.g., `("channel", "geo")`).
    The optimizer replaces `channel_data` with the optimization variable under the hood.
- Posterior must contain a response variable (default: `"total_contribution"`) or any custom
  `response_variable` you pass, and the required MMM deterministics (e.g. `channel_contribution`).
- For time distribution: pass a DataArray with dims `("date", *budget_dims)` and values along
  `date` summing to 1 for each budget cell.
- Bounds can be a dict only for single‑dimensional budgets; otherwise use an xarray.DataArray
  (use `optimizer_xarray_builder(...)`).

Notes
-----
- If `budgets_to_optimize` is not provided, the optimizer auto‑detects cells with historical
  information using `idata.posterior.channel_contribution.mean(("chain","draw","date")).astype(bool)`.
- Default bounds are `[0, total_budget]` on each optimized cell.
- Set `callback=True` in `allocate_budget(...)` to receive per‑iteration diagnostics
  (objective, gradient, constraints) for monitoring.
"""

import warnings
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from arviz import InferenceData
from pydantic import BaseModel, ConfigDict, Field, InstanceOf, PrivateAttr
from pymc import Model, do
from pymc.model.fgraph import clone_model
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.pytensorf import rewrite_pregrad
from pytensor import function
from pytensor.compile.sharedvalue import SharedVariable, shared
from scipy.optimize import OptimizeResult, minimize
from xarray import DataArray

from pymc_marketing.mmm.constraints import (
    Constraint,
    build_default_sum_constraint,
    compile_constraints_for_scipy,
)
from pymc_marketing.mmm.utility import UtilityFunctionType, average_response
from pymc_marketing.pytensor_utils import merge_models

# Delayed import inside methods to avoid circular dependency on pytensor_utils


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


@runtime_checkable
class OptimizerCompatibleModelWrapper(Protocol):
    """Protocol for marketing mix model wrappers compatible with the BudgetOptimizer."""

    adstock: Any
    _channel_scales: Any
    idata: InferenceData

    def _set_predictors_for_optimization(self, num_periods: int) -> Model:
        """Set the predictors for optimization."""


class BuildMergedModel(OptimizerCompatibleModelWrapper):
    """Merge multiple optimizer-compatible models into a single model.

    This wrapper combines several optimizer-compatible MMM wrappers by:
    - Merging their posterior `InferenceData` with per-model prefixes
    - Optionally thinning posterior draws via ``use_every_n_draw``
    - Exposing a persistent merged PyMC ``Model`` for optimization through
      ``_set_predictors_for_optimization`` and a dynamic ``model`` property for
      inspection when needed

    Parameters
    ----------
    models : list[OptimizerCompatibleModelWrapper]
        A list of wrappers that each expose ``idata`` and
        ``_set_predictors_for_optimization(num_periods: int) -> Model``.
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
    models : list[OptimizerCompatibleModelWrapper]
        The provided list of wrappers.
    num_models : int
        Number of models being merged.
    num_periods : int | None
        Number of forecast periods inferred from the primary model (if available).
    idata : arviz.InferenceData
        The merged and prefixed posterior (and data) container.
    adstock : Any
        Carried over from the primary model when available.
    model : pymc.Model
        Property returning a merged PyMC model; see Notes.

    Examples
    --------
    Merge three multidimensional MMMs into a single optimizer model:

    .. code-block:: python

        from pymc_marketing.mmm.multidimensional import (
            MMM,
            MultiDimensionalBudgetOptimizerWrapper,
        )
        from pymc_marketing.mmm.budget_optimizer import (
            BuildMergedModel,
            BudgetOptimizer,
        )

        # Assume m1, m2, m3 are already fitted MMM instances
        w1 = MultiDimensionalBudgetOptimizerWrapper(
            model=m1, start_date=start, end_date=end
        )
        w2 = MultiDimensionalBudgetOptimizerWrapper(
            model=m2, start_date=start, end_date=end
        )
        w3 = MultiDimensionalBudgetOptimizerWrapper(
            model=m3, start_date=start, end_date=end
        )

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
        m_opt = merged_single._set_predictors_for_optimization(
            num_periods=merged_single.num_periods
        )

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
        models: list[OptimizerCompatibleModelWrapper],
        prefixes: list[str] | None = None,
        merge_on: str | None = "channel_data",
        use_every_n_draw: int = 1,
    ) -> None:
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

        if hasattr(self.primary_model, "adstock"):
            self.adstock = self.primary_model.adstock

        # Signal to BudgetOptimizer to enforce mask validation
        self.enforce_budget_mask_validation = False

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
                    prefixed_idata[group] = prefixed_idata[group].rename(rename_dict)

        return prefixed_idata

    def _set_predictors_for_optimization(self, num_periods: int) -> Model:
        # If we already built a persistent model for this horizon, reuse it
        if (
            self._persistent_merged_model is not None
            and self._persistent_num_periods == int(num_periods)
        ):
            return self._persistent_merged_model

        # Build per-model optimization models
        pymc_models = [
            m._set_predictors_for_optimization(num_periods=num_periods)
            for m in self.models
        ]
        if self.num_models == 1:
            self._persistent_merged_model = freeze_dims_and_data(pymc_models[0])
        else:
            self._persistent_merged_model = merge_models(
                models=pymc_models, prefixes=self.prefixes, merge_on=self.merge_on
            )

        self._persistent_num_periods = int(num_periods)
        return self._persistent_merged_model

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
            return self._set_predictors_for_optimization(int(self.num_periods))

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
    num_periods : int
        Number of time units at the desired time granularity to allocate budget for.
    model : MMMModel
        The marketing mix model to optimize.
    response_variable : str, optional
        The response variable to optimize. Default is "total_contribution".
    utility_function : UtilityFunctionType, optional
        The utility function to maximize. Default is the mean of the response distribution.
    budgets_to_optimize : xarray.DataArray, optional
        Mask defining a subset of budgets to optimize. Non-optimized budgets remain fixed at 0.
    custom_constraints : Sequence[Constraint], optional
        Custom constraints for the optimizer.
    default_constraints : bool, optional
        Whether to add a default sum constraint on the total budget. Default is True.
    budget_distribution_over_period : xarray.DataArray, optional
        Distribution factors for budget allocation over time. Should have dims ("date", *budget_dims)
        where date dimension has length num_periods. Values along date dimension should sum to 1 for
        each combination of other dimensions. If None, budget is distributed evenly across periods.
    """

    num_periods: int = Field(
        ...,
        gt=0,
        description="Number of time units at the desired time granularity to allocate budget for.",
    )

    mmm_model: InstanceOf[OptimizerCompatibleModelWrapper] = Field(
        ...,
        description="The marketing mix model to optimize.",
        arbitrary_types_allowed=True,
        alias="model",
    )

    response_variable: str = Field(
        default="total_contribution",
        description="The response variable to optimize.",
    )

    utility_function: UtilityFunctionType = Field(
        default=average_response,
        description="Utility function to maximize.",
        arbitrary_types_allowed=True,
    )

    budgets_to_optimize: DataArray | None = Field(
        default=None,
        description="Mask defining a subset of budgets to optimize. Non-optimized budgets remain fixed at 0.",
    )

    custom_constraints: Sequence[Constraint] = Field(
        default=(),
        description="Custom constraints for the optimizer.",
        arbitrary_types_allowed=True,
    )

    default_constraints: bool = Field(
        default=True,
        description="Whether to add a default sum constraint on the total budget.",
    )

    budget_distribution_over_period: DataArray | None = Field(
        default=None,
        description=(
            "Distribution factors for budget allocation over time. Should have dims ('date', *budget_dims) "
            "where date dimension has length num_periods. Values along date dimension should sum to 1 for "
            "each combination of other dimensions. If None, budget is distributed evenly across periods."
        ),
    )

    compile_kwargs: dict | None = Field(
        default=None,
        description="Keyword arguments for the model compilation. Specially usefull to pass compilation mode",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    DEFAULT_MINIMIZE_KWARGS: ClassVar[dict] = {
        "method": "SLSQP",
        "options": {"ftol": 1e-9, "maxiter": 1_000},
    }

    def __init__(self, **data):
        super().__init__(**data)
        # 1. Prepare model with time dimension for optimization
        pymc_model = self.mmm_model._set_predictors_for_optimization(
            self.num_periods
        )  # TODO: Once multidimensional class becomes the main class.

        # 2. Shared variable for total_budget: Use annotation to avoid type checking
        self._total_budget: SharedVariable = shared(
            np.array(0.0, dtype="float64"), name="total_budget"
        )  # type: ignore

        # 3. Identify budget dimensions and shapes
        self._budget_dims = [
            dim
            for dim in pymc_model.named_vars_to_dims["channel_data"]
            if dim != "date"
        ]
        self._budget_coords = {
            dim: list(pymc_model.coords[dim]) for dim in self._budget_dims
        }
        self._budget_shape = tuple(len(coord) for coord in self._budget_coords.values())

        # 4. Ensure that we only optmize over non-zero channels
        # Only perform non-zero channel detection for MMM instances.
        # For OptimizerCompatibleModelWrapper, default to optimizing all channels unless a mask is provided.
        is_wrapper = (
            "channel_contribution" not in self.mmm_model.idata.posterior.data_vars
        )

        if self.budgets_to_optimize is None:
            if is_wrapper:
                # Wrapper path: default to all True over budget dims
                ones = np.ones(self._budget_shape, dtype=bool)
                self.budgets_to_optimize = xr.DataArray(
                    ones, coords=self._budget_coords, dims=self._budget_dims
                )
            else:
                # If no mask is provided, optimize all non-zero channels in the model
                self.budgets_to_optimize = (
                    self.mmm_model.idata.posterior.channel_contribution.mean(
                        ("chain", "draw", "date")
                    ).astype(bool)
                )
        elif not is_wrapper:
            # If a mask is provided for MMM instances, ensure it has the correct shape
            expected_mask = self.mmm_model.idata.posterior.channel_contribution.mean(
                ("chain", "draw", "date")
            ).astype(bool)

            # Check if we are asking to optimize over channels that are not present in the model
            if np.any(self.budgets_to_optimize.values > expected_mask.values):
                raise ValueError(
                    "budgets_to_optimize mask contains True values at coordinates where the model has no "
                    "information."
                )

        size_budgets = self.budgets_to_optimize.sum().item()

        self._budgets_flat = pt.tensor("budgets_flat", shape=(size_budgets,))

        # Fill a zero array, then set only the True positions
        budgets_zeros = pt.zeros(self._budget_shape)
        budgets_zeros.name = "budgets_zeros"
        bool_mask = np.asarray(self.budgets_to_optimize).astype(bool)
        self._budgets = budgets_zeros[bool_mask].set(self._budgets_flat)

        # 5. Validate and process budget_distribution_over_period
        self._budget_distribution_over_period_tensor = (
            self._validate_and_process_budget_distribution(
                budget_distribution_over_period=self.budget_distribution_over_period,
                num_periods=self.num_periods,
                budget_dims=self._budget_dims,
                budgets_to_optimize=self.budgets_to_optimize,
            )
        )

        # 6. Replace channel_data with budgets in the PyMC model
        self._pymc_model = self._replace_channel_data_by_optimization_variable(
            pymc_model
        )

        # 7. Compile objective & gradient
        self._compiled_functions = {}
        self._compile_objective_and_grad()

        # 8. Build constraints
        self._constraints = {}
        self.set_constraints(
            default=self.default_constraints, constraints=self.custom_constraints
        )

    def set_constraints(self, constraints, default=None) -> None:
        """Set constraints for the optimizer."""
        self._constraints = {}
        if default is None:
            default = False if constraints else True

        for c in constraints:
            new_constraint = Constraint(
                key=c.key,
                constraint_fun=c.constraint_fun,
                constraint_type=c.constraint_type if c.constraint_type else "eq",
            )
            self._constraints[c.key] = new_constraint

        if default:
            self._constraints["default"] = build_default_sum_constraint("default")
            warnings.warn(
                "Using default equality constraint",
                UserWarning,
                stacklevel=2,
            )

        # Compile constraints to be used by SciPy
        self._compiled_constraints = compile_constraints_for_scipy(
            constraints=self._constraints, optimizer=self
        )

    def _validate_and_process_budget_distribution(
        self,
        budget_distribution_over_period: DataArray | None,
        num_periods: int,
        budget_dims: list[str],
        budgets_to_optimize: DataArray,
    ) -> pt.TensorVariable | None:
        """Validate and process budget distribution over periods.

        Parameters
        ----------
        budget_distribution_over_period : DataArray | None
            Distribution factors for budget allocation over time.
        num_periods : int
            Number of time periods to allocate budget for.
        budget_dims : list[str]
            List of budget dimensions (excluding 'date').
        budgets_to_optimize : DataArray
            Mask defining which budgets to optimize.

        Returns
        -------
        pt.TensorVariable | None
            Processed tensor containing masked time factors, or None if no distribution provided.
        """
        if budget_distribution_over_period is None:
            return None

        # Validate dimensions - date should be first
        expected_dims = ("date", *budget_dims)
        if set(budget_distribution_over_period.dims) != set(expected_dims):
            raise ValueError(
                f"budget_distribution_over_period must have dims {expected_dims}, "
                f"but got {budget_distribution_over_period.dims}"
            )

        # Validate date dimension length
        if len(budget_distribution_over_period.coords["date"]) != num_periods:
            raise ValueError(
                f"budget_distribution_over_period date dimension must have length {num_periods}, "
                f"but got {len(budget_distribution_over_period.coords['date'])}"
            )

        # Validate that factors sum to 1 along date dimension
        sums = budget_distribution_over_period.sum(dim="date")
        if not np.allclose(sums.values, 1.0, rtol=1e-5):
            raise ValueError(
                "budget_distribution_over_period must sum to 1 along the date dimension "
                "for each combination of other dimensions"
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
        return pt.constant(time_factors_masked, name="budget_distribution_over_period")

    def _apply_budget_distribution_over_period(
        self,
        budgets: pt.TensorVariable,
        num_periods: int,
        date_dim_idx: int,
    ) -> pt.TensorVariable:
        """Apply budget distribution over periods to budgets across time periods.

        Parameters
        ----------
        budgets : pt.TensorVariable
            The scaled budget tensor with shape matching budget dimensions.
        num_periods : int
            Number of time periods to distribute budget across.
        date_dim_idx : int
            Index position where the date dimension should be inserted.

        Returns
        -------
        pt.TensorVariable
            Budget tensor repeated across time periods with distribution factors applied.
            Shape will be (*budget_dims[:date_dim_idx], num_periods, *budget_dims[date_dim_idx:])
        """
        # Apply time distribution factors
        # The time factors are already masked and have shape (num_periods, num_optimized_budgets)
        # budgets has full shape (e.g., (2, 2) for geo x channel)
        # We need to extract only the optimized budgets

        # Get the optimized budget values
        bool_mask = np.asarray(self.budgets_to_optimize).astype(bool)
        budgets_optimized = budgets[bool_mask]  # Shape: (num_optimized_budgets,)

        # Now multiply budgets by time factors
        budgets_expanded = pt.expand_dims(
            budgets_optimized, 0
        )  # Shape: (1, num_optimized_budgets)
        repeated_budgets_flat = (
            budgets_expanded * self._budget_distribution_over_period_tensor
        )  # Shape: (num_periods, num_optimized_budgets)

        # Reconstruct the full shape for each time period
        repeated_budgets_list = []
        for t in range(num_periods):
            # Create a zero tensor with the full budget shape
            budgets_t = pt.zeros_like(budgets)
            # Set the optimized values
            budgets_t = budgets_t[bool_mask].set(repeated_budgets_flat[t])
            repeated_budgets_list.append(budgets_t)

        # Stack the time periods
        repeated_budgets = pt.stack(repeated_budgets_list, axis=date_dim_idx)
        repeated_budgets *= num_periods

        return repeated_budgets

    def _replace_channel_data_by_optimization_variable(self, model: Model) -> Model:
        """Replace `channel_data` in the model graph with our newly created `_budgets` variable."""
        num_periods = self.num_periods
        max_lag = self.mmm_model.adstock.l_max
        channel_data_dims = model.named_vars_to_dims["channel_data"]
        date_dim_idx = list(channel_data_dims).index("date")
        channel_scales = self.mmm_model._channel_scales
        channel_data_dtype = model["channel_data"].dtype
        if np.dtype(channel_data_dtype).kind != "f":
            raise ValueError(
                f"Optimization requires channel data of float type, got {channel_data_dtype}"
            )

        # Scale budgets by channel_scales
        budgets = self._budgets
        budgets /= channel_scales

        # Repeat budgets over num_periods
        repeated_budgets_shape = list(tuple(budgets.shape))
        repeated_budgets_shape.insert(date_dim_idx, num_periods)

        if self._budget_distribution_over_period_tensor is not None:
            # Apply time distribution factors
            repeated_budgets = self._apply_budget_distribution_over_period(
                budgets, num_periods, date_dim_idx
            )
        else:
            # Default behavior: distribute evenly across periods
            repeated_budgets = pt.broadcast_to(
                pt.expand_dims(budgets, date_dim_idx),
                shape=repeated_budgets_shape,
            )

        repeated_budgets.name = "repeated_budgets"

        # Pad the repeated budgets with zeros to account for carry-over effects
        # We set the repeated budgehts in a zero-filled tensor to achieve this
        repeated_budgets_with_carry_over_shape = list(tuple(budgets.shape))
        repeated_budgets_with_carry_over_shape.insert(
            date_dim_idx, num_periods + max_lag
        )

        # Get the dtype from the model's channel_data to ensure type compatibility
        channel_data_dtype = model["channel_data"].dtype

        repeated_budgets_with_carry_over = pt.zeros(
            repeated_budgets_with_carry_over_shape,
            dtype=channel_data_dtype,  # Use the same dtype as channel_data
        )
        set_idxs = (*((slice(None),) * date_dim_idx), slice(None, num_periods))
        repeated_budgets_with_carry_over = repeated_budgets_with_carry_over[
            set_idxs
        ].set(
            pt.cast(repeated_budgets, channel_data_dtype)
        )  # Cast to ensure type compatibility
        repeated_budgets_with_carry_over.name = "repeated_budgets_with_carry_over"

        # Freeze dims & data in the underlying PyMC model
        model = freeze_dims_and_data(model, data=[])

        # Use `do(...)` to replace `channel_data` with repeated_budgets_with_carry_over
        return do(model, {"channel_data": repeated_budgets_with_carry_over})

    def extract_response_distribution(
        self, response_variable: str
    ) -> pt.TensorVariable:
        """Extract the response distribution graph, conditioned on posterior parameters.

        Example:
        --------
        `BudgetOptimizer(...).extract_response_distribution("channel_contribution")`
        returns a graph that computes `"channel_contribution"` as a function of both
        the newly introduced budgets and the posterior of model parameters.
        """
        # Local import to avoid circular import at module load time
        from pymc_marketing.pytensor_utils import extract_response_distribution

        return extract_response_distribution(
            pymc_model=self._pymc_model,
            idata=self.mmm_model.idata,
            response_variable=response_variable,
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
        objective_grad = pt.grad(rewrite_pregrad(objective), budgets_flat)

        if self.compile_kwargs and (self.compile_kwargs["mode"]).lower() == "jax":
            # Use PyMC's JAX infrastructure for robust compilation
            from pymc.sampling.jax import get_jaxified_graph

            objective_and_grad_func = get_jaxified_graph(
                inputs=[budgets_flat],
                outputs=[objective, objective_grad],
                **{k: v for k, v in self.compile_kwargs.items() if k != "mode"} or {},
            )
        else:
            # Standard PyTensor compilation
            objective_and_grad_func = function(
                [budgets_flat], [objective, objective_grad], **self.compile_kwargs or {}
            )

        # Avoid repeated input validation for performance
        if hasattr(objective_and_grad_func, "trust_input"):
            objective_and_grad_func.trust_input = True

        self._compiled_functions[self.utility_function] = {
            "objective_and_grad": objective_and_grad_func,
        }

    def allocate_budget(
        self,
        total_budget: float,
        budget_bounds: DataArray | dict[str, tuple[float, float]] | None = None,
        x0: np.ndarray | None = None,
        minimize_kwargs: dict[str, Any] | None = None,
        return_if_fail: bool = False,
        callback: bool = False,
    ) -> (
        tuple[DataArray, OptimizeResult]
        | tuple[DataArray, OptimizeResult, list[dict[str, Any]]]
    ):
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
            - If an xarray.DataArray, must have dims (*budget_dims, "bound"), specifying [low, high] per channel cell.
        x0 : np.ndarray, optional
            Initial guess. Array of real elements of size (n,), where n is the number of driver budgets to optimize. If
            None, the total budget is spread uniformly across all drivers to be optimized.
        minimize_kwargs : dict, optional
            Extra kwargs for `scipy.optimize.minimize`. Defaults to method="SLSQP",
            ftol=1e-9, maxiter=1_000.
        return_if_fail : bool, optional
            Return output even if optimization fails. Default is False.
        callback : bool, optional
            Whether to return callback information tracking optimization progress. When True, returns a third
            element containing a list of dictionaries with optimization information at each iteration including
            'x' (parameter values), 'fun' (objective value), 'jac' (gradient), and constraint information.
            Default is False for backward compatibility.

        Returns
        -------
        optimal_budgets : xarray.DataArray
            The optimized budget allocation across channels.
        result : OptimizeResult
            The raw scipy optimization result.
        callback_info : list[dict[str, Any]], optional
            Only returned if callback=True. List of dictionaries containing optimization
            information at each iteration.

        Raises
        ------
        MinimizeException
            If the optimization fails for any reason, the exception message will contain the details.
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
        if budget_bounds is None:
            warnings.warn(
                "No budget bounds provided. Using default bounds (0, total_budget) for each channel.",
                UserWarning,
                stacklevel=2,
            )
            budget_bounds_array = np.broadcast_to(
                [0, total_budget],
                (*self._budget_shape, 2),
            )
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
                    for channel in self.mmm_model.channel_columns
                ],
                (*self._budget_shape, 2),
            )
        elif isinstance(budget_bounds, DataArray):
            # Must have dims (*self._budget_dims, "bound")
            if set(budget_bounds.dims) != set([*self._budget_dims, "bound"]):
                raise ValueError(
                    f"budget_bounds must be a DataArray with dims {(*self._budget_dims, 'bound')}"
                )
            budget_bounds_array = budget_bounds.transpose(
                *self._budget_dims, "bound"
            ).values
        else:
            raise ValueError(
                "budget_bounds must be a dictionary or an xarray.DataArray"
            )

        # 2. Build the final bounds list
        bounds = [
            (low, high)
            for (low, high) in budget_bounds_array[self.budgets_to_optimize.values]  # type: ignore
        ]

        # 3. Determine how many budget entries we optimize
        budgets_size = self.budgets_to_optimize.sum().item()  # type: ignore

        # 4. Construct the initial guess (x0) if not provided
        if x0 is None:
            x0 = (np.ones(budgets_size) * (total_budget / budgets_size)).astype(
                self._budgets_flat.type.dtype
            )

        # filter x0 based on shape/type of self._budgets_flat
        # will raise a TypeError if x0 does not have acceptable shape and/or type
        x0 = self._budgets_flat.type.filter(x0)

        # 5. Set up callback tracking if requested
        callback_info = []

        def track_progress(xk):
            """Track optimization progress at each iteration."""
            # Evaluate objective and gradient
            obj_val, grad_val = self._compiled_functions[self.utility_function][
                "objective_and_grad"
            ](xk)

            # Store iteration info
            iter_info = {
                "x": np.array(xk),  # Current parameter values
                "fun": float(obj_val),  # Objective function value (scalar)
                "jac": np.array(grad_val),  # Gradient values
            }

            # Evaluate constraint values and gradients if available
            if self._compiled_constraints:
                constraint_info = []

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
            fun=self._compiled_functions[self.utility_function]["objective_and_grad"],
            x0=x0,
            jac=True,
            bounds=bounds,
            constraints=self._compiled_constraints,
            callback=scipy_callback,
            **minimize_kwargs,
        )

        # 6. Process results
        if result.success or return_if_fail:
            # Fill zeros, then place the solution in masked positions
            optimal_budgets = np.zeros_like(
                self.budgets_to_optimize.values,  # type: ignore
                dtype=float,
            )
            optimal_budgets[self.budgets_to_optimize.values] = result.x  # type: ignore

            optimal_budgets = DataArray(
                optimal_budgets, dims=self._budget_dims, coords=self._budget_coords
            )

            if callback:
                return optimal_budgets, result, callback_info
            else:
                return optimal_budgets, result

        else:
            raise MinimizeException(f"Optimization failed: {result.message}")


class CustomModelWrapper(BaseModel):
    """Wrapper for the BudgetOptimizer to handle custom PyMC models."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    base_model: Model = Field(
        ...,
        description="Underlying PyMC model to be cloned for optimization.",
    )
    idata: InferenceData
    channel_columns: list[str] = Field(
        ...,
        description="Channel labels used for budget optimization.",
    )
    adstock: Any = Field(
        default_factory=lambda: type("Adstock", (), {"l_max": 0})(),
        description="Default adstock placeholder with zero carryover.",
    )

    _channel_scales: int = PrivateAttr(default=1.0)

    def __init__(
        self,
        base_model: Model,
        idata: InferenceData,
        channels: Sequence[str],
    ) -> None:
        super().__init__(
            base_model=base_model,
            idata=idata,
            channel_columns=list(channels),
        )

    def _set_predictors_for_optimization(self, num_periods: int) -> pm.Model:
        coords = {"date": np.arange(num_periods), "channel": self.channel_columns}
        model_clone = clone_model(self.base_model)
        pm.set_data(
            {"channel_data": np.zeros((num_periods, len(self.channel_columns)))},
            model=model_clone,
            coords=coords,
        )
        return model_clone


OptimizerCompatibleModelWrapper.register(CustomModelWrapper)
