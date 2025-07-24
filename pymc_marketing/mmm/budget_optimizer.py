#   Copyright 2022 - 2025 The PyMC Labs Developers
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

"""Budget optimization module."""

import warnings
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

import arviz as az
import numpy as np
import pytensor.tensor as pt
import xarray as xr
from arviz import InferenceData
from pydantic import BaseModel, ConfigDict, Field, InstanceOf
from pymc import Model, do
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.pytensorf import rewrite_pregrad
from pytensor import clone_replace, function
from pytensor.compile.sharedvalue import SharedVariable, shared
from pytensor.graph import rewrite_graph, vectorize_graph
from pytensor.graph.basic import ancestors
from scipy.optimize import OptimizeResult, minimize
from xarray import DataArray

try:
    from pymc.pytensorf import rvs_in_graph
except ImportError:
    from pymc.logprob.utils import rvs_in_graph

from pymc_marketing.mmm.constraints import (
    Constraint,
    build_default_sum_constraint,
    compile_constraints_for_scipy,
)
from pymc_marketing.mmm.utility import UtilityFunctionType, average_response


def optimizer_xarray_builder(value, **kwargs):
    """
    Create an xarray.DataArray with flexible dimensions and coordinates.

    Parameters
    ----------
    - value (array-like): The data values for the DataArray. Shape must match the dimensions implied by the kwargs.
    - **kwargs: Key-value pairs representing dimension names and their corresponding coordinates.

    Returns
    -------
    - xarray.DataArray: The resulting DataArray with the specified dimensions and values.

    Raises
    ------
    - ValueError: If the shape of `value` doesn't match the lengths of the specified coordinates.
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
        if self.budgets_to_optimize is None:
            # If no mask is provided, we optimize all channels
            self.budgets_to_optimize = (
                self.mmm_model.idata.posterior.channel_contribution.mean(
                    ("chain", "draw", "date")
                ).astype(bool)
            )
        else:
            # If a mask is provided, ensure it has the correct shape
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
        model = self._pymc_model
        # Convert InferenceData to a sample-major xarray
        posterior = az.extract(self.mmm_model.idata).transpose("sample", ...)  # type: ignore

        # The PyMC variable to extract
        response_var = model[response_variable]

        # Identify which free RVs are needed to compute `response_var`
        free_rvs = set(model.free_RVs)
        needed_rvs = [
            rv for rv in ancestors([response_var], blockers=free_rvs) if rv in free_rvs
        ]
        placeholder_replace_dict = {
            model[rv.name]: pt.tensor(name=rv.name, shape=rv.type.shape, dtype=rv.dtype)
            for rv in needed_rvs
        }

        [response_var] = clone_replace(
            [response_var],
            replace=placeholder_replace_dict,
        )

        if rvs_in_graph([response_var]):
            raise RuntimeError("RVs found in the extracted graph, this is likely a bug")

        # Cleanup graph
        response_var = rewrite_graph(response_var, include=("canonicalize", "ShapeOpt"))

        # Replace placeholders with actual posterior samples
        replace_dict = {}
        for placeholder in placeholder_replace_dict.values():
            replace_dict[placeholder] = pt.constant(
                posterior[placeholder.name].astype(placeholder.dtype),
                name=placeholder.name,
            )

        # Vectorize across samples
        response_distribution = vectorize_graph(response_var, replace=replace_dict)

        # Final cleanup
        response_distribution = rewrite_graph(
            response_distribution,
            include=(
                "useless",
                "local_eager_useless_unbatched_blockwise",
                "local_useless_unbatched_blockwise",
            ),
        )

        return response_distribution

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

        objective_and_grad_func = function([budgets_flat], [objective, objective_grad])

        # Avoid repeated input validation for performance
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
