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
from typing import Any, ClassVar

import arviz as az
import numpy as np
import pytensor.tensor as pt
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field
from pymc import Model, do
from pymc.logprob.utils import rvs_in_graph
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.pytensorf import rewrite_pregrad
from pytensor import clone_replace, function
from pytensor.compile.sharedvalue import SharedVariable, shared
from pytensor.graph import rewrite_graph, vectorize_graph
from pytensor.graph.basic import ancestors
from scipy.optimize import OptimizeResult, minimize
from xarray import DataArray

from pymc_marketing.mmm.constraints import (
    Constraint,
    build_default_sum_constraint,
    compile_constraints_for_scipy,
)
from pymc_marketing.mmm.mmm import MMM
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
    model: MMMModel
        The marketing mix model to optimize.
    utility_function : UtilityFunctionType, optional
        The utility function to maximize. Default is the mean of the response distribution.
    """

    num_periods: int = Field(
        ...,
        gt=0,
        description="Number of time units at the desired time granularity to allocate budget for.",
    )

    mmm_model: MMM = Field(
        ...,
        description="The marketing mix model to optimize.",
        arbitrary_types_allowed=True,
        alias="model",
    )

    response_variable: str = Field(
        default="total_contributions",
        description="The response variable to optimize.",
    )

    utility_function: UtilityFunctionType = Field(
        default=average_response,
        description="Utility function to maximize.",
        arbitrary_types_allowed=True,
    )

    budgets_to_optimize: DataArray | None = Field(
        default=None,
        description="Mask that defines a subset of budgets to optimize. Non-optimized budgets remain fixed at 0.",
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    DEFAULT_MINIMIZE_KWARGS: ClassVar[dict] = {
        "method": "SLSQP",
        "options": {"ftol": 1e-9, "maxiter": 1_000},
    }

    def __init__(self, **data):
        super().__init__(**data)
        # 1. Prepare model with time dimension for optimization
        pymc_model = self.mmm_model._set_predictors_for_optimization(self.num_periods)

        # 2. Shared variable for total_budget
        self._total_budget: SharedVariable = shared(
            np.array(0.0, dtype="float64"), name="total_budget"
        )

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

        # 4. Handle mask vs. no mask in flattening
        if self.budgets_to_optimize is None:
            size_budgets = np.prod(self._budget_shape)
        else:
            size_budgets = self.budgets_to_optimize.sum().item()

        self._budgets_flat = pt.tensor("budgets_flat", shape=(size_budgets,))

        if self.budgets_to_optimize is None:
            # No mask case: reshape entire vector
            self._budgets = self._budgets_flat.reshape(self._budget_shape)
        else:
            # Masked case: fill a zero array, then set only the True positions
            budgets_zeros = pt.zeros(self._budget_shape, name="budgets_zeros")
            bool_mask = np.asarray(self.budgets_to_optimize).astype(bool)
            self._budgets = budgets_zeros[bool_mask].set(self._budgets_flat)

        # 5. Replace channel_data with budgets in the PyMC model
        self._pymc_model = self._replace_channel_data_by_optimization_variable(
            pymc_model
        )

        # 6. Compile objective & gradient
        self._compiled_functions = {}
        self._compile_objective_and_grad()

        # 7. Build constraints
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
        repeated_budgets_with_carry_over = pt.zeros(
            repeated_budgets_with_carry_over_shape
        )
        set_idxs = (*((slice(None),) * date_dim_idx), slice(None, num_periods))
        repeated_budgets_with_carry_over = repeated_budgets_with_carry_over[
            set_idxs
        ].set(repeated_budgets)
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
        `BudgetOptimizer(...).extract_response_distribution("channel_contributions")`
        returns a graph that computes `"channel_contributions"` as a function of both
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

        objective_func = function([budgets_flat], objective)
        grad_func = function([budgets_flat], objective_grad)

        # Avoid repeated input validation for performance
        objective_func.trust_input = True
        grad_func.trust_input = True

        self._compiled_functions[self.utility_function] = {
            "objective": objective_func,
            "gradient": grad_func,
        }

    def allocate_budget(
        self,
        total_budget: float,
        budget_bounds: DataArray | dict[str, tuple[float, float]] | None = None,
        minimize_kwargs: dict[str, Any] | None = None,
    ) -> tuple[DataArray, OptimizeResult]:
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
        minimize_kwargs : dict, optional
            Extra kwargs for `scipy.optimize.minimize`. Defaults to method "SLSQP",
            ftol=1e-9, maxiter=1_000.

        Returns
        -------
        optimal_budgets : xarray.DataArray
            The optimized budget allocation across channels.
        result : OptimizeResult
            The raw scipy optimization result.

        Raises
        ------
        MinimizeException
            If the optimization fails for any reason, the exception message will contain the details.
        """
        self._total_budget.set_value(np.asarray(total_budget, dtype="float64"))

        # 1. Process budget bounds
        if budget_bounds is None:
            warnings.warn(
                "No budget bounds provided. Using default bounds (0, total_budget) for each channel.",
                UserWarning,
                stacklevel=2,
            )
            budget_bounds_array = np.array(
                [[0, total_budget]] * np.prod(self._budget_shape)
            )
        elif isinstance(budget_bounds, dict):
            if len(self._budget_dims) > 1:
                raise ValueError(
                    f"Dict approach to budget_bounds is not supported for multi-dimensional budgets. "
                    f"budget_dims = {self._budget_dims}. Pass an xarray.DataArray instead."
                )
            # Flatten each channel's bounds into an array
            budget_bounds_array = np.concatenate(
                [
                    np.asarray(budget_bounds[channel])
                    for channel in self.mmm_model.channel_columns
                ],
                axis=0,
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
        if self.budgets_to_optimize is None:
            bounds = [(low, high) for (low, high) in budget_bounds_array.reshape(-1, 2)]
        else:
            # Only gather bounds for the True positions in the mask
            bounds = [
                (low, high)
                for (low, high) in budget_bounds_array[self.budgets_to_optimize.values]
            ]

        # 4. Determine how many budget entries we optimize
        if self.budgets_to_optimize is None:
            budgets_size = np.prod(self._budget_shape)
        else:
            budgets_size = self.budgets_to_optimize.sum().item()

        # 5. Create an initial guess
        initial_guess = np.ones(budgets_size) * (total_budget / budgets_size)
        initial_guess = initial_guess.astype(self._budgets_flat.type.dtype)

        if minimize_kwargs is None:
            minimize_kwargs = self.DEFAULT_MINIMIZE_KWARGS.copy()
        else:
            # Merge with defaults (preferring user-supplied keys)
            minimize_kwargs = {**self.DEFAULT_MINIMIZE_KWARGS, **minimize_kwargs}

        # 6. Run the SciPy optimizer
        result = minimize(
            fun=self._compiled_functions[self.utility_function]["objective"],
            jac=self._compiled_functions[self.utility_function]["gradient"],
            x0=initial_guess,
            bounds=bounds,
            constraints=self._compiled_constraints,
            **minimize_kwargs,
        )

        # 7. Process results
        if result.success:
            if self.budgets_to_optimize is None:
                # Reshape the entire optimized solution
                optimal_budgets = np.reshape(result.x, self._budget_shape)
            else:
                # Fill zeros, then place the solution in masked positions
                optimal_budgets = np.zeros_like(
                    self.budgets_to_optimize.values, dtype=float
                )
                optimal_budgets[self.budgets_to_optimize.values] = result.x

            optimal_budgets = DataArray(
                optimal_budgets, dims=self._budget_dims, coords=self._budget_coords
            )
            return optimal_budgets, result

        else:
            raise MinimizeException(f"Optimization failed: {result.message}")
