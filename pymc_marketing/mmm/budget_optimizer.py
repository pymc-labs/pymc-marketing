#   Copyright 2025 The PyMC Labs Developers
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
from typing import Any, ClassVar

import arviz as az
import numpy as np
import pytensor.tensor as pt
from pydantic import BaseModel, ConfigDict, Field
from pymc import Model, do
from pymc.logprob.utils import rvs_in_graph
from pymc.model.transform.optimization import freeze_dims_and_data
from pytensor import clone_replace, function
from pytensor.graph import rewrite_graph, vectorize_graph
from pytensor.graph.basic import ancestors
from scipy.optimize import OptimizeResult, minimize
from xarray import DataArray

from pymc_marketing.mmm.mmm import MMM
from pymc_marketing.mmm.utility import UtilityFunctionType, average_response


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
        description="The number of time units at time granularity which the budget is to be allocated.",
    )

    mmm_model: MMM = Field(
        ...,
        description="The marketing mix model to optimize.",
        arbitrary_types_allowed=True,
        alias="model",
    )

    response_variable: str = Field(
        default="channel_contributions",
        description="The response variable to optimize.",
    )

    utility_function: UtilityFunctionType = Field(
        default=average_response,
        description="Utility function to maximize.",
        arbitrary_types_allowed=True,
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    DEFAULT_MINIMIZE_KWARGS: ClassVar[dict] = {
        "method": "SLSQP",
        "options": {"ftol": 1e-9, "maxiter": 1_000},
    }

    def __init__(self, **data):
        super().__init__(**data)
        pymc_model = self.mmm_model._set_predictors_for_optimization(self.num_periods)
        self._budget_dims = [
            dim
            for dim in pymc_model.named_vars_to_dims["channel_data"]
            if dim != "date"
        ]
        self._budget_coords = {
            dim: list(pymc_model.coords[dim]) for dim in self._budget_dims
        }
        self._budget_shape = tuple(len(coord) for coord in self._budget_coords.values())
        # Flattened variable that will be optimized
        self._budgets_flat = pt.tensor(
            "budgets_flat", shape=(np.prod(self._budget_shape),)
        )
        # Unflattened variable used in utility/constraint functions
        self._budgets = self._budgets_flat.reshape(self._budget_shape)
        self._pymc_model = self._replace_channel_data_by_optimization_variable(
            pymc_model
        )

        self._compiled_functions = {}
        self._compile_objective_and_grad()

    def _replace_channel_data_by_optimization_variable(self, model: Model) -> Model:
        """Replace the deterministic `"channel_data"` in the model by the optimization variable."""
        num_periods = self.num_periods
        max_lag = self.mmm_model.adstock.l_max
        channel_data_dims = model.named_vars_to_dims["channel_data"]
        date_dim_idx = list(channel_data_dims).index("date")
        channel_scales = self.mmm_model._channel_scales

        # The optimization budgets variable has the same shape as the channel_data, but without the date dimension
        budgets = self._budgets
        budgets /= channel_scales

        # Replicate the budgets over num_periods and append zeros to also quantify carry-over effects
        # If there's a single channel, and num_peridos=2, and max_lag=1, then budget_full==[budgets[0], budget[0], 0]
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

        # Freeze the dimensions of the model to simplify graph
        # We don't try to freeze data variables because they may have incorrect shapes on the date dimension
        model = freeze_dims_and_data(model, data=[])

        # Replace the deterministic channel_data by the optimization variable using do
        return do(model, {"channel_data": repeated_budgets_with_carry_over})

    def extract_response_distribution(
        self, response_variable: str
    ) -> pt.TensorVariable:
        """Extract the response distribution graph, conditioned on posterior parameters and the optimization variable.

        `BudgetOptimizer(...).extract_response_distribution("channel_contributions")`
        returns the graph that computes the `"channel_contributions"` variable defined in the PyMC model,
        as a function of the optimization variable and the posterior of the model parameters.
        The graph will have shape `(chains * draws, *channel_contributions.shape)`.

        """
        model = self._pymc_model
        # Stack chain/draws dimensions
        posterior = az.extract(self.mmm_model.idata).transpose("sample", ...)  # type: ignore

        response_variable = model[response_variable]

        # Replace model free_RVs that are needed to compute `response_variable` by placeholder variables
        free_rvs = set(model.free_RVs)
        needed_rvs = [
            rv
            for rv in ancestors([response_variable], blockers=free_rvs)
            if rv in free_rvs
        ]
        placeholder_replace_dict = {
            model[rv.name]: pt.tensor(name=rv.name, shape=rv.type.shape, dtype=rv.dtype)
            for rv in needed_rvs
        }

        [response_variable] = clone_replace(
            [response_variable],
            replace=placeholder_replace_dict,
        )

        if rvs_in_graph([response_variable]):
            raise RuntimeError("RVs found in the extracted graph, this is likely a bug")

        # Cleanup graph prior to vectorization
        response_variable = rewrite_graph(
            response_variable, include=("canonicalize", "ShapeOpt")
        )

        # Replace dummy variables by posterior constants (while vectorizing graph)
        replace_dict = {}
        for placeholder in placeholder_replace_dict.values():
            replace_dict[placeholder] = pt.constant(
                posterior[placeholder.name].astype(placeholder.dtype),
                name=placeholder.name,
            )

        response_variable_distribution = vectorize_graph(
            response_variable, replace=replace_dict
        )

        # Final cleanup of the vectorize graph.
        # This shouldn't be needed, vectorize should just not do anything if there are no batch dims!
        response_variable_distribution = rewrite_graph(
            response_variable_distribution,
            include=(
                "useless",
                "local_eager_useless_unbatched_blockwise",
                "local_useless_unbatched_blockwise",
            ),
        )

        return response_variable_distribution

    def _compile_objective_and_grad(self):
        """Compile the objective function and its gradient using symbolic computation."""
        budgets_flat = self._budgets_flat
        budgets = self._budgets
        response_distribution = self.extract_response_distribution(
            self.response_variable
        )

        objective = -self.utility_function(
            samples=response_distribution, budgets=budgets
        )

        # Compute gradient symbolically
        objective_grad = pt.grad(objective, budgets_flat)

        # Compile the functions
        objective_func = function([budgets_flat], objective)
        grad_func = function([budgets_flat], objective_grad)

        # Set trust_input=True to avoid expensive input validation in every call
        objective_func.trust_input = True
        grad_func.trust_input = True

        # Store the compiled functions
        self._compiled_functions[self.utility_function] = {
            "objective": objective_func,
            "gradient": grad_func,
        }

    def allocate_budget(
        self,
        total_budget: float,
        budget_bounds: DataArray | dict[str, tuple[float, float]] | None = None,
        custom_constraints: dict[Any, Any] | None = None,
        minimize_kwargs: dict[str, Any] | None = None,
    ) -> tuple[DataArray, OptimizeResult]:
        """Allocate the budget based on the total budget, budget bounds, and custom constraints.

        The default budget bounds are (0, total_budget) for each channel.

        The default constraint ensures the sum of all budgets equals the total budget.

        The optimization is done using the Sequential Least Squares Quadratic Programming (SLSQP) method
        and it's constrained such that:
        1. The sum of budgets across all channels equals the total available budget.
        2. The budget allocated to each individual channel lies within its specified range.

        Parameters
        ----------
        total_budget : float
            The total budget.
        budget_bounds : DataArray or dict, optional
            DataArray or dict containing the budget bounds for each channel. Default is None.
        custom_constraints : dict, optional
            Custom constraints for the optimization. Default is None.
        minimize_kwargs : dict, optional
            Additional keyword arguments for the `scipy.optimize.minimize` function. If None, default values are used.
            Method is set to "SLSQP", ftol is set to 1e-9, and maxiter is set to 1_000.

        Returns
        -------
        DataArray, OptimizeResult
            The optimal budgets for each channel and the optimization result object.

        Raises
        ------
        MinimizeException
            If the optimization fails, an exception is raised with the reason for the failure.

        """
        if budget_bounds is None:
            warnings.warn(
                "No budget bounds provided. Using default bounds (0, total_budget) for each channel.",
                UserWarning,
                stacklevel=2,
            )
            budget_bounds_array = np.full(
                fill_value=[0, total_budget], shape=(*self._budget_shape, 2)
            )

        elif isinstance(budget_bounds, dict):
            # For backwards compatibility
            if len(self._budget_dims) > 1:
                raise ValueError(
                    f"Dict approach to budget_bounds is not supported for budgets with more than one dimension. "
                    f"budget_dims = {self._budget_dims}. Pass an xarray.Datarray instead"
                )
            budget_bounds_array = np.concatenate(
                [
                    np.asarray(budget_bounds[channel])
                    for channel in self.mmm_model.channel_columns
                ],
                axis=0,
            )

        elif isinstance(budget_bounds, DataArray):
            if not set(budget_bounds.dims) == {*self._budget_dims, "bound"}:
                raise ValueError(
                    f"budget_bounds must be an DataArray with dims {(*self._budget_dims, 'bound')}"
                )
            budget_bounds_array = budget_bounds.transpose(
                *self._budget_dims, "bound"
            ).values
        else:
            raise ValueError(
                "budget_bounds must be a dictionary or an xarray.DataArray"
            )

        bounds = [(low, high) for low, high in budget_bounds_array.reshape(-1, 2)]  # type: ignore

        if custom_constraints is None:
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}
            warnings.warn(
                "Using default equality constraint: The sum of all budgets should be equal to the total budget.",
                UserWarning,
                stacklevel=2,
            )
        elif not isinstance(custom_constraints, dict):
            raise TypeError("`custom_constraints` should be a dictionary.")
        else:
            constraints = custom_constraints

        budgets_size = np.prod(self._budget_shape)
        initial_guess = np.ones(budgets_size) * total_budget / budgets_size
        initial_guess = initial_guess.astype(self._budgets_flat.type.dtype)

        if minimize_kwargs is None:
            minimize_kwargs = self.DEFAULT_MINIMIZE_KWARGS.copy()
        else:
            minimize_kwargs = {**self.DEFAULT_MINIMIZE_KWARGS, **minimize_kwargs}

        result = minimize(
            fun=self._compiled_functions[self.utility_function]["objective"],
            jac=self._compiled_functions[self.utility_function]["gradient"],
            x0=initial_guess,
            bounds=bounds,
            constraints=constraints,
            **minimize_kwargs,
        )

        if result.success:
            optimal_budgets = np.reshape(result.x, self._budget_shape)
            optimal_budgets = DataArray(
                optimal_budgets, dims=self._budget_dims, coords=self._budget_coords
            )

            return optimal_budgets, result
        else:
            raise MinimizeException(f"Optimization failed: {result.message}")
