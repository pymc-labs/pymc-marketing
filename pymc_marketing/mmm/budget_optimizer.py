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

import numpy as np
import pytensor.tensor as pt
from pydantic import BaseModel, ConfigDict, Field
from pymc.logprob.utils import rvs_in_graph
from pymc.model.transform.optimization import freeze_dims_and_data
from pytensor import clone_replace, function
from pytensor.graph import rewrite_graph, vectorize_graph
from pytensor.graph.basic import get_var_by_name
from scipy.optimize import OptimizeResult, minimize
from xarray import DataArray

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

    hmm_model: Any = Field(
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
        self._pymc_model = self.hmm_model._set_predictors_for_optimization(
            self.num_periods
        )
        self._budget_dims = [
            dim
            for dim in self._pymc_model.named_vars_to_dims["channel_data"]
            if dim != "date"
        ]
        self._budget_coords = {
            dim: list(self._pymc_model._coords[dim]) for dim in self._budget_dims
        }
        self._budget_shape = tuple(len(coord) for coord in self._budget_coords.values())

        self._compiled_functions = {}
        self._compile_objective_and_grad()

    def _create_budget_variable(self):
        size_budgets = np.prod(self._budget_shape)
        budgets_flat = pt.tensor("budgets_flat", shape=(size_budgets,))
        budgets = budgets_flat.reshape(self._budget_shape)
        return budgets

    def _compile_objective_and_grad(self):
        """Compile the objective function and its gradient using symbolic computation."""
        budgets = self._create_budget_variable()
        response_distribution = self.extract_response_distribution(budgets=budgets)

        objective_value = -self.utility_function(
            samples=response_distribution, budgets=budgets
        )

        # Compute gradient symbolically
        [budgets_flat] = get_var_by_name([budgets], "budgets_flat")
        grad_obj = pt.grad(objective_value, budgets_flat)

        # Compile the functions
        utility_func = function([budgets_flat], objective_value)
        grad_func = function([budgets_flat], grad_obj)

        # Cache the compiled functions
        self._compiled_functions[self.utility_function] = {
            "objective": utility_func,
            "gradient": grad_func,
        }

    def _objective(self, budgets: pt.TensorVariable) -> float:
        """Objective function for the budget optimization."""
        return self._compiled_functions[self.utility_function]["objective"](
            budgets
        ).item()

    def _gradient(self, budgets: pt.TensorVariable) -> pt.TensorVariable:
        """Gradient of the objective function."""
        return self._compiled_functions[self.utility_function]["gradient"](budgets)

    def extract_response_distribution(
        self, budgets: pt.TensorVariable
    ) -> pt.TensorVariable:
        """Extract the response graph, conditioned on the posterior draws and a placeholder budget variable."""
        if not (isinstance(budgets, pt.TensorVariable)):  # and budgets.type.ndim == 1):
            raise ValueError("budgets must be a TensorVariable")

        num_periods = self.num_periods
        model = self._pymc_model
        posterior = self.hmm_model.idata.posterior  # type: ignore
        max_lag = self.hmm_model.adstock.l_max
        channel_scales = self.hmm_model.scaler.input_scales[
            self.hmm_model.channel_columns
        ].values

        # Freeze all but date dims for a more succinct graph
        coords = self._pymc_model._coords
        model = freeze_dims_and_data(
            model, data=[], dims=[dim for dim in coords if dim != "date"]
        )

        response_variable = model[self.response_variable]

        # Replicate the budget over num_periods and append zeros to also quantify carry-over effects
        channel_data_dims = model.named_vars_to_dims["channel_data"]
        date_dim_idx = list(channel_data_dims).index("date")

        budgets_tiled_shape = list(tuple(budgets.shape))
        budgets_tiled_shape.insert(date_dim_idx, num_periods)
        # TODO: If scales become part of the model, we don't need to transform it here
        budgets /= channel_scales[:, None]
        budgets_tiled = pt.broadcast_to(
            pt.expand_dims(budgets, date_dim_idx), budgets_tiled_shape
        )

        budget_full_shape = list(tuple(budgets.shape))
        budget_full_shape.insert(date_dim_idx, num_periods + max_lag)
        budgets_full = pt.zeros(budget_full_shape)
        set_idxs = (*((slice(None),) * date_dim_idx), slice(None, num_periods))
        budgets_full = budgets_full[set_idxs].set(budgets_tiled)
        budgets_full.name = "budgets_full"

        # Replace model free_RVs by placeholder variables
        placeholder_replace_dict = {
            model[free_RV.name]: pt.tensor(
                name=free_RV.name,
                shape=free_RV.type.shape,
                dtype=free_RV.dtype,
            )
            for free_RV in model.free_RVs
        }

        # Replace the channel_data by the budget variable
        placeholder_replace_dict[model["channel_data"]] = budgets_full

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

        # Replace dummy variables by posterior constants (and vectorize graph)
        replace_dict = {}
        for placeholder in placeholder_replace_dict.values():
            if placeholder.name == "budgets_full":
                continue
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
                "local_eager_useless_unbatched_blockwise",
                "local_useless_unbatched_blockwise",
            ),
        )

        return response_variable_distribution

    def allocate_budget(
        self,
        total_budget: float,
        budget_bounds: dict[str, tuple[float, float]] | None = None,
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
        budget_bounds : DataArray, optional
            Array containing the budget bounds for each channel. Default is None.
        custom_constraints : dict, optional
            Custom constraints for the optimization. Default is None.
        minimize_kwargs : dict, optional
            Additional keyword arguments for the `scipy.optimize.minimize` function. If None, default values are used.
            Method is set to "SLSQP", ftol is set to 1e-9, and maxiter is set to 1_000.

        Returns
        -------
        tuple[dict[str, float], float]
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
            budget_bounds = DataArray(
                np.full(fill_value=[0, total_budget], shape=(*self._budget_shape, 2)),
                dims=[*self._budget_dims, "bound"],
            )
        elif isinstance(budget_bounds, dict):
            # TODO: Backwards compatibility with dict approach?
            raise NotImplementedError("Dict approach is not supported anymore.")
        elif isinstance(budget_bounds, DataArray):
            if not set(budget_bounds.dims) == {*self._budget_dims, "bound"}:
                raise ValueError(
                    f"budget_bounds must be an DataArray with dims {(*self._budget_dims, 'bound')}"
                )
            budget_bounds = budget_bounds.transpose(*self._budget_dims, "bound")
        else:
            raise ValueError("budget_bounds must be an DataArray")

        bounds = [(low, high) for low, high in budget_bounds.values.reshape(-1, 2)]  # type: ignore

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

        [budgets_size] = get_var_by_name(
            [self._create_budget_variable()], "budgets_flat"
        )[0].type.shape
        initial_guess = np.ones(budgets_size) * total_budget / budgets_size

        if minimize_kwargs is None:
            minimize_kwargs = self.DEFAULT_MINIMIZE_KWARGS.copy()
        else:
            minimize_kwargs = {**self.DEFAULT_MINIMIZE_KWARGS, **minimize_kwargs}

        result = minimize(
            fun=self._objective,
            x0=initial_guess,
            bounds=bounds,
            constraints=constraints,
            jac=self._gradient,
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
