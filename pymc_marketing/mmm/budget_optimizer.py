#   Copyright 2024 The PyMC Labs Developers
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
from pytensor import function
from scipy.optimize import minimize

from pymc_marketing.mmm.components.adstock import AdstockTransformation
from pymc_marketing.mmm.components.saturation import SaturationTransformation
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
    adstock : AdstockTransformation
        The adstock class.
    saturation : SaturationTransformation
        The saturation class.
    num_periods : int
        The number of time units.
    parameters : dict
        A dictionary of parameters for each channel.
    scales : np.ndarray
        The scale parameter for each channel variable.
    response_scaler : float, optional
        The scaling factor for the target response variable. Default is 1.
    adstock_first : bool, optional
        Whether to apply adstock transformation first or saturation transformation first.
        Default is True.
    utility_function : UtilityFunctionType, optional
        The utility function to maximize. Default is the mean of the response distribution.

    """

    adstock: AdstockTransformation = Field(
        ..., description="The adstock transformation class."
    )
    saturation: SaturationTransformation = Field(
        ..., description="The saturation transformation class."
    )
    num_periods: int = Field(
        ...,
        gt=0,
        description="The number of time units at time granularity which the budget is to be allocated.",
    )
    parameters: dict[str, Any] = Field(
        ..., description="A dictionary of parameters for each channel."
    )
    scales: np.ndarray = Field(
        ..., description="The scale parameter for each channel variable"
    )
    response_scaler: float = Field(
        default=1.0,
        description="Scaling factor for the target response variable. Defaults to 1.",
    )
    adstock_first: bool = Field(
        True,
        description="Whether to apply adstock transformation first or saturation transformation first.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    response_scaler_sym: pt.TensorVariable = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Response scaler tensor variable.",
    )

    utility_function: UtilityFunctionType = Field(
        default=average_response,
        description="Utility function to maximize.",
        arbitrary_types_allowed=True,
    )

    DEFAULT_MINIMIZE_KWARGS: ClassVar[dict] = {
        "method": "SLSQP",
        "options": {"ftol": 1e-9, "maxiter": 1_000},
    }

    def __init__(self, **data):
        super().__init__(**data)
        self.response_scaler_sym = pt.as_tensor_variable(self.response_scaler)
        self._compiled_functions = {}
        self._compile_objective_and_grad()

    def _compile_objective_and_grad(self):
        """Compile the objective function and its gradient using symbolic computation."""
        budgets_sym = pt.vector("budgets")

        _response_distribution = self._estimate_response(budgets=budgets_sym)

        response_distribution = _response_distribution.sum(axis=(2, 3)).flatten()

        objective_value = -self.utility_function(
            samples=response_distribution, budgets=budgets_sym
        )

        # Compute gradient symbolically
        grad_obj = pt.grad(objective_value, budgets_sym)

        # Compile the functions
        utility_func = function([budgets_sym], objective_value)
        grad_func = function([budgets_sym], grad_obj)

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

    def _estimate_response(self, budgets: list[float]) -> np.ndarray:
        """Calculate the total response during a period of time given the budgets.

        It considers the saturation and adstock transformations.

        Parameters
        ----------
        budgets : list[float]
            The budgets for each channel.

        Returns
        -------
        np.ndarray
            The estimated response distribution.

        """
        first_transform, second_transform = (
            (self.adstock, self.saturation)
            if self.adstock_first
            else (self.saturation, self.adstock)
        )

        # Convert scales to a tensor variable when needed
        budget = budgets / pt.as_tensor_variable(self.scales)

        # Convert parameters to tensor variables if necessary
        def convert_params(params):
            return {
                k: (pt.as_tensor_variable(v) if isinstance(v, np.ndarray) else v)
                for k, v in params.items()
            }

        first_params = convert_params(
            self.parameters["adstock_params"]
            if self.adstock_first
            else self.parameters["saturation_params"]
        )
        second_params = convert_params(
            self.parameters["saturation_params"]
            if self.adstock_first
            else self.parameters["adstock_params"]
        )

        spend = pt.tile(budget, (self.num_periods, 1))
        spend_extended = pt.concatenate(
            [spend, pt.zeros((self.adstock.l_max, spend.shape[1]))], axis=0
        )

        _response = first_transform.function(x=spend_extended, **first_params)

        for param_name, param_value in second_params.items():
            if isinstance(param_value, pt.TensorVariable) and param_value.ndim == 3:
                param_value = param_value.dimshuffle(0, 1, "x", 2)
                second_params[param_name] = param_value

        # Multiply by the response_scaler_sym
        return (
            second_transform.function(x=_response, **second_params)
            * self.response_scaler_sym
        )

    def allocate_budget(
        self,
        total_budget: float,
        budget_bounds: dict[str, tuple[float, float]] | None = None,
        custom_constraints: dict[Any, Any] | None = None,
        minimize_kwargs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, float], float]:
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
        budget_bounds : dict[str, tuple[float, float]], optional
            The budget bounds for each channel. Default is None.
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
            budget_bounds = {
                channel: (0, total_budget) for channel in self.parameters["channels"]
            }
            warnings.warn(
                "No budget bounds provided. Using default bounds (0, total_budget) for each channel.",
                UserWarning,
                stacklevel=2,
            )
        elif not isinstance(budget_bounds, dict):
            raise TypeError("`budget_bounds` should be a dictionary.")

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

        num_channels = len(self.parameters["channels"])
        initial_guess = np.ones(num_channels) * total_budget / num_channels
        bounds = [
            (
                (budget_bounds[channel][0], budget_bounds[channel][1])
                if channel in budget_bounds
                else (0, total_budget)
            )
            for channel in self.parameters["channels"]
        ]

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
            optimal_budgets = {
                name: budget
                for name, budget in zip(
                    self.parameters["channels"], result.x, strict=False
                )
            }
            return optimal_budgets, result
        else:
            raise MinimizeException(f"Optimization failed: {result.message}")
