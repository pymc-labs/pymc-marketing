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

import numpy as np
from pandas import DataFrame
from scipy.optimize import minimize

from pymc_marketing.mmm.transformers import michaelis_menten
from pymc_marketing.mmm.utils import sigmoid_saturation


def calculate_expected_contribution(
    method: str,
    parameters: dict[str, tuple[float, float]],
    budget: dict[str, float],
) -> dict[str, float]:
    """
    Calculate expected contributions using the specified model.

    This function calculates the expected contributions for each channel
    based on the chosen model. The selected model can be either the Michaelis-Menten
    model or the sigmoid model, each described by specific parameters.
    As the allocated budget varies, the expected contribution is computed according
    to the chosen model.

    Parameters
    ----------
    method : str
        The model to use for contribution estimation. Choose from 'michaelis-menten' or 'sigmoid'.
    parameters : Dict
        Model-specific parameters for each channel. For 'michaelis-menten', each entry is a tuple (L, k) where:
        - L is the maximum potential contribution.
        - k is the budget at which the contribution is half of its maximum.

        For 'sigmoid', each entry is a tuple (alpha, lam) where:
        - alpha controls the slope of the curve.
        - lam is the budget at which the curve transitions.
    budget : Dict
        The total budget.

    Returns
    -------
    Dict
        A dictionary with channels as keys and their respective contributions as values.
        The key 'total' contains the total expected contribution across all channels.

    Raises
    ------
    ValueError
        If the specified `method` is not recognized.
    """

    total_expected_contribution = 0.0
    contributions = {}

    for channel, channe_budget in budget.items():
        if method == "michaelis-menten":
            L, k = parameters[channel]
            contributions[channel] = michaelis_menten(channe_budget, L, k)

        elif method == "sigmoid":
            alpha, lam = parameters[channel]
            contributions[channel] = sigmoid_saturation(channe_budget, alpha, lam)

        else:
            raise ValueError("`method` must be either 'michaelis-menten' or 'sigmoid'.")

        total_expected_contribution += contributions[channel]

    contributions["total"] = total_expected_contribution

    return contributions


def objective_distribution(
    x: list[float],
    method: str,
    channels: list[str],
    parameters: dict[str, tuple[float, float]],
) -> float:
    """
    Compute the total contribution for a given budget distribution.

    This function calculates the negative sum of contributions for a proposed budget
    distribution using the Michaelis-Menten model. This value will be minimized in
    the optimization process to maximize the total expected contribution.

    Parameters
    ----------
    x : List of float
        The proposed budget distribution across channels.
    channels : List of str
        The List of channels for which the budget is being optimized.
    parameters : Dict
        Michaelis-Menten parameters for each channel as described in `calculate_expected_contribution`.

    Returns
    -------
    float
        Negative of the total expected contribution for the given budget distribution.
    """

    sum_contributions = 0.0

    for channel, budget in zip(channels, x, strict=False):
        if method == "michaelis-menten":
            L, k = parameters[channel]
            sum_contributions += michaelis_menten(budget, L, k)

        elif method == "sigmoid":
            alpha, lam = parameters[channel]
            sum_contributions += sigmoid_saturation(budget, alpha, lam)

        else:
            raise ValueError("`method` must be either 'michaelis-menten' or 'sigmoid'.")

    return -1 * sum_contributions


def optimize_budget_distribution(
    method: str,
    total_budget: int,
    budget_ranges: dict[str, tuple[float, float]] | None,
    parameters: dict[str, tuple[float, float]],
    channels: list[str],
) -> dict[str, float]:
    """
    Optimize the budget allocation across channels to maximize total contribution.

    Using the Michaelis-Menten or Sigmoid function, this function seeks the best budget distribution across
    channels that maximizes the total expected contribution.

    This function leverages the Sequential Least Squares Quadratic Programming (SLSQP) optimization
    algorithm to find the best budget distribution across channels that maximizes the total
    expected contribution based on the Michaelis-Menten or Sigmoid functions.

    The optimization is constrained such that:
    1. The sum of budgets across all channels equals the total available budget.
    2. The budget allocated to each individual channel lies within its specified range.

    The SLSQP method is particularly suited for this kind of problem as it can handle
    both equality and inequality constraints.

    Parameters
    ----------
    total_budget : int
        The total budget to be distributed across channels.
    budget_ranges : Dict or None
        An optional dictionary defining the minimum and maximum budget for each channel.
        If not provided, the budget for each channel is constrained between 0 and its L value.
    parameters : Dict
        Michaelis-Menten parameters for each channel as described in `calculate_expected_contribution`.
    channels : list of str
        The list of channels for which the budget is being optimized.

    Returns
    -------
    Dict
        A dictionary with channels as keys and the optimal budget for each channel as values.
    """

    # Check if budget_ranges is the correct type
    if not isinstance(budget_ranges, dict | type(None)):
        raise TypeError("`budget_ranges` should be a dictionary or None.")

    if budget_ranges is None:
        budget_ranges = {
            channel: (0, min(total_budget, parameters[channel][0]))
            for channel in channels
        }

    initial_guess = [total_budget // len(channels)] * len(channels)

    bounds = [budget_ranges[channel] for channel in channels]

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}

    result = minimize(
        lambda x: objective_distribution(x, method, channels, parameters),
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return {
        channel: budget for channel, budget in zip(channels, result.x, strict=False)
    }


def budget_allocator(
    method: str,
    total_budget: int,
    channels: list[str],
    parameters: dict[str, tuple[float, float]],
    budget_ranges: dict[str, tuple[float, float]] | None,
) -> DataFrame:
    optimal_budget = optimize_budget_distribution(
        method=method,
        total_budget=total_budget,
        budget_ranges=budget_ranges,
        parameters=parameters,
        channels=channels,
    )

    expected_contribution = calculate_expected_contribution(
        method=method, parameters=parameters, budget=optimal_budget
    )

    optimal_budget.update({"total": sum(optimal_budget.values())})

    return DataFrame(
        {
            "estimated_contribution": expected_contribution,
            "optimal_budget": optimal_budget,
        }
    )
