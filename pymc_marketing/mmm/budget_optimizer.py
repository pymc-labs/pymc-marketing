# optimization_utils.py
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pandas import DataFrame
from scipy.optimize import minimize

from pymc_marketing.mmm.utils import michaelis_menten


def calculate_expected_contribution(parameters, optimal_budget):
    """
    Calculate the total expected contribution of budget allocations across various channels.

    Returns
    -------
    dict
        A dictionary with channels as keys and their respective contributions as values.
        The key 'total' contains the total expected contribution.
    """

    total_expected_contribution = 0
    contributions = {}

    for channel, budget in optimal_budget.items():
        L, k = parameters[channel]
        contributions[channel] = michaelis_menten(budget, L, k)
        total_expected_contribution += contributions[channel]

    contributions["total"] = total_expected_contribution

    return contributions


def objective_distribution(x, channels, parameters):
    """
    Calculate the objective function value for a given budget distribution.

    Parameters
    ----------
    x : list of float
        The budget distribution across channels.

    Returns
    -------
    float
        The value of the objective function given the budget distribution.
    """

    sum_contributions = 0

    for channel, budget in zip(channels, x):
        L, k = parameters[channel]
        sum_contributions += michaelis_menten(budget, L, k)

    return -1 * sum_contributions


def optimize_budget_distribution(total_budget, budget_ranges, parameters, channels):
    """
    Calculate the optimal budget distribution that minimizes the objective function.

    Returns
    -------
    dict
        A dictionary with channels as keys and the optimal budget for each channel as values.
    """

    # Check if budget_ranges is the correct type
    if not isinstance(budget_ranges, (dict, type(None))):
        raise TypeError("`budget_ranges` should be a dictionary or None.")

    if budget_ranges is None:
        budget_ranges = {
            channel: [0, min(total_budget, parameters[channel][0])]
            for channel in channels
        }

    initial_guess = [total_budget / len(channels)] * len(channels)

    bounds = [budget_ranges[channel] for channel in channels]

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}

    result = minimize(
        objective_distribution,
        initial_guess,
        args=(channels, parameters),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return {channel: budget for channel, budget in zip(channels, result.x)}


def budget_allocator(
    total_budget: int = 1000,
    channels: Union[List[str], Tuple[str]] = [],
    parameters: Optional[Dict[str, Tuple[float, float]]] = {},
    budget_ranges: Optional[Dict[str, Tuple[float, float]]] = {},
) -> DataFrame:

    optimal_budget = optimize_budget_distribution(
        total_budget, budget_ranges, parameters, channels
    )

    return DataFrame(
        {
            "estimated_contribution": calculate_expected_contribution(
                parameters, optimal_budget
            ),
            "optimal_budget": optimal_budget,
        }
    )
