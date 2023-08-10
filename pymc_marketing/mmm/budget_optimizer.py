# optimization_utils.py
from typing import Dict, List, Optional, Tuple

import numpy as np
from pandas import DataFrame
from scipy.optimize import minimize

from pymc_marketing.mmm.utils import michaelis_menten


def calculate_expected_contribution(
    parameters: Dict[str, Tuple[float, float]], optimal_budget: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate expected contributions using the Michaelis-Menten model.

    The Michaelis-Menten model describes the relationship between the allocated budget and
    its expected contribution. As the budget increases, the contribution initially rises quickly
    but eventually plateaus, highlighting diminishing returns on investment.

    Parameters
    ----------
    parameters : Dict
        The Michaelis-Menten parameters for each channel. Each entry is a tuple (L, k) where:
        - L is the maximum potential contribution.
        - k is the budget at which the contribution is half of its maximum.
    optimal_budget : Dict
        The optimized budget allocations for each channel.

    Returns
    -------
    Dict
        A dictionary with channels as keys and their respective contributions as values.
        The key 'total' contains the total expected contribution.
    """

    total_expected_contribution = 0.0
    contributions = {}

    for channel, budget in optimal_budget.items():
        L, k = parameters[channel]
        contributions[channel] = michaelis_menten(budget, L, k)
        total_expected_contribution += contributions[channel]

    contributions["total"] = total_expected_contribution

    return contributions


def objective_distribution(
    x: List[float], channels: List[str], parameters: Dict[str, Tuple[float, float]]
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

    for channel, budget in zip(channels, x):
        L, k = parameters[channel]
        sum_contributions += michaelis_menten(budget, L, k)

    return -1 * sum_contributions


def optimize_budget_distribution(
    total_budget: int,
    budget_ranges: Optional[Dict[str, Tuple[float, float]]],
    parameters: Dict[str, Tuple[float, float]],
    channels: List[str],
) -> Dict[str, float]:
    """
    Optimize the budget allocation across channels to maximize total contribution.

    Using the Michaelis-Menten model, this function seeks the best budget distribution across
    channels that maximizes the total expected contribution.

    This function leverages the Sequential Least Squares Quadratic Programming (SLSQP) optimization
    algorithm to find the best budget distribution across channels that maximizes the total
    expected contribution based on the Michaelis-Menten model.

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
    if not isinstance(budget_ranges, (dict, type(None))):
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
        objective_distribution,
        initial_guess,
        args=(channels, parameters),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return {channel: budget for channel, budget in zip(channels, result.x)}


def budget_allocator(
    total_budget: int,
    channels: List[str],
    parameters: Dict[str, Tuple[float, float]],
    budget_ranges: Optional[Dict[str, Tuple[float, float]]],
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
