from typing import Dict, List, Optional, Tuple
import numpy as np
from pandas import DataFrame
from scipy.optimize import minimize
from pymc_marketing.mmm.utils import extense_sigmoid, michaelis_menten

def calculate_expected_contribution(
    method: str,
    parameters: Dict[str, Tuple[float, float]],
    budget: Dict[str, float],
) -> Dict[str, float]:
    total_expected_contribution = 0.0
    contributions = {}

    for channel, channe_budget in budget.items():
        if method == "michaelis-menten":
            L, k = parameters[channel]
            contributions[channel] = michaelis_menten(channe_budget, L, k)

        elif method == "sigmoid":
            alpha, lam = parameters[channel]
            contributions[channel] = extense_sigmoid(channe_budget, alpha, lam)

        else:
            raise ValueError("`method` must be either 'michaelis-menten' or 'sigmoid'.")

        total_expected_contribution += contributions[channel]

    contributions["total"] = total_expected_contribution

    return contributions

def objective_distribution(
    x: List[float],
    method: str,
    channels: List[str],
    parameters: Dict[str, Tuple[float, float]],
) -> float:
    sum_contributions = 0.0

    for channel, budget in zip(channels, x):
        if method == "michaelis-menten":
            L, k = parameters[channel]
            sum_contributions += michaelis_menten(budget, L, k)

        elif method == "sigmoid":
            alpha, lam = parameters[channel]
            sum_contributions += extense_sigmoid(budget, alpha, lam)

        else:
            raise ValueError("`method` must be either 'michaelis-menten' or 'sigmoid'.")

    return -1 * sum_contributions

def optimize_budget_distribution(
    method: str,
    total_budget: int,
    budget_ranges: Optional[Dict[str, Tuple[float, float]]],
    parameters: Dict[str, Tuple[float, float]],
    channels: List[str],
    emissions_per_channel: Dict[str, float],
    max_emissions: float
) -> Dict[str, float]:
    if not isinstance(budget_ranges, (dict, type(None))):
        raise TypeError("`budget_ranges` should be a dictionary or None.")

    if budget_ranges is None:
        budget_ranges = {
            channel: (0, min(total_budget, parameters[channel][0]))
            for channel in channels
        }

    initial_guess = [total_budget // len(channels)] * len(channels)

    bounds = [budget_ranges[channel] for channel in channels]

    budget_constraint = {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}

    emissions_constraint = {'type': 'ineq', 'fun': lambda x: max_emissions - sum(x[i]*emissions_per_channel[channels[i]] for i in range(len(x)))}

    constraints = [budget_constraint, emissions_constraint]

    result = minimize(
        lambda x: objective_distribution(x, method, channels, parameters),
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return {channel: budget for channel, budget in zip(channels, result.x)}

def budget_allocator(
    method: str,
    total_budget: int,
    channels: List[str],
    parameters: Dict[str, Tuple[float, float]],
    budget_ranges: Optional[Dict[str, Tuple[float, float]]],
    emissions_per_channel: Dict[str, float],
    max_emissions: float
) -> DataFrame:
    optimal_budget = optimize_budget_distribution(
        method=method,
        total_budget=total_budget,
        budget_ranges=budget_ranges,
        parameters=parameters,
        channels=channels,
        emissions_per_channel=emissions_per_channel,
        max_emissions=max_emissions
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