# optimization_utils.py
from typing import List

import numpy as np
from pandas import DataFrame
from xarray import DataArray

from pymc_marketing.mmm.utils import CurveCalculator


def cost_function(
    budget_allocations: List[float],
    df: DataFrame,
    data: DataFrame,
    channel_contributions: DataArray,
) -> float:
    """
    Compute the total collaboration of budget allocation across various channels.

    Parameters
    ----------
    budget_allocations : list of float
        The budget allocated to each channel.
    df : pandas DataFrame
        The DataFrame containing data about each channel.
    data : pandas DataFrame
        The DataFrame containing the x-values for the quadratic fit.
    channel_contributions : xarray DataArray
        The channel contributions.

    Returns
    -------
    float
        The total collaboration of the budget allocations.

    Notes
    -----
    This function calculates the total collaboration of budget allocations for each channel based on a quadratic curve fitted to the channel data.
    The total collaboration is computed as the sum of the polynomial evaluation at the allocation for each channel.
    """
    total_contribution = 0
    for i, allocation in enumerate(budget_allocations):
        # Retrieve the data for this channel
        channel = df.loc[i, "channel"]
        x = data[channel].to_numpy().copy()
        y = np.array(channel_contributions.sel(channel=channel).copy())

        # Fit a quadratic curve to the data
        calculator = CurveCalculator(x, y)
        polynomial = calculator.polynomial

        # Calculate the contribution for this channel based on its quadratic curve
        total_contribution -= polynomial(allocation)
    return total_contribution


def budget_constraint(budget_allocations: List[float], total_budget: float) -> float:
    """
    Compute the budget constraint for the optimization problem.

    Parameters
    ----------
    budget_allocations : list of float
        The budget allocated to each channel.
    total_budget : float
        The total available budget.

    Returns
    -------
    float
        The difference between the total allocated budget and the total available budget.

    Notes
    -----
    This function calculates the difference between the total allocated budget and the total available budget.
    This difference should be zero in the optimization problem.
    """
    total_budget_allocated = np.sum(budget_allocations)
    return total_budget_allocated - total_budget  # this should be zero
