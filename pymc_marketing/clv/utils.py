from typing import Union

import numpy as np
import pandas as pd
import xarray


def to_xarray(customer_id, *arrays, dim: str = "customer_id"):
    """Convert vector arrays to xarray with a common dim (default "customer_id")."""
    dims = (dim,)
    coords = {dim: np.asarray(customer_id)}

    res = tuple(
        xarray.DataArray(data=array, coords=coords, dims=dims) for array in arrays
    )

    if len(arrays) == 1:
        return res[0]
    return res


def customer_lifetime_value(
    transaction_model,
    customer_id: Union[pd.Series, np.ndarray],
    frequency: Union[pd.Series, np.ndarray],
    recency: Union[pd.Series, np.ndarray],
    T: Union[pd.Series, np.ndarray],
    monetary_value: Union[pd.Series, np.ndarray, xarray.DataArray],
    time: int = 12,
    discount_rate: float = 0.01,
    freq: str = "D",
) -> xarray.DataArray:
    """
    Compute the average lifetime value for a group of one or more customers.
    This method computes the average lifetime value for a group of one or more customers.
    It also applies Discounted Cash Flow.

    Adapted from lifetimes package
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L449

    Parameters
    ----------
    transaction_model: CLVModel
        The model to predict future transactions
    customer_id: array_like
        Customer unique identifiers. Must not repeat.
    frequency: array_like
        The frequency vector of customers' purchases (denoted x in literature).
    recency: array_like
        The recency vector of customers' purchases (denoted t_x in literature).
    T: array_like
        The vector of customers' age (time since first purchase)
    monetary_value: array_like
        The monetary value vector of customer's purchases (denoted m in literature).
    time: int, optional
        The lifetime expected for the user in months. Default: 12
    discount_rate: float, optional
        The monthly adjusted discount rate. Default: 1
    freq: string, optional
        Frequency of discrete time steps used to estimate the customer lifetime value.
        Defaults to "D" for daily. Other options are "W" (weekly), "M" (monthly), and "H" (hourly).
        Smaller time frames estimate better the effects of discounting rate, at the cost of more
        evaluations.

    Returns
    -------
    xarray
        DataArray with the estimated customer lifetime values
    """

    steps = np.arange(1, time + 1)
    factor = {"W": 4.345, "M": 1.0, "D": 30, "H": 30 * 24}[freq]

    model_dims = transaction_model.fit_result.posterior.dims
    model_coords = transaction_model.fit_result.posterior.coords
    clv_coords = {
        "chain": model_coords["chain"],
        "draw": model_coords["draw"],
        "customer_id": np.asarray(customer_id),
    }
    clv = xarray.DataArray(
        np.zeros((model_dims["chain"], model_dims["draw"], len(customer_id))),
        dims=("chain", "draw", "customer_id"),
        coords=clv_coords,
    )

    # Monetary value can be passed as a DataArray, with entries per chain and draw or as a simple vector
    if not isinstance(monetary_value, xarray.DataArray):
        monetary_value = to_xarray(customer_id, monetary_value)

    frequency, recency, T = to_xarray(customer_id, frequency, recency, T)

    # TODO: Vectorize computation so that we perform a single call to expected_num_purchases
    prev_expected_num_purchases = transaction_model.expected_num_purchases(
        customer_id=customer_id,
        frequency=frequency,
        recency=recency,
        T=T,
        t=0,
    )
    for i in steps * factor:
        # since the prediction of number of transactions is cumulative, we have to subtract off the previous periods
        new_expected_num_purchases = transaction_model.expected_num_purchases(
            customer_id=customer_id,
            frequency=frequency,
            recency=recency,
            T=T,
            t=i,
        )
        expected_transactions = new_expected_num_purchases - prev_expected_num_purchases
        prev_expected_num_purchases = new_expected_num_purchases

        # sum up the CLV estimates of all the periods and apply discounted cash flow
        clv += (monetary_value * expected_transactions) / (1 + discount_rate) ** (
            i / factor
        )

    return clv
