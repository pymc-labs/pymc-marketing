from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
import xarray

__all__ = ["to_xarray", "customer_lifetime_value", "clv_summary"]


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


def _find_first_transactions(
    transactions: pd.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str = None,
    datetime_format: str = None,
    observation_period_end: Union[str, pd.Period, datetime] = None,
    time_unit: str = "D",
) -> pd.DataFrame:
    """
    Return dataframe with first transactions.

    This takes a DataFrame of transaction data of the form:
        customer_id, datetime [, monetary_value]
    and appends a column named 'repeated' to the transaction log which indicates which rows
    are repeated transactions for that customer_id.

    Adapted from lifetimes package
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L148

    Parameters
    ----------
    transactions: :obj: DataFrame
        A Pandas DataFrame that contains the customer_id col and the datetime col.
    customer_id_col: string
        Column in the transactions DataFrame that denotes the customer_id.
    datetime_col:  string
        Column in the transactions DataFrame that denotes the datetime the purchase was made.
    monetary_value_col: string, optional
        Column in the transactions DataFrame that denotes the monetary value of the transaction.
        Optional; only needed for spend estimation models like the Gamma-Gamma model.
    observation_period_end: :obj: datetime
        A string or datetime to denote the final date of the study.
        Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    datetime_format: string, optional
        A string that represents the timestamp format. Useful if Pandas can't understand
        the provided format.
    time_unit: string, optional
        Time granularity for study.
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    """

    select_columns = [customer_id_col, datetime_col]

    if monetary_value_col:
        select_columns.append(monetary_value_col)

    transactions = transactions[select_columns].sort_values(select_columns).copy()

    # convert date column into a DateTimeIndex for time-wise grouping and truncating
    transactions[datetime_col] = pd.to_datetime(
        transactions[datetime_col], format=datetime_format
    )
    transactions = (
        transactions.set_index(datetime_col).to_period(time_unit).to_timestamp()
    )

    transactions = transactions.loc[
        (transactions.index <= observation_period_end)
    ].reset_index()

    period_groupby = transactions.groupby(
        [datetime_col, customer_id_col], sort=False, as_index=False
    )

    if monetary_value_col:
        # when processing a monetary column, make sure to sum together transactions made in the same period
        period_transactions = period_groupby.sum()
    else:
        # by calling head() on the groupby object, the datetime and customer_id columns
        # will be reduced to the first transaction of that time period
        period_transactions = period_groupby.head(1)

    # create a new column for flagging first transactions
    period_transactions = period_transactions.copy()
    period_transactions.loc[:, "first"] = False
    # find all first transactions and store as an index
    first_transactions = (
        period_transactions.groupby(customer_id_col, sort=True, as_index=False)
        .head(1)
        .index
    )
    # flag first transactions as True
    period_transactions.loc[first_transactions, "first"] = True
    select_columns.append("first")
    # reset datetime_col to period
    period_transactions.loc[:, datetime_col] = pd.Index(
        period_transactions[datetime_col]
    ).to_period(time_unit)

    return period_transactions[select_columns]


def clv_summary(
    transactions: pd.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str = None,
    datetime_format: str = None,
    observation_period_end: Union[str, pd.Period, datetime] = None,
    time_unit: str = "D",
    time_scaler: float = 1,
) -> pd.DataFrame:
    """
    Summarize transaction data for modeling.

    This transforms a DataFrame of transaction data of the form:
        customer_id, datetime [, monetary_value]
    to a DataFrame of the form:
        customer_id, frequency, recency, T [, monetary_value]

    Adapted from lifetimes package
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L230

    Parameters
    ----------
    transactions: :obj: DataFrame
        A Pandas DataFrame that contains the customer_id col and the datetime col.
    customer_id_col: string
        Column in the transactions DataFrame that denotes the customer_id.
    datetime_col:  string
        Column in the transactions DataFrame that denotes the datetime the purchase was made.
    monetary_value_col: string, optional
        Column in the transactions DataFrame that denotes the monetary value of the transaction.
        Optional; only needed for spend estimation models like the Gamma-Gamma model.
    observation_period_end: datetime, optional
         A string or datetime to denote the final date of the study.
         Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    datetime_format: string, optional
        A string that represents the timestamp format. Useful if Pandas can't understand
        the provided format.
    time_unit: string, optional
        Time granularity for study.
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    time_scaler: int, optional
        Default: 1. Useful for scaling recency & T to a different time granularity. Example:
        With freq='D' and freq_multiplier=1, we get recency=591 and T=632
        With freq='h' and freq_multiplier=24, we get recency=590.125 and T=631.375
        This is useful if predictions in a different time granularity are desired,
        and can also help with model convergence for study periods of many years.

    Returns
    -------
    :obj: DataFrame:
        customer_id, frequency, recency, T [, monetary_value]
    """

    if observation_period_end is None:
        observation_period_end = (
            pd.to_datetime(transactions[datetime_col].max(), format=datetime_format)
            .to_period(time_unit)
            .to_timestamp()
        )
    else:
        observation_period_end = (
            pd.to_datetime(observation_period_end, format=datetime_format)
            .to_period(time_unit)
            .to_timestamp()
        )

    # label repeated transactions
    repeated_transactions = _find_first_transactions(
        transactions,
        customer_id_col,
        datetime_col,
        monetary_value_col,
        datetime_format,
        observation_period_end,
        time_unit,
    )
    # reset datetime_col to timestamp
    repeated_transactions[datetime_col] = pd.Index(
        repeated_transactions[datetime_col]
    ).to_timestamp()

    # count all orders by customer
    customers = repeated_transactions.groupby(customer_id_col, sort=False)[
        datetime_col
    ].agg(["min", "max", "count"])

    # subtract 1 from count for non-repeat customers
    customers["frequency"] = customers["count"] - 1

    customers["T"] = (
        (observation_period_end - customers["min"])
        / np.timedelta64(1, time_unit)
        / time_scaler
    )
    customers["recency"] = (
        (customers["max"] - customers["min"])
        / np.timedelta64(1, time_unit)
        / time_scaler
    )

    summary_columns = ["frequency", "recency", "T"]

    if monetary_value_col:
        # create an index of first purchases
        first_purchases = repeated_transactions[repeated_transactions["first"]].index
        # Exclude first purchases from the mean value calculation,
        # by setting as null, then imputing with zero
        repeated_transactions.loc[first_purchases, monetary_value_col] = np.nan
        customers["monetary_value"] = (
            repeated_transactions.groupby(customer_id_col)[monetary_value_col]
            .mean()
            .fillna(0)
        )
        summary_columns.append("monetary_value")

    summary_df = customers[summary_columns].astype(float)

    return summary_df.reset_index()
