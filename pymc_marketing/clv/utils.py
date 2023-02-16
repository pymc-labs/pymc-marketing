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


def _find_first_transactions(
    transactions,
    customer_id_col,
    datetime_col,
    monetary_value_col=None,
    datetime_format=None,
    observation_period_end=None,
    freq="D",
):
    """
    Return dataframe with first transactions.

    This takes a DataFrame of transaction data of the form:
        customer_id, datetime [, monetary_value]
    and appends a column named 'repeated' to the transaction log which indicates which rows
    are repeated transactions for that customer_id.

    Parameters
    ----------
    transactions: :obj: DataFrame
        a Pandas DataFrame that contains the customer_id col and the datetime col.
    customer_id_col: string
        the column in transactions DataFrame that denotes the customer_id
    datetime_col:  string
        the column in transactions that denotes the datetime the purchase was made.
    monetary_value_col: string, optional
        the column in transactions that denotes the monetary value of the transaction.
        Optional, only needed for customer lifetime value estimation models.
    observation_period_end: :obj: datetime
        a string or datetime to denote the final date of the study.
        Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    datetime_format: string, optional
        a string that represents the timestamp format. Useful if Pandas can't understand
        the provided format.
    freq: string, optional
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    """

    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()

    if type(observation_period_end) == pd.Period:
        observation_period_end = observation_period_end.to_timestamp()

    select_columns = [customer_id_col, datetime_col]

    if monetary_value_col:
        select_columns.append(monetary_value_col)

    transactions = transactions[select_columns].sort_values(select_columns).copy()

    # make sure the date column uses datetime objects, and use Pandas' DateTimeIndex.to_period()
    # to convert the column to a PeriodIndex which is useful for time-wise grouping and truncating
    transactions[datetime_col] = pd.to_datetime(
        transactions[datetime_col], format=datetime_format
    )
    transactions = transactions.set_index(datetime_col).to_period(freq).to_timestamp()

    transactions = transactions.loc[
        (transactions.index <= observation_period_end)
    ].reset_index()

    period_groupby = transactions.groupby(
        [datetime_col, customer_id_col], sort=False, as_index=False
    )

    if monetary_value_col:
        # when we have a monetary column, make sure to sum together any values in the same period
        period_transactions = period_groupby.sum()
    else:
        # by calling head() on the groupby object, the datetime_col and customer_id_col columns
        # will be reduced
        period_transactions = period_groupby.head(1)

    # initialize a new column where we will indicate which are the first transactions
    period_transactions = period_transactions.copy()
    period_transactions.loc[:, "first"] = False
    # find all of the initial transactions and store as an index
    first_transactions = (
        period_transactions.groupby(customer_id_col, sort=True, as_index=False)
        .head(1)
        .index
    )
    # mark the initial transactions as True
    period_transactions.loc[first_transactions, "first"] = True
    select_columns.append("first")
    # reset datetime_col to period
    period_transactions.loc[:, datetime_col] = pd.Index(
        period_transactions[datetime_col]
    ).to_period(freq)

    return period_transactions[select_columns]


def clv_summary(
    transactions,
    customer_id_col,
    datetime_col,
    monetary_value_col=None,
    datetime_format=None,
    observation_period_end=None,
    freq="D",
    freq_multiplier=1,
    include_first_transaction=False,
):
    """
    Return summary data from transactions.

    This transforms a DataFrame of transaction data of the form:
        customer_id, datetime [, monetary_value]
    to a DataFrame of the form:
        customer_id, frequency, recency, T [, monetary_value]

    Parameters
    ----------
    transactions: :obj: DataFrame
        a Pandas DataFrame that contains the customer_id col and the datetime col.
    customer_id_col: string
        the column in transactions DataFrame that denotes the customer_id
    datetime_col:  string
        the column in transactions that denotes the datetime the purchase was made.
    monetary_value_col: string, optional
        the columns in the transactions that denotes the monetary value of the transaction.
        Optional, only needed for customer lifetime value estimation models.
    observation_period_end: datetime, optional
         a string or datetime to denote the final date of the study.
         Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    datetime_format: string, optional
        a string that represents the timestamp format. Useful if Pandas can't understand
        the provided format.
    freq: string, optional
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    freq_multiplier: int, optional
        Default: 1. Useful for getting exact recency & T. Example:
        With freq='D' and freq_multiplier=1, we get recency=591 and T=632
        With freq='h' and freq_multiplier=24, we get recency=590.125 and T=631.375
    include_first_transaction: bool, optional
        Default: False
        By default the first transaction is not included while calculating frequency and
        monetary_value. Can be set to True to include it.
        Should be False if you are going to use this data with any fitters in BTYD package

    Returns
    -------
    :obj: DataFrame:
        customer_id, frequency, recency, T [, monetary_value]
    """

    if observation_period_end is None:
        observation_period_end = (
            pd.to_datetime(transactions[datetime_col].max(), format=datetime_format)
            .to_period(freq)
            .to_timestamp()
        )
    else:
        observation_period_end = (
            pd.to_datetime(observation_period_end, format=datetime_format)
            .to_period(freq)
            .to_timestamp()
        )

    # label all of the repeated transactions
    repeated_transactions = _find_first_transactions(
        transactions,
        customer_id_col,
        datetime_col,
        monetary_value_col,
        datetime_format,
        observation_period_end,
        freq,
    )
    # reset datetime_col to timestamp
    repeated_transactions[datetime_col] = pd.Index(
        repeated_transactions[datetime_col]
    ).to_timestamp()

    # count all orders by customer.
    customers = repeated_transactions.groupby(customer_id_col, sort=False)[
        datetime_col
    ].agg(["min", "max", "count"])

    if not include_first_transaction:
        # subtract 1 from count, as we ignore their first order.
        customers["frequency"] = customers["count"] - 1
    else:
        customers["frequency"] = customers["count"]

    customers["T"] = (
        (observation_period_end - customers["min"])
        / np.timedelta64(1, freq)
        / freq_multiplier
    )
    customers["recency"] = (
        (customers["max"] - customers["min"])
        / np.timedelta64(1, freq)
        / freq_multiplier
    )

    summary_columns = ["frequency", "recency", "T"]

    if monetary_value_col:
        if not include_first_transaction:
            # create an index of all the first purchases
            first_purchases = repeated_transactions[
                repeated_transactions["first"]
            ].index
            # by setting the monetary_value cells of all the first purchases to NaN,
            # those values will be excluded from the mean value calculation
            repeated_transactions.loc[first_purchases, monetary_value_col] = np.nan
        customers["monetary_value"] = (
            repeated_transactions.groupby(customer_id_col)[monetary_value_col]
            .mean()
            .fillna(0)
        )
        summary_columns.append("monetary_value")

    return customers[summary_columns].astype(float)