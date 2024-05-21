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
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd
import xarray
from numpy import datetime64

__all__ = [
    "to_xarray",
    "customer_lifetime_value",
    "rfm_summary",
    "rfm_train_test_split",
]


def to_xarray(customer_id, *arrays, dim: str = "customer_id"):
    """Convert vector arrays to xarray with a common dim (default "customer_id")."""
    dims = (dim,)
    coords = {dim: np.asarray(customer_id)}

    res = tuple(
        xarray.DataArray(data=array, coords=coords, dims=dims) for array in arrays
    )

    return res[0] if len(arrays) == 1 else res


def customer_lifetime_value(
    transaction_model,
    customer_id: pd.Series | np.ndarray,
    frequency: pd.Series | np.ndarray,
    recency: pd.Series | np.ndarray,
    T: pd.Series | np.ndarray,
    monetary_value: pd.Series | np.ndarray | xarray.DataArray,
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
        The monthly adjusted discount rate. Default: 0.01
    freq: string, optional
        Unit of time of the purchase history. Defaults to "D" for daily.
        Other options are "W" (weekly), "M" (monthly), and "H" (hourly).
        Example: If your dataset contains information about weekly purchases,
        you should use "W".

    Returns
    -------
    xarray
        DataArray with the estimated customer lifetime values
    """

    def _squeeze_dims(x: xarray.DataArray):
        dims_to_squeeze: tuple[str, ...] = ()
        if "chain" in x.dims and len(x.chain) == 1:
            dims_to_squeeze += ("chain",)
        if "draw" in x.dims and len(x.draw) == 1:
            dims_to_squeeze += ("draw",)
        if dims_to_squeeze:
            x = x.squeeze(dims_to_squeeze)
        return x

    if discount_rate == 0.0:
        # no discount rate: just compute a single time step from 0 to `time`
        steps = np.arange(time, time + 1)
    else:
        steps = np.arange(1, time + 1)

    factor = {"W": 4.345, "M": 1.0, "D": 30, "H": 30 * 24}[freq]

    # Monetary value can be passed as a DataArray, with entries per chain and draw or as a simple vector
    if not isinstance(monetary_value, xarray.DataArray):
        monetary_value = to_xarray(customer_id, monetary_value)
    monetary_value = _squeeze_dims(monetary_value)

    frequency, recency, T = to_xarray(customer_id, frequency, recency, T)

    clv = xarray.DataArray(0.0)

    # FIXME: This is a hotfix for ParetoNBDModel, as it has a different API from BetaGeoModel
    #  We should harmonize them!
    from pymc_marketing.clv.models import ParetoNBDModel

    if isinstance(transaction_model, ParetoNBDModel):
        transaction_data = pd.DataFrame(
            {
                "customer_id": customer_id,
                "frequency": frequency,
                "recency": recency,
                "T": T,
            }
        )

        def expected_purchases(*, t, **kwargs):
            return transaction_model.expected_purchases(
                future_t=t,
                data=transaction_data,
            )
    else:
        expected_purchases = transaction_model.expected_num_purchases

    # TODO: Vectorize computation so that we perform a single call to expected_num_purchases
    prev_expected_num_purchases = _squeeze_dims(
        expected_purchases(
            customer_id=customer_id,
            frequency=frequency,
            recency=recency,
            T=T,
            t=0,
        )
    )
    for i in steps * factor:
        # since the prediction of number of transactions is cumulative, we have to subtract off the previous periods
        new_expected_num_purchases = _squeeze_dims(
            expected_purchases(
                customer_id=customer_id,
                frequency=frequency,
                recency=recency,
                T=T,
                t=i,
            )
        )
        expected_transactions = new_expected_num_purchases - prev_expected_num_purchases
        prev_expected_num_purchases = new_expected_num_purchases

        # sum up the CLV estimates of all the periods and apply discounted cash flow
        clv = clv + (monetary_value * expected_transactions) / (1 + discount_rate) ** (
            i / factor
        )

    # Add squeezed chain/draw dims
    if "draw" not in clv.dims:
        clv = clv.expand_dims({"draw": 1})
    if "chain" not in clv.dims:
        clv = clv.expand_dims({"chain": 1})

    return clv.transpose("chain", "draw", "customer_id")


def _find_first_transactions(
    transactions: pd.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None = None,
    datetime_format: str | None = None,
    observation_period_end: str | pd.Period | datetime | None = None,
    time_unit: str = "D",
    sort_transactions: bool | None = True,
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
    datetime_format: string, optional
        A string that represents the timestamp format. Useful if Pandas can't understand
        the provided format.
    observation_period_end: Union[str, pd.Period, datetime], optional
        A string or datetime to denote the final date of the study.
        Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    time_unit: string, optional
        Time granularity for study.
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    sort_transactions: bool, optional
        Default: True
        If raw data is already sorted in chronological order, set to `False` to improve computational efficiency.
    """

    select_columns = [customer_id_col, datetime_col]

    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()

    if isinstance(observation_period_end, pd.Period):
        observation_period_end = observation_period_end.to_timestamp()
    if isinstance(observation_period_end, str):
        observation_period_end = pd.to_datetime(observation_period_end)

    if monetary_value_col:
        select_columns.append(monetary_value_col)

    if sort_transactions:
        transactions = transactions[select_columns].sort_values(select_columns).copy()

    # convert date column into a DateTimeIndex for time-wise grouping and truncating
    transactions[datetime_col] = pd.to_datetime(
        transactions[datetime_col], format=datetime_format
    )
    transactions = (
        transactions.set_index(datetime_col).to_period(time_unit).to_timestamp()
    )

    mask = pd.to_datetime(transactions.index) <= pd.to_datetime(observation_period_end)

    transactions = transactions.loc[mask].reset_index()

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
    period_transactions[datetime_col] = period_transactions[datetime_col].dt.to_period(
        time_unit
    )

    return period_transactions[select_columns]


def clv_summary(*args, **kwargs):
    warnings.warn("clv_summary was renamed to rfm_summary", UserWarning, stacklevel=1)
    return rfm_summary(*args, **kwargs)


def rfm_summary(
    transactions: pd.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None = None,
    datetime_format: str | None = None,
    observation_period_end: str | pd.Period | datetime | None = None,
    time_unit: str = "D",
    time_scaler: float | None = 1,
    include_first_transaction: bool | None = False,
    sort_transactions: bool | None = True,
) -> pd.DataFrame:
    """
    Summarize transaction data for use in CLV modeling and/or RFM segmentation.

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
    observation_period_end: Union[str, pd.Period, datetime], optional
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
    include_first_transaction: bool, optional
        Default: False
        For predictive CLV modeling, this should be False.
        Set to True if performing RFM segmentation.
    sort_transactions: bool, optional
        Default: True
        If raw data is already sorted in chronological order, set to `False` to improve computational efficiency.

    Returns
    -------
    :obj: DataFrame:
        customer_id, frequency, recency, T [, monetary_value]
    """

    if observation_period_end is None:
        observation_period_end_ts = (
            pd.to_datetime(transactions[datetime_col].max(), format=datetime_format)
            .to_period(time_unit)
            .to_timestamp()
        )
    elif isinstance(observation_period_end, pd.Period):
        observation_period_end_ts = observation_period_end.to_timestamp()
    else:
        observation_period_end_ts = (
            pd.to_datetime(observation_period_end, format=datetime_format)
            .to_period(time_unit)
            .to_timestamp()
        )

    # label repeated transactions
    repeated_transactions = _find_first_transactions(  # type: ignore
        transactions,
        customer_id_col,
        datetime_col,
        monetary_value_col,
        datetime_format,
        observation_period_end_ts,
        time_unit,
        sort_transactions,
    )
    # reset datetime_col to timestamp
    repeated_transactions[datetime_col] = repeated_transactions[
        datetime_col
    ].dt.to_timestamp()

    # count all orders by customer
    customers = repeated_transactions.groupby(customer_id_col, sort=False)[
        datetime_col
    ].agg(["min", "max", "count"])

    if not include_first_transaction:
        # subtract 1 from count, as we ignore their first order.
        customers["frequency"] = customers["count"] - 1
    else:
        customers["frequency"] = customers["count"]

    customers["T"] = (
        (observation_period_end_ts - customers["min"])
        / np.timedelta64(1, time_unit)
        / time_scaler
    )
    customers["recency"] = (
        (pd.to_datetime(customers["max"]) - pd.to_datetime(customers["min"]))  # type: ignore
        / np.timedelta64(1, time_unit)
        / time_scaler
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

    summary_df = customers[summary_columns].astype(float)
    summary_df = summary_df.reset_index().rename(
        columns={customer_id_col: "customer_id"}
    )

    return summary_df


def rfm_train_test_split(
    transactions: pd.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    train_period_end: float | str | datetime | datetime64 | date,
    test_period_end: float | str | datetime | datetime64 | date | None = None,
    time_unit: str = "D",
    time_scaler: float | None = 1,
    datetime_format: str | None = None,
    monetary_value_col: str | None = None,
    include_first_transaction: bool | None = False,
    sort_transactions: bool | None = True,
) -> pd.DataFrame:
    """
    Summarize transaction data and split into training and tests datasets for CLV modeling.
    This can also be used to evaluate the impact of a time-based intervention like a marketing campaign.

    This transforms a DataFrame of transaction data of the form:
        customer_id, datetime [, monetary_value]
    to a DataFrame of the form:
        customer_id, frequency, recency, T [, monetary_value], test_frequency [, test_monetary_value], test_T

    Note this function will exclude new customers whose first transactions occurred during the test period.

    Adapted from lifetimes package
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L27

    Parameters
    ----------
    transactions: :obj: DataFrame
        A Pandas DataFrame that contains the customer_id col and the datetime col.
    customer_id_col: string
        Column in the transactions DataFrame that denotes the customer_id.
    datetime_col:  string
        Column in the transactions DataFrame that denotes the datetime the purchase was made.
    train_period_end: Union[str, pd.Period, datetime], optional
        A string or datetime to denote the final time period for the training data.
        Events after this time period are used for the test data.
    test_period_end: Union[str, pd.Period, datetime], optional
        A string or datetime to denote the final time period of the study.
        Events after this date are truncated. If not given, defaults to the max of 'datetime_col'.
    time_unit: string, optional
        Time granularity for study.
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    time_scaler: int, optional
        Default: 1. Useful for scaling recency & T to a different time granularity. Example:
        With freq='D' and freq_multiplier=1, we get recency=591 and T=632
        With freq='h' and freq_multiplier=24, we get recency=590.125 and T=631.375
        This is useful if predictions in months or years are desired,
        and can also help with model convergence for study periods of many years.
    datetime_format: string, optional
        A string that represents the timestamp format. Useful if Pandas can't understand
        the provided format.
    monetary_value_col: string, optional
        Column in the transactions DataFrame that denotes the monetary value of the transaction.
        Optional; only needed for spend estimation models like the Gamma-Gamma model.
    include_first_transaction: bool, optional
        Default: False
        For predictive CLV modeling, this should be False.
        Set to True if performing RFM segmentation.
    sort_transactions: bool, optional
        Default: True
        If raw data is already sorted in chronological order, set to `False` to improve computational efficiency.

    Returns
    -------
    :obj: DataFrame:
        customer_id, frequency, recency, T, test_frequency, test_T [, monetary_value, test_monetary_value]
    """

    if test_period_end is None:
        test_period_end = transactions[datetime_col].max()

    transaction_cols = [customer_id_col, datetime_col]
    if monetary_value_col:
        transaction_cols.append(monetary_value_col)
    transactions = transactions[transaction_cols].copy()

    transactions[datetime_col] = pd.to_datetime(
        transactions[datetime_col], format=datetime_format
    )
    test_period_end = pd.to_datetime(test_period_end, format=datetime_format)
    train_period_end = pd.to_datetime(train_period_end, format=datetime_format)

    # create training dataset
    training_transactions = transactions.loc[
        transactions[datetime_col] <= train_period_end
    ]

    if training_transactions.empty:
        error_msg = """No data available. Check `test_transactions` and `train_period_end`
        and confirm values in `transactions` occur prior to those time periods."""
        raise ValueError(error_msg)

    training_rfm_data = rfm_summary(
        training_transactions,
        customer_id_col,
        datetime_col,
        monetary_value_col=monetary_value_col,
        datetime_format=datetime_format,
        observation_period_end=train_period_end,
        time_unit=time_unit,
        time_scaler=time_scaler,
        include_first_transaction=include_first_transaction,
        sort_transactions=sort_transactions,
    )

    # create test dataset
    test_transactions = transactions.loc[
        (test_period_end >= transactions[datetime_col])
        & (transactions[datetime_col] > train_period_end)
    ].copy()

    if test_transactions.empty:
        error_msg = """No data available. Check `test_transactions` and `train_period_end`
        and confirm values in `transactions` occur prior to those time periods."""
        raise ValueError(error_msg)

    test_transactions[datetime_col] = test_transactions[datetime_col].dt.to_period(
        time_unit
    )
    # create dataframe with customer_id and test_frequency columns
    test_rfm_data = (
        test_transactions.groupby([customer_id_col, datetime_col], sort=False)[
            datetime_col
        ]
        .agg(lambda r: 1)
        .groupby(level=customer_id_col)
        .count()
    ).reset_index()

    test_rfm_data = test_rfm_data.rename(
        columns={"id": "customer_id", "date": "test_frequency"}
    )

    if monetary_value_col:
        test_monetary_value = (
            test_transactions.groupby([customer_id_col, datetime_col])[
                monetary_value_col
            ]
            .sum()
            .groupby(customer_id_col)
            .mean()
        )

        test_rfm_data = test_rfm_data.merge(
            test_monetary_value,
            left_on="customer_id",
            right_on=customer_id_col,
            how="inner",
        )
        test_rfm_data = test_rfm_data.rename(
            columns={monetary_value_col: "test_monetary_value"}
        )

    train_test_rfm_data = training_rfm_data.merge(
        test_rfm_data, on="customer_id", how="left"
    )
    train_test_rfm_data.fillna(0, inplace=True)

    time_delta = (
        test_period_end.to_period(time_unit) - train_period_end.to_period(time_unit)
    ).n
    train_test_rfm_data["test_T"] = time_delta / time_scaler  # type: ignore

    return train_test_rfm_data
