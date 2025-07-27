#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""Utilities for the CLV module."""

import warnings
from datetime import date, datetime

import numpy as np
import pandas
import xarray
from numpy import datetime64

__all__ = [
    "customer_lifetime_value",
    "rfm_segments",
    "rfm_summary",
    "rfm_train_test_split",
    "to_xarray",
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
    data: pandas.DataFrame,
    future_t: int = 12,
    discount_rate: float = 0.00,
    time_unit: str = "D",
) -> xarray.DataArray:
    """
    Compute customer lifetime value.

    Compute the average lifetime value for a group of one or more customers
    and apply a discount rate for net present value estimations.
    Note `future_t` is measured in months regardless of `time_unit` specified.

    Adapted from lifetimes package
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L449

    Parameters
    ----------
    transaction_model : ~CLVModel
        Predictive model for future transactions. `BetaGeoModel` and `ParetoNBDModel` are currently supported.
    data : ~pandas.DataFrame
        DataFrame containing the following columns:

        * `customer_id`: Unique customer identifier
        * `frequency`: Number of repeat purchases observed for each customer
        * `recency`: Time between the first and the last purchase
        * `T`: Time between the first purchase and the end of the observation period
        * `future_spend`: Predicted monetary values for each customer
    future_t : int, optional
        The lifetime expected for the user in months. Default: 12
    discount_rate : float, optional
        The monthly adjusted discount rate. Default: 0.00
    time_unit : string, optional
        Unit of time of the purchase history. Defaults to "D" for daily.
        Other options are "W" (weekly), "M" (monthly), and "H" (hourly).
        Example: If your dataset contains information about weekly purchases,
        you should use "W".

    Returns
    -------
    xarray
        DataArray containing estimated customer lifetime values

    """
    if "future_spend" not in data.columns:
        raise ValueError("Required column future_spend missing")

    def _squeeze_dims(x: xarray.DataArray):
        """
        Squeeze dimensions for MAP-fitted model predictions.

        This utility is required for MAP-fitted model predictions to broadcast properly.

        Parameters
        ----------
        x : xarray.DataArray
            DataArray to squeeze dimensions for.

        Returns
        -------
        xarray.DataArray
            DataArray with squeezed dimensions.
        """
        dims_to_squeeze: tuple[str, ...] = ()
        if "chain" in x.dims and len(x.chain) == 1:
            dims_to_squeeze += ("chain",)
        if "draw" in x.dims and len(x.draw) == 1:
            dims_to_squeeze += ("draw",)
        x = x.squeeze(dims_to_squeeze)
        return x

    if discount_rate == 0.0:
        # no discount rate: just compute a single time step from 0 to `time`
        steps = np.arange(future_t, future_t + 1)
    else:
        steps = np.arange(1, future_t + 1)

    factor = {"W": 4.345, "M": 1.0, "D": 30, "H": 30 * 24}[time_unit]

    monetary_value = to_xarray(data["customer_id"], data["future_spend"])

    clv = xarray.DataArray(0.0)

    # TODO: Add an IF block to support ShiftedBetaGeoModelIndividual

    # initialize FOR loop with 0 purchases at future_t = 0
    prev_expected_purchases = 0

    for i in steps * factor:
        # since the prediction of number of transactions is cumulative, we have to subtract off the previous periods
        new_expected_purchases = _squeeze_dims(
            transaction_model.expected_purchases(
                data=data,
                future_t=i,
            )
        )
        expected_transactions = new_expected_purchases - prev_expected_purchases
        prev_expected_purchases = new_expected_purchases

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
    transactions: pandas.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None = None,
    datetime_format: str | None = None,
    observation_period_end: str | pandas.Period | datetime | None = None,
    time_unit: str = "D",
    sort_transactions: bool | None = True,
) -> pandas.DataFrame:
    """Return dataframe with first transactions.

    This takes a DataFrame of transaction data of the form:
        *customer_id, datetime [, monetary_value]*
    and appends a column named *repeated* to the transaction log to indicate which rows
    are repeated transactions for each *customer_id*.

    Adapted from lifetimes package
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L148

    Parameters
    ----------
    transactions : ~pandas.DataFrame
        A Pandas DataFrame containing *customer_id_col* and *datetime_col*.
    customer_id_col : string
        Column in the *transactions* DataFrame denoting the *customer_id*.
    datetime_col :  string
        Column in the *transactions* DataFrame denoting datetimes purchase were made.
    monetary_value_col : string, optional
        Column in the *transactions* DataFrame that denotes the monetary value of the transaction.
        Optional; only needed for spend estimation models like the Gamma-Gamma model.
    datetime_format : string, optional
        A string that represents the timestamp format. Useful if Pandas can't understand
        the provided format.
    observation_period_end : Union[str, pandas.Period, datetime], optional
        A string or datetime to denote the final date of the study.
        Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    time_unit : string, optional
        Time granularity for study.
        Default : 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    sort_transactions : bool, optional
        Default: True
        If raw data is already sorted in chronological order, set to `False` to improve computational efficiency.

    """
    select_columns = [customer_id_col, datetime_col]

    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()

    if isinstance(observation_period_end, pandas.Period):
        observation_period_end = observation_period_end.to_timestamp()
    if isinstance(observation_period_end, str):
        observation_period_end = pandas.to_datetime(observation_period_end)

    if monetary_value_col:
        select_columns.append(monetary_value_col)

    if sort_transactions:
        transactions = transactions[select_columns].sort_values(select_columns).copy()

    # convert date column into a DateTimeIndex for time-wise grouping and truncating
    transactions[datetime_col] = pandas.to_datetime(
        transactions[datetime_col], format=datetime_format
    )
    transactions = (
        transactions.set_index(datetime_col).to_period(time_unit).to_timestamp()
    )

    mask = pandas.to_datetime(transactions.index) <= pandas.to_datetime(
        observation_period_end
    )

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


def rfm_summary(
    transactions: pandas.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None = None,
    datetime_format: str | None = None,
    observation_period_end: str | pandas.Period | datetime | None = None,
    time_unit: str = "D",
    time_scaler: float | None = 1,
    include_first_transaction: bool | None = False,
    sort_transactions: bool | None = True,
) -> pandas.DataFrame:
    """Summarize transaction data for use in CLV modeling or RFM segmentation.

    This transforms a DataFrame of transaction data of the form:
        *customer_id, datetime [, monetary_value]*
    to a DataFrame for CLV modeling:
        *customer_id, frequency, recency, T [, monetary_value]*

    If the `include_first_transaction = True` argument is specified, a DataFrame for RFM segmentation is returned:
        *customer_id, frequency, recency, monetary_value*

    This function is not required if using the `clv.rfm_segments` utility.

    Adapted from lifetimes package
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L230

    Parameters
    ----------
    transactions : ~pandas.DataFrame
        A Pandas DataFrame containing *customer_id_col* and *datetime_col*.
    customer_id_col : string
        Column in the *transactions* DataFrame denoting the *customer_id*.
    datetime_col :  string
        Column in the *transactions* DataFrame denoting datetimes purchase were made.
    monetary_value_col : string, optional
        Column in the transactions DataFrame denoting the monetary value of the transaction.
        Optional; only needed for RFM segmentation and spend estimation models like the Gamma-Gamma model.
    observation_period_end : Union[str, pandas.Period, datetime], optional
        A string or datetime to denote the final date of the study.
        Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    datetime_format : string, optional
        A string that represents the timestamp format. Useful if Pandas doesn't recognize the provided format.
    time_unit : string, optional
        Time granularity for study.
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    time_scaler : int, optional
        Default: 1. Scales *recency* & *T* to a different time granularity.
        This is useful for datasets spanning many years, and running predictions in different time scales.
    datetime_format : string, optional
        A string that represents the timestamp format. Useful if Pandas doesn't recognize the provided format.
    monetary_value_col : string, optional
        Column in the *transactions* DataFrame that denotes the monetary value of the transaction.
        Optional; only needed for spend estimation models like the Gamma-Gamma model.
    include_first_transaction : bool, optional
        Default: *False*
        For predictive CLV modeling, this should be *False*.
        Set to *True* if performing RFM segmentation.
    sort_transactions : bool, optional
        Default: *True*
        If raw data is already sorted in chronological order, set to *False* to improve computational efficiency.

    Returns
    -------
    DataFrame
        Dataframe containing summarized RFM data, and test columns for *frequency*, *T*,
        and *monetary_value* if specified

    """
    if observation_period_end is None:
        observation_period_end_ts = (
            pandas.to_datetime(transactions[datetime_col].max(), format=datetime_format)
            .to_period(time_unit)
            .to_timestamp()
        )
    elif isinstance(observation_period_end, pandas.Period):
        observation_period_end_ts = observation_period_end.to_timestamp()
    else:
        observation_period_end_ts = (
            pandas.to_datetime(observation_period_end, format=datetime_format)
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

    # subtract 1 from count, as we ignore the first order.
    customers["frequency"] = customers["count"] - 1

    customers["recency"] = (
        (pandas.to_datetime(customers["max"]) - pandas.to_datetime(customers["min"]))
        / np.timedelta64(1, time_unit)  # type: ignore[call-overload]
        / time_scaler
    )

    customers["T"] = (
        (observation_period_end_ts - customers["min"])
        / np.timedelta64(1, time_unit)  # type: ignore[call-overload]
        / time_scaler
    )

    summary_columns = ["frequency", "recency", "T"]

    if include_first_transaction:
        # add the first order back to the frequency count
        customers["frequency"] = customers["frequency"] + 1

        # change recency to segmentation definition
        customers["recency"] = customers["T"] - customers["recency"]

        # T column is not used for segmentation
        summary_columns = ["frequency", "recency"]

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
    transactions: pandas.DataFrame,
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
) -> pandas.DataFrame:
    """Summarize transaction data and split into training and tests datasets for CLV modeling.

    This can also be used to evaluate the impact of a time-based intervention like a marketing campaign.

    This transforms a DataFrame of transaction data of the form:
        *customer_id, datetime [, monetary_value]*
    to a DataFrame of the form:
        *customer_id, frequency, recency, T [, monetary_value], test_frequency [, test_monetary_value], test_T*

    Note this function will exclude new customers whose first transactions occurred during the test period.

    Adapted from lifetimes package
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L27

    Parameters
    ----------
    transactions : ~pandas.DataFrame
        A Pandas DataFrame containing *customer_id_col* and *datetime_col*.
    customer_id_col : string
        Column in the *transactions* DataFrame denoting the customer_id.
    datetime_col :  string
        Column in the *transactions* DataFrame denoting datetimes purchases were made.
    train_period_end : Union[str, pandas.Period, datetime], optional
        A string or datetime to denote the final time period for the training data.
        Events after this time period are used for the test data.
    test_period_end : Union[str, pandas.Period, datetime], optional
        A string or datetime to denote the final time period of the study.
        Events after this date are truncated. If not given, defaults to the max of *datetime_col*.
    time_unit : string, optional
        Time granularity for study.
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    time_scaler : int, optional
        Default: 1. Scales *recency* & *T* to a different time granularity.
        This is useful for datasets spanning many years, and running predictions in different time scales.
    datetime_format : string, optional
        A string that represents the timestamp format. Useful if Pandas doesn't recognize the provided format.
    monetary_value_col : string, optional
        Column in the *transactions* DataFrame that denotes the monetary value of the transaction.
        Optional; only needed for spend estimation models like the Gamma-Gamma model.
    include_first_transaction : bool, optional
        Default: *False*
        For predictive CLV modeling, this should be *False*.
        Set to *True* if performing RFM segmentation.
    sort_transactions : bool, optional
        Default: *True*
        If raw data is already sorted in chronological order, set to *False* to improve computational efficiency.

    Returns
    -------
    DataFrame
        Dataframe containing summarized RFM data, and test columns for *frequency*, *T*,
        and *monetary_value* if specified

    """
    if test_period_end is None:
        test_period_end = transactions[datetime_col].max()

    transaction_cols = [customer_id_col, datetime_col]
    if monetary_value_col:
        transaction_cols.append(monetary_value_col)
    transactions = transactions[transaction_cols].copy()

    transactions[datetime_col] = pandas.to_datetime(
        transactions[datetime_col], format=datetime_format
    )
    test_period_end = pandas.to_datetime(test_period_end, format=datetime_format)
    train_period_end = pandas.to_datetime(train_period_end, format=datetime_format)

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
        columns={customer_id_col: "customer_id", datetime_col: "test_frequency"}
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


def rfm_segments(
    transactions: pandas.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str,
    segment_config: dict | None = None,
    observation_period_end: str | pandas.Period | datetime | None = None,
    datetime_format: str | None = None,
    time_unit: str = "D",
    time_scaler: float | None = 1,
    sort_transactions: bool | None = True,
) -> pandas.DataFrame:
    """Assign customers to segments based on spending behavior derived from RFM scores.

    This transforms a DataFrame of transaction data of the form:
        *customer_id, datetime, monetary_value*
    to a DataFrame of the form:
        *customer_id, frequency, recency, monetary_value, rfm_score, segment*

    Customer purchasing data is aggregated into three variables: `recency`, `frequency`, and `monetary_value`.
    Quartiles are estimated for each variable, and a three-digit RFM score is then assigned to each customer.
    For example, a customer with a score of '234' is in the second quartile for `recency`, third quartile for
    `frequency`, and fourth quartile for `monetary_value`.
    RFM scores corresponding to segments such as "Top Spender", "Frequent Buyer", or "At-Risk" are determined, and
    customers are then segmented based on their RFM score.

    By default, the following segments are created:
        - "Premium Customer": Customers in top 2 quartiles for all variables.
        - "Repeat Customer": Customers in top 2 quartiles for frequency, and either recency or monetary value.
        - "Top Spender": Customers in top 2 quartiles for monetary value, and either frequency or recency.
        - "At-Risk Customer": Customers in bottom 2 quartiles for two or more variables.
        - "Inactive Customer": Customers in bottom quartile for two or more variables.
        - Customers with unspecified RFM scores will be assigned to a segment named "Other".

    If an alternative segmentation approach is desired, use
    `rfm_summary(include_first_transaction=True, *args, **kwargs)` instead to preprocess data for segmentation.
    In either case, the returned DataFrame cannot be used for modeling.
    If assigning model predictions to RFM segments, create a separate DataFrame for modeling and join by Customer ID.

    Parameters
    ----------
    transactions : ~pandas.DataFrame
        A Pandas DataFrame containing *customer_id_col* and *datetime_col*.
    customer_id_col : string
        Column in the *transactions* DataFrame denoting the *customer_id*.
    datetime_col :  string
        Column in the *transactions* DataFrame denoting datetimes purchase were made.
    monetary_value_col : string
        Column in the *transactions* DataFrame that denotes the monetary value of the transaction.
    segment_config : dict, optional
        Dictionary containing segment names and list of RFM score assignments;
        key/value pairs should be formatted as `{"segment": ['111', '123', '321'], ...}`.
        If not provided, default segment names and definitions are applied.
    observation_period_end : Union[str, pandas.Period, datetime, None], optional
        A string or datetime to denote the final date of the study.
        Events after this date are truncated. If not given, defaults to the max of *datetime_col*.
    datetime_format : string, optional
        A string that represents the timestamp format. Useful if Pandas doesn't recognize the provided format.
    time_unit : string, optional
        Time granularity for study.
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    time_scaler : int, optional
        Default: 1. Scales *recency* & *T* to a different time granularity.
        This is useful for datasets spanning many years, and running predictions in different time scales.
    sort_transactions : bool, optional
        Default: *True*
        If raw data is already sorted in chronological order, set to *False* to improve computational efficiency.

    Returns
    -------
    DataFrame
        Dataframe containing summarized RFM data, RFM scores, and segment assignments

    """
    rfm_data = rfm_summary(
        transactions,
        customer_id_col=customer_id_col,
        datetime_col=datetime_col,
        monetary_value_col=monetary_value_col,
        observation_period_end=observation_period_end,
        datetime_format=datetime_format,
        time_unit=time_unit,
        time_scaler=time_scaler,
        include_first_transaction=True,
        sort_transactions=sort_transactions,
    )

    # iteratively assign quartile labels for each row/variable
    for column_name in zip(
        ["r_quartile", "f_quartile", "m_quartile"],
        ["recency", "frequency", "monetary_value"],
        strict=False,
    ):
        # If data has many repeat values, fewer than 4 bins will be returned.
        # These try blocks will modify labelling for fewer bins.
        try:
            labels = _rfm_quartile_labels(column_name[0], 5)
            rfm_data[column_name[0]] = pandas.qcut(
                rfm_data[column_name[1]], q=4, labels=labels, duplicates="drop"
            ).astype(str)
        except ValueError:
            try:
                labels = _rfm_quartile_labels(column_name[0], 4)
                rfm_data[column_name[0]] = pandas.qcut(
                    rfm_data[column_name[1]], q=4, labels=labels, duplicates="drop"
                ).astype(str)
            except ValueError:
                labels = _rfm_quartile_labels(column_name[0], 3)
                rfm_data[column_name[0]] = pandas.qcut(
                    rfm_data[column_name[1]], q=4, labels=labels, duplicates="drop"
                ).astype(str)
                warnings.warn(
                    f"RFM score will not exceed 2 for {column_name[0]}. Specify a custom segment_config",
                    UserWarning,
                    stacklevel=1,
                )

    rfm_data = pandas.eval(  # type: ignore
        "rfm_score = rfm_data.r_quartile + rfm_data.f_quartile + rfm_data.m_quartile",
        target=rfm_data,
    )

    if segment_config is None:
        segment_config = _default_rfm_segment_config

    segment_names = list(segment_config.keys())

    # create catch-all "Other" segment and assign defined segments from config
    rfm_data["segment"] = "Other"

    for key in segment_names:
        rfm_data.loc[rfm_data["rfm_score"].isin(segment_config[key]), "segment"] = key

    # drop unnecessary columns
    rfm_data = rfm_data.drop(columns=["r_quartile", "f_quartile", "m_quartile"])

    return rfm_data


def _rfm_quartile_labels(column_name, max_label_range):
    """
    Label quartiles for each variable.

    Called internally by rfm_segments to label quartiles for each variable.

    Parameters
    ----------
    column_name : str
        The name of the column to label.
    max_label_range : int
        The maximum range of labels to create.

    Returns
    -------
    list[int]
        A list of labels for the column.
    """
    # recency labels must be reversed because lower values are more desirable
    if column_name == "r_quartile":
        return list(range(max_label_range - 1, 0, -1))
    else:
        return range(1, max_label_range)


_default_rfm_segment_config = {
    "Premium Customer": [
        "334",
        "443",
        "444",
        "344",
        "434",
        "433",
        "343",
        "333",
    ],
    "Repeat Customer": ["244", "234", "232", "332", "143", "233", "243"],
    "Top Spender": [
        "424",
        "414",
        "144",
        "314",
        "324",
        "124",
        "224",
        "423",
        "413",
        "133",
        "323",
        "313",
        "134",
    ],
    "At Risk Customer": [
        "422",
        "223",
        "212",
        "122",
        "222",
        "132",
        "322",
        "312",
        "412",
        "123",
        "214",
    ],
    "Inactive Customer": ["411", "111", "113", "114", "112", "211", "311"],
}


def _expected_cumulative_transactions(
    model,
    transactions: pandas.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    t: int,
    datetime_format: str | None = None,
    time_unit: str = "D",
    time_scaler: float | None = 1,
    sort_transactions: bool | None = True,
    set_index_date: bool | None = False,
):
    """
    Aggregate actual and expected cumulative transactions over time for a fitted ``BetaGeoModel`` or ``ParetoNBDModel``.

    This function follows the formulation on page 8 of [1]_. Specifically, we take only customers who have made their
    first transaction before the specified number of ``t`` time periods, run ``expected_purchases_new_customer()``
    for all remaining time periods, then sum across the customer population.

    Adapted from legacy ``lifetimes`` library:
    https://github.com/CamDavidsonPilon/lifetimes/blob/master/lifetimes/utils.py#L506

    Parameters
    ----------
    model:
        A fitted ``BetaGeoModel`` or ``ParetoNBDModel``.
    transactions : ~pandas.DataFrame
        A Pandas DataFrame containing *customer_id_col* and *datetime_col*.
    customer_id_col : string
        Column in the *transactions* DataFrame denoting the *customer_id*.
    datetime_col :  string
        Column in the *transactions* DataFrame denoting datetimes purchase were made.
    t: int
        Number of time units since earliest transaction for which we want to aggregate cumulative transactions.
    datetime_format : string, optional
        A string that represents the timestamp format. Useful if Pandas doesn't recognize the provided format.
    time_unit : string, optional
        Time granularity for study.
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    time_scaler : int, optional
        Default: 1. Scales *recency* & *T* to a different time granularity.
        This is useful for datasets spanning many years, and running predictions in different time scales.
    sort_transactions : bool, optional
        Default: *True*
        If raw data is already sorted in chronological order, set to *False* to improve computational efficiency.
    set_index_date: bool, optional
        Set to True to return a dataframe with a datetime index.

    Returns
    -------
    DataFrame
        Dataframe containing columns for actual and predicted values

    References
    ----------
    .. [1] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005),
    A Note on Implementing the Pareto/NBD Model in MATLAB.
    http://brucehardie.com/notes/008/
    """
    start_date = pandas.to_datetime(
        transactions[datetime_col], format=datetime_format
    ).min()
    start_period = start_date.to_period(time_unit)
    observation_period_end = start_period + t

    # Has an extra column (besides the id and the date)
    # with a boolean for when it is a first transaction
    repeated_and_first_transactions = _find_first_transactions(  # type: ignore
        transactions,
        customer_id_col,
        datetime_col,
        datetime_format=datetime_format,
        observation_period_end=observation_period_end,
        time_unit=time_unit,
        sort_transactions=sort_transactions,
    )

    # Mask, first transactions and repeated transactions
    first_trans_mask = repeated_and_first_transactions["first"]
    repeated_transactions = repeated_and_first_transactions[~first_trans_mask]
    first_transactions = repeated_and_first_transactions[first_trans_mask]

    date_range = pandas.date_range(start_date, periods=t + 1, freq=time_unit)
    date_periods = date_range.to_period(time_unit)

    pred_cum_transactions = np.array([])

    # First Transactions on Each Day/Freq
    first_trans_size = first_transactions.groupby(datetime_col).size()

    # In the loop below, we calculate the expected number of purchases for customers
    # who have made their first purchases on a date before the one being evaluated.
    # Then we sum them to get the cumulative sum up to the specific period.
    for i, period in enumerate(date_periods):  # index of period and its date
        if i % time_scaler == 0 and i > 0:  # type: ignore
            # Periods before the one being evaluated
            times = np.array([d.n for d in period - first_trans_size.index])
            times = times[times > 0].astype(float) / time_scaler

            # create arbitrary dataframe from array of n time periods for predictions
            pred_data = pandas.DataFrame(
                {
                    "customer_id": times,
                    "t": times,
                }
            )

            # Array of different expected number of purchases for different times
            # TODO: This does not currently support a covariate model
            expected_trans_array = model.expected_purchases_new_customer(
                pred_data
            ).mean(dim=("chain", "draw"))

            # Mask for the number of customers with 1st transactions up to the period
            mask = first_trans_size.index < period
            masked_first_trans = first_trans_size[mask].values  # type: ignore
            # ``expected_trans`` is an xarray with the cumulative sum of expected transactions
            expected_trans = (expected_trans_array * masked_first_trans).sum()
            pred_cum_transactions = np.append(
                pred_cum_transactions, expected_trans.values
            )

    act_trans = repeated_transactions.groupby(datetime_col).size()
    act_tracking_transactions = act_trans.reindex(date_periods, fill_value=0)

    act_cum_transactions = []
    for j in range(1, t // time_scaler + 1):  # type: ignore
        sum_trans = sum(act_tracking_transactions.iloc[: j * time_scaler])  # type: ignore
        act_cum_transactions.append(sum_trans)

    if set_index_date:
        index = date_periods[time_scaler - 1 : -1 : time_scaler]  # type: ignore
    else:
        index = range(0, t // time_scaler)  # type: ignore

    df_cum_transactions = pandas.DataFrame(
        {"actual": act_cum_transactions, "predicted": pred_cum_transactions},
        index=index,
    )

    return df_cum_transactions
