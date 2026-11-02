#   Copyright 2022 - 2026 The PyMC Labs Developers
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
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
import pandas
import xarray
from numpy import datetime64

from pymc_marketing.clv._narwhals_utils import (
    TIME_UNIT_DAYS,
    Backend,
    _coerce_period_end,
    _finalize,
    _period_start,
    _period_start_scalar,
    _prepare_datetime_column,
    _resolve_time_spec,
    _validate_input,
)

__all__ = [
    "customer_lifetime_value",
    "rfm_segments",
    "rfm_summary",
    "rfm_train_test_split",
    "to_xarray",
]

# Average length of a calendar month: 365.25 / 12, where 365.25 accounts for
# leap years. Using a flat 30 here understates a year by 5.25 days and inflates
# CLV estimates for the "D", "W" and "h" time units.
_DAYS_PER_MONTH = 30.4375

# Values accepted as the end of an observation, training or test period.
PeriodEnd = str | date | datetime | pandas.Period | pandas.Timestamp

# pandas 3 removed the "H" period alias ("h" replaces it) and numpy never
# accepted "H" as a datetime unit. Normalise once at the public entry points.
_PERIOD_ALIASES = {"H": "h"}
# "M" stays the monthly period alias but pandas 3 needs "ME" for offsets.
_OFFSET_ALIASES = {"M": "ME"}


def _normalize_time_unit(time_unit: str) -> str:
    """Return the pandas period alias for a public ``time_unit`` value."""
    return _PERIOD_ALIASES.get(time_unit, time_unit)


def _resolve_dims(ndim: int, dims: tuple[str, ...]) -> tuple[str, ...]:
    """Return the dim names labelling an array of ``ndim`` dimensions."""
    if ndim == 1:
        return dims[:1]
    if ndim == len(dims):
        return dims
    if ndim == 0:
        raise ValueError(
            f"Cannot convert a 0-dimensional array: every array needs a leading "
            f"{dims[0]!r} axis."
        )
    if len(dims) == 1:
        raise ValueError(
            f"Cannot label an array of {ndim} dimensions with {dims}. Pass "
            "'extra_dims' naming its trailing axes."
        )
    raise ValueError(
        f"Cannot label an array of {ndim} dimensions with {dims}: every array must "
        f"have either 1 or {len(dims)} dimensions. Arrays whose trailing axes differ "
        "need one call each, so that every axis is named unambiguously."
    )


def to_xarray(
    customer_id,
    *arrays,
    dim: str = "customer_id",
    extra_dims: Sequence[str] | None = None,
):
    """Convert arrays to xarray objects sharing a common dim (default "customer_id").

    The leading axis of every array is labelled ``dim`` and takes ``customer_id`` as
    its coordinate. Arrays with more than one axis are supported, but the names of
    their trailing axes must be supplied through ``extra_dims``. Auto-generated names
    are deliberately not provided: they would make the unrelated trailing axes of two
    arrays returned by the same call align with each other.

    For the same reason every array must be one-dimensional or carry the full set of
    trailing axes; a rank in between is rejected rather than guessed at, since
    ``extra_dims`` would not say which of its names the missing axis dropped. Arrays
    with different trailing axes need one call each.

    Parameters
    ----------
    customer_id : array-like
        Coordinate values for ``dim``. Must match the length of the leading axis of
        every array.
    *arrays : array-like
        Arrays to convert. Each must be one-dimensional or have
        ``1 + len(extra_dims)`` dimensions.
    dim : str, default "customer_id"
        Name of the leading dimension.
    extra_dims : sequence of str, optional
        Names of the trailing dimensions, e.g. ``("channel",)``. Required as soon as
        an array is multidimensional. A bare string is rejected: it would be read as
        one name per character.

    Returns
    -------
    xarray.DataArray or tuple of xarray.DataArray
        One ``DataArray`` per entry in ``arrays``, or a bare ``DataArray`` when a
        single array is passed.

    Raises
    ------
    TypeError
        If ``extra_dims`` is a string rather than a sequence of names.
    ValueError
        If ``extra_dims`` repeats a name or clashes with ``dim``, or if an array is
        neither one-dimensional nor of dimension ``1 + len(extra_dims)``.

    Examples
    --------
    .. code-block:: python

        frequency, recency = to_xarray(
            data["customer_id"], data["frequency"], data["recency"]
        )
        spend = to_xarray(data["customer_id"], channel_matrix, extra_dims=("channel",))

    Trailing dimensions are named but not populated; attach their labels with
    ``assign_coords``:

    .. code-block:: python

        spend = to_xarray(
            data["customer_id"], channel_df, extra_dims=("channel",)
        ).assign_coords(channel=channel_df.columns)

    """
    if isinstance(extra_dims, str):
        raise TypeError(
            "extra_dims must be a sequence of dimension names, not a string; pass "
            f"extra_dims=({extra_dims!r},) to name a single trailing axis."
        )

    dims: tuple[str, ...] = (dim, *(extra_dims or ()))
    if len(set(dims)) != len(dims):
        raise ValueError(f"Dimension names must be unique, got {dims}.")

    coords = {dim: np.asarray(customer_id)}

    res = tuple(
        xarray.DataArray(
            data=array, coords=coords, dims=_resolve_dims(np.ndim(array), dims)
        )
        for array in arrays
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

    Note: ``future_t`` is always in months regardless of the ``time_unit``
    used for the transaction model. The value is converted internally to
    the given ``time_unit`` for computing expected purchases per period.

    Adapted from the legacy ``lifetimes`` library:
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L449

    Parameters
    ----------
    transaction_model : ~CLVModel
        Predictive model for future transactions. `BetaGeoModel`,
        `ModifiedBetaGeoModel`, and `ParetoNBDModel` are currently supported.
    data : ~pandas.DataFrame
        DataFrame containing the following columns:

        * `customer_id`: Unique customer identifier
        * `frequency`: Number of repeat purchases observed for each customer
        * `recency`: Time between the first and the last purchase
        * `T`: Time between the first purchase and the end of the observation period
        * `future_spend`: Predicted monetary values for each customer
    future_t : int, optional
        The number of months to project lifetime value for. This is always
        specified in months, independent of ``time_unit``. Default: 12
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
    time_unit = _normalize_time_unit(time_unit)
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

    factor = {
        "W": _DAYS_PER_MONTH / 7,
        "M": 1.0,
        "D": _DAYS_PER_MONTH,
        "h": _DAYS_PER_MONTH * 24,
    }[time_unit]

    monetary_value = to_xarray(data["customer_id"], data["future_spend"])

    clv = xarray.DataArray(0.0)

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

    Adapted from the legacy ``lifetimes`` library:
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
        Default: 'D' for days. Other options are 'W' (weekly), 'M' (monthly)
        and 'H' (hourly).
    sort_transactions : bool, optional
        Default: True
        If raw data is already sorted in chronological order, set to `False` to improve computational efficiency.

    """
    time_unit = _normalize_time_unit(time_unit)
    select_columns = [customer_id_col, datetime_col]

    if monetary_value_col:
        select_columns.append(monetary_value_col)

    transactions = transactions[select_columns].copy()
    transactions[datetime_col] = pandas.to_datetime(
        transactions[datetime_col], format=datetime_format
    )

    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()

    if isinstance(observation_period_end, pandas.Period):
        observation_period_end = observation_period_end.to_timestamp()
    if isinstance(observation_period_end, str):
        observation_period_end = pandas.to_datetime(
            observation_period_end, format=datetime_format
        )

    if sort_transactions:
        transactions = transactions.sort_values(select_columns)

    # convert date column into a DateTimeIndex for time-wise grouping and truncating
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
    period_transactions["first"] = False
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


def _rfm_summary_legacy(
    transactions: pandas.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None = None,
    datetime_format: str | None = None,
    observation_period_end: PeriodEnd | None = None,
    time_unit: str = "D",
    time_scaler: float | None = 1,
    include_first_transaction: bool | None = False,
    sort_transactions: bool | None = True,
) -> pandas.DataFrame:
    """Pandas-only implementation of :func:`rfm_summary`, kept verbatim until v2.0."""
    time_unit = _normalize_time_unit(time_unit)
    if observation_period_end is None:
        observation_period_end_ts = (
            pandas.to_datetime(transactions[datetime_col], format=datetime_format)
            .max()
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

    # Count recency and T in calendar periods, like the frequency column does.
    # Every timestamp here is already aligned to a period start, so this equals
    # the timedelta division for "D", "W" and "h" and also works for "M", which
    # has no fixed numpy duration.
    first_period = customers["min"].dt.to_period(time_unit).array.asi8
    last_period = customers["max"].dt.to_period(time_unit).array.asi8
    end_period = observation_period_end_ts.to_period(time_unit).ordinal

    customers["recency"] = (last_period - first_period) / time_scaler
    customers["T"] = (end_period - first_period) / time_scaler

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


def _rfm_train_test_split_legacy(
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
    """Pandas-only implementation of :func:`rfm_train_test_split`, kept verbatim until v2.0."""
    time_unit = _normalize_time_unit(time_unit)
    transaction_cols = [customer_id_col, datetime_col]
    if monetary_value_col:
        transaction_cols.append(monetary_value_col)
    transactions = transactions[transaction_cols].copy()

    transactions[datetime_col] = pandas.to_datetime(
        transactions[datetime_col], format=datetime_format
    )
    if test_period_end is None:
        test_period_end = transactions[datetime_col].max()

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

    training_rfm_data = _rfm_summary_legacy(
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


def _rfm_segments_legacy(
    transactions: pandas.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str,
    segment_config: dict | None = None,
    observation_period_end: PeriodEnd | None = None,
    datetime_format: str | None = None,
    time_unit: str = "D",
    time_scaler: float | None = 1,
    sort_transactions: bool | None = True,
) -> pandas.DataFrame:
    """Pandas-only implementation of :func:`rfm_segments`, kept verbatim until v2.0."""
    time_unit = _normalize_time_unit(time_unit)
    rfm_data = _rfm_summary_legacy(
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


# --------------------------------------------------------------------------- #
# Dataframe-agnostic (Narwhals) path and the dispatchers in front of it.
# --------------------------------------------------------------------------- #

# pandas period aliases accepted by the legacy path.
_LEGACY_TIME_UNITS = frozenset({"D", "W", "M", "H", "h"})
# Legacy aliases with an exact counterpart on the Narwhals path.
_LEGACY_TO_NEW: dict[str, tuple[str, str]] = {
    "D": ("1d", "day"),
    "W": ("1w", "week"),
}
# ``time_scaler`` values that have a named ``time_unit`` equivalent.
_KNOWN_TIME_SCALERS: dict[float, str] = {
    7.0: "week",
    30.0: "month",
    30.4375: "month",
    91.3125: "quarter",
    365.0: "year",
    365.25: "year",
}
_SEGMENT_METRICS = (
    ("r_quartile", "recency"),
    ("f_quartile", "frequency"),
    ("m_quartile", "monetary_value"),
)
_QUARTILE_PROBABILITIES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _deprecation_message(param: str, value: Any) -> str:
    """Build the ``FutureWarning`` text for a legacy parameter."""
    base = f"'{param}' is deprecated and will be removed in pymc-marketing v2.0. "
    if param == "time_unit":
        if value in _LEGACY_TO_NEW:
            granularity, unit = _LEGACY_TO_NEW[value]
            hint = f"Use granularity={granularity!r}, time_unit={unit!r} instead."
        else:
            hint = (
                "Use granularity=... / time_unit=... instead; note that 'M' and 'h' "
                "have no exact equivalent (months become 30.4375-day floats and "
                "hourly periods are not supported)."
            )
    elif param == "time_scaler":
        named = next(
            (
                unit
                for scaler, unit in _KNOWN_TIME_SCALERS.items()
                if abs(float(value) - scaler) < 1e-6
            ),
            None,
        )
        hint = (
            f"Use time_unit={named!r} instead."
            if named is not None
            else "Use granularity=... / time_unit=... instead."
        )
    elif param == "datetime_format":
        hint = (
            "Convert the column with pandas.to_datetime(..., format=...) first instead."
        )
    else:  # sort_transactions
        hint = (
            "The Narwhals-based path does not depend on row order; drop the argument."
        )
    return base + hint


def _route(
    transactions: Any,
    *,
    time_unit: str,
    time_scaler: float | None,
    datetime_format: str | None,
    sort_transactions: bool | None,
    granularity: str | None,
    week_start: str | None,
    backend: Backend | None,
    stacklevel: int = 3,
) -> tuple[Literal["legacy", "nw"], dict[str, Any]]:
    """Decide between the legacy pandas path and the Narwhals path.

    Returns the path name and the keyword arguments resolved for it. Legacy
    parameters passed at their default values count as not passed.
    """
    if time_unit in TIME_UNIT_DAYS:
        named_time_unit: str | None = time_unit
    elif time_unit in _LEGACY_TIME_UNITS:
        named_time_unit = None
    else:
        raise ValueError(
            f"Unsupported time_unit {time_unit!r}. Expected a named unit "
            f"({sorted(TIME_UNIT_DAYS)}) or a legacy pandas alias "
            f"({sorted(_LEGACY_TIME_UNITS)})."
        )

    legacy_passed: dict[str, Any] = {}
    if named_time_unit is None and time_unit != "D":
        legacy_passed["time_unit"] = time_unit
    if time_scaler is not None and time_scaler != 1:
        legacy_passed["time_scaler"] = time_scaler
    if datetime_format is not None:
        legacy_passed["datetime_format"] = datetime_format
    if sort_transactions is not None and sort_transactions is not True:
        legacy_passed["sort_transactions"] = sort_transactions

    new_passed: dict[str, Any] = {
        name: value
        for name, value in (
            ("granularity", granularity),
            ("week_start", week_start),
            ("backend", backend),
            ("time_unit", named_time_unit),
        )
        if value is not None
    }

    if legacy_passed and new_passed:
        raise TypeError(
            f"Legacy parameter(s) {sorted(legacy_passed)} cannot be combined with "
            f"{sorted(new_passed)}. Use either the legacy pandas arguments or "
            "granularity / time_unit / week_start / backend, not both."
        )

    if isinstance(transactions, pandas.DataFrame) and not new_passed:
        for name, value in legacy_passed.items():
            warnings.warn(
                _deprecation_message(name, value), FutureWarning, stacklevel=stacklevel
            )
        return "legacy", {
            "time_unit": time_unit,
            "time_scaler": time_scaler,
            "datetime_format": datetime_format,
            "sort_transactions": sort_transactions,
        }

    resolved: dict[str, Any] = {
        "granularity": "1d",
        "time_unit": "day",
        "week_start": None,
        "backend": None,
        "extra_divisor": 1.0,
    }
    if new_passed:
        resolved.update(new_passed)
        return "nw", resolved

    # Non-pandas input with legacy arguments: translate what has an exact
    # equivalent, refuse the rest.
    if time_unit not in _LEGACY_TO_NEW:
        raise TypeError(
            f"time_unit={time_unit!r} is only supported for pandas input. Use "
            "granularity=... / time_unit=... on other dataframe libraries."
        )
    resolved["granularity"], resolved["time_unit"] = _LEGACY_TO_NEW[time_unit]
    if datetime_format is not None:
        raise TypeError(
            "datetime_format is only supported for pandas input; cast the column to "
            "a datetime dtype first (e.g. pl.col(...).str.to_datetime(format=...))."
        )
    if time_scaler is not None and time_scaler != 1:
        resolved["extra_divisor"] = float(time_scaler)
    if "sort_transactions" in legacy_passed:
        warnings.warn(
            "sort_transactions is ignored: the Narwhals-based path does not depend "
            "on row order.",
            UserWarning,
            stacklevel=stacklevel,
        )
    for name, value in legacy_passed.items():
        warnings.warn(
            _deprecation_message(name, value), FutureWarning, stacklevel=stacklevel
        )
    return "nw", resolved


def _rfm_summary_frame(
    frame: nw.LazyFrame[Any],
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None,
    observation_period_end: PeriodEnd | None,
    include_first_transaction: bool,
    unit: str,
    week_start_offset: int,
    divisor: float,
) -> nw.LazyFrame[Any]:
    """Build the lazy RFM summary shared by the Narwhals implementations.

    The result has the legacy column names and dtypes, sorted by ``customer_id``.
    """
    columns = [customer_id_col, datetime_col]
    if monetary_value_col:
        columns.append(monetary_value_col)
    schema = _validate_input(frame, columns, datetime_col)
    frame = frame.select(columns)
    frame, time_zone = _prepare_datetime_column(
        frame, datetime_col, schema[datetime_col]
    )

    frame = frame.with_columns(
        _period_start(nw.col(datetime_col), unit, week_start_offset).alias("_period")
    )
    if observation_period_end is None:
        end_expr = nw.col("_period").max()
    else:
        end = _coerce_period_end(
            observation_period_end, time_zone, "observation_period_end"
        )
        end_expr = nw.lit(_period_start_scalar(end, unit, week_start_offset))
    frame = frame.with_columns(end_expr.alias("_end")).filter(
        nw.col("_period") <= nw.col("_end")
    )

    # One row per customer and period. Spend inside a period is summed, as the
    # legacy path does.
    period_aggs = [nw.col("_end").max().alias("_end")]
    if monetary_value_col:
        period_aggs.append(nw.col(monetary_value_col).sum().alias("_monetary"))
    periods = frame.group_by(customer_id_col, "_period").agg(*period_aggs)

    customer_aggs = [
        nw.col("_period").min().alias("_first"),
        nw.col("_period").max().alias("_last"),
        nw.len().alias("_n_periods"),
        nw.col("_end").max().alias("_end"),
    ]
    if monetary_value_col:
        # Spend in the customer's first period, isolated with a window function so
        # that no join is needed (every period row but the first contributes 0).
        periods = periods.with_columns(
            nw.when(nw.col("_period") == nw.col("_period").min().over(customer_id_col))
            .then(nw.col("_monetary"))
            .otherwise(nw.lit(0.0))
            .alias("_first_monetary")
        )
        customer_aggs.extend(
            [
                nw.col("_monetary").sum().alias("_monetary_sum"),
                nw.col("_first_monetary").sum().alias("_first_monetary"),
            ]
        )
    customers = periods.group_by(customer_id_col).agg(*customer_aggs)

    n_periods = nw.col("_n_periods").cast(nw.Float64)
    recency = (nw.col("_last") - nw.col("_first")).cast(nw.Float64) / divisor
    t_value = (nw.col("_end") - nw.col("_first")).cast(nw.Float64) / divisor

    if include_first_transaction:
        frequency = n_periods
        # Segmentation definition: time since the last purchase.
        recency = t_value - recency
        summary = [frequency.alias("frequency"), recency.alias("recency")]
    else:
        frequency = n_periods - 1
        summary = [
            frequency.alias("frequency"),
            recency.alias("recency"),
            t_value.alias("T"),
        ]

    if monetary_value_col:
        if include_first_transaction:
            monetary = nw.col("_monetary_sum") / n_periods
        else:
            monetary = (
                nw.when(nw.col("_n_periods") > 1)
                .then(
                    (nw.col("_monetary_sum") - nw.col("_first_monetary"))
                    / (n_periods - 1)
                )
                .otherwise(nw.lit(0.0))
            )
        summary.append(monetary.cast(nw.Float64).alias("monetary_value"))

    return customers.select(
        nw.col(customer_id_col).alias("customer_id"), *summary
    ).sort("customer_id")


def _rfm_summary_nw(
    transactions: Any,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None = None,
    observation_period_end: PeriodEnd | None = None,
    include_first_transaction: bool | None = False,
    *,
    granularity: str | None = None,
    time_unit: str | None = None,
    week_start: str | None = None,
    backend: Backend | None = None,
    extra_divisor: float = 1.0,
) -> Any:
    """Dataframe-agnostic implementation of :func:`rfm_summary`."""
    unit, time_unit, week_start_offset = _resolve_time_spec(
        granularity, time_unit, week_start
    )
    divisor = TIME_UNIT_DAYS[time_unit] * extra_divisor

    frame = nw.from_native(transactions)
    eager_input = isinstance(frame, nw.DataFrame)
    summary = _rfm_summary_frame(
        frame.lazy(),
        customer_id_col,
        datetime_col,
        monetary_value_col,
        observation_period_end,
        bool(include_first_transaction),
        unit,
        week_start_offset,
        divisor,
    )
    return _finalize(summary, backend, eager_input=eager_input)


_NO_DATA_MESSAGE = """No data available. Check `test_transactions` and `train_period_end`
        and confirm values in `transactions` occur prior to those time periods."""


def _rfm_train_test_split_nw(
    transactions: Any,
    customer_id_col: str,
    datetime_col: str,
    train_period_end: PeriodEnd,
    test_period_end: PeriodEnd | None = None,
    monetary_value_col: str | None = None,
    include_first_transaction: bool | None = False,
    *,
    granularity: str | None = None,
    time_unit: str | None = None,
    week_start: str | None = None,
    backend: Backend | None = None,
    extra_divisor: float = 1.0,
) -> Any:
    """Dataframe-agnostic implementation of :func:`rfm_train_test_split`."""
    unit, time_unit, week_start_offset = _resolve_time_spec(
        granularity, time_unit, week_start
    )
    divisor = TIME_UNIT_DAYS[time_unit] * extra_divisor

    frame = nw.from_native(transactions)
    eager_input = isinstance(frame, nw.DataFrame)

    columns = [customer_id_col, datetime_col]
    if monetary_value_col:
        columns.append(monetary_value_col)
    schema = _validate_input(frame, columns, datetime_col)
    frame = frame.select(columns)
    frame, time_zone = _prepare_datetime_column(
        frame, datetime_col, schema[datetime_col]
    )

    train_end = _coerce_period_end(train_period_end, time_zone, "train_period_end")
    train_end_period = _period_start_scalar(train_end, unit, week_start_offset)

    frame = frame.with_columns(
        _period_start(nw.col(datetime_col), unit, week_start_offset).alias("_period")
    )
    train_mask = nw.col(datetime_col) <= nw.lit(train_end)
    test_mask = nw.col(datetime_col) > nw.lit(train_end)
    if test_period_end is not None:
        test_end = _coerce_period_end(test_period_end, time_zone, "test_period_end")
        test_mask = test_mask & (nw.col(datetime_col) <= nw.lit(test_end))

    if eager_input:
        # Only an eager input can be inspected without materialising a query.
        eager = frame  # type: ignore[assignment]
        if eager.filter(train_mask).is_empty():  # type: ignore[union-attr]
            raise ValueError(_NO_DATA_MESSAGE)
        if eager.filter(test_mask).is_empty():  # type: ignore[union-attr]
            raise ValueError(_NO_DATA_MESSAGE)

    lazy = frame.lazy()
    training = _rfm_summary_frame(
        lazy.filter(train_mask).drop("_period"),
        customer_id_col,
        datetime_col,
        monetary_value_col,
        train_end,
        bool(include_first_transaction),
        unit,
        week_start_offset,
        divisor,
    )

    test_period_aggs = []
    if monetary_value_col:
        test_period_aggs.append(nw.col(monetary_value_col).sum().alias("_monetary"))
    test_periods = lazy.filter(test_mask).group_by(customer_id_col, "_period")
    test_periods_frame = (
        test_periods.agg(*test_period_aggs) if test_period_aggs else test_periods.agg()
    )
    test_aggs = [
        nw.len().alias("test_frequency"),
        # Last period with a test transaction; its maximum over all customers is
        # the end of the study when ``test_period_end`` is not given.
        nw.col("_period").max().alias("_test_end"),
    ]
    if monetary_value_col:
        test_aggs.append(nw.col("_monetary").mean().alias("test_monetary_value"))
    test_data = (
        test_periods_frame.group_by(customer_id_col)
        .agg(*test_aggs)
        .rename({customer_id_col: "customer_id"})
    )

    fill_columns = ["test_frequency"]
    if monetary_value_col:
        fill_columns.append("test_monetary_value")
    result = training.join(test_data, on="customer_id", how="left").with_columns(
        *[nw.col(name).fill_null(0.0).cast(nw.Float64) for name in fill_columns]
    )

    if test_period_end is None:
        # The study ends at the last transaction of the whole log, which lies in
        # the test window whenever that window is not empty.
        result = result.with_columns(
            (
                (nw.col("_test_end").max() - nw.lit(train_end_period)).cast(nw.Float64)
                / divisor
            ).alias("test_T")
        )
    else:
        test_end_period = _period_start_scalar(test_end, unit, week_start_offset)
        result = result.with_columns(
            nw.lit(
                (test_end_period - train_end_period) / divisor, dtype=nw.Float64
            ).alias("test_T")
        )

    output_columns = [
        *training.collect_schema().names(),
        *fill_columns,
        "test_T",
    ]
    return _finalize(
        result.select(output_columns).sort("customer_id"),
        backend,
        eager_input=eager_input,
    )


def _rfm_segments_nw(
    transactions: Any,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str,
    segment_config: dict | None = None,
    observation_period_end: PeriodEnd | None = None,
    *,
    granularity: str | None = None,
    time_unit: str | None = None,
    week_start: str | None = None,
    backend: Backend | None = None,
    extra_divisor: float = 1.0,
    stacklevel: int = 2,
) -> Any:
    """Dataframe-agnostic implementation of :func:`rfm_segments`.

    The quartile edges are a one-row aggregate that has to be materialised: the
    number of distinct edges decides how many score labels exist, exactly as
    ``pandas.qcut(duplicates="drop")`` does on the legacy path. Lazy backends
    therefore evaluate the summary twice, once for the edges and once for the
    final result.
    """
    unit, time_unit, week_start_offset = _resolve_time_spec(
        granularity, time_unit, week_start
    )
    divisor = TIME_UNIT_DAYS[time_unit] * extra_divisor

    frame = nw.from_native(transactions)
    eager_input = isinstance(frame, nw.DataFrame)
    summary = _rfm_summary_frame(
        frame.lazy(),
        customer_id_col,
        datetime_col,
        monetary_value_col,
        observation_period_end,
        True,
        unit,
        week_start_offset,
        divisor,
    )

    edges = summary.select(
        *[
            nw.col(metric)
            .quantile(probability, interpolation="linear")
            .alias(f"{metric}_{index}")
            for _, metric in _SEGMENT_METRICS
            for index, probability in enumerate(_QUARTILE_PROBABILITIES)
        ]
    ).collect()
    edge_values = {
        name: values[0] for name, values in edges.to_dict(as_series=False).items()
    }

    score_columns = []
    for score_name, metric in _SEGMENT_METRICS:
        # pandas.qcut drops duplicated edges and bins right-closed, with the
        # lowest edge included in the first bin.
        unique_edges = sorted(
            {float(edge_values[f"{metric}_{index}"]) for index in range(5)}
        )
        n_bins = len(unique_edges) - 1
        if n_bins < 2:
            raise ValueError(
                "Bin labels must be one fewer than the number of bin edges"
            )
        if n_bins == 2:
            warnings.warn(
                f"RFM score will not exceed 2 for {score_name}. Specify a custom segment_config",
                UserWarning,
                stacklevel=stacklevel,
            )
        labels = [str(label) for label in _rfm_quartile_labels(score_name, n_bins + 1)]
        score: Any = nw.when(nw.col(metric) <= unique_edges[1]).then(nw.lit(labels[0]))
        for index in range(2, n_bins):
            score = score.when(nw.col(metric) <= unique_edges[index]).then(
                nw.lit(labels[index - 1])
            )
        score_columns.append(score.otherwise(nw.lit(labels[-1])).alias(score_name))

    if segment_config is None:
        segment_config = _default_rfm_segment_config

    # Later segments win when a score is listed twice, like the legacy loop.
    segment: Any = nw.lit("Other")
    for name, scores in segment_config.items():
        segment = (
            nw.when(nw.col("rfm_score").is_in(list(scores)))
            .then(nw.lit(name))
            .otherwise(segment)
        )

    result = (
        summary.with_columns(*score_columns)
        .with_columns(
            nw.concat_str(
                nw.col("r_quartile"), nw.col("f_quartile"), nw.col("m_quartile")
            ).alias("rfm_score")
        )
        .with_columns(segment.alias("segment"))
        .select(
            "customer_id",
            "frequency",
            "recency",
            "monetary_value",
            "rfm_score",
            "segment",
        )
    )
    return _finalize(result, backend, eager_input=eager_input)


def rfm_summary(
    transactions: Any,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None = None,
    datetime_format: str | None = None,
    observation_period_end: PeriodEnd | None = None,
    time_unit: str = "D",
    time_scaler: float | None = 1,
    include_first_transaction: bool | None = False,
    sort_transactions: bool | None = True,
    *,
    granularity: str | None = None,
    week_start: str | None = None,
    backend: Backend | None = None,
) -> Any:
    """Summarize transaction data for use in CLV modeling or RFM segmentation.

    This transforms a DataFrame of transaction data of the form:
        *customer_id, datetime [, monetary_value]*
    to a DataFrame for CLV modeling:
        *customer_id, frequency, recency, T [, monetary_value]*

    If the `include_first_transaction = True` argument is specified, a DataFrame for RFM segmentation is returned:
        *customer_id, frequency, recency, monetary_value*

    This function is not required if using the `clv.rfm_segments` utility.

    Any dataframe supported by `Narwhals <https://narwhals-dev.github.io/narwhals/>`_
    is accepted: pandas, Polars (eager or lazy), PyArrow, DuckDB, Ibis, PySpark and
    more. The result is returned in the same library as the input (a lazy input
    gives a lazy result) unless ``backend`` is given.

    Adapted from the legacy ``lifetimes`` library:
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L230

    Parameters
    ----------
    transactions : DataFrame
        A dataframe containing *customer_id_col* and *datetime_col*. On every
        library except pandas, *datetime_col* must already have a datetime (or
        date) dtype.
    customer_id_col : string
        Column in the *transactions* DataFrame denoting the *customer_id*.
    datetime_col :  string
        Column in the *transactions* DataFrame denoting datetimes purchase were made.
    monetary_value_col : string, optional
        Column in the transactions DataFrame denoting the monetary value of the transaction.
        Optional; only needed for RFM segmentation and spend estimation models like the Gamma-Gamma model.
    datetime_format : string, optional
        A string that represents the timestamp format. Useful if Pandas doesn't recognize the provided format.

        .. deprecated:: 1.4.0
            Only supported for pandas input; removed in v2.0. Cast the column to a
            datetime dtype before calling instead.
    observation_period_end : Union[str, date, datetime, pandas.Period], optional
        A string or datetime to denote the final date of the study.
        Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
        On the Narwhals path strings must be ISO-8601 and the time-zone awareness must
        match the column's.
    time_unit : string, optional
        Unit of *recency* and *T*. Default ``"D"``.

        Named units select the Narwhals path and keep *recency* and *T* as continuous
        floats: ``"day"``, ``"week"`` (7 days), ``"month"`` (30.4375 days),
        ``"quarter"`` (91.3125 days) and ``"year"`` (365.25 days). The legacy pandas
        aliases ``"D"``, ``"W"``, ``"M"`` and ``"h"`` count calendar periods on the
        legacy path; ``"D"`` and ``"W"`` are translated for non-pandas input.

        .. deprecated:: 1.4.0
            The legacy aliases are removed in v2.0. Use a named unit together with
            ``granularity``.
    time_scaler : int, optional
        Default: 1. Scales *recency* & *T* to a different time granularity.
        This is useful for datasets spanning many years, and running predictions in different time scales.

        .. deprecated:: 1.4.0
            Removed in v2.0. Use ``time_unit="week"`` (and so on) instead.
    include_first_transaction : bool, optional
        Default: *False*
        For predictive CLV modeling, this should be *False*.
        Set to *True* if performing RFM segmentation.
    sort_transactions : bool, optional
        Default: *True*
        If raw data is already sorted in chronological order, set to *False* to improve computational efficiency.

        .. deprecated:: 1.4.0
            Removed in v2.0. The Narwhals path never depends on row order.
    granularity : string, optional
        Calendar bucket that collapses repeat purchases into one transaction:
        ``"1d"`` (default), ``"1w"``, ``"1mo"``, ``"1q"`` or ``"1y"``. Selects the
        Narwhals path. *recency* and *T* are measured in days between the starts of
        the first, last and final periods, then divided by the ``time_unit`` constant.
    week_start : string, optional
        First day of the week for ``granularity="1w"``, e.g. ``"wednesday"``.
        Default ``"monday"``, which matches the legacy ``"W"`` periods.
    backend : {"pandas", "polars", "pyarrow"}, optional
        Collect the result into this library instead of returning the input's own
        type.

    Returns
    -------
    DataFrame
        Dataframe containing summarized RFM data, and test columns for *frequency*, *T*,
        and *monetary_value* if specified. Same library as the input unless ``backend``
        is given; on the Narwhals path rows are sorted by ``customer_id``.

    Notes
    -----
    pandas input with legacy arguments only keeps using the pandas implementation
    unchanged. Passing any of ``granularity``, ``week_start``, ``backend`` or a named
    ``time_unit``, or any non-pandas dataframe, uses the Narwhals implementation.
    Legacy and new parameters cannot be mixed in one call.

    """
    path, resolved = _route(
        transactions,
        time_unit=time_unit,
        time_scaler=time_scaler,
        datetime_format=datetime_format,
        sort_transactions=sort_transactions,
        granularity=granularity,
        week_start=week_start,
        backend=backend,
    )
    if path == "legacy":
        return _rfm_summary_legacy(
            transactions,
            customer_id_col,
            datetime_col,
            monetary_value_col=monetary_value_col,
            observation_period_end=observation_period_end,
            include_first_transaction=include_first_transaction,
            **resolved,
        )
    return _rfm_summary_nw(
        transactions,
        customer_id_col,
        datetime_col,
        monetary_value_col=monetary_value_col,
        observation_period_end=observation_period_end,
        include_first_transaction=include_first_transaction,
        **resolved,
    )


def rfm_train_test_split(
    transactions: Any,
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
    *,
    granularity: str | None = None,
    week_start: str | None = None,
    backend: Backend | None = None,
) -> Any:
    """Summarize transaction data and split into training and tests datasets for CLV modeling.

    This can also be used to evaluate the impact of a time-based intervention like a marketing campaign.

    This transforms a DataFrame of transaction data of the form:
        *customer_id, datetime [, monetary_value]*
    to a DataFrame of the form:
        *customer_id, frequency, recency, T [, monetary_value], test_frequency [, test_monetary_value], test_T*

    Note this function will exclude new customers whose first transactions occurred during the test period.

    Any dataframe supported by Narwhals is accepted; see :func:`rfm_summary` for the
    routing rules shared by the CLV utilities.

    Adapted from the legacy ``lifetimes`` library:
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L27

    Parameters
    ----------
    transactions : DataFrame
        A dataframe containing *customer_id_col* and *datetime_col*.
    customer_id_col : string
        Column in the *transactions* DataFrame denoting the customer_id.
    datetime_col :  string
        Column in the *transactions* DataFrame denoting datetimes purchases were made.
    train_period_end : Union[str, date, datetime, pandas.Period]
        A string or datetime to denote the final time period for the training data.
        Events after this time period are used for the test data.
    test_period_end : Union[str, date, datetime, pandas.Period], optional
        A string or datetime to denote the final time period of the study.
        Events after this date are truncated. If not given, defaults to the max of *datetime_col*.
    time_unit : string, optional
        Unit of *recency*, *T* and *test_T*; see :func:`rfm_summary`. Default ``"D"``.
    time_scaler : int, optional
        Default: 1. Scales *recency* & *T* to a different time granularity.
        This is useful for datasets spanning many years, and running predictions in different time scales.

        .. deprecated:: 1.4.0
            Removed in v2.0. Use a named ``time_unit`` instead.
    datetime_format : string, optional
        A string that represents the timestamp format. Useful if Pandas doesn't recognize the provided format.

        .. deprecated:: 1.4.0
            Only supported for pandas input; removed in v2.0.
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

        .. deprecated:: 1.4.0
            Removed in v2.0. The Narwhals path never depends on row order.
    granularity : string, optional
        Calendar bucket, ``"1d"`` (default) to ``"1y"``; see :func:`rfm_summary`.
    week_start : string, optional
        First day of the week for ``granularity="1w"``.
    backend : {"pandas", "polars", "pyarrow"}, optional
        Collect the result into this library instead of returning the input's own type.

    Returns
    -------
    DataFrame
        Dataframe containing summarized RFM data, and test columns for *frequency*, *T*,
        and *monetary_value* if specified. On the Narwhals path *test_frequency* is a
        float and rows are sorted by ``customer_id``.

    """
    path, resolved = _route(
        transactions,
        time_unit=time_unit,
        time_scaler=time_scaler,
        datetime_format=datetime_format,
        sort_transactions=sort_transactions,
        granularity=granularity,
        week_start=week_start,
        backend=backend,
    )
    if path == "legacy":
        return _rfm_train_test_split_legacy(
            transactions,
            customer_id_col,
            datetime_col,
            train_period_end,
            test_period_end=test_period_end,
            monetary_value_col=monetary_value_col,
            include_first_transaction=include_first_transaction,
            **resolved,
        )
    return _rfm_train_test_split_nw(
        transactions,
        customer_id_col,
        datetime_col,
        train_period_end,  # type: ignore[arg-type]
        test_period_end=test_period_end,  # type: ignore[arg-type]
        monetary_value_col=monetary_value_col,
        include_first_transaction=include_first_transaction,
        **resolved,
    )


def rfm_segments(
    transactions: Any,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str,
    segment_config: dict | None = None,
    observation_period_end: PeriodEnd | None = None,
    datetime_format: str | None = None,
    time_unit: str = "D",
    time_scaler: float | None = 1,
    sort_transactions: bool | None = True,
    *,
    granularity: str | None = None,
    week_start: str | None = None,
    backend: Backend | None = None,
) -> Any:
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

    Any dataframe supported by Narwhals is accepted; see :func:`rfm_summary` for the
    routing rules shared by the CLV utilities.

    Parameters
    ----------
    transactions : DataFrame
        A dataframe containing *customer_id_col* and *datetime_col*.
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
    observation_period_end : Union[str, date, datetime, pandas.Period], optional
        A string or datetime to denote the final date of the study.
        Events after this date are truncated. If not given, defaults to the max of *datetime_col*.
    datetime_format : string, optional
        A string that represents the timestamp format. Useful if Pandas doesn't recognize the provided format.

        .. deprecated:: 1.4.0
            Only supported for pandas input; removed in v2.0.
    time_unit : string, optional
        Unit of *recency*; see :func:`rfm_summary`. Default ``"D"``.
    time_scaler : int, optional
        Default: 1. Scales *recency* & *T* to a different time granularity.
        This is useful for datasets spanning many years, and running predictions in different time scales.

        .. deprecated:: 1.4.0
            Removed in v2.0. Use a named ``time_unit`` instead.
    sort_transactions : bool, optional
        Default: *True*
        If raw data is already sorted in chronological order, set to *False* to improve computational efficiency.

        .. deprecated:: 1.4.0
            Removed in v2.0. The Narwhals path never depends on row order.
    granularity : string, optional
        Calendar bucket, ``"1d"`` (default) to ``"1y"``; see :func:`rfm_summary`.
    week_start : string, optional
        First day of the week for ``granularity="1w"``.
    backend : {"pandas", "polars", "pyarrow"}, optional
        Collect the result into this library instead of returning the input's own type.

    Returns
    -------
    DataFrame
        Dataframe containing summarized RFM data, RFM scores, and segment assignments.
        On the Narwhals path rows are sorted by ``customer_id``.

    """
    path, resolved = _route(
        transactions,
        time_unit=time_unit,
        time_scaler=time_scaler,
        datetime_format=datetime_format,
        sort_transactions=sort_transactions,
        granularity=granularity,
        week_start=week_start,
        backend=backend,
    )
    if path == "legacy":
        return _rfm_segments_legacy(
            transactions,
            customer_id_col,
            datetime_col,
            monetary_value_col,
            segment_config=segment_config,
            observation_period_end=observation_period_end,
            **resolved,
        )
    return _rfm_segments_nw(
        transactions,
        customer_id_col,
        datetime_col,
        monetary_value_col,
        segment_config=segment_config,
        observation_period_end=observation_period_end,
        stacklevel=3,
        **resolved,
    )


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
        Default: 'D' for days. Other options are 'W' (weekly), 'M' (monthly)
        and 'H' (hourly).
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
    time_unit = _normalize_time_unit(time_unit)
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

    date_range = pandas.date_range(
        start_date, periods=t + 1, freq=_OFFSET_ALIASES.get(time_unit, time_unit)
    )
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
            times = np.array([d.n for d in period - first_trans_size.index])  # type: ignore[operator]
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
