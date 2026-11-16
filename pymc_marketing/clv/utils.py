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
from typing import Any, overload

import narwhals.stable.v2 as nw
import numpy as np
import pandas
import xarray
from narwhals.stable.v2.typing import IntoFrameT
from numpy import datetime64

from pymc_marketing.clv._narwhals_utils import (
    Backend,
    _coerce_period_end,
    _elapsed,
    _finalize,
    _period_start,
    _period_start_scalar,
    _prepare_datetime_column,
    _resolve_time_spec,
    _validate_backend,
    _validate_input,
)
from pymc_marketing.constants import DAYS_IN_MONTH, DAYS_IN_WEEK
from pymc_marketing.names.clv import Col, Dim

__all__ = [
    "customer_lifetime_value",
    "rfm_segments",
    "rfm_summary",
    "rfm_train_test_split",
    "to_xarray",
]

# Values accepted as the end of an observation, training or test period.
PeriodEnd = str | date | datetime | datetime64 | pandas.Period | pandas.Timestamp

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
    if Col.FUTURE_SPEND not in data.columns:
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

    # DAYS_IN_MONTH is the average calendar month (365.25 / 12). A flat 30 days
    # understates a year by 5.25 days and inflates CLV for the "D", "W" and "h" units.
    factor = {
        "W": DAYS_IN_MONTH / DAYS_IN_WEEK,
        "M": 1.0,
        "D": DAYS_IN_MONTH,
        "h": DAYS_IN_MONTH * 24,
    }[time_unit]

    monetary_value = to_xarray(data[Col.CUSTOMER_ID], data[Col.FUTURE_SPEND])

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

    return clv.transpose("chain", "draw", Dim.CUSTOMER_ID)


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
    period_transactions[Col.FIRST] = False
    # find all first transactions and store as an index
    first_transactions = (
        period_transactions.groupby(customer_id_col, sort=True, as_index=False)
        .head(1)
        .index
    )
    # flag first transactions as True
    period_transactions.loc[first_transactions, Col.FIRST] = True
    select_columns.append(Col.FIRST)
    # reset datetime_col to period
    period_transactions[datetime_col] = period_transactions[datetime_col].dt.to_period(
        time_unit
    )

    return period_transactions[select_columns]


def _rfm_quartile_labels(column_name: str, max_label_range: int) -> list[int] | range:
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
    if column_name == Col.R_QUARTILE:
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


_SEGMENT_METRICS = (
    (Col.R_QUARTILE, Col.RECENCY),
    (Col.F_QUARTILE, Col.FREQUENCY),
    (Col.M_QUARTILE, Col.MONETARY_VALUE),
)
_QUARTILE_PROBABILITIES = (0.0, 0.25, 0.5, 0.75, 1.0)
# Relative slack on a quartile edge when edges and bins come from two evaluations.
_EDGE_TOLERANCE = 1e-12

_NO_DATA_MESSAGE = """No data available. Check `test_transactions` and `train_period_end`
        and confirm values in `transactions` occur prior to those time periods."""


def _rfm_summary_frame(
    frame: nw.LazyFrame[Any],
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None,
    observation_period_end: PeriodEnd | None,
    include_first_transaction: bool,
    unit: str,
    week_start_offset: int,
    time_unit: str,
) -> nw.LazyFrame[Any]:
    """Build the lazy RFM summary shared by the public RFM functions.

    The result is sorted by ``customer_id``.
    """
    columns = [customer_id_col, datetime_col]
    if monetary_value_col:
        columns.append(monetary_value_col)
    schema = _validate_input(frame, columns, datetime_col)
    # Rows without a customer or a timestamp cannot be summarised.
    frame = frame.select(columns).drop_nulls([customer_id_col, datetime_col])
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

    # Purchases inside one period count as a single transaction.
    customer_aggs = [
        nw.col("_period").min().alias("_first"),
        nw.col("_period").max().alias("_last"),
        nw.col("_period").n_unique().alias("_n_periods"),
        nw.col("_end").max().alias("_end"),
    ]
    if monetary_value_col:
        spend = nw.col(monetary_value_col)
        if not include_first_transaction:
            # Spend after the customer's first period, isolated with a window
            # function so that neither a join nor a subtraction is needed.
            frame = frame.with_columns(
                nw.when(
                    nw.col("_period") != nw.col("_period").min().over(customer_id_col)
                )
                .then(spend)
                .otherwise(nw.lit(0.0))
                .alias("_repeat_monetary")
            )
            spend = nw.col("_repeat_monetary")
        customer_aggs.append(spend.sum().alias("_monetary_sum"))
    customers = frame.group_by(customer_id_col).agg(*customer_aggs)

    n_periods = nw.col("_n_periods").cast(nw.Float64)
    recency = _elapsed(
        (nw.col("_last") - nw.col("_first")).cast(nw.Float64), unit, time_unit
    )
    t_value = _elapsed(
        (nw.col("_end") - nw.col("_first")).cast(nw.Float64), unit, time_unit
    )

    if include_first_transaction:
        # Segmentation definition: time since the last purchase.
        summary = [
            n_periods.alias(Col.FREQUENCY),
            (t_value - recency).alias(Col.RECENCY),
        ]
    else:
        summary = [
            (n_periods - 1).alias(Col.FREQUENCY),
            recency.alias(Col.RECENCY),
            t_value.alias(Col.T),
        ]

    if monetary_value_col:
        if include_first_transaction:
            monetary = nw.col("_monetary_sum") / n_periods
        else:
            monetary = (
                nw.when(nw.col("_n_periods") > 1)
                .then(nw.col("_monetary_sum") / (n_periods - 1))
                .otherwise(nw.lit(0.0))
            )
        summary.append(monetary.cast(nw.Float64).alias(Col.MONETARY_VALUE))

    return customers.select(
        nw.col(customer_id_col).alias(Col.CUSTOMER_ID), *summary
    ).sort(Col.CUSTOMER_ID)


@overload
def rfm_summary(
    transactions: IntoFrameT,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None = None,
    *,
    observation_period_end: PeriodEnd | None = None,
    include_first_transaction: bool = False,
    granularity: str | None = None,
    time_unit: str = "day",
    week_start: str | None = None,
    backend: None = None,
) -> IntoFrameT: ...
@overload
def rfm_summary(
    transactions: Any,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None = None,
    *,
    observation_period_end: PeriodEnd | None = None,
    include_first_transaction: bool = False,
    granularity: str | None = None,
    time_unit: str = "day",
    week_start: str | None = None,
    backend: Backend,
) -> Any: ...
def rfm_summary(
    transactions: Any,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None = None,
    *,
    observation_period_end: PeriodEnd | None = None,
    include_first_transaction: bool = False,
    granularity: str | None = None,
    time_unit: str = "day",
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
        A dataframe containing *customer_id_col* and *datetime_col*. *datetime_col*
        must have a datetime or date dtype; cast strings and integers first, e.g.
        with ``pandas.to_datetime``. Rows with a missing customer id or datetime are
        dropped.
    customer_id_col : string
        Column in the *transactions* DataFrame denoting the *customer_id*.
    datetime_col :  string
        Column in the *transactions* DataFrame denoting datetimes purchase were made.
    monetary_value_col : string, optional
        Column in the transactions DataFrame denoting the monetary value of the transaction.
        Optional; only needed for RFM segmentation and spend estimation models like the Gamma-Gamma model.
    observation_period_end : Union[str, date, datetime, numpy.datetime64, pandas.Period], optional
        A string or datetime to denote the final date of the study.
        Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
        Strings must be ISO-8601 and the time-zone awareness must match the column's.
    include_first_transaction : bool, optional
        Default: *False*
        For predictive CLV modeling, this should be *False*.
        Set to *True* if performing RFM segmentation.
    granularity : string, optional
        Calendar bucket that collapses repeat purchases into one transaction:
        ``"1d"``, ``"1w"``, ``"1mo"``, ``"1q"`` or ``"1y"``. Defaults to the bucket of
        ``time_unit``.

        When the bucket and ``time_unit`` name the same period, *recency* and *T*
        are whole period counts. Otherwise they are the days between the starts of
        the first, last and final periods divided by a fixed length: 7 days per
        week, 30.4375 per month, 91.3125 per quarter and 365.25 per year. For
        example ``granularity="1d", time_unit="week"`` gives fractional weeks.
    time_unit : {"day", "week", "month", "quarter", "year"}, optional
        Unit of *recency* and *T*. Default ``"day"``. Without ``granularity`` it also
        sets the calendar bucket, so ``time_unit="week"`` counts whole weeks.
    week_start : string, optional
        First day of the week for weekly buckets, e.g. ``"wednesday"``.
        Default ``"monday"``.
    backend : {"pandas", "polars", "pyarrow"}, optional
        Collect the result into this library instead of returning the input's own
        type.

    Returns
    -------
    DataFrame
        Dataframe containing summarized RFM data, sorted by ``customer_id``. Same
        library as the input unless ``backend`` is given.

    Notes
    -----
    A time-zone-aware *datetime_col* is bucketed by its wall-clock time in the
    column's own zone. DuckDB reports ``TIMESTAMPTZ`` columns in the connection's
    session ``TimeZone``, so set that explicitly (or cast to a plain ``TIMESTAMP``)
    to get the same buckets on every machine.

    Examples
    --------
    Daily periods on any dataframe library:

    .. code-block:: python

        import polars as pl
        from pymc_marketing.clv import rfm_summary

        transactions = pl.read_csv("transactions.csv", try_parse_dates=True)
        rfm = rfm_summary(transactions, "customer_id", "date", "spend")

    Whole weeks starting on Sunday, returned as a pandas DataFrame:

    .. code-block:: python

        rfm = rfm_summary(
            transactions,
            "customer_id",
            "date",
            time_unit="week",
            week_start="sunday",
            backend="pandas",
        )

    Monthly purchase periods with *recency* and *T* in days:

    .. code-block:: python

        rfm = rfm_summary(
            transactions, "customer_id", "date", granularity="1mo", time_unit="day"
        )

    """
    _validate_backend(backend)
    unit, time_unit, week_start_offset = _resolve_time_spec(
        granularity, time_unit, week_start
    )

    frame = nw.from_native(transactions)
    summary = _rfm_summary_frame(
        frame.lazy(),
        customer_id_col,
        datetime_col,
        monetary_value_col,
        observation_period_end,
        include_first_transaction,
        unit,
        week_start_offset,
        time_unit,
    )
    return _finalize(summary, backend, eager_input=isinstance(frame, nw.DataFrame))


@overload
def rfm_train_test_split(
    transactions: IntoFrameT,
    customer_id_col: str,
    datetime_col: str,
    train_period_end: PeriodEnd,
    test_period_end: PeriodEnd | None = None,
    *,
    monetary_value_col: str | None = None,
    include_first_transaction: bool = False,
    granularity: str | None = None,
    time_unit: str = "day",
    week_start: str | None = None,
    backend: None = None,
) -> IntoFrameT: ...
@overload
def rfm_train_test_split(
    transactions: Any,
    customer_id_col: str,
    datetime_col: str,
    train_period_end: PeriodEnd,
    test_period_end: PeriodEnd | None = None,
    *,
    monetary_value_col: str | None = None,
    include_first_transaction: bool = False,
    granularity: str | None = None,
    time_unit: str = "day",
    week_start: str | None = None,
    backend: Backend,
) -> Any: ...
def rfm_train_test_split(
    transactions: Any,
    customer_id_col: str,
    datetime_col: str,
    train_period_end: PeriodEnd,
    test_period_end: PeriodEnd | None = None,
    *,
    monetary_value_col: str | None = None,
    include_first_transaction: bool = False,
    granularity: str | None = None,
    time_unit: str = "day",
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

    Any dataframe supported by Narwhals is accepted; see :func:`rfm_summary`.

    Adapted from the legacy ``lifetimes`` library:
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L27

    Parameters
    ----------
    transactions : DataFrame
        A dataframe containing *customer_id_col* and *datetime_col*, which must have
        a datetime or date dtype.
    customer_id_col : string
        Column in the *transactions* DataFrame denoting the customer_id.
    datetime_col :  string
        Column in the *transactions* DataFrame denoting datetimes purchases were made.
    train_period_end : Union[str, date, datetime, numpy.datetime64, pandas.Period]
        A string or datetime to denote the final time period for the training data.
        Events after this time period are used for the test data.
    test_period_end : Union[str, date, datetime, numpy.datetime64, pandas.Period], optional
        A string or datetime to denote the final time period of the study.
        Events after this date are truncated. If not given, defaults to the max of *datetime_col*.
    monetary_value_col : string, optional
        Column in the *transactions* DataFrame that denotes the monetary value of the transaction.
        Optional; only needed for spend estimation models like the Gamma-Gamma model.
    include_first_transaction : bool, optional
        Default: *False*
        For predictive CLV modeling, this should be *False*.
        Set to *True* if performing RFM segmentation.
    granularity : string, optional
        Calendar bucket, ``"1d"`` to ``"1y"``; see :func:`rfm_summary`.
    time_unit : {"day", "week", "month", "quarter", "year"}, optional
        Unit of *recency*, *T* and *test_T*; see :func:`rfm_summary`. Default ``"day"``.
    week_start : string, optional
        First day of the week for weekly buckets.
    backend : {"pandas", "polars", "pyarrow"}, optional
        Collect the result into this library instead of returning the input's own type.

    Returns
    -------
    DataFrame
        Dataframe containing summarized RFM data, and test columns for *frequency*, *T*,
        and *monetary_value* if specified, sorted by ``customer_id``.

    Raises
    ------
    ValueError
        If the training or the test period holds no transactions. For a lazy input
        (Polars ``LazyFrame``, DuckDB, ...) this check runs two one-row queries
        when the function is called; the result itself stays lazy.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.clv import rfm_train_test_split

        rfm = rfm_train_test_split(
            transactions,
            "customer_id",
            "date",
            train_period_end="2015-06-30",
            time_unit="week",
        )

    """
    _validate_backend(backend)
    unit, time_unit, week_start_offset = _resolve_time_spec(
        granularity, time_unit, week_start
    )

    frame = nw.from_native(transactions)
    eager_input = isinstance(frame, nw.DataFrame)

    columns = [customer_id_col, datetime_col]
    if monetary_value_col:
        columns.append(monetary_value_col)
    schema = _validate_input(frame, columns, datetime_col)
    frame = frame.select(columns).drop_nulls([customer_id_col, datetime_col])
    frame, time_zone = _prepare_datetime_column(
        frame, datetime_col, schema[datetime_col]
    )

    train_end = _coerce_period_end(train_period_end, time_zone, "train_period_end")
    train_end_period = _period_start_scalar(train_end, unit, week_start_offset)

    train_mask = nw.col(datetime_col) <= nw.lit(train_end)
    test_mask = nw.col(datetime_col) > nw.lit(train_end)
    if test_period_end is not None:
        test_end = _coerce_period_end(test_period_end, time_zone, "test_period_end")
        test_mask = test_mask & (nw.col(datetime_col) <= nw.lit(test_end))

    lazy = frame.lazy()
    # One row per period is enough to tell; a lazy input runs two small queries.
    for mask in (train_mask, test_mask):
        if lazy.filter(mask).head(1).collect().is_empty():
            raise ValueError(_NO_DATA_MESSAGE)

    training = _rfm_summary_frame(
        lazy.filter(train_mask),
        customer_id_col,
        datetime_col,
        monetary_value_col,
        train_end,
        include_first_transaction,
        unit,
        week_start_offset,
        time_unit,
    )

    test_rows = lazy.filter(test_mask).with_columns(
        _period_start(nw.col(datetime_col), unit, week_start_offset).alias("_period")
    )
    test_aggs = [nw.col("_period").n_unique().alias("test_frequency")]
    if monetary_value_col:
        test_aggs.append(nw.col(monetary_value_col).sum().alias("test_monetary_value"))
    test_data = (
        test_rows.group_by(customer_id_col)
        .agg(*test_aggs)
        .rename({customer_id_col: Col.CUSTOMER_ID})
    )
    result = training.join(test_data, on=Col.CUSTOMER_ID, how="left")

    fill_columns = ["test_frequency"]
    if monetary_value_col:
        # Mean spend per test period.
        result = result.with_columns(
            nw.col("test_monetary_value") / nw.col("test_frequency")
        )
        fill_columns.append("test_monetary_value")
    result = result.with_columns(
        *[nw.col(name).fill_null(0.0).cast(nw.Float64) for name in fill_columns]
    )

    if test_period_end is None:
        # The study ends at the last transaction of the whole log, whoever made it:
        # customers who first buy in the test window count here even though they
        # have no row in the result.
        study_end = test_rows.select(nw.col("_period").max().alias("_test_end"))
        result = result.join(study_end, how="cross").with_columns(
            _elapsed(
                (nw.col("_test_end") - nw.lit(train_end_period)).cast(nw.Float64),
                unit,
                time_unit,
            ).alias("test_T")
        )
    else:
        test_end_period = _period_start_scalar(test_end, unit, week_start_offset)
        result = result.with_columns(
            nw.lit(
                _elapsed(test_end_period - train_end_period, unit, time_unit),
                dtype=nw.Float64,
            ).alias("test_T")
        )

    output_columns = [
        *training.collect_schema().names(),
        *fill_columns,
        "test_T",
    ]
    return _finalize(
        result.select(output_columns).sort(Col.CUSTOMER_ID),
        backend,
        eager_input=eager_input,
    )


@overload
def rfm_segments(
    transactions: IntoFrameT,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str,
    segment_config: dict | None = None,
    *,
    observation_period_end: PeriodEnd | None = None,
    granularity: str | None = None,
    time_unit: str = "day",
    week_start: str | None = None,
    backend: None = None,
) -> IntoFrameT: ...
@overload
def rfm_segments(
    transactions: Any,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str,
    segment_config: dict | None = None,
    *,
    observation_period_end: PeriodEnd | None = None,
    granularity: str | None = None,
    time_unit: str = "day",
    week_start: str | None = None,
    backend: Backend,
) -> Any: ...
def rfm_segments(
    transactions: Any,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str,
    segment_config: dict | None = None,
    *,
    observation_period_end: PeriodEnd | None = None,
    granularity: str | None = None,
    time_unit: str = "day",
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

    Any dataframe supported by Narwhals is accepted; see :func:`rfm_summary`.

    Parameters
    ----------
    transactions : DataFrame
        A dataframe containing *customer_id_col* and *datetime_col*, which must have
        a datetime or date dtype.
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
    observation_period_end : Union[str, date, datetime, numpy.datetime64, pandas.Period], optional
        A string or datetime to denote the final date of the study.
        Events after this date are truncated. If not given, defaults to the max of *datetime_col*.
    granularity : string, optional
        Calendar bucket, ``"1d"`` to ``"1y"``; see :func:`rfm_summary`.
    time_unit : {"day", "week", "month", "quarter", "year"}, optional
        Unit of *recency*; see :func:`rfm_summary`. Default ``"day"``.
    week_start : string, optional
        First day of the week for weekly buckets.
    backend : {"pandas", "polars", "pyarrow"}, optional
        Collect the result into this library instead of returning the input's own type.

    Returns
    -------
    DataFrame
        Dataframe containing summarized RFM data, RFM scores, and segment assignments,
        sorted by ``customer_id``.

    Notes
    -----
    A lazy input (Polars ``LazyFrame``, DuckDB, ...) stays lazy and is evaluated
    twice, once for the quartile edges and once for the result. Engines that add
    floats in parallel can return a *monetary_value* that differs in its last bits
    from one run to the next, so the edges of a metric are compared with a
    tolerance of 1e-12 times its largest edge, which keeps a customer sitting
    exactly on an edge in the same bin. Edges closer to each other than that
    tolerance count as one.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.clv import rfm_segments

        segments = rfm_segments(transactions, "customer_id", "date", "spend")

    """
    _validate_backend(backend)
    unit, time_unit, week_start_offset = _resolve_time_spec(
        granularity, time_unit, week_start
    )

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
        time_unit,
    )
    # Parallel engines return float sums that differ in their last bits from one
    # evaluation to the next, which moves a customer sitting on a quartile edge to
    # another bin. Each edge therefore gets a slack, sized by the largest edge: the
    # error of a float sum follows the size of the values added, not the size of
    # the result, so an edge at zero made by refunds cancelling spend needs as much
    # slack as any other. An eager input is also summarised only once; a lazy input
    # stays lazy, as its summary may not fit in memory.
    if eager_input:
        summary = summary.collect().lazy()

    # The edges are a one-row aggregate that has to be materialised: the number of
    # distinct edges decides how many score labels exist.
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
        # Like pandas.qcut(duplicates="drop"): duplicated edges are dropped and bins
        # are right-closed, with the lowest edge included in the first bin.
        metric_edges = sorted(
            {float(edge_values[f"{metric}_{index}"]) for index in range(5)}
        )
        slack = _EDGE_TOLERANCE * max(abs(edge) for edge in metric_edges)
        # Edges closer than the slack are one edge, or the number of bins would
        # change between two runs.
        unique_edges = metric_edges[:1]
        for edge in metric_edges[1:]:
            if edge - unique_edges[-1] > slack:
                unique_edges.append(edge)
        n_bins = len(unique_edges) - 1
        unique_edges = [edge + slack for edge in unique_edges]
        if n_bins < 2:
            raise ValueError(
                "Bin labels must be one fewer than the number of bin edges"
            )
        if n_bins == 2:
            warnings.warn(
                f"RFM score will not exceed 2 for {score_name}. Specify a custom segment_config",
                UserWarning,
                stacklevel=2,
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

    # Later segments win when a score is listed twice.
    segment: Any = nw.lit("Other")
    for name, scores in segment_config.items():
        segment = (
            nw.when(nw.col(Col.RFM_SCORE).is_in(list(scores)))
            .then(nw.lit(name))
            .otherwise(segment)
        )

    result = (
        summary.with_columns(*score_columns)
        .with_columns(
            nw.concat_str(
                nw.col(Col.R_QUARTILE), nw.col(Col.F_QUARTILE), nw.col(Col.M_QUARTILE)
            ).alias(Col.RFM_SCORE)
        )
        .with_columns(segment.alias(Col.SEGMENT))
        .select(
            Col.CUSTOMER_ID,
            Col.FREQUENCY,
            Col.RECENCY,
            Col.MONETARY_VALUE,
            Col.RFM_SCORE,
            Col.SEGMENT,
        )
    )
    return _finalize(result, backend, eager_input=eager_input)


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
    first_trans_mask = repeated_and_first_transactions[Col.FIRST]
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
                    Col.CUSTOMER_ID: times,
                    Col.NEW_CUSTOMER_T: times,
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
