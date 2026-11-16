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
"""RFM summaries of transaction data, on any Narwhals dataframe."""

import warnings
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

import narwhals.stable.v2 as nw
import pandas
from narwhals.stable.v2.typing import IntoFrameT
from numpy import datetime64

from pymc_marketing.clv._narwhals_utils import (
    TimeSpec,
    _coerce_period_end,
    _finalize,
    _prepare_transactions,
    _resolve_time_spec,
)
from pymc_marketing.names.clv import Col

__all__ = ["rfm_segments", "rfm_summary", "rfm_train_test_split"]

# Values accepted as the end of an observation, training or test period.
PeriodEnd = str | date | datetime | datetime64 | pandas.Period | pandas.Timestamp

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


# Score column of each metric, in the order the digits of the RFM score are written.
_SCORED_METRICS = (
    (Col.R_QUARTILE, Col.RECENCY),
    (Col.F_QUARTILE, Col.FREQUENCY),
    (Col.M_QUARTILE, Col.MONETARY_VALUE),
)
# Slack on a quartile edge, as a share of the largest edge of the metric.
_EDGE_TOLERANCE = 1e-12


def _end_period(
    period_end: PeriodEnd | None, time_zone: str | None, name: str, spec: TimeSpec
) -> int | None:
    """Period a user-supplied study end falls in, or ``None`` when not given."""
    if period_end is None:
        return None
    return spec.period_of(_coerce_period_end(period_end, time_zone, name))


def _with_study_end(
    frame: nw.LazyFrame[Any], transactions: nw.LazyFrame[Any], end_period: int | None
) -> nw.LazyFrame[Any]:
    """Add ``_end``, the period the study ends in, to every row of ``frame``.

    Without ``end_period`` the study ends in the last period of ``transactions``.
    That is a one-row aggregate, cross joined. Broadcasting the max over the
    transactions instead is a window without a partition, which Spark runs in a
    single task.
    """
    if end_period is not None:
        return frame.with_columns(nw.lit(end_period, dtype=nw.Int64).alias("_end"))
    study_end = transactions.select(nw.col("_period").max().alias("_end"))
    return frame.join(study_end, how="cross")


def _customer_periods(
    frame: nw.LazyFrame[Any],
    customer_id_col: str,
    monetary_value_col: str | None,
    include_first_transaction: bool,
) -> nw.LazyFrame[Any]:
    """Aggregate prepared transactions into one row per customer.

    The row holds the customer's first and last period, the number of periods with
    a purchase (purchases inside one period count as a single transaction) and,
    with ``monetary_value_col``, the spend: all of it, or only what follows the
    first period when ``include_first_transaction`` is False.
    """
    aggs = [
        nw.col("_period").min().alias("_first"),
        nw.col("_period").max().alias("_last"),
        nw.col("_period").n_unique().alias("_n_periods"),
    ]
    if monetary_value_col:
        spend = nw.col(monetary_value_col)
        if not include_first_transaction:
            # A window function isolates the repeat spend, so that neither a join
            # nor a subtraction is needed.
            first_period = nw.col("_period").min().over(customer_id_col)
            frame = frame.with_columns(
                nw.when(nw.col("_period") != first_period)
                .then(spend)
                .otherwise(nw.lit(0.0))
                .alias("_repeat_monetary")
            )
            spend = nw.col("_repeat_monetary")
        aggs.append(spend.sum().alias("_monetary_sum"))
    return frame.group_by(customer_id_col).agg(*aggs)


def _summary_columns(
    monetary_value: bool, include_first_transaction: bool, spec: TimeSpec
) -> list[nw.Expr]:
    """Build the RFM columns from the columns of :func:`_customer_periods` and ``_end``.

    For modeling, *frequency* counts repeat periods, *recency* runs from the first
    to the last purchase and *monetary_value* is the mean repeat spend. With
    ``include_first_transaction``, the segmentation definition applies: every
    period counts, *recency* is the time since the last purchase and there is no *T*.
    """
    n_periods = nw.col("_n_periods").cast(nw.Float64)
    recency = spec.elapsed((nw.col("_last") - nw.col("_first")).cast(nw.Float64))
    t_value = spec.elapsed((nw.col("_end") - nw.col("_first")).cast(nw.Float64))
    monetary_sum = nw.col("_monetary_sum")

    if include_first_transaction:
        columns = [
            n_periods.alias(Col.FREQUENCY),
            (t_value - recency).alias(Col.RECENCY),
        ]
        monetary = monetary_sum / n_periods
    else:
        columns = [
            (n_periods - 1).alias(Col.FREQUENCY),
            recency.alias(Col.RECENCY),
            t_value.alias(Col.T),
        ]
        monetary = (
            nw.when(nw.col("_n_periods") > 1)
            .then(monetary_sum / (n_periods - 1))
            .otherwise(nw.lit(0.0))
        )
    if monetary_value:
        columns.append(monetary.cast(nw.Float64).alias(Col.MONETARY_VALUE))
    return columns


def _rfm_summary_frame(
    frame: nw.LazyFrame[Any],
    customer_id_col: str,
    monetary_value_col: str | None,
    end_period: int | None,
    include_first_transaction: bool,
    spec: TimeSpec,
) -> nw.LazyFrame[Any]:
    """Summarise prepared transactions into one row per customer.

    ``frame`` comes from ``_prepare_transactions``. ``end_period`` is the period the
    study ends in; ``None`` means the last period that holds a transaction.
    """
    if end_period is not None:
        frame = frame.filter(nw.col("_period") <= end_period)
    customers = _customer_periods(
        frame, customer_id_col, monetary_value_col, include_first_transaction
    )
    return _with_study_end(customers, frame, end_period).select(
        nw.col(customer_id_col).alias(Col.CUSTOMER_ID),
        *_summary_columns(bool(monetary_value_col), include_first_transaction, spec),
    )


def _test_period_summary(
    test_rows: nw.LazyFrame[Any], customer_id_col: str, monetary_value_col: str | None
) -> nw.LazyFrame[Any]:
    """Summarise the transactions of the test period into one row per customer.

    *test_frequency* counts the periods with a purchase and *test_monetary_value*
    is the mean spend per such period.
    """
    aggs = [nw.col("_period").n_unique().alias(Col.TEST_FREQUENCY)]
    if monetary_value_col:
        aggs.append(nw.col(monetary_value_col).sum().alias(Col.TEST_MONETARY_VALUE))
    summary = test_rows.group_by(customer_id_col).agg(*aggs)
    if monetary_value_col:
        summary = summary.with_columns(
            nw.col(Col.TEST_MONETARY_VALUE) / nw.col(Col.TEST_FREQUENCY)
        )
    return summary.rename({customer_id_col: Col.CUSTOMER_ID})


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
) -> IntoFrameT:
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
    more. The result is returned in the same library as the input, and a lazy input
    gives a lazy result.

    Adapted from the legacy ``lifetimes`` library:
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L230

    Parameters
    ----------
    transactions : DataFrame
        A dataframe containing *customer_id_col* and *datetime_col*. *datetime_col*
        must have a datetime or date dtype; cast strings and integers first, e.g.
        with ``pandas.to_datetime``. Rows with a missing customer id or datetime are
        dropped, and a missing monetary value adds nothing to the spend. ``NaN``
        counts as missing on every library.
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

    Returns
    -------
    DataFrame
        Dataframe containing summarized RFM data, sorted by ``customer_id`` (ids of
        mixed types, which have no order, are left as they come).

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

    Whole weeks starting on Sunday, from a lazy query to a pandas DataFrame:

    .. code-block:: python

        rfm = (
            rfm_summary(
                transactions.lazy(),
                "customer_id",
                "date",
                time_unit="week",
                week_start="sunday",
            )
            .collect()
            .to_pandas()
        )

    Monthly purchase periods with *recency* and *T* in days:

    .. code-block:: python

        rfm = rfm_summary(
            transactions, "customer_id", "date", granularity="1mo", time_unit="day"
        )

    """
    spec = _resolve_time_spec(granularity, time_unit, week_start)
    frame, time_zone = _prepare_transactions(
        transactions, customer_id_col, datetime_col, monetary_value_col, spec
    )
    end_period = _end_period(
        observation_period_end, time_zone, "observation_period_end", spec
    )
    summary = _rfm_summary_frame(
        frame,
        customer_id_col,
        monetary_value_col,
        end_period,
        include_first_transaction,
        spec,
    )
    return _finalize(summary, transactions)


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
) -> IntoFrameT:
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
    spec = _resolve_time_spec(granularity, time_unit, week_start)
    frame, time_zone = _prepare_transactions(
        transactions, customer_id_col, datetime_col, monetary_value_col, spec
    )

    train_end = _coerce_period_end(train_period_end, time_zone, "train_period_end")
    train_end_period = spec.period_of(train_end)
    train_rows = frame.filter(nw.col(datetime_col) <= nw.lit(train_end))
    test_rows = frame.filter(nw.col(datetime_col) > nw.lit(train_end))
    test_end_period = None
    if test_period_end is not None:
        test_end = _coerce_period_end(test_period_end, time_zone, "test_period_end")
        test_end_period = spec.period_of(test_end)
        test_rows = test_rows.filter(nw.col(datetime_col) <= nw.lit(test_end))

    # One row per period is enough to tell; a lazy input runs two small queries.
    for rows in (train_rows, test_rows):
        if rows.head(1).collect().is_empty():
            raise ValueError(
                "No data available. Check `train_period_end` and `test_period_end` "
                "and confirm `transactions` holds purchases on both sides of the "
                "split."
            )

    training = _rfm_summary_frame(
        train_rows,
        customer_id_col,
        monetary_value_col,
        train_end_period,
        include_first_transaction,
        spec,
    )
    test = _test_period_summary(test_rows, customer_id_col, monetary_value_col)
    test_columns = test.collect_schema().names()[1:]
    result = training.join(test, on=Col.CUSTOMER_ID, how="left").with_columns(
        # Customers without a purchase in the test period.
        *[nw.col(name).fill_null(0.0).cast(nw.Float64) for name in test_columns]
    )

    # Without ``test_period_end`` the study ends at the last transaction of the
    # whole log, whoever made it: customers who first buy in the test window count
    # here even though they have no row in the result.
    test_days = (nw.col("_end") - train_end_period).cast(nw.Float64)
    result = _with_study_end(result, test_rows, test_end_period).with_columns(
        spec.elapsed(test_days).alias(Col.TEST_T)
    )

    columns = [*training.collect_schema().names(), *test_columns, Col.TEST_T]
    return _finalize(result.select(columns), transactions)


def _bin_edges(edges: Sequence[float]) -> list[float]:
    """Turn the five quartile edges of a metric into the bounds of its bins.

    Like ``pandas.qcut(duplicates="drop")``, equal edges count once, so a metric
    with many ties gets fewer than four bins. The result has one bound more than
    there are bins, lowest first.

    Parallel engines return float sums that differ in their last bits from one
    evaluation to the next, and the edges and the binned values come from two
    evaluations. Each bound is therefore raised by a slack, so that a customer
    sitting on an edge stays in the same bin, and edges closer than the slack count
    as one, so that the number of bins stays the same. The slack follows the largest
    edge: the error of a float sum follows the size of the values added, not the size
    of the result, so an edge at zero made by refunds cancelling spend needs as much
    slack as any other.
    """
    slack = _EDGE_TOLERANCE * max(abs(edge) for edge in edges)
    distinct: list[float] = []
    for edge in sorted(edges):
        if not distinct or edge - distinct[-1] > slack:
            distinct.append(edge)
    return [edge + slack for edge in distinct]


def _quartile_score(score_name: str, metric: str, bounds: list[float]) -> nw.Expr:
    """Label each customer with the bin its ``metric`` falls in, as a string digit.

    Bins are closed on the right, and the first one includes the lowest bound.
    """
    n_bins = len(bounds) - 1
    if n_bins < 2:
        raise ValueError("Bin labels must be one fewer than the number of bin edges")
    if n_bins == 2:
        warnings.warn(
            f"RFM score will not exceed 2 for {score_name}. Specify a custom segment_config",
            UserWarning,
            stacklevel=3,
        )
    labels = [str(label) for label in range(1, n_bins + 1)]
    if score_name == Col.R_QUARTILE:
        # A low recency is good, so its labels run backwards.
        labels.reverse()
    score: Any = nw.when(nw.col(metric) <= bounds[1]).then(nw.lit(labels[0]))
    for bound, label in zip(bounds[2:-1], labels[1:-1], strict=True):
        score = score.when(nw.col(metric) <= bound).then(nw.lit(label))
    return score.otherwise(nw.lit(labels[-1])).alias(score_name)


def _quartile_edges(summary: nw.LazyFrame[Any]) -> dict[str, list[float]]:
    """Collect the five quartile edges of each scored metric, lowest first.

    The edges have to be materialised: how many of them are distinct decides how
    many score labels exist. The outer two are a plain min and max because Spark
    computes every quantile exactly, which is far more work.
    """
    metrics = [metric for _, metric in _SCORED_METRICS]
    row = (
        summary.select(
            *[nw.col(metric).min().alias(f"{metric}_0") for metric in metrics],
            *[
                nw.col(metric)
                .quantile(probability, interpolation="linear")
                .alias(f"{metric}_{position}")
                for metric in metrics
                for position, probability in enumerate((0.25, 0.5, 0.75), start=1)
            ],
            *[nw.col(metric).max().alias(f"{metric}_4") for metric in metrics],
        )
        .collect()
        .rows(named=True)[0]
    )
    # An empty summary has no edges: None, or NaN on pandas.
    if any(edge is None or edge != edge for edge in row.values()):
        raise ValueError("No transactions to segment. Check `transactions`.")
    return {
        metric: [float(row[f"{metric}_{position}"]) for position in range(5)]
        for metric in metrics
    }


def _segment_names(segment_config: Mapping[str, Sequence[str]]) -> nw.Expr:
    """Map each RFM score to the name of its segment, or to "Other".

    Later segments win when a score is listed twice.
    """
    segment: Any = nw.lit("Other")
    for name, scores in segment_config.items():
        segment = (
            nw.when(nw.col(Col.RFM_SCORE).is_in(list(scores)))
            .then(nw.lit(name))
            .otherwise(segment)
        )
    return segment.alias(Col.SEGMENT)


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
) -> IntoFrameT:
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
    spec = _resolve_time_spec(granularity, time_unit, week_start)
    frame, time_zone = _prepare_transactions(
        transactions, customer_id_col, datetime_col, monetary_value_col, spec
    )
    end_period = _end_period(
        observation_period_end, time_zone, "observation_period_end", spec
    )
    summary = _rfm_summary_frame(
        frame, customer_id_col, monetary_value_col, end_period, True, spec
    )
    # An in-memory input is summarised only once. A lazy input stays lazy, as its
    # summary may not fit in memory, and is evaluated for the edges and again for
    # the result.
    if isinstance(nw.from_native(transactions), nw.DataFrame):
        summary = summary.collect().lazy()

    edges = _quartile_edges(summary)
    scores = [
        _quartile_score(score_name, metric, _bin_edges(edges[metric]))
        for score_name, metric in _SCORED_METRICS
    ]
    result = (
        summary.with_columns(*scores)
        .with_columns(
            nw.concat_str(*[nw.col(name) for name, _ in _SCORED_METRICS]).alias(
                Col.RFM_SCORE
            )
        )
        .with_columns(_segment_names(segment_config or _default_rfm_segment_config))
        .select(
            Col.CUSTOMER_ID,
            Col.FREQUENCY,
            Col.RECENCY,
            Col.MONETARY_VALUE,
            Col.RFM_SCORE,
            Col.SEGMENT,
        )
    )
    return _finalize(result, transactions)
