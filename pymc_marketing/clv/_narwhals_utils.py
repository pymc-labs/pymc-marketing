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
"""Private Narwhals helpers shared by the dataframe-agnostic CLV utilities.

Every time computation here works on *integer day numbers* (days since 1970-01-01)
derived from ``dt.year()``, ``dt.month()`` and ``dt.day()`` with integer arithmetic.
That keeps the whole pipeline portable: not every lazy backend supports
``dt.timestamp``, ``Duration`` arithmetic or weekly ``dt.truncate``, but all of them
expose the calendar fields, and integer subtraction is exact everywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo

import narwhals.stable.v2 as nw
import numpy as np
import pandas

from pymc_marketing.names.clv import Col

# Mean length in days of each time unit.
TIME_UNIT_DAYS: dict[str, float] = {
    "day": 1.0,
    "week": 7.0,
    "month": 30.4375,  # 365.25 / 12
    "quarter": 91.3125,  # 365.25 / 4
    "year": 365.25,
}
# The period each ``granularity`` buckets by. Multiples such as "2w" are left out:
# each dataframe backend anchors a multi-period bucket differently.
GRANULARITY_UNIT: dict[str, str] = {
    "1d": "day",
    "1w": "week",
    "1mo": "month",
    "1q": "quarter",
    "1y": "year",
}
# Names of the temporary columns the RFM pipeline adds to the transactions.
RESERVED_COLUMNS = frozenset(
    {
        "_period",
        "_end",
        "_first",
        "_last",
        "_n_periods",
        "_monetary_sum",
        "_repeat_monetary",
    }
)
WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

# Days from 0000-03-01 to 1970-01-01 in the civil-date algorithm below.
_CIVIL_EPOCH_SHIFT = 719468
# 1970-01-01 is a Thursday, so the Monday-anchored week containing it starts three
# days earlier. ``_WEEK_SHIFT`` is ``719467 + 3``: a multiple of seven plus that
# offset, large enough to keep floor division on non-negative operands for every
# date of the Common Era (some backends truncate negative integer division).
_WEEK_SHIFT = 719470


def _days_from_civil(year: Any, month: Any, day: Any) -> Any:
    """Days since 1970-01-01 of a proleptic Gregorian date.

    Works on Narwhals expressions and on plain integers alike, using only
    multiplication and floor division on non-negative operands (Howard Hinnant's
    ``days_from_civil`` algorithm).
    """
    if any(isinstance(part, nw.Expr) for part in (year, month, day)):
        # Scalars become literals so that every comparison below is an expression.
        year, month, day = (
            part.cast(nw.Int64)
            if isinstance(part, nw.Expr)
            else nw.lit(part, dtype=nw.Int64)
            for part in (year, month, day)
        )
        # No modulo: pandas columns with a PyArrow dtype do not implement it.
        shifted_year = nw.when(month <= 2).then(year - 1).otherwise(year)
        month_index = nw.when(month <= 2).then(month + 9).otherwise(month - 3)
    else:
        shifted_year = year - 1 if month <= 2 else year
        month_index = month + 9 if month <= 2 else month - 3

    # month_index: March -> 0 ... February -> 11
    era = shifted_year // 400
    year_of_era = shifted_year - era * 400
    day_of_year = (153 * month_index + 2) // 5 + day - 1
    day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
    return era * 146097 + day_of_era - _CIVIL_EPOCH_SHIFT


@dataclass(frozen=True)
class TimeSpec:
    """How transactions are bucketed into periods and how durations are reported.

    Parameters
    ----------
    unit : str
        Period that collapses purchases into one transaction: a key of
        ``TIME_UNIT_DAYS``.
    time_unit : str
        Unit durations are reported in: a key of ``TIME_UNIT_DAYS``.
    week_start_offset : int
        First day of a weekly period, in days after Monday.
    """

    unit: str
    time_unit: str
    week_start_offset: int = 0

    def _period_start(self, year: Any, month: Any, day: Any) -> Any:
        """Day number of the first day of the period holding a calendar date."""
        if self.unit == "day":
            return _days_from_civil(year, month, day)
        if self.unit == "week":
            shift = _WEEK_SHIFT - self.week_start_offset
            return ((_days_from_civil(year, month, day) + shift) // 7) * 7 - shift
        if self.unit == "month":
            return _days_from_civil(year, month, 1)
        if self.unit == "quarter":
            return _days_from_civil(year, ((month - 1) // 3) * 3 + 1, 1)
        return _days_from_civil(year, 1, 1)

    def period(self, expr: nw.Expr) -> nw.Expr:
        """Period of each datetime in a column, as the day number of its first day."""
        return self._period_start(
            expr.dt.year().cast(nw.Int64),
            expr.dt.month().cast(nw.Int64),
            expr.dt.day().cast(nw.Int64),
        )

    def period_of(self, value: datetime) -> int:
        """Period of one datetime, as the day number of its first day."""
        return self._period_start(value.year, value.month, value.day)

    def elapsed(self, days: Any) -> Any:
        """Express a day count between two period starts in ``time_unit``.

        ``days`` is a Narwhals expression or a plain number. When ``time_unit`` is
        the period itself the result is a whole number of periods, which
        discrete-time models such as BG/BB need.
        """
        elapsed = days / TIME_UNIT_DAYS[self.time_unit]
        if self.unit == self.time_unit and self.unit in ("month", "quarter", "year"):
            # Starts of n calendar periods are within a few days of n times the mean
            # period length, far less than half a period, so rounding is exact.
            elapsed = (
                elapsed.round()
                if isinstance(elapsed, nw.Expr)
                else float(round(elapsed))
            )
        return elapsed


def _resolve_time_spec(
    granularity: str | None, time_unit: str, week_start: str | None
) -> TimeSpec:
    """Validate the time parameters of the public functions.

    Without ``granularity`` the period follows ``time_unit``, so ``time_unit="week"``
    alone counts whole weeks.
    """
    if time_unit not in TIME_UNIT_DAYS:
        raise ValueError(
            f"Unsupported time_unit {time_unit!r}. Expected one of "
            f"{sorted(TIME_UNIT_DAYS)}."
        )
    if granularity is None:
        unit = time_unit
    elif granularity in GRANULARITY_UNIT:
        unit = GRANULARITY_UNIT[granularity]
    else:
        raise ValueError(
            f"Unsupported granularity {granularity!r}. Expected one of "
            f"{sorted(GRANULARITY_UNIT)}."
        )

    offset = 0
    if week_start is not None:
        if unit != "week":
            raise ValueError(
                "week_start only applies to weekly granularity; pass "
                "granularity='1w' or drop week_start."
            )
        if not isinstance(week_start, str) or week_start.lower() not in WEEKDAYS:
            raise ValueError(
                f"Unsupported week_start {week_start!r}. Expected one of {WEEKDAYS}."
            )
        offset = WEEKDAYS.index(week_start.lower())

    return TimeSpec(unit, time_unit, offset)


def _validate_input(
    frame: nw.DataFrame[Any] | nw.LazyFrame[Any],
    required_cols: list[str],
    datetime_col: str,
) -> nw.Schema:
    """Check the required columns exist and ``datetime_col`` is typed as a date/datetime."""
    reserved = sorted(RESERVED_COLUMNS.intersection(required_cols))
    if reserved:
        raise ValueError(
            f"Column name(s) {reserved} are reserved for internal use; rename them "
            "before calling."
        )
    schema = frame.collect_schema()
    missing = [col for col in required_cols if col not in schema]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in transactions. Available columns: "
            f"{list(schema.names())}."
        )
    dtype = schema[datetime_col]
    if not isinstance(dtype, (nw.Datetime, nw.Date)):
        raise TypeError(
            f"Column {datetime_col!r} must have a datetime or date dtype, got {dtype}. "
            "Cast it first, e.g. ``pd.to_datetime(df[col], format=...)`` for pandas, "
            "``pl.col(col).str.to_datetime(format=...)`` for Polars or "
            "``pc.strptime(table[col], format=..., unit='us')`` for PyArrow."
        )
    return schema


def _prepare_transactions(
    transactions: Any,
    customer_id_col: str,
    datetime_col: str,
    monetary_value_col: str | None,
    spec: TimeSpec,
) -> tuple[nw.LazyFrame[Any], str | None]:
    """Validate the transactions and tag each one with its period.

    Returns a lazy frame of the columns in use plus ``_period``, and the time zone
    of ``datetime_col``.

    A time-zone-aware column keeps its wall-clock time: periods are calendar
    buckets in the column's own zone. For DuckDB ``TIMESTAMPTZ`` columns that zone
    is the connection's session ``TimeZone``.
    """
    frame = nw.from_native(transactions)
    columns = [customer_id_col, datetime_col]
    if monetary_value_col:
        columns.append(monetary_value_col)
    schema = _validate_input(frame, columns, datetime_col)
    # Rows without a customer or a timestamp cannot be summarised.
    frame = frame.lazy().select(columns).drop_nulls([customer_id_col, datetime_col])
    # Polars keeps NaN apart from null; it is a missing value here as well.
    if schema[customer_id_col].is_float():
        frame = frame.filter(~nw.col(customer_id_col).is_nan())
    if monetary_value_col and schema[monetary_value_col].is_float():
        frame = frame.with_columns(nw.col(monetary_value_col).fill_nan(None))

    dtype = schema[datetime_col]
    time_zone = getattr(dtype, "time_zone", None)
    if isinstance(dtype, nw.Date):
        frame = frame.with_columns(nw.col(datetime_col).cast(nw.Datetime))
    elif time_zone is not None:
        frame = frame.with_columns(nw.col(datetime_col).dt.replace_time_zone(None))
    period = spec.period(nw.col(datetime_col))
    return frame.with_columns(period.alias("_period")), time_zone


_FIXED_OFFSET_RE = re.compile(r"^(?:UTC)?([+-])(\d{2}):?(\d{2})$")


def _zone(name: str) -> tzinfo:
    """Time zone named by a column dtype: UTC, a fixed offset or an IANA key."""
    if name in ("UTC", "utc"):
        return UTC
    # pandas reports a fixed-offset zone as e.g. "UTC+02:00", which is no IANA key.
    match = _FIXED_OFFSET_RE.match(name)
    if match is None:
        return ZoneInfo(name)
    offset = timedelta(hours=int(match.group(2)), minutes=int(match.group(3)))
    return timezone(-offset if match.group(1) == "-" else offset)


def _coerce_period_end(
    value: str | date | datetime | np.datetime64 | pandas.Period | pandas.Timestamp,
    column_time_zone: str | None,
    name: str,
) -> datetime:
    """Turn a user-supplied period boundary into a naive ``datetime``.

    Accepts ``datetime``, ``date``, ISO-8601 strings, ``numpy.datetime64``,
    ``pandas.Timestamp`` and ``pandas.Period``. The time-zone awareness must match
    the column's: an aware value is converted to the column's zone and stripped, so
    it lines up with the wall-clock buckets of :func:`_prepare_transactions`.
    """
    if isinstance(value, pandas.Period):
        value = value.to_timestamp()
    if isinstance(value, np.datetime64):
        value = pandas.Timestamp(value)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be an ISO-8601 string (e.g. '2015-02-07'), a date or a "
                f"datetime, got {value!r}. Parse other formats yourself first."
            ) from exc
    if isinstance(value, pandas.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        pass
    elif isinstance(value, date):
        value = datetime(value.year, value.month, value.day)
    else:
        raise TypeError(
            f"{name} must be a str, date, datetime, numpy.datetime64, "
            f"pandas.Timestamp or pandas.Period, got {type(value).__name__}."
        )

    value_is_aware = value.tzinfo is not None
    if value_is_aware and column_time_zone is None:
        raise ValueError(
            f"{name} is time-zone aware but the datetime column is naive; pass a "
            "naive value or localise the column."
        )
    if not value_is_aware and column_time_zone is not None:
        raise ValueError(
            f"{name} is naive but the datetime column is time-zone aware "
            f"({column_time_zone}); pass a value with tzinfo."
        )
    if value_is_aware:
        value = value.astimezone(_zone(str(column_time_zone))).replace(tzinfo=None)
    return value


def _finalize(frame: nw.LazyFrame[Any], transactions: Any) -> Any:
    """Return ``frame`` sorted by customer, in the library of ``transactions``.

    An in-memory input gives an in-memory result and a lazy input stays lazy. A
    pandas result gets a fresh ``RangeIndex``.
    """
    # The sort is the last step of the query: an engine need not keep the order of
    # a subquery. Ids of mixed types (a pandas object column) cannot be ordered.
    if not isinstance(frame.collect_schema()[Col.CUSTOMER_ID], nw.Object):
        frame = frame.sort(Col.CUSTOMER_ID)
    result: nw.DataFrame[Any] | nw.LazyFrame[Any] = frame
    if isinstance(nw.from_native(transactions), nw.DataFrame):
        result = frame.collect()
    native = result.to_native()
    if isinstance(native, pandas.DataFrame):
        native = native.reset_index(drop=True)
    return native
