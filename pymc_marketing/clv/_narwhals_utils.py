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
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from typing import Any, Literal, get_args
from zoneinfo import ZoneInfo

import narwhals.stable.v2 as nw
import numpy as np
import pandas
from narwhals.stable.v2.typing import FrameT

Backend = Literal["pandas", "polars", "pyarrow"]

# Fixed divisors turning a duration in days into the named ``time_unit``.
TIME_UNIT_DAYS: dict[str, float] = {
    "day": 1.0,
    "week": 7.0,
    "month": 30.4375,  # 365.25 / 12
    "quarter": 91.3125,  # 365.25 / 4
    "year": 365.25,
}

# Granularity unit a named ``time_unit`` implies when ``granularity`` is not given.
TIME_UNIT_GRANULARITY: dict[str, str] = {
    "day": "d",
    "week": "w",
    "month": "mo",
    "quarter": "q",
    "year": "y",
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
        "_test_end",
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

_GRANULARITY_RE = re.compile(r"^(\d+)(d|w|mo|q|y)$")

# ``date.toordinal()`` counts days from 0001-01-01 (= 1); 1970-01-01 is day 719163.
_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
# Days from 0000-03-01 to 1970-01-01 in the civil-date algorithm below.
_CIVIL_EPOCH_SHIFT = 719468
# 1970-01-01 is a Thursday, so the Monday-anchored week containing it starts three
# days earlier. ``_WEEK_SHIFT`` is ``719467 + 3``: a multiple of seven plus that
# offset, large enough to keep floor division on non-negative operands for every
# date of the Common Era (some backends truncate negative integer division).
_WEEK_SHIFT = 719470


def _parse_granularity(granularity: str) -> tuple[int, str]:
    """Split ``"1d"``, ``"1w"``, ``"1mo"``, ``"1q"`` or ``"1y"`` into its parts.

    Multiples larger than one are rejected for now: the anchor of a multi-period
    bucket differs between backends, so results would not be comparable.
    """
    if not isinstance(granularity, str):
        raise TypeError(
            f"granularity must be a string such as '1d' or '1w', got {granularity!r}."
        )
    match = _GRANULARITY_RE.match(granularity)
    if match is None:
        raise ValueError(
            f"Unsupported granularity {granularity!r}. Expected one of "
            "'1d', '1w', '1mo', '1q' or '1y'."
        )
    multiple, unit = int(match.group(1)), match.group(2)
    if multiple != 1:
        raise ValueError(
            f"granularity {granularity!r} is not supported yet: only single periods "
            "('1d', '1w', '1mo', '1q', '1y') are accepted, because multi-period "
            "buckets are anchored differently by each dataframe backend."
        )
    return multiple, unit


def _resolve_time_spec(
    granularity: str | None,
    time_unit: str | None,
    week_start: str | None,
) -> tuple[str, str, int]:
    """Validate the new-path time parameters and fill in their defaults.

    Returns ``(unit, time_unit, week_start_offset)`` where ``unit`` is the
    granularity unit (``"d"``, ``"w"``, ``"mo"``, ``"q"`` or ``"y"``) and
    ``week_start_offset`` counts days from Monday. Without ``granularity`` the
    bucket follows ``time_unit``, so ``time_unit="week"`` alone counts whole weeks.
    """
    time_unit = "day" if time_unit is None else time_unit
    if time_unit not in TIME_UNIT_DAYS:
        raise ValueError(
            f"Unsupported time_unit {time_unit!r}. Expected one of "
            f"{sorted(TIME_UNIT_DAYS)}."
        )
    if granularity is None:
        unit = TIME_UNIT_GRANULARITY[time_unit]
    else:
        _, unit = _parse_granularity(granularity)

    if week_start is None:
        offset = 0
    else:
        if unit != "w":
            raise ValueError(
                "week_start only applies to weekly granularity; pass "
                "granularity='1w' or drop week_start."
            )
        if not isinstance(week_start, str) or week_start.lower() not in WEEKDAYS:
            raise ValueError(
                f"Unsupported week_start {week_start!r}. Expected one of {WEEKDAYS}."
            )
        offset = WEEKDAYS.index(week_start.lower())

    return unit, time_unit, offset


def _elapsed(days: Any, unit: str, time_unit: str, extra_divisor: float = 1.0) -> Any:
    """Express a day count between two period starts in ``time_unit``.

    ``days`` is a Narwhals expression or a plain number. When ``time_unit`` names the
    granularity itself the result is a whole number of periods, which discrete-time
    models such as BG/BB need.
    """
    elapsed = days / TIME_UNIT_DAYS[time_unit]
    if unit in ("mo", "q", "y") and TIME_UNIT_GRANULARITY[time_unit] == unit:
        # Starts of n calendar periods are within a few days of n times the mean
        # period length, far less than half a period, so rounding is exact.
        elapsed = (
            elapsed.round() if isinstance(elapsed, nw.Expr) else float(round(elapsed))
        )
    return elapsed / extra_divisor


def _days_from_civil(year: Any, month: Any, day: Any) -> Any:
    """Days since 1970-01-01 of a proleptic Gregorian date.

    Works on Narwhals expressions and on plain integers alike, using only
    multiplication, floor division and modulo on non-negative operands (Howard
    Hinnant's ``days_from_civil`` algorithm).
    """
    if any(isinstance(part, nw.Expr) for part in (year, month, day)):
        # Scalars become literals so that every comparison below is an expression.
        year, month, day = (
            part.cast(nw.Int64)
            if isinstance(part, nw.Expr)
            else nw.lit(part, dtype=nw.Int64)
            for part in (year, month, day)
        )
        shifted_year = nw.when(month <= 2).then(year - 1).otherwise(year)
    else:
        shifted_year = year - 1 if month <= 2 else year

    era = shifted_year // 400
    year_of_era = shifted_year - era * 400
    month_index = (month + 9) % 12  # March -> 0 ... February -> 11
    day_of_year = (153 * month_index + 2) // 5 + day - 1
    day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
    return era * 146097 + day_of_era - _CIVIL_EPOCH_SHIFT


def _day_number(expr: nw.Expr) -> nw.Expr:
    """Integer days since 1970-01-01 of a datetime (or date) expression."""
    return _days_from_civil(expr.dt.year(), expr.dt.month(), expr.dt.day())


def _week_start(day_number: Any, week_start_offset: int) -> Any:
    """First day of the week containing ``day_number``.

    Weeks start on Monday plus ``week_start_offset`` days.
    """
    shift = _WEEK_SHIFT - week_start_offset
    return ((day_number + shift) // 7) * 7 - shift


def _period_start(expr: nw.Expr, unit: str, week_start_offset: int = 0) -> nw.Expr:
    """Day number of the first day of the period containing ``expr``."""
    year, month = expr.dt.year(), expr.dt.month()
    if unit == "d":
        return _day_number(expr)
    if unit == "w":
        return _week_start(_day_number(expr), week_start_offset)
    if unit == "mo":
        return _days_from_civil(year, month, 1)
    if unit == "q":
        quarter_month = ((month.cast(nw.Int64) - 1) // 3) * 3 + 1
        return _days_from_civil(year, quarter_month, 1)
    if unit == "y":
        return _days_from_civil(year, 1, 1)
    raise ValueError(f"Unknown granularity unit {unit!r}.")  # pragma: no cover


def _period_start_scalar(value: datetime, unit: str, week_start_offset: int = 0) -> int:
    """Python-side twin of :func:`_period_start` for a single naive datetime."""
    day_number = value.date().toordinal() - _EPOCH_ORDINAL
    if unit == "d":
        return day_number
    if unit == "w":
        return _week_start(day_number, week_start_offset)
    if unit == "mo":
        return date(value.year, value.month, 1).toordinal() - _EPOCH_ORDINAL
    if unit == "q":
        quarter_month = ((value.month - 1) // 3) * 3 + 1
        return date(value.year, quarter_month, 1).toordinal() - _EPOCH_ORDINAL
    if unit == "y":
        return date(value.year, 1, 1).toordinal() - _EPOCH_ORDINAL
    raise ValueError(f"Unknown granularity unit {unit!r}.")  # pragma: no cover


def _validate_backend(backend: str | None) -> None:
    """Reject a ``backend`` the result cannot be collected into."""
    if backend is not None and backend not in get_args(Backend):
        raise ValueError(
            f"Unsupported backend {backend!r}. Expected one of 'pandas', 'polars', "
            "'pyarrow' or None."
        )


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


def _prepare_datetime_column(
    frame: FrameT, datetime_col: str, dtype: nw.dtypes.DType
) -> tuple[FrameT, str | None]:
    """Normalise ``datetime_col`` to a naive datetime, returning its original time zone.

    A time-zone-aware column keeps its wall-clock time: periods are calendar
    buckets in the column's own zone. For DuckDB ``TIMESTAMPTZ`` columns that zone
    is the connection's session ``TimeZone``.
    """
    time_zone = getattr(dtype, "time_zone", None)
    if isinstance(dtype, nw.Date):
        frame = frame.with_columns(nw.col(datetime_col).cast(nw.Datetime))
    elif time_zone is not None:
        frame = frame.with_columns(nw.col(datetime_col).dt.replace_time_zone(None))
    return frame, time_zone


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
    it lines up with the wall-clock buckets of :func:`_prepare_datetime_column`.
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


def _finalize(
    frame: nw.LazyFrame[Any],
    backend: Backend | None,
    *,
    eager_input: bool,
) -> Any:
    """Return ``frame`` as a native object.

    ``backend=None`` keeps the input's own library and laziness. Otherwise the result
    is collected into the requested eager library. A pandas result gets a fresh
    ``RangeIndex``, like the legacy path.
    """
    result: nw.DataFrame[Any] | nw.LazyFrame[Any]
    if backend is not None:
        result = frame.collect(backend=backend)
    else:
        result = frame.collect() if eager_input else frame
    native = result.to_native()
    if isinstance(native, pandas.DataFrame):
        native = native.reset_index(drop=True)
    return native
