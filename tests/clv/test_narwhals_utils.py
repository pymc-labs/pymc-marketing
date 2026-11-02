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
"""Unit tests for the private Narwhals helpers behind the CLV utilities."""

from datetime import UTC, date, datetime, timedelta, timezone

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest

from pymc_marketing.clv._narwhals_utils import (
    TIME_UNIT_DAYS,
    WEEKDAYS,
    _coerce_period_end,
    _day_number,
    _days_from_civil,
    _finalize,
    _parse_granularity,
    _period_start,
    _period_start_scalar,
    _prepare_datetime_column,
    _resolve_time_spec,
    _validate_input,
)
from tests.clv.conftest import collect_sorted

EPOCH = date(1970, 1, 1)


@pytest.mark.parametrize(
    "granularity, expected",
    [
        ("1d", (1, "d")),
        ("1w", (1, "w")),
        ("1mo", (1, "mo")),
        ("1q", (1, "q")),
        ("1y", (1, "y")),
    ],
)
def test_parse_granularity_accepts_single_periods(granularity, expected):
    assert _parse_granularity(granularity) == expected


@pytest.mark.parametrize("granularity", ["2w", "3d", "12mo"])
def test_parse_granularity_rejects_multiples(granularity):
    with pytest.raises(ValueError, match="only single periods"):
        _parse_granularity(granularity)


@pytest.mark.parametrize("granularity", ["1h", "week", "1W", "", "1"])
def test_parse_granularity_rejects_unknown_units(granularity):
    with pytest.raises(ValueError, match="Unsupported granularity"):
        _parse_granularity(granularity)


def test_parse_granularity_rejects_non_strings():
    with pytest.raises(TypeError, match="granularity must be a string"):
        _parse_granularity(1)


def test_resolve_time_spec_defaults():
    assert _resolve_time_spec(None, None, None) == ("d", "day", 0)


def test_resolve_time_spec_week_start_offsets():
    for offset, weekday in enumerate(WEEKDAYS):
        assert _resolve_time_spec("1w", "week", weekday) == ("w", "week", offset)
    assert _resolve_time_spec("1w", "week", "Wednesday") == ("w", "week", 2)


def test_resolve_time_spec_week_start_requires_weekly_granularity():
    with pytest.raises(ValueError, match="week_start only applies to weekly"):
        _resolve_time_spec("1d", "day", "monday")


def test_resolve_time_spec_rejects_unknown_week_start():
    with pytest.raises(ValueError, match="Unsupported week_start"):
        _resolve_time_spec("1w", "week", "someday")


def test_resolve_time_spec_rejects_unknown_time_unit():
    with pytest.raises(ValueError, match="Unsupported time_unit"):
        _resolve_time_spec("1d", "D", None)


def test_time_unit_days_constants():
    assert TIME_UNIT_DAYS == {
        "day": 1.0,
        "week": 7.0,
        "month": 30.4375,
        "quarter": 91.3125,
        "year": 365.25,
    }


@pytest.mark.parametrize(
    "value",
    [
        date(1, 1, 1),
        date(1600, 2, 29),
        date(1899, 12, 31),
        date(1969, 12, 31),
        date(1970, 1, 1),
        date(2000, 2, 29),
        date(2015, 2, 7),
        date(2100, 3, 1),
        date(9999, 12, 31),
    ],
)
def test_days_from_civil_matches_python_ordinal(value):
    expected = (value - EPOCH).days
    assert _days_from_civil(value.year, value.month, value.day) == expected


def _day_range() -> pd.DataFrame:
    """Every day of a span covering leap years and both century rules."""
    days = pd.date_range("1896-01-01", "2104-12-31", freq="D")
    return pd.DataFrame({"ts": days + pd.Timedelta(hours=13, minutes=37)})


def test_day_number_matches_python_on_every_backend(to_backend):
    frame = _day_range()
    native = to_backend(frame)
    result = collect_sorted(
        nw.from_native(native).lazy().select(_day_number(nw.col("ts")).alias("n")),
        by="n",
    )
    expected = np.sort(
        np.array([(ts.date() - EPOCH).days for ts in frame["ts"]], dtype="int64")
    )
    np.testing.assert_array_equal(result["n"].to_numpy(), expected)


@pytest.mark.parametrize("unit", ["d", "w", "mo", "q", "y"])
@pytest.mark.parametrize("week_start_offset", [0, 2, 6])
def test_period_start_expression_matches_scalar_twin(
    to_backend, unit, week_start_offset
):
    if unit != "w" and week_start_offset:
        pytest.skip("week_start only matters for weekly granularity")
    frame = _day_range()
    native = to_backend(frame)
    result = collect_sorted(
        nw.from_native(native)
        .lazy()
        .select(
            _day_number(nw.col("ts")).alias("n"),
            _period_start(nw.col("ts"), unit, week_start_offset).alias("p"),
        ),
        by="n",
    )
    expected = np.array(
        [
            _period_start_scalar(ts.to_pydatetime(), unit, week_start_offset)
            for ts in frame["ts"]
        ],
        dtype="int64",
    )
    np.testing.assert_array_equal(result["p"].to_numpy(), expected)


def test_period_start_scalar_weekly_anchors():
    # 2015-01-01 is a Thursday. Monday-start weeks begin on 2014-12-29, which is
    # the same bucket pandas assigns with the "W" (W-SUN) period alias.
    thursday = datetime(2015, 1, 1)
    monday = (date(2014, 12, 29) - EPOCH).days
    assert _period_start_scalar(thursday, "w", 0) == monday
    # Wednesday-start weeks containing that Thursday begin on 2014-12-31.
    assert _period_start_scalar(thursday, "w", 2) == monday + 2
    # A Monday belongs to the Wednesday-start week that began the previous Wednesday.
    assert _period_start_scalar(datetime(2014, 12, 29), "w", 2) == monday - 5
    # Sunday-start weeks: Sunday 2014-12-28 opens the week holding 2015-01-01.
    assert _period_start_scalar(thursday, "w", 6) == monday - 1


def test_period_start_scalar_calendar_units():
    value = datetime(2015, 8, 17, 23, 59)
    assert _period_start_scalar(value, "d") == (date(2015, 8, 17) - EPOCH).days
    assert _period_start_scalar(value, "mo") == (date(2015, 8, 1) - EPOCH).days
    assert _period_start_scalar(value, "q") == (date(2015, 7, 1) - EPOCH).days
    assert _period_start_scalar(value, "y") == (date(2015, 1, 1) - EPOCH).days


def test_period_start_weekly_matches_pandas_w_sun(to_backend):
    frame = _day_range()
    result = collect_sorted(
        nw.from_native(to_backend(frame))
        .lazy()
        .select(
            _day_number(nw.col("ts")).alias("n"),
            _period_start(nw.col("ts"), "w", 0).alias("p"),
        ),
        by="n",
    )
    pandas_week_start = frame["ts"].dt.to_period("W").dt.start_time
    expected = ((pandas_week_start - pd.Timestamp(EPOCH)).dt.days).to_numpy()
    np.testing.assert_array_equal(result["p"].to_numpy(), expected)


def test_validate_input_missing_column():
    frame = nw.from_native(
        pd.DataFrame({"id": [1], "date": pd.to_datetime(["2020-01-01"])})
    )
    with pytest.raises(KeyError, match=r"\['spend'\] not found"):
        _validate_input(frame, ["id", "date", "spend"], "date")


def test_validate_input_string_datetime_column_has_cast_hint(to_backend):
    native = to_backend(pd.DataFrame({"id": [1], "date": ["2020-01-01"]}))
    frame = nw.from_native(native)
    with pytest.raises(TypeError, match=r"pd\.to_datetime"):
        _validate_input(frame, ["id", "date"], "date")


def test_validate_input_accepts_datetime_and_date(to_backend):
    native = to_backend(
        pd.DataFrame(
            {
                "id": [1],
                "dt": pd.to_datetime(["2020-01-01"]),
                "d": pd.to_datetime(["2020-01-01"]).date,
            }
        )
    )
    frame = nw.from_native(native)
    schema = _validate_input(frame, ["id", "dt"], "dt")
    assert isinstance(schema["dt"], nw.Datetime)
    if frame.collect_schema()["d"] == nw.Date:
        _validate_input(frame, ["id", "d"], "d")


def test_prepare_datetime_column_strips_time_zone():
    frame = nw.from_native(
        pd.DataFrame(
            {"ts": pd.to_datetime(["2020-01-01 23:00"]).tz_localize("Asia/Tokyo")}
        )
    ).lazy()
    dtype = frame.collect_schema()["ts"]
    prepared, time_zone = _prepare_datetime_column(frame, "ts", dtype)
    assert time_zone == "Asia/Tokyo"
    out = prepared.collect().to_native()
    assert out["ts"].dt.tz is None
    # Wall-clock time is kept, so the calendar bucket is the Tokyo day.
    assert out["ts"].iloc[0] == pd.Timestamp("2020-01-01 23:00")


def test_prepare_datetime_column_casts_dates():
    frame = nw.from_native(pd.DataFrame({"d": [date(2020, 1, 1)]})).lazy()
    frame = frame.with_columns(nw.col("d").cast(nw.Date))
    dtype = frame.collect_schema()["d"]
    prepared, time_zone = _prepare_datetime_column(frame, "d", dtype)
    assert time_zone is None
    assert isinstance(prepared.collect_schema()["d"], nw.Datetime)


@pytest.mark.parametrize(
    "value",
    [
        "2015-02-07",
        "2015-02-07T10:30:00",
        date(2015, 2, 7),
        datetime(2015, 2, 7),
        pd.Timestamp("2015-02-07"),
        pd.Period("2015-02-07", "D"),
    ],
)
def test_coerce_period_end_naive_inputs(value):
    result = _coerce_period_end(value, None, "observation_period_end")
    assert result.tzinfo is None
    assert result.date() == date(2015, 2, 7)


def test_coerce_period_end_converts_aware_values_to_column_zone():
    value = datetime(2015, 2, 7, 23, 0, tzinfo=UTC)
    result = _coerce_period_end(value, "Asia/Tokyo", "observation_period_end")
    assert result == datetime(2015, 2, 8, 8, 0)


def test_coerce_period_end_utc_column():
    value = datetime(2015, 2, 7, 1, 0, tzinfo=timezone(timedelta(hours=2)))
    assert _coerce_period_end(value, "UTC", "x") == datetime(2015, 2, 6, 23, 0)


def test_coerce_period_end_time_zone_mismatch_raises():
    with pytest.raises(ValueError, match="aware but the datetime column is naive"):
        _coerce_period_end(datetime(2015, 2, 7, tzinfo=UTC), None, "x")
    with pytest.raises(
        ValueError, match="naive but the datetime column is time-zone aware"
    ):
        _coerce_period_end(datetime(2015, 2, 7), "UTC", "x")


def test_coerce_period_end_rejects_bad_values():
    with pytest.raises(ValueError, match="ISO-8601"):
        _coerce_period_end("02/07/2015", None, "x")
    with pytest.raises(TypeError, match="must be a str, date, datetime"):
        _coerce_period_end(42, None, "x")


def test_finalize_keeps_native_type(to_backend):
    native = to_backend(pd.DataFrame({"a": [1, 2]}))
    frame = nw.from_native(native)
    eager_input = isinstance(frame, nw.DataFrame)
    out = _finalize(frame.lazy(), None, eager_input=eager_input)
    assert type(out) is type(native)


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_finalize_collects_to_requested_backend(to_backend, backend):
    pl = pytest.importorskip("polars")
    pa = pytest.importorskip("pyarrow")
    expected_type = {
        "pandas": pd.DataFrame,
        "polars": pl.DataFrame,
        "pyarrow": pa.Table,
    }[backend]
    native = to_backend(pd.DataFrame({"a": [1, 2]}))
    frame = nw.from_native(native)
    out = _finalize(frame.lazy(), backend, eager_input=isinstance(frame, nw.DataFrame))
    assert isinstance(out, expected_type)


def test_finalize_rejects_unknown_backend():
    frame = nw.from_native(pd.DataFrame({"a": [1]})).lazy()
    with pytest.raises(ValueError, match="Unsupported backend"):
        _finalize(frame, "dask", eager_input=True)
