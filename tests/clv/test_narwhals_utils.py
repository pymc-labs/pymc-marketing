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
"""Tests for the dataframe-agnostic (Narwhals) path of the RFM utilities.

The legacy pandas behaviour is covered by ``tests/clv/test_utils.py``. This module
checks the private helpers in ``pymc_marketing.clv._narwhals_utils``, that every
supported backend reproduces the legacy results, that routing between the two paths
behaves, and that the new ``granularity`` / ``time_unit`` API is right.
"""

import warnings
from datetime import UTC, date, datetime, timedelta, timezone

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pymc_marketing.clv._narwhals_utils import (
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
    _validate_backend,
    _validate_input,
)
from pymc_marketing.clv.utils import (
    _rfm_segments_legacy,
    _rfm_segments_nw,
    _rfm_summary_legacy,
    _rfm_summary_nw,
    _rfm_train_test_split_legacy,
    _rfm_train_test_split_nw,
    rfm_segments,
    rfm_summary,
    rfm_train_test_split,
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


@pytest.mark.parametrize(
    "time_unit, unit",
    [("day", "d"), ("week", "w"), ("month", "mo"), ("quarter", "q"), ("year", "y")],
)
def test_resolve_time_spec_granularity_follows_time_unit(time_unit, unit):
    assert _resolve_time_spec(None, time_unit, None) == (unit, time_unit, 0)
    assert _resolve_time_spec("1d", time_unit, None) == ("d", time_unit, 0)


def test_resolve_time_spec_week_start_with_weekly_time_unit():
    assert _resolve_time_spec(None, "week", "sunday") == ("w", "week", 6)


def test_days_from_civil_mixes_scalars_and_expressions():
    frame = nw.from_native(pd.DataFrame({"month": [1, 2, 3, 12]}))
    actual = frame.select(_days_from_civil(2016, nw.col("month"), 1).alias("n"))
    expected = [(date(2016, m, 1) - EPOCH).days for m in [1, 2, 3, 12]]
    assert actual["n"].to_list() == expected


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


def test_finalize_resets_the_pandas_index():
    frame = nw.from_native(pd.DataFrame({"a": [2, 1]}, index=[5, 3])).lazy()
    out = _finalize(frame.sort("a"), None, eager_input=True)
    assert out.index.equals(pd.RangeIndex(2))


def test_validate_backend():
    _validate_backend(None)
    _validate_backend("polars")
    with pytest.raises(ValueError, match="Unsupported backend"):
        _validate_backend("dask")


def test_validate_input_rejects_reserved_column_names():
    frame = nw.from_native(pd.DataFrame({"id": [1], "_period": [datetime(2015, 1, 1)]}))
    with pytest.raises(ValueError, match="reserved for internal use"):
        _validate_input(frame, ["id", "_period"], "_period")


def test_coerce_period_end_accepts_numpy_datetime64():
    actual = _coerce_period_end(np.datetime64("2015-02-07T03:00"), None, "x")
    assert actual == datetime(2015, 2, 7, 3)


@pytest.mark.parametrize("zone", ["UTC+02:00", "+02:00", "+0200"])
def test_coerce_period_end_fixed_offset_column(zone):
    actual = _coerce_period_end(datetime(2015, 2, 7, 23, tzinfo=UTC), zone, "x")
    assert actual == datetime(2015, 2, 8, 1)


def test_coerce_period_end_negative_fixed_offset_column():
    actual = _coerce_period_end(datetime(2015, 2, 7, 1, tzinfo=UTC), "UTC-05:30", "x")
    assert actual == datetime(2015, 2, 6, 19, 30)


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #

pl = pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")


def sorted_frame(df: pd.DataFrame, by: str = "customer_id") -> pd.DataFrame:
    return df.sort_values(by).reset_index(drop=True)


@pytest.fixture(scope="module")
def transaction_data() -> pd.DataFrame:
    """The lifetimes fixture of ``TestRFM`` with a typed datetime column."""
    d = [
        [1, "2015-01-01", 1],
        [1, "2015-02-06", 2],
        [2, "2015-01-01", 2],
        [3, "2015-01-01", 3],
        [3, "2015-01-02", 1],
        [3, "2015-01-05", 5],
        [4, "2015-01-16", 6],
        [4, "2015-02-02", 3],
        [4, "2015-02-05", 3],
        [4, "2015-02-05", 6],
        [5, "2015-01-16", 3],
        [5, "2015-01-17", 1],
        [5, "2015-01-18", 8],
        [6, "2015-02-02", 5],
    ]
    df = pd.DataFrame(d, columns=["identifier", "date", "monetary_value"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def cdnow_datetime(cdnow_trans) -> pd.DataFrame:
    df = cdnow_trans.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


@pytest.fixture(scope="module")
def heavy_ties() -> pd.DataFrame:
    """Many customers sharing identical recency, frequency and spend values.

    pandas.qcut drops duplicated quantile edges here, so the score labels depend on
    exact tie handling.
    """
    rows = []
    for customer in range(1, 41):
        bucket = customer % 4
        # Four customers per bucket repeat the exact same purchase pattern.
        rows.append([customer, "2015-01-01", 10.0 * (bucket + 1)])
        if bucket >= 1:
            rows.append([customer, "2015-01-10", 10.0])
        if bucket >= 2:
            rows.append([customer, "2015-01-20", 10.0])
        if bucket == 3:
            rows.append([customer, "2015-01-30", 10.0])
    df = pd.DataFrame(rows, columns=["id", "date", "spend"])
    df["date"] = pd.to_datetime(df["date"])
    return df


EXPECTED_DAILY = pd.DataFrame(
    [
        [1, 1.0, 36.0, 37.0],
        [2, 0.0, 0.0, 37.0],
        [3, 2.0, 4.0, 37.0],
        [4, 2.0, 20.0, 22.0],
        [5, 2.0, 2.0, 22.0],
        [6, 0.0, 0.0, 5.0],
    ],
    columns=["customer_id", "frequency", "recency", "T"],
)

EXPECTED_WEEKLY = pd.DataFrame(
    [
        [1, 1.0, 5.0, 5.0],
        [2, 0.0, 0.0, 5.0],
        [3, 1.0, 1.0, 5.0],
        [4, 1.0, 3.0, 3.0],
        [5, 0.0, 0.0, 3.0],
        [6, 0.0, 0.0, 0.0],
    ],
    columns=["customer_id", "frequency", "recency", "T"],
)

EXPECTED_MONETARY = pd.DataFrame(
    [
        [1, 1.0, 36.0, 37.0, 2.0],
        [2, 0.0, 0.0, 37.0, 0.0],
        [3, 2.0, 4.0, 37.0, 3.0],
        [4, 2.0, 20.0, 22.0, 6.0],
        [5, 2.0, 2.0, 22.0, 4.5],
        [6, 0.0, 0.0, 5.0, 0.0],
    ],
    columns=["customer_id", "frequency", "recency", "T", "monetary_value"],
)

EXPECTED_FIRST_TRANSACTION = pd.DataFrame(
    [
        [1, 2.0, 1.0, 1.5],
        [2, 1.0, 37.0, 2.0],
        [3, 3.0, 33.0, 3.0],
        [4, 3.0, 2.0, 6.0],
        [5, 3.0, 20.0, 4.0],
        [6, 1.0, 5.0, 5.0],
    ],
    columns=["customer_id", "frequency", "recency", "monetary_value"],
)


# --------------------------------------------------------------------------- #
# Cross-backend equivalence with the legacy expected outputs
# --------------------------------------------------------------------------- #


class TestCrossBackendSummary:
    @pytest.mark.parametrize(
        "today",
        ["2015-02-07", date(2015, 2, 7), datetime(2015, 2, 7), pd.Period("2015-02-07")],
        ids=["string", "date", "datetime", "period"],
    )
    def test_daily(self, to_backend, transaction_data, today):
        actual = rfm_summary(
            to_backend(transaction_data),
            "identifier",
            "date",
            observation_period_end=today,
            granularity="1d",
            time_unit="day",
        )
        assert_frame_equal(collect_sorted(actual), EXPECTED_DAILY)

    def test_weekly(self, to_backend, transaction_data):
        actual = rfm_summary(
            to_backend(transaction_data),
            "identifier",
            "date",
            observation_period_end="2015-02-07",
            granularity="1w",
            time_unit="week",
        )
        assert_frame_equal(collect_sorted(actual), EXPECTED_WEEKLY)

    def test_monetary_values_and_first_transaction(self, to_backend, transaction_data):
        native = to_backend(transaction_data)
        actual = rfm_summary(
            native,
            "identifier",
            "date",
            monetary_value_col="monetary_value",
            observation_period_end="2015-02-07",
            granularity="1d",
        )
        assert_frame_equal(collect_sorted(actual), EXPECTED_MONETARY)

        actual_first = rfm_summary(
            native,
            "identifier",
            "date",
            monetary_value_col="monetary_value",
            observation_period_end="2015-02-07",
            include_first_transaction=True,
            granularity="1d",
        )
        assert_frame_equal(collect_sorted(actual_first), EXPECTED_FIRST_TRANSACTION)

    def test_observation_period_end_defaults_to_last_transaction(
        self, to_backend, transaction_data
    ):
        actual = collect_sorted(
            rfm_summary(
                to_backend(transaction_data), "identifier", "date", granularity="1d"
            )
        )
        # The last transaction is on 2015-02-06, one day before the fixed end above.
        expected = EXPECTED_DAILY.assign(T=EXPECTED_DAILY["T"] - 1)
        assert_frame_equal(actual, expected)

    def test_string_customer_ids(self, to_backend):
        df = pd.DataFrame(
            {
                "identifier": ["X", "X", "Y", "Y", "Y", "Y"],
                "date": pd.to_datetime(
                    [
                        "2015-02-01",
                        "2015-02-06",
                        "2015-01-01",
                        "2015-01-01",
                        "2015-01-02",
                        "2015-01-05",
                    ]
                ),
            }
        )
        actual = collect_sorted(
            rfm_summary(to_backend(df), "identifier", "date", granularity="1d")
        )
        assert actual["customer_id"].tolist() == ["X", "Y"]
        assert actual["frequency"].tolist() == [1.0, 2.0]
        assert actual["recency"].tolist() == [5.0, 4.0]
        assert actual["T"].tolist() == [5.0, 36.0]

    def test_unordered_rows_choose_the_correct_first_order(self, to_backend):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2014-04-09", "2014-03-14", "2014-05-21"]),
                "identifier": [2, 2, 2],
                "sales": [20, 10, 25],
            }
        )
        actual = collect_sorted(
            rfm_summary(to_backend(df), "identifier", "date", "sales", granularity="1d")
        )
        assert actual.loc[0, "monetary_value"] == 22.5

    def test_hardie_paper_statistics(self, to_backend, cdnow_datetime):
        summary = collect_sorted(
            rfm_summary(
                to_backend(cdnow_datetime),
                "id",
                "date",
                "spent",
                observation_period_end="1997-10-01",
                granularity="1d",
            )
        )
        results = summary[summary["frequency"] > 0]["monetary_value"].describe()
        assert np.round(results.loc["mean"]) == 35
        assert np.round(results.loc["std"]) == 30
        assert np.round(results.loc["min"]) == 3
        assert np.round(results.loc["50%"]) == 27
        assert np.round(results.loc["max"]) == 300
        assert np.round(results.loc["count"]) == 946

    @pytest.mark.parametrize(
        "granularity, time_unit, expected_recency, expected_T",
        [
            ("1d", "day", [65, 103, 0], [118, 103, 81]),
            ("1w", "week", [9, 15, 0], [17, 15, 12]),
        ],
    )
    def test_sparse_monthly_fixture(
        self, to_backend, granularity, time_unit, expected_recency, expected_T
    ):
        df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2, 3],
                "date": pd.to_datetime(
                    [
                        "2024-01-05",
                        "2024-03-10",
                        "2024-01-20",
                        "2024-05-02",
                        "2024-02-11",
                    ]
                ),
            }
        )
        summary = collect_sorted(
            rfm_summary(
                to_backend(df),
                "id",
                "date",
                granularity=granularity,
                time_unit=time_unit,
            )
        )
        assert summary["customer_id"].tolist() == [1, 2, 3]
        assert summary["frequency"].tolist() == [1, 1, 0]
        np.testing.assert_allclose(summary["recency"], expected_recency)
        np.testing.assert_allclose(summary["T"], expected_T)


class TestCrossBackendTrainTestSplit:
    def test_basic_split(self, to_backend, transaction_data):
        actual = collect_sorted(
            rfm_train_test_split(
                to_backend(transaction_data),
                "identifier",
                "date",
                "2015-02-01",
                granularity="1d",
            )
        ).set_index("customer_id")
        assert actual.loc[1, "test_frequency"] == 1
        assert actual.loc[2, "test_frequency"] == 0
        # Customer 6 only buys in the test period and is excluded.
        assert 6 not in actual.index
        assert actual["test_T"].tolist() == [5.0] * 5

    def test_weekly(self, to_backend, transaction_data):
        actual = collect_sorted(
            rfm_train_test_split(
                to_backend(transaction_data),
                "identifier",
                "date",
                "2015-02-01",
                test_period_end="2015-02-07",
                granularity="1w",
                time_unit="week",
            )
        )
        expected = pd.DataFrame(
            [
                [1, 0.0, 0.0, 4.0, 1.0, 1.0],
                [2, 0.0, 0.0, 4.0, 0.0, 1.0],
                [3, 1.0, 1.0, 4.0, 0.0, 1.0],
                [4, 0.0, 0.0, 2.0, 1.0, 1.0],
                [5, 0.0, 0.0, 2.0, 0.0, 1.0],
            ],
            columns=[
                "customer_id",
                "frequency",
                "recency",
                "T",
                "test_frequency",
                "test_T",
            ],
        )
        assert_frame_equal(actual, expected)

    def test_date_boundaries(self, to_backend, transaction_data):
        actual = collect_sorted(
            rfm_train_test_split(
                to_backend(transaction_data),
                "identifier",
                "date",
                train_period_end="2015-02-01",
                test_period_end="2015-02-04",
                granularity="1d",
            )
        ).set_index("customer_id")
        assert actual.loc[2, "test_frequency"] == 0
        assert actual.loc[4, "test_frequency"] == 1
        assert actual["test_T"].tolist() == [3.0] * 5

    def test_monetary_value(self, to_backend, transaction_data):
        actual = collect_sorted(
            rfm_train_test_split(
                to_backend(transaction_data),
                "identifier",
                "date",
                "2015-02-01",
                test_period_end="2015-02-07",
                monetary_value_col="monetary_value",
                granularity="1d",
            )
        )
        assert actual["monetary_value"].tolist() == [0, 0, 3, 0, 4.5]
        assert actual["test_monetary_value"].tolist() == [2, 0, 0, 6, 0]
        assert list(actual.columns) == [
            "customer_id",
            "frequency",
            "recency",
            "T",
            "monetary_value",
            "test_frequency",
            "test_monetary_value",
            "test_T",
        ]

    def test_include_first_transaction_drops_T(self, to_backend, transaction_data):
        actual = collect_sorted(
            rfm_train_test_split(
                to_backend(transaction_data),
                "identifier",
                "date",
                "2015-02-01",
                include_first_transaction=True,
                granularity="1d",
            )
        )
        assert list(actual.columns) == [
            "customer_id",
            "frequency",
            "recency",
            "test_frequency",
            "test_T",
        ]

    def test_empty_training_period_raises_on_eager_input(
        self, to_backend, transaction_data
    ):
        native = to_backend(transaction_data)
        if isinstance(nw.from_native(native), nw.LazyFrame):
            pytest.skip("lazy inputs cannot be inspected without materialising")
        with pytest.raises(ValueError, match="No data available"):
            rfm_train_test_split(
                native, "identifier", "date", "2014-02-07", granularity="1d"
            )

    def test_empty_test_period_raises_on_eager_input(
        self, to_backend, transaction_data
    ):
        native = to_backend(transaction_data)
        if isinstance(nw.from_native(native), nw.LazyFrame):
            pytest.skip("lazy inputs cannot be inspected without materialising")
        with pytest.raises(ValueError, match="No data available"):
            rfm_train_test_split(
                native,
                "identifier",
                "date",
                "2015-02-08",
                test_period_end="2014-02-07",
                granularity="1d",
            )


class TestCrossBackendSegments:
    @pytest.mark.parametrize("config", (None, "custom"))
    def test_segment_config(self, to_backend, transaction_data, config):
        if config is not None:
            config = {"Test Segment": ["111", "222", "333", "444"]}
            expected = ["Other", "Test Segment", "Other", "Other", "Other", "Other"]
        else:
            expected = [
                "Other",
                "Inactive Customer",
                "At Risk Customer",
                "Premium Customer",
                "Repeat Customer",
                "Top Spender",
            ]
        actual = collect_sorted(
            rfm_segments(
                to_backend(transaction_data),
                "identifier",
                "date",
                "monetary_value",
                segment_config=config,
                granularity="1d",
            )
        )
        assert actual["segment"].tolist() == expected
        assert list(actual.columns) == [
            "customer_id",
            "frequency",
            "recency",
            "monetary_value",
            "rfm_score",
            "segment",
        ]

    def test_two_bin_warning(self, to_backend):
        d = [
            [1, "2015-01-01", 1],
            [1, "2015-02-06", 2],
            [2, "2015-01-01", 2],
            [3, "2015-01-02", 1],
            [3, "2015-01-05", 3],
            [4, "2015-01-16", 4],
            [4, "2015-02-05", 5],
            [5, "2015-01-17", 1],
            [5, "2015-01-18", 2],
            [5, "2015-01-19", 2],
        ]
        repetitive = pd.DataFrame(d, columns=["id", "date", "monetary_value"])
        repetitive["date"] = pd.to_datetime(repetitive["date"])
        with pytest.warns(
            UserWarning, match="RFM score will not exceed 2 for f_quartile"
        ):
            rfm_segments(
                to_backend(repetitive), "id", "date", "monetary_value", granularity="1d"
            )

    def test_single_bin_raises_like_pandas_qcut(self, to_backend):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "date": pd.to_datetime(["2015-01-01"] * 4),
                "spend": [1.0, 1.0, 1.0, 1.0],
            }
        )
        with pytest.raises(ValueError, match="Bin labels must be one fewer"):
            rfm_segments(to_backend(df), "id", "date", "spend", granularity="1d")


# --------------------------------------------------------------------------- #
# Legacy-to-new parity on pandas input
# --------------------------------------------------------------------------- #

PARITY_CASES = [
    pytest.param("D", None, False, None, id="D"),
    pytest.param("D", "monetary_value", False, "2015-02-07", id="D-monetary-end"),
    pytest.param("D", "monetary_value", True, "2015-02-07", id="D-monetary-first"),
    pytest.param("W", None, False, "2015-02-07", id="W-end"),
    pytest.param("W", "monetary_value", False, None, id="W-monetary"),
    pytest.param("W", "monetary_value", True, None, id="W-monetary-first"),
    pytest.param("M", "monetary_value", False, "2015-03-15", id="M-monetary-end"),
]


def translate(time_unit: str) -> dict:
    return {
        "D": {"granularity": "1d", "time_unit": "day"},
        "W": {"granularity": "1w", "time_unit": "week"},
        "M": {"granularity": "1mo", "time_unit": "month"},
    }[time_unit]


class TestLegacyParity:
    @pytest.mark.parametrize("time_unit, monetary, include_first, end", PARITY_CASES)
    def test_summary(self, transaction_data, time_unit, monetary, include_first, end):
        legacy = _rfm_summary_legacy(
            transaction_data,
            "identifier",
            "date",
            monetary_value_col=monetary,
            observation_period_end=end,
            time_unit=time_unit,
            include_first_transaction=include_first,
        )
        new = _rfm_summary_nw(
            transaction_data,
            "identifier",
            "date",
            monetary_value_col=monetary,
            observation_period_end=end,
            include_first_transaction=include_first,
            **translate(time_unit),
        )
        assert_frame_equal(sorted_frame(legacy), sorted_frame(new))

    @pytest.mark.parametrize("time_unit", ["D", "W"])
    @pytest.mark.parametrize("time_scaler", [1, 7, 30.4375])
    def test_summary_time_scaler(self, cdnow_datetime, time_unit, time_scaler):
        legacy = _rfm_summary_legacy(
            cdnow_datetime,
            "id",
            "date",
            "spent",
            observation_period_end="1997-10-01",
            time_unit=time_unit,
            time_scaler=time_scaler,
        )
        new = _rfm_summary_nw(
            cdnow_datetime,
            "id",
            "date",
            "spent",
            observation_period_end="1997-10-01",
            extra_divisor=float(time_scaler),
            **translate(time_unit),
        )
        assert_frame_equal(sorted_frame(legacy), sorted_frame(new))

    @pytest.mark.parametrize("time_unit", ["D", "W"])
    @pytest.mark.parametrize("monetary", [None, "monetary_value"])
    @pytest.mark.parametrize("test_end", [None, "2015-02-07", "2015-02-04"])
    def test_train_test_split(self, transaction_data, time_unit, monetary, test_end):
        legacy = _rfm_train_test_split_legacy(
            transaction_data,
            "identifier",
            "date",
            "2015-02-01",
            test_period_end=test_end,
            time_unit=time_unit,
            monetary_value_col=monetary,
        )
        new = _rfm_train_test_split_nw(
            transaction_data,
            "identifier",
            "date",
            "2015-02-01",
            test_period_end=test_end,
            monetary_value_col=monetary,
            **translate(time_unit),
        )
        # The legacy path leaves test_frequency as int64 whenever every customer
        # buys in the test period; the new path always returns floats.
        assert_frame_equal(sorted_frame(legacy), sorted_frame(new), check_dtype=False)

    def test_train_test_split_cdnow(self, cdnow_datetime):
        legacy = _rfm_train_test_split_legacy(
            cdnow_datetime,
            "id",
            "date",
            "1997-09-30",
            test_period_end="1998-06-30",
            time_unit="W",
            monetary_value_col="spent",
        )
        new = _rfm_train_test_split_nw(
            cdnow_datetime,
            "id",
            "date",
            "1997-09-30",
            test_period_end="1998-06-30",
            monetary_value_col="spent",
            granularity="1w",
            time_unit="week",
        )
        assert_frame_equal(sorted_frame(legacy), sorted_frame(new), check_dtype=False)

    @pytest.mark.parametrize("time_unit", ["D", "W"])
    def test_segments(self, transaction_data, time_unit):
        legacy = _rfm_segments_legacy(
            transaction_data,
            "identifier",
            "date",
            "monetary_value",
            time_unit=time_unit,
        )
        new = _rfm_segments_nw(
            transaction_data,
            "identifier",
            "date",
            "monetary_value",
            **translate(time_unit),
        )
        assert_frame_equal(sorted_frame(legacy), sorted_frame(new))

    def test_segments_cdnow(self, cdnow_datetime):
        legacy = _rfm_segments_legacy(
            cdnow_datetime, "id", "date", "spent", observation_period_end="1997-10-01"
        )
        new = _rfm_segments_nw(
            cdnow_datetime,
            "id",
            "date",
            "spent",
            observation_period_end="1997-10-01",
            granularity="1d",
        )
        assert_frame_equal(sorted_frame(legacy), sorted_frame(new))

    def test_segments_heavy_ties(self, heavy_ties):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            legacy = _rfm_segments_legacy(heavy_ties, "id", "date", "spend")
            new = _rfm_segments_nw(heavy_ties, "id", "date", "spend", granularity="1d")
        assert_frame_equal(sorted_frame(legacy), sorted_frame(new))

    def test_segments_heavy_ties_on_every_backend(self, to_backend, heavy_ties):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            legacy = _rfm_segments_legacy(heavy_ties, "id", "date", "spend")
            new = rfm_segments(
                to_backend(heavy_ties), "id", "date", "spend", granularity="1d"
            )
        assert_frame_equal(sorted_frame(legacy), collect_sorted(new))


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


class TestRouting:
    @pytest.mark.parametrize(
        "legacy_kwargs",
        [
            {"time_unit": "W"},
            {"time_scaler": 7},
            {"datetime_format": "%Y-%m-%d"},
            {"sort_transactions": False},
        ],
    )
    @pytest.mark.parametrize(
        "new_kwargs",
        [
            {"granularity": "1d"},
            {"week_start": "monday"},
            {"backend": "pandas"},
            {"time_unit": "week"},
        ],
    )
    def test_mixed_parameters_raise(self, transaction_data, legacy_kwargs, new_kwargs):
        if "time_unit" in legacy_kwargs and "time_unit" in new_kwargs:
            pytest.skip("one keyword cannot carry two values")
        with pytest.raises(TypeError, match="cannot be combined"):
            rfm_summary(
                transaction_data, "identifier", "date", **legacy_kwargs, **new_kwargs
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"time_unit": "W"},
            {"time_unit": "M"},
            # pandas aliases without a named counterpart keep working.
            {"time_unit": "Q"},
            {"time_unit": "W-WED"},
            {"time_unit": "2D"},
            {"time_scaler": 7},
            {"datetime_format": "%Y-%m-%d"},
            {"sort_transactions": False},
        ],
    )
    def test_pandas_legacy_parameters_keep_legacy_output(
        self, transaction_data, kwargs
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            actual = rfm_summary(transaction_data, "identifier", "date", **kwargs)
        expected = _rfm_summary_legacy(transaction_data, "identifier", "date", **kwargs)
        assert_frame_equal(actual, expected)

    def test_non_pandas_translates_legacy_time_unit(self, transaction_data):
        native = pl.from_pandas(transaction_data)
        actual = rfm_summary(
            native,
            "identifier",
            "date",
            observation_period_end="2015-02-07",
            time_unit="W",
        )
        assert_frame_equal(collect_sorted(actual), EXPECTED_WEEKLY)

    def test_non_pandas_monthly_alias_counts_whole_months(self, transaction_data):
        legacy = _rfm_summary_legacy(
            transaction_data, "identifier", "date", time_unit="M"
        )
        actual = rfm_summary(
            pl.from_pandas(transaction_data), "identifier", "date", time_unit="M"
        )
        assert_frame_equal(collect_sorted(actual), sorted_frame(legacy))

    def test_non_pandas_default_time_unit_is_silent(self, transaction_data):
        native = pl.from_pandas(transaction_data)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            actual = rfm_summary(
                native, "identifier", "date", observation_period_end="2015-02-07"
            )
        assert_frame_equal(collect_sorted(actual), EXPECTED_DAILY)

    def test_non_pandas_translates_time_scaler(self, transaction_data):
        native = pl.from_pandas(transaction_data)
        actual = collect_sorted(
            rfm_summary(
                native,
                "identifier",
                "date",
                observation_period_end="2015-02-07",
                time_scaler=7,
            )
        )
        expected = EXPECTED_DAILY.assign(
            recency=EXPECTED_DAILY["recency"] / 7, T=EXPECTED_DAILY["T"] / 7
        )
        assert_frame_equal(actual, expected)

    @pytest.mark.parametrize("time_unit", ["Q", "H", "h", "fortnight"])
    def test_non_pandas_rejects_untranslatable_time_unit(
        self, transaction_data, time_unit
    ):
        with pytest.raises(TypeError, match="only supported for pandas input"):
            rfm_summary(
                pl.from_pandas(transaction_data),
                "identifier",
                "date",
                time_unit=time_unit,
            )

    def test_non_pandas_rejects_datetime_format(self, transaction_data):
        with pytest.raises(TypeError, match="cast the column to a datetime dtype"):
            rfm_summary(
                pl.from_pandas(transaction_data),
                "identifier",
                "date",
                datetime_format="%Y-%m-%d",
            )

    def test_non_pandas_ignores_sort_transactions_with_warning(self, transaction_data):
        native = pl.from_pandas(transaction_data)
        with pytest.warns(UserWarning, match="sort_transactions is ignored"):
            actual = rfm_summary(
                native,
                "identifier",
                "date",
                observation_period_end="2015-02-07",
                sort_transactions=False,
            )
        assert_frame_equal(collect_sorted(actual), EXPECTED_DAILY)

    def test_unknown_time_unit_raises(self, transaction_data):
        with pytest.raises(ValueError, match="Invalid frequency"):
            rfm_summary(transaction_data, "identifier", "date", time_unit="fortnight")

    def test_named_time_unit_matches_its_legacy_alias(self, transaction_data):
        """``"week"`` buckets by week like ``"W"``, on the Narwhals path."""
        actual = rfm_summary(
            transaction_data,
            "identifier",
            "date",
            observation_period_end="2015-02-07",
            time_unit="week",
        )
        assert_frame_equal(actual, EXPECTED_WEEKLY)

    def test_explicit_granularity_overrides_the_implied_one(self, transaction_data):
        actual = rfm_summary(
            transaction_data,
            "identifier",
            "date",
            observation_period_end="2015-02-07",
            granularity="1d",
            time_unit="week",
        )
        # Daily periods, measured in weeks.
        expected = EXPECTED_DAILY.assign(
            recency=EXPECTED_DAILY["recency"] / 7, T=EXPECTED_DAILY["T"] / 7
        )
        assert_frame_equal(actual, expected)

    def test_invalid_backend_fails_before_any_work(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            rfm_summary(object(), "identifier", "date", backend="dask")


# --------------------------------------------------------------------------- #
# Return types
# --------------------------------------------------------------------------- #


class TestReturnTypes:
    @pytest.mark.parametrize("function", [rfm_summary, rfm_segments])
    def test_native_in_native_out(self, to_backend, transaction_data, function):
        native = to_backend(transaction_data)
        kwargs = {"granularity": "1d"}
        if function is rfm_segments:
            result = function(native, "identifier", "date", "monetary_value", **kwargs)
        else:
            result = function(native, "identifier", "date", **kwargs)
        assert type(result) is type(native)

    def test_train_test_split_native_in_native_out(self, to_backend, transaction_data):
        native = to_backend(transaction_data)
        result = rfm_train_test_split(
            native, "identifier", "date", "2015-02-01", granularity="1d"
        )
        assert type(result) is type(native)

    def test_lazy_in_lazy_out(self, transaction_data):
        result = rfm_summary(
            pl.from_pandas(transaction_data).lazy(), "identifier", "date"
        )
        assert isinstance(result, pl.LazyFrame)

    @pytest.mark.parametrize(
        "backend, expected_type",
        [("pandas", pd.DataFrame), ("polars", pl.DataFrame), ("pyarrow", pa.Table)],
    )
    def test_backend_collects(
        self, to_backend, transaction_data, backend, expected_type
    ):
        native = to_backend(transaction_data)
        summary = rfm_summary(native, "identifier", "date", backend=backend)
        split = rfm_train_test_split(
            native, "identifier", "date", "2015-02-01", backend=backend
        )
        segments = rfm_segments(
            native, "identifier", "date", "monetary_value", backend=backend
        )
        assert isinstance(summary, expected_type)
        assert isinstance(split, expected_type)
        assert isinstance(segments, expected_type)

    def test_invalid_backend_raises(self, transaction_data):
        with pytest.raises(ValueError, match="Unsupported backend"):
            rfm_summary(transaction_data, "identifier", "date", backend="dask")

    def test_pandas_output_dtypes(self, transaction_data):
        actual = rfm_summary(
            transaction_data, "identifier", "date", "monetary_value", granularity="1d"
        )
        assert actual["customer_id"].dtype == "int64"
        assert all(
            actual[col].dtype == "float64"
            for col in ["frequency", "recency", "T", "monetary_value"]
        )


# --------------------------------------------------------------------------- #
# New API
# --------------------------------------------------------------------------- #


class TestNewApi:
    @pytest.mark.parametrize("granularity", [None, "1mo"])
    def test_monthly_time_unit_counts_whole_months(self, to_backend, granularity):
        df = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "date": pd.to_datetime(["2015-01-01", "2015-01-20", "2015-02-01"]),
            }
        )
        actual = collect_sorted(
            rfm_summary(
                to_backend(df),
                "id",
                "date",
                observation_period_end="2015-03-15",
                granularity=granularity,
                time_unit="month",
            )
        )
        # January and February are two periods, so one repeat purchase.
        assert actual["frequency"].tolist() == [1.0]
        assert actual["recency"].tolist() == [1.0]
        # The observation period is truncated to March: two months after January.
        assert actual["T"].tolist() == [2.0]

    def test_monthly_granularity_in_days(self, to_backend):
        df = pd.DataFrame(
            {"id": [1, 1], "date": pd.to_datetime(["2015-01-20", "2015-02-01"])}
        )
        actual = collect_sorted(
            rfm_summary(
                to_backend(df),
                "id",
                "date",
                observation_period_end="2015-03-15",
                granularity="1mo",
                time_unit="day",
            )
        )
        # Days between the starts of the months.
        assert actual["recency"].tolist() == [31.0]
        assert actual["T"].tolist() == [59.0]

    def test_whole_periods_over_many_years(self):
        df = pd.DataFrame(
            {"id": [1, 1], "date": pd.to_datetime(["1990-02-28", "2024-02-29"])}
        )
        for time_unit, expected in [
            ("month", 408.0),
            ("quarter", 136.0),
            ("year", 34.0),
        ]:
            actual = rfm_summary(df, "id", "date", time_unit=time_unit)
            assert actual["recency"].tolist() == [expected]

    def test_quarterly_granularity(self, to_backend):
        df = pd.DataFrame(
            {"id": [1, 1], "date": pd.to_datetime(["2015-01-15", "2015-05-02"])}
        )
        actual = collect_sorted(
            rfm_summary(
                to_backend(df),
                "id",
                "date",
                observation_period_end="2015-12-31",
                granularity="1q",
                time_unit="quarter",
            )
        )
        assert actual["frequency"].tolist() == [1.0]
        assert actual["recency"].tolist() == [1.0]
        assert actual["T"].tolist() == [3.0]

    def test_yearly_granularity(self, to_backend):
        df = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "date": pd.to_datetime(["2015-06-15", "2015-11-02", "2016-02-01"]),
            }
        )
        actual = collect_sorted(
            rfm_summary(
                to_backend(df), "id", "date", granularity="1y", time_unit="year"
            )
        )
        assert actual["frequency"].tolist() == [1.0]
        assert actual["recency"].tolist() == [1.0]
        assert actual["T"].tolist() == [1.0]

    def test_week_start_changes_buckets(self, to_backend):
        # Tuesday 2015-01-06 and Wednesday 2015-01-07.
        df = pd.DataFrame(
            {"id": [1, 1], "date": pd.to_datetime(["2015-01-06", "2015-01-07"])}
        )
        native = to_backend(df)
        monday = collect_sorted(
            rfm_summary(native, "id", "date", granularity="1w", time_unit="day")
        )
        wednesday = collect_sorted(
            rfm_summary(
                native,
                "id",
                "date",
                granularity="1w",
                time_unit="day",
                week_start="wednesday",
            )
        )
        assert monday["frequency"].tolist() == [0.0]
        assert monday["recency"].tolist() == [0.0]
        assert wednesday["frequency"].tolist() == [1.0]
        assert wednesday["recency"].tolist() == [7.0]
        assert wednesday["T"].tolist() == [7.0]

    @pytest.mark.parametrize(
        "time_unit, divisor",
        [
            ("day", 1.0),
            ("week", 7.0),
            ("month", 30.4375),
            ("quarter", 91.3125),
            ("year", 365.25),
        ],
    )
    def test_time_unit_constants(self, to_backend, time_unit, divisor):
        df = pd.DataFrame(
            {"id": [1, 1], "date": pd.to_datetime(["2015-01-01", "2016-01-01"])}
        )
        actual = collect_sorted(
            rfm_summary(
                to_backend(df),
                "id",
                "date",
                observation_period_end="2016-07-01",
                granularity="1d",
                time_unit=time_unit,
            )
        )
        assert actual["recency"].iloc[0] == pytest.approx(365 / divisor)
        assert actual["T"].iloc[0] == pytest.approx(547 / divisor)

    def test_same_period_purchases_collapse(self, to_backend):
        df = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "date": pd.to_datetime(
                    ["2015-01-01 09:00", "2015-01-01 17:00", "2015-01-03 08:00"]
                ),
                "spend": [10.0, 20.0, 5.0],
            }
        )
        actual = collect_sorted(
            rfm_summary(to_backend(df), "id", "date", "spend", granularity="1d")
        )
        assert actual["frequency"].tolist() == [1.0]
        assert actual["recency"].tolist() == [2.0]
        # The two same-day purchases are summed and count as the first transaction.
        assert actual["monetary_value"].tolist() == [5.0]

    def test_purchases_on_and_after_observation_period_end(self, to_backend):
        df = pd.DataFrame(
            {
                "id": [1, 1, 1, 2],
                "date": pd.to_datetime(
                    [
                        "2015-01-01 00:00",
                        "2015-01-10 23:59",
                        "2015-01-11 00:00",
                        "2015-01-12 00:00",
                    ]
                ),
            }
        )
        actual = collect_sorted(
            rfm_summary(
                to_backend(df),
                "id",
                "date",
                observation_period_end="2015-01-10",
                granularity="1d",
            )
        )
        # The purchase on the final day counts; later ones (and customer 2) are dropped.
        assert actual["customer_id"].tolist() == [1]
        assert actual["frequency"].tolist() == [1.0]
        assert actual["recency"].tolist() == [9.0]
        assert actual["T"].tolist() == [9.0]

    def test_weekly_end_includes_the_whole_final_week(self, to_backend):
        # Like the legacy "W" path, a purchase later in the week of
        # observation_period_end still belongs to the observed final period.
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "date": pd.to_datetime(["2015-01-05", "2015-01-18"]),
            }  # Mon, Sun
        )
        actual = collect_sorted(
            rfm_summary(
                to_backend(df),
                "id",
                "date",
                observation_period_end="2015-01-14",
                granularity="1w",
                time_unit="week",
            )
        )
        assert actual["frequency"].tolist() == [1.0]
        assert actual["recency"].tolist() == [1.0]
        assert actual["T"].tolist() == [1.0]

    def test_time_zone_aware_column(self, to_backend):
        if to_backend.backend_name == "duckdb":
            pytest.skip(
                "DuckDB keeps the time zone in the connection, not the column: "
                "wall-clock buckets follow the session zone rather than Asia/Tokyo"
            )
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "date": pd.to_datetime(
                    ["2015-01-01 23:30", "2015-01-05 00:30"]
                ).tz_localize("Asia/Tokyo"),
            }
        )
        native = to_backend(df)
        end = datetime(2015, 1, 10, 12, 0, tzinfo=UTC)
        actual = collect_sorted(
            rfm_summary(
                native, "id", "date", observation_period_end=end, granularity="1d"
            )
        )
        # Wall-clock days in Tokyo: Jan 1, Jan 5 and (Jan 10 21:00 Tokyo) Jan 10.
        assert actual["recency"].tolist() == [4.0]
        assert actual["T"].tolist() == [9.0]

    def test_time_zone_mismatch_raises(self, to_backend):
        aware = pd.DataFrame(
            {"id": [1], "date": pd.to_datetime(["2015-01-01"]).tz_localize("UTC")}
        )
        naive = pd.DataFrame({"id": [1], "date": pd.to_datetime(["2015-01-01"])})
        with pytest.raises(
            ValueError, match="naive but the datetime column is time-zone aware"
        ):
            rfm_summary(
                to_backend(aware),
                "id",
                "date",
                observation_period_end="2015-01-02",
                granularity="1d",
            )
        with pytest.raises(ValueError, match="aware but the datetime column is naive"):
            rfm_summary(
                to_backend(naive),
                "id",
                "date",
                observation_period_end=datetime(2015, 1, 2, tzinfo=UTC),
                granularity="1d",
            )

    def test_date_column(self):
        df = pl.DataFrame({"id": [1, 1], "date": [date(2015, 1, 1), date(2015, 1, 8)]})
        actual = rfm_summary(df, "id", "date", granularity="1w", time_unit="week")
        assert actual["frequency"].to_list() == [1.0]
        assert actual["recency"].to_list() == [1.0]

    def test_string_datetime_column_raises_with_cast_hint(self, to_backend):
        df = pd.DataFrame({"id": [1], "date": ["2015-01-01"]})
        with pytest.raises(TypeError, match=r"pl\.col\(col\)\.str\.to_datetime"):
            rfm_summary(to_backend(df), "id", "date", granularity="1d")

    def test_missing_column_raises(self, to_backend, transaction_data):
        with pytest.raises(KeyError, match="spend"):
            rfm_summary(
                to_backend(transaction_data),
                "identifier",
                "date",
                "spend",
                granularity="1d",
            )

    def test_multi_period_granularity_raises(self, transaction_data):
        with pytest.raises(ValueError, match="only single periods"):
            rfm_summary(transaction_data, "identifier", "date", granularity="2w")

    def test_week_start_requires_weekly_granularity(self, transaction_data):
        with pytest.raises(ValueError, match="week_start only applies"):
            rfm_summary(
                transaction_data,
                "identifier",
                "date",
                granularity="1d",
                week_start="monday",
            )

    def test_non_iso_string_period_end_raises(self, transaction_data):
        with pytest.raises(ValueError, match="ISO-8601"):
            rfm_summary(
                transaction_data,
                "identifier",
                "date",
                observation_period_end="02/07/2015",
                granularity="1d",
            )


# --------------------------------------------------------------------------- #
# Regressions: public functions on every backend against the legacy path
# --------------------------------------------------------------------------- #


def random_transactions(seed: int) -> pd.DataFrame:
    """Random log with several cohorts, float spend and late-joining customers."""
    rng = np.random.default_rng(seed)
    n_rows = 400
    customers = rng.integers(1, 61, size=n_rows)
    # Each customer joins on their own day, so cohorts spread over the whole log
    # and the last purchase is often made by a customer who is new in the test window.
    joined = rng.integers(0, 150, size=61)[customers]
    days = joined + rng.integers(0, 180 - joined)
    return pd.DataFrame(
        {
            "id": customers,
            "date": pd.Timestamp("2015-01-01") + pd.to_timedelta(days, unit="D"),
            # Multiples of 0.25 sum exactly, whatever order a backend adds them in.
            "spend": rng.integers(1, 400, size=n_rows) / 4,
        }
    )


class TestRandomLogsMatchLegacy:
    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("time_unit", ["D", "W", "M"])
    def test_summary(self, to_backend, seed, time_unit):
        df = random_transactions(seed)
        legacy = _rfm_summary_legacy(df, "id", "date", "spend", time_unit=time_unit)
        actual = rfm_summary(
            to_backend(df), "id", "date", "spend", **translate(time_unit)
        )
        assert_frame_equal(collect_sorted(actual), sorted_frame(legacy))

    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("time_unit", ["D", "W"])
    @pytest.mark.parametrize("test_end", [None, "2015-06-10"])
    def test_train_test_split(self, to_backend, seed, time_unit, test_end):
        df = random_transactions(seed)
        legacy = _rfm_train_test_split_legacy(
            df,
            "id",
            "date",
            "2015-04-15",
            test_period_end=test_end,
            time_unit=time_unit,
            monetary_value_col="spend",
        )
        actual = rfm_train_test_split(
            to_backend(df),
            "id",
            "date",
            "2015-04-15",
            test_period_end=test_end,
            monetary_value_col="spend",
            **translate(time_unit),
        )
        assert_frame_equal(
            collect_sorted(actual), sorted_frame(legacy), check_dtype=False
        )

    @pytest.mark.parametrize("seed", range(8))
    def test_segments(self, to_backend, seed):
        df = random_transactions(seed)
        legacy = _rfm_segments_legacy(df, "id", "date", "spend")
        actual = rfm_segments(to_backend(df), "id", "date", "spend", granularity="1d")
        assert_frame_equal(collect_sorted(actual), sorted_frame(legacy))


class TestRegressions:
    def test_test_T_counts_new_customers_last_purchase(self, to_backend):
        """The study ends at the last purchase of the log, whoever made it."""
        df = pd.DataFrame(
            {
                "id": [1, 1, 2, 9],
                "date": pd.to_datetime(
                    ["2015-01-01", "2015-01-20", "2015-01-05", "2015-03-31"]
                ),
            }
        )
        actual = collect_sorted(
            rfm_train_test_split(
                to_backend(df), "id", "date", "2015-01-10", granularity="1d"
            )
        )
        assert actual["customer_id"].tolist() == [1, 2]
        assert actual["test_T"].tolist() == [80.0, 80.0]

    def test_test_T_when_only_new_customers_buy_in_the_test_window(self, to_backend):
        df = pd.DataFrame(
            {
                "id": [1, 2, 9],
                "date": pd.to_datetime(["2015-01-01", "2015-01-05", "2015-03-31"]),
            }
        )
        actual = collect_sorted(
            rfm_train_test_split(
                to_backend(df), "id", "date", "2015-01-10", granularity="1d"
            )
        )
        assert actual["test_frequency"].tolist() == [0.0, 0.0]
        assert actual["test_T"].tolist() == [80.0, 80.0]

    def test_rows_without_timestamp_or_customer_are_dropped(self, to_backend):
        df = pd.DataFrame(
            {
                "id": ["a", "a", None, "b", "b"],
                "date": pd.to_datetime(
                    ["2015-01-01", "2015-01-11", "2015-01-05", "2015-01-03", None]
                ),
                "spend": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        legacy = _rfm_summary_legacy(df, "id", "date", "spend")
        native = to_backend(df)
        summary = collect_sorted(
            rfm_summary(native, "id", "date", "spend", granularity="1d")
        )
        assert summary["customer_id"].tolist() == ["a", "b"]
        assert_frame_equal(summary, sorted_frame(legacy))

        split = collect_sorted(
            rfm_train_test_split(native, "id", "date", "2015-01-04", granularity="1d")
        )
        assert split["customer_id"].tolist() == ["a", "b"]
        assert split["test_frequency"].tolist() == [1.0, 0.0]

    def test_pandas_result_has_a_range_index(self):
        df = pd.DataFrame(
            {
                "id": ["c", "b", "a", "a"],
                "date": pd.to_datetime(
                    ["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-09"]
                ),
                "spend": [1.0, 2.0, 3.0, 4.0],
            }
        )
        results = [
            rfm_summary(df, "id", "date", granularity="1d"),
            rfm_train_test_split(df, "id", "date", "2015-01-05", granularity="1d"),
            rfm_summary(pl.from_pandas(df), "id", "date", backend="pandas"),
        ]
        for result in results:
            assert result["customer_id"].tolist() == sorted(result["customer_id"])
            assert result.index.equals(pd.RangeIndex(len(result)))

    def test_segments_are_reproducible_with_float_spend(self, to_backend):
        rng = np.random.default_rng(0)
        df = random_transactions(0).assign(spend=rng.gamma(2.0, 17.3, size=400))
        native = to_backend(df)
        if isinstance(native, pl.LazyFrame) or to_backend.backend_name == "duckdb":
            pytest.skip("lazy inputs evaluate the summary twice; see rfm_segments")
        first = collect_sorted(rfm_segments(native, "id", "date", "spend"))
        for _ in range(10):
            again = collect_sorted(rfm_segments(native, "id", "date", "spend"))
            assert_frame_equal(again, first)
        legacy = _rfm_segments_legacy(df, "id", "date", "spend")
        assert first["rfm_score"].tolist() == sorted_frame(legacy)["rfm_score"].tolist()

    def test_first_period_spend_does_not_swamp_repeat_spend(self, to_backend):
        df = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "date": pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03"]),
                "spend": [1e17, 1.0, 3.0],
            }
        )
        actual = collect_sorted(
            rfm_summary(to_backend(df), "id", "date", "spend", granularity="1d")
        )
        assert actual["monetary_value"].tolist() == [2.0]

    def test_numpy_datetime64_period_ends(self, to_backend, transaction_data):
        native = to_backend(transaction_data)
        actual = rfm_train_test_split(
            native,
            "identifier",
            "date",
            np.datetime64("2015-02-01"),
            test_period_end=transaction_data["date"].to_numpy().max(),
            granularity="1d",
        )
        expected = rfm_train_test_split(
            native,
            "identifier",
            "date",
            "2015-02-01",
            test_period_end="2015-02-06",
            granularity="1d",
        )
        assert_frame_equal(collect_sorted(actual), collect_sorted(expected))

    def test_pandas_fixed_offset_time_zone(self, transaction_data):
        df = transaction_data.assign(
            date=transaction_data["date"].dt.tz_localize(timezone(timedelta(hours=2)))
        )
        actual = rfm_summary(
            df,
            "identifier",
            "date",
            observation_period_end=datetime(2015, 2, 6, 22, tzinfo=UTC),
            granularity="1d",
        )
        # 22:00 UTC on February 6 is already February 7 in the column's zone.
        assert_frame_equal(actual, EXPECTED_DAILY)

    def test_reserved_column_name_raises(self, transaction_data):
        df = transaction_data.rename(columns={"date": "_period"})
        with pytest.raises(ValueError, match="reserved for internal use"):
            rfm_summary(df, "identifier", "_period", granularity="1d")
