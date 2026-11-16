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
"""Tests for the RFM utilities, which accept any Narwhals dataframe.

This module checks the private helpers in ``pymc_marketing.clv._narwhals_utils``,
that every supported backend gives the expected results, and the ``granularity`` /
``time_unit`` API.
"""

from datetime import UTC, date, datetime, timedelta, timezone

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pymc_marketing.clv._narwhals_utils import (
    WEEKDAYS,
    TimeSpec,
    _coerce_period_end,
    _days_from_civil,
    _resolve_time_spec,
)
from pymc_marketing.clv.rfm import (
    _bin_edges,
    _default_rfm_segment_config,
    rfm_segments,
    rfm_summary,
    rfm_train_test_split,
)
from tests.clv.conftest import collect_sorted

EPOCH = date(1970, 1, 1)


def test_resolve_time_spec_period_follows_time_unit():
    assert _resolve_time_spec(None, "day", None) == TimeSpec("day", "day", 0)
    assert _resolve_time_spec(None, "quarter", None) == TimeSpec("quarter", "quarter")
    # An explicit granularity wins over the one the time unit implies.
    assert _resolve_time_spec("1mo", "day", None) == TimeSpec("month", "day")


def test_resolve_time_spec_week_start_offsets():
    for offset, weekday in enumerate(WEEKDAYS):
        assert _resolve_time_spec("1w", "week", weekday).week_start_offset == offset
    assert _resolve_time_spec(None, "week", "Wednesday").week_start_offset == 2


@pytest.mark.parametrize(
    "granularity, time_unit, week_start, match",
    [
        ("1d", "day", "monday", "week_start only applies to weekly"),
        ("1w", "week", "someday", "Unsupported week_start"),
        ("1d", "D", None, "Unsupported time_unit"),
        ("1d", "weeks", None, "Unsupported time_unit"),
        # Multiples are anchored differently by each backend, so none is accepted.
        ("2w", "week", None, "Unsupported granularity"),
        ("1h", "day", None, "Unsupported granularity"),
        ("week", "week", None, "Unsupported granularity"),
        (7, "day", None, "Unsupported granularity"),
    ],
)
def test_resolve_time_spec_rejects(granularity, time_unit, week_start, match):
    with pytest.raises(ValueError, match=match):
        _resolve_time_spec(granularity, time_unit, week_start)


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


# Each period, with the pandas period alias that buckets the same way.
PERIODS = [
    (TimeSpec("day", "day"), "D"),
    (TimeSpec("week", "week"), "W-SUN"),  # weeks that start on Monday
    (TimeSpec("week", "week", 2), "W-TUE"),  # ... on Wednesday
    (TimeSpec("week", "week", 6), "W-SAT"),  # ... on Sunday
    (TimeSpec("month", "month"), "M"),
    (TimeSpec("quarter", "quarter"), "Q"),
    (TimeSpec("year", "year"), "Y"),
]


@pytest.mark.parametrize("spec, freq", PERIODS, ids=[freq for _, freq in PERIODS])
def test_period_matches_pandas_on_every_backend(to_backend, spec, freq):
    frame = _day_range()
    result = (
        nw.from_native(to_backend(frame))
        .lazy()
        .select(nw.col("ts"), spec.period(nw.col("ts")).alias("period"))
        .collect()
        .sort("ts")
    )
    start_time = frame["ts"].dt.to_period(freq).dt.start_time
    expected = (start_time - pd.Timestamp(EPOCH)).dt.days.to_numpy()
    np.testing.assert_array_equal(result["period"].to_numpy(), expected)
    # One datetime goes through the same rule.
    for position in (0, 59, 20_000, len(frame) - 1):
        assert (
            spec.period_of(frame["ts"][position].to_pydatetime()) == expected[position]
        )


def test_elapsed_counts_whole_periods_when_the_unit_is_the_period():
    # 2015-01-01 to 2016-03-01 is 425 days: 14 whole months, 13.96 mean months.
    assert TimeSpec("month", "month").elapsed(425) == 14.0
    assert TimeSpec("day", "month").elapsed(425) == pytest.approx(425 / 30.4375)
    assert TimeSpec("week", "week").elapsed(21) == 3.0


@pytest.mark.parametrize(
    "edges, expected",
    [
        ([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]),
        # Ties: equal edges count once, as pandas.qcut(duplicates="drop") does.
        ([0.0, 0.0, 0.0, 3.0, 4.0], [0.0, 3.0, 4.0]),
        # Edges apart by float noise are one edge, whatever order they come in.
        ([1e-15, -1e-15, 50.0, 75.0, 100.0], [-1e-15, 50.0, 75.0, 100.0]),
    ],
)
def test_bin_edges_are_the_distinct_edges(edges, expected):
    np.testing.assert_allclose(_bin_edges(edges), expected, rtol=0, atol=1e-9)


def test_bin_edges_leave_room_for_float_noise():
    """A value just above an edge, as a re-evaluated float sum can be, stays below."""
    bounds = _bin_edges([0.0, 25.0, 50.0, 75.0, 100.0])
    assert all(
        bound > edge for bound, edge in zip(bounds, [0, 25, 50, 75, 100], strict=True)
    )
    assert bounds[1] > 25.0 + 1e-14
    # The slack follows the largest edge, so an edge at zero gets as much.
    assert bounds[0] == pytest.approx(1e-10)


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


def test_coerce_period_end_rejects_bad_values():
    with pytest.raises(ValueError, match="ISO-8601"):
        _coerce_period_end("02/07/2015", None, "x")
    with pytest.raises(TypeError, match="must be a str, date, datetime"):
        _coerce_period_end(42, None, "x")


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
# Every backend against hand-written expected outputs
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

    def test_empty_training_period_raises(self, to_backend, transaction_data):
        native = to_backend(transaction_data)
        with pytest.raises(ValueError, match="No data available"):
            rfm_train_test_split(
                native, "identifier", "date", "2014-02-07", granularity="1d"
            )

    def test_no_transactions_after_training_period_raises(
        self, to_backend, transaction_data
    ):
        with pytest.raises(ValueError, match="No data available"):
            rfm_train_test_split(
                to_backend(transaction_data),
                "identifier",
                "date",
                "2015-02-08",
                granularity="1d",
            )

    def test_empty_test_period_raises(self, to_backend, transaction_data):
        native = to_backend(transaction_data)
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


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


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


def _warehouse_frame(backend: str, df: pd.DataFrame):
    """Wrap ``df`` in a SQL-only lazy frame, like a table in a warehouse."""
    if backend == "ibis":
        return pytest.importorskip("ibis").memtable(df)
    return pytest.importorskip("sqlframe.duckdb").DuckDBSession().createDataFrame(df)


@pytest.mark.parametrize("backend", ["sqlframe", "ibis"])
@pytest.mark.parametrize(
    "function, args",
    [
        (rfm_summary, ("identifier", "date", "monetary_value")),
        (rfm_train_test_split, ("identifier", "date", "2015-02-01")),
        (rfm_segments, ("identifier", "date", "monetary_value")),
    ],
    ids=["summary", "train_test_split", "segments"],
)
def test_warehouse_pushdown(backend, function, args, transaction_data):
    """SQLFrame (the PySpark API) and Ibis stand in for Spark and Snowflake.

    Only these few cases run on them: each query takes about a second, so the full
    cross-backend matrix above stays on the fast in-process backends.
    """
    native = _warehouse_frame(backend, transaction_data)
    result = function(native, *args, granularity="1d")
    # Still a lazy query: no row left the engine before the caller collects.
    assert type(result) is type(native)
    expected = function(transaction_data, *args, granularity="1d")
    assert_frame_equal(
        collect_sorted(result), collect_sorted(expected), check_dtype=False
    )


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
        # A purchase later in the week of
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
        with pytest.raises(ValueError, match="Unsupported granularity '2w'"):
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
# Regressions: public functions on every backend
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
            # Multiples of 0.25 sum exactly, whatever order a backend adds them in.
            "spend": rng.integers(1, 400, size=n_rows) / 4,
            # Purchases happen at any time of day.
            "date": pd.Timestamp("2015-01-01")
            + pd.to_timedelta(days, unit="D")
            + pd.to_timedelta(rng.integers(0, 86_400, size=n_rows), unit="s"),
        }
    )


# pandas period alias of each time unit the reference implementation covers.
REFERENCE_FREQ = {"day": "D", "week": "W", "month": "M"}


def n_periods(difference: pd.Series) -> pd.Series:
    """Whole periods in a Series of period differences."""
    return difference.map(lambda offset: offset.n).astype(float)


def reference_summary(
    df: pd.DataFrame, time_unit: str, end=None, include_first: bool = False
) -> pd.DataFrame:
    """RFM summary written with pandas periods, independent of the library code."""
    freq = REFERENCE_FREQ[time_unit]
    df = df.assign(period=df["date"].dt.to_period(freq))
    end = df["period"].max() if end is None else pd.Timestamp(end).to_period(freq)
    spend = (
        df[df["period"] <= end].groupby(["id", "period"], as_index=False)["spend"].sum()
    )
    first = spend.groupby("id")["period"].transform("min")
    customers = spend.groupby("id").agg(
        first=("period", "min"), last=("period", "max"), n=("period", "count")
    )
    recency = n_periods(customers["last"] - customers["first"])
    t_value = n_periods(end - customers["first"])
    n = customers["n"].astype(float)
    if include_first:
        columns = {
            "frequency": n,
            "recency": t_value - recency,
            "monetary_value": spend.groupby("id")["spend"].sum() / n,
        }
    else:
        repeat_spend = spend[spend["period"] != first].groupby("id")["spend"].sum()
        columns = {
            "frequency": n - 1,
            "recency": recency,
            "T": t_value,
            "monetary_value": (repeat_spend / (n - 1)).reindex(n.index).fillna(0.0),
        }
    return (
        pd.DataFrame(columns)
        .rename_axis("customer_id")
        .reset_index()
        .sort_values("customer_id", ignore_index=True)
    )


def reference_split(
    df: pd.DataFrame, time_unit: str, train_end: str, test_end: str | None
) -> pd.DataFrame:
    freq = REFERENCE_FREQ[time_unit]
    in_test = df["date"] > pd.Timestamp(train_end)
    if test_end is not None:
        in_test &= df["date"] <= pd.Timestamp(test_end)
    test = df[in_test].assign(period=df["date"].dt.to_period(freq))
    result = reference_summary(
        df[df["date"] <= pd.Timestamp(train_end)], time_unit, end=train_end
    ).set_index("customer_id")
    test_frequency = test.groupby("id")["period"].nunique()
    result["test_frequency"] = (
        test_frequency.reindex(result.index).fillna(0.0).astype(float)
    )
    result["test_monetary_value"] = (
        (test.groupby("id")["spend"].sum() / test_frequency)
        .reindex(result.index)
        .fillna(0.0)
    )
    study_end = (
        test["period"].max()
        if test_end is None
        else pd.Timestamp(test_end).to_period(freq)
    )
    result["test_T"] = float((study_end - pd.Timestamp(train_end).to_period(freq)).n)
    return result.reset_index()


def reference_segments(df: pd.DataFrame) -> pd.DataFrame:
    result = reference_summary(df, "day", include_first=True)
    score = ""
    for metric in ("recency", "frequency", "monetary_value"):
        bins = pd.qcut(result[metric], q=4, labels=False, duplicates="drop")
        # A low recency is good, so its labels run backwards.
        label = bins.max() + 1 - bins if metric == "recency" else bins + 1
        score = score + label.astype(str)
    result["rfm_score"] = score
    result["segment"] = "Other"
    for name, scores in _default_rfm_segment_config.items():
        result.loc[result["rfm_score"].isin(scores), "segment"] = name
    return result


class TestRandomLogsMatchReference:
    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("time_unit", ["day", "week", "month"])
    @pytest.mark.parametrize("end", [None, "2015-05-13"])
    @pytest.mark.parametrize("include_first", [False, True])
    def test_summary(self, to_backend, seed, time_unit, end, include_first):
        df = random_transactions(seed)
        actual = rfm_summary(
            to_backend(df),
            "id",
            "date",
            "spend",
            observation_period_end=end,
            include_first_transaction=include_first,
            time_unit=time_unit,
        )
        assert_frame_equal(
            collect_sorted(actual), reference_summary(df, time_unit, end, include_first)
        )

    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("time_unit", ["day", "week", "month"])
    @pytest.mark.parametrize("test_end", [None, "2015-06-10"])
    def test_train_test_split(self, to_backend, seed, time_unit, test_end):
        df = random_transactions(seed)
        actual = rfm_train_test_split(
            to_backend(df),
            "id",
            "date",
            "2015-04-15",
            test_period_end=test_end,
            monetary_value_col="spend",
            time_unit=time_unit,
        )
        assert_frame_equal(
            collect_sorted(actual),
            reference_split(df, time_unit, "2015-04-15", test_end),
        )

    @pytest.mark.parametrize("seed", range(8))
    def test_segments(self, to_backend, seed):
        df = random_transactions(seed)
        actual = rfm_segments(to_backend(df), "id", "date", "spend")
        assert_frame_equal(collect_sorted(actual), reference_segments(df))


class TestTimeUnit:
    def test_unknown_time_unit_raises(self, transaction_data):
        with pytest.raises(ValueError, match="Unsupported time_unit 'fortnight'"):
            rfm_summary(transaction_data, "identifier", "date", time_unit="fortnight")

    @pytest.mark.parametrize(
        "removed",
        [
            {"time_scaler": 7},
            {"datetime_format": "%Y-%m-%d"},
            {"sort_transactions": False},
        ],
    )
    def test_removed_parameters_raise(self, transaction_data, removed):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            rfm_summary(transaction_data, "identifier", "date", **removed)

    def test_explicit_granularity_overrides_the_implied_one(self, transaction_data):
        daily = rfm_summary(
            transaction_data, "identifier", "date", observation_period_end="2015-02-07"
        )
        actual = rfm_summary(
            transaction_data,
            "identifier",
            "date",
            observation_period_end="2015-02-07",
            granularity="1d",
            time_unit="week",
        )
        # Daily periods, measured in weeks.
        assert_frame_equal(
            actual, daily.assign(recency=daily["recency"] / 7, T=daily["T"] / 7)
        )

    def test_monthly_split_counts_the_test_window_in_months(self, to_backend):
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
        actual = collect_sorted(
            rfm_train_test_split(
                to_backend(df),
                "id",
                "date",
                train_period_end="2024-03-31",
                test_period_end="2024-05-31",
                time_unit="month",
            )
        )
        assert actual["test_T"].tolist() == [2.0, 2.0, 2.0]
        assert actual["test_frequency"].tolist() == [0.0, 1.0, 0.0]
        assert actual["recency"].tolist() == [2.0, 0.0, 0.0]
        assert actual["T"].tolist() == [2.0, 2.0, 1.0]


class TestRegressions:
    def test_default_period_end_adds_no_global_window(self, transaction_data):
        """A max over all transactions is a window without a partition.

        Spark runs such a window in a single task, so the default study end has to
        come from a one-row aggregate instead.
        """
        ibis = pytest.importorskip("ibis")
        result = rfm_summary(ibis.memtable(transaction_data), "identifier", "date")
        assert "OVER" not in str(ibis.to_sql(result))

    @pytest.mark.parametrize(
        "function, args",
        [
            (rfm_summary, ("identifier", "date", "monetary_value")),
            (rfm_train_test_split, ("identifier", "date", "2015-02-01")),
            (rfm_segments, ("identifier", "date", "monetary_value")),
        ],
        ids=["summary", "train_test_split", "segments"],
    )
    def test_the_sort_is_the_last_step_of_the_query(
        self, function, args, transaction_data
    ):
        """An engine need not keep the order of a subquery, so sort once, last."""
        ibis = pytest.importorskip("ibis")
        sql = str(ibis.to_sql(function(ibis.memtable(transaction_data), *args)))
        assert sql.count("ORDER BY") == 1
        assert "ORDER BY" in sql.rsplit(")", 1)[-1]

    def test_split_accepts_pandas_periods(self, to_backend, transaction_data):
        native = to_backend(transaction_data)
        actual = rfm_train_test_split(
            native,
            "identifier",
            "date",
            pd.Period("2015-02-01", "D"),
            pd.Period("2015-02-05", "D"),
        )
        expected = rfm_train_test_split(
            native, "identifier", "date", "2015-02-01", "2015-02-05"
        )
        assert_frame_equal(collect_sorted(actual), collect_sorted(expected))

    def test_customer_ids_of_mixed_types(self):
        """Such ids cannot be ordered, so the result comes back unsorted."""
        df = pd.DataFrame(
            {
                "id": [1, "b", 1, "b", 2.5],
                "date": pd.to_datetime(
                    [
                        "2015-01-01",
                        "2015-01-01",
                        "2015-01-03",
                        "2015-01-06",
                        "2015-01-02",
                    ]
                ),
            }
        )
        actual = rfm_summary(df, "id", "date").set_index("customer_id")
        assert actual["frequency"].to_dict() == {1: 1.0, "b": 1.0, 2.5: 0.0}
        assert actual["recency"].to_dict() == {1: 2.0, "b": 5.0, 2.5: 0.0}

    def test_polars_nan_is_a_missing_value(self):
        """Polars keeps NaN apart from null; pandas reads both as missing."""
        df = pl.DataFrame(
            {
                "id": [1.0, 1.0, 1.0, float("nan"), 2.0],
                "date": [date(2015, 1, day) for day in (1, 2, 3, 4, 5)],
                "spend": [9.0, float("nan"), 3.0, 4.0, 5.0],
            }
        )
        for frame in (df, df.lazy()):
            actual = collect_sorted(rfm_summary(frame, "id", "date", "spend"))
            assert actual["customer_id"].tolist() == [1.0, 2.0]
            # Two repeat periods, one of them without a known spend.
            assert actual["monetary_value"].tolist() == [1.5, 0.0]

    def test_segments_of_no_transactions_raise(self, to_backend, transaction_data):
        empty = to_backend(transaction_data.head(0))
        with pytest.raises(ValueError, match="No transactions to segment"):
            rfm_segments(empty, "identifier", "date", "monetary_value")

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
        native = to_backend(df)
        summary = collect_sorted(
            rfm_summary(native, "id", "date", "spend", granularity="1d")
        )
        assert summary["customer_id"].tolist() == ["a", "b"]
        assert summary["frequency"].tolist() == [1.0, 0.0]
        assert summary["recency"].tolist() == [10.0, 0.0]
        assert summary["T"].tolist() == [10.0, 8.0]
        assert summary["monetary_value"].tolist() == [2.0, 0.0]

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
            rfm_summary(pl.from_pandas(df), "id", "date").to_pandas(),
        ]
        for result in results:
            assert result["customer_id"].tolist() == sorted(result["customer_id"])
            assert result.index.equals(pd.RangeIndex(len(result)))

    @pytest.mark.parametrize("seed", [0, 2, 6])
    def test_segments_are_reproducible_with_float_spend(self, to_backend, seed):
        """Float sums differ in their last bits between runs of a parallel engine.

        With an odd number of customers the median is one customer's own spend, so
        that customer sits exactly on a quartile edge.
        """
        rng = np.random.default_rng(seed)
        n_rows = 20_000
        df = pd.DataFrame(
            {
                "id": rng.integers(0, 401, n_rows),
                "date": pd.Timestamp("2015-01-01")
                + pd.to_timedelta(rng.integers(0, 300, n_rows), unit="D"),
                "spend": rng.gamma(2.0, 17.3, n_rows),
            }
        )
        native = to_backend(df)
        scores = {
            tuple(
                collect_sorted(rfm_segments(native, "id", "date", "spend"))["rfm_score"]
            )
            for _ in range(15)
        }
        assert len(scores) == 1

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_segments_are_reproducible_with_refunded_spend(self, to_backend, seed):
        """Spend that refunds cancel out sums to zero give or take the last bits.

        With over a quarter of the customers fully refunded, the lowest quartile
        edge is such a sum.
        """
        rng = np.random.default_rng(seed)
        n_refunded, n_regular, n_pairs = 150, 251, 60
        spend = rng.gamma(2.0, 17.3, (n_refunded, n_pairs))
        df = pd.DataFrame(
            {
                "id": np.r_[
                    np.repeat(np.arange(n_refunded), 2 * n_pairs),
                    np.repeat(np.arange(n_refunded, n_refunded + n_regular), 40),
                ],
                "spend": np.r_[
                    np.c_[spend, -spend].ravel(),
                    rng.gamma(2.0, 17.3, n_regular * 40),
                ],
            }
        )
        df["date"] = pd.Timestamp("2015-01-01") + pd.to_timedelta(
            rng.integers(0, 300, len(df)), unit="D"
        )
        native = to_backend(df.sample(frac=1, random_state=seed))
        scores = {
            tuple(
                collect_sorted(rfm_segments(native, "id", "date", "spend"))["rfm_score"]
            )
            for _ in range(15)
        }
        assert len(scores) == 1

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
