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

The legacy pandas behaviour is covered by ``tests/clv/test_utils.py``; this module
checks that every supported backend reproduces it, that routing between the two
paths behaves, and that the new ``granularity`` / ``time_unit`` API is right.
"""

import warnings
from datetime import UTC, date, datetime

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pymc_marketing.clv.utils import (
    _rfm_segments_legacy,
    _rfm_segments_nw,
    _rfm_summary_legacy,
    _rfm_summary_nw,
    _rfm_train_test_split_legacy,
    _rfm_train_test_split_nw,
    _route,
    rfm_segments,
    rfm_summary,
    rfm_train_test_split,
)
from tests.clv.conftest import collect_sorted

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
]


def translate(time_unit: str) -> dict:
    return {
        "D": {"granularity": "1d", "time_unit": "day"},
        "W": {"granularity": "1w", "time_unit": "week"},
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

    def test_pandas_defaults_take_legacy_path_without_warning(self, transaction_data):
        path, resolved = _route(
            transaction_data,
            time_unit="D",
            time_scaler=1,
            datetime_format=None,
            sort_transactions=True,
            granularity=None,
            week_start=None,
            backend=None,
        )
        assert path == "legacy"
        assert resolved == {
            "time_unit": "D",
            "time_scaler": 1,
            "datetime_format": None,
            "sort_transactions": True,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rfm_summary(transaction_data, "identifier", "date")

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (
                {"time_unit": "W"},
                r"'time_unit' is deprecated.*granularity='1w', time_unit='week'",
            ),
            ({"time_unit": "M"}, r"'time_unit' is deprecated.*no exact equivalent"),
            ({"time_scaler": 7}, r"'time_scaler' is deprecated.*time_unit='week'"),
            (
                {"time_scaler": 30.4375},
                r"'time_scaler' is deprecated.*time_unit='month'",
            ),
            ({"time_scaler": 365}, r"'time_scaler' is deprecated.*time_unit='year'"),
            (
                {"time_scaler": 12},
                r"'time_scaler' is deprecated.*granularity=\.\.\. / time_unit=\.\.\.",
            ),
            (
                {"datetime_format": "%Y-%m-%d"},
                r"'datetime_format' is deprecated.*pandas.to_datetime",
            ),
            (
                {"sort_transactions": False},
                r"'sort_transactions' is deprecated.*row order",
            ),
        ],
    )
    def test_pandas_legacy_parameters_warn(self, transaction_data, kwargs, match):
        with pytest.warns(FutureWarning, match=match) as record:
            legacy_output = rfm_summary(
                transaction_data, "identifier", "date", **kwargs
            )
        assert all(
            "v2.0" in str(w.message) for w in record if w.category is FutureWarning
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            expected = _rfm_summary_legacy(
                transaction_data, "identifier", "date", **kwargs
            )
        assert_frame_equal(legacy_output, expected)

    def test_pandas_legacy_warning_points_at_caller(self, transaction_data):
        with pytest.warns(FutureWarning) as record:
            rfm_summary(transaction_data, "identifier", "date", time_unit="W")
        assert record[0].filename == __file__

    def test_train_test_split_warns_once_per_parameter(self, transaction_data):
        with pytest.warns(FutureWarning) as record:
            rfm_train_test_split(
                transaction_data, "identifier", "date", "2015-02-01", time_unit="W"
            )
        assert len([w for w in record if w.category is FutureWarning]) == 1

    def test_non_pandas_translates_legacy_time_unit(self, transaction_data):
        native = pl.from_pandas(transaction_data)
        with pytest.warns(FutureWarning, match="granularity='1w', time_unit='week'"):
            actual = rfm_summary(
                native,
                "identifier",
                "date",
                observation_period_end="2015-02-07",
                time_unit="W",
            )
        assert_frame_equal(collect_sorted(actual), EXPECTED_WEEKLY)

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
        with pytest.warns(FutureWarning, match="time_unit='week'"):
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

    @pytest.mark.parametrize("time_unit", ["M", "H", "h"])
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
        with (
            pytest.warns(UserWarning, match="sort_transactions is ignored"),
            pytest.warns(FutureWarning, match="'sort_transactions' is deprecated"),
        ):
            actual = rfm_summary(
                native,
                "identifier",
                "date",
                observation_period_end="2015-02-07",
                sort_transactions=False,
            )
        assert_frame_equal(collect_sorted(actual), EXPECTED_DAILY)

    def test_unknown_time_unit_raises(self, transaction_data):
        with pytest.raises(ValueError, match="Unsupported time_unit"):
            rfm_summary(transaction_data, "identifier", "date", time_unit="fortnight")

    def test_pandas_named_time_unit_takes_new_path(self, transaction_data):
        actual = rfm_summary(
            transaction_data,
            "identifier",
            "date",
            observation_period_end="2015-02-07",
            time_unit="week",
        )
        # Daily periods, measured in weeks.
        expected = EXPECTED_DAILY.assign(
            recency=EXPECTED_DAILY["recency"] / 7, T=EXPECTED_DAILY["T"] / 7
        )
        assert_frame_equal(actual, expected)


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
    def test_monthly_granularity_keeps_floats(self, to_backend):
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
                granularity="1mo",
                time_unit="month",
            )
        )
        # January and February are two periods, so one repeat purchase.
        assert actual["frequency"].tolist() == [1.0]
        assert actual["recency"].iloc[0] == pytest.approx(31 / 30.4375)
        # The observation period is truncated to 2015-03-01: 59 days after Jan 1.
        assert actual["T"].iloc[0] == pytest.approx(59 / 30.4375)

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
        assert actual["recency"].iloc[0] == pytest.approx(90 / 91.3125)
        assert actual["T"].iloc[0] == pytest.approx(273 / 91.3125)

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
        assert actual["recency"].iloc[0] == pytest.approx(365 / 365.25)
        assert actual["T"].iloc[0] == pytest.approx(365 / 365.25)

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

    def test_lazy_split_with_empty_period_returns_empty(self, transaction_data):
        result = rfm_train_test_split(
            pl.from_pandas(transaction_data).lazy(), "identifier", "date", "2014-02-07"
        )
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().height == 0
