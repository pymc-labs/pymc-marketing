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


import warnings

import numpy as np
import pandas as pd
import pytest
from pymc.testing import mock_sample_setup_and_teardown

from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm.time_slice_cross_validation import (
    TimeSliceCrossValidationResult,
    TimeSliceCrossValidator,
)


# Helper wrapper that matches the run(...) API which expects an object
# with a build_model(...) method that returns an MMM-like object.
class _MMMBuilder:
    def __init__(self, mmm):
        self._mmm = mmm

    def build_model(self, X, y):
        # Build the internal MMM and return the MMM instance so the
        # TimeSliceCrossValidator receives a fit-capable object.
        try:
            # Some MMM implementations build in-place and return None; call and ignore return.
            _ = self._mmm.build_model(X, y)
        except Exception as e:
            # If build_model fails for any reason, warn so the test output
            # remains informative instead of silently swallowing the error.
            warnings.warn(
                f"_MMMBuilder.build_model: underlying build_model raised: {e}",
                stacklevel=2,
            )
        return self._mmm


# Mock sampling for faster tests
mock_pymc_sample = pytest.fixture(scope="module")(mock_sample_setup_and_teardown)


@pytest.fixture
def target_column():
    return "y_named"


@pytest.fixture
def mmm_fixture(target_column):
    return MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        dims=("country",),
        target_column=target_column,
        adstock=GeometricAdstock(l_max=3),
        saturation=LogisticSaturation(),
    )


@pytest.fixture
def XY(target_column) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=3, freq="W-MON").rename("date")
    df = pd.DataFrame(
        {
            ("A", "C1"): [1, 2, 3],
            ("B", "C1"): [4, 5, 6],
            ("A", "C2"): [7, 8, 9],
            ("B", "C2"): [10, 11, 12],
        },
        index=dates,
    )
    df.columns.names = ["country", "channel"]

    y = pd.DataFrame(
        {
            ("A", target_column): [1, 2, 3],
            ("B", target_column): [4, 5, 6],
        },
        index=dates,
    )
    y.columns.names = ["country", target_column]

    return (
        df.stack("country", future_stack=True).reset_index(),
        y.stack("country", future_stack=True).reset_index()[target_column],
    )


@pytest.fixture
def cv(XY):
    """Create a TimeSliceCrossValidator instance for testing."""
    return TimeSliceCrossValidator(n_init=1, forecast_horizon=1, date_column="date")


class TestTimeSliceCrossValidator:
    """Test the TimeSliceCrossValidator class."""

    def test_initialization(self):
        """Test TimeSliceCrossValidator initialization."""
        cv = TimeSliceCrossValidator(n_init=1, forecast_horizon=1, date_column="date")

        assert cv.n_init == 1
        assert cv.forecast_horizon == 1
        assert cv.date_column == "date"
        assert cv.step_size == 1  # Default value

    def test_initialization_with_step_size(self):
        """Test TimeSliceCrossValidator initialization with step_size."""
        cv = TimeSliceCrossValidator(
            n_init=1, forecast_horizon=1, date_column="date", step_size=4
        )
        assert cv.n_init == 1
        assert cv.forecast_horizon == 1
        assert cv.date_column == "date"
        assert cv.step_size == 4

    def test_get_n_splits(self, cv, XY):
        """Test get_n_splits method."""
        X, y = XY
        n_splits = cv.get_n_splits(X, y)
        expected_splits = (
            len(pd.to_datetime(X[cv.date_column]).unique())
            - cv.n_init
            - cv.forecast_horizon
        ) // cv.step_size + 1
        assert n_splits == expected_splits

    def test_split_generator(self, XY, cv):
        """Test split generator yields correct train/test indices."""
        X, y = XY
        # Use specific parameters for testing split generator
        splits = list(cv.split(X, y))

        # Check we get the expected number of splits
        expected_splits = cv.get_n_splits(X, y)
        assert len(splits) == expected_splits

        # Check first split
        train_idx, test_idx = splits[0]
        assert len(train_idx) == cv.n_init * 2
        assert len(test_idx) == cv.forecast_horizon * 2
        # basic ordering checks
        assert int(train_idx.max()) < int(test_idx.min())

    def test_fit_mmm(self, cv, XY, mmm_fixture, mock_pymc_sample):
        """Test _fit_mmm method."""
        X, y = XY

        fitted_mmm = cv._fit_mmm(mmm_fixture, X, y)

        # Check that the MMM was fitted
        assert fitted_mmm.idata is not None

    def test_time_slice_step(self, cv, XY, mmm_fixture, mock_pymc_sample):
        """Test _time_slice_step method."""
        X, y = XY
        X_train = X.iloc[:-1]
        y_train = y.iloc[:-1]
        X_test = X.iloc[-1:]
        y_test = y.iloc[-1:]

        result = cv._time_slice_step(mmm_fixture, X_train, y_train, X_test, y_test)

        assert isinstance(result, TimeSliceCrossValidationResult)
        assert result.X_train.equals(X_train)
        assert result.y_train.equals(y_train)
        assert result.X_test.equals(X_test)
        assert result.y_test.equals(y_test)
        assert result.idata is not None

    def test_run_method_basic(self, cv, XY, mmm_fixture, mock_pymc_sample):
        """Test run method returns correct number of results."""
        X, y = XY
        results = cv.run(X, y, mmm=_MMMBuilder(mmm_fixture))

        expected_splits = cv.get_n_splits(X, y)
        assert len(results) == expected_splits

        # Check that all results are TimeSliceCrossValidationResult instances
        for result in results:
            assert isinstance(result, TimeSliceCrossValidationResult)

    def test_run_method_idata_not_identical(
        self, cv, XY, mmm_fixture, mock_pymc_sample
    ):
        """Test that idata objects in results are not identical."""
        X, y = XY
        results = cv.run(X, y, mmm=_MMMBuilder(mmm_fixture))

        # Check that we have at least 2 results to compare
        assert len(results) >= 2, "Need at least 2 results to test idata uniqueness"

        # Check that idata objects are not identical
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                # Check that the idata objects are different instances
                assert results[i].idata is not results[j].idata, (
                    f"Results {i} and {j} have identical idata objects"
                )

                # Check that the idata objects have different memory addresses
                assert id(results[i].idata) != id(results[j].idata), (
                    f"Results {i} and {j} have idata objects with same memory address"
                )


class TestStepSize:
    """Test the step_size functionality."""

    @pytest.fixture
    def large_XY(self, target_column):
        """Create a larger dataset for testing step_size."""
        dates = pd.date_range("2025-01-01", periods=20, freq="D").rename("date")
        df = pd.DataFrame(
            {
                ("A", "C1"): range(20),
                ("B", "C1"): range(20, 40),
                ("A", "C2"): range(40, 60),
                ("B", "C2"): range(60, 80),
            },
            index=dates,
        )
        df.columns.names = ["country", "channel"]

        y = pd.DataFrame(
            {
                ("A", target_column): range(20),
                ("B", target_column): range(20, 40),
            },
            index=dates,
        )
        y.columns.names = ["country", target_column]

        return (
            df.stack("country", future_stack=True).reset_index(),
            y.stack("country", future_stack=True).reset_index()[target_column],
        )

    def test_step_size_default(self, mmm_fixture):
        """Test that step_size defaults to 1."""
        cv = TimeSliceCrossValidator(n_init=5, forecast_horizon=3, date_column="date")
        assert cv.step_size == 1

    def test_step_size_custom(self, mmm_fixture):
        """Test custom step_size values."""
        for step_size in [2, 3, 4, 5]:
            cv = TimeSliceCrossValidator(
                n_init=5, forecast_horizon=3, date_column="date", step_size=step_size
            )
            assert cv.step_size == step_size

    def test_get_n_splits_with_step_size(self, large_XY, mmm_fixture):
        """Test get_n_splits with different step_size values."""
        X, y = large_XY

        # Test with step_size=1 (default)
        cv1 = TimeSliceCrossValidator(
            n_init=5, forecast_horizon=3, date_column="date", step_size=1
        )
        n_splits1 = cv1.get_n_splits(X, y)

        # Test with step_size=4
        cv4 = TimeSliceCrossValidator(
            n_init=5, forecast_horizon=3, date_column="date", step_size=4
        )
        n_splits4 = cv4.get_n_splits(X, y)

        # With step_size=4, we should have fewer splits
        assert n_splits4 < n_splits1
        assert n_splits4 == 4  # (20 - 5 - 3) // 4 + 1 = 4
        assert n_splits1 == 13  # (20 - 5 - 3) + 1 = 13

    def test_split_generator_with_step_size(self, large_XY, mmm_fixture):
        """Test split generator with step_size."""
        X, y = large_XY

        # Test with step_size=1
        cv1 = TimeSliceCrossValidator(
            n_init=5, forecast_horizon=3, date_column="date", step_size=1
        )
        splits1 = list(cv1.split(X, y))

        # Test with step_size=4
        cv4 = TimeSliceCrossValidator(
            n_init=5, forecast_horizon=3, date_column="date", step_size=4
        )
        splits4 = list(cv4.split(X, y))

        # Check number of splits
        assert len(splits1) == 13
        assert len(splits4) == 4

        # Check that step_size=4 skips intermediate splits
        # First split should be the same (compare arrays element-wise)
        assert np.array_equal(splits1[0][0], splits4[0][0])
        assert np.array_equal(splits1[0][1], splits4[0][1])

        # Second split with step_size=4 should be the 5th split with step_size=1
        assert np.array_equal(splits1[4][0], splits4[1][0])
        assert np.array_equal(splits1[4][1], splits4[1][1])

        # Third split with step_size=4 should be the 9th split with step_size=1
        assert np.array_equal(splits1[8][0], splits4[2][0])
        assert np.array_equal(splits1[8][1], splits4[2][1])

    def test_step_size_edge_cases(self, large_XY, mmm_fixture):
        """Test step_size edge cases."""
        X, y = large_XY

        # Test with step_size larger than available data
        cv_large = TimeSliceCrossValidator(
            n_init=5,
            forecast_horizon=3,
            date_column="date",
            step_size=100,  # Much larger than available data
        )
        n_splits_large = cv_large.get_n_splits(X, y)
        assert (
            n_splits_large >= 1
        )  # With large step size we should still have at least one split

        # Test with step_size=0 (should raise assertion error)
        with pytest.raises(ValueError, match="step_size must be a positive integer"):
            TimeSliceCrossValidator(
                n_init=5, forecast_horizon=3, date_column="date", step_size=0
            )

    def test_step_size_with_run_method(self, large_XY, mmm_fixture, mock_pymc_sample):
        """Test that run method works correctly with step_size."""
        X, y = large_XY

        # Test with step_size=4
        cv4 = TimeSliceCrossValidator(
            n_init=5, forecast_horizon=3, date_column="date", step_size=4
        )

        results = cv4.run(X, y, mmm=_MMMBuilder(mmm_fixture))

        # Should have 4 results (as calculated in get_n_splits test)
        assert len(results) == 4

        # All results should be TimeSliceCrossValidationResult instances
        for result in results:
            assert isinstance(result, TimeSliceCrossValidationResult)

    def test_step_size_backward_compatibility(self, large_XY, mmm_fixture):
        """Test that step_size=1 maintains backward compatibility."""
        X, y = large_XY

        # Create two validators - one with explicit step_size=1, one without
        cv_explicit = TimeSliceCrossValidator(
            n_init=5, forecast_horizon=3, date_column="date", step_size=1
        )

        cv_implicit = TimeSliceCrossValidator(
            n_init=5, forecast_horizon=3, date_column="date"
        )

        # Both should produce identical results
        splits_explicit = list(cv_explicit.split(X, y))
        splits_implicit = list(cv_implicit.split(X, y))

        assert len(splits_explicit) == len(splits_implicit)
        for i, (split_explicit, split_implicit) in enumerate(
            zip(splits_explicit, splits_implicit, strict=True)
        ):
            assert np.array_equal(split_explicit[0], split_implicit[0]), (
                f"Split {i} train differs"
            )
            assert np.array_equal(split_explicit[1], split_implicit[1]), (
                f"Split {i} test differs"
            )

    def test_step_size_single_split(self, XY, mmm_fixture):
        """Test that with n_init=1, forecast_horizon=1, step_size=2, we get only one split."""
        X, y = XY

        # With 3 periods of data, n_init=1, forecast_horizon=1, step_size=2
        # We should only have one split: train on first period, test on second period
        # (third period is not used because we need n_init + forecast_horizon = 2 periods minimum)
        cv = TimeSliceCrossValidator(
            n_init=1, forecast_horizon=1, date_column="date", step_size=2
        )

        n_splits = cv.get_n_splits(X, y)
        assert n_splits == 1, f"Expected 1 split, got {n_splits}"

        # Verify we can actually generate the splits
        splits = list(cv.split(X, y))
        assert len(splits) == 1, f"Expected 1 split from generator, got {len(splits)}"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self, mmm_fixture):
        """Test behavior with empty DataFrame raises assertion error."""
        X = pd.DataFrame({"date": [], "channel_1": []})
        y = pd.Series([], name="target")

        cv = TimeSliceCrossValidator(n_init=5, forecast_horizon=3, date_column="date")

        with pytest.raises(ValueError):
            cv.run(X, y)

    def test_single_row_dataframe(self, mmm_fixture):
        """Test behavior with single row DataFrame raises assertion error."""
        X = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "channel_1": [1.0]})
        y = pd.Series([1.0], name="target")

        cv = TimeSliceCrossValidator(n_init=5, forecast_horizon=3, date_column="date")

        with pytest.raises(ValueError):
            cv.run(X, y)

    def test_invalid_date_column(self, XY, mmm_fixture):
        """Test behavior with invalid date column."""
        X, y = XY

        cv = TimeSliceCrossValidator(
            n_init=5, forecast_horizon=3, date_column="nonexistent_column"
        )

        # This should raise a KeyError when trying to access the column
        with pytest.raises(KeyError):
            cv.run(X, y, mmm=_MMMBuilder(mmm_fixture))

    def test_negative_n_init(self, mmm_fixture):
        """Test behavior with negative n_init raises assertion error."""
        with pytest.raises(ValueError, match="n_init must be a positive integer"):
            TimeSliceCrossValidator(
                n_init=-1,  # Invalid
                forecast_horizon=3,
                date_column="date",
            )

    def test_zero_forecast_horizon(self, mmm_fixture):
        """Test behavior with zero forecast horizon raises assertion error."""
        with pytest.raises(
            ValueError, match="forecast_horizon must be a positive integer"
        ):
            TimeSliceCrossValidator(
                n_init=5,
                forecast_horizon=0,  # Invalid
                date_column="date",
            )
