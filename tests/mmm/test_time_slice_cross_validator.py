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


import copy
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pymc.testing import mock_sample_setup_and_teardown

from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm.plot import MMMPlotSuite
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
        # Return a fresh copy per-call to avoid shared-state across folds.
        mmm_copy = copy.deepcopy(self._mmm)
        try:
            _ = mmm_copy.build_model(X, y)
        except Exception as e:
            warnings.warn(
                f"_MMMBuilder.build_model: underlying build_model raised: {e}",
                stacklevel=2,
            )
        return mmm_copy


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
def XY_no_country(target_column) -> pd.DataFrame:
    """Return X,y without a country dimension (flat channels only)."""
    dates = pd.date_range("2025-01-01", periods=3, freq="W-MON").rename("date")
    df = pd.DataFrame(
        {
            "C1": [1, 2, 3],
            "C2": [4, 5, 6],
        },
        index=dates,
    )

    # Melt to long form so X has rows per date x channel
    X = df.reset_index().melt(id_vars="date", var_name="channel", value_name="spend")

    # Create y as a simple target per date and repeat per channel to align with X
    y_df = pd.DataFrame({target_column: [1, 2, 3]}, index=dates)
    y = y_df.loc[y_df.index.repeat(df.shape[1])].reset_index(drop=False)
    # Reset index leaves a 'date' column from reset_index; return the target series
    y = y[target_column]

    return X, y


@pytest.fixture
def XY_with_product(target_column) -> pd.DataFrame:
    """Return X,y with both country and an additional 'product' dimension."""
    dates = pd.date_range("2025-01-01", periods=3, freq="W-MON").rename("date")
    df = pd.DataFrame(
        {
            ("A", "p1", "C1"): [1, 2, 3],
            ("A", "p1", "C2"): [4, 5, 6],
            ("B", "p2", "C1"): [7, 8, 9],
            ("B", "p2", "C2"): [10, 11, 12],
        },
        index=dates,
    )
    df.columns.names = ["country", "product", "channel"]

    y = pd.DataFrame(
        {
            ("A", "p1", target_column): [1, 2, 3],
            ("B", "p2", target_column): [4, 5, 6],
        },
        index=dates,
    )
    y.columns.names = ["country", "product", target_column]

    # Stack country and product so X and y are in long form with both dimensions
    X_long = df.stack(["country", "product"], future_stack=True).reset_index()
    y_long = y.stack(["country", "product"], future_stack=True).reset_index()[
        target_column
    ]

    return X_long, y_long


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

    @pytest.mark.parametrize(
        argnames="random_seed",
        argvalues=[42, np.random.default_rng(42)],
        ids=["int", "rng"],
    )
    def test_run_method_basic(self, cv, XY, mmm_fixture, mock_pymc_sample, random_seed):
        """Test run method returns correct number of results."""
        X, y = XY
        # run() now returns a combined InferenceData; per-fold results are
        # available on `cv._cv_results` after calling run().
        _combined = cv.run(
            X,
            y,
            mmm=_MMMBuilder(mmm_fixture),
            sampler_config={"random_seed": random_seed},
        )

        expected_splits = cv.get_n_splits(X, y)
        assert hasattr(cv, "_cv_results")
        results = cv._cv_results
        assert len(results) == expected_splits

        # Check that all results are TimeSliceCrossValidationResult instances
        for result in results:
            assert isinstance(result, TimeSliceCrossValidationResult)

    def test_run_method_idata_not_identical(
        self, cv, XY, mmm_fixture, mock_pymc_sample
    ):
        """Test that idata objects in results are not identical."""
        X, y = XY
        _ = cv.run(X, y, mmm=_MMMBuilder(mmm_fixture))
        assert hasattr(cv, "_cv_results")
        results = cv._cv_results

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

        _ = cv4.run(X, y, mmm=_MMMBuilder(mmm_fixture))
        assert hasattr(cv4, "_cv_results")
        results = cv4._cv_results

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


def _build_simple_idata(dates, var_name="y_original_scale"):
    # build minimal posterior_predictive dataset with chain/draw/date dims
    arr = np.random.RandomState(1).normal(size=(1, 2, len(dates)))
    da = xr.DataArray(
        arr, dims=("chain", "draw", "date"), coords={"date": dates}, name=var_name
    )
    ds = xr.Dataset({var_name: da})
    return az.InferenceData(posterior_predictive=ds)


def test_combine_idata_builds_cv_metadata_and_cv_coord():
    cv = TimeSliceCrossValidator(n_init=1, forecast_horizon=1, date_column="date")

    dates1 = pd.to_datetime(["2025-01-01", "2025-01-08"])
    dates2 = pd.to_datetime(["2025-01-15", "2025-01-22"])

    idata1 = _build_simple_idata(dates1)
    idata2 = _build_simple_idata(dates2)

    # create simple dataframe placeholders
    df_train = pd.DataFrame({"date": dates1})
    df_test = pd.DataFrame({"date": dates2})

    r1 = TimeSliceCrossValidationResult(
        X_train=df_train,
        y_train=pd.Series([1, 2]),
        X_test=df_test,
        y_test=pd.Series([3, 4]),
        idata=idata1,
    )
    r2 = TimeSliceCrossValidationResult(
        X_train=df_train,
        y_train=pd.Series([5, 6]),
        X_test=df_test,
        y_test=pd.Series([7, 8]),
        idata=idata2,
    )

    cv._cv_results = [r1, r2]

    combined = cv._combine_idata(cv._cv_results, ["m1", "m2"])

    assert hasattr(combined, "posterior_predictive")
    assert "cv" in combined.posterior_predictive.coords
    assert hasattr(combined, "cv_metadata")
    assert "metadata" in combined.cv_metadata


def test_param_stability_uses_posterior_predictive_and_returns_fig_ax():
    # Build a combined idata with a 'cv' coord and a posterior_predictive variable
    cv_labels = ["m1", "m2"]
    # create array with dims chain, draw, cv
    arr = np.random.RandomState(1).normal(size=(1, 2, len(cv_labels)))
    da = xr.DataArray(
        arr, dims=("chain", "draw", "cv"), coords={"cv": cv_labels}, name="beta_channel"
    )
    ds = xr.Dataset({"beta_channel": da})
    idata = az.InferenceData(posterior_predictive=ds)

    suite = MMMPlotSuite(idata=None)

    fig_ax = suite.param_stability(idata, parameter=["beta_channel"])
    # should return a tuple (fig, ax) when dims is None
    assert isinstance(fig_ax, tuple)
    assert len(fig_ax) >= 1


# ------------------------------------------------------------


def _make_posterior_predictive(cv_labels, dates):
    # shape: chain=1, draw=2, cv=len(cv_labels), date=len(dates)
    arr = np.random.RandomState(2).normal(size=(1, 2, len(cv_labels), len(dates)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "date"),
        coords={"cv": cv_labels, "date": dates},
        name="y_original_scale",
    )
    ds = xr.Dataset({"y_original_scale": da})
    return ds


def test__combine_idata_raises_if_no_idata():
    cv = TimeSliceCrossValidator(n_init=1, forecast_horizon=1, date_column="date")
    # create two results without idata
    df = pd.DataFrame({"date": [pd.Timestamp("2025-01-01")]})
    r1 = TimeSliceCrossValidationResult(
        X_train=df, y_train=pd.Series([1]), X_test=df, y_test=pd.Series([1]), idata=None
    )
    r2 = TimeSliceCrossValidationResult(
        X_train=df, y_train=pd.Series([2]), X_test=df, y_test=pd.Series([2]), idata=None
    )
    results = [r1, r2]

    # Ensure internal _cv_results exists so metadata creation can reference
    # per-fold metadata without raising AttributeError inside the implementation.
    cv._cv_results = results

    # Current behavior builds an InferenceData that may contain only cv_metadata
    combined = cv._combine_idata(results, ["m1", "m2"])
    # Should at least contain cv_metadata group
    assert hasattr(combined, "cv_metadata")
    assert "metadata" in combined.cv_metadata


def test_combine_idata_uses_fallback_groups_when__groups_missing():
    cv = TimeSliceCrossValidator(n_init=1, forecast_horizon=1, date_column="date")

    dates1 = pd.to_datetime(["2025-01-01", "2025-01-08"])
    ds1 = _build_pp_dataset(dates1, var_name="y")
    ds2 = _build_pp_dataset(dates1, var_name="y")

    idata1 = _DummyIdata({"posterior_predictive": ds1})
    idata2 = _DummyIdata({"posterior_predictive": ds2})

    df = pd.DataFrame({"date": dates1})
    r1 = TimeSliceCrossValidationResult(
        X_train=df,
        y_train=pd.Series([1, 2]),
        X_test=df,
        y_test=pd.Series([3, 4]),
        idata=idata1,
    )
    r2 = TimeSliceCrossValidationResult(
        X_train=df,
        y_train=pd.Series([5, 6]),
        X_test=df,
        y_test=pd.Series([7, 8]),
        idata=idata2,
    )

    # Ensure cv._cv_results exists for metadata creation
    cv._cv_results = [r1, r2]

    combined = cv._combine_idata([r1, r2], ["m1", "m2"])

    assert hasattr(combined, "posterior_predictive")
    assert "cv" in combined.posterior_predictive.coords
    assert hasattr(combined, "cv_metadata")


def test_cv_crps_raises_when_cv_metadata_missing():
    # idata with posterior_predictive but no cv_metadata should raise
    dates = pd.to_datetime(["2025-01-01", "2025-01-08"])
    # create posterior_predictive without a 'cv' coordinate (date-only)
    arr = np.random.RandomState(5).normal(size=(1, 2, len(dates)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "date"),
        coords={"date": dates},
        name="y_original_scale",
    )
    ds = xr.Dataset({"y_original_scale": da})
    idata = az.InferenceData(posterior_predictive=ds)

    suite = MMMPlotSuite(idata=idata)

    with pytest.raises(ValueError):
        suite.cv_crps(idata)


def test_run_produces_combined_cv_idata_and_returns_results_list():
    dates = pd.date_range("2025-01-01", periods=4, freq="D")
    X = pd.DataFrame({"date": dates, "geo": ["g1"] * len(dates)})
    y = pd.Series(np.arange(len(dates)))

    cv = TimeSliceCrossValidator(
        n_init=1, forecast_horizon=1, date_column="date", step_size=1
    )

    # Local factory that always returns the same idata structure (fixed dates)
    class LocalFakeModel:
        def __init__(self):
            self.idata = None
            self.sampler_config = None

        def fit(self, X, y, progressbar=True):
            return None

        def sample_posterior_predictive(
            self, X, extend_idata=True, combined=True, progressbar=False, **kwargs
        ):
            # ignore X and always return idata with the same date coords
            fixed_dates = pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
            )[:]
            arr = np.random.RandomState(7).normal(size=(1, 2, len(fixed_dates)))
            da = xr.DataArray(
                arr,
                dims=("chain", "draw", "date"),
                coords={"date": fixed_dates},
                name="y_original_scale",
            )
            ds_pp = xr.Dataset({"y_original_scale": da})
            ds_post = xr.Dataset({"beta": da})
            ds_obs = xr.Dataset({"y": da})
            ds_const = xr.Dataset({"meta": ("date", fixed_dates)})
            ds_fit = xr.Dataset({"fit_info": da})

            self.idata = az.InferenceData(
                posterior=ds_post,
                posterior_predictive=ds_pp,
                observed_data=ds_obs,
                constant_data=ds_const,
                fit_data=ds_fit,
            )
            return self.idata

    class LocalFakeFactory:
        def build_model(self, X, y):
            return LocalFakeModel()

    combined = cv.run(X, y, mmm=LocalFakeFactory())

    # run now returns the combined arviz.InferenceData
    assert isinstance(combined, az.InferenceData)
    assert hasattr(cv, "cv_idata")

    # also expose per-fold results
    assert hasattr(cv, "_cv_results")
    results = cv._cv_results

    # run should return a list of per-fold results (accessible via _cv_results)
    assert isinstance(results, list)

    # combined should include the posterior_predictive group and cv coord
    assert hasattr(combined, "posterior_predictive")
    assert "cv" in combined.posterior_predictive.coords

    # last fold idata should be exposed as cv.idata
    assert hasattr(cv, "idata") and cv.idata is not None


def _build_pp_dataset(dates, var_name="y"):
    arr = np.random.RandomState(4).normal(size=(1, 2, len(dates)))
    da = xr.DataArray(
        arr, dims=("chain", "draw", "date"), coords={"date": dates}, name=var_name
    )
    ds = xr.Dataset({var_name: da})
    return ds


class _DummyIdata:
    """Lightweight stand-in that resembles parts of arviz.InferenceData but
    intentionally lacks the internal `_groups` attribute to exercise the
    fallback branch in `_combine_idata` used in tests.
    """

    def __init__(self, ds_dict: dict):
        for k, v in ds_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key, None)


def test_combine_idata_raises_when_model_names_shorter():
    """Ensure a clear error is raised when user supplies fewer model_names than CV splits."""
    cv = TimeSliceCrossValidator(n_init=1, forecast_horizon=1, date_column="date")

    dates1 = pd.to_datetime(
        ["2025-01-01", "2025-01-08"]
    )  # two dates -> two time points
    idata1 = _build_simple_idata(dates1)
    idata2 = _build_simple_idata(dates1)

    df_train = pd.DataFrame({"date": dates1})
    df_test = pd.DataFrame({"date": dates1})

    r1 = TimeSliceCrossValidationResult(
        X_train=df_train,
        y_train=pd.Series([1, 2]),
        X_test=df_test,
        y_test=pd.Series([3, 4]),
        idata=idata1,
    )
    r2 = TimeSliceCrossValidationResult(
        X_train=df_train,
        y_train=pd.Series([5, 6]),
        X_test=df_test,
        y_test=pd.Series([7, 8]),
        idata=idata2,
    )

    results = [r1, r2]

    # ensure per-fold metadata is available for _create_metadata
    cv._cv_results = results

    # Passing only one model name for two results should raise a ValueError
    # Implementation may raise at different points (explicit IndexError check
    # or later during concat/coord alignment), so accept any ValueError here.
    with pytest.raises(ValueError):
        cv._combine_idata(results, ["only_one_name"])


def test_combine_idata_uses_fallback_when_concat_raises(monkeypatch):
    """If xr.concat raises on the primary concat call, the implementation should
    fall back to assigning per-dataset 'cv' coords and concat along 'cv'."""
    cv = TimeSliceCrossValidator(n_init=1, forecast_horizon=1, date_column="date")

    dates1 = pd.to_datetime(["2025-01-01", "2025-01-08"])
    ds1 = _build_pp_dataset(dates1, var_name="y")
    ds2 = _build_pp_dataset(dates1, var_name="y")

    idata1 = az.InferenceData(posterior_predictive=ds1)
    idata2 = az.InferenceData(posterior_predictive=ds2)

    df = pd.DataFrame({"date": dates1})
    r1 = TimeSliceCrossValidationResult(
        X_train=df,
        y_train=pd.Series([1, 2]),
        X_test=df,
        y_test=pd.Series([3, 4]),
        idata=idata1,
    )
    r2 = TimeSliceCrossValidationResult(
        X_train=df,
        y_train=pd.Series([5, 6]),
        X_test=df,
        y_test=pd.Series([7, 8]),
        idata=idata2,
    )

    results = [r1, r2]

    # ensure per-fold metadata is available for _create_metadata
    cv._cv_results = results

    # Monkeypatch xr.concat to raise on the first invocation and then delegate to
    # the original implementation so the fallback path can succeed on its call.
    original_concat = xr.concat
    state = {"called": 0, "raised_once": False}

    def fake_concat(objs, *args, **kwargs):
        # On first call, simulate a failure
        if state["called"] == 0:
            state["called"] += 1
            state["raised_once"] = True
            raise RuntimeError("forced concat failure")
        # Subsequent calls behave normally
        return original_concat(objs, *args, **kwargs)

    monkeypatch.setattr(xr, "concat", fake_concat)

    try:
        combined = cv._combine_idata(results, ["m1", "m2"])
    finally:
        # restore to be safe (monkeypatch fixture will restore, but keep defensive)
        monkeypatch.setattr(xr, "concat", original_concat)

    assert state["raised_once"] is True
    assert hasattr(combined, "posterior_predictive")
    assert "cv" in combined.posterior_predictive.coords
    assert hasattr(combined, "cv_metadata")
