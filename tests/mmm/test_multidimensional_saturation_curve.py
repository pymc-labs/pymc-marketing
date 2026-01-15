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
"""Tests for MMM.sample_saturation_curve() method."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_mmm_data():
    """Create simple single-dimension MMM data."""
    date_range = pd.date_range("2023-01-01", periods=14)
    np.random.seed(42)

    channel_1 = np.random.randint(100, 500, size=len(date_range))
    channel_2 = np.random.randint(100, 500, size=len(date_range))
    channel_3 = np.random.randint(100, 500, size=len(date_range))

    X = pd.DataFrame(
        {
            "date": date_range,
            "channel_1": channel_1,
            "channel_2": channel_2,
            "channel_3": channel_3,
        }
    )
    y = pd.Series(
        channel_1
        + channel_2
        + channel_3
        + np.random.randint(100, 300, size=len(date_range)),
        name="target",
    )

    return {"X": X, "y": y}


@pytest.fixture
def panel_mmm_data():
    """Create panel (multidimensional) MMM data with country dimension."""
    date_range = pd.date_range("2023-01-01", periods=7)
    countries = ["US", "UK"]
    np.random.seed(123)

    records = []
    for country in countries:
        for date in date_range:
            channel_1 = np.random.randint(100, 500)
            channel_2 = np.random.randint(100, 500)
            target = channel_1 + channel_2 + np.random.randint(50, 150)
            records.append((date, country, channel_1, channel_2, target))

    df = pd.DataFrame(
        records,
        columns=["date", "country", "channel_1", "channel_2", "target"],
    )

    X = df[["date", "country", "channel_1", "channel_2"]].copy()
    y = df["target"].copy()

    return {"X": X, "y": y}


@pytest.fixture
def simple_fitted_mmm(simple_mmm_data, mock_pymc_sample):
    """Create a simple fitted MMM for testing."""
    X = simple_mmm_data["X"]
    y = simple_mmm_data["y"]

    mmm = MMM(
        channel_columns=["channel_1", "channel_2", "channel_3"],
        date_column="date",
        target_column="target",
        control_columns=None,
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    mmm.fit(X, y)

    return mmm


@pytest.fixture
def panel_fitted_mmm(panel_mmm_data, mock_pymc_sample):
    """Create a panel (multidimensional) fitted MMM for testing."""
    X = panel_mmm_data["X"]
    y = panel_mmm_data["y"]

    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        dims=("country",),
        control_columns=None,
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    mmm.fit(X, y)

    return mmm


# ============================================================================
# Basic Functionality Tests
# ============================================================================


def test_sample_saturation_curve_returns_dataarray(simple_fitted_mmm):
    """Test that sample_saturation_curve returns xr.DataArray."""
    # Act
    curves = simple_fitted_mmm.sample_saturation_curve()

    # Assert
    assert isinstance(curves, xr.DataArray)


def test_sample_saturation_curve_has_correct_dims_simple_model(simple_fitted_mmm):
    """Test that curves have correct dimensions for simple model.

    Note: The dimensions depend on how the saturation transformation's
    priors are configured. With default priors without channel dims,
    the output will be (x, sample). With channel-specific priors,
    it would include a channel dimension.
    """
    # Act
    curves = simple_fitted_mmm.sample_saturation_curve(num_points=100)

    # Assert - should have sample and x dims at minimum
    assert "sample" in curves.dims
    assert "x" in curves.dims
    # Verify the expected shape
    assert curves.sizes["x"] == 100


def test_sample_saturation_curve_has_correct_dims_panel_model(panel_fitted_mmm):
    """Test that curves have correct dimensions for panel model.

    Note: The dimensions depend on how the saturation transformation's
    priors are configured. Default priors result in (x, sample).
    """
    # Act
    curves = panel_fitted_mmm.sample_saturation_curve(num_points=100)

    # Assert - should have sample and x dims at minimum
    assert "sample" in curves.dims
    assert "x" in curves.dims
    # Verify the expected shape
    assert curves.sizes["x"] == 100


def test_sample_saturation_curve_num_points_controls_shape(simple_fitted_mmm):
    """Test that num_points parameter controls number of points."""
    # Arrange
    num_points = 50

    # Act
    curves = simple_fitted_mmm.sample_saturation_curve(num_points=num_points)

    # Assert
    assert curves.sizes["x"] == num_points


def test_sample_saturation_curve_num_samples_controls_shape(simple_fitted_mmm):
    """Test that num_samples parameter controls number of posterior samples.

    Note: With mock_pymc_sample, we get a small number of samples.
    This test verifies that when num_samples < total, we get num_samples.
    """
    # Arrange - use a smaller num_samples than available
    total_available = (
        simple_fitted_mmm.idata.posterior.sizes["chain"]
        * simple_fitted_mmm.idata.posterior.sizes["draw"]
    )
    num_samples = min(5, total_available - 1)  # Request fewer than available

    # Skip if we don't have enough samples to test subsampling
    if total_available <= 2:
        pytest.skip("Not enough posterior samples to test subsampling")

    # Act
    curves = simple_fitted_mmm.sample_saturation_curve(num_samples=num_samples)

    # Assert - should have exactly num_samples
    assert curves.sizes["sample"] == num_samples


def test_sample_saturation_curve_uses_all_samples_when_num_samples_exceeds_total(
    simple_fitted_mmm,
):
    """Test that all samples are used when num_samples > total available."""
    # Arrange
    # Request more samples than available
    num_samples = 10000

    # Act
    curves = simple_fitted_mmm.sample_saturation_curve(num_samples=num_samples)

    # Assert - Should get all available samples, not num_samples
    total_available = (
        simple_fitted_mmm.idata.posterior.sizes["chain"]
        * simple_fitted_mmm.idata.posterior.sizes["draw"]
    )
    assert curves.sizes["sample"] == total_available


def test_sample_saturation_curve_uses_all_samples_when_num_samples_is_none(
    simple_fitted_mmm,
):
    """Test that all samples are used when num_samples is None."""
    # Act
    curves = simple_fitted_mmm.sample_saturation_curve(num_samples=None)

    # Assert - Should get all available samples
    total_available = (
        simple_fitted_mmm.idata.posterior.sizes["chain"]
        * simple_fitted_mmm.idata.posterior.sizes["draw"]
    )
    assert curves.sizes["sample"] == total_available


def test_sample_saturation_curve_random_state_reproducibility(simple_fitted_mmm):
    """Test that random_state produces reproducible results."""
    # Arrange
    num_samples = 50
    random_state = 42

    # Act - Sample twice with same random_state
    curves1 = simple_fitted_mmm.sample_saturation_curve(
        num_samples=num_samples, random_state=random_state
    )
    curves2 = simple_fitted_mmm.sample_saturation_curve(
        num_samples=num_samples, random_state=random_state
    )

    # Assert - Results should be identical
    xr.testing.assert_equal(curves1, curves2)


def test_sample_saturation_curve_random_state_different_seeds_differ(simple_fitted_mmm):
    """Test that different random_state values produce different results.

    Note: This only works when subsampling (num_samples < total available).
    With mock_pymc_sample, we may have limited samples.
    """
    # Arrange - use smaller num_samples to ensure subsampling happens
    total_available = (
        simple_fitted_mmm.idata.posterior.sizes["chain"]
        * simple_fitted_mmm.idata.posterior.sizes["draw"]
    )
    num_samples = min(5, total_available - 1)

    # Skip if we don't have enough samples to test different subsampling
    if total_available <= 5:
        pytest.skip("Not enough posterior samples to test different subsampling")

    # Act - Sample with different random states
    curves1 = simple_fitted_mmm.sample_saturation_curve(
        num_samples=num_samples, random_state=42
    )
    curves2 = simple_fitted_mmm.sample_saturation_curve(
        num_samples=num_samples, random_state=123
    )

    # Assert - Results should differ (different posterior samples selected)
    assert not np.allclose(curves1.values, curves2.values)


def test_sample_saturation_curve_random_state_with_generator(simple_fitted_mmm):
    """Test that random_state accepts numpy Generator."""
    # Arrange - use smaller num_samples to ensure subsampling happens
    total_available = (
        simple_fitted_mmm.idata.posterior.sizes["chain"]
        * simple_fitted_mmm.idata.posterior.sizes["draw"]
    )
    num_samples = min(5, total_available - 1)

    # Skip if we don't have enough samples to test subsampling
    if total_available <= 2:
        pytest.skip("Not enough posterior samples to test subsampling")

    rng = np.random.default_rng(42)

    # Act - Should not raise
    curves = simple_fitted_mmm.sample_saturation_curve(
        num_samples=num_samples, random_state=rng
    )

    # Assert
    assert curves.sizes["sample"] == num_samples


def test_sample_saturation_curve_x_coordinate_range(simple_fitted_mmm):
    """Test that x coordinate spans from 0 to max_value in scaled space."""
    # Arrange
    max_value = 2.0

    # Act
    curves = simple_fitted_mmm.sample_saturation_curve(
        max_value=max_value,
        original_scale=False,  # Keep in scaled space
    )

    # Assert
    x_coords = curves.coords["x"].values
    assert x_coords[0] == pytest.approx(0.0)
    assert np.max(x_coords) == pytest.approx(max_value)


# ============================================================================
# Scaling Behavior Tests
# ============================================================================


def test_sample_saturation_curve_original_scale_differs_from_scaled(simple_fitted_mmm):
    """Test that original_scale=True produces different values than False."""
    # Act
    curves_original = simple_fitted_mmm.sample_saturation_curve(
        max_value=1.0,
        original_scale=True,
    )
    curves_scaled = simple_fitted_mmm.sample_saturation_curve(
        max_value=1.0,
        original_scale=False,
    )

    # Assert - Values should differ
    assert not np.allclose(curves_original.values, curves_scaled.values)


def test_sample_saturation_curve_x_coords_same_regardless_of_original_scale(
    simple_fitted_mmm,
):
    """Test that x coordinates remain in scaled space regardless of original_scale.

    Note: x coordinates always remain in scaled space (same as max_value) since
    converting them would require per-channel scaling which complicates plotting.
    """
    # Arrange
    max_value = 1.0

    # Act
    curves_scaled = simple_fitted_mmm.sample_saturation_curve(
        max_value=max_value,
        original_scale=False,
    )
    curves_original = simple_fitted_mmm.sample_saturation_curve(
        max_value=max_value,
        original_scale=True,
    )

    # Assert - x coordinates should be the same regardless of original_scale
    x_scaled = curves_scaled.coords["x"].values
    x_original = curves_original.coords["x"].values

    np.testing.assert_array_equal(x_scaled, x_original)


def test_sample_saturation_curve_original_scale_uses_target_scale(simple_fitted_mmm):
    """Test that original_scale=True scales y values by target_scale."""
    # Act
    curves_scaled = simple_fitted_mmm.sample_saturation_curve(
        max_value=1.0,
        original_scale=False,
    )
    curves_original = simple_fitted_mmm.sample_saturation_curve(
        max_value=1.0,
        original_scale=True,
    )

    # Assert - y values should be scaled by target_scale
    # Take mean across samples for simpler comparison
    mean_scaled = curves_scaled.mean(dim="sample")
    mean_original = curves_original.mean(dim="sample")

    # Original should be roughly scaled * target_scale
    # (Broadcasting makes exact comparison complex, so check magnitude)
    assert np.mean(np.abs(mean_original.values)) > np.mean(np.abs(mean_scaled.values))


# ============================================================================
# Parameter Validation Tests
# ============================================================================


@pytest.mark.parametrize("max_value", [0, -1])
def test_sample_saturation_curve_raises_on_invalid_max_value(
    simple_fitted_mmm, max_value
):
    """Test that invalid max_value raises ValidationError."""
    # Act & Assert
    with pytest.raises(ValidationError):
        simple_fitted_mmm.sample_saturation_curve(max_value=max_value)


@pytest.mark.parametrize("num_points", [0, -1])
def test_sample_saturation_curve_raises_on_invalid_num_points(
    simple_fitted_mmm, num_points
):
    """Test that invalid num_points raises ValidationError."""
    # Act & Assert
    with pytest.raises(ValidationError):
        simple_fitted_mmm.sample_saturation_curve(num_points=num_points)


@pytest.mark.parametrize("num_samples", [0, -1])
def test_sample_saturation_curve_raises_on_invalid_num_samples(
    simple_fitted_mmm, num_samples
):
    """Test that invalid num_samples raises ValidationError.

    Note: None is valid (uses all samples), but 0 and negative are not.
    """
    # Act & Assert
    with pytest.raises(ValidationError):
        simple_fitted_mmm.sample_saturation_curve(num_samples=num_samples)


def test_sample_saturation_curve_raises_on_unfitted_model():
    """Test that calling on unfitted model raises ValueError."""
    # Arrange - Create unfitted MMM
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    # Act & Assert
    with pytest.raises(ValueError, match="idata does not exist"):
        mmm.sample_saturation_curve()


def test_sample_saturation_curve_raises_when_no_posterior(simple_mmm_data):
    """Test that calling raises ValueError when idata exists but has no posterior.

    This can happen if only sample_prior_predictive() was called but not fit().
    """
    import arviz as az

    # Arrange - Create MMM with idata but no posterior
    X = simple_mmm_data["X"]
    y = simple_mmm_data["y"]

    mmm = MMM(
        channel_columns=["channel_1", "channel_2", "channel_3"],
        date_column="date",
        target_column="target",
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    # Build model and create empty idata (simulating prior-only sampling)
    mmm.build_model(X, y)
    mmm.idata = az.InferenceData()  # Empty idata without posterior

    # Act & Assert
    with pytest.raises(ValueError, match="posterior not found in idata"):
        mmm.sample_saturation_curve()


# ============================================================================
# Integration Tests
# ============================================================================


def test_sample_saturation_curve_curves_are_monotonic_increasing(simple_fitted_mmm):
    """Test that sampled curves are monotonically increasing.

    Saturation curves should always increase (or at least not decrease)
    as spend increases.

    Note: The saturation transformation's sample_curve may not have channel
    dimensions if the priors don't have them. We check the mean curve directly.
    """
    # Act
    curves = simple_fitted_mmm.sample_saturation_curve(
        num_points=100, original_scale=False
    )

    # Assert - Check monotonicity for mean curve
    mean_curve = curves.mean(dim="sample")

    # Values should be non-decreasing along x dimension
    diffs = np.diff(mean_curve.values)
    assert np.all(diffs >= -1e-6)  # Allow tiny numerical errors


def test_sample_saturation_curve_with_very_large_max_value(simple_fitted_mmm):
    """Test that method handles very large max_value without overflow."""
    # Act
    curves = simple_fitted_mmm.sample_saturation_curve(
        max_value=1e6,
        original_scale=False,
    )

    # Assert - Should not have NaN or inf values
    assert not np.any(np.isnan(curves.values))
    assert not np.any(np.isinf(curves.values))


def test_sample_saturation_curve_has_x_and_sample_dims(simple_fitted_mmm):
    """Test that output has required x and sample dimensions.

    Note: The saturation transformation's sample_curve returns (x, sample)
    dimensions. Channel/custom dimensions depend on how the saturation
    priors are configured.
    """
    # Act
    curves = simple_fitted_mmm.sample_saturation_curve()

    # Assert - Should always have x and sample dimensions
    assert "x" in curves.dims
    assert "sample" in curves.dims


def test_sample_saturation_curve_with_panel_model_works(panel_fitted_mmm):
    """Test that panel model curves can be sampled.

    Note: The saturation transformation's sample_curve returns (x, sample)
    dimensions. Custom dimensions (like country) would only appear if
    the saturation priors were configured with those dimensions.
    """
    # Act
    curves = panel_fitted_mmm.sample_saturation_curve()

    # Assert - Should have x and sample dimensions
    assert "x" in curves.dims
    assert "sample" in curves.dims
    # Should be a valid DataArray with values
    assert curves.sizes["x"] > 0
    assert curves.sizes["sample"] > 0


def test_sample_saturation_curve_default_parameters(simple_fitted_mmm):
    """Test that default parameters produce expected output."""
    # Act
    curves = simple_fitted_mmm.sample_saturation_curve()

    # Assert - default num_points is 100
    assert curves.sizes["x"] == 100
    # x values remain in scaled space (max 1.0 by default)
    assert np.max(curves.coords["x"].values) == pytest.approx(1.0)


def test_sample_saturation_curve_small_num_points(simple_fitted_mmm):
    """Test that small num_points works correctly."""
    # Act
    curves = simple_fitted_mmm.sample_saturation_curve(num_points=5)

    # Assert
    assert curves.sizes["x"] == 5
    # x coords should still span from 0 to max
    assert curves.coords["x"].values[0] == 0.0


def test_sample_saturation_curve_can_be_used_for_plotting(simple_fitted_mmm):
    """Test that the returned array can be used for common plotting operations."""
    # Act
    curves = simple_fitted_mmm.sample_saturation_curve()

    # Assert - These operations should work without error
    mean_curves = curves.mean(dim="sample")
    assert isinstance(mean_curves, xr.DataArray)

    lower = curves.quantile(0.05, dim="sample")
    upper = curves.quantile(0.95, dim="sample")
    assert isinstance(lower, xr.DataArray)
    assert isinstance(upper, xr.DataArray)

    # x coordinate should be usable for plotting
    x_values = curves.coords["x"].values
    assert len(x_values) > 0
    assert x_values[0] == 0.0  # Starts at 0
