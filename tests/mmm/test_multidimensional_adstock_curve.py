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
"""Tests for MMM.sample_adstock_curve() method.

Note: Fixtures `simple_mmm_data`, `panel_mmm_data`, `simple_fitted_mmm`, and
`panel_fitted_mmm` are defined in tests/mmm/conftest.py and automatically
available to all tests in this module.
"""

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM

# ============================================================================
# Basic Functionality Tests
# ============================================================================


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_returns_dataarray(fitted_mmm, request):
    """Test that sample_adstock_curve returns xr.DataArray."""
    mmm = request.getfixturevalue(fitted_mmm)
    # Act
    curves = mmm.sample_adstock_curve()

    # Assert
    assert isinstance(curves, xr.DataArray)


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_has_correct_dims(fitted_mmm, request):
    """Test that curves have correct dimensions.

    Note: The dimensions depend on how the adstock transformation's
    priors are configured. With default priors without channel dims,
    the output will be (time since exposure, sample). With channel-specific
    priors, it would include a channel dimension.
    """
    mmm = request.getfixturevalue(fitted_mmm)
    # Act
    curves = mmm.sample_adstock_curve()

    # Assert - should have sample and time since exposure dims at minimum
    assert "sample" in curves.dims
    assert "time since exposure" in curves.dims


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_num_samples_controls_shape(fitted_mmm, request):
    """Test that num_samples parameter controls number of posterior samples.

    Note: With mock_pymc_sample, we get a small number of samples.
    This test verifies that when num_samples < total, we get num_samples.
    """
    mmm = request.getfixturevalue(fitted_mmm)
    # Arrange - use a smaller num_samples than available
    total_available = (
        mmm.idata.posterior.sizes["chain"] * mmm.idata.posterior.sizes["draw"]
    )
    num_samples = min(5, total_available - 1)  # Request fewer than available

    # Skip if we don't have enough samples to test subsampling
    if total_available <= 2:
        pytest.skip("Not enough posterior samples to test subsampling")

    # Act
    curves = mmm.sample_adstock_curve(num_samples=num_samples)

    # Assert - should have exactly num_samples
    assert curves.sizes["sample"] == num_samples


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_uses_all_samples_when_num_samples_exceeds_total(
    fitted_mmm, request
):
    """Test that all samples are used when num_samples > total available."""
    mmm = request.getfixturevalue(fitted_mmm)
    # Arrange - Request more samples than available
    num_samples = 10000

    # Act
    curves = mmm.sample_adstock_curve(num_samples=num_samples)

    # Assert - Should get all available samples, not num_samples
    total_available = (
        mmm.idata.posterior.sizes["chain"] * mmm.idata.posterior.sizes["draw"]
    )
    assert curves.sizes["sample"] == total_available


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_uses_all_samples_when_num_samples_is_none(
    fitted_mmm, request
):
    """Test that all samples are used when num_samples is None."""
    mmm = request.getfixturevalue(fitted_mmm)
    # Act
    curves = mmm.sample_adstock_curve(num_samples=None)

    # Assert - Should get all available samples
    total_available = (
        mmm.idata.posterior.sizes["chain"] * mmm.idata.posterior.sizes["draw"]
    )
    assert curves.sizes["sample"] == total_available


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_random_state_reproducibility(fitted_mmm, request):
    """Test that random_state produces reproducible results."""
    mmm = request.getfixturevalue(fitted_mmm)
    # Arrange
    num_samples = 50
    random_state = 42

    # Act - Sample twice with same random_state
    curves1 = mmm.sample_adstock_curve(
        num_samples=num_samples, random_state=random_state
    )
    curves2 = mmm.sample_adstock_curve(
        num_samples=num_samples, random_state=random_state
    )

    # Assert - Results should be identical
    xr.testing.assert_equal(curves1, curves2)


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_random_state_different_seeds_differ(fitted_mmm, request):
    """Test that different random_state values produce different results.

    Note: This only works when subsampling (num_samples < total available).
    With mock_pymc_sample, we may have limited samples.
    """
    mmm = request.getfixturevalue(fitted_mmm)
    # Arrange - use smaller num_samples to ensure subsampling happens
    total_available = (
        mmm.idata.posterior.sizes["chain"] * mmm.idata.posterior.sizes["draw"]
    )
    num_samples = min(5, total_available - 1)

    # Skip if we don't have enough samples to test different subsampling
    if total_available <= 5:
        pytest.skip("Not enough posterior samples to test different subsampling")

    # Act - Sample with different random states
    curves1 = mmm.sample_adstock_curve(num_samples=num_samples, random_state=42)
    curves2 = mmm.sample_adstock_curve(num_samples=num_samples, random_state=123)

    # Assert - Results should differ (different posterior samples selected)
    assert not np.allclose(curves1.values, curves2.values)


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_random_state_with_generator(fitted_mmm, request):
    """Test that random_state accepts numpy Generator."""
    mmm = request.getfixturevalue(fitted_mmm)
    # Arrange - use smaller num_samples to ensure subsampling happens
    total_available = (
        mmm.idata.posterior.sizes["chain"] * mmm.idata.posterior.sizes["draw"]
    )
    num_samples = min(5, total_available - 1)

    # Skip if we don't have enough samples to test subsampling
    if total_available <= 2:
        pytest.skip("Not enough posterior samples to test subsampling")

    rng = np.random.default_rng(42)

    # Act - Should not raise
    curves = mmm.sample_adstock_curve(num_samples=num_samples, random_state=rng)

    # Assert
    assert curves.sizes["sample"] == num_samples


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_time_coordinate_range(fitted_mmm, request):
    """Test that time coordinate spans from 0 to l_max-1."""
    mmm = request.getfixturevalue(fitted_mmm)
    # Act
    curves = mmm.sample_adstock_curve()

    # Assert
    time_coords = curves.coords["time since exposure"].values
    assert time_coords[0] == pytest.approx(0.0)
    # Maximum time should be l_max - 1 (0-indexed)
    l_max = mmm.adstock.l_max
    assert np.max(time_coords) == pytest.approx(l_max - 1)


# ============================================================================
# Parameter Validation Tests
# ============================================================================


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
@pytest.mark.parametrize("amount", [0, -1])
def test_sample_adstock_curve_raises_on_invalid_amount(fitted_mmm, request, amount):
    """Test that invalid amount raises ValidationError."""
    mmm = request.getfixturevalue(fitted_mmm)
    # Act & Assert
    with pytest.raises(ValidationError):
        mmm.sample_adstock_curve(amount=amount)


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
@pytest.mark.parametrize("num_samples", [0, -1])
def test_sample_adstock_curve_raises_on_invalid_num_samples(
    fitted_mmm, request, num_samples
):
    """Test that invalid num_samples raises ValidationError.

    Note: None is valid (uses all samples), but 0 and negative are not.
    """
    mmm = request.getfixturevalue(fitted_mmm)
    # Act & Assert
    with pytest.raises(ValidationError):
        mmm.sample_adstock_curve(num_samples=num_samples)


def test_sample_adstock_curve_raises_on_unfitted_model():
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
        mmm.sample_adstock_curve()


def test_sample_adstock_curve_raises_when_no_posterior(simple_mmm_data):
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
        mmm.sample_adstock_curve()


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_curves_decay_over_time(fitted_mmm, request):
    """Test that sampled curves decay over time (adstock-specific behavior).

    Adstock curves should generally decrease as time since exposure increases,
    showing the carryover effect diminishing over time.

    Note: We check that the curve starts higher and ends lower, which is the
    characteristic behavior of adstock transformations.
    """
    mmm = request.getfixturevalue(fitted_mmm)
    # Act
    curves = mmm.sample_adstock_curve()

    # Assert - Check that mean curve shows decay behavior
    mean_curve = curves.mean(dim="sample")

    # First value (time 0) should be higher than last value (time l_max-1)
    # This is the characteristic decay of adstock effects
    first_val = mean_curve.isel({"time since exposure": 0}).values
    last_val = mean_curve.isel({"time since exposure": -1}).values

    assert np.all(first_val > last_val), "Adstock curve should decay over time"


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_amount_scales_linearly(fitted_mmm, request):
    """Test that doubling amount approximately doubles curve values (linearity).

    Adstock transformations should be linear with respect to the input amount.
    """
    mmm = request.getfixturevalue(fitted_mmm)
    # Act
    curves_1x = mmm.sample_adstock_curve(amount=1.0)
    curves_2x = mmm.sample_adstock_curve(amount=2.0)

    # Assert - 2x amount should give approximately 2x values
    # Use mean across samples for simpler comparison
    mean_1x = curves_1x.mean(dim="sample")
    mean_2x = curves_2x.mean(dim="sample")

    # Check approximate 2x relationship (within 10% tolerance for numerical stability)
    ratio = mean_2x / mean_1x
    assert np.allclose(ratio.values, 2.0, rtol=0.1)


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_works(fitted_mmm, request):
    """Test that model curves can be sampled.

    Note: The adstock transformation's sample_curve returns
    (time since exposure, sample) dimensions. Custom dimensions
    (like country) would only appear if the adstock priors were
    configured with those dimensions.
    """
    mmm = request.getfixturevalue(fitted_mmm)
    # Act
    curves = mmm.sample_adstock_curve()

    # Assert - Should have time since exposure and sample dimensions
    assert "time since exposure" in curves.dims
    assert "sample" in curves.dims
    # Should be a valid DataArray with values
    assert curves.sizes["time since exposure"] > 0
    assert curves.sizes["sample"] > 0


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_default_parameters(fitted_mmm, request):
    """Test that default parameters produce expected output."""
    mmm = request.getfixturevalue(fitted_mmm)
    # Act
    curves = mmm.sample_adstock_curve()

    # Assert - time coordinate spans from 0 to l_max-1
    l_max = mmm.adstock.l_max
    assert curves.sizes["time since exposure"] == l_max
    # Time values start at 0
    assert curves.coords["time since exposure"].values[0] == pytest.approx(0.0)


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_can_be_used_for_plotting(fitted_mmm, request):
    """Test that the returned array can be used for common plotting operations."""
    mmm = request.getfixturevalue(fitted_mmm)
    # Act
    curves = mmm.sample_adstock_curve()

    # Assert - These operations should work without error
    mean_curves = curves.mean(dim="sample")
    assert isinstance(mean_curves, xr.DataArray)

    lower = curves.quantile(0.05, dim="sample")
    upper = curves.quantile(0.95, dim="sample")
    assert isinstance(lower, xr.DataArray)
    assert isinstance(upper, xr.DataArray)

    # time coordinate should be usable for plotting
    time_values = curves.coords["time since exposure"].values
    assert len(time_values) > 0
    assert time_values[0] == 0.0  # Starts at 0
