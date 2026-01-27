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
"""Tests for MMM.sample_saturation_curve() method.

Note: Fixtures `simple_mmm_data`, `panel_mmm_data`, `simple_fitted_mmm`, and
`panel_fitted_mmm` are defined in tests/mmm/conftest.py and automatically
available to all tests in this module.

This is a reduced test suite that removes redundancies while maintaining
full coverage of the method's functionality.
"""

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
class TestSampleSaturationCurveBasics:
    def test_returns_dataarray_with_correct_dims(self, fitted_mmm, request):
        """Test return type and dimensions."""
        mmm = request.getfixturevalue(fitted_mmm)
        curves = mmm.sample_saturation_curve(num_points=100)

        assert isinstance(curves, xr.DataArray)
        assert "sample" in curves.dims
        assert "x" in curves.dims
        assert curves.sizes["x"] == 100
        assert curves.sizes["sample"] > 0

    def test_num_points_controls_x_dimension(self, fitted_mmm, request):
        """Test that num_points parameter controls x dimension size."""
        mmm = request.getfixturevalue(fitted_mmm)
        curves = mmm.sample_saturation_curve(num_points=50)
        assert curves.sizes["x"] == 50

    def test_x_coordinate_range(self, fitted_mmm, request):
        """Test that x coordinates span from 0 to max_value."""
        mmm = request.getfixturevalue(fitted_mmm)
        max_value = 2.0
        curves = mmm.sample_saturation_curve(max_value=max_value, original_scale=False)

        x_coords = curves.coords["x"].values
        assert x_coords[0] == pytest.approx(0.0)
        assert np.max(x_coords) == pytest.approx(max_value)

    def test_curves_are_monotonic_increasing(self, fitted_mmm, request):
        """Test that saturation curves are monotonically increasing."""
        mmm = request.getfixturevalue(fitted_mmm)
        curves = mmm.sample_saturation_curve(num_points=100, original_scale=False)
        mean_curve = curves.mean(dim="sample")

        diffs = np.diff(mean_curve.values, axis=0)
        assert np.all(diffs >= -1e-6)  # Allow tiny numerical errors


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
class TestSampleSaturationCurveNumSamples:
    def test_num_samples_controls_sample_dimension(self, fitted_mmm, request):
        """Test that num_samples controls the number of posterior samples."""
        mmm = request.getfixturevalue(fitted_mmm)
        total_available = (
            mmm.idata.posterior.sizes["chain"] * mmm.idata.posterior.sizes["draw"]
        )
        num_samples = min(5, total_available - 1)

        curves = mmm.sample_saturation_curve(num_samples=num_samples)
        assert curves.sizes["sample"] == num_samples

    def test_uses_all_samples_when_num_samples_exceeds_or_none(
        self, fitted_mmm, request
    ):
        """Test behavior when num_samples exceeds available or is None."""
        mmm = request.getfixturevalue(fitted_mmm)
        total_available = (
            mmm.idata.posterior.sizes["chain"] * mmm.idata.posterior.sizes["draw"]
        )

        # When exceeds available
        curves_exceed = mmm.sample_saturation_curve(num_samples=10000)
        assert curves_exceed.sizes["sample"] == total_available

        # When None
        curves_none = mmm.sample_saturation_curve(num_samples=None)
        assert curves_none.sizes["sample"] == total_available


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
class TestSampleSaturationCurveRandomState:
    def test_reproducibility_and_different_seeds(self, fitted_mmm, request):
        """Test that same seed reproduces results and different seeds differ."""
        mmm = request.getfixturevalue(fitted_mmm)
        total_available = (
            mmm.idata.posterior.sizes["chain"] * mmm.idata.posterior.sizes["draw"]
        )
        num_samples = min(5, total_available - 1)

        if total_available <= 5:
            pytest.skip("Not enough posterior samples to test subsampling")

        # Same seed should reproduce
        curves1 = mmm.sample_saturation_curve(num_samples=num_samples, random_state=42)
        curves2 = mmm.sample_saturation_curve(num_samples=num_samples, random_state=42)
        xr.testing.assert_equal(curves1, curves2)

        # Different seeds should differ
        curves3 = mmm.sample_saturation_curve(num_samples=num_samples, random_state=123)
        assert not np.allclose(curves1.values, curves3.values)

        # Numpy Generator should also work
        rng = np.random.default_rng(42)
        curves_rng = mmm.sample_saturation_curve(
            num_samples=num_samples, random_state=rng
        )
        assert curves_rng.sizes["sample"] == num_samples


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
class TestSampleSaturationCurveOriginalScale:
    def test_original_scale_affects_values_not_x_coords(self, fitted_mmm, request):
        """Test that original_scale affects y values but not x coordinates."""
        mmm = request.getfixturevalue(fitted_mmm)
        max_value = 1.0

        curves_scaled = mmm.sample_saturation_curve(
            max_value=max_value, original_scale=False
        )
        curves_original = mmm.sample_saturation_curve(
            max_value=max_value, original_scale=True
        )

        # X coordinates should be identical
        np.testing.assert_array_equal(
            curves_scaled.coords["x"].values, curves_original.coords["x"].values
        )

        # Y values should differ
        assert not np.allclose(curves_original.values, curves_scaled.values)

        # Original scale should have larger magnitude (scaled by target_scale)
        mean_scaled = curves_scaled.mean(dim="sample")
        mean_original = curves_original.mean(dim="sample")
        assert np.mean(np.abs(mean_original.values)) > np.mean(
            np.abs(mean_scaled.values)
        )


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
class TestSampleSaturationCurveValidation:
    @pytest.mark.parametrize(
        "param,invalid_values",
        [
            ("max_value", [0, -1]),
            ("num_points", [0, -1]),
            ("num_samples", [0, -1]),
        ],
    )
    def test_raises_on_invalid_parameters(
        self, fitted_mmm, request, param, invalid_values
    ):
        """Test that invalid parameters raise ValidationError."""
        mmm = request.getfixturevalue(fitted_mmm)
        for invalid_value in invalid_values:
            with pytest.raises(ValidationError):
                mmm.sample_saturation_curve(**{param: invalid_value})


class TestSampleSaturationCurveErrorCases:
    def test_raises_on_unfitted_model(self):
        """Test that calling on unfitted model raises ValueError."""
        mmm = MMM(
            channel_columns=["channel_1", "channel_2"],
            date_column="date",
            target_column="target",
            adstock=GeometricAdstock(l_max=10),
            saturation=LogisticSaturation(),
        )

        with pytest.raises(ValueError, match="idata does not exist"):
            mmm.sample_saturation_curve()

    def test_raises_when_no_posterior(self, simple_mmm_data):
        """Test that calling raises ValueError when idata has no posterior."""
        import arviz as az

        X = simple_mmm_data["X"]
        y = simple_mmm_data["y"]

        mmm = MMM(
            channel_columns=["channel_1", "channel_2", "channel_3"],
            date_column="date",
            target_column="target",
            adstock=GeometricAdstock(l_max=10),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X, y)
        mmm.idata = az.InferenceData()  # Empty idata without posterior

        with pytest.raises(ValueError, match="posterior not found in idata"):
            mmm.sample_saturation_curve()


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
class TestSampleSaturationCurveEdgeCases:
    def test_handles_very_large_max_value(self, fitted_mmm, request):
        """Test that method handles very large max_value without overflow."""
        mmm = request.getfixturevalue(fitted_mmm)
        curves = mmm.sample_saturation_curve(max_value=1e6, original_scale=False)

        assert not np.any(np.isnan(curves.values))
        assert not np.any(np.isinf(curves.values))

    def test_idata_argument_uses_provided_posterior(self, fitted_mmm, request):
        """Test that idata argument uses provided posterior samples."""
        mmm = request.getfixturevalue(fitted_mmm)

        modified_idata = mmm.idata.copy()
        modified_posterior = modified_idata.posterior.copy()
        if "saturation_lam" in modified_posterior:
            modified_posterior["saturation_lam"] = (
                modified_posterior["saturation_lam"] * 2
            )
        if "saturation_beta" in modified_posterior:
            modified_posterior["saturation_beta"] = (
                modified_posterior["saturation_beta"] * 2
            )
        modified_idata.posterior = modified_posterior

        curves_default = mmm.sample_saturation_curve(random_state=42)
        curves_with_idata = mmm.sample_saturation_curve(
            idata=modified_idata, random_state=42
        )

        assert not np.allclose(curves_default.values, curves_with_idata.values)

    def test_can_be_used_for_plotting_operations(self, fitted_mmm, request):
        """Test that returned array supports common plotting operations."""
        mmm = request.getfixturevalue(fitted_mmm)
        curves = mmm.sample_saturation_curve()

        # These operations should work without error
        mean_curves = curves.mean(dim="sample")
        assert isinstance(mean_curves, xr.DataArray)

        lower = curves.quantile(0.05, dim="sample")
        upper = curves.quantile(0.95, dim="sample")
        assert isinstance(lower, xr.DataArray)
        assert isinstance(upper, xr.DataArray)

        x_values = curves.coords["x"].values
        assert len(x_values) > 0
        assert x_values[0] == 0.0
