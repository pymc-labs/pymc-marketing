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
class TestSampleAdstockCurveBasics:
    """Basic functionality tests for sample_adstock_curve."""

    def test_returns_dataarray_with_correct_dims(self, fitted_mmm, request):
        """Test return type and dimensions."""
        mmm = request.getfixturevalue(fitted_mmm)
        curves = mmm.sample_adstock_curve()

        assert isinstance(curves, xr.DataArray)
        assert "sample" in curves.dims
        assert "time since exposure" in curves.dims
        assert curves.sizes["time since exposure"] > 0
        assert curves.sizes["sample"] > 0

    def test_time_coordinate_range(self, fitted_mmm, request):
        """Test that time coordinates span from 0 to l_max-1."""
        mmm = request.getfixturevalue(fitted_mmm)
        curves = mmm.sample_adstock_curve()

        time_coords = curves.coords["time since exposure"].values
        assert time_coords[0] == pytest.approx(0.0)
        assert curves.sizes["time since exposure"] == mmm.adstock.l_max
        assert np.max(time_coords) == pytest.approx(mmm.adstock.l_max - 1)


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
class TestSampleAdstockCurveNumSamples:
    """Tests for num_samples parameter behavior."""

    def test_num_samples_controls_sample_dimension(self, fitted_mmm, request):
        """Test that num_samples controls the number of posterior samples."""
        mmm = request.getfixturevalue(fitted_mmm)
        total_available = (
            mmm.idata.posterior.sizes["chain"] * mmm.idata.posterior.sizes["draw"]
        )
        num_samples = min(5, total_available - 1)

        if total_available <= 2:
            pytest.skip("Not enough posterior samples to test subsampling")

        curves = mmm.sample_adstock_curve(num_samples=num_samples)
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
        curves_exceed = mmm.sample_adstock_curve(num_samples=10000)
        assert curves_exceed.sizes["sample"] == total_available

        # When None
        curves_none = mmm.sample_adstock_curve(num_samples=None)
        assert curves_none.sizes["sample"] == total_available


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
class TestSampleAdstockCurveValidation:
    """Tests for input validation."""

    @pytest.mark.parametrize(
        "param,invalid_values",
        [
            ("amount", [0, -1]),
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
                mmm.sample_adstock_curve(**{param: invalid_value})


class TestSampleAdstockCurveErrorCases:
    """Tests for error conditions."""

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
            mmm.sample_adstock_curve()

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
            mmm.sample_adstock_curve()


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
class TestSampleAdstockCurveEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_idata_argument_uses_provided_posterior(self, fitted_mmm, request):
        """Test that idata argument uses provided posterior samples."""
        mmm = request.getfixturevalue(fitted_mmm)

        modified_idata = mmm.idata.copy()
        modified_posterior = modified_idata.posterior.copy()
        if "adstock_alpha" in modified_posterior:
            modified_posterior["adstock_alpha"] = np.clip(
                modified_posterior["adstock_alpha"] * 0.5, 0.01, 0.99
            )
        modified_idata.posterior = modified_posterior

        curves_default = mmm.sample_adstock_curve(random_state=42)
        curves_with_idata = mmm.sample_adstock_curve(
            idata=modified_idata, random_state=42
        )

        assert not np.allclose(curves_default.values, curves_with_idata.values), (
            "Curves with modified idata should differ from default"
        )
