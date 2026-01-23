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


@pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
def test_sample_adstock_curve_basic_functionality(fitted_mmm, request):
    """Test basic functionality: return type, dimensions, and coordinate range."""
    mmm = request.getfixturevalue(fitted_mmm)

    curves = mmm.sample_adstock_curve()

    # Return type and required dimensions
    assert isinstance(curves, xr.DataArray)
    assert "sample" in curves.dims
    assert "time since exposure" in curves.dims
    assert curves.sizes["time since exposure"] > 0
    assert curves.sizes["sample"] > 0

    # Time coordinate range: 0 to l_max-1
    time_coords = curves.coords["time since exposure"].values
    assert time_coords[0] == pytest.approx(0.0)
    assert curves.sizes["time since exposure"] == mmm.adstock.l_max
    assert np.max(time_coords) == pytest.approx(mmm.adstock.l_max - 1)


@pytest.mark.parametrize(
    "num_samples_mode",
    ["less_than_total", "greater_than_total", "none"],
)
def test_sample_adstock_curve_num_samples_behavior(simple_fitted_mmm, num_samples_mode):
    """Test num_samples parameter behavior for different scenarios."""
    mmm = simple_fitted_mmm
    total_available = (
        mmm.idata.posterior.sizes["chain"] * mmm.idata.posterior.sizes["draw"]
    )

    if num_samples_mode == "less_than_total":
        if total_available <= 2:
            pytest.skip("Not enough posterior samples to test subsampling")
        num_samples = min(5, total_available - 1)
        expected_samples = num_samples
    elif num_samples_mode == "greater_than_total":
        num_samples = 10000
        expected_samples = total_available
    else:  # none
        num_samples = None
        expected_samples = total_available

    curves = mmm.sample_adstock_curve(num_samples=num_samples)
    assert curves.sizes["sample"] == expected_samples


@pytest.mark.parametrize("amount", [0, -1])
def test_sample_adstock_curve_raises_on_invalid_amount(simple_fitted_mmm, amount):
    """Test that invalid amount raises ValidationError."""
    with pytest.raises(ValidationError):
        simple_fitted_mmm.sample_adstock_curve(amount=amount)


@pytest.mark.parametrize("num_samples", [0, -1])
def test_sample_adstock_curve_raises_on_invalid_num_samples(
    simple_fitted_mmm, num_samples
):
    """Test that invalid num_samples raises ValidationError."""
    with pytest.raises(ValidationError):
        simple_fitted_mmm.sample_adstock_curve(num_samples=num_samples)


def test_sample_adstock_curve_raises_on_unfitted_model():
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


def test_sample_adstock_curve_raises_when_no_posterior(simple_mmm_data):
    """Test that calling raises ValueError when idata exists but has no posterior."""
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
    mmm.idata = az.InferenceData()

    with pytest.raises(ValueError, match="posterior not found in idata"):
        mmm.sample_adstock_curve()


def test_sample_adstock_curve_with_idata_argument(simple_fitted_mmm):
    """Test that idata argument uses provided InferenceData instead of self.idata."""
    mmm = simple_fitted_mmm

    modified_idata = mmm.idata.copy()
    modified_posterior = modified_idata.posterior.copy()
    if "adstock_alpha" in modified_posterior:
        modified_posterior["adstock_alpha"] = np.clip(
            modified_posterior["adstock_alpha"] * 0.5, 0.01, 0.99
        )
    modified_idata.posterior = modified_posterior

    curves_default = mmm.sample_adstock_curve(random_state=42)
    curves_with_idata = mmm.sample_adstock_curve(idata=modified_idata, random_state=42)

    assert not np.allclose(curves_default.values, curves_with_idata.values), (
        "Curves with modified idata should differ from default"
    )
