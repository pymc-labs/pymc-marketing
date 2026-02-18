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
"""Test for GitHub issue #1749.

The `sample_prior` method must succeed when a transformation is initialized
with constant tensor parameters instead of probability distributions.
"""

import numpy as np
import pytensor.tensor as pt
import xarray as xr

from pymc_marketing.mmm import GeometricAdstock


def test_geometric_adstock_sample_prior_with_constant_tensor() -> None:
    """Test that sample_prior works with constant tensor parameters.

    Regression test for GitHub issue #1749. When GeometricAdstock is
    initialized with a constant pt.as_tensor_variable (not a distribution),
    sample_prior should run without error and return the expected shape.
    """
    # Initialize with constant tensor (not a distribution)
    alpha = pt.as_tensor_variable([0.5, 0.3, 0.2])
    adstock = GeometricAdstock(l_max=4, priors={"alpha": alpha})
    coords = {"channel": ["A", "B", "C"]}

    # This should not raise an error
    prior = adstock.sample_prior(coords=coords)

    # Assert it returns the expected structure
    assert isinstance(prior, xr.Dataset)
    assert "adstock_alpha" in prior.data_vars
    assert prior.sizes["chain"] == 1
    assert prior.sizes["draw"] >= 1
    # Expected shape: (chain=1, draw=N, channel=3)
    assert prior["adstock_alpha"].shape == (1, prior.sizes["draw"], 3)
    # Each draw should contain the same constant values
    np.testing.assert_allclose(prior["adstock_alpha"].values[0, 0], [0.5, 0.3, 0.2])
