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
"""Tests for pymc_marketing.mmm.plotting._helpers."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pymc_marketing.mmm.plotting._helpers import _validate_dims


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"y": (["date", "channel", "geo"], np.random.randn(10, 3, 2))},
        coords={
            "date": range(10),
            "channel": ["tv", "radio", "social"],
            "geo": ["US", "UK"],
        },
    )


class TestValidateDims:
    def test_none_dims_is_noop(self, sample_dataset):
        _validate_dims(sample_dataset, None)

    def test_empty_dims_is_noop(self, sample_dataset):
        _validate_dims(sample_dataset, {})

    def test_valid_single_value(self, sample_dataset):
        _validate_dims(sample_dataset, {"channel": "tv"})

    def test_valid_list_value(self, sample_dataset):
        _validate_dims(sample_dataset, {"channel": ["tv", "radio"]})

    def test_valid_multiple_dims(self, sample_dataset):
        _validate_dims(sample_dataset, {"channel": ["tv"], "geo": "US"})

    def test_invalid_dim_name_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="Dimension 'region' not found"):
            _validate_dims(sample_dataset, {"region": "US"})

    def test_invalid_dim_value_scalar_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="Value 'FR' not found"):
            _validate_dims(sample_dataset, {"geo": "FR"})

    def test_invalid_dim_value_in_list_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="Value 'FR' not found"):
            _validate_dims(sample_dataset, {"geo": ["US", "FR"]})

    def test_numpy_array_values(self, sample_dataset):
        _validate_dims(sample_dataset, {"channel": np.array(["tv", "radio"])})

    def test_validates_against_provided_dataset_not_posterior(self):
        """Ensure validation uses the dataset passed in, not some hardcoded group."""
        ds_with_different_coords = xr.Dataset(
            {"x": (["fold"], [1, 2, 3])},
            coords={"fold": [1, 2, 3]},
        )
        _validate_dims(ds_with_different_coords, {"fold": 2})
        with pytest.raises(ValueError, match="Dimension 'channel' not found"):
            _validate_dims(ds_with_different_coords, {"channel": "tv"})
