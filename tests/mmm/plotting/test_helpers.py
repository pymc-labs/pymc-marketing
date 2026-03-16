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

from pymc_marketing.mmm.plotting._helpers import _validate_dims, channel_color_map


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


class TestChannelColorMap:
    def test_returns_dict_with_correct_keys(self):
        result = channel_color_map(["tv", "radio", "social"])
        assert set(result.keys()) == {"tv", "radio", "social"}

    def test_assigns_distinct_colors(self):
        result = channel_color_map(["tv", "radio", "social"])
        colors = list(result.values())
        assert len(set(colors)) == 3

    def test_deterministic_ordering(self):
        channels = ["tv", "radio", "social"]
        result1 = channel_color_map(channels)
        result2 = channel_color_map(channels)
        assert result1 == result2

    def test_single_channel(self):
        result = channel_color_map(["tv"])
        assert "tv" in result
        assert len(result) == 1

    def test_wraps_after_cycle_length(self):
        many_channels = [f"ch_{i}" for i in range(15)]
        result = channel_color_map(many_channels)
        assert len(result) == 15
        assert result["ch_0"] == result["ch_10"]

    def test_empty_channels(self):
        result = channel_color_map([])
        assert result == {}

    def test_preserves_channel_order(self):
        channels = ["social", "tv", "radio"]
        result = channel_color_map(channels)
        assert list(result.keys()) == ["social", "tv", "radio"]
