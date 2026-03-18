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

import warnings

import matplotlib
import numpy as np
import pytest
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pymc_marketing.mmm.plotting._helpers import (
    _dims_to_sel_kwargs,
    _extract_matplotlib_result,
    _process_plot_params,
    _select_dims,
    _validate_dims,
    channel_color_map,
)

matplotlib.use("Agg")


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


class TestProcessPlotParams:
    def test_no_args_returns_empty_dict(self):
        result = _process_plot_params(
            figsize=None,
            plot_collection=None,
            backend=None,
            return_as_pc=False,
        )
        assert result == {}

    def test_figsize_injected_into_figure_kwargs(self):
        result = _process_plot_params(
            figsize=(12, 6),
            plot_collection=None,
            backend=None,
            return_as_pc=False,
        )
        assert result == {"figure_kwargs": {"figsize": (12, 6)}}

    def test_figsize_overrides_existing_figure_kwargs(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _process_plot_params(
                figsize=(12, 6),
                plot_collection=None,
                backend=None,
                return_as_pc=False,
                figure_kwargs={"figsize": (8, 4), "dpi": 100},
            )
        assert result["figure_kwargs"]["figsize"] == (12, 6)
        assert result["figure_kwargs"]["dpi"] == 100
        assert len(w) == 1
        assert "overrides" in str(w[0].message).lower()

    def test_figsize_ignored_when_plot_collection_provided(self):
        pc = PlotCollection.wrap(
            xr.Dataset({"x": (["a"], [1, 2, 3])}),
            backend="matplotlib",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _process_plot_params(
                figsize=(12, 6),
                plot_collection=pc,
                backend=None,
                return_as_pc=False,
            )
        assert "figure_kwargs" not in result
        assert len(w) == 1
        assert "ignored" in str(w[0].message).lower()

    def test_non_matplotlib_backend_without_return_as_pc_raises(self):
        with pytest.raises(ValueError, match="return_as_pc=True"):
            _process_plot_params(
                figsize=None,
                plot_collection=None,
                backend="plotly",
                return_as_pc=False,
            )

    def test_non_matplotlib_backend_with_return_as_pc_ok(self):
        result = _process_plot_params(
            figsize=None,
            plot_collection=None,
            backend="plotly",
            return_as_pc=True,
        )
        assert result == {}

    def test_matplotlib_backend_explicit_ok(self):
        result = _process_plot_params(
            figsize=None,
            plot_collection=None,
            backend="matplotlib",
            return_as_pc=False,
        )
        assert result == {}

    def test_none_backend_ok(self):
        result = _process_plot_params(
            figsize=None,
            plot_collection=None,
            backend=None,
            return_as_pc=False,
        )
        assert result == {}

    def test_extra_pc_kwargs_forwarded(self):
        result = _process_plot_params(
            figsize=None,
            plot_collection=None,
            backend=None,
            return_as_pc=False,
            col_wrap=3,
        )
        assert result == {"col_wrap": 3}

    def test_figsize_merged_with_extra_kwargs(self):
        result = _process_plot_params(
            figsize=(10, 5),
            plot_collection=None,
            backend=None,
            return_as_pc=False,
            col_wrap=2,
        )
        assert result == {
            "figure_kwargs": {"figsize": (10, 5)},
            "col_wrap": 2,
        }


class TestExtractMatplotlibResult:
    @pytest.fixture
    def single_panel_pc(self):
        data = xr.Dataset({"x": (["a"], [1, 2, 3])})
        return PlotCollection.wrap(data, backend="matplotlib")

    @pytest.fixture
    def multi_panel_pc(self):
        data = xr.Dataset(
            {"x": (["a", "b"], np.random.randn(3, 2))},
            coords={"a": [0, 1, 2], "b": ["p", "q"]},
        )
        return PlotCollection.grid(data, cols=["b"], backend="matplotlib")

    def test_return_as_pc_true_returns_plot_collection(self, single_panel_pc):
        result = _extract_matplotlib_result(single_panel_pc, return_as_pc=True)
        assert isinstance(result, PlotCollection)
        assert result is single_panel_pc

    def test_return_as_pc_false_returns_tuple(self, single_panel_pc):
        result = _extract_matplotlib_result(single_panel_pc, return_as_pc=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        fig, axes = result
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(a, Axes) for a in axes.flat)

    def test_single_panel_wraps_in_ndarray(self, single_panel_pc):
        _, axes = _extract_matplotlib_result(single_panel_pc, return_as_pc=False)
        assert axes.shape == (1,)

    def test_multi_panel_returns_correct_count(self, multi_panel_pc):
        _, axes = _extract_matplotlib_result(multi_panel_pc, return_as_pc=False)
        assert axes.size == 2


class TestPublicAPI:
    def test_all_helpers_importable_from_helpers_module(self):
        from pymc_marketing.mmm.plotting._helpers import (
            _extract_matplotlib_result,
            _process_plot_params,
            _select_dims,
            _validate_dims,
            channel_color_map,
        )

        assert callable(_validate_dims)
        assert callable(_select_dims)
        assert callable(_process_plot_params)
        assert callable(_extract_matplotlib_result)
        assert callable(channel_color_map)

    def test_plotting_package_importable(self):
        import pymc_marketing.mmm.plotting  # noqa: F401


class TestDimsToSelKwargs:
    def test_none_returns_empty_dict(self):
        assert _dims_to_sel_kwargs(None) == {}

    def test_empty_returns_empty_dict(self):
        assert _dims_to_sel_kwargs({}) == {}

    def test_scalar_wrapped_in_list(self):
        result = _dims_to_sel_kwargs({"channel": "tv"})
        assert result == {"channel": ["tv"]}

    def test_list_preserved(self):
        result = _dims_to_sel_kwargs({"channel": ["tv", "radio"]})
        assert result == {"channel": ["tv", "radio"]}

    def test_tuple_preserved(self):
        result = _dims_to_sel_kwargs({"channel": ("tv", "radio")})
        assert result == {"channel": ("tv", "radio")}

    def test_numpy_array_preserved(self):
        arr = np.array(["tv", "radio"])
        result = _dims_to_sel_kwargs({"channel": arr})
        np.testing.assert_array_equal(result["channel"], arr)

    def test_multiple_dims_mixed(self):
        result = _dims_to_sel_kwargs({"channel": "tv", "geo": ["US", "UK"]})
        assert result["channel"] == ["tv"]
        assert result["geo"] == ["US", "UK"]

    def test_integer_scalar_wrapped(self):
        result = _dims_to_sel_kwargs({"fold": 2})
        assert result == {"fold": [2]}


class TestSelectDims:
    """Tests for the combined validate + select helper."""

    def test_none_dims_returns_unchanged(self, sample_dataset):
        result = _select_dims(sample_dataset, None)
        assert result is sample_dataset

    def test_empty_dims_returns_unchanged(self, sample_dataset):
        result = _select_dims(sample_dataset, {})
        assert result is sample_dataset

    def test_selects_valid_dims(self, sample_dataset):
        result = _select_dims(sample_dataset, {"channel": "tv"})
        assert list(result.coords["channel"].values) == ["tv"]
        assert "date" in result.dims

    def test_preserves_dim_as_size_one(self, sample_dataset):
        result = _select_dims(sample_dataset, {"geo": "US"})
        assert result.sizes["geo"] == 1

    def test_ignores_extra_dims(self, sample_dataset):
        result = _select_dims(sample_dataset, {"channel": "tv", "region": "US"})
        assert list(result.coords["channel"].values) == ["tv"]

    def test_all_extra_returns_unchanged(self, sample_dataset):
        result = _select_dims(sample_dataset, {"region": "US"})
        assert result is sample_dataset

    def test_validates_matching_values(self, sample_dataset):
        with pytest.raises(ValueError, match="Value 'FR' not found"):
            _select_dims(sample_dataset, {"geo": "FR"})

    def test_works_with_dataarray(self):
        da = xr.DataArray(
            np.random.randn(3, 2),
            dims=("channel", "geo"),
            coords={"channel": ["tv", "radio", "social"], "geo": ["US", "UK"]},
        )
        result = _select_dims(da, {"channel": "tv"})
        assert isinstance(result, xr.DataArray)
        assert list(result.coords["channel"].values) == ["tv"]

    def test_dataarray_ignores_extra_dims(self):
        da = xr.DataArray(
            np.random.randn(3),
            dims=("channel",),
            coords={"channel": ["tv", "radio", "social"]},
        )
        result = _select_dims(da, {"channel": "tv", "geo": "US"})
        assert list(result.coords["channel"].values) == ["tv"]

    def test_multiple_dims_selected(self, sample_dataset):
        result = _select_dims(sample_dataset, {"channel": "tv", "geo": "US"})
        assert result.sizes["channel"] == 1
        assert result.sizes["geo"] == 1
