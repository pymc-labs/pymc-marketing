#   Copyright 2022 - 2025 The PyMC Labs Developers
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.plot import (
    drop_scalar_coords,
    plot_curve,
    plot_hdi,
    plot_samples,
    random_samples,
    selections,
)


@pytest.mark.parametrize(
    "coords, expected",
    [
        ({}, [{}]),
        ({"channel": [1, 2, 3]}, [{"channel": 1}, {"channel": 2}, {"channel": 3}]),
        (
            {"channel": [1, 2], "country": ["A", "B"]},
            [
                {"channel": 1, "country": "A"},
                {"channel": 1, "country": "B"},
                {"channel": 2, "country": "A"},
                {"channel": 2, "country": "B"},
            ],
        ),
    ],
)
def test_selections(coords, expected) -> None:
    assert list(selections(coords)) == expected


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [
            range(1),
            range(10),
        ]
    )
    return pd.DataFrame(
        np.random.randn(10, 1),
        index=index,
        columns=["y"],
    )


def test_random_samples(sample_frame) -> None:
    rng = np.random.default_rng(0)
    n = 5
    idx = random_samples(rng, n=n, n_chains=1, n_draws=10)

    assert len(idx) == n
    assert idx == [
        (0, 4),
        (0, 7),
        (0, 2),
        (0, 3),
        (0, 5),
    ]
    df_sub = sample_frame.loc[idx, :]
    assert len(df_sub) == n


@pytest.fixture(scope="module")
def mock_curve() -> xr.DataArray:
    coords = {
        "chain": np.arange(1),
        "draw": np.arange(15),
        "geo": np.arange(5),
        "day": np.arange(31),
    }
    return xr.DataArray(
        np.ones((1, 15, 5, 31)),
        coords=coords,
    )


@pytest.mark.parametrize("plot_func", [plot_samples, plot_hdi])
@pytest.mark.parametrize(
    "same_axes", [True, False], ids=["same_axes", "different_axes"]
)
@pytest.mark.parametrize("legend", [True, False], ids=["legend", "no_legend"])
def test_plot_functions(mock_curve, plot_func, same_axes: bool, legend: bool) -> None:
    fig, axes = plot_func(
        mock_curve,
        non_grid_names={"day"},
        same_axes=same_axes,
        legend=legend,
    )

    assert axes.size == (1 if same_axes else mock_curve.sizes["geo"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.mark.parametrize(
    "non_grid_names",
    [pytest.param("day", id="string"), pytest.param({"day"}, id="set")],
)
def test_plot_curve_non_grid_names_type(mock_curve, non_grid_names) -> None:
    fig, axes = plot_curve(mock_curve, non_grid_names)

    assert axes.size == mock_curve.sizes["geo"]
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_curve_supply_axes_same_axes(mock_curve) -> None:
    _, ax = plt.subplots()
    axes = np.array([ax])

    fig, modified_axes = plot_curve(
        mock_curve,
        non_grid_names={"day"},
        axes=axes,
        same_axes=True,
    )

    np.testing.assert_equal(axes, modified_axes)
    plt.close(fig)


def test_plot_curve_custom_colors(mock_curve) -> None:
    colors = ["red", "blue", "green", "yellow", "purple"]

    fig, axes = plot_curve(mock_curve, non_grid_names={"day"}, colors=colors)

    for ax, color in zip(axes, colors, strict=True):
        for line in ax.get_lines():
            assert line.get_color() == color

    plt.close(fig)


def test_plot_curve_custom_sel_to_string(mock_curve) -> None:
    def custom_sel_to_string(sel):
        return ", ".join(f"{key}: {value}" for key, value in sel.items())

    fig, axes = plot_curve(
        mock_curve,
        non_grid_names={"day"},
        sel_to_string=custom_sel_to_string,
    )

    titles = [ax.get_title() for ax in axes]

    assert titles == [
        "geo: 0",
        "geo: 1",
        "geo: 2",
        "geo: 3",
        "geo: 4",
    ]

    plt.close(fig)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"n_samples": 3}, id="n_samples"),
        pytest.param({"hdi_probs": [0.5, 0.9]}, id="hdi_probs"),
    ],
)
def test_plot_curve_exposed_parameters(mock_curve, kwargs) -> None:
    fig, axes = plot_curve(
        mock_curve,
        non_grid_names={"day"},
        **kwargs,
    )

    assert axes.size == mock_curve.sizes["geo"]
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.fixture
def mock_curve_with_scalars() -> xr.DataArray:
    coords = {
        "x": [1, 2, 3],
        "y": [10, 20, 30],
        "scalar1": 42,  # Scalar coordinate
        "scalar2": 3.14,  # Another scalar coordinate
    }
    data = np.random.rand(3, 3)
    return xr.DataArray(data, coords=coords, dims=["x", "y"])


def test_drop_scalar_coords(mock_curve_with_scalars) -> None:
    original_curve = mock_curve_with_scalars.copy(deep=True)  # Make a deep copy
    curve = drop_scalar_coords(mock_curve_with_scalars)

    # Ensure scalar coordinates are removed
    assert "scalar1" not in curve.coords
    assert "scalar2" not in curve.coords

    # Ensure other coordinates are still present
    assert "x" in curve.coords
    assert "y" in curve.coords

    # Ensure data shape is unchanged
    assert curve.shape == (3, 3)

    # Ensure the original DataArray was not modified
    xr.testing.assert_identical(mock_curve_with_scalars, original_curve)
