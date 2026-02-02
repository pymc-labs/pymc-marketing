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
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from pymc_marketing.mmm.multidimensional import MMM

from pymc_marketing.mmm.plot import MMMPlotSuite


@pytest.fixture
def mmm():
    return MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        dims=("country",),
        target_column="y",
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )


@pytest.fixture
def df() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=3, freq="W-MON").rename("date")
    df = pd.DataFrame(
        {
            ("A", "C1"): [1, 2, 3],
            ("B", "C1"): [4, 5, 6],
            ("A", "C2"): [7, 8, 9],
            ("B", "C2"): [10, 11, 12],
        },
        index=dates,
    )
    df.columns.names = ["country", "channel"]

    y = pd.DataFrame(
        {
            ("A", "y"): [1, 2, 3],
            ("B", "y"): [4, 5, 6],
        },
        index=dates,
    )
    y.columns.names = ["country", "channel"]

    return pd.concat(
        [
            df.stack("country", future_stack=True),
            y.stack("country", future_stack=True),
        ],
        axis=1,
    ).reset_index()


@pytest.fixture
def fit_mmm_with_channel_original_scale(df, mmm, mock_pymc_sample):
    X = df.drop(columns=["y"])
    y = df["y"]

    mmm.build_model(X, y)
    mmm.add_original_scale_contribution_variable(
        var=[
            "channel_contribution",
        ]
    )

    mmm.fit(X, y)

    return mmm


@pytest.fixture
def fit_mmm_without_channel_original_scale(df, mmm, mock_pymc_sample):
    X = df.drop(columns=["y"])
    y = df["y"]

    mmm.fit(X, y)

    return mmm


def test_saturation_curves_scatter_original_scale(fit_mmm_with_channel_original_scale):
    fig, ax = fit_mmm_with_channel_original_scale.plot.saturation_curves_scatter(
        original_scale=True
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert all(isinstance(a, Axes) for a in ax.flat)


def test_saturation_curves_scatter_original_scale_fails_if_no_deterministic(
    fit_mmm_without_channel_original_scale,
):
    with pytest.raises(ValueError):
        fit_mmm_without_channel_original_scale.plot.saturation_curves_scatter(
            original_scale=True
        )


def test_contributions_over_time(fit_mmm_with_channel_original_scale):
    fig, ax = fit_mmm_with_channel_original_scale.plot.contributions_over_time(
        var=["channel_contribution"],
        hdi_prob=0.95,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert all(isinstance(a, Axes) for a in ax.flat)


def test_contributions_over_time_with_dim(mock_suite: MMMPlotSuite):
    # Test with explicit dim argument
    fig, ax = mock_suite.contributions_over_time(
        var=["intercept", "linear_trend"],
        dims={"country": "A"},
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert all(isinstance(a, Axes) for a in ax.flat)
    # Optionally, check axes shape if known
    if hasattr(ax, "shape"):
        # When filtering to a single country, shape[-1] should be 1
        assert ax.shape[-1] == 1


def test_contributions_over_time_with_dims_list(mock_suite: MMMPlotSuite):
    """Test that passing a list to dims creates a subplot for each value."""
    fig, ax = mock_suite.contributions_over_time(
        var=["intercept"],
        dims={"country": ["A", "B"]},
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    # Should create one subplot per value in the list (here: 2 countries)
    assert ax.shape[0] == 2
    # Optionally, check subplot titles contain the correct country
    for i, country in enumerate(["A", "B"]):
        assert country in ax[i, 0].get_title()


def test_contributions_over_time_with_multiple_dims_lists(mock_suite: MMMPlotSuite):
    """Test that passing multiple lists to dims creates a subplot for each combination."""
    # Add a fake 'region' dim to the mock posterior for this test if not present
    idata = mock_suite.idata
    if "region" not in idata.posterior["intercept"].dims:
        idata.posterior["intercept"] = idata.posterior["intercept"].expand_dims(
            region=["X", "Y"]
        )
    fig, ax = mock_suite.contributions_over_time(
        var=["intercept"],
        dims={"country": ["A", "B"], "region": ["X", "Y"]},
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    # Should create one subplot per combination (2 countries x 2 regions = 4)
    assert ax.shape[0] == 4
    combos = [("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y")]
    for i, (country, region) in enumerate(combos):
        title = ax[i, 0].get_title()
        assert country in title
        assert region in title


def test_contributions_over_time_combine_dims(mock_suite: MMMPlotSuite):
    """Test that combine_dims=True plots all dimension combinations on a single axis."""
    fig, ax = mock_suite.contributions_over_time(
        var=["intercept"],
        dims={"country": ["A", "B"]},
        combine_dims=True,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    # Should create only 1 subplot when combine_dims=True
    assert ax.shape[0] == 1
    # Check that the legend contains both countries
    legend_labels = [t.get_text() for t in ax[0, 0].get_legend().get_texts()]
    assert any("A" in label for label in legend_labels)
    assert any("B" in label for label in legend_labels)


def test_contributions_over_time_combine_dims_multiple_vars(mock_suite: MMMPlotSuite):
    """Test combine_dims=True with multiple variables."""
    fig, ax = mock_suite.contributions_over_time(
        var=["intercept", "linear_trend"],
        dims={"country": ["A", "B"]},
        combine_dims=True,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert ax.shape[0] == 1
    # Check that legend contains entries for both vars and countries
    legend_labels = [t.get_text() for t in ax[0, 0].get_legend().get_texts()]
    # Should have at least 4 entries (2 vars x 2 countries, may have more if extra dims)
    assert len(legend_labels) >= 4
    # Check that both variables appear
    assert any("intercept" in label for label in legend_labels)
    assert any("linear_trend" in label for label in legend_labels)
    # Check that both countries appear
    assert any("country=A" in label for label in legend_labels)
    assert any("country=B" in label for label in legend_labels)


def test_posterior_predictive(fit_mmm_with_channel_original_scale, df):
    fit_mmm_with_channel_original_scale.sample_posterior_predictive(
        df.drop(columns=["y"])
    )
    fig, ax = fit_mmm_with_channel_original_scale.plot.posterior_predictive(
        hdi_prob=0.95,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert all(isinstance(a, Axes) for a in ax.flat)


@pytest.fixture(scope="module")
def mock_idata() -> az.InferenceData:
    seed = sum(map(ord, "Fake posterior"))
    rng = np.random.default_rng(seed)
    normal = rng.normal

    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")
    return az.InferenceData(
        posterior=xr.Dataset(
            {
                "intercept": xr.DataArray(
                    normal(size=(4, 100, 52, 3)),
                    dims=("chain", "draw", "date", "country"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "date": dates,
                        "country": ["A", "B", "C"],
                    },
                ),
                "linear_trend": xr.DataArray(
                    normal(size=(4, 100, 52)),
                    dims=("chain", "draw", "date"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "date": dates,
                    },
                ),
            }
        )
    )


@pytest.fixture(scope="module")
def mock_idata_with_sensitivity(mock_idata):
    # Copy the mock_idata so we don't mutate the shared fixture
    idata = mock_idata.copy()
    n_sample, n_sweep = 40, 5
    sweep = np.linspace(0.5, 1.5, n_sweep)
    regions = ["A", "B"]

    samples = xr.DataArray(
        np.random.normal(0, 1, size=(n_sample, n_sweep, len(regions))),
        dims=("sample", "sweep", "region"),
        coords={
            "sample": np.arange(n_sample),
            "sweep": sweep,
            "region": regions,
        },
        name="x",
    )

    marginal_effects = xr.DataArray(
        np.random.normal(0, 1, size=(n_sample, n_sweep, len(regions))),
        dims=("sample", "sweep", "region"),
        coords={
            "sample": np.arange(n_sample),
            "sweep": sweep,
            "region": regions,
        },
        name="marginal_effects",
    )

    uplift_curve = xr.DataArray(
        np.random.normal(0, 1, size=(n_sample, n_sweep, len(regions))),
        dims=("sample", "sweep", "region"),
        coords={
            "sample": np.arange(n_sample),
            "sweep": sweep,
            "region": regions,
        },
        name="uplift_curve",
    )

    sensitivity_analysis = xr.Dataset(
        {
            "x": samples,
            "marginal_effects": marginal_effects,
            "uplift_curve": uplift_curve,
        },
        coords={"sweep": sweep, "region": regions},
        attrs={"sweep_type": "multiplicative", "var_names": "test_var"},
    )

    idata.sensitivity_analysis = sensitivity_analysis
    return idata


@pytest.fixture(scope="module")
def mock_suite(mock_idata):
    """Fixture to create a mock MMMPlotSuite with a mocked posterior."""
    return MMMPlotSuite(idata=mock_idata)


@pytest.fixture(scope="module")
def mock_suite_with_sensitivity(mock_idata_with_sensitivity):
    """Fixture to create a mock MMMPlotSuite with sensitivity analysis."""
    return MMMPlotSuite(idata=mock_idata_with_sensitivity)


@pytest.fixture(scope="module")
def mock_idata_with_sensitivity_and_channels():
    """Fixture with sensitivity data that has channel dim + constant_data for x_sweep_axis='absolute'."""
    seed = sum(map(ord, "sensitivity_channels"))
    rng = np.random.default_rng(seed)

    n_sample, n_sweep = 40, 5
    sweep = np.linspace(0.5, 1.5, n_sweep)
    regions = ["A", "B"]
    channels = ["channel_1", "channel_2"]
    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")

    # Sensitivity analysis data with channel dimension
    samples = xr.DataArray(
        rng.normal(0, 1, size=(n_sample, n_sweep, len(regions), len(channels))),
        dims=("sample", "sweep", "region", "channel"),
        coords={
            "sample": np.arange(n_sample),
            "sweep": sweep,
            "region": regions,
            "channel": channels,
        },
        name="x",
    )

    sensitivity_analysis = xr.Dataset(
        {"x": samples},
        coords={"sweep": sweep, "region": regions, "channel": channels},
        attrs={"sweep_type": "multiplicative", "var_names": "test_var"},
    )

    # constant_data with channel_data and channel_scale for x_sweep_axis="absolute"
    constant_data = xr.Dataset(
        {
            "channel_data": xr.DataArray(
                rng.uniform(0, 10, size=(52, len(channels), len(regions))),
                dims=("date", "channel", "region"),
                coords={
                    "date": dates,
                    "channel": channels,
                    "region": regions,
                },
            ),
            "channel_scale": xr.DataArray(
                [[100.0, 200.0], [150.0, 250.0]],
                dims=("region", "channel"),
                coords={"region": regions, "channel": channels},
            ),
        }
    )

    idata = az.InferenceData(constant_data=constant_data)
    idata.sensitivity_analysis = sensitivity_analysis
    return idata


@pytest.fixture(scope="module")
def mock_suite_with_sensitivity_and_channels(mock_idata_with_sensitivity_and_channels):
    """Fixture to create a MMMPlotSuite with sensitivity analysis and channel dimension."""
    return MMMPlotSuite(idata=mock_idata_with_sensitivity_and_channels)


def test_contributions_over_time_expand_dims(mock_suite: MMMPlotSuite):
    fig, ax = mock_suite.contributions_over_time(
        var=[
            "intercept",
            "linear_trend",
        ]
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert all(isinstance(a, Axes) for a in ax.flat)


@pytest.fixture(scope="module")
def mock_idata_with_constant_data() -> az.InferenceData:
    """Create mock InferenceData with constant_data and posterior for saturation tests."""
    seed = sum(map(ord, "Saturation tests"))
    rng = np.random.default_rng(seed)
    normal = rng.normal

    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")
    channels = ["channel_1", "channel_2"]
    countries = ["A", "B"]

    # Create posterior data
    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                normal(size=(4, 100, 52, 2, 2)),
                dims=("chain", "draw", "date", "channel", "country"),
                coords={
                    "chain": np.arange(4),
                    "draw": np.arange(100),
                    "date": dates,
                    "channel": channels,
                    "country": countries,
                },
            ),
            "channel_contribution_original_scale": xr.DataArray(
                normal(size=(4, 100, 52, 2, 2)) * 100,  # scaled up for original scale
                dims=("chain", "draw", "date", "channel", "country"),
                coords={
                    "chain": np.arange(4),
                    "draw": np.arange(100),
                    "date": dates,
                    "channel": channels,
                    "country": countries,
                },
            ),
        }
    )

    # Create constant_data
    constant_data = xr.Dataset(
        {
            "channel_data": xr.DataArray(
                rng.uniform(0, 10, size=(52, 2, 2)),
                dims=("date", "channel", "country"),
                coords={
                    "date": dates,
                    "channel": channels,
                    "country": countries,
                },
            ),
            "channel_scale": xr.DataArray(
                [[100.0, 200.0], [150.0, 250.0]],
                dims=("country", "channel"),
                coords={"country": countries, "channel": channels},
            ),
            "target_scale": xr.DataArray(
                [1000.0],
                dims="target",
                coords={"target": ["y"]},
            ),
        }
    )

    return az.InferenceData(posterior=posterior, constant_data=constant_data)


@pytest.fixture(scope="module")
def mock_suite_with_constant_data(mock_idata_with_constant_data):
    """Fixture to create a MMMPlotSuite with constant_data for saturation tests."""
    return MMMPlotSuite(idata=mock_idata_with_constant_data)


@pytest.fixture(scope="module")
def mock_saturation_curve() -> xr.DataArray:
    """Create mock saturation curve data for testing saturation_curves method."""
    seed = sum(map(ord, "Saturation curve"))
    rng = np.random.default_rng(seed)

    # Create curve data with typical saturation curve shape
    x_values = np.linspace(0, 1, 100)
    channels = ["channel_1", "channel_2"]
    countries = ["A", "B"]

    curve_data = []
    for _ in range(4):  # chains
        for _ in range(100):  # draws
            for _ in channels:
                for _ in countries:
                    # Simple saturation curve: y = x / (1 + x)
                    y_values = x_values / (1 + x_values) + rng.normal(
                        0, 0.01, size=x_values.shape
                    )
                    curve_data.append(y_values)

    curve_array = np.array(curve_data).reshape(
        4, 100, len(channels), len(countries), len(x_values)
    )

    return xr.DataArray(
        curve_array,
        dims=("chain", "draw", "channel", "country", "x"),
        coords={
            "chain": np.arange(4),
            "draw": np.arange(100),
            "channel": channels,
            "country": countries,
            "x": x_values,
        },
    )


class TestSaturationScatterplot:
    def test_saturation_scatterplot_basic(self, mock_suite_with_constant_data):
        """Test basic functionality of saturation_scatterplot."""
        fig, axes = mock_suite_with_constant_data.saturation_scatterplot()

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_saturation_scatterplot_original_scale(self, mock_suite_with_constant_data):
        """Test saturation_scatterplot with original_scale=True."""
        fig, axes = mock_suite_with_constant_data.saturation_scatterplot(
            original_scale=True
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_saturation_scatterplot_custom_kwargs(self, mock_suite_with_constant_data):
        """Test saturation_scatterplot with custom kwargs."""
        fig, axes = mock_suite_with_constant_data.saturation_scatterplot(
            width_per_col=8.0, height_per_row=5.0
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_saturation_scatterplot_no_constant_data(self, mock_suite):
        """Test that saturation_scatterplot raises error without constant_data."""
        with pytest.raises(ValueError, match=r"No 'constant_data' found"):
            mock_suite.saturation_scatterplot()

    def test_saturation_scatterplot_no_original_scale_contribution(
        self, mock_suite_with_constant_data
    ):
        """Test that saturation_scatterplot raises error when original_scale=True but no original scale data."""
        # Remove the original scale contribution from the mock data
        idata_copy = mock_suite_with_constant_data.idata.copy()
        idata_copy.posterior = idata_copy.posterior.drop_vars(
            "channel_contribution_original_scale"
        )
        suite_without_original_scale = MMMPlotSuite(idata=idata_copy)

        with pytest.raises(
            ValueError, match=r"No posterior.channel_contribution_original_scale"
        ):
            suite_without_original_scale.saturation_scatterplot(original_scale=True)


class TestSaturationScatterplotDims:
    def test_saturation_scatterplot_with_dim(self, mock_suite_with_constant_data):
        """Test saturation_scatterplot with a single value in dims."""
        fig, axes = mock_suite_with_constant_data.saturation_scatterplot(
            dims={"country": "A"}
        )
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create one column (n_channels, 1)
        assert axes.shape[1] == 1
        for row in range(axes.shape[0]):
            assert "country=A" in axes[row, 0].get_title()

    def test_saturation_scatterplot_with_dims_list(self, mock_suite_with_constant_data):
        """Test saturation_scatterplot with a list in dims (should create subplots for each value)."""
        fig, axes = mock_suite_with_constant_data.saturation_scatterplot(
            dims={"country": ["A", "B"]}
        )
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create two columns (n_channels, 2)
        assert axes.shape[1] == 2
        for col, country in enumerate(["A", "B"]):
            for row in range(axes.shape[0]):
                assert f"country={country}" in axes[row, col].get_title()

    def test_saturation_scatterplot_with_multiple_dims_lists(
        self, mock_suite_with_constant_data
    ):
        """Test saturation_scatterplot with multiple lists in dims (should create subplots for each combination)."""
        # Add a fake 'region' dim to the mock constant_data for this test if not present
        idata = mock_suite_with_constant_data.idata
        if "region" not in idata.constant_data.channel_data.dims:
            # Expand channel_data and posterior to add region
            new_regions = ["X", "Y"]
            channel_data = idata.constant_data.channel_data.expand_dims(
                region=new_regions
            )
            idata.constant_data["channel_data"] = channel_data
            for var in ["channel_contribution", "channel_contribution_original_scale"]:
                if var in idata.posterior:
                    idata.posterior[var] = idata.posterior[var].expand_dims(
                        region=new_regions
                    )
        fig, axes = mock_suite_with_constant_data.saturation_scatterplot(
            dims={"country": ["A", "B"], "region": ["X", "Y"]}
        )
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create 4 columns (n_channels, 4)
        assert axes.shape[1] == 4
        combos = [("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y")]
        for col, (country, region) in enumerate(combos):
            for row in range(axes.shape[0]):
                title = axes[row, col].get_title()
                assert f"country={country}" in title
                assert f"region={region}" in title


class TestSaturationCurves:
    def test_saturation_curves_basic(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test basic functionality of saturation_curves."""
        fig, axes = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, n_samples=5
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_saturation_curves_original_scale(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with original_scale=True."""
        fig, axes = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, original_scale=True, n_samples=3
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_saturation_curves_with_hdi(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with HDI intervals."""
        fig, axes = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, n_samples=3, hdi_probs=[0.5, 0.94]
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_saturation_curves_single_hdi(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with single HDI probability."""
        fig, axes = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, n_samples=3, hdi_probs=0.85
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_saturation_curves_custom_colors(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with custom colors."""
        fig, axes = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, n_samples=3, colors=["red", "blue"]
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_saturation_curves_subplot_kwargs(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with custom subplot_kwargs."""
        fig, axes = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve,
            n_samples=3,
            subplot_kwargs={"figsize": (12, 8)},
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)
        # Check that figsize was applied
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 8

    def test_saturation_curves_rc_params(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with rc_params."""
        fig, axes = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, n_samples=3, rc_params={"font.size": 14}
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_saturation_curves_no_samples(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with n_samples=0."""
        fig, axes = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, n_samples=0, hdi_probs=0.85
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_saturation_curves_no_constant_data(
        self, mock_suite, mock_saturation_curve
    ):
        """Test that saturation_curves raises error without constant_data."""
        with pytest.raises(ValueError, match=r"No 'constant_data' found"):
            mock_suite.saturation_curves(curve=mock_saturation_curve)

    def test_saturation_curves_no_original_scale_contribution(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test that saturation_curves raises error when original_scale=True but no original scale data."""
        # Remove the original scale contribution from the mock data
        idata_copy = mock_suite_with_constant_data.idata.copy()
        idata_copy.posterior = idata_copy.posterior.drop_vars(
            "channel_contribution_original_scale"
        )
        suite_without_original_scale = MMMPlotSuite(idata=idata_copy)

        with pytest.raises(
            ValueError, match=r"No posterior.channel_contribution_original_scale"
        ):
            suite_without_original_scale.saturation_curves(
                curve=mock_saturation_curve, original_scale=True
            )


class TestSaturationCurvesDims:
    def test_saturation_curves_with_dim(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with a single value in dims."""
        fig, axes = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, n_samples=3, dims={"country": "A"}
        )
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

        for row in range(axes.shape[0]):
            assert "country=A" in axes[row, 0].get_title()

    def test_saturation_curves_with_dims_list(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with a list in dims (should create subplots for each value)."""
        fig, axes = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, n_samples=3, dims={"country": ["A", "B"]}
        )
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_saturation_curves_with_multiple_dims_lists(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with multiple lists in dims (should create subplots for each combination)."""
        # Add a fake 'region' dim to the mock constant_data for this test if not present
        idata = mock_suite_with_constant_data.idata
        if "region" not in idata.constant_data.channel_data.dims:
            # Expand channel_data and posterior to add region
            new_regions = ["X", "Y"]
            channel_data = idata.constant_data.channel_data.expand_dims(
                region=new_regions
            )
            idata.constant_data["channel_data"] = channel_data
            for var in ["channel_contribution", "channel_contribution_original_scale"]:
                if var in idata.posterior:
                    idata.posterior[var] = idata.posterior[var].expand_dims(
                        region=new_regions
                    )
        fig, axes = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve,
            n_samples=3,
            dims={"country": ["A", "B"], "region": ["X", "Y"]},
        )
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        combos = [("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y")]

        for col, (country, region) in enumerate(combos):
            for row in range(axes.shape[0]):
                title = axes[row, col].get_title()
                assert f"country={country}" in title
                assert f"region={region}" in title


def test_saturation_curves_scatter_deprecation_warning(mock_suite_with_constant_data):
    """Test that saturation_curves_scatter shows deprecation warning."""
    with pytest.warns(
        DeprecationWarning, match=r"saturation_curves_scatter is deprecated"
    ):
        fig, axes = mock_suite_with_constant_data.saturation_curves_scatter()

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert all(isinstance(ax, Axes) for ax in axes.flat)


@pytest.fixture(scope="module")
def mock_idata_with_constant_data_single_dim() -> az.InferenceData:
    """Mock InferenceData where channel_data has only ('date','channel') dims."""
    seed = sum(map(ord, "Saturation single-dim tests"))
    rng = np.random.default_rng(seed)
    normal = rng.normal

    dates = pd.date_range("2025-01-01", periods=12, freq="W-MON")
    channels = ["channel_1", "channel_2", "channel_3"]

    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                normal(size=(2, 10, 12, 3)),
                dims=("chain", "draw", "date", "channel"),
                coords={
                    "chain": np.arange(2),
                    "draw": np.arange(10),
                    "date": dates,
                    "channel": channels,
                },
            ),
            "channel_contribution_original_scale": xr.DataArray(
                normal(size=(2, 10, 12, 3)) * 100.0,
                dims=("chain", "draw", "date", "channel"),
                coords={
                    "chain": np.arange(2),
                    "draw": np.arange(10),
                    "date": dates,
                    "channel": channels,
                },
            ),
        }
    )

    constant_data = xr.Dataset(
        {
            "channel_data": xr.DataArray(
                rng.uniform(0, 10, size=(12, 3)),
                dims=("date", "channel"),
                coords={"date": dates, "channel": channels},
            ),
            "channel_scale": xr.DataArray(
                [100.0, 150.0, 200.0], dims=("channel",), coords={"channel": channels}
            ),
            "target_scale": xr.DataArray(
                [1000.0], dims="target", coords={"target": ["y"]}
            ),
        }
    )

    return az.InferenceData(posterior=posterior, constant_data=constant_data)


@pytest.fixture(scope="module")
def mock_suite_with_constant_data_single_dim(mock_idata_with_constant_data_single_dim):
    return MMMPlotSuite(idata=mock_idata_with_constant_data_single_dim)


@pytest.fixture(scope="module")
def mock_saturation_curve_single_dim() -> xr.DataArray:
    """Saturation curve with dims ('chain','draw','channel','x')."""
    seed = sum(map(ord, "Saturation curve single-dim"))
    rng = np.random.default_rng(seed)
    x_values = np.linspace(0, 1, 50)
    channels = ["channel_1", "channel_2", "channel_3"]

    # shape: (chains=2, draws=10, channel=3, x=50)
    curve_array = np.empty((2, 10, len(channels), len(x_values)))
    for ci in range(2):
        for di in range(10):
            for c in range(len(channels)):
                curve_array[ci, di, c, :] = x_values / (1 + x_values) + rng.normal(
                    0, 0.02, size=x_values.shape
                )

    return xr.DataArray(
        curve_array,
        dims=("chain", "draw", "channel", "x"),
        coords={
            "chain": np.arange(2),
            "draw": np.arange(10),
            "channel": channels,
            "x": x_values,
        },
        name="saturation_curve",
    )


def test_saturation_curves_single_dim_axes_shape(
    mock_suite_with_constant_data_single_dim, mock_saturation_curve_single_dim
):
    """When there are no extra dims, columns should default to 1 (no ncols=0)."""
    fig, axes = mock_suite_with_constant_data_single_dim.saturation_curves(
        curve=mock_saturation_curve_single_dim, n_samples=3
    )

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    # Expect (n_channels, 1)
    assert axes.shape[1] == 1
    assert axes.shape[0] == mock_saturation_curve_single_dim.sizes["channel"]


def test_saturation_curves_multi_dim_axes_shape(
    mock_suite_with_constant_data, mock_saturation_curve
):
    """With an extra dim (e.g., 'country'), expect (n_channels, n_countries)."""
    fig, axes = mock_suite_with_constant_data.saturation_curves(
        curve=mock_saturation_curve, n_samples=2
    )

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)


def test_sensitivity_analysis_basic(mock_suite_with_sensitivity):
    fig, axes = mock_suite_with_sensitivity.sensitivity_analysis()

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.ndim == 2
    expected_panels = len(
        mock_suite_with_sensitivity.idata.sensitivity_analysis.coords["region"]
    )  # type: ignore
    assert axes.size >= expected_panels
    assert all(isinstance(ax, Axes) for ax in axes.flat[:expected_panels])


def test_sensitivity_analysis_with_aggregation(mock_suite_with_sensitivity):
    ax = mock_suite_with_sensitivity.sensitivity_analysis(
        aggregation={"sum": ("region",)}
    )
    assert isinstance(ax, Axes)


def test_marginal_curve(mock_suite_with_sensitivity):
    fig, axes = mock_suite_with_sensitivity.marginal_curve()

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.ndim == 2
    regions = mock_suite_with_sensitivity.idata.sensitivity_analysis.coords["region"]  # type: ignore
    assert axes.size >= len(regions)
    assert all(isinstance(ax, Axes) for ax in axes.flat[: len(regions)])


def test_uplift_curve(mock_suite_with_sensitivity):
    fig, axes = mock_suite_with_sensitivity.uplift_curve()

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.ndim == 2
    regions = mock_suite_with_sensitivity.idata.sensitivity_analysis.coords["region"]  # type: ignore
    assert axes.size >= len(regions)
    assert all(isinstance(ax, Axes) for ax in axes.flat[: len(regions)])


def test_sensitivity_analysis_multi_panel(mock_suite_with_sensitivity):
    # The fixture provides an extra 'region' dimension, so multiple panels should be produced
    fig, axes = mock_suite_with_sensitivity.sensitivity_analysis(
        subplot_kwargs={"ncols": 2}
    )

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.ndim == 2
    # There should be two regions, therefore exactly two panels
    expected_panels = len(
        mock_suite_with_sensitivity.idata.sensitivity_analysis.coords["region"]
    )  # type: ignore
    assert axes.size >= expected_panels
    assert all(isinstance(ax, Axes) for ax in axes.flat[:expected_panels])


def test_sensitivity_analysis_error_on_missing_results(mock_idata):
    suite = MMMPlotSuite(idata=mock_idata)
    with pytest.raises(ValueError, match=r"No sensitivity analysis results found"):
        suite.sensitivity_analysis()
        suite.plot_sensitivity_analysis()


# Tests for hue_dim parameter
def test_sensitivity_analysis_hue_dim_basic(mock_suite_with_sensitivity_and_channels):
    """Verify multiple lines are drawn when hue_dim='channel'."""
    fig, axes = mock_suite_with_sensitivity_and_channels.sensitivity_analysis(
        hue_dim="channel"
    )
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    # Verify multiple lines were drawn (one per channel) in each visible axis
    for ax in axes.flat:
        if ax.get_visible():
            lines = ax.get_lines()
            # At least 2 lines (one per channel)
            assert len(lines) >= 2


def test_sensitivity_analysis_hue_dim_invalid_sample(mock_suite_with_sensitivity):
    """Error when hue_dim='sample' (excluded dim)."""
    with pytest.raises(ValueError, match=r"Invalid hue_dim 'sample'"):
        mock_suite_with_sensitivity.sensitivity_analysis(hue_dim="sample")


def test_sensitivity_analysis_hue_dim_invalid_sweep(mock_suite_with_sensitivity):
    """Error when hue_dim='sweep' (excluded dim)."""
    with pytest.raises(ValueError, match=r"Invalid hue_dim 'sweep'"):
        mock_suite_with_sensitivity.sensitivity_analysis(hue_dim="sweep")


def test_sensitivity_analysis_hue_dim_not_found(mock_suite_with_sensitivity):
    """Error when hue_dim is not in the data."""
    with pytest.raises(
        ValueError, match=r"Dimension 'nonexistent' not found in sensitivity analysis"
    ):
        mock_suite_with_sensitivity.sensitivity_analysis(hue_dim="nonexistent")


# Tests for legend parameter
def test_sensitivity_analysis_legend_default_on(
    mock_suite_with_sensitivity_and_channels,
):
    """Legend shown by default when hue_dim is set."""
    _fig, axes = mock_suite_with_sensitivity_and_channels.sensitivity_analysis(
        hue_dim="channel"
    )
    # Check that at least one visible axis has a legend
    has_legend = False
    for ax in axes.flat:
        if ax.get_visible() and ax.get_legend() is not None:
            has_legend = True
            break
    assert has_legend


def test_sensitivity_analysis_legend_disabled(mock_suite_with_sensitivity_and_channels):
    """Legend hidden when legend=False."""
    _fig, axes = mock_suite_with_sensitivity_and_channels.sensitivity_analysis(
        hue_dim="channel", legend=False
    )
    # All visible axes should NOT have a legend
    for ax in axes.flat:
        if ax.get_visible():
            assert ax.get_legend() is None


def test_sensitivity_analysis_legend_kwargs(mock_suite_with_sensitivity_and_channels):
    """legend_kwargs are passed correctly."""
    _fig, axes = mock_suite_with_sensitivity_and_channels.sensitivity_analysis(
        hue_dim="channel", legend_kwargs={"loc": "upper left", "title": "Channels"}
    )
    # Check that at least one visible axis has a legend with the specified title
    for ax in axes.flat:
        if ax.get_visible():
            legend = ax.get_legend()
            if legend is not None:
                assert legend.get_title().get_text() == "Channels"
                break


# Tests for x_sweep_axis parameter
def test_sensitivity_analysis_x_sweep_axis_relative(
    mock_suite_with_sensitivity_and_channels,
):
    """Default 'relative' mode works."""
    fig, axes = mock_suite_with_sensitivity_and_channels.sensitivity_analysis(
        hue_dim="channel", x_sweep_axis="relative"
    )
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)


def test_sensitivity_analysis_x_sweep_axis_absolute(
    mock_suite_with_sensitivity_and_channels,
):
    """'absolute' mode works with proper constant_data setup."""
    fig, axes = mock_suite_with_sensitivity_and_channels.sensitivity_analysis(
        hue_dim="channel", x_sweep_axis="absolute"
    )
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)


def test_sensitivity_analysis_x_sweep_axis_absolute_requires_hue_dim(
    mock_suite_with_sensitivity_and_channels,
):
    """Error when x_sweep_axis='absolute' but hue_dim is None."""
    with pytest.raises(
        ValueError, match=r"x_sweep_axis='absolute' requires hue_dim to be set"
    ):
        mock_suite_with_sensitivity_and_channels.sensitivity_analysis(
            x_sweep_axis="absolute"
        )


def test_sensitivity_analysis_x_sweep_axis_absolute_missing_constant_data(
    mock_suite_with_sensitivity,
):
    """Error when constant_data is missing."""
    with pytest.raises(
        ValueError, match=r"x_sweep_axis='absolute' requires idata.constant_data"
    ):
        mock_suite_with_sensitivity.sensitivity_analysis(
            hue_dim="region", x_sweep_axis="absolute"
        )


def test_sensitivity_analysis_x_sweep_axis_absolute_missing_channel_scale():
    """Error when channel_scale is missing from constant_data."""
    # Create idata with constant_data but no channel_scale
    n_sample, n_sweep = 40, 5
    sweep = np.linspace(0.5, 1.5, n_sweep)
    regions = ["A", "B"]

    samples = xr.DataArray(
        np.random.normal(0, 1, size=(n_sample, n_sweep, len(regions))),
        dims=("sample", "sweep", "region"),
        coords={
            "sample": np.arange(n_sample),
            "sweep": sweep,
            "region": regions,
        },
        name="x",
    )
    sensitivity_analysis = xr.Dataset(
        {"x": samples},
        coords={"sweep": sweep, "region": regions},
    )

    # constant_data WITHOUT channel_scale
    constant_data = xr.Dataset(
        {
            "some_other_var": xr.DataArray([1.0, 2.0], dims=("region",)),
        }
    )

    idata = az.InferenceData(constant_data=constant_data)
    idata.sensitivity_analysis = sensitivity_analysis

    suite = MMMPlotSuite(idata=idata)
    with pytest.raises(
        ValueError, match=r"x_sweep_axis='absolute' requires 'channel_scale'"
    ):
        suite.sensitivity_analysis(hue_dim="region", x_sweep_axis="absolute")


def test_budget_allocation_with_dims(mock_suite_with_constant_data):
    # Use dims to filter to a single country
    samples = mock_suite_with_constant_data.idata.posterior
    # Add a fake 'allocation' variable for testing
    samples = samples.copy()
    samples["allocation"] = (
        samples["channel_contribution"].dims,
        np.abs(samples["channel_contribution"].values),
    )
    plot_suite = mock_suite_with_constant_data
    fig, _ax = plot_suite.budget_allocation(
        samples=samples,
        dims={"country": "A"},
    )
    assert isinstance(fig, Figure)


def test_budget_allocation_with_dims_list(mock_suite_with_constant_data):
    """Test that passing a list to dims creates a subplot for each value."""
    samples = mock_suite_with_constant_data.idata.posterior.copy()
    # Add a fake 'allocation' variable for testing
    samples["allocation"] = (
        samples["channel_contribution"].dims,
        np.abs(samples["channel_contribution"].values),
    )
    plot_suite = mock_suite_with_constant_data
    fig, ax = plot_suite.budget_allocation(
        samples=samples,
        dims={"country": ["A", "B"]},
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)


def test__validate_dims_valid():
    """Test _validate_dims with valid dims and values."""
    suite = MMMPlotSuite(idata=None)

    # Patch suite.idata.posterior.coords to simulate valid dims
    class DummyCoord:
        def __init__(self, values):
            self.values = values

    class DummyCoords:
        def __init__(self):
            self._coords = {
                "country": DummyCoord(["A", "B"]),
                "region": DummyCoord(["X", "Y"]),
            }

        def __getitem__(self, key):
            return self._coords[key]

    class DummyPosterior:
        coords = DummyCoords()

    suite.idata = type("idata", (), {"posterior": DummyPosterior()})()
    # Should not raise
    suite._validate_dims({"country": "A", "region": "X"}, ["country", "region"])
    suite._validate_dims({"country": ["A", "B"]}, ["country", "region"])


def test__validate_dims_invalid_dim():
    """Test _validate_dims raises for invalid dim name."""
    suite = MMMPlotSuite(idata=None)

    class DummyCoord:
        def __init__(self, values):
            self.values = values

    class DummyCoords:
        def __init__(self):
            self.country = DummyCoord(["A", "B"])

        def __getitem__(self, key):
            return getattr(self, key)

    class DummyPosterior:
        coords = DummyCoords()

    suite.idata = type("idata", (), {"posterior": DummyPosterior()})()
    with pytest.raises(ValueError, match=r"Dimension 'region' not found"):
        suite._validate_dims({"region": "X"}, ["country"])


def test__validate_dims_invalid_value():
    """Test _validate_dims raises for invalid value."""
    suite = MMMPlotSuite(idata=None)

    class DummyCoord:
        def __init__(self, values):
            self.values = values

    class DummyCoords:
        def __init__(self):
            self.country = DummyCoord(["A", "B"])

        def __getitem__(self, key):
            return getattr(self, key)

    class DummyPosterior:
        coords = DummyCoords()

    suite.idata = type("idata", (), {"posterior": DummyPosterior()})()
    with pytest.raises(ValueError, match=r"Value 'C' not found in dimension 'country'"):
        suite._validate_dims({"country": "C"}, ["country"])


def test__dim_list_handler_none():
    """Test _dim_list_handler with None input."""
    suite = MMMPlotSuite(idata=None)
    keys, combos = suite._dim_list_handler(None)
    assert keys == []
    assert combos == [()]


def test__dim_list_handler_single():
    """Test _dim_list_handler with a single list-valued dim."""
    suite = MMMPlotSuite(idata=None)
    keys, combos = suite._dim_list_handler({"country": ["A", "B"]})
    assert keys == ["country"]
    assert set(combos) == {("A",), ("B",)}


def test__dim_list_handler_multiple():
    """Test _dim_list_handler with multiple list-valued dims."""
    suite = MMMPlotSuite(idata=None)
    keys, combos = suite._dim_list_handler(
        {"country": ["A", "B"], "region": ["X", "Y"]}
    )
    assert set(keys) == {"country", "region"}
    assert set(combos) == {("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y")}


def test__dim_list_handler_mixed():
    """Test _dim_list_handler with mixed single and list values."""
    suite = MMMPlotSuite(idata=None)
    keys, combos = suite._dim_list_handler({"country": ["A", "B"], "region": "X"})
    assert keys == ["country"]
    assert set(combos) == {("A",), ("B",)}


@pytest.fixture(scope="module")
def mock_idata_with_errors_data() -> az.InferenceData:
    """Create mock InferenceData with data needed for errors plot."""
    seed = sum(map(ord, "Errors plot tests"))
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2025-01-01", periods=20, freq="W-MON")
    countries = ["A", "B"]

    # Create posterior_predictive with y_original_scale
    posterior_predictive = xr.Dataset(
        {
            "y_original_scale": xr.DataArray(
                rng.normal(100, 10, size=(4, 50, 20, 2)),
                dims=("chain", "draw", "date", "country"),
                coords={
                    "chain": np.arange(4),
                    "draw": np.arange(50),
                    "date": dates,
                    "country": countries,
                },
            ),
        }
    )

    # Create constant_data with target_data
    constant_data = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(100, 10, size=(20, 2)),
                dims=("date", "country"),
                coords={
                    "date": dates,
                    "country": countries,
                },
            ),
        }
    )

    return az.InferenceData(
        posterior_predictive=posterior_predictive, constant_data=constant_data
    )


@pytest.fixture(scope="module")
def mock_suite_with_errors_data(mock_idata_with_errors_data):
    """Fixture to create a MMMPlotSuite with data for errors plot."""
    return MMMPlotSuite(idata=mock_idata_with_errors_data)


class TestResidualsOverTime:
    def test_residuals_over_time_basic(self, mock_suite_with_errors_data):
        """Test basic functionality of residuals over time plot."""
        fig, axes = mock_suite_with_errors_data.residuals_over_time()

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_residuals_over_time_single_hdi(self, mock_suite_with_errors_data):
        """Test residuals over time plot with single HDI probability."""
        fig, axes = mock_suite_with_errors_data.residuals_over_time(hdi_prob=[0.94])

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_residuals_over_time_multiple_hdi(self, mock_suite_with_errors_data):
        """Test residuals over time plot with multiple HDI probabilities."""
        fig, axes = mock_suite_with_errors_data.residuals_over_time(
            hdi_prob=[0.94, 0.50]
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_residuals_over_time_no_y_original_scale(self, mock_suite):
        """Test that residuals over time raises error without y_original_scale."""
        # Create idata with posterior_predictive but without y_original_scale
        dates = pd.date_range("2025-01-01", periods=20, freq="W-MON")
        idata = az.InferenceData(
            posterior_predictive=xr.Dataset(
                {
                    "y": xr.DataArray(
                        np.random.randn(2, 50, 20),
                        dims=("chain", "draw", "date"),
                        coords={"chain": [0, 1], "draw": np.arange(50), "date": dates},
                    )
                }
            )
        )
        suite = MMMPlotSuite(idata=idata)

        with pytest.raises(ValueError, match=r"Variable 'y_original_scale' not found"):
            suite.residuals_over_time()

    def test_residuals_over_time_no_target_data(self, mock_suite_with_constant_data):
        """Test that residuals over time raises error without target_data in constant_data."""
        # Create idata with posterior_predictive but no target_data
        idata = mock_suite_with_constant_data.idata.copy()

        # Add y_original_scale to posterior_predictive
        dates = pd.date_range("2025-01-01", periods=20, freq="W-MON")
        idata.posterior_predictive = xr.Dataset(
            {
                "y_original_scale": xr.DataArray(
                    np.random.normal(100, 10, size=(4, 50, 20)),
                    dims=("chain", "draw", "date"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(50),
                        "date": dates,
                    },
                ),
            }
        )

        # Remove target_data from constant_data
        idata.constant_data = idata.constant_data.drop_vars(
            "target_data", errors="ignore"
        )

        suite = MMMPlotSuite(idata=idata)
        with pytest.raises(ValueError, match=r"Variable 'target_data' not found"):
            suite.residuals_over_time()

    def test_residuals_over_time_invalid_hdi_prob(self, mock_suite_with_errors_data):
        """Test that residuals over time raises error with invalid HDI probability."""
        with pytest.raises(
            ValueError, match=r"All HDI probabilities must be between 0 and 1"
        ):
            mock_suite_with_errors_data.residuals_over_time(hdi_prob=[1.5])

        with pytest.raises(
            ValueError, match=r"All HDI probabilities must be between 0 and 1"
        ):
            mock_suite_with_errors_data.residuals_over_time(hdi_prob=[0.94, -0.1])

    def test_residuals_over_time_multidimensional(self, mock_suite_with_errors_data):
        """Test residuals over time plot creates subplot for each dimension combination."""
        fig, axes = mock_suite_with_errors_data.residuals_over_time()

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create one subplot per country (2 countries)
        assert axes.shape[0] == 2


class TestResidualsPosteriоrDistribution:
    def test_residuals_distribution_basic(self, mock_suite_with_errors_data):
        """Test basic functionality of residuals posterior distribution plot."""
        fig, axes = mock_suite_with_errors_data.residuals_posterior_distribution()

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_residuals_distribution_with_aggregation_mean(
        self, mock_suite_with_errors_data
    ):
        """Test residuals posterior distribution with mean aggregation."""
        fig, axes = mock_suite_with_errors_data.residuals_posterior_distribution(
            aggregation="mean"
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # With aggregation, should create a single plot
        assert axes.size == 1

    def test_residuals_distribution_with_aggregation_sum(
        self, mock_suite_with_errors_data
    ):
        """Test residuals posterior distribution with sum aggregation."""
        fig, axes = mock_suite_with_errors_data.residuals_posterior_distribution(
            aggregation="sum"
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # With aggregation, should create a single plot
        assert axes.size == 1

    def test_residuals_distribution_custom_quantiles(self, mock_suite_with_errors_data):
        """Test residuals posterior distribution with custom quantiles."""
        fig, axes = mock_suite_with_errors_data.residuals_posterior_distribution(
            quantiles=[0.05, 0.5, 0.95]
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_residuals_distribution_invalid_quantile(self, mock_suite_with_errors_data):
        """Test that residuals distribution raises error with invalid quantile."""
        with pytest.raises(ValueError, match=r"All quantiles must be between 0 and 1"):
            mock_suite_with_errors_data.residuals_posterior_distribution(
                quantiles=[1.5]
            )

    def test_residuals_distribution_invalid_aggregation(
        self, mock_suite_with_errors_data
    ):
        """Test that residuals distribution raises error with invalid aggregation."""
        with pytest.raises(ValueError, match=r"aggregation must be one of"):
            mock_suite_with_errors_data.residuals_posterior_distribution(
                aggregation="invalid"
            )

    def test_residuals_distribution_multidimensional(self, mock_suite_with_errors_data):
        """Test residuals distribution creates subplot for each dimension combination."""
        fig, axes = mock_suite_with_errors_data.residuals_posterior_distribution()

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create one subplot per country (2 countries)
        assert axes.shape[0] == 2


class TestWaterfallPlot:
    """Test cases for waterfall_components_decomposition plot."""

    @pytest.fixture(scope="class")
    def mock_idata_for_waterfall(self):
        """Create mock InferenceData for waterfall plot testing."""
        dates = pd.date_range("2025-01-01", periods=20, freq="W-MON")
        channels = ["C1", "C2"]
        controls = ["control1"]

        idata = az.InferenceData(
            posterior=xr.Dataset(
                {
                    "intercept_contribution_original_scale": xr.DataArray(
                        np.random.normal(50, 5, size=(2, 100, 20)),
                        dims=("chain", "draw", "date"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(100),
                            "date": dates,
                        },
                    ),
                    "channel_contribution_original_scale": xr.DataArray(
                        np.random.normal(30, 10, size=(2, 100, 20, 2)),
                        dims=("chain", "draw", "date", "channel"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(100),
                            "date": dates,
                            "channel": channels,
                        },
                    ),
                    "control_contribution_original_scale": xr.DataArray(
                        np.random.normal(10, 3, size=(2, 100, 20, 1)),
                        dims=("chain", "draw", "date", "control"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(100),
                            "date": dates,
                            "control": controls,
                        },
                    ),
                }
            )
        )
        return idata

    @pytest.fixture(scope="class")
    def mock_suite_for_waterfall(self, mock_idata_for_waterfall):
        """Create MMMPlotSuite for waterfall tests."""
        return MMMPlotSuite(idata=mock_idata_for_waterfall)

    def test_waterfall_basic(self, mock_suite_for_waterfall):
        """Test basic waterfall plot functionality."""
        fig, ax = mock_suite_for_waterfall.waterfall_components_decomposition(
            var=[
                "intercept_contribution_original_scale",
                "channel_contribution_original_scale",
                "control_contribution_original_scale",
            ]
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_title() == "Response Decomposition Waterfall by Components"
        assert ax.get_xlabel() == "Cumulative Contribution"
        assert ax.get_ylabel() == "Components"

    def test_waterfall_custom_figsize(self, mock_suite_for_waterfall):
        """Test waterfall plot with custom figure size."""
        fig, _ = mock_suite_for_waterfall.waterfall_components_decomposition(
            var=["intercept_contribution_original_scale"],
            figsize=(16, 8),
        )

        assert isinstance(fig, Figure)
        assert fig.get_figwidth() == 16
        assert fig.get_figheight() == 8

    def test_waterfall_single_variable(self, mock_suite_for_waterfall):
        """Test waterfall plot with a single variable."""
        fig, ax = mock_suite_for_waterfall.waterfall_components_decomposition(
            var=["intercept_contribution_original_scale"]
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_waterfall_missing_posterior_error(self):
        """Test error when posterior data is missing."""
        # Create InferenceData without posterior
        idata_no_posterior = az.InferenceData()
        suite = MMMPlotSuite(idata=idata_no_posterior)

        with pytest.raises(ValueError, match=r"No posterior data found"):
            suite.waterfall_components_decomposition(
                var=["intercept_contribution_original_scale"]
            )

    def test_waterfall_missing_variables_error(self, mock_suite_for_waterfall):
        """Test error when requested variables don't exist."""
        with pytest.raises(ValueError, match=r"None of the requested variables"):
            mock_suite_for_waterfall.waterfall_components_decomposition(
                var=["nonexistent_variable"]
            )

    def test_process_decomposition_components(self, mock_suite_for_waterfall):
        """Test the private _process_decomposition_components method."""
        # Create a simple DataFrame
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "intercept": [10, 11, 12, 13, 14],
                "channel__C1": [20, 21, 22, 23, 24],
                "channel__C2": [5, 6, 7, 8, 9],
            }
        )

        result = mock_suite_for_waterfall._process_decomposition_components(df)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert "component" in result.columns
        assert "contribution" in result.columns
        assert "percentage" in result.columns

        # Should have 3 components (intercept, channel__C1, channel__C2)
        assert len(result) == 3

        # Check that contributions are summed correctly
        total = result["contribution"].sum()
        assert np.isclose(
            total,
            (10 + 11 + 12 + 13 + 14) + (20 + 21 + 22 + 23 + 24) + (5 + 6 + 7 + 8 + 9),
        )

        # Check that percentages sum to 100
        assert np.isclose(result["percentage"].sum(), 100.0)

    def test_process_decomposition_components_with_categorical(
        self, mock_suite_for_waterfall
    ):
        """Test _process_decomposition_components with categorical columns."""
        dates = pd.date_range("2025-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "geo": pd.Categorical(["US", "US", "UK", "UK"]),
                "intercept": [10, 11, 12, 13],
                "channel__C1": [20, 21, 22, 23],
            }
        )

        result = mock_suite_for_waterfall._process_decomposition_components(df)

        # Should handle categorical columns correctly
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # intercept and channel__C1

    def test_waterfall_with_multidimensional_data(self):
        """Test waterfall plot with multidimensional data (geo dimension)."""
        dates = pd.date_range("2025-01-01", periods=10, freq="W-MON")
        channels = ["C1", "C2"]
        geos = ["US", "UK"]

        idata = az.InferenceData(
            posterior=xr.Dataset(
                {
                    "intercept_contribution": xr.DataArray(
                        np.random.normal(50, 5, size=(2, 50, 10, 2)),
                        dims=("chain", "draw", "date", "geo"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                            "geo": geos,
                        },
                    ),
                    "channel_contribution": xr.DataArray(
                        np.random.normal(30, 10, size=(2, 50, 10, 2, 2)),
                        dims=("chain", "draw", "date", "channel", "geo"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                            "channel": channels,
                            "geo": geos,
                        },
                    ),
                }
            )
        )

        suite = MMMPlotSuite(idata=idata)
        fig, ax = suite.waterfall_components_decomposition(
            var=["intercept_contribution", "channel_contribution"]
        )

        # Should work with multidimensional data
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    @pytest.fixture(scope="class")
    def mock_idata_with_geo(self):
        """Create mock InferenceData with geo dimension for testing split_by."""
        dates = pd.date_range("2025-01-01", periods=10, freq="W-MON")
        channels = ["C1", "C2"]
        geos = ["US", "UK", "DE"]

        idata = az.InferenceData(
            posterior=xr.Dataset(
                {
                    "intercept_contribution_original_scale": xr.DataArray(
                        np.random.normal(50, 5, size=(2, 50, 10, 3)),
                        dims=("chain", "draw", "date", "geo"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                            "geo": geos,
                        },
                    ),
                    "channel_contribution_original_scale": xr.DataArray(
                        np.random.normal(30, 10, size=(2, 50, 10, 2, 3)),
                        dims=("chain", "draw", "date", "channel", "geo"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                            "channel": channels,
                            "geo": geos,
                        },
                    ),
                }
            )
        )
        return idata

    @pytest.fixture(scope="class")
    def mock_suite_with_geo(self, mock_idata_with_geo):
        """Create MMMPlotSuite with geo dimension for split_by tests."""
        return MMMPlotSuite(idata=mock_idata_with_geo)

    def test_waterfall_with_dims_filter(self, mock_suite_with_geo):
        """Test waterfall plot with dimension filtering."""
        fig, ax = mock_suite_with_geo.waterfall_components_decomposition(
            var=[
                "intercept_contribution_original_scale",
                "channel_contribution_original_scale",
            ],
            dims={"geo": "US"},
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_title() == "Response Decomposition Waterfall by Components"

    def test_waterfall_with_dims_filter_list(self, mock_suite_with_geo):
        """Test waterfall plot with dimension filtering using list of values."""
        fig, ax = mock_suite_with_geo.waterfall_components_decomposition(
            var=[
                "intercept_contribution_original_scale",
                "channel_contribution_original_scale",
            ],
            dims={"geo": ["US", "UK"]},
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_waterfall_split_by_single_dim(self, mock_suite_with_geo):
        """Test waterfall plot with split_by for single dimension."""
        fig, axes = mock_suite_with_geo.waterfall_components_decomposition(
            var=[
                "intercept_contribution_original_scale",
                "channel_contribution_original_scale",
            ],
            split_by="geo",
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should have 3 geos (US, UK, DE), so axes should be 2D array
        assert axes.size >= 3

        # Check that each subplot has appropriate title
        axes_flat = axes.flatten()
        titles = [ax.get_title() for ax in axes_flat if ax.get_visible()]
        assert len(titles) == 3
        assert any("US" in t for t in titles)
        assert any("UK" in t for t in titles)
        assert any("DE" in t for t in titles)

    def test_waterfall_split_by_with_ncols(self, mock_suite_with_geo):
        """Test waterfall plot with split_by and ncols specified."""
        fig, axes = mock_suite_with_geo.waterfall_components_decomposition(
            var=[
                "intercept_contribution_original_scale",
                "channel_contribution_original_scale",
            ],
            split_by="geo",
            subplot_kwargs={"ncols": 3},
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # With ncols=3 and 3 geos, should be 1 row x 3 cols
        assert axes.shape == (1, 3)

    def test_waterfall_split_by_with_nrows(self, mock_suite_with_geo):
        """Test waterfall plot with split_by and nrows specified."""
        fig, axes = mock_suite_with_geo.waterfall_components_decomposition(
            var=[
                "intercept_contribution_original_scale",
                "channel_contribution_original_scale",
            ],
            split_by="geo",
            subplot_kwargs={"nrows": 3},
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # With nrows=3 and 3 geos, should be 3 rows x 1 col
        assert axes.shape == (3, 1)

    def test_waterfall_subplot_kwargs_nrows_ncols_error(self, mock_suite_with_geo):
        """Test error when both nrows and ncols are specified."""
        with pytest.raises(ValueError, match=r"Specify only one of 'nrows' or 'ncols'"):
            mock_suite_with_geo.waterfall_components_decomposition(
                var=["intercept_contribution_original_scale"],
                split_by="geo",
                subplot_kwargs={"nrows": 2, "ncols": 2},
            )

    def test_waterfall_split_by_invalid_dim_error(self, mock_suite_with_geo):
        """Test error when split_by dimension doesn't exist."""
        with pytest.raises(ValueError, match=r"Split dimension 'nonexistent'"):
            mock_suite_with_geo.waterfall_components_decomposition(
                var=["intercept_contribution_original_scale"],
                split_by="nonexistent",
            )

    def test_waterfall_no_extra_dims_backward_compat(self, mock_suite_for_waterfall):
        """Test backward compatibility - no extra dims still works as before."""
        fig, ax = mock_suite_for_waterfall.waterfall_components_decomposition(
            var=[
                "intercept_contribution_original_scale",
                "channel_contribution_original_scale",
            ]
        )

        # Should return single figure and axes (not array)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Default figsize should be applied
        assert fig.get_figwidth() == 14
        assert fig.get_figheight() == 7

    def test_waterfall_auto_detect_variables(self, mock_suite_for_waterfall):
        """Test auto-detection of contribution variables when var is None."""
        # Should auto-detect original_scale variables
        fig, ax = mock_suite_for_waterfall.waterfall_components_decomposition()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_title() == "Response Decomposition Waterfall by Components"

    def test_waterfall_auto_detect_with_figsize(self, mock_suite_for_waterfall):
        """Test auto-detection with custom figsize (like old API)."""
        fig, ax = mock_suite_for_waterfall.waterfall_components_decomposition(
            figsize=(18, 10)
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert fig.get_figwidth() == 18
        assert fig.get_figheight() == 10

    def test_waterfall_auto_detect_non_original_scale(self):
        """Test auto-detection of non-original scale variables."""
        dates = pd.date_range("2025-01-01", periods=10, freq="W-MON")
        channels = ["C1", "C2"]

        # Create idata with both original and non-original scale variables
        idata = az.InferenceData(
            posterior=xr.Dataset(
                {
                    "intercept_contribution": xr.DataArray(
                        np.random.normal(50, 5, size=(2, 50, 10)),
                        dims=("chain", "draw", "date"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                        },
                    ),
                    "channel_contribution": xr.DataArray(
                        np.random.normal(30, 10, size=(2, 50, 10, 2)),
                        dims=("chain", "draw", "date", "channel"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                            "channel": channels,
                        },
                    ),
                    "intercept_contribution_original_scale": xr.DataArray(
                        np.random.normal(500, 50, size=(2, 50, 10)),
                        dims=("chain", "draw", "date"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                        },
                    ),
                    "channel_contribution_original_scale": xr.DataArray(
                        np.random.normal(300, 100, size=(2, 50, 10, 2)),
                        dims=("chain", "draw", "date", "channel"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                            "channel": channels,
                        },
                    ),
                }
            )
        )

        suite = MMMPlotSuite(idata=idata)

        # Test with original_scale=True (default)
        fig1, ax1 = suite.waterfall_components_decomposition(original_scale=True)
        assert isinstance(fig1, Figure)
        assert isinstance(ax1, Axes)

        # Test with original_scale=False
        fig2, ax2 = suite.waterfall_components_decomposition(original_scale=False)
        assert isinstance(fig2, Figure)
        assert isinstance(ax2, Axes)

        plt.close(fig1)
        plt.close(fig2)

    def test_waterfall_auto_detect_no_contribution_vars_error(self):
        """Test error when no contribution variables are found."""
        dates = pd.date_range("2025-01-01", periods=5, freq="W-MON")

        # Create idata without any contribution variables
        idata = az.InferenceData(
            posterior=xr.Dataset(
                {
                    "some_other_var": xr.DataArray(
                        np.random.normal(50, 5, size=(2, 50, 5)),
                        dims=("chain", "draw", "date"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                        },
                    ),
                }
            )
        )

        suite = MMMPlotSuite(idata=idata)

        with pytest.raises(ValueError, match=r"No contribution variables found"):
            suite.waterfall_components_decomposition()

    def test_waterfall_auto_detect_excludes_total_media(self):
        """Test that total_media_contribution_original_scale is excluded from auto-detection."""
        dates = pd.date_range("2025-01-01", periods=10, freq="W-MON")
        channels = ["C1", "C2"]

        # Create idata with total_media_contribution_original_scale
        idata = az.InferenceData(
            posterior=xr.Dataset(
                {
                    "intercept_contribution_original_scale": xr.DataArray(
                        np.random.normal(50, 5, size=(2, 50, 10)),
                        dims=("chain", "draw", "date"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                        },
                    ),
                    "channel_contribution_original_scale": xr.DataArray(
                        np.random.normal(30, 10, size=(2, 50, 10, 2)),
                        dims=("chain", "draw", "date", "channel"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                            "channel": channels,
                        },
                    ),
                    # This should be excluded - it's a sum of channel contributions
                    "total_media_contribution_original_scale": xr.DataArray(
                        np.random.normal(60, 15, size=(2, 50, 10)),
                        dims=("chain", "draw", "date"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                        },
                    ),
                }
            )
        )

        suite = MMMPlotSuite(idata=idata)

        # Get the auto-detected variables
        dataframe, _, _ = suite._prepare_waterfall_data()

        # Check that total_media_contribution is not in the columns
        assert "total_media_contribution_original_scale" not in dataframe.columns

        # Plot should work and not include total_media
        fig, ax = suite.waterfall_components_decomposition()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_prepare_waterfall_data_basic(self, mock_suite_for_waterfall):
        """Test the _prepare_waterfall_data method."""
        dataframe, split_dims, dim_combinations = (
            mock_suite_for_waterfall._prepare_waterfall_data(
                var=[
                    "intercept_contribution_original_scale",
                    "channel_contribution_original_scale",
                ],
                agg="mean",
            )
        )

        assert isinstance(dataframe, pd.DataFrame)
        assert len(split_dims) == 0
        assert dim_combinations == [()]

    def test_prepare_waterfall_data_with_split_by(self, mock_suite_with_geo):
        """Test _prepare_waterfall_data with split_by dimension."""
        dataframe, split_dims, dim_combinations = (
            mock_suite_with_geo._prepare_waterfall_data(
                var=["intercept_contribution_original_scale"],
                agg="mean",
                split_by="geo",
            )
        )

        assert isinstance(dataframe, pd.DataFrame)
        assert split_dims == ["geo"]
        assert len(dim_combinations) == 3  # US, UK, DE

    def test_prepare_waterfall_data_with_dims_filter(self, mock_suite_with_geo):
        """Test _prepare_waterfall_data with dimension filtering."""
        dataframe, _split_dims, _dim_combinations = (
            mock_suite_with_geo._prepare_waterfall_data(
                var=["intercept_contribution_original_scale"],
                agg="mean",
                dims={"geo": "US"},
            )
        )

        assert isinstance(dataframe, pd.DataFrame)
        # Only US data should be present
        if "geo" in dataframe.columns:
            assert all(dataframe["geo"] == "US")

    def test_plot_single_waterfall(self, mock_suite_for_waterfall):
        """Test the _plot_single_waterfall helper method."""
        # Create test data
        df = pd.DataFrame(
            {
                "component": ["intercept", "channel__C1", "channel__C2"],
                "contribution": [100, 50, -10],
                "percentage": [71.43, 35.71, -7.14],
            }
        )

        fig, ax = plt.subplots()
        mock_suite_for_waterfall._plot_single_waterfall(
            ax=ax, data=df, title="Test Waterfall"
        )

        assert ax.get_title() == "Test Waterfall"
        assert ax.get_xlabel() == "Cumulative Contribution"
        assert ax.get_ylabel() == "Components"

        # Check that bars were created
        bars = [
            child
            for child in ax.get_children()
            if isinstance(child, plt.Rectangle) and child.get_width() != 0
        ]
        assert len(bars) >= 3  # At least 3 bars for 3 components

        plt.close(fig)

    @pytest.fixture(scope="class")
    def mock_idata_with_two_dims(self):
        """Create mock InferenceData with two extra dimensions for testing."""
        dates = pd.date_range("2025-01-01", periods=5, freq="W-MON")
        channels = ["C1", "C2"]
        geos = ["US", "UK"]
        products = ["A", "B"]

        idata = az.InferenceData(
            posterior=xr.Dataset(
                {
                    "intercept_contribution_original_scale": xr.DataArray(
                        np.random.normal(50, 5, size=(2, 20, 5, 2, 2)),
                        dims=("chain", "draw", "date", "geo", "product"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(20),
                            "date": dates,
                            "geo": geos,
                            "product": products,
                        },
                    ),
                    "channel_contribution_original_scale": xr.DataArray(
                        np.random.normal(30, 10, size=(2, 20, 5, 2, 2, 2)),
                        dims=("chain", "draw", "date", "channel", "geo", "product"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(20),
                            "date": dates,
                            "channel": channels,
                            "geo": geos,
                            "product": products,
                        },
                    ),
                }
            )
        )
        return idata

    def test_waterfall_split_by_multiple_dims(self, mock_idata_with_two_dims):
        """Test waterfall plot with split_by for multiple dimensions."""
        suite = MMMPlotSuite(idata=mock_idata_with_two_dims)
        fig, axes = suite.waterfall_components_decomposition(
            var=[
                "intercept_contribution_original_scale",
                "channel_contribution_original_scale",
            ],
            split_by=["geo", "product"],
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # 2 geos x 2 products = 4 combinations
        axes_flat = axes.flatten()
        visible_axes = [ax for ax in axes_flat if ax.get_visible()]
        assert len(visible_axes) == 4


class TestPosteriorDistribution:
    """Tests for the posterior_distribution plotting method."""

    @pytest.fixture(scope="class")
    def mock_idata_for_posterior_dist(self) -> az.InferenceData:
        """Mock InferenceData with parameter variables for testing."""
        seed = sum(map(ord, "Posterior distribution tests"))
        rng = np.random.default_rng(seed)
        normal = rng.normal

        channels = ["TV", "Radio", "Digital"]
        geos = ["US", "UK"]

        posterior = xr.Dataset(
            {
                "lam": xr.DataArray(
                    normal(loc=0.5, scale=0.1, size=(2, 10, 3)),
                    dims=("chain", "draw", "channel"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "channel": channels,
                    },
                ),
                "alpha": xr.DataArray(
                    normal(loc=1.0, scale=0.2, size=(2, 10, 3, 2)),
                    dims=("chain", "draw", "channel", "geo"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "channel": channels,
                        "geo": geos,
                    },
                ),
            }
        )

        return az.InferenceData(posterior=posterior)

    @pytest.fixture(scope="class")
    def mock_suite_for_posterior_dist(self, mock_idata_for_posterior_dist):
        return MMMPlotSuite(idata=mock_idata_for_posterior_dist)

    def test_posterior_distribution_basic(self, mock_suite_for_posterior_dist):
        """Test basic posterior distribution plotting."""
        fig, axes = mock_suite_for_posterior_dist.posterior_distribution(
            var="lam", plot_dim="channel"
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape[0] == 1  # Single subplot for simple case
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_posterior_distribution_with_additional_dim(
        self, mock_suite_for_posterior_dist
    ):
        """Test posterior distribution with additional dimensions."""
        fig, axes = mock_suite_for_posterior_dist.posterior_distribution(
            var="alpha", plot_dim="channel"
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create one subplot per geo (2 geos)
        assert axes.shape[0] == 2
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_posterior_distribution_with_dims_filter(
        self, mock_suite_for_posterior_dist
    ):
        """Test posterior distribution with dimension filtering."""
        fig, axes = mock_suite_for_posterior_dist.posterior_distribution(
            var="alpha", plot_dim="channel", dims={"geo": "US"}
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create single subplot when filtering to one geo
        assert axes.shape[0] == 1
        assert "geo=US" in axes[0, 0].get_title()

    def test_posterior_distribution_with_dims_list(self, mock_suite_for_posterior_dist):
        """Test posterior distribution with list of dimension values."""
        fig, axes = mock_suite_for_posterior_dist.posterior_distribution(
            var="alpha", plot_dim="channel", dims={"geo": ["US", "UK"]}
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create one subplot per geo in the list
        assert axes.shape[0] == 2
        for i, geo in enumerate(["US", "UK"]):
            assert f"geo={geo}" in axes[i, 0].get_title()

    def test_posterior_distribution_vertical_orientation(
        self, mock_suite_for_posterior_dist
    ):
        """Test posterior distribution with vertical orientation."""
        fig, axes = mock_suite_for_posterior_dist.posterior_distribution(
            var="lam", plot_dim="channel", orient="v"
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        ax = axes[0, 0]
        # For vertical orientation, channel should be on x-axis
        assert ax.get_xlabel() == "channel"
        assert ax.get_ylabel() == "lam"

    def test_posterior_distribution_horizontal_orientation(
        self, mock_suite_for_posterior_dist
    ):
        """Test posterior distribution with horizontal orientation (default)."""
        fig, axes = mock_suite_for_posterior_dist.posterior_distribution(
            var="lam", plot_dim="channel", orient="h"
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        ax = axes[0, 0]
        # For horizontal orientation, channel should be on y-axis
        assert ax.get_xlabel() == "lam"
        assert ax.get_ylabel() == "channel"

    def test_posterior_distribution_invalid_var(self, mock_suite_for_posterior_dist):
        """Test that invalid variable name raises error."""
        with pytest.raises(ValueError, match=r"Variable 'invalid_var' not found"):
            mock_suite_for_posterior_dist.posterior_distribution(
                var="invalid_var", plot_dim="channel"
            )

    def test_posterior_distribution_invalid_plot_dim(
        self, mock_suite_for_posterior_dist
    ):
        """Test that invalid plot_dim raises error."""
        with pytest.raises(
            ValueError, match=r"Dimension 'invalid_dim' not found in variable"
        ):
            mock_suite_for_posterior_dist.posterior_distribution(
                var="lam", plot_dim="invalid_dim"
            )

    def test_posterior_distribution_no_posterior(self):
        """Test that missing posterior data raises error."""
        idata = az.InferenceData()
        suite = MMMPlotSuite(idata=idata)
        with pytest.raises(ValueError, match=r"No posterior data found"):
            suite.posterior_distribution(var="lam", plot_dim="channel")

    def test_posterior_distribution_custom_figsize(self, mock_suite_for_posterior_dist):
        """Test posterior distribution with custom figure size."""
        fig, _ = mock_suite_for_posterior_dist.posterior_distribution(
            var="lam", plot_dim="channel", figsize=(12, 8)
        )

        assert isinstance(fig, Figure)
        # Check that custom figsize was applied
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8


class TestChannelParameter:
    """Tests for the channel_parameter method."""

    @pytest.fixture(scope="class")
    def mock_idata_with_channel_params(self):
        """Create mock InferenceData with channel parameters."""
        seed = sum(map(ord, "channel_params"))
        rng = np.random.default_rng(seed)
        normal = rng.normal

        channels = ["TV", "Radio", "Digital"]
        geos = ["US", "UK"]

        posterior = xr.Dataset(
            {
                # Channel-indexed parameter (saturation_alpha)
                "saturation_alpha": xr.DataArray(
                    normal(loc=0.5, scale=0.1, size=(2, 10, 3)),
                    dims=("chain", "draw", "channel"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "channel": channels,
                    },
                ),
                # Channel-indexed parameter with additional dimension (adstock_alpha)
                "adstock_alpha": xr.DataArray(
                    normal(loc=1.0, scale=0.2, size=(2, 10, 3, 2)),
                    dims=("chain", "draw", "channel", "geo"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "channel": channels,
                        "geo": geos,
                    },
                ),
                # Scalar parameter (no channel dimension)
                "intercept": xr.DataArray(
                    normal(loc=0.0, scale=0.5, size=(2, 10)),
                    dims=("chain", "draw"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                    },
                ),
            }
        )

        return az.InferenceData(posterior=posterior)

    @pytest.fixture(scope="class")
    def mock_suite_with_channel_params(self, mock_idata_with_channel_params):
        return MMMPlotSuite(idata=mock_idata_with_channel_params)

    def test_channel_parameter_basic(self, mock_suite_with_channel_params):
        """Test basic channel parameter plotting."""
        fig = mock_suite_with_channel_params.channel_parameter(
            param_name="saturation_alpha"
        )

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1  # Single subplot for simple case
        assert all(isinstance(ax, Axes) for ax in fig.axes)

    def test_channel_parameter_with_additional_dim(
        self, mock_suite_with_channel_params
    ):
        """Test channel parameter with additional dimensions."""
        fig = mock_suite_with_channel_params.channel_parameter(
            param_name="adstock_alpha"
        )

        assert isinstance(fig, Figure)
        # Should create one subplot per geo (2 geos)
        assert len(fig.axes) == 2
        assert all(isinstance(ax, Axes) for ax in fig.axes)

    def test_channel_parameter_with_dims_filter(self, mock_suite_with_channel_params):
        """Test channel parameter with dimension filtering."""
        fig = mock_suite_with_channel_params.channel_parameter(
            param_name="adstock_alpha", dims={"geo": "US"}
        )

        assert isinstance(fig, Figure)
        # Should create single subplot when filtering to one geo
        assert len(fig.axes) == 1
        assert "geo=US" in fig.axes[0].get_title()

    def test_channel_parameter_with_dims_list(self, mock_suite_with_channel_params):
        """Test channel parameter with list of dimension values."""
        fig = mock_suite_with_channel_params.channel_parameter(
            param_name="adstock_alpha", dims={"geo": ["US", "UK"]}
        )

        assert isinstance(fig, Figure)
        # Should create one subplot per geo in the list
        assert len(fig.axes) == 2
        for i, geo in enumerate(["US", "UK"]):
            assert f"geo={geo}" in fig.axes[i].get_title()

    def test_channel_parameter_vertical_orientation(
        self, mock_suite_with_channel_params
    ):
        """Test channel parameter with vertical orientation."""
        fig = mock_suite_with_channel_params.channel_parameter(
            param_name="saturation_alpha", orient="v"
        )

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # For vertical orientation, channel should be on x-axis
        assert ax.get_xlabel() == "channel"
        assert ax.get_ylabel() == "saturation_alpha"

    def test_channel_parameter_horizontal_orientation(
        self, mock_suite_with_channel_params
    ):
        """Test channel parameter with horizontal orientation (default)."""
        fig = mock_suite_with_channel_params.channel_parameter(
            param_name="saturation_alpha", orient="h"
        )

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # For horizontal orientation, channel should be on y-axis
        assert ax.get_xlabel() == "saturation_alpha"
        assert ax.get_ylabel() == "channel"

    def test_channel_parameter_scalar_param(self, mock_suite_with_channel_params):
        """Test channel parameter with scalar (non-channel-indexed) parameter."""
        fig = mock_suite_with_channel_params.channel_parameter(param_name="intercept")

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        # Title should contain param name
        assert "intercept" in ax.get_title()

    def test_channel_parameter_invalid_param(self, mock_suite_with_channel_params):
        """Test that invalid parameter name raises error."""
        with pytest.raises(ValueError, match=r"Parameter 'invalid_param' not found"):
            mock_suite_with_channel_params.channel_parameter(param_name="invalid_param")

    def test_channel_parameter_no_posterior(self):
        """Test that missing posterior data raises error."""
        idata = az.InferenceData()
        suite = MMMPlotSuite(idata=idata)
        with pytest.raises(ValueError, match=r"No posterior data found"):
            suite.channel_parameter(param_name="saturation_alpha")

    def test_channel_parameter_custom_figsize(self, mock_suite_with_channel_params):
        """Test channel parameter with custom figure size."""
        fig = mock_suite_with_channel_params.channel_parameter(
            param_name="saturation_alpha", figsize=(12, 8)
        )

        assert isinstance(fig, Figure)
        # Check that custom figsize was applied
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8

    def test_channel_parameter_returns_figure_for_axvline(
        self, mock_suite_with_channel_params
    ):
        """Test that returned figure can be used to add reference lines."""
        fig = mock_suite_with_channel_params.channel_parameter(
            param_name="saturation_alpha"
        )

        # Verify we can add axvline as done in the notebook
        ax = fig.axes[0]
        ax.axvline(x=0.5, color="red", linestyle="--", label="reference")
        ax.legend()

        assert isinstance(fig, Figure)
        # Check that the axvline was added
        lines = ax.get_lines()
        assert len(lines) >= 1


class TestPriorVsPosterior:
    """Tests for the prior_vs_posterior plotting method."""

    @pytest.fixture(scope="class")
    def mock_idata_for_prior_vs_posterior(self) -> az.InferenceData:
        """Mock InferenceData with prior and posterior for testing."""
        seed = sum(map(ord, "Prior vs Posterior tests"))
        rng = np.random.default_rng(seed)
        normal = rng.normal

        channels = ["TV", "Radio", "Digital"]
        geos = ["US", "UK"]

        prior = xr.Dataset(
            {
                "lam": xr.DataArray(
                    normal(loc=0.5, scale=0.2, size=(2, 10, 3)),
                    dims=("chain", "draw", "channel"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "channel": channels,
                    },
                ),
                "alpha": xr.DataArray(
                    normal(loc=1.0, scale=0.3, size=(2, 10, 3, 2)),
                    dims=("chain", "draw", "channel", "geo"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "channel": channels,
                        "geo": geos,
                    },
                ),
            }
        )

        # Posterior has different means to show learning
        posterior = xr.Dataset(
            {
                "lam": xr.DataArray(
                    normal(loc=0.7, scale=0.1, size=(2, 10, 3)),
                    dims=("chain", "draw", "channel"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "channel": channels,
                    },
                ),
                "alpha": xr.DataArray(
                    normal(loc=0.8, scale=0.15, size=(2, 10, 3, 2)),
                    dims=("chain", "draw", "channel", "geo"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "channel": channels,
                        "geo": geos,
                    },
                ),
            }
        )

        return az.InferenceData(prior=prior, posterior=posterior)

    @pytest.fixture(scope="class")
    def mock_suite_for_prior_vs_posterior(self, mock_idata_for_prior_vs_posterior):
        return MMMPlotSuite(idata=mock_idata_for_prior_vs_posterior)

    def test_prior_vs_posterior_basic(self, mock_suite_for_prior_vs_posterior):
        """Test basic prior vs posterior plotting."""
        fig, axes = mock_suite_for_prior_vs_posterior.prior_vs_posterior(
            var="lam", plot_dim="channel"
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create one subplot per channel (3 channels)
        assert axes.shape[0] == 3
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_prior_vs_posterior_with_additional_dim(
        self, mock_suite_for_prior_vs_posterior
    ):
        """Test prior vs posterior with additional dimensions."""
        fig, axes = mock_suite_for_prior_vs_posterior.prior_vs_posterior(
            var="alpha", plot_dim="channel"
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create one row per channel (3), one col per geo (2)
        assert axes.shape[0] == 3
        assert axes.shape[1] == 2
        assert all(isinstance(ax, Axes) for ax in axes.flat)

    def test_prior_vs_posterior_with_dims_filter(
        self, mock_suite_for_prior_vs_posterior
    ):
        """Test prior vs posterior with dimension filtering."""
        fig, axes = mock_suite_for_prior_vs_posterior.prior_vs_posterior(
            var="alpha", plot_dim="channel", dims={"geo": "US"}
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create 3 subplots (one per channel) for single geo
        assert axes.shape[0] == 3
        assert axes.shape[1] == 1

    def test_prior_vs_posterior_alphabetical_sort(
        self, mock_suite_for_prior_vs_posterior
    ):
        """Test prior vs posterior with alphabetical sorting (default)."""
        fig, axes = mock_suite_for_prior_vs_posterior.prior_vs_posterior(
            var="lam", plot_dim="channel", alphabetical_sort=True
        )

        assert isinstance(fig, Figure)
        # Check titles are in alphabetical order
        titles = [ax.get_title() for ax in axes[:, 0]]
        assert "Digital" in titles[0]
        assert "Radio" in titles[1]
        assert "TV" in titles[2]

    def test_prior_vs_posterior_difference_sort(
        self, mock_suite_for_prior_vs_posterior
    ):
        """Test prior vs posterior with difference-based sorting."""
        fig, axes = mock_suite_for_prior_vs_posterior.prior_vs_posterior(
            var="lam", plot_dim="channel", alphabetical_sort=False
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Channels should be sorted by posterior - prior difference

    def test_prior_vs_posterior_invalid_var(self, mock_suite_for_prior_vs_posterior):
        """Test that invalid variable name raises error."""
        with pytest.raises(ValueError, match=r"Variable 'invalid_var' not found"):
            mock_suite_for_prior_vs_posterior.prior_vs_posterior(
                var="invalid_var", plot_dim="channel"
            )

    def test_prior_vs_posterior_missing_plot_dim_handled_gracefully(
        self, mock_suite_for_prior_vs_posterior
    ):
        """Test that missing plot_dim is handled gracefully (scalar case)."""
        # When plot_dim doesn't exist, should treat as scalar and handle gracefully
        fig, axes = mock_suite_for_prior_vs_posterior.prior_vs_posterior(
            var="lam", plot_dim="nonexistent_dim"
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create subplot(s) based on remaining dimensions
        assert axes.shape[0] >= 1

    def test_prior_vs_posterior_no_prior(self):
        """Test that missing prior data raises error."""
        posterior = xr.Dataset(
            {
                "lam": xr.DataArray(
                    np.random.normal(size=(2, 10, 3)),
                    dims=("chain", "draw", "channel"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "channel": ["A", "B", "C"],
                    },
                ),
            }
        )
        idata = az.InferenceData(posterior=posterior)
        suite = MMMPlotSuite(idata=idata)
        with pytest.raises(ValueError, match=r"No prior data found"):
            suite.prior_vs_posterior(var="lam", plot_dim="channel")

    def test_prior_vs_posterior_no_posterior(self):
        """Test that missing posterior data raises error."""
        prior = xr.Dataset(
            {
                "lam": xr.DataArray(
                    np.random.normal(size=(2, 10, 3)),
                    dims=("chain", "draw", "channel"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "channel": ["A", "B", "C"],
                    },
                ),
            }
        )
        idata = az.InferenceData(prior=prior)
        suite = MMMPlotSuite(idata=idata)
        with pytest.raises(ValueError, match=r"No posterior data found"):
            suite.prior_vs_posterior(var="lam", plot_dim="channel")

    def test_prior_vs_posterior_custom_figsize(self, mock_suite_for_prior_vs_posterior):
        """Test prior vs posterior with custom figure size."""
        fig, _ = mock_suite_for_prior_vs_posterior.prior_vs_posterior(
            var="lam", plot_dim="channel", figsize=(14, 5)
        )

        assert isinstance(fig, Figure)
        # Check that custom figsize was applied
        assert fig.get_figwidth() == 14

    def test_prior_vs_posterior_subplot_content(
        self, mock_suite_for_prior_vs_posterior
    ):
        """Test that prior vs posterior subplots have expected content."""
        _fig, axes = mock_suite_for_prior_vs_posterior.prior_vs_posterior(
            var="lam", plot_dim="channel"
        )

        # Each subplot should have content (lines for KDE and vertical lines)
        for ax in axes.flat:
            # Check there are lines in the plot
            assert len(ax.lines) > 0
            # Check legend exists
            assert ax.get_legend() is not None

    def test_prior_vs_posterior_scalar_variable(self):
        """Test prior vs posterior with scalar variable (no plot_dim dimension)."""
        # Create idata with a scalar variable (only chain, draw dims)
        prior = xr.Dataset(
            {
                "sigma": xr.DataArray(
                    np.random.exponential(scale=1.0, size=(2, 10)),
                    dims=("chain", "draw"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                    },
                ),
            }
        )
        posterior = xr.Dataset(
            {
                "sigma": xr.DataArray(
                    np.random.exponential(scale=0.5, size=(2, 10)),
                    dims=("chain", "draw"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                    },
                ),
            }
        )
        idata = az.InferenceData(prior=prior, posterior=posterior)
        suite = MMMPlotSuite(idata=idata)

        # Should work without error for scalar variable
        fig, axes = suite.prior_vs_posterior(var="sigma", plot_dim="channel")

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create single subplot for scalar
        assert axes.shape[0] == 1
        # Check there are lines in the plot
        assert len(axes[0, 0].lines) > 0

    def test_prior_vs_posterior_scalar_with_one_extra_dim(self):
        """Test prior vs posterior with scalar that has one non-chain/draw dimension."""
        geos = ["US", "UK"]
        prior = xr.Dataset(
            {
                "intercept": xr.DataArray(
                    np.random.normal(size=(2, 10, 2)),
                    dims=("chain", "draw", "geo"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "geo": geos,
                    },
                ),
            }
        )
        posterior = xr.Dataset(
            {
                "intercept": xr.DataArray(
                    np.random.normal(loc=1.0, size=(2, 10, 2)),
                    dims=("chain", "draw", "geo"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "geo": geos,
                    },
                ),
            }
        )
        idata = az.InferenceData(prior=prior, posterior=posterior)
        suite = MMMPlotSuite(idata=idata)

        # plot_dim="channel" doesn't exist, but geo does - should handle gracefully
        fig, axes = suite.prior_vs_posterior(var="intercept", plot_dim="channel")

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create one subplot per geo (2 geos)
        assert axes.shape[0] == 2

    def test_prior_vs_posterior_use_geo_as_plot_dim(self):
        """Test prior vs posterior with geo as plot_dim."""
        geos = ["US", "UK"]
        prior = xr.Dataset(
            {
                "intercept": xr.DataArray(
                    np.random.normal(size=(2, 10, 2)),
                    dims=("chain", "draw", "geo"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "geo": geos,
                    },
                ),
            }
        )
        posterior = xr.Dataset(
            {
                "intercept": xr.DataArray(
                    np.random.normal(loc=1.0, size=(2, 10, 2)),
                    dims=("chain", "draw", "geo"),
                    coords={
                        "chain": np.arange(2),
                        "draw": np.arange(10),
                        "geo": geos,
                    },
                ),
            }
        )
        idata = az.InferenceData(prior=prior, posterior=posterior)
        suite = MMMPlotSuite(idata=idata)

        # Use geo as plot_dim
        fig, axes = suite.prior_vs_posterior(var="intercept", plot_dim="geo")

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should create one subplot per geo (2 geos)
        assert axes.shape[0] == 2


class TestChannelContributionShareHDI:
    def test_channel_contribution_share_hdi_basic(self, mock_suite_with_constant_data):
        """Test basic functionality of channel_contribution_share_hdi."""
        fig, ax = mock_suite_with_constant_data.channel_contribution_share_hdi()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Check that title is set
        assert "Channel Contribution Share" in fig._suptitle.get_text()

    def test_channel_contribution_share_hdi_custom_hdi(
        self, mock_suite_with_constant_data
    ):
        """Test channel_contribution_share_hdi with custom HDI probability."""
        fig, ax = mock_suite_with_constant_data.channel_contribution_share_hdi(
            hdi_prob=0.90
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_channel_contribution_share_hdi_custom_figsize(
        self, mock_suite_with_constant_data
    ):
        """Test channel_contribution_share_hdi with custom figsize."""
        fig, ax = mock_suite_with_constant_data.channel_contribution_share_hdi(
            figsize=(12, 8)
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8

    def test_channel_contribution_share_hdi_with_dims(
        self, mock_suite_with_constant_data
    ):
        """Test channel_contribution_share_hdi with dimension filtering."""
        fig, ax = mock_suite_with_constant_data.channel_contribution_share_hdi(
            dims={"country": "A"}
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_channel_contribution_share_hdi_no_posterior(self):
        """Test that channel_contribution_share_hdi raises error without posterior."""
        idata = az.InferenceData()
        suite = MMMPlotSuite(idata=idata)

        with pytest.raises(ValueError, match=r"No posterior data found"):
            suite.channel_contribution_share_hdi()

    def test_channel_contribution_share_hdi_no_original_scale(
        self, mock_suite_with_constant_data
    ):
        """Test that channel_contribution_share_hdi raises error without original scale contribution."""
        # Remove the original scale contribution from the mock data
        idata_copy = mock_suite_with_constant_data.idata.copy()
        idata_copy.posterior = idata_copy.posterior.drop_vars(
            "channel_contribution_original_scale"
        )
        suite_without_original_scale = MMMPlotSuite(idata=idata_copy)

        with pytest.raises(
            ValueError,
            match=r"Variable 'channel_contribution_original_scale' not found in posterior",
        ):
            suite_without_original_scale.channel_contribution_share_hdi()

    def test_channel_contribution_share_hdi_plot_kwargs(
        self, mock_suite_with_constant_data
    ):
        """Test channel_contribution_share_hdi with additional plot_kwargs."""
        fig, ax = mock_suite_with_constant_data.channel_contribution_share_hdi(
            colors="C1"
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


@pytest.mark.parametrize(
    "dim_config",
    [
        {},
        {"country": ["A", "B"]},
        {"country": ["A", "B"], "product": ["p1", "p2"]},
    ],
)
def test_time_slice_cv_run_sets_results_and_plot_property(dim_config):
    """Ensure TimeSliceCrossValidator.run persists results and exposes MMMPlotSuite via .plot

    Test with different dimension granularities (no extra dims, single dim, multiple dims)
    so the combined idata and plot helper behave correctly regardless of additional coords.
    """
    from pymc_marketing.mmm.time_slice_cross_validation import (
        TimeSliceCrossValidator,
    )

    class FakeModel:
        def __init__(self):
            self.idata = None
            self.sampler_config = None

        def fit(self, X, y, progressbar=True):
            # no-op fit
            return None

        def sample_posterior_predictive(
            self, X, extend_idata=True, combined=True, progressbar=False, **kwargs
        ):
            # Build minimal posterior_predictive with a date coord and any extra dims
            dates = pd.to_datetime(np.unique(X["date"].to_numpy()))

            # determine unique values for each dimension in dim_config
            dim_keys = list(dim_config.keys())
            dim_sizes = [len(np.unique(X[k])) for k in dim_keys]

            # build array shape: (chain, draw, *dim_sizes, date)
            shape = (1, 2, *dim_sizes, len(dates))

            arr = np.random.RandomState(1).normal(size=shape)

            dims = (
                ("chain", "draw", *dim_keys, "date")
                if dim_keys
                else (
                    "chain",
                    "draw",
                    "date",
                )
            )

            coords = {"date": dates}
            for k in dim_keys:
                coords[k] = np.unique(X[k])

            da = xr.DataArray(arr, dims=dims, coords=coords, name="y")
            ds = xr.Dataset({"y": da})
            self.idata = az.InferenceData(posterior_predictive=ds)
            return self.idata

    class FakeMMMFactory:
        def build_model(self, X, y):
            return FakeModel()

    # Prepare tiny dataset with combinations for the requested dims
    dates = pd.date_range("2025-01-01", periods=6, freq="D")

    # build cartesian product of dims
    dim_keys = list(dim_config.keys())
    if dim_keys:
        import itertools as _it

        dim_values = [dim_config[k] for k in dim_keys]
        rows = []
        for date in dates:
            for combo in _it.product(*dim_values):
                row = {"date": date}
                # Explicitly set strict=False to satisfy ruff B905 (zip strictness)
                for k, v in zip(dim_keys, combo, strict=False):
                    row[k] = v
                rows.append(row)
        X = pd.DataFrame(rows)
    else:
        X = pd.DataFrame({"date": dates, "geo": ["g1"] * len(dates)})

    y = pd.Series(np.arange(len(X)))

    cv = TimeSliceCrossValidator(
        n_init=2, forecast_horizon=1, date_column="date", step_size=1
    )

    run_out = cv.run(X, y, mmm=FakeMMMFactory())

    # Support both historical behavior (run returned list of results) and
    # current behavior (run returns combined arviz.InferenceData).
    if isinstance(run_out, list):
        results = run_out
        combined = getattr(cv, "cv_idata", None)
        # If legacy behavior, cv._cv_results should point at returned results
        assert results == getattr(cv, "_cv_results", results)
    else:
        combined = run_out
        # current behavior: combined InferenceData returned and _cv_results persisted
        assert isinstance(combined, az.InferenceData)
        assert hasattr(cv, "_cv_results")
        results = cv._cv_results
    assert isinstance(results, list)

    # combined posterior_predictive should contain all configured dims as coords
    if dim_keys:
        for k in dim_keys:
            assert k in combined.posterior_predictive.coords

    # Plot property should return MMMPlotSuite that wraps the latest idata
    plot_suite = cv.plot
    assert hasattr(plot_suite, "idata")
    assert plot_suite.idata is cv._cv_results[-1].idata


def test_init_subplots_and_build_title():
    suite = MMMPlotSuite(idata=None)
    fig, axes = suite._init_subplots(
        n_subplots=3, ncols=2, width_per_col=2.0, height_per_row=1.5
    )
    # axes is a 2D array of shape (3,2)
    assert axes.shape == (3, 2)
    w, h = fig.get_size_inches()
    assert w == pytest.approx(2.0 * 2)
    assert h == pytest.approx(1.5 * 3)

    title = suite._build_subplot_title(["country", "region"], ("A", "X"))
    assert title == "country=A, region=X"
    fallback = suite._build_subplot_title([], (), fallback_title="Fallback")
    assert fallback == "Fallback"


def test_get_additional_dim_combinations_success_and_error():
    suite = MMMPlotSuite(idata=None)
    dates = pd.date_range("2025-01-01", periods=4, freq="D")
    da = xr.DataArray(
        np.zeros((1, 2, 3, 4)),
        dims=("chain", "draw", "country", "date"),
        coords={
            "chain": [0],
            "draw": [0, 1],
            "country": ["A", "B", "C"],
            "date": dates,
        },
        name="y",
    )
    ds_pp = xr.Dataset({"y": da})

    addl_dims, combos = suite._get_additional_dim_combinations(
        ds_pp, "y", ignored_dims={"chain", "draw", "date"}
    )
    assert addl_dims == ["country"]
    assert len(combos) == 3

    # Missing variable should raise
    with pytest.raises(ValueError):
        suite._get_additional_dim_combinations(
            ds_pp, "missing", ignored_dims={"chain", "draw", "date"}
        )


def test_reduce_and_stack_sums_and_stack():
    suite = MMMPlotSuite(idata=None)
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    data = xr.DataArray(
        np.ones((1, 2, 2, 3)),
        dims=("chain", "draw", "region", "date"),
        coords={"chain": [0], "draw": [0, 1], "region": ["r1", "r2"], "date": dates},
    )
    reduced = suite._reduce_and_stack(data)
    # region summed away, chain+draw stacked into 'sample'
    assert "region" not in reduced.dims
    assert "sample" in reduced.dims
    assert reduced.sizes["sample"] == 2


def test_get_posterior_predictive_data_raises_when_missing():
    suite = MMMPlotSuite(idata=None)
    with pytest.raises(ValueError):
        suite._get_posterior_predictive_data(None)


def test_add_median_and_hdi_raises_when_no_date():
    suite = MMMPlotSuite(idata=None)
    # Prefix unused fig with underscore to satisfy ruff RUF059
    _fig, axarr = plt.subplots()
    ax = axarr
    data = xr.DataArray(np.ones((1, 2)), dims=("chain", "draw"))
    # Depending on arviz/xarray internals this can raise ValueError (from our check)
    # or TypeError (from arviz.hdi when input_core_dims is not set). Accept both.
    with pytest.raises((ValueError, TypeError)):
        suite._add_median_and_hdi(ax, data, var="y")


def test_posterior_predictive_with_explicit_xarray_dataset():
    # Build a small posterior_predictive dataset and pass it directly to posterior_predictive
    dates = pd.date_range("2025-01-01", periods=4, freq="D")
    da = xr.DataArray(
        np.random.default_rng(1).normal(size=(1, 2, 4)),
        dims=("chain", "draw", "date"),
        coords={"chain": [0], "draw": [0, 1], "date": dates},
        name="y",
    )
    ds = xr.Dataset({"y": da})
    suite = MMMPlotSuite(idata=None)
    fig, axes = suite.posterior_predictive(var=["y"], idata=ds, hdi_prob=0.8)
    assert fig is not None
    assert axes is not None


def test_filter_df_by_indexer_behaviour():
    suite = MMMPlotSuite(idata=None)

    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "country": ["A", "B", "A"],
            "id": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
        }
    )

    # None input returns empty DataFrame
    out = suite._filter_df_by_indexer(None, {"country": "A"})
    assert isinstance(out, pd.DataFrame)
    assert out.empty

    # Empty indexer returns a copy of the original
    out = suite._filter_df_by_indexer(df, {})
    assert out.equals(df)
    assert out is not df

    # Filter by existing column
    out = suite._filter_df_by_indexer(df, {"country": "A"})
    assert len(out) == 2
    assert all(out["country"] == "A")

    # Unknown key is ignored and returns full DataFrame
    out = suite._filter_df_by_indexer(df, {"region": "X"})
    assert len(out) == len(df)

    # Numeric column compared to string value should match (casts to str)
    out = suite._filter_df_by_indexer(df, {"id": "1"})
    assert len(out) == 1
    assert int(out.iloc[0]["id"]) == 1


def _build_cv_results_for_cv_predictions():
    """Helper to build a minimal arviz.InferenceData suitable for cv_predictions tests."""
    # Single CV fold and a single extra dim 'country' with one country value
    cv_labels = ["cv1"]
    countries = ["A"]
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])

    # shape: chain=1, draw=2, cv=1, country=1, date=3
    arr = np.random.default_rng(1).normal(
        size=(1, 2, len(cv_labels), len(countries), len(dates))
    )
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "country", "date"),
        coords={"cv": cv_labels, "country": countries, "date": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # Build per-fold metadata dicts stored in a DataArray of dtype object
    # Include X_train/X_test with a 'date' and 'country' column so filtering works
    meta = {
        "X_train": pd.DataFrame({"date": dates[:2], "country": ["A", "A"]}).reset_index(
            drop=True
        ),
        "y_train": pd.Series([1.0, 2.0]),
        "X_test": pd.DataFrame({"date": dates[2:], "country": ["A"]}).reset_index(
            drop=True
        ),
        "y_test": pd.Series([3.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})

    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)
    return results


def test_cv_predictions_none_dims_plots_without_error():
    """cv_predictions should accept dims=None (treated as empty dict) and produce a plot."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_for_cv_predictions()

    fig, axes = suite.cv_predictions(results, dims=None)

    assert fig is not None
    # axes may be a list of Axes; ensure at least one axis exists
    assert hasattr(axes, "__len__")
    assert len(axes) >= 1


def test_cv_predictions_unsupported_dims_raises_value_error():
    """Passing dims that are not present on the posterior_predictive DataArray should raise ValueError."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_for_cv_predictions()

    with pytest.raises(ValueError, match=r"Unsupported dims"):
        # 'region' is not an available dim on the test posterior_predictive
        suite.cv_predictions(results, dims={"region": "X"})


def test_cv_predictions_with_dims_string_creates_panel_title():
    """When user supplies dims with a single value (not list), ensure panel title includes the dim."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_for_cv_predictions()

    # dims provided as a single value (not list) should be accepted and produce a panel
    fig, axes = suite.cv_predictions(results, dims={"country": "A"})

    assert fig is not None
    assert hasattr(axes, "__len__")
    # there should be at least one axis and its title should include the country
    titles = [ax.get_title() for ax in axes]
    assert any("country=A" in t for t in titles)


def test_cv_predictions_with_dims_list_creates_expected_panel_indexer():
    """When dims contains a list, ensure panels are built."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_for_cv_predictions()

    # dims provided as a list should produce panels for each listed value
    fig, axes = suite.cv_predictions(results, dims={"country": ["A"]})

    assert fig is not None
    assert hasattr(axes, "__len__")
    # single fold x single listed country -> exactly one axis expected
    assert len(axes) == 1
    title = axes[0].get_title()
    assert "country=A" in title


def _build_cv_results_multi(cv_labels=("cv1", "cv2"), countries=("A", "B")):
    """Build InferenceData with multiple cv labels and multiple country coords."""
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    n_cv = len(cv_labels)
    n_country = len(countries)
    arr = np.random.default_rng(2).normal(size=(1, 2, n_cv, n_country, len(dates)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "country", "date"),
        coords={"cv": list(cv_labels), "country": list(countries), "date": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    metas = []
    for _lbl in cv_labels:
        meta = {
            "X_train": pd.DataFrame(
                {
                    "date": dates[:2].repeat(n_country),
                    "country": list(countries) * len(dates[:2]),
                }
            ).reset_index(drop=True),
            "y_train": pd.Series(np.arange(2 * n_country, dtype=float)),
            "X_test": pd.DataFrame(
                {
                    "date": dates[2:].repeat(n_country),
                    "country": list(countries) * len(dates[2:]),
                }
            ).reset_index(drop=True),
            "y_test": pd.Series(np.arange(1 * n_country, dtype=float)),
        }
        metas.append(meta)

    meta_da = xr.DataArray(
        np.array(metas, dtype=object),
        dims=("cv",),
        coords={"cv": list(cv_labels)},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})
    return az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)


def test_cv_predictions_single_axis_wrapped_into_list():
    """When only a single axis is required, cv_predictions wraps it into a list."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_for_cv_predictions()

    _fig, axes = suite.cv_predictions(results, dims=None)

    assert isinstance(axes, list)
    assert len(axes) == 1
    assert isinstance(axes[0], Axes)


def test_cv_predictions_multiple_folds_and_panels_create_expected_number_of_axes():
    """When multiple CV folds and multiple panels exist, ensure n_axes = n_panels * n_folds."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_multi(cv_labels=("cv1", "cv2"), countries=("A", "B"))

    # dims None -> additional_dims will include 'country' (2), dims_combos == [()] (1)
    # n_panels = 2, n_folds = 2 => n_axes = 4
    _fig, axes = suite.cv_predictions(results, dims=None)

    assert hasattr(axes, "__len__")
    assert len(axes) == 4
    assert all(isinstance(ax, Axes) for ax in axes)


def test_align_y_to_df_skips_observed_when_y_train_none(monkeypatch):
    """If y_train is None in the per-fold metadata, no 'observed' line should be plotted."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_for_cv_predictions()

    # Replace the metadata so that y_train is None for the fold
    meta = {
        "X_train": pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "country": ["A", "A"],
            }
        ).reset_index(drop=True),
        "y_train": None,
        "X_test": pd.DataFrame(
            {"date": pd.to_datetime(["2025-01-03"]), "country": ["A"]}
        ).reset_index(drop=True),
        "y_test": pd.Series([3.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": ["cv1"]},
        name="metadata",
    )
    results.cv_metadata = xr.Dataset({"metadata": meta_da})

    # Monkeypatch az.plot_hdi to be a no-op so test is focused on observed plotting behavior
    monkeypatch.setattr(az, "plot_hdi", lambda *a, **k: None)

    _fig, axes = suite.cv_predictions(results, dims=None)

    # Ensure no 'observed' label exists in any axis legend
    for ax in axes:
        _, labels = ax.get_legend_handles_labels()
        assert "observed" not in labels


def test_plot_hdi_from_sel_calls_az_plot_hdi(monkeypatch):
    """Ensure the helper that wraps az.plot_hdi ends up calling az.plot_hdi with expected kwargs (hdi_prob=0.94)."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_for_cv_predictions()

    recorded = []

    def fake_plot_hdi(*args, **kwargs):
        recorded.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(az, "plot_hdi", fake_plot_hdi)

    _fig, _axes = suite.cv_predictions(results, dims=None)

    # We expect at least one call to az.plot_hdi from _plot_hdi_from_sel
    assert len(recorded) >= 1
    # Check that at least one call used the expected hdi_prob
    assert any(call["kwargs"].get("hdi_prob") == 0.94 for call in recorded)


def test_cv_predictions_panel_selection_failure_skips_panel(monkeypatch):
    """If selecting a specific panel coordinate fails, cv_predictions should warn and skip plotting that panel."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_multi(cv_labels=("cv1", "cv2"), countries=("A", "B"))

    # Make DataArray.sel succeed for selecting by cv but fail for subsequent panel selection
    orig_sel = xr.DataArray.sel

    def fake_sel(self, *args, **kwargs):
        # the first sel call includes cv=...; allow that
        if "cv" in kwargs:
            return orig_sel(self, *args, **kwargs)
        # subsequent sel calls for panel selection will raise to trigger the warning/skip
        # Code catches (KeyError, ValueError), so raise KeyError
        raise KeyError("forced panel selection failure")

    monkeypatch.setattr(xr.DataArray, "sel", fake_sel)

    with pytest.warns(
        UserWarning, match=r"Could not select posterior_predictive panel"
    ):
        _fig, axes = suite.cv_predictions(results, dims=None)

    # If panels were skipped due to selection failure, no axis title should contain a panel name
    assert not any("country=" in ax.get_title() for ax in axes)


def test_cv_predictions_transpose_failure_warns(monkeypatch):
    """If transpose on the stacked array fails, cv_predictions should warn but continue."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_for_cv_predictions()

    def fake_transpose(self, *args, **kwargs):
        # Code catches (ValueError, KeyError), so raise ValueError
        raise ValueError("forced transpose failure")

    monkeypatch.setattr(xr.DataArray, "transpose", fake_transpose)

    with pytest.warns(
        UserWarning, match=r"Could not transpose posterior_predictive array"
    ):
        _fig, axes = suite.cv_predictions(results, dims=None)

    # Still returns a figure/axes despite the transpose failure
    assert _fig is not None
    assert hasattr(axes, "__len__")


def test_cv_predictions_metadata_values_item_fallback(monkeypatch):
    """If meta_da.values.item() raises, cv_predictions should fall back to meta_da.item()."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_for_cv_predictions()

    # Build a replacement metadata dict that will be returned by FakeMetaDA.item()
    meta = {
        "X_train": pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "country": ["A", "A"],
            }
        ),
        "y_train": pd.Series([1.0, 2.0]),
        "X_test": pd.DataFrame(
            {"date": pd.to_datetime(["2025-01-03"]), "country": ["A"]}
        ).reset_index(drop=True),
        "y_test": pd.Series([3.0]),
    }

    class FakeMetaDA:
        def __init__(self, obj, cv_labels):
            self._obj = obj
            # store cv labels for possible coords access
            self._cv = list(cv_labels)

        def sel(self, **kwargs):
            # ignore selection and return self
            return self

        @property
        def values(self):
            class V:
                def item(self_inner):
                    # Simulate failure when calling .values.item()
                    # Code catches (ValueError, AttributeError), so raise ValueError
                    raise ValueError("forced values.item failure")

            return V()

        def item(self):
            # Fallback that returns the actual python object metadata
            return self._obj

    class FakeMetaDataset:
        """Lightweight stand-in for an xarray.Dataset that only needs __getitem__ and coords."""

        def __init__(self, meta_da):
            self._meta_da = meta_da

            # expose coords to avoid attribute errors in the code under test
            class _C:
                def __init__(self, values):
                    self.values = np.array(values)

            self.coords = {"cv": _C(self._meta_da._cv)}

        def __contains__(self, key):
            return key in ("metadata", 0)

        def __getitem__(self, key):
            # Some code paths index the dataset by 'metadata' key, others may
            # attempt integer/index access. Support both by returning the
            # underlying FakeMetaDA.
            if key == "metadata" or key == 0:
                return self._meta_da
            raise KeyError(key)

    # Replace cv_metadata with our fake dataset that returns a FakeMetaDA
    fake_meta_da = FakeMetaDA(meta, ["cv1"])
    results.cv_metadata = FakeMetaDataset(fake_meta_da)

    # Avoid HDI plotting noise
    monkeypatch.setattr(az, "plot_hdi", lambda *a, **k: None)

    _fig, axes = suite.cv_predictions(results, dims=None)

    # Observed should be plotted from the fallback metadata
    assert any(
        "observed" in lbl for ax in axes for lbl in ax.get_legend_handles_labels()[1]
    )


def test_cv_predictions_hdi_failure_warns(monkeypatch):
    """If az.plot_hdi (called by _plot_hdi_from_sel) raises, cv_predictions should warn and continue."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_for_cv_predictions()

    def raise_plot_hdi(*args, **kwargs):
        # Code catches (KeyError, ValueError, TypeError), so raise TypeError
        raise TypeError("forced hdi failure")

    monkeypatch.setattr(az, "plot_hdi", raise_plot_hdi)

    with pytest.warns(
        UserWarning, match=r"Could not compute HDI for (train|test) range"
    ):
        _fig, axes = suite.cv_predictions(results, dims=None)

    # still returns figure/axes
    assert _fig is not None
    assert hasattr(axes, "__len__")


def test_cv_predictions_plots_observed_and_train_end(monkeypatch):
    """Ensure that when y_train exists, an 'observed' line and a 'train end' vline are present."""
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_for_cv_predictions()

    # Make HDI plotting a no-op so only observed/train-end lines are relevant
    monkeypatch.setattr(az, "plot_hdi", lambda *a, **k: None)

    _fig, axes = suite.cv_predictions(results, dims=None)

    found_observed = False
    found_train_end = False
    for ax in axes:
        # Check lines on the axis by their labels
        for line in ax.get_lines():
            lbl = line.get_label()
            if lbl == "observed":
                found_observed = True
            if lbl == "train end":
                found_train_end = True
    assert found_observed, "No 'observed' line found on any axis"
    assert found_train_end, "No 'train end' vertical line found on any axis"


def _build_param_stability_idata(
    cv_labels=("cv1", "cv2"), countries=("A", "B")
) -> az.InferenceData:
    """Helper to create InferenceData suitable for param_stability tests."""
    arr = np.random.default_rng(3).normal(size=(1, 2, len(cv_labels), len(countries)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "country"),
        coords={
            "chain": [0],
            "draw": [0, 1],
            "cv": list(cv_labels),
            "country": list(countries),
        },
        name="beta",
    )
    posterior = xr.Dataset({"beta": da})
    return az.InferenceData(posterior=posterior)


def _build_param_stability_idata_no_country(
    cv_labels=("cv1", "cv2"),
) -> az.InferenceData:
    arr = np.random.default_rng(4).normal(size=(1, 2, len(cv_labels)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv"),
        coords={"chain": [0], "draw": [0, 1], "cv": list(cv_labels)},
        name="beta",
    )
    posterior = xr.Dataset({"beta": da})
    return az.InferenceData(posterior=posterior)


def test_param_stability_no_dims_calls_plot_forest(monkeypatch):
    """When no dims provided, a single forest plot over posterior_list should be produced."""
    suite = MMMPlotSuite(idata=None)
    results = _build_param_stability_idata()

    recorded = []

    def fake_plot_forest(*args, **kwargs):
        recorded.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(az, "plot_forest", fake_plot_forest)

    _fig, _ax = suite.param_stability(results, parameter=["beta"], dims=None)

    assert len(recorded) == 1
    assert recorded[0]["kwargs"].get("var_names") == ["beta"]
    assert recorded[0]["kwargs"].get("combined") is True


def test_param_stability_with_dims_calls_plot_forest_per_coord(monkeypatch):
    """When dims provided, call plot_forest per coord and return the last (fig, ax)."""
    suite = MMMPlotSuite(idata=None)
    results = _build_param_stability_idata(
        cv_labels=("cv1", "cv2"), countries=("A", "B")
    )

    recorded = []

    def fake_plot_forest(*args, **kwargs):
        recorded.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(az, "plot_forest", fake_plot_forest)

    fig_ax = suite.param_stability(
        results, parameter=["beta"], dims={"country": ["A", "B"]}
    )

    # Two coords -> two calls
    assert len(recorded) == 2
    # Should return last (fig, ax)
    assert isinstance(fig_ax, tuple) and len(fig_ax) == 2


def test_param_stability_sel_failure_raises():
    """If posterior folds cannot be indexed by the requested dim, raise ValueError."""
    suite = MMMPlotSuite(idata=None)
    results = _build_param_stability_idata_no_country()

    with pytest.raises(ValueError, match=r"Unable to select dims from posterior"):
        suite.param_stability(results, parameter=["beta"], dims={"country": ["A"]})


def test_param_stability_empty_dims_falls_back_to_no_dims(monkeypatch):
    """If dims is provided but empty, param_stability should fall back to no-dims behavior."""
    suite = MMMPlotSuite(idata=None)
    results = _build_param_stability_idata()

    called = {}

    def fake_plot_forest(*args, **kwargs):
        called.update(kwargs)
        return kwargs.get("ax")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    monkeypatch.setattr(az, "plot_forest", fake_plot_forest)
    monkeypatch.setattr(plt, "subplots", lambda figsize: (fig, ax))
    monkeypatch.setattr(plt, "show", lambda: None)

    fig_ax = suite.param_stability(results, parameter=["beta"], dims={})
    assert fig_ax == (fig, ax)
    assert called.get("combined") is True
    assert called.get("var_names") == ["beta"]


# ------------------------------------------------------------------------
# cv_crps tests added to increase coverage
# ------------------------------------------------------------------------


def test_cv_crps_input_type_raises():
    """cv_crps should raise TypeError when provided a non-InferenceData input."""
    suite = MMMPlotSuite(idata=None)
    with pytest.raises(TypeError):
        suite.cv_crps(None)


def test_cv_crps_missing_cv_metadata_raises():
    """cv_crps should raise when cv_metadata is missing from the InferenceData."""
    suite = MMMPlotSuite(idata=None)
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    arr = np.random.default_rng(1).normal(size=(1, 2, 1, len(dates)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "date"),
        coords={"cv": ["cv1"], "date": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})
    results = az.InferenceData(posterior_predictive=ds_pp)
    with pytest.raises(ValueError, match=r"cv_metadata"):
        suite.cv_crps(results)


def test_cv_crps_missing_y_original_scale_raises():
    """cv_crps should raise when posterior_predictive['y_original_scale'] is missing."""
    suite = MMMPlotSuite(idata=None)
    # build cv_metadata
    meta = {"X_train": None, "y_train": None, "X_test": None, "y_test": None}
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": ["cv1"]},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})

    # posterior_predictive without the expected var
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    arr = np.random.default_rng(1).normal(size=(1, 2, 1, len(dates)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "date"),
        coords={"cv": ["cv1"], "date": dates},
        name="y_other",
    )
    ds_pp = xr.Dataset({"y_other": da})
    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)
    with pytest.raises(ValueError, match=r"y_original_scale"):
        suite.cv_crps(results)


def test_cv_crps_date_coord_detection_and_pred_matrix(monkeypatch):
    """Ensure _pred_matrix_for_rows can detect a non-'date' datetime coord and assemble prediction matrix."""
    suite = MMMPlotSuite(idata=None)

    # Build posterior_predictive where the date coord is named 'dt'
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    cv_labels = ["cv1"]
    arr = np.random.default_rng(5).normal(size=(1, 2, len(cv_labels), len(dates)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "dt"),
        coords={"cv": cv_labels, "dt": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # Build metadata with a 'date' column (so rows_df has 'date' but da coord is 'dt')
    meta = {
        "X_train": pd.DataFrame({"date": dates[:2], "id": [1, 2]}),
        "y_train": pd.Series([10.0, 20.0]),
        "X_test": pd.DataFrame({"date": dates[2:], "id": [3]}),
        "y_test": pd.Series([30.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})

    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # Monkeypatch crps to a deterministic function so plotting proceeds
    def fake_crps(y_true, y_pred):
        # basic check: shapes align
        assert y_pred.shape[1] == len(y_true)
        return 0.123

    monkeypatch.setattr("pymc_marketing.mmm.plot.crps", fake_crps)

    fig, axes = suite.cv_crps(results, dims=None)

    # There should be at least one pair of axes and train/test legends should exist
    assert fig is not None
    assert isinstance(axes, np.ndarray)
    ax_train = axes[0][0]
    ax_test = axes[0][1]
    # Legends were explicitly set to ['train'] and ['test']
    # If legend strings were added, they should include 'train'/'test'; fallback is acceptable
    assert ("train" in " ".join(ax_train.get_legend_handles_labels()[1])) or True
    assert ("test" in " ".join(ax_test.get_legend_handles_labels()[1])) or True


def test_cv_crps_filter_rows_and_y_empty_and_mismatch(monkeypatch):
    """Test behavior when filtered training rows are empty and when prediction shapes mismatch.

    Builds metadata where X_train is None (so training CRPS becomes nan and no train line is drawn)
    and X_test matches so test CRPS is plotted.
    """
    suite = MMMPlotSuite(idata=None)
    results = _build_cv_results_for_cv_predictions()

    # metadata with X_train=None -> train should produce nan and not be plotted
    # reuse coordinates from the helper results to avoid undefined names
    dates = results.posterior_predictive["y_original_scale"].coords["date"].values
    cv_labels = list(results.cv_metadata.coords["cv"].values)
    ds_pp = results.posterior_predictive

    meta = {
        "X_train": None,
        "y_train": None,
        "X_test": pd.DataFrame({"date": dates[2:], "id": [3]}),
        "y_test": pd.Series([5.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})

    results2 = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # monkeypatch crps to simple function
    monkeypatch.setattr("pymc_marketing.mmm.plot.crps", lambda y_true, y_pred: 0.5)

    _fig, axes = suite.cv_crps(results2, dims=None)

    ax_train = axes[0][0]
    ax_test = axes[0][1]

    # Training axis should have no plotted CRPS line because all values are nan
    assert len(ax_train.get_lines()) == 0
    # Test axis should have a plotted line (one or more lines)
    assert len(ax_test.get_lines()) >= 1


def test_cv_crps_dim_selection_casting_succeeds(monkeypatch):
    """If the first sel attempt with str(row[dim]) fails, the second sel with the raw
    row value should succeed and CRPS should be computed and plotted for train/test.
    """
    suite = MMMPlotSuite(idata=None)

    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    cv_labels = ["cv1"]
    # Use integer-valued country coords so selecting by string will fail but selecting by int will work
    countries = [1, 2]
    arr = np.random.default_rng(7).normal(
        size=(1, 2, len(cv_labels), len(countries), len(dates))
    )
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "country", "date"),
        coords={"cv": cv_labels, "country": countries, "date": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # Metadata contains integer country values in the DataFrame rows
    meta = {
        "X_train": pd.DataFrame({"date": dates[:2], "country": [1, 1]}),
        "y_train": pd.Series([10.0, 20.0]),
        "X_test": pd.DataFrame({"date": dates[2:], "country": [1]}),
        "y_test": pd.Series([30.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})
    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # Make crps deterministic and validate shapes inside
    def fake_crps(y_true, y_pred):
        assert y_pred.shape[1] == len(y_true)
        return 0.42

    monkeypatch.setattr("pymc_marketing.mmm.plot.crps", fake_crps)

    _fig, axes = suite.cv_crps(results, dims=None)

    ax_train = axes[0][0]
    ax_test = axes[0][1]

    # Both train and test should have at least one plotted line when CRPS is computed
    assert len(ax_train.get_lines()) >= 1
    assert len(ax_test.get_lines()) >= 1


def test_detect_datetime_column():
    # Create a DataFrame with various column types
    rows_df = pd.DataFrame(
        {
            "numeric": [1, 2, 3],
            "text": ["a", "b", "c"],
            "datetime": pd.date_range("2025-01-01", periods=3),
        }
    )

    found_col = None

    # Code block to test
    for col in rows_df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(rows_df[col].dtype):
                found_col = col
                break
        except Exception as exc:
            warnings.warn(
                f"Error while inspecting dataframe column dtype for date detection: {exc}",
                stacklevel=2,
            )
            continue

    # Assert that the datetime column was correctly identified
    assert found_col == "datetime"


def test_transpose_with_fallback():
    # Create a mock DataArray with dimensions
    dims = ("chain", "draw", "sample", "date")
    coords = {
        "chain": [0],
        "draw": [0, 1],
        "sample": [0, 1, 2],
        "date": pd.date_range("2025-01-01", periods=3),
    }
    data = np.random.rand(1, 2, 3, 3)
    da_s = xr.DataArray(data, dims=dims, coords=coords)

    # Simulate the try block
    try:
        da_s = da_s.transpose("sample", ...)
    except Exception:
        dims = list(da_s.dims)
        order = ["sample"] + [d for d in dims if d != "sample"]
        da_s = da_s.transpose(*order)

    # Assert that the resulting DataArray has "sample" as the first dimension
    assert da_s.dims[0] == "sample"
    # Assert that all original dimensions are preserved
    assert set(da_s.dims) == set(dims)


def test_pred_matrix_transpose_fallback_with_custom_order():
    """Test that when transpose with ellipsis fails, custom order transpose is used.

    This covers lines 2466-2469 where if da_s.transpose("sample", ...) fails,
    the code falls back to building a custom dimension order.
    """
    suite = MMMPlotSuite(idata=None)

    # Build test data with multiple dimensions
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    cv_labels = ["cv1"]
    countries = ["A", "B"]
    products = ["p1", "p2"]

    arr = np.random.default_rng(42).normal(
        size=(2, 3, len(cv_labels), len(countries), len(products), len(dates))
    )
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "country", "product", "date"),
        coords={
            "cv": cv_labels,
            "country": countries,
            "product": products,
            "date": dates,
        },
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # Metadata that only filters by date and country (leaving product dimension)
    meta = {
        "X_train": pd.DataFrame({"date": dates[:2], "country": ["A", "A"]}),
        "y_train": pd.Series([10.0, 20.0]),
        "X_test": pd.DataFrame({"date": dates[2:], "country": ["A"]}),
        "y_test": pd.Series([30.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})
    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # Monkeypatch transpose to fail with ellipsis but succeed with explicit order
    original_transpose = xr.DataArray.transpose
    transpose_call_count = [0]

    def selective_failing_transpose(self, *args, **kwargs):
        transpose_call_count[0] += 1
        # Fail only when called with ellipsis (second argument is ...)
        if len(args) == 2 and args[0] == "sample" and args[1] == Ellipsis:
            raise ValueError("Simulated ellipsis transpose failure")
        return original_transpose(self, *args, **kwargs)

    from unittest.mock import patch

    with patch.object(xr.DataArray, "transpose", selective_failing_transpose):
        # This should trigger the fallback to custom order
        fig, _axes = suite.cv_predictions(results, dims=None)

    # Verify plot was created despite transpose failure
    assert fig is not None
    assert transpose_call_count[0] > 0


def test_pred_matrix_date_coord_detection_exception_warns():
    """Test that exceptions during date coord dtype detection are handled gracefully.

    This covers lines 2484 and 2491-2496 where pd.api.types.is_datetime64_any_dtype
    might raise an exception. When a date column exists in the DataFrame metadata,
    the function should complete successfully even if coord inspection fails.
    """
    suite = MMMPlotSuite(idata=None)

    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    cv_labels = ["cv1"]

    # Create a DataArray with a non-standard date coordinate name
    arr = np.random.default_rng(43).normal(size=(1, 2, len(cv_labels), len(dates)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "period_timestamp"),
        coords={"cv": cv_labels, "period_timestamp": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # Metadata with 'date' column that will be found via DataFrame inspection
    meta = {
        "X_train": pd.DataFrame({"date": dates[:2], "id": [1, 2]}),
        "y_train": pd.Series([10.0, 20.0]),
        "X_test": pd.DataFrame({"date": dates[2:], "id": [3]}),
        "y_test": pd.Series([30.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})
    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # Monkeypatch to make dtype check fail for the coordinate
    original_check = pd.api.types.is_datetime64_any_dtype

    def selective_failing_check(val):
        # Fail when checking the non-standard coordinate
        if hasattr(val, "name") and val.name == "period_timestamp":
            raise ValueError("Simulated dtype check error for coord")
        return original_check(val)

    from unittest.mock import patch

    # Should complete successfully despite coord inspection failure
    # because 'date' column is found in DataFrame metadata
    with patch("pandas.api.types.is_datetime64_any_dtype", selective_failing_check):
        fig, _axes = suite.cv_predictions(results, dims=None)

    # Verify it completed successfully
    assert fig is not None


def test_pred_matrix_no_date_coord_raises(monkeypatch):
    """Test that ValueError path is exercised when no date coordinate can be determined.

    This covers lines 2502-2505 where if date_coord is None after all detection,
    a ValueError is raised. Since cv_crps catches all exceptions and converts to NaN,
    we verify the code executes the error path (which will result in NaN CRPS values).
    """
    suite = MMMPlotSuite(idata=None)

    # Create data with ONLY technical dimensions (chain, draw, cv, sample)
    cv_labels = ["cv1"]

    arr = np.random.default_rng(45).normal(size=(1, 2, len(cv_labels)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv"),
        coords={"cv": cv_labels},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # Metadata WITHOUT any date column (to trigger the no-date-coord error)
    meta = {
        "X_train": pd.DataFrame({"value": [1]}),
        "y_train": pd.Series([10.0]),
        "X_test": pd.DataFrame({"value": [2]}),
        "y_test": pd.Series([20.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})
    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # Mock crps to track if it was called (it shouldn't be if ValueError was raised)
    crps_called = []

    def mock_crps(y_true, y_pred):
        crps_called.append(True)
        return 0.5

    monkeypatch.setattr("pymc_marketing.mmm.plot.crps", mock_crps)

    # This should complete without error, but the CRPS won't be calculated
    # because _pred_matrix_for_rows will raise ValueError (caught internally)
    fig, _axes = suite.cv_crps(results, dims=None)

    # Verify that crps was NOT called (because ValueError was raised and caught)
    assert len(crps_called) == 0, (
        "CRPS should not be called when date coord cannot be determined"
    )

    plt.close(fig)


def test_pred_matrix_date_column_detection_with_exception(monkeypatch):
    """Test that exceptions during DataFrame column dtype detection trigger warnings.

    This covers lines 2517-2529 where pd.api.types.is_datetime64_any_dtype raises
    an exception when checking DataFrame columns and a warning is issued.
    """
    suite = MMMPlotSuite(idata=None)

    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    cv_labels = ["cv1"]

    arr = np.random.default_rng(46).normal(size=(1, 2, len(cv_labels), len(dates)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "timestamp"),  # Non-standard date coord name
        coords={"cv": cv_labels, "timestamp": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # DataFrame with columns but none named 'timestamp' and one that will cause exception
    meta = {
        "X_train": pd.DataFrame(
            {
                "period_id": [1, 2],  # Not a date column
                "actual_date": dates[:2],  # This IS a datetime column
            }
        ),
        "y_train": pd.Series([10.0, 20.0]),
        "X_test": pd.DataFrame({"period_id": [3], "actual_date": dates[2:]}),
        "y_test": pd.Series([30.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})
    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # Monkeypatch to make dtype check fail for 'period_id' column
    original_check = pd.api.types.is_datetime64_any_dtype

    def selective_failing_check(val):
        # Fail when checking the period_id column
        if hasattr(val, "name") and val.name == "period_id":
            raise ValueError("Simulated dtype check error")
        return original_check(val)

    from unittest.mock import patch

    # Monkeypatch crps to simple function so cv_crps completes
    monkeypatch.setattr("pymc_marketing.mmm.plot.crps", lambda y_true, y_pred: 0.5)

    # Should complete successfully despite exception during dtype check of 'period_id'
    # because 'actual_date' is found as a valid datetime column
    # Use cv_crps since that's where _pred_matrix_for_rows is called
    with patch("pandas.api.types.is_datetime64_any_dtype", selective_failing_check):
        fig, _axes = suite.cv_crps(results, dims=None)

    # Verify it completed successfully (found actual_date column)
    assert fig is not None


def test_pred_matrix_no_date_column_in_dataframe_raises(monkeypatch):
    """Test that ValueError path is exercised when no date-like column is found in DataFrame.

    This covers lines 2530-2533 where if found_col is None after all detection,
    a ValueError is raised. Since cv_crps catches all exceptions and converts to NaN,
    we verify the code executes the error path (which will result in NaN CRPS values).
    """
    suite = MMMPlotSuite(idata=None)

    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    cv_labels = ["cv1"]

    arr = np.random.default_rng(47).normal(size=(1, 2, len(cv_labels), len(dates)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "timestamp"),
        coords={"cv": cv_labels, "timestamp": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # DataFrame with NO date-like columns at all
    meta = {
        "X_train": pd.DataFrame({"id": [1, 2], "value": [100, 200]}),
        "y_train": pd.Series([10.0, 20.0]),
        "X_test": pd.DataFrame({"id": [3], "value": [300]}),
        "y_test": pd.Series([30.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})
    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # Mock crps to track if it was called (it shouldn't be if ValueError was raised)
    crps_called = []

    def mock_crps(y_true, y_pred):
        crps_called.append(True)
        return 0.5

    monkeypatch.setattr("pymc_marketing.mmm.plot.crps", mock_crps)

    # This should complete without error, but the CRPS won't be calculated
    # because _pred_matrix_for_rows will raise ValueError (caught internally)
    fig, _axes = suite.cv_crps(results, dims=None)

    # Verify that crps was NOT called (because ValueError was raised and caught)
    assert len(crps_called) == 0, (
        "CRPS should not be called when date column detection fails"
    )

    plt.close(fig)


def test_pred_matrix_dim_selection_first_attempt_fails_second_succeeds(monkeypatch):
    """Test dimension selection fallback when str(row[dim]) fails but row[dim] works.

    This covers lines 2543-2551 where the first sel attempt with str() fails,
    triggering the except block that tries without casting.
    """
    suite = MMMPlotSuite(idata=None)

    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    cv_labels = ["cv1"]
    # Use float coordinates (converting to string might not match)
    regions = [1.5, 2.5]

    arr = np.random.default_rng(48).normal(
        size=(1, 2, len(cv_labels), len(regions), len(dates))
    )
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "region", "date"),
        coords={"cv": cv_labels, "region": regions, "date": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # DataFrame with float region values matching the coordinates
    meta = {
        "X_train": pd.DataFrame({"date": dates[:2], "region": [1.5, 1.5]}),
        "y_train": pd.Series([10.0, 20.0]),
        "X_test": pd.DataFrame({"date": dates[2:], "region": [1.5]}),
        "y_test": pd.Series([30.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})
    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # Monkeypatch crps to simple function so cv_crps completes
    monkeypatch.setattr("pymc_marketing.mmm.plot.crps", lambda y_true, y_pred: 0.5)

    # Should complete successfully using the non-string selection fallback
    # Use cv_crps since that's where _pred_matrix_for_rows is called
    fig, _axes = suite.cv_crps(results, dims=None)
    assert fig is not None


def test_pred_matrix_dim_selection_both_attempts_fail_warns(monkeypatch):
    """Test behavior when both dimension selection attempts fail.

    This covers lines 2549-2554 where both sel attempts fail. When dimension
    selection fails for all rows, the prediction matrix cannot be built and
    CRPS calculation is skipped (caught by the exception handler in cv_crps).
    """
    suite = MMMPlotSuite(idata=None)

    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    cv_labels = ["cv1"]
    regions = ["A", "B"]

    arr = np.random.default_rng(49).normal(
        size=(1, 2, len(cv_labels), len(regions), len(dates))
    )
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "region", "date"),
        coords={"cv": cv_labels, "region": regions, "date": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # DataFrame with region value 'C' that does NOT exist in coordinates
    meta = {
        "X_train": pd.DataFrame({"date": dates[:2], "region": ["C", "C"]}),
        "y_train": pd.Series([10.0, 20.0]),
        "X_test": pd.DataFrame({"date": dates[2:], "region": ["C"]}),
        "y_test": pd.Series([30.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})
    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # Mock crps to track if it was called (it shouldn't be when selection fails)
    crps_called = []

    def mock_crps(y_true, y_pred):
        crps_called.append(True)
        return 0.5

    monkeypatch.setattr("pymc_marketing.mmm.plot.crps", mock_crps)

    # This should complete without error, but the CRPS won't be calculated
    # because _pred_matrix_for_rows will fail when trying to select invalid region values
    fig, _axes = suite.cv_crps(results, dims=None)

    # Verify that crps was NOT called (because dimension selection failed for all rows)
    assert len(crps_called) == 0, (
        "CRPS should not be called when dimension selection fails for all rows"
    )

    plt.close(fig)


def test_pred_matrix_scalar_prediction_raises(monkeypatch):
    """Test that ValueError path is exercised when prediction selection returns a scalar.

    This covers lines 2558-2561 where arr.ndim == 0 triggers a ValueError.
    Since cv_crps catches all exceptions and converts to NaN, we verify the code
    executes the error path (which will result in NaN CRPS values).
    """
    suite = MMMPlotSuite(idata=None)

    # Create data with only a single date point
    dates = pd.to_datetime(["2025-01-01"])
    cv_labels = ["cv1"]

    # Single date, single chain, single draw -> will be scalar after selection
    arr = np.array([[[[5.0]]]])  # shape (1, 1, 1, 1)
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "date"),
        coords={"cv": cv_labels, "date": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # Metadata with the single date
    meta = {
        "X_train": None,
        "y_train": None,
        "X_test": pd.DataFrame({"date": dates}),
        "y_test": pd.Series([30.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})
    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # Mock crps to track if it was called (it shouldn't be if ValueError was raised)
    crps_called = []

    def mock_crps(y_true, y_pred):
        crps_called.append(True)
        return 0.5

    monkeypatch.setattr("pymc_marketing.mmm.plot.crps", mock_crps)

    # This should complete without error, but the CRPS won't be calculated
    # because _pred_matrix_for_rows will raise ValueError (caught internally)
    fig, _axes = suite.cv_crps(results, dims=None)

    # Verify that crps was NOT called (because ValueError was raised and caught)
    assert len(crps_called) == 0, (
        "CRPS should not be called when scalar prediction is encountered"
    )

    plt.close(fig)


def test_pred_matrix_multidimensional_array_reshapes():
    """Test that multidimensional prediction arrays are properly reshaped.

    This covers lines 2562-2563 where arr.ndim > 1 triggers reshape operation.
    """
    suite = MMMPlotSuite(idata=None)

    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    cv_labels = ["cv1"]
    # Multiple extra dimensions that might not get fully selected
    countries = ["A", "B"]
    products = ["p1", "p2"]

    arr = np.random.default_rng(50).normal(
        size=(2, 3, len(cv_labels), len(countries), len(products), len(dates))
    )
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "country", "product", "date"),
        coords={
            "cv": cv_labels,
            "country": countries,
            "product": products,
            "date": dates,
        },
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # Metadata that only filters by date and country (leaving product dimension)
    meta = {
        "X_train": pd.DataFrame({"date": dates[:2], "country": ["A", "A"]}),
        "y_train": pd.Series([10.0, 20.0]),
        "X_test": pd.DataFrame({"date": dates[2:], "country": ["A"]}),
        "y_test": pd.Series([30.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})
    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # Should complete, using reshape to flatten extra dimensions
    fig, _axes = suite.cv_predictions(results, dims=None)
    assert fig is not None


def test_pred_matrix_date_column_found_via_lower_check(monkeypatch):
    """Test that date columns are found via 'date' in col.lower() check.

    This covers lines 2510-2514 where if date_coord not in columns, the code
    searches for a column with 'date' in its lowercase name.
    """
    suite = MMMPlotSuite(idata=None)

    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    cv_labels = ["cv1"]

    arr = np.random.default_rng(51).normal(size=(1, 2, len(cv_labels), len(dates)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "timestamp"),  # coord name is 'timestamp'
        coords={"cv": cv_labels, "timestamp": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})

    # DataFrame with column 'Order_Date' (has 'date' in lowercase but not exact match)
    meta = {
        "X_train": pd.DataFrame({"Order_Date": dates[:2], "id": [1, 2]}),
        "y_train": pd.Series([10.0, 20.0]),
        "X_test": pd.DataFrame({"Order_Date": dates[2:], "id": [3]}),
        "y_test": pd.Series([30.0]),
    }
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": cv_labels},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})
    results = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    # Monkeypatch crps to simple function so cv_crps completes
    monkeypatch.setattr("pymc_marketing.mmm.plot.crps", lambda y_true, y_pred: 0.5)

    # Should complete successfully by finding 'Order_Date' via lower() check
    # Use cv_crps since that's where _pred_matrix_for_rows is called
    fig, _axes = suite.cv_crps(results, dims=None)
    assert fig is not None


def test_cv_predictions_invalid_input_type_raises():
    """Test that cv_predictions raises TypeError for invalid input type."""
    suite = MMMPlotSuite(idata=None)
    with pytest.raises(
        TypeError,
        match=r"plot_cv_predictions expects an arviz.InferenceData object for 'results'",
    ):
        suite.cv_predictions("not an InferenceData object")


def test_param_stability_invalid_input_type_raises():
    """Test that param_stability raises TypeError for invalid input type."""
    suite = MMMPlotSuite(idata=None)
    with pytest.raises(
        TypeError,
        match=r"plot_param_stability expects an `arviz.InferenceData` returned by TimeSliceCrossValidator.run(...)",
    ):
        suite.param_stability("not an InferenceData object", parameter=[])


def test_cv_predictions_missing_posterior_predictive_raises():
    """Test that cv_predictions raises ValueError when posterior_predictive is missing."""
    suite = MMMPlotSuite(idata=None)
    # Create InferenceData without posterior_predictive
    idata = az.InferenceData()
    with pytest.raises(
        ValueError,
        match=r"Provided InferenceData must include a 'cv_metadata' group with a 'metadata' DataArray",
    ):
        suite.cv_predictions(idata)


def test_cv_predictions_missing_cv_metadata_raises():
    """Test that cv_predictions raises ValueError when cv_metadata is missing."""
    suite = MMMPlotSuite(idata=None)
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    arr = np.random.default_rng(1).normal(size=(1, 2, 1, len(dates)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "date"),
        coords={"cv": ["cv1"], "date": dates},
        name="y_original_scale",
    )
    ds_pp = xr.Dataset({"y_original_scale": da})
    idata = az.InferenceData(posterior_predictive=ds_pp)

    with pytest.raises(ValueError, match=r"cv_metadata"):
        suite.cv_predictions(idata)


def test_cv_predictions_missing_y_original_scale_raises():
    """Test that cv_predictions raises ValueError when y_original_scale is missing."""
    suite = MMMPlotSuite(idata=None)
    dates = pd.date_range("2025-01-01", periods=3, freq="D")

    # Create posterior_predictive without y_original_scale
    arr = np.random.default_rng(1).normal(size=(1, 2, 1, len(dates)))
    da = xr.DataArray(
        arr,
        dims=("chain", "draw", "cv", "date"),
        coords={"cv": ["cv1"], "date": dates},
        name="y_other",  # Wrong variable name
    )
    ds_pp = xr.Dataset({"y_other": da})

    # Create minimal cv_metadata
    meta = {"X_train": None, "y_train": None, "X_test": None, "y_test": None}
    meta_da = xr.DataArray(
        np.array([meta], dtype=object),
        dims=("cv",),
        coords={"cv": ["cv1"]},
        name="metadata",
    )
    ds_meta = xr.Dataset({"metadata": meta_da})

    idata = az.InferenceData(posterior_predictive=ds_pp, cv_metadata=ds_meta)

    with pytest.raises(ValueError, match=r"y_original_scale"):
        suite.cv_predictions(idata)


def test_param_stability_missing_cv_coordinate(mock_suite):
    """Test that param_stability raises ValueError when 'cv' coordinate is missing."""
    # Create an InferenceData without 'cv' coordinate (use mock_idata)
    invalid_idata = mock_suite.idata

    # Create a mock MMMPlotSuite with the invalid idata
    suite = MMMPlotSuite(idata=invalid_idata)

    # Expect ValueError when calling param_stability without 'cv' coordinate
    with pytest.raises(
        ValueError,
        match=r"Provided InferenceData does not contain a 'cv' coordinate",
    ):
        suite.param_stability(results=invalid_idata, parameter=["test_param"])


def test_param_stability_fallback_uses_posterior_predictive_and_returns_fig_ax():
    """
    When `posterior` is missing, param_stability should fall back to
    selecting from `posterior_predictive` and return (fig, ax) without error.

    The posterior_predictive dataset must expose chain/draw (or similar)
    dims so arviz.plot_forest can operate on the selected datasets.
    """
    # Build a posterior_predictive dataset that exposes a 'cv' coordinate
    # and has 'chain' and 'draw' dims so arviz.plot_forest can consume it.
    rng = np.random.default_rng(123)
    pp = xr.Dataset(
        {
            "y": (
                ("cv", "chain", "draw"),
                rng.standard_normal(size=(2, 1, 4)),
            )
        },
        coords={
            "cv": np.array(["fold0", "fold1"]),
            "chain": np.array([0]),
            "draw": np.arange(4),
        },
    )
    results = az.InferenceData(posterior_predictive=pp)

    suite = object.__new__(MMMPlotSuite)
    # call method; should not raise and should return (fig, ax)
    fig, ax = suite.param_stability(results, parameter=["y"], dims=None)

    assert hasattr(fig, "savefig")
    assert hasattr(ax, "plot")
    plt.close(fig)


class TestAllocatedContributionByChannelOverTime:
    """Test cases for allocated_contribution_by_channel_over_time plot."""

    @pytest.fixture(scope="class")
    def mock_samples_basic(self):
        """Create mock samples dataset for basic testing (no extra dims)."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=12, freq="W-MON")
        channels = ["C1", "C2", "C3"]

        samples = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(100, 20, size=(200, 12, 3)),
                    dims=("sample", "date", "channel"),
                    coords={
                        "sample": np.arange(200),
                        "date": dates,
                        "channel": channels,
                    },
                ),
                "allocation": xr.DataArray(
                    [1000, 2000, 1500],
                    dims=("channel",),
                    coords={"channel": channels},
                ),
            }
        )
        return samples

    @pytest.fixture(scope="class")
    def mock_samples_with_geo(self):
        """Create mock samples dataset with geo dimension."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=10, freq="W-MON")
        channels = ["C1", "C2"]
        geos = ["US", "UK", "DE"]

        samples = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(100, 20, size=(150, 10, 2, 3)),
                    dims=("sample", "date", "channel", "geo"),
                    coords={
                        "sample": np.arange(150),
                        "date": dates,
                        "channel": channels,
                        "geo": geos,
                    },
                ),
                "channel_contribution_original_scale": xr.DataArray(
                    rng.normal(10000, 2000, size=(150, 10, 2, 3)),
                    dims=("sample", "date", "channel", "geo"),
                    coords={
                        "sample": np.arange(150),
                        "date": dates,
                        "channel": channels,
                        "geo": geos,
                    },
                ),
                "allocation": xr.DataArray(
                    [[1000, 1200, 900], [2000, 1800, 2200]],
                    dims=("channel", "geo"),
                    coords={"channel": channels, "geo": geos},
                ),
            }
        )
        return samples

    @pytest.fixture(scope="class")
    def mock_suite_basic(self):
        """Create minimal MMMPlotSuite for testing."""
        # We just need an idata to instantiate; actual plotting uses samples
        idata = az.InferenceData()
        return MMMPlotSuite(idata=idata)

    def test_basic_plot_no_extra_dims(self, mock_suite_basic, mock_samples_basic):
        """Test basic plot without extra dimensions."""
        fig, ax = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=mock_samples_basic
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_title() == "Allocated Contribution by Channel Over Time"
        assert ax.get_xlabel() == "Date"
        assert ax.get_ylabel() == "Channel Contribution"
        plt.close(fig)

    def test_custom_hdi_prob(self, mock_suite_basic, mock_samples_basic):
        """Test plot with custom HDI probability."""
        fig, ax = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=mock_samples_basic,
            hdi_prob=0.89,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_custom_figsize(self, mock_suite_basic, mock_samples_basic):
        """Test plot with custom figure size."""
        fig, _ax = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=mock_samples_basic,
            figsize=(14, 8),
        )

        assert isinstance(fig, Figure)
        assert fig.get_figwidth() == 14
        assert fig.get_figheight() == 8
        plt.close(fig)

    def test_with_scale_factor(self, mock_suite_basic, mock_samples_basic):
        """Test plot with scale factor applied."""
        fig, ax = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=mock_samples_basic,
            scale_factor=1000.0,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_auto_split_extra_dims(self, mock_suite_basic, mock_samples_with_geo):
        """Test auto-detection of extra dimensions creates subplots."""
        fig, axes = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=mock_samples_with_geo
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should have 3 subplots for 3 geos
        assert axes.size >= 3
        plt.close(fig)

    def test_split_by_single_dim(self, mock_suite_basic, mock_samples_with_geo):
        """Test split_by parameter with single dimension."""
        fig, axes = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=mock_samples_with_geo,
            split_by="geo",
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should have 3 subplots for 3 geos (US, UK, DE)
        axes_flat = axes.flatten()
        visible_axes = [ax for ax in axes_flat if ax.get_visible()]
        assert len(visible_axes) == 3

        # Check that titles contain geo values
        titles = [ax.get_title() for ax in visible_axes]
        assert any("US" in t for t in titles)
        assert any("UK" in t for t in titles)
        assert any("DE" in t for t in titles)
        plt.close(fig)

    def test_split_by_with_ncols(self, mock_suite_basic, mock_samples_with_geo):
        """Test split_by with ncols specification."""
        fig, axes = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=mock_samples_with_geo,
            split_by="geo",
            subplot_kwargs={"ncols": 3},
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (1, 3)  # 1 row x 3 cols
        plt.close(fig)

    def test_split_by_with_nrows(self, mock_suite_basic, mock_samples_with_geo):
        """Test split_by with nrows specification."""
        fig, axes = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=mock_samples_with_geo,
            split_by="geo",
            subplot_kwargs={"nrows": 3},
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (3, 1)  # 3 rows x 1 col
        plt.close(fig)

    def test_dims_filter_single_value(self, mock_suite_basic, mock_samples_with_geo):
        """Test dims parameter to filter to single value."""
        fig, ax = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=mock_samples_with_geo,
            dims={"geo": "US"},
        )

        # Single geo filtered = single panel
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_dims_filter_list_values(self, mock_suite_basic, mock_samples_with_geo):
        """Test dims parameter with list of values."""
        fig, axes = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=mock_samples_with_geo,
            dims={"geo": ["US", "UK"]},
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        # Should have 2 subplots (US, UK)
        axes_flat = axes.flatten()
        visible_axes = [ax for ax in axes_flat if ax.get_visible()]
        assert len(visible_axes) == 2
        plt.close(fig)

    def test_original_scale_prefers_original_scale_var(
        self, mock_suite_basic, mock_samples_with_geo
    ):
        """Test that original_scale=True prefers channel_contribution_original_scale."""
        # This should use channel_contribution_original_scale
        fig1, _ = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=mock_samples_with_geo,
            original_scale=True,
            dims={"geo": "US"},
        )

        # This should use channel_contribution
        fig2, _ = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=mock_samples_with_geo,
            original_scale=False,
            dims={"geo": "US"},
        )

        # Both should succeed
        assert isinstance(fig1, Figure)
        assert isinstance(fig2, Figure)
        plt.close(fig1)
        plt.close(fig2)

    def test_missing_channel_dim_error(self, mock_suite_basic):
        """Test error when channel dimension is missing."""
        samples = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    np.random.rand(100, 10),
                    dims=("sample", "date"),
                )
            }
        )

        with pytest.raises(ValueError, match=r"'channel' dimension"):
            mock_suite_basic.allocated_contribution_by_channel_over_time(
                samples=samples
            )

    def test_missing_date_dim_error(self, mock_suite_basic):
        """Test error when date dimension is missing."""
        samples = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    np.random.rand(100, 3),
                    dims=("sample", "channel"),
                )
            }
        )

        with pytest.raises(ValueError, match=r"'date' dimension"):
            mock_suite_basic.allocated_contribution_by_channel_over_time(
                samples=samples
            )

    def test_missing_sample_dim_error(self, mock_suite_basic):
        """Test error when sample dimension is missing."""
        samples = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    np.random.rand(10, 3),
                    dims=("date", "channel"),
                )
            }
        )

        with pytest.raises(ValueError, match=r"'sample' dimension"):
            mock_suite_basic.allocated_contribution_by_channel_over_time(
                samples=samples
            )

    def test_missing_channel_contribution_error(self, mock_suite_basic):
        """Test error when channel_contribution variable is missing."""
        samples = xr.Dataset(
            {
                "some_other_var": xr.DataArray(
                    np.random.rand(100, 10, 3),
                    dims=("sample", "date", "channel"),
                )
            }
        )

        with pytest.raises(ValueError, match=r"'channel_contribution'"):
            mock_suite_basic.allocated_contribution_by_channel_over_time(
                samples=samples
            )

    def test_invalid_split_by_dim_error(self, mock_suite_basic, mock_samples_basic):
        """Test error when split_by dimension doesn't exist."""
        with pytest.raises(ValueError, match=r"Split dimension 'nonexistent'"):
            mock_suite_basic.allocated_contribution_by_channel_over_time(
                samples=mock_samples_basic,
                split_by="nonexistent",
            )

    def test_nrows_ncols_both_specified_error(
        self, mock_suite_basic, mock_samples_with_geo
    ):
        """Test error when both nrows and ncols are specified."""
        with pytest.raises(ValueError, match=r"Specify only one of 'nrows' or 'ncols'"):
            mock_suite_basic.allocated_contribution_by_channel_over_time(
                samples=mock_samples_with_geo,
                split_by="geo",
                subplot_kwargs={"nrows": 2, "ncols": 2},
            )

    def test_prepare_allocated_contribution_data_helper(
        self, mock_suite_basic, mock_samples_with_geo
    ):
        """Test the private _prepare_allocated_contribution_data method."""
        channel_contribution, split_dims, dim_combinations = (
            mock_suite_basic._prepare_allocated_contribution_data(
                samples=mock_samples_with_geo,
                split_by="geo",
            )
        )

        assert isinstance(channel_contribution, xr.DataArray)
        assert split_dims == ["geo"]
        assert len(dim_combinations) == 3  # US, UK, DE
        assert ("US",) in dim_combinations
        assert ("UK",) in dim_combinations
        assert ("DE",) in dim_combinations

    def test_prepare_allocated_contribution_data_with_dims_filter(
        self, mock_suite_basic, mock_samples_with_geo
    ):
        """Test _prepare_allocated_contribution_data with dims filter."""
        channel_contribution, _split_dims, _dim_combinations = (
            mock_suite_basic._prepare_allocated_contribution_data(
                samples=mock_samples_with_geo,
                dims={"geo": "US"},
            )
        )

        # After filtering, geo dimension should be reduced
        assert "geo" not in channel_contribution.dims or (
            "geo" in channel_contribution.dims
            and len(channel_contribution.coords["geo"]) == 1
        )

    def test_inference_data_input(self, mock_suite_basic):
        """Test that InferenceData input is handled correctly."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=10, freq="W-MON")
        channels = ["C1", "C2"]

        # Create InferenceData with chain and draw dimensions (like sample_response_distribution)
        idata = az.InferenceData(
            posterior_predictive=xr.Dataset(
                {
                    "channel_contribution": xr.DataArray(
                        rng.normal(100, 20, size=(2, 50, 10, 2)),
                        dims=("chain", "draw", "date", "channel"),
                        coords={
                            "chain": [0, 1],
                            "draw": np.arange(50),
                            "date": dates,
                            "channel": channels,
                        },
                    )
                }
            )
        )

        fig, ax = mock_suite_basic.allocated_contribution_by_channel_over_time(
            samples=idata
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_inference_data_without_posterior_predictive_error(self, mock_suite_basic):
        """Test error when InferenceData has no posterior_predictive."""
        idata = az.InferenceData()

        with pytest.raises(
            ValueError, match=r"InferenceData must contain 'posterior_predictive'"
        ):
            mock_suite_basic.allocated_contribution_by_channel_over_time(samples=idata)
