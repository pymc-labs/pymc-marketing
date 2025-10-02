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
import warnings

import arviz as az
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
    n_chain, n_draw, n_sweep = 2, 10, 5
    sweep = np.linspace(0.5, 1.5, n_sweep)
    # Add a single extra dim for multi-panel test
    extra_dim = ["A", "B"]
    # y and marginal_effects: dims (chain, draw, sweep, extra)
    y = xr.DataArray(
        np.random.normal(0, 1, size=(n_chain, n_draw, n_sweep, len(extra_dim))),
        dims=("chain", "draw", "sweep", "region"),
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "sweep": sweep,
            "region": extra_dim,
        },
    )
    marginal_effects = xr.DataArray(
        np.random.normal(0, 1, size=(n_chain, n_draw, n_sweep, len(extra_dim))),
        dims=("chain", "draw", "sweep", "region"),
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "sweep": sweep,
            "region": extra_dim,
        },
    )
    # Add sweep_type and var_names as attrs/coords
    sensitivity_analysis = xr.Dataset(
        {"y": y, "marginal_effects": marginal_effects},
        coords={"sweep": sweep, "region": extra_dim},
        attrs={"sweep_type": "multiplicative", "var_names": "test_var"},
    )
    # Attach to idata
    idata.sensitivity_analysis = sensitivity_analysis
    # Add posterior_predictive for percentage test
    idata.posterior_predictive = xr.Dataset(
        {
            "y": xr.DataArray(
                np.abs(
                    np.random.normal(
                        10, 2, size=(n_chain, n_draw, n_sweep, len(extra_dim))
                    )
                ),
                dims=("chain", "draw", "sweep", "region"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "sweep": sweep,
                    "region": extra_dim,
                },
            )
        }
    )
    return idata


@pytest.fixture(scope="module")
def mock_suite(mock_idata):
    """Fixture to create a mock MMMPlotSuite with a mocked posterior."""
    return MMMPlotSuite(idata=mock_idata)


@pytest.fixture(scope="module")
def mock_suite_with_sensitivity(mock_idata_with_sensitivity):
    """Fixture to create a mock MMMPlotSuite with sensitivity analysis."""
    return MMMPlotSuite(idata=mock_idata_with_sensitivity)


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


def test_plot_sensitivity_analysis_basic(mock_suite_with_sensitivity):
    # Should return (fig, axes) for multi-panel
    fig, axes = mock_suite_with_sensitivity.plot_sensitivity_analysis()
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert all(isinstance(ax, Axes) for ax in axes.flat)


def test_plot_sensitivity_analysis_marginal(mock_suite_with_sensitivity):
    fig, axes = mock_suite_with_sensitivity.plot_sensitivity_analysis(marginal=True)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)


def test_plot_sensitivity_analysis_percentage(mock_suite_with_sensitivity):
    fig, axes = mock_suite_with_sensitivity.plot_sensitivity_analysis(percentage=True)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)


def test_plot_sensitivity_analysis_error_on_both_modes(mock_suite_with_sensitivity):
    with pytest.raises(
        ValueError, match=r"Not implemented marginal effects in percentage scale."
    ):
        mock_suite_with_sensitivity.plot_sensitivity_analysis(
            marginal=True, percentage=True
        )


def test_plot_sensitivity_analysis_error_on_missing_results(mock_idata):
    suite = MMMPlotSuite(idata=mock_idata)
    with pytest.raises(ValueError, match=r"No sensitivity analysis results found"):
        suite.plot_sensitivity_analysis()


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
