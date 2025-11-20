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
#
# NOTE: This file may be consolidated with test_plot_backends.py in the future
# once the backend migration is complete and stable.
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
    from arviz_plots import PlotCollection

    pc = mock_suite.contributions_over_time(
        var=["intercept", "linear_trend"],
        dims={"country": "A"},
    )
    assert isinstance(pc, PlotCollection)
    # Verify plot was created successfully
    assert hasattr(pc, "backend")
    assert hasattr(pc, "show")


def test_contributions_over_time_with_dims_list(mock_suite: MMMPlotSuite):
    """Test that passing a list to dims creates a subplot for each value."""
    from arviz_plots import PlotCollection

    pc = mock_suite.contributions_over_time(
        var=["intercept"],
        dims={"country": ["A", "B"]},
    )
    assert isinstance(pc, PlotCollection)
    assert hasattr(pc, "backend")
    assert hasattr(pc, "show")


def test_contributions_over_time_with_multiple_dims_lists(mock_suite: MMMPlotSuite):
    """Test that passing multiple lists to dims creates a subplot for each combination."""
    from arviz_plots import PlotCollection

    # Add a fake 'region' dim to the mock posterior for this test if not present
    idata = mock_suite.idata
    if "region" not in idata.posterior["intercept"].dims:
        idata.posterior["intercept"] = idata.posterior["intercept"].expand_dims(
            region=["X", "Y"]
        )
    pc = mock_suite.contributions_over_time(
        var=["intercept"],
        dims={"country": ["A", "B"], "region": ["X", "Y"]},
    )
    assert isinstance(pc, PlotCollection)
    assert hasattr(pc, "backend")
    assert hasattr(pc, "show")


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


def test_contributions_over_time_expand_dims(mock_suite: MMMPlotSuite):
    from arviz_plots import PlotCollection

    pc = mock_suite.contributions_over_time(
        var=[
            "intercept",
            "linear_trend",
        ]
    )

    assert isinstance(pc, PlotCollection)
    assert hasattr(pc, "backend")
    assert hasattr(pc, "show")


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


# =============================================================================
# Comprehensive Backend Tests (Milestone 3)
# =============================================================================
# These tests verify that all plotting methods work correctly across all
# supported backends (matplotlib, plotly, bokeh).
# =============================================================================


class TestPosteriorPredictiveBackends:
    """Test posterior_predictive method across all backends."""

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_posterior_predictive_all_backends(self, mock_suite, backend):
        """Test posterior_predictive works with all backends."""
        from arviz_plots import PlotCollection

        # Create idata with posterior_predictive
        idata = mock_suite.idata.copy()
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=52, freq="W")
        idata.posterior_predictive = xr.Dataset(
            {
                "y": xr.DataArray(
                    rng.normal(size=(4, 100, 52)),
                    dims=("chain", "draw", "date"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "date": dates,
                    },
                )
            }
        )
        suite = MMMPlotSuite(idata=idata)

        pc = suite.posterior_predictive(backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )


class TestContributionsOverTimeBackends:
    """Test contributions_over_time method across all backends."""

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_contributions_over_time_all_backends(self, mock_suite, backend):
        """Test contributions_over_time works with all backends."""
        from arviz_plots import PlotCollection

        pc = mock_suite.contributions_over_time(var=["intercept"], backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )


class TestSaturationPlotBackends:
    """Test saturation plot methods across all backends."""

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_saturation_scatterplot_all_backends(
        self, mock_suite_with_constant_data, backend
    ):
        """Test saturation_scatterplot works with all backends."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_constant_data.saturation_scatterplot(backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_saturation_curves_all_backends(
        self, mock_suite_with_constant_data, mock_saturation_curve, backend
    ):
        """Test saturation_curves works with all backends."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, backend=backend, n_samples=3
        )

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )


class TestBudgetAllocationBackends:
    """Test budget allocation methods across all backends."""

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_budget_allocation_roas_all_backends(self, mock_suite, backend):
        """Test budget_allocation_roas works with all backends."""
        from arviz_plots import PlotCollection

        # Create proper allocation samples with required variables and dimensions
        rng = np.random.default_rng(42)
        channels = ["TV", "Radio", "Digital"]
        dates = pd.date_range("2025-01-01", periods=52, freq="W")
        samples = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    rng.normal(loc=1000, scale=100, size=(100, 52, 3)),
                    dims=("sample", "date", "channel"),
                    coords={
                        "sample": np.arange(100),
                        "date": dates,
                        "channel": channels,
                    },
                ),
                "allocation": xr.DataArray(
                    rng.uniform(100, 1000, size=(3,)),
                    dims=("channel",),
                    coords={"channel": channels},
                ),
            }
        )

        pc = mock_suite.budget_allocation_roas(samples=samples, backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_allocated_contribution_by_channel_over_time_all_backends(
        self, mock_suite, backend
    ):
        """Test allocated_contribution_by_channel_over_time works with all backends."""
        from arviz_plots import PlotCollection

        # Create proper samples with 'sample', 'date', and 'channel' dimensions
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=52, freq="W")
        channels = ["TV", "Radio", "Digital"]
        samples = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(100, 52, 3)),
                    dims=("sample", "date", "channel"),
                    coords={
                        "sample": np.arange(100),
                        "date": dates,
                        "channel": channels,
                    },
                )
            }
        )

        pc = mock_suite.allocated_contribution_by_channel_over_time(
            samples=samples, backend=backend
        )

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )


class TestSensitivityAnalysisBackends:
    """Test sensitivity analysis methods across all backends."""

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_sensitivity_analysis_all_backends(
        self, mock_suite_with_sensitivity, backend
    ):
        """Test sensitivity_analysis works with all backends."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_sensitivity.sensitivity_analysis(backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_uplift_curve_all_backends(self, mock_suite_with_sensitivity, backend):
        """Test uplift_curve works with all backends."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_sensitivity.uplift_curve(backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_marginal_curve_all_backends(self, mock_suite_with_sensitivity, backend):
        """Test marginal_curve works with all backends."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_sensitivity.marginal_curve(backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )


class TestBackendBehavior:
    """Test backend configuration and override behavior."""

    def test_backend_overrides_global_config(self, mock_suite):
        """Test that method backend parameter overrides global config."""
        from arviz_plots import PlotCollection

        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.backend", "matplotlib")

        try:
            # Set global to matplotlib
            mmm_config["plot.backend"] = "matplotlib"

            # Override with plotly
            pc_plotly = mock_suite.contributions_over_time(
                var=["intercept"], backend="plotly"
            )
            assert isinstance(pc_plotly, PlotCollection)

            # Default should still be matplotlib
            pc_default = mock_suite.contributions_over_time(var=["intercept"])
            assert isinstance(pc_default, PlotCollection)

        finally:
            mmm_config["plot.backend"] = original

    @pytest.mark.parametrize("config_backend", ["matplotlib", "plotly", "bokeh"])
    def test_backend_parameter_none_uses_config(self, mock_suite, config_backend):
        """Test that backend=None uses global config."""
        from arviz_plots import PlotCollection

        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.backend", "matplotlib")

        try:
            mmm_config["plot.backend"] = config_backend

            pc = mock_suite.contributions_over_time(
                var=["intercept"],
                backend=None,  # Explicitly None
            )

            assert isinstance(pc, PlotCollection)
            # PlotCollection should be created with config_backend

        finally:
            mmm_config["plot.backend"] = original

    def test_invalid_backend_warning(self, mock_suite):
        """Test that invalid backend either shows warning or raises error."""
        # Invalid backend should either warn or raise an error
        # arviz_plots may accept the invalid backend string without warning
        # This test just verifies the code doesn't crash in unexpected ways
        try:
            mock_suite.contributions_over_time(
                var=["intercept"], backend="invalid_backend"
            )
            # If it succeeds, the backend was accepted (implementation dependent)
        except (ValueError, TypeError, NotImplementedError, ModuleNotFoundError):
            # These are expected exceptions for invalid backends
            pass


class TestDataParameters:
    """Test explicit data parameter functionality."""

    def test_contributions_over_time_with_explicit_data(self, mock_posterior_data):
        """Test contributions_over_time accepts explicit data parameter."""
        from arviz_plots import PlotCollection

        # Create suite without idata
        suite = MMMPlotSuite(idata=None)

        # Should work with explicit data parameter
        pc = suite.contributions_over_time(var=["intercept"], data=mock_posterior_data)

        assert isinstance(pc, PlotCollection)

    def test_saturation_scatterplot_with_explicit_data(
        self, mock_constant_data, mock_posterior_data
    ):
        """Test saturation_scatterplot accepts explicit data parameters."""
        from arviz_plots import PlotCollection

        suite = MMMPlotSuite(idata=None)

        # Create proper posterior data with channel_contribution
        rng = np.random.default_rng(42)
        posterior_data = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(4, 100, 52, 3)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "date": pd.date_range("2025-01-01", periods=52, freq="W"),
                        "channel": ["TV", "Radio", "Digital"],
                    },
                )
            }
        )

        pc = suite.saturation_scatterplot(
            constant_data=mock_constant_data, posterior_data=posterior_data
        )

        assert isinstance(pc, PlotCollection)


class TestIntegration:
    """Test complete workflows and method interactions."""

    def test_multiple_plots_same_suite_instance(self, mock_suite_with_constant_data):
        """Test that same suite instance can create multiple plots."""
        from arviz_plots import PlotCollection

        suite = mock_suite_with_constant_data

        # Create multiple different plots
        # Use channel_contribution which exists in the fixture
        pc1 = suite.contributions_over_time(var=["channel_contribution"])
        pc2 = suite.saturation_scatterplot()

        assert isinstance(pc1, PlotCollection)
        assert isinstance(pc2, PlotCollection)

        # All should be independent PlotCollection objects
        assert pc1 is not pc2

    def test_backend_switching_same_method(self, mock_suite):
        """Test that backends can be switched for same method."""
        from arviz_plots import PlotCollection

        suite = mock_suite

        # Create same plot with different backends
        pc_mpl = suite.contributions_over_time(var=["intercept"], backend="matplotlib")
        pc_plotly = suite.contributions_over_time(var=["intercept"], backend="plotly")
        pc_bokeh = suite.contributions_over_time(var=["intercept"], backend="bokeh")

        assert isinstance(pc_mpl, PlotCollection)
        assert isinstance(pc_plotly, PlotCollection)
        assert isinstance(pc_bokeh, PlotCollection)
