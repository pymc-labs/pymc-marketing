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


class TestResidualsPosteriорDistribution:
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
