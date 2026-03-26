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
from __future__ import annotations

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.figure import Figure

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.plotting.diagnostics import (
    DiagnosticsPlots,
    _get_posterior_predictive,
    _get_prior_predictive,
)

matplotlib.use("Agg")

SEED = sum(map(ord, "DiagnosticsPlots tests"))


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def simple_idata() -> az.InferenceData:
    """InferenceData with (chain, draw, date) dims — no extra dims."""
    rng = np.random.default_rng(SEED)
    n_chain, n_draw, n_date = 2, 50, 20
    dates = np.arange(n_date)
    coords = {"chain": np.arange(n_chain), "draw": np.arange(n_draw), "date": dates}
    base_shape = (n_chain, n_draw, n_date)

    pp = xr.Dataset(
        {
            "y": xr.DataArray(
                rng.normal(size=base_shape),
                dims=("chain", "draw", "date"),
                coords=coords,
            ),
            "y_original_scale": xr.DataArray(
                rng.normal(size=base_shape) * 100 + 500,
                dims=("chain", "draw", "date"),
                coords=coords,
            ),
        }
    )
    prior = xr.Dataset(
        {
            "y": xr.DataArray(
                rng.normal(size=base_shape),
                dims=("chain", "draw", "date"),
                coords=coords,
            )
        }
    )
    const = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(500, 50, size=(n_date,)),
                dims=("date",),
                coords={"date": dates},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )
    return az.InferenceData(
        posterior_predictive=pp, prior_predictive=prior, constant_data=const
    )


@pytest.fixture(scope="module")
def panel_idata() -> az.InferenceData:
    """InferenceData with extra 'geo' dim — (chain, draw, date, geo)."""
    rng = np.random.default_rng(SEED + 1)
    n_chain, n_draw, n_date = 2, 30, 15
    dates = np.arange(n_date)
    geos = ["CA", "NY"]
    coords = {
        "chain": np.arange(n_chain),
        "draw": np.arange(n_draw),
        "date": dates,
        "geo": geos,
    }
    base_shape = (n_chain, n_draw, n_date, 2)

    pp = xr.Dataset(
        {
            "y": xr.DataArray(
                rng.normal(size=base_shape),
                dims=("chain", "draw", "date", "geo"),
                coords=coords,
            ),
            "y_original_scale": xr.DataArray(
                rng.normal(size=base_shape) * 100 + 500,
                dims=("chain", "draw", "date", "geo"),
                coords=coords,
            ),
        }
    )
    prior = xr.Dataset(
        {
            "y": xr.DataArray(
                rng.normal(size=base_shape),
                dims=("chain", "draw", "date", "geo"),
                coords=coords,
            )
        }
    )
    const = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(500, 50, size=(n_date, 2)),
                dims=("date", "geo"),
                coords={"date": dates, "geo": geos},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )
    return az.InferenceData(
        posterior_predictive=pp, prior_predictive=prior, constant_data=const
    )


@pytest.fixture(scope="module")
def simple_data(simple_idata) -> MMMIDataWrapper:
    return MMMIDataWrapper(simple_idata, validate_on_init=False)


@pytest.fixture(scope="module")
def panel_data(panel_idata) -> MMMIDataWrapper:
    return MMMIDataWrapper(panel_idata, validate_on_init=False)


@pytest.fixture(scope="module")
def simple_plots(simple_data) -> DiagnosticsPlots:
    return DiagnosticsPlots(simple_data)


@pytest.fixture(scope="module")
def panel_plots(panel_data) -> DiagnosticsPlots:
    return DiagnosticsPlots(panel_data)


# ============================================================================
# Helper tests
# ============================================================================


class TestGetPosteriorPredictive:
    def test_returns_dataset_with_y(self, simple_data):
        result = _get_posterior_predictive(simple_data)
        assert isinstance(result, xr.Dataset)
        assert "y" in result

    def test_raises_when_missing(self):
        data = MMMIDataWrapper(az.InferenceData(), validate_on_init=False)
        with pytest.raises(ValueError, match="posterior_predictive"):
            _get_posterior_predictive(data)


class TestGetPriorPredictive:
    def test_returns_dataset_with_y(self, simple_data):
        result = _get_prior_predictive(simple_data)
        assert isinstance(result, xr.Dataset)
        assert "y" in result

    def test_raises_when_missing(self):
        data = MMMIDataWrapper(az.InferenceData(), validate_on_init=False)
        with pytest.raises(ValueError, match="prior_predictive"):
            _get_prior_predictive(data)


class TestComputeResiduals:
    def test_returns_dataarray_named_residuals(self, simple_plots, simple_data):
        result = simple_plots._compute_residuals(simple_data)
        assert isinstance(result, xr.DataArray)
        assert result.name == "residuals"

    def test_has_chain_draw_date_dims(self, simple_plots, simple_data):
        result = simple_plots._compute_residuals(simple_data)
        assert {"chain", "draw", "date"}.issubset(result.dims)

    def test_custom_pp_var(self, simple_plots, simple_data):
        """pp_var parameter allows non-hardcoded variable name."""
        result = simple_plots._compute_residuals(simple_data, pp_var="y_original_scale")
        assert result.name == "residuals"

    def test_raises_on_missing_pp_var(self, simple_plots, simple_data):
        with pytest.raises(ValueError, match="y_nonexistent"):
            simple_plots._compute_residuals(simple_data, pp_var="y_nonexistent")


class TestDiagnosticsPlotsConstructor:
    def test_stores_data(self, simple_data):
        plots = DiagnosticsPlots(simple_data)
        assert plots._data is simple_data


# ============================================================================
# posterior_predictive tests
# ============================================================================


class TestPosteriorPredictiveBasic:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.posterior_predictive()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_single_panel_no_extra_dims(self, simple_plots):
        _, axes = simple_plots.posterior_predictive()
        assert axes.size == 1

    def test_return_as_pc(self, simple_plots):
        result = simple_plots.posterior_predictive(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_raises_when_y_original_scale_missing(self):
        """original_scale=True must raise clearly when y_original_scale is absent."""
        rng = np.random.default_rng(0)
        n_chain, n_draw, n_date = 2, 10, 5
        coords = {
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "date": np.arange(n_date),
        }
        pp = xr.Dataset(
            {
                "y": xr.DataArray(
                    rng.normal(size=(n_chain, n_draw, n_date)),
                    dims=("chain", "draw", "date"),
                    coords=coords,
                ),
            }
        )
        const = xr.Dataset(
            {
                "target_data": xr.DataArray(
                    rng.normal(size=(n_date,)),
                    dims=("date",),
                    coords={"date": np.arange(n_date)},
                ),
                "target_scale": xr.DataArray(1.0),
            }
        )
        idata = az.InferenceData(posterior_predictive=pp, constant_data=const)
        data = MMMIDataWrapper(idata, validate_on_init=False)
        plots = DiagnosticsPlots(data)
        with pytest.raises(ValueError, match="y_original_scale"):
            plots.posterior_predictive(original_scale=True)

    def test_raises_when_y_missing(self):
        """original_scale=False must raise clearly when y is absent."""
        rng = np.random.default_rng(0)
        n_chain, n_draw, n_date = 2, 10, 5
        coords = {
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "date": np.arange(n_date),
        }
        pp = xr.Dataset(
            {
                "y_original_scale": xr.DataArray(
                    rng.normal(size=(n_chain, n_draw, n_date)) * 100 + 500,
                    dims=("chain", "draw", "date"),
                    coords=coords,
                ),
            }
        )
        const = xr.Dataset(
            {
                "target_data": xr.DataArray(
                    rng.normal(size=(n_date,)),
                    dims=("date",),
                    coords={"date": np.arange(n_date)},
                ),
                "target_scale": xr.DataArray(1.0),
            }
        )
        idata = az.InferenceData(posterior_predictive=pp, constant_data=const)
        data = MMMIDataWrapper(idata, validate_on_init=False)
        plots = DiagnosticsPlots(data)
        with pytest.raises(ValueError, match="'y' not found"):
            plots.posterior_predictive(original_scale=False)

    def test_original_scale_false_plots_y(self, simple_plots, simple_data):
        """original_scale=False must plot the 'y' (scaled) variable."""
        _, axes = simple_plots.posterior_predictive(original_scale=False)
        ax = axes.flat[0]
        expected = (
            simple_data.idata.posterior_predictive["y"]
            .mean(dim=("chain", "draw"))
            .values
        )
        line_y_arrays = [line.get_ydata() for line in ax.lines]
        assert any(np.allclose(y, expected, equal_nan=True) for y in line_y_arrays), (
            "No line matches y (scaled) posterior mean when original_scale=False"
        )


class TestPosteriorPredictiveElements:
    def test_predicted_mean_line_present(self, simple_plots, simple_data):
        """The predicted mean line y-data must match pp_ds['y_original_scale'].mean(chain/draw)."""
        _, axes = simple_plots.posterior_predictive()
        ax = axes.flat[0]
        expected = (
            simple_data.idata.posterior_predictive["y_original_scale"]
            .mean(dim=("chain", "draw"))
            .values
        )
        line_y_arrays = [line.get_ydata() for line in ax.lines]
        assert any(np.allclose(y, expected, equal_nan=True) for y in line_y_arrays), (
            "No line matches the posterior predictive mean (y_original_scale)"
        )

    def test_hdi_band_present(self, simple_plots):
        """HDI fill_between band must create at least one collection in the axes."""
        _, axes = simple_plots.posterior_predictive()
        ax = axes.flat[0]
        assert len(ax.collections) > 0, (
            "No HDI band found in axes (expected fill_between collection)"
        )

    def test_x_data_length_matches_dates(self, simple_plots):
        """All plotted lines must span the full date range (n_date=20)."""
        _, axes = simple_plots.posterior_predictive()
        ax = axes.flat[0]
        n_date = 20  # matches simple_idata fixture
        for line in ax.lines:
            assert len(line.get_xdata()) == n_date

    def test_narrower_hdi_prob_gives_narrower_band(self, simple_plots):
        """A 50% HDI band must be narrower than the default 94% band."""
        _, axes_94 = simple_plots.posterior_predictive(hdi_prob=0.94)
        _, axes_50 = simple_plots.posterior_predictive(hdi_prob=0.50)
        ax_94 = axes_94.flat[0]
        ax_50 = axes_50.flat[0]

        def band_height(ax):
            verts = ax.collections[0].get_paths()[0].vertices
            return verts[:, 1].max() - verts[:, 1].min()

        assert band_height(ax_94) > band_height(ax_50), (
            "94% HDI band should be wider than 50% HDI band"
        )


class TestPosteriorPredictiveDims:
    def test_panel_idata_creates_multiple_panels(self, panel_plots):
        _, axes = panel_plots.posterior_predictive()
        assert axes.size >= 2  # one per geo value

    def test_dims_filter_single_value(self, panel_plots):
        _, axes = panel_plots.posterior_predictive(dims={"geo": ["CA"]})
        assert axes.size == 1

    def test_invalid_dim_raises(self, panel_plots):
        with pytest.raises(ValueError, match="nonexistent"):
            panel_plots.posterior_predictive(dims={"nonexistent": "CA"})


class TestPosteriorPredictiveCustomization:
    def test_figsize_sets_figure_dimensions(self, simple_plots):
        """figsize must propagate to the returned Figure."""
        fig, _ = simple_plots.posterior_predictive(figsize=(10, 4))
        w, h = fig.get_size_inches()
        assert abs(w - 10) < 0.1 and abs(h - 4) < 0.1

    def test_non_matplotlib_backend_without_return_as_pc_raises(self, simple_plots):
        with pytest.raises(ValueError, match="return_as_pc"):
            simple_plots.posterior_predictive(backend="plotly")

    def test_line_kwargs_color_applied(self, simple_plots):
        """line_kwargs color must appear on the predictive mean line."""
        _, axes = simple_plots.posterior_predictive(line_kwargs={"color": "blue"})
        ax = axes.flat[0]
        colors = [line.get_color() for line in ax.lines]
        assert any(c == "blue" for c in colors), (
            "No line with color='blue' found — line_kwargs not applied"
        )

    def test_hdi_kwargs_alpha_applied(self, simple_plots):
        """hdi_kwargs alpha must appear on the HDI collection."""
        _, axes = simple_plots.posterior_predictive(hdi_kwargs={"alpha": 0.05})
        ax = axes.flat[0]
        alphas = [col.get_alpha() for col in ax.collections]
        assert any(a is not None and abs(a - 0.05) < 1e-6 for a in alphas), (
            "No collection with alpha=0.05 found — hdi_kwargs not applied"
        )

    def test_observed_kwargs_color_applied(self, simple_plots):
        """observed_kwargs color must appear on the observed data line."""
        _, axes = simple_plots.posterior_predictive(observed_kwargs={"color": "red"})
        ax = axes.flat[0]
        colors = [line.get_color() for line in ax.lines]
        assert any(c == "red" for c in colors), (
            "No line with color='red' found — observed_kwargs not applied"
        )


class TestPosteriorPredictiveObserved:
    def test_observed_values_match_target(self, simple_plots, simple_data):
        """The observed line y-data must match data.get_target(original_scale=True)
        when original_scale=True (the default)."""
        _, axes = simple_plots.posterior_predictive()
        ax = axes.flat[0]
        expected = simple_data.get_target(original_scale=True).values
        line_y_arrays = [line.get_ydata() for line in ax.lines]
        assert any(np.allclose(y, expected, equal_nan=True) for y in line_y_arrays), (
            "No line in the axes matches the observed target values"
        )

    def test_observed_present_in_panel_plot(self, panel_plots, panel_data):
        """Multi-panel plot: observed data is present in each panel."""
        _, axes = panel_plots.posterior_predictive()
        observed = panel_data.get_target(original_scale=True)
        for ax in axes.flat:
            line_y_arrays = [line.get_ydata() for line in ax.lines]
            # At least one line in each panel has the same length as the date axis
            assert any(len(y) == observed.sizes["date"] for y in line_y_arrays)

    def test_observed_respects_original_scale_false(self, simple_plots, simple_data):
        """When plotting scaled predictions (original_scale=False), the observed line
        must use scaled target values."""
        _, axes = simple_plots.posterior_predictive(original_scale=False)
        ax = axes.flat[0]
        expected = simple_data.get_target(original_scale=False).values
        line_y_arrays = [line.get_ydata() for line in ax.lines]
        assert any(np.allclose(y, expected, equal_nan=True) for y in line_y_arrays), (
            "Observed line did not use scaled target when original_scale=False"
        )


class TestPosteriorPredictiveIdataOverride:
    def test_idata_override_uses_different_data(self, simple_plots, panel_idata):
        """panel_idata has a geo dim — should create more panels than simple_idata."""
        _, axes = simple_plots.posterior_predictive(idata=panel_idata)
        assert axes.size >= 2

    def test_idata_override_does_not_mutate_self(
        self, simple_plots, simple_idata, panel_idata
    ):
        """self._data must not be mutated after an idata override call."""
        simple_plots.posterior_predictive(idata=panel_idata)
        assert simple_plots._data.idata is simple_idata


# ============================================================================
# prior_predictive tests
# ============================================================================


class TestPriorPredictiveBasic:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.prior_predictive()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_single_panel_no_extra_dims(self, simple_plots):
        _, axes = simple_plots.prior_predictive()
        assert axes.size == 1

    def test_return_as_pc(self, simple_plots):
        result = simple_plots.prior_predictive(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_raises_on_missing_var(self, simple_plots):
        with pytest.raises(ValueError, match="nonexistent"):
            simple_plots.prior_predictive(target_var="nonexistent")

    def test_error_messages_reference_prior(self):
        """prior_predictive error messages must say 'prior', not 'posterior'."""
        with pytest.raises(ValueError, match="prior_predictive"):
            data = MMMIDataWrapper(az.InferenceData(), validate_on_init=False)
            DiagnosticsPlots(data).prior_predictive()


class TestPriorPredictiveElements:
    def test_predicted_mean_line_present(self, simple_plots, simple_data):
        """The prior mean line y-data must match prior_ds['y'].mean(chain/draw)."""
        _, axes = simple_plots.prior_predictive()
        ax = axes.flat[0]
        expected = (
            simple_data.idata.prior_predictive["y"].mean(dim=("chain", "draw")).values
        )
        line_y_arrays = [line.get_ydata() for line in ax.lines]
        assert any(np.allclose(y, expected, equal_nan=True) for y in line_y_arrays), (
            "No line matches the prior predictive mean"
        )

    def test_hdi_band_present(self, simple_plots):
        """HDI fill_between band must create at least one collection in the axes."""
        _, axes = simple_plots.prior_predictive()
        ax = axes.flat[0]
        assert len(ax.collections) > 0, (
            "No HDI band found in axes (expected fill_between collection)"
        )

    def test_x_data_length_matches_dates(self, simple_plots):
        """All plotted lines must span the full date range (n_date=20)."""
        _, axes = simple_plots.prior_predictive()
        ax = axes.flat[0]
        n_date = 20  # matches simple_idata fixture
        for line in ax.lines:
            assert len(line.get_xdata()) == n_date

    def test_narrower_hdi_prob_gives_narrower_band(self, simple_plots):
        """A 50% HDI band must be narrower than the default 94% band."""
        _, axes_94 = simple_plots.prior_predictive(hdi_prob=0.94)
        _, axes_50 = simple_plots.prior_predictive(hdi_prob=0.50)
        ax_94 = axes_94.flat[0]
        ax_50 = axes_50.flat[0]

        def band_height(ax):
            verts = ax.collections[0].get_paths()[0].vertices
            return verts[:, 1].max() - verts[:, 1].min()

        assert band_height(ax_94) > band_height(ax_50), (
            "94% HDI band should be wider than 50% HDI band"
        )


class TestPriorPredictiveDims:
    def test_panel_idata_creates_multiple_panels(self, panel_plots):
        _, axes = panel_plots.prior_predictive()
        assert axes.size >= 2

    def test_dims_filter_single_value(self, panel_plots):
        _, axes = panel_plots.prior_predictive(dims={"geo": ["CA"]})
        assert axes.size == 1


class TestPriorPredictiveCustomization:
    def test_figsize_sets_figure_dimensions(self, simple_plots):
        fig, _ = simple_plots.prior_predictive(figsize=(10, 4))
        w, h = fig.get_size_inches()
        assert abs(w - 10) < 0.1 and abs(h - 4) < 0.1

    def test_line_kwargs_color_applied(self, simple_plots):
        """line_kwargs color must appear on the prior mean line."""
        _, axes = simple_plots.prior_predictive(line_kwargs={"color": "green"})
        ax = axes.flat[0]
        colors = [line.get_color() for line in ax.lines]
        assert any(c == "green" for c in colors), (
            "No line with color='green' found — line_kwargs not applied"
        )

    def test_observed_kwargs_color_applied(self, simple_plots):
        _, axes = simple_plots.prior_predictive(observed_kwargs={"color": "gray"})
        ax = axes.flat[0]
        colors = [line.get_color() for line in ax.lines]
        assert any(c == "gray" for c in colors), (
            "No line with color='gray' found — observed_kwargs not applied"
        )


class TestPriorPredictiveObserved:
    def test_observed_values_match_target(self, simple_plots, simple_data):
        """The observed line y-data must match data.get_target(original_scale=False)
        when target_var='y' (scaled predictions)."""
        _, axes = simple_plots.prior_predictive(target_var="y")
        ax = axes.flat[0]
        expected = simple_data.get_target(original_scale=False).values
        line_y_arrays = [line.get_ydata() for line in ax.lines]
        assert any(np.allclose(y, expected, equal_nan=True) for y in line_y_arrays), (
            "No line in the axes matches the observed target values"
        )

    def test_observed_present_in_panel_plot(self, panel_plots, panel_data):
        """Multi-panel plot: observed data is present in each panel."""
        _, axes = panel_plots.prior_predictive()
        observed = panel_data.get_target(original_scale=True)
        for ax in axes.flat:
            line_y_arrays = [line.get_ydata() for line in ax.lines]
            assert any(len(y) == observed.sizes["date"] for y in line_y_arrays)


class TestPriorPredictiveIdataOverride:
    def test_idata_override_does_not_mutate_self(
        self, simple_plots, simple_idata, panel_idata
    ):
        simple_plots.prior_predictive(idata=panel_idata)
        assert simple_plots._data.idata is simple_idata


# ============================================================================
# residuals tests
# ============================================================================


class TestResidualsBasic:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.residuals()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_single_panel_no_extra_dims(self, simple_plots):
        _, axes = simple_plots.residuals()
        assert axes.size == 1

    def test_return_as_pc(self, simple_plots):
        result = simple_plots.residuals(return_as_pc=True)
        assert isinstance(result, PlotCollection)


class TestResidualsElements:
    def test_mean_residuals_line_present(self, simple_plots, simple_data):
        """Mean residuals line y-data must match _compute_residuals().mean(chain/draw)."""
        from pymc_marketing.mmm.plotting.diagnostics import _compute_residuals

        _, axes = simple_plots.residuals()
        ax = axes.flat[0]
        expected = _compute_residuals(simple_data).mean(dim=("chain", "draw")).values
        line_y_arrays = [line.get_ydata() for line in ax.lines]
        assert any(np.allclose(y, expected, equal_nan=True) for y in line_y_arrays), (
            "No line matches the mean residuals"
        )

    def test_zero_hline_present(self, simple_plots):
        """A horizontal line at y=0 (reference line) must be present."""
        _, axes = simple_plots.residuals()
        ax = axes.flat[0]
        # axhline creates a Line2D whose y-data is constant 0.0
        hline_y_arrays = [line.get_ydata() for line in ax.lines]
        assert any(np.all(np.isclose(y, 0.0)) for y in hline_y_arrays), (
            "No zero reference line found at y=0"
        )

    def test_hdi_band_present(self, simple_plots):
        """HDI fill_between band must create at least one collection in the axes."""
        _, axes = simple_plots.residuals()
        ax = axes.flat[0]
        assert len(ax.collections) > 0, (
            "No HDI band found in residuals axes (expected fill_between collection)"
        )

    def test_narrower_hdi_prob_gives_narrower_band(self, simple_plots):
        """A 50% HDI band must be narrower than the default 94% band."""
        _, axes_94 = simple_plots.residuals(hdi_prob=0.94)
        _, axes_50 = simple_plots.residuals(hdi_prob=0.50)
        ax_94 = axes_94.flat[0]
        ax_50 = axes_50.flat[0]

        def band_height(ax):
            verts = ax.collections[0].get_paths()[0].vertices
            return verts[:, 1].max() - verts[:, 1].min()

        assert band_height(ax_94) > band_height(ax_50), (
            "94% HDI band should be wider than 50% HDI band"
        )

    def test_x_data_length_matches_dates(self, simple_plots):
        """All plotted lines must span the full date range (n_date=20)."""
        _, axes = simple_plots.residuals()
        ax = axes.flat[0]
        n_date = 20
        for line in ax.lines:
            assert len(line.get_xdata()) == n_date


class TestResidualsDims:
    def test_panel_idata_creates_multiple_panels(self, panel_plots):
        _, axes = panel_plots.residuals()
        assert axes.size >= 2

    def test_dims_filter(self, panel_plots):
        _, axes = panel_plots.residuals(dims={"geo": ["CA"]})
        assert axes.size == 1


class TestResidualsCustomization:
    def test_figsize_sets_figure_dimensions(self, simple_plots):
        fig, _ = simple_plots.residuals(figsize=(10, 4))
        w, h = fig.get_size_inches()
        assert abs(w - 10) < 0.1 and abs(h - 4) < 0.1

    def test_hdi_kwargs_alpha_applied(self, simple_plots):
        """hdi_kwargs alpha must appear on the HDI collection."""
        _, axes = simple_plots.residuals(hdi_kwargs={"alpha": 0.05})
        ax = axes.flat[0]
        alphas = [col.get_alpha() for col in ax.collections]
        assert any(a is not None and abs(a - 0.05) < 1e-6 for a in alphas), (
            "No collection with alpha=0.05 found — hdi_kwargs not applied"
        )

    def test_line_kwargs_color_applied(self, simple_plots):
        """line_kwargs color must appear on the mean residuals line."""
        _, axes = simple_plots.residuals(line_kwargs={"color": "red"})
        ax = axes.flat[0]
        colors = [line.get_color() for line in ax.lines]
        assert any(c == "red" for c in colors), (
            "No line with color='red' found — line_kwargs not applied"
        )


class TestResidualsIdataOverride:
    def test_idata_override_uses_different_data(self, simple_plots, panel_idata):
        _, axes = simple_plots.residuals(idata=panel_idata)
        assert axes.size >= 2

    def test_idata_override_does_not_mutate_self(
        self, simple_plots, simple_idata, panel_idata
    ):
        simple_plots.residuals(idata=panel_idata)
        assert simple_plots._data.idata is simple_idata


# ============================================================================
# residuals_distribution tests
# ============================================================================


class TestResidualsDistributionBasic:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.residuals_distribution()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_no_extra_dims_single_panel(self, simple_plots):
        _, axes = simple_plots.residuals_distribution()
        assert axes.size == 1

    def test_return_as_pc_returns_plot_collection(self, simple_plots):
        result = simple_plots.residuals_distribution(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_invalid_quantile_raises(self, simple_plots):
        with pytest.raises(ValueError, match="quantile"):
            simple_plots.residuals_distribution(quantiles=[0.5, 1.5])

    def test_invalid_aggregation_dim_raises(self, simple_plots):
        with pytest.raises(ValueError, match="aggregation"):
            simple_plots.residuals_distribution(aggregation=["nonexistent"])

    def test_non_matplotlib_backend_without_return_as_pc_raises(self, simple_plots):
        with pytest.raises(ValueError, match="return_as_pc"):
            simple_plots.residuals_distribution(backend="plotly")


class TestResidualsDistributionElements:
    def test_kde_curve_present(self, simple_plots):
        """KDE must render at least one line (the density curve) in the axes."""
        _, axes = simple_plots.residuals_distribution()
        ax = axes.flat[0]
        assert len(ax.lines) > 0, (
            "No KDE curve found — expected at least one Line2D from azp.plot_dist"
        )

    def test_default_three_quantile_lines_present(self, simple_plots):
        """Default quantiles=[0.25, 0.5, 0.75] must produce exactly 3 reference lines
        beyond the KDE curve (total >= 4 lines)."""
        _, axes = simple_plots.residuals_distribution()
        ax = axes.flat[0]
        # KDE curve + 3 quantile reference lines
        assert len(ax.lines) >= 4, (
            f"Expected KDE + 3 quantile lines (>=4 total), found {len(ax.lines)}"
        )

    def test_custom_quantile_count_matches(self, simple_plots):
        """Passing 2 custom quantiles must produce 2 reference lines (KDE + 2 = >=3)."""
        _, axes = simple_plots.residuals_distribution(quantiles=[0.1, 0.9])
        ax = axes.flat[0]
        assert len(ax.lines) >= 3, (
            f"Expected KDE + 2 quantile lines (>=3 total), found {len(ax.lines)}"
        )

    def test_quantile_lines_are_vertical(self, simple_plots):
        """Quantile reference lines must be vertical (constant x across all y-points)."""
        _, axes = simple_plots.residuals_distribution()
        ax = axes.flat[0]
        # Skip the first line (KDE curve); quantile lines have identical x values
        vertical_lines = [
            line for line in ax.lines if len(set(np.round(line.get_xdata(), 6))) == 1
        ]
        assert len(vertical_lines) >= 3, (
            f"Expected >=3 vertical quantile lines, found {len(vertical_lines)}"
        )

    def test_quantile_x_positions_match_computed_quantiles(
        self, simple_plots, simple_data
    ):
        """The x-positions of quantile lines must match the computed quantile values."""
        from pymc_marketing.mmm.plotting.diagnostics import _compute_residuals

        _, axes = simple_plots.residuals_distribution()
        ax = axes.flat[0]
        residuals = _compute_residuals(simple_data)
        expected_quantiles = np.quantile(residuals.values.ravel(), [0.25, 0.5, 0.75])
        vertical_x = sorted(
            [
                line.get_xdata()[0]
                for line in ax.lines
                if len(set(np.round(line.get_xdata(), 6))) == 1
            ]
        )
        assert len(vertical_x) == len(expected_quantiles), (
            f"Expected {len(expected_quantiles)} quantile lines, got {len(vertical_x)}"
        )
        assert np.allclose(vertical_x, np.sort(expected_quantiles), rtol=1e-3), (
            f"Quantile line positions {vertical_x} don't match expected {expected_quantiles}"
        )


class TestResidualsDistributionAggregation:
    def test_aggregation_none_panel_idata_multiple_panels(self, panel_plots):
        _, axes = panel_plots.residuals_distribution()
        assert axes.size >= 2  # one per geo value — geo is structural by default

    def test_aggregation_geo_collapses_to_single_panel(self, panel_plots):
        _, axes = panel_plots.residuals_distribution(aggregation=["geo"])
        assert axes.size == 1


class TestResidualsDistributionDims:
    def test_dims_filter(self, panel_plots):
        _, axes = panel_plots.residuals_distribution(dims={"geo": ["CA"]})
        assert axes.size == 1


class TestResidualsDistributionIdataOverride:
    def test_idata_override_does_not_mutate_self(
        self, simple_plots, simple_idata, panel_idata
    ):
        simple_plots.residuals_distribution(idata=panel_idata)
        assert simple_plots._data.idata is simple_idata


# ============================================================================
# Package-level import test
# ============================================================================


def test_diagnostics_plots_importable_from_package():
    """DiagnosticsPlots must be importable from pymc_marketing.mmm.plotting."""
    from pymc_marketing.mmm.plotting import DiagnosticsPlots as DP

    assert DP is DiagnosticsPlots
