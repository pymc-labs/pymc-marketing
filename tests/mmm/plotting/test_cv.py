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
import matplotlib

matplotlib.use("Agg")

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.figure import Figure

SEED = 42


@pytest.fixture(scope="module")
def cv_results_idata():
    """Minimal az.InferenceData for MMMCVPlotSuite tests.

    Three folds over 30 daily dates:
      fold_0 — train 0-19, test 20-29
      fold_1 — train 0-24, test 25-29
      fold_2 — train 0-29, test [] (degenerate fold, no test rows)
    """
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    cv_labels = ["fold_0", "fold_1", "fold_2"]
    channels = ["tv", "radio"]
    n_chains, n_draws = 2, 50

    posterior_ds = xr.Dataset(
        {
            "beta_channel": xr.DataArray(
                rng.normal(size=(3, n_chains, n_draws, 2)),
                dims=["cv", "chain", "draw", "channel"],
                coords={
                    "cv": cv_labels,
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "channel": channels,
                },
            )
        }
    )

    pp_ds = xr.Dataset(
        {
            "y_original_scale": xr.DataArray(
                rng.normal(100, 10, size=(3, n_chains, n_draws, 30)),
                dims=["cv", "chain", "draw", "date"],
                coords={
                    "cv": cv_labels,
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "date": dates,
                },
            )
        }
    )

    fold_specs = [(20, 20), (25, 25), (30, 30)]
    meta_arr = np.empty(3, dtype=object)
    for i, (train_end, test_start) in enumerate(fold_specs):
        X_train = pd.DataFrame({"date": dates[:train_end]})
        y_train = pd.Series(rng.normal(100, 10, size=train_end), name="y")
        if test_start < 30:
            X_test = pd.DataFrame({"date": dates[test_start:]})
            y_test = pd.Series(rng.normal(100, 10, size=30 - test_start), name="y")
        else:
            X_test = pd.DataFrame({"date": pd.DatetimeIndex([])})
            y_test = pd.Series([], name="y", dtype=float)
        meta_arr[i] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    cv_metadata_ds = xr.Dataset(
        {
            "metadata": xr.DataArray(
                meta_arr,
                dims=["cv"],
                coords={"cv": cv_labels},
            )
        }
    )

    return az.InferenceData(
        posterior=posterior_ds,
        posterior_predictive=pp_ds,
        cv_metadata=cv_metadata_ds,
    )


@pytest.fixture(scope="module")
def cv_plot(cv_results_idata):
    from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

    return MMMCVPlotSuite(cv_results_idata)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


class TestInit:
    def test_stores_cv_data(self, cv_results_idata):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

        suite = MMMCVPlotSuite(cv_results_idata)
        assert suite.cv_data is cv_results_idata

    def test_raises_type_error_for_non_idata(self):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

        with pytest.raises(TypeError, match=r"az\.InferenceData"):
            MMMCVPlotSuite({"not": "idata"})

    def test_raises_value_error_without_cv_metadata(self):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

        bad = az.InferenceData(posterior=xr.Dataset())
        with pytest.raises(ValueError, match="cv_metadata"):
            MMMCVPlotSuite(bad)


class TestPredictions:
    def test_returns_tuple(self, cv_plot):
        result = cv_plot.predictions()
        assert isinstance(result, tuple) and len(result) == 2
        fig, axes = result
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_return_as_pc(self, cv_plot):
        result = cv_plot.predictions(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_n_axes_equals_n_folds(self, cv_plot):
        _fig, axes = cv_plot.predictions()
        # 3 folds → at least 3 axes (one per fold panel)
        assert len(axes) >= 3

    def test_train_test_colors_differ(self, cv_plot):
        _fig, axes = cv_plot.predictions()
        ax = axes[0]
        colors = set()
        for coll in ax.collections:
            fc = coll.get_facecolor()
            if fc is not None and len(fc) > 0:
                colors.add(tuple(np.round(fc[0][:3], 2)))
        assert len(colors) >= 2, "Expected at least two fill colors (train/test)"

    def test_missing_cv_metadata_raises(self, cv_plot, cv_results_idata):
        bad = az.InferenceData(
            posterior_predictive=cv_results_idata.posterior_predictive
        )
        # bad has no cv_metadata — _validate_cv_results raises ValueError
        with pytest.raises((TypeError, ValueError)):
            cv_plot.predictions(cv_data=bad)

    def test_missing_posterior_predictive_raises(self, cv_plot, cv_results_idata):
        bad = az.InferenceData(cv_metadata=cv_results_idata.cv_metadata)
        with pytest.raises(ValueError, match="posterior_predictive"):
            cv_plot.predictions(cv_data=bad)

    def test_dims_filtering(self, cv_plot):
        import pandas as pd

        # Filter to a single date — date dim becomes size-1, plot still renders
        single_date = pd.Timestamp("2024-01-05")
        fig, _axes = cv_plot.predictions(dims={"date": [single_date]})
        assert isinstance(fig, Figure)

    def test_split_line_present(self, cv_plot):
        _fig, axes = cv_plot.predictions()
        for ax in axes:
            dashed_lines = [
                line for line in ax.lines if line.get_linestyle() in ("--", "dashed")
            ]
            assert len(dashed_lines) == 1, (
                "Expected exactly one dashed vertical split line per subplot"
            )

    def test_subplot_titles_contain_fold(self, cv_plot):
        _fig, axes = cv_plot.predictions()
        titles = [ax.get_title() for ax in axes]
        for lbl in ("fold_0", "fold_1", "fold_2"):
            assert any(lbl in t for t in titles), f"No subplot title contains '{lbl}'"

    def test_subplot_titles_geo(self, cv_results_idata_geo):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

        suite = MMMCVPlotSuite(cv_results_idata_geo)
        _fig, axes = suite.predictions()
        titles = [ax.get_title() for ax in axes]
        assert any("geo_a" in t for t in titles), "No title contains 'geo_a'"
        assert any("geo_b" in t for t in titles), "No title contains 'geo_b'"
        assert any("fold_0" in t for t in titles), "No title contains 'fold_0'"

    def test_figure_renders_without_overflow(self, cv_plot):
        """Saving the figure must not raise OverflowError.

        Regression: azp.add_lines used to receive datetime64[ns] values which
        set the x-axis limits to nanosecond timestamps (~1.7e18), causing
        matplotlib's AutoDateLocator to overflow when formatting date ticks.
        """
        import io

        fig, _axes = cv_plot.predictions()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")


class TestParamStability:
    def test_returns_tuple(self, cv_plot):
        fig, axes = cv_plot.param_stability()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_return_as_pc(self, cv_plot):
        result = cv_plot.param_stability(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_var_names(self, cv_plot):
        # Should run without error when restricting to known variable
        fig, _axes = cv_plot.param_stability(var_names=["beta_channel"])
        assert isinstance(fig, Figure)

    def test_dims_filtering(self, cv_plot):
        # Filter posterior to a single channel before plotting
        fig, _axes = cv_plot.param_stability(dims={"channel": ["tv"]})
        assert isinstance(fig, Figure)

    def test_no_cv_coord_raises(self, cv_results_idata):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

        # Strip cv coordinate from posterior
        posterior = cv_results_idata.posterior.isel(cv=0, drop=True)
        bad = az.InferenceData(
            posterior=posterior,
            cv_metadata=cv_results_idata.cv_metadata,
        )
        suite = MMMCVPlotSuite(bad)
        with pytest.raises(ValueError, match="cv"):
            suite.param_stability()

    def test_single_figure(self, cv_plot):
        import matplotlib.pyplot as plt

        plt.close("all")
        cv_plot.param_stability()
        assert len(plt.get_fignums()) == 1


@pytest.fixture(scope="module")
def cv_results_idata_geo():
    """az.InferenceData with an extra 'geo' dimension in y_original_scale.

    Mirrors a real multidimensional MMM where the model is fit per geo.
    Used to reproduce the CRPS all-NaN bug.
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    cv_labels = ["fold_0", "fold_1"]
    geos = ["geo_a", "geo_b"]
    n_chains, n_draws = 2, 10

    posterior_ds = xr.Dataset(
        {
            "beta_channel": xr.DataArray(
                rng.normal(size=(2, n_chains, n_draws, 1)),
                dims=["cv", "chain", "draw", "channel"],
                coords={
                    "cv": cv_labels,
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "channel": ["tv"],
                },
            )
        }
    )

    pp_ds = xr.Dataset(
        {
            "y_original_scale": xr.DataArray(
                rng.normal(100, 10, size=(2, n_chains, n_draws, 20, 2)),
                dims=["cv", "chain", "draw", "date", "geo"],
                coords={
                    "cv": cv_labels,
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "date": dates,
                    "geo": geos,
                },
            )
        }
    )

    # X_train / X_test rows include a 'geo' column — one row per (date, geo) combo
    fold_specs = [(15, 15), (20, 20)]
    meta_arr = np.empty(2, dtype=object)
    for i, (train_end, test_start) in enumerate(fold_specs):
        rows_train = [(d, g) for d in dates[:train_end] for g in geos]
        X_train = pd.DataFrame(
            {"date": [r[0] for r in rows_train], "geo": [r[1] for r in rows_train]}
        )
        y_train = pd.Series(rng.normal(100, 10, size=len(rows_train)), name="y")
        if test_start < 20:
            rows_test = [(d, g) for d in dates[test_start:] for g in geos]
            X_test = pd.DataFrame(
                {"date": [r[0] for r in rows_test], "geo": [r[1] for r in rows_test]}
            )
            y_test = pd.Series(rng.normal(100, 10, size=len(rows_test)), name="y")
        else:
            X_test = pd.DataFrame(
                {"date": pd.DatetimeIndex([]), "geo": pd.Series([], dtype=str)}
            )
            y_test = pd.Series([], name="y", dtype=float)
        meta_arr[i] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    cv_metadata_ds = xr.Dataset(
        {"metadata": xr.DataArray(meta_arr, dims=["cv"], coords={"cv": cv_labels})}
    )
    return az.InferenceData(
        posterior=posterior_ds, posterior_predictive=pp_ds, cv_metadata=cv_metadata_ds
    )


class TestCRPS:
    def test_returns_tuple(self, cv_plot):
        fig, axes = cv_plot.crps()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_return_as_pc(self, cv_plot):
        result = cv_plot.crps(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_line_count(self, cv_plot):
        # 1x2 grid: left panel = train, right panel = test; one line each
        _fig, axes = cv_plot.crps()
        assert len(axes) == 2
        assert len(axes[0].lines) == 1
        assert len(axes[1].lines) == 1

    def test_train_test_colors_differ(self, cv_plot):
        _fig, axes = cv_plot.crps()
        colors = {ax.lines[0].get_color() for ax in axes}
        assert len(colors) == 2, "Expected train and test panels in distinct colors"

    def test_subplot_titles(self, cv_plot):
        _fig, axes = cv_plot.crps()
        titles = [ax.get_title() for ax in axes]
        assert any("train" in t for t in titles), "No subplot title contains 'train'"
        assert any("test" in t for t in titles), "No subplot title contains 'test'"

    def test_subplot_titles_geo(self, cv_results_idata_geo):
        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

        suite = MMMCVPlotSuite(cv_results_idata_geo)
        _fig, axes = suite.crps()
        titles = [ax.get_title() for ax in axes]
        assert any("geo_a" in t for t in titles), "No title contains 'geo_a'"
        assert any("geo_b" in t for t in titles), "No title contains 'geo_b'"
        assert any("train" in t for t in titles), "No title contains 'train'"
        assert any("test" in t for t in titles), "No title contains 'test'"

    def test_missing_cv_metadata_raises(self, cv_plot, cv_results_idata):
        bad = az.InferenceData(
            posterior_predictive=cv_results_idata.posterior_predictive
        )
        with pytest.raises((TypeError, ValueError)):
            cv_plot.crps(cv_data=bad)

    def test_nan_tolerant(self, cv_plot, cv_results_idata):
        # fold_2 has an empty test set → test CRPS must be NaN
        from pymc_marketing.mmm.plotting.cv import (
            _crps_for_split,
            _extract_cv_labels,
            _read_fold_meta,
        )

        cv_labels = _extract_cv_labels(cv_results_idata)
        _, _, X_test, y_test = _read_fold_meta(
            cv_results_idata, cv_labels[-1]
        )  # fold_2
        result = _crps_for_split(cv_results_idata, cv_labels[-1], X_test, y_test, {})
        assert np.isnan(result), "fold_2 test CRPS must be NaN (empty test set)"
        # And rendering must not crash
        fig, _axes = cv_plot.crps()
        assert isinstance(fig, Figure)

    def test_crps_multidim_geo_no_nan(self, cv_results_idata_geo):
        """crps() must produce finite values for multidimensional models.

        2 geos x 2 splits = 4 panels; each panel must have exactly one line
        and at least one finite CRPS value (fold_1 test is legitimately NaN
        because it has no test rows, but fold_0 test is finite).

        Regression: _pred_matrix_for_rows only selected by 'date', returning a
        2-D array (n_samples, n_geo) per observation. This caused every CRPS
        computation to fail silently, leaving all scores as NaN.
        """
        import warnings

        from pymc_marketing.mmm.plotting.cv import MMMCVPlotSuite

        suite = MMMCVPlotSuite(cv_results_idata_geo)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _fig, axes = suite.crps()  # must not warn about failed CRPS
        assert len(axes) == 4, "Expected 2 geos x 2 splits = 4 panels"
        for ax in axes:
            assert len(ax.lines) == 1
            y_vals = ax.lines[0].get_ydata()
            assert np.any(np.isfinite(y_vals)), (
                "CRPS panel contains only NaN — CRPS computation failed"
            )
