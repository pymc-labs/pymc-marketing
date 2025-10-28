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
"""Time-slice cross-validation utilities for PyMC-Marketing MMM.

This module provides the TimeSliceCrossValidator which can run rolling
time-slice cross-validation for media-mix models built with the library.
The validator does not retain a fitted MMM instance; models may be
constructed per-fold from a YAML configuration or supplied to ``run()``.
"""

from dataclasses import dataclass
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from pymc_marketing.metrics import crps
from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml


@dataclass
class TimeSliceCrossValidationResult:
    """Container for the results of one time-slice CV step."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    idata: az.InferenceData


class TimeSliceCrossValidator:
    """
    Time-Slice Cross Validator for Media Mix Models (MMM).

    Provides scikit-learn-style API:
        - split(X, y): yields train/test indices
        - get_n_splits(): returns number of splits
        - run(): executes full CV loop and returns fitted models and predictions

    Optional sampler configuration
    ------------------------------
    This validator accepts an optional ``sampler_config`` either at construction
    time or when calling ``run``. The configuration is applied to the underlying
    MMM object by setting ``mmm.sampler_config`` before each fold is fitted.

    Example (set on construction):

        cv = TimeSliceCrossValidator(
            n_init=158,
            forecast_horizon=10,
            date_column='date',
            step_size=50,
            sampler_config={
                'tune': 500,
                'draws': 200,
                'chains': 4,
                'random_seed': 123,
                'target_accept': 0.90,
                'nuts_sampler': 'nutpie',
                'nuts_sampler_kwargs': {'backend': 'jax', 'gradient_backend': 'jax'},
            }
        )

    Example (override per run):

        cv.run(X, y, sampler_config={'draws': 1000, 'tune': 1000})
    """

    def __init__(
        self,
        n_init: int,
        forecast_horizon: int,
        date_column: str,
        step_size: int = 1,
        sampler_config: dict | None = None,
    ):
        if not isinstance(step_size, int) or step_size <= 0:
            raise ValueError("step_size must be a positive integer")
        if not isinstance(n_init, int) or n_init <= 0:
            raise ValueError("n_init must be a positive integer")
        if not isinstance(forecast_horizon, int) or forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be a positive integer")
        self.n_init = n_init
        self.forecast_horizon = forecast_horizon
        self.date_column = date_column
        self.step_size = step_size
        # Optional sampler configuration that will be applied to the MMM prior to fitting
        # Can be provided here at construction or passed to run() to override per-run.
        self.sampler_config = sampler_config

    # Model helpers
    def _fit_mmm(self, mmm, X, y, sampler_config: dict | None = None):
        """Fit the MMM and sample posterior predictive.

        sampler_config, if provided, will be set on the MMM instance before fitting.
        """
        # Determine which sampler config to apply (explicit override takes precedence)
        effective_sampler_config = (
            sampler_config if sampler_config is not None else self.sampler_config
        )
        if effective_sampler_config is not None:
            # Set the sampler config on the model prior to fitting
            mmm.sampler_config = effective_sampler_config

        _ = mmm.fit(
            X,
            y,
            progressbar=True,
        )
        _ = mmm.sample_posterior_predictive(
            X,
            extend_idata=True,
            combined=True,
            progressbar=False,
        )
        return mmm

    def _time_slice_step(
        self, mmm, X_train, y_train, X_test, y_test, sampler_config: dict | None = None
    ):
        """Run one CV step and return results."""
        # Fit the model for this fold. sampler_config can override the validator-level config.
        mmm = self._fit_mmm(mmm, X_train, y_train, sampler_config=sampler_config)

        # Combine train and test data for posterior predictions
        X_combined = pd.concat([X_train, X_test], ignore_index=True)

        # Run posterior predictions on combined data with extend_idata=True
        _ = mmm.sample_posterior_predictive(
            X=X_combined,
            include_last_observations=False,
            extend_idata=True,
            progressbar=False,
        )

        return TimeSliceCrossValidationResult(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            idata=mmm.idata,
        )

    def get_n_splits(self, X, y=None):
        """Return number of possible rolling splits."""
        total_dates = len(X[self.date_column].unique())
        # Calculate how many splits we can make with the given step_size
        # We need at least n_init + forecast_horizon dates for one split
        # With step_size, we can make splits at positions: 0, step_size, 2*step_size, ...
        # The last possible split position is: total_dates - n_init - forecast_horizon
        # So the number of splits is: floor((total_dates - n_init - forecast_horizon) / step_size) + 1

        max_splits = (
            total_dates - self.n_init - self.forecast_horizon
        ) // self.step_size + 1
        return max(0, max_splits)

    def split(self, X, y):
        """Yield (train_idx, test_idx) pairs for each time-slice split.

        This implementation selects rows by date masks so that all coordinate
        levels (e.g. multiple geos) for the selected date ranges are included
        in each fold. It returns integer positions suitable for use with
        ``DataFrame.iloc``.
        """
        n_splits = self.get_n_splits(X, y)
        if n_splits <= 0:
            raise ValueError(
                "No splits possible with the given n_init, forecast_horizon and step_size"
            )

        # unique sorted dates
        udates = np.unique(pd.to_datetime(X[self.date_column].to_numpy()))

        for i in range(n_splits):
            start_date = udates[i * self.step_size + self.n_init]
            end_date = udates[
                i * self.step_size + self.n_init + self.forecast_horizon - 1
            ]

            # boolean masks selecting rows for train and test ranges (preserve all geos)
            train_mask = pd.to_datetime(X[self.date_column]) < start_date
            test_mask = (pd.to_datetime(X[self.date_column]) >= start_date) & (
                pd.to_datetime(X[self.date_column]) <= end_date
            )

            train_idx = np.flatnonzero(train_mask.to_numpy())
            test_idx = np.flatnonzero(test_mask.to_numpy())

            yield train_idx, test_idx

    # Run CV
    def run(
        self,
        X,
        y,
        sampler_config: dict | None = None,
        yaml_path: str | None = None,
        mmm: Any | None = None,
    ) -> list[TimeSliceCrossValidationResult]:
        """Run the complete time-slice CV loop.

        If `yaml_path` is provided, the validator will rebuild the MMM from the
        YAML for each fold using the training data before calling `_fit_mmm`.

        sampler_config: Optional dict to override the validator-level sampler configuration
                        for all folds in this run. If provided here it takes precedence
                        over the configuration passed at construction time.

        Example:

            # Use a lighter sampler for quick checks
            cv.run(X, y, sampler_config={
                'tune': 300,
                'draws': 300,
                'chains': 2,
            })

        Returns
        -------
            List[TimeSliceCrossValidationResult]
        """
        results = []
        for _i, (train_idx, test_idx) in enumerate(tqdm(self.split(X, y))):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            # Optionally (re)build the model from yaml using the training fold
            if yaml_path is not None:
                fold_mmm = build_mmm_from_yaml(
                    config_path=yaml_path, X=X_train, y=y_train
                )
            elif mmm is not None:
                # use provided mmm instance (do not store it on self)
                fold_mmm = mmm.build_model(X_train, y_train)
            else:
                raise ValueError(
                    "Either provide an `mmm` instance to run(...) or a `yaml_path` to build the model per-fold."
                )

            result = self._time_slice_step(
                fold_mmm,
                X_train,
                y_train,
                X_test,
                y_test,
                sampler_config=sampler_config,
            )
            results.append(result)
        return results

    # Visualization helpers
    def plot_predictions(self, results, dims: dict[str, list[str]] | None = None):
        """Plot posterior predictive predictions across CV folds.

        Args:
            results: List of TimeSliceCrossValidationResult objects returned by `run()`.
            dims: Optional dict specifying dimensions to filter when plotting.
                Currently only supports {"geo": [...]}.
                If omitted, all geos present in the posterior predictive
                `y_original_scale` are plotted.

        The plot shows per-geo, per-fold posterior predictive HDI (3%-97%)
        for train (blue) and test (orange) ranges as shaded bands, the
        posterior mean as dashed lines, and observed values as black lines.
        A vertical dashed green line marks the end of the training period for
        each fold.
        """
        # Support optional dims filtering (currently only supports filtering by 'geo')
        if dims is None:
            geos = list(
                results[0]
                .idata.posterior_predictive["y_original_scale"]
                .coords["geo"]
                .values
            )
        else:
            # Only 'geo' filtering is supported for predictions plotting
            unsupported = [d for d in dims.keys() if d != "geo"]
            if unsupported:
                raise ValueError(
                    f"plot_predictions only supports dims with 'geo' key. Unsupported dims: {unsupported}"
                )
            geos = list(dims.get("geo", []))
            if not geos:
                # fallback to all geos if empty list provided
                geos = list(
                    results[0]
                    .idata.posterior_predictive["y_original_scale"]
                    .coords["geo"]
                    .values
                )

        n_folds = len(results)
        n_axes = len(geos) * n_folds

        fig, axes = plt.subplots(n_axes, 1, figsize=(12, 4 * n_axes), sharex=True)
        if n_axes == 1:
            axes = [axes]

        # Helper to align y Series to a DataFrame's rows without using reindex (avoids duplicate-index errors)
        def _align_y_to_df(y_series, df):
            y_df = y_series.reset_index()
            y_df.columns = ["orig_index", "y_value"]
            df_idx = pd.DataFrame({"orig_index": df.index, "date": df["date"].values})
            merged = df_idx.merge(y_df, on="orig_index", how="left")
            return merged["y_value"], merged["date"]

        for geo_idx, geo in enumerate(geos):
            for fold_idx, result in enumerate(results):
                ax_i = geo_idx * n_folds + fold_idx
                ax = axes[ax_i]

                arr = result.idata.posterior_predictive["y_original_scale"]
                # Stack chain/draw -> sample for quantile computation
                arr_s = arr.stack(sample=("chain", "draw")).transpose(
                    "sample", "date", "geo"
                )

                # Train / Test dates for this fold & geo
                if "geo" in result.X_train.columns:
                    train_df_geo = result.X_train[
                        result.X_train["geo"].astype(str) == str(geo)
                    ]
                else:
                    train_df_geo = result.X_train.copy()
                if "geo" in result.X_test.columns:
                    test_df_geo = result.X_test[
                        result.X_test["geo"].astype(str) == str(geo)
                    ]
                else:
                    test_df_geo = result.X_test.copy()

                train_dates = (
                    pd.to_datetime(train_df_geo["date"].values)
                    if not train_df_geo.empty
                    else pd.DatetimeIndex([])
                )
                test_dates = (
                    pd.to_datetime(test_df_geo["date"].values)
                    if not test_df_geo.empty
                    else pd.DatetimeIndex([])
                )
                train_dates = train_dates.sort_values().unique()
                test_dates = test_dates.sort_values().unique()

                # Build selection dict for arr_s.sel; we always set geo and date
                # Note: arr_s has coordinates named 'date' and 'geo' after transpose
                # Additional dims are not supported here
                # Compute and plot HDI for train (blue) if train_dates exist
                # Compute and plot HDI with arviz.plot_hdi (fallback to manual if unavailable)

                def _plot_hdi_from_sel(sel, ax, color, label):
                    """
                    Robust wrapper to call arviz.plot_hdi from an xarray DataArray `sel`.

                    Ensures the data passed to az.plot_hdi has shape (n_samples, n_dates).
                    If sel collapses to a scalar (0-d), the function will skip plotting.
                    """
                    # Squeeze out any length-1 dimensions (e.g. geo if it became a scalar coord)
                    try:
                        sel2 = sel.squeeze()
                    except Exception:
                        sel2 = sel

                    # Extract numpy array
                    arr = getattr(sel2, "values", sel2)

                    # If scalar, nothing to plot
                    if getattr(arr, "ndim", 0) == 0:
                        return

                    # If 1D, decide whether it's (samples,) or (dates,)
                    if arr.ndim == 1:
                        # prefer to treat as (samples, 1) if there is a sample coord
                        if hasattr(sel2, "coords") and "sample" in sel2.coords:
                            arr = arr.reshape((-1, 1))
                            x = (
                                sel2.coords["date"].values
                                if "date" in sel2.coords
                                else [sel2.coords.get("date")]
                            )
                        else:
                            # otherwise assume it's (dates,) -> make (1, dates)
                            arr = arr.reshape((1, -1))
                            x = (
                                sel2.coords["date"].values
                                if "date" in sel2.coords
                                else [sel2.coords.get("date")]
                            )
                    else:
                        # arr.ndim >= 2. Ensure ordering is (sample, date)
                        if hasattr(sel2, "dims"):
                            dims = list(sel2.dims)
                            if dims == ["date", "sample"]:
                                arr = arr.T
                            elif dims != ["sample", "date"]:
                                # try to transpose into desired order if possible
                                try:
                                    sel2 = sel2.transpose("sample", "date")
                                    arr = sel2.values
                                except Exception as e:
                                    # fallback: leave arr as-is
                                    raise e
                        x = (
                            sel2.coords["date"].values
                            if hasattr(sel2, "coords") and "date" in sel2.coords
                            else None
                        )

                    # Finally call arviz. If x is None, let arviz handle it (will use integer indices)
                    az.plot_hdi(
                        y=arr,
                        x=x,
                        ax=ax,
                        hdi_prob=0.94,
                        color=color,
                        smooth=False,
                        fill_kwargs={"alpha": 0.25, "label": label},
                        plot_kwargs={"color": color, "linestyle": "--", "linewidth": 1},
                    )

                if train_dates.size:
                    sel = arr_s.sel(date=train_dates, geo=geo)
                    _plot_hdi_from_sel(sel, ax, "C0", "HDI (train)")

                if test_dates.size:
                    sel = arr_s.sel(date=test_dates, geo=geo)
                    _plot_hdi_from_sel(sel, ax, "C1", "HDI (test)")

                # Plot observed actuals in black (train + test) as lines (no markers)
                if not train_df_geo.empty:
                    y_train_vals, train_plot_dates = _align_y_to_df(
                        result.y_train, train_df_geo
                    )
                    y_train_vals = y_train_vals.dropna()
                    if not y_train_vals.empty:
                        dates_to_plot = pd.to_datetime(
                            train_plot_dates.loc[y_train_vals.index].values
                        )
                        ax.plot(
                            dates_to_plot,
                            y_train_vals.values,
                            color="black",
                            linestyle="-",
                            linewidth=1.5,
                            label="observed",
                        )

                if not test_df_geo.empty:
                    y_test_vals, test_plot_dates = _align_y_to_df(
                        result.y_test, test_df_geo
                    )
                    y_test_vals = y_test_vals.dropna()
                    if not y_test_vals.empty:
                        dates_to_plot = pd.to_datetime(
                            test_plot_dates.loc[y_test_vals.index].values
                        )
                        ax.plot(
                            dates_to_plot,
                            y_test_vals.values,
                            color="black",
                            linestyle="-",
                            linewidth=1.5,
                        )

                # Vertical line marking end of training
                if train_dates.size:
                    end_train_date = pd.to_datetime(train_dates.max())
                    ax.axvline(
                        end_train_date,
                        color="green",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.9,
                        label="train end",
                    )

                ax.set_title(f"{geo} — Fold {fold_idx} — Posterior Predictive")
                ax.set_ylabel("y_original_scale")

        # Build a single unique legend placed at the bottom of the figure
        handles, labels = [], []
        for ax in axes:
            h, _l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(_l)
        by_label = dict(zip(labels, handles, strict=False))
        if by_label:
            plt.tight_layout(rect=[0, 0.07, 1, 1])
            ncol = min(4, len(by_label))
            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc="lower center",
                ncol=ncol,
                bbox_to_anchor=(0.5, 0.01),
            )
        else:
            plt.tight_layout()

        axes[-1].set_xlabel("date")
        plt.show()

    def plot_param_stability(
        self, results, parameter: list[str], dims: dict[str, list[str]] | None = None
    ):
        """
        Plot parameter stability across CV iterations.

        Args:
            results: list of TimeSliceCrossValidationResult or similar objects containing idata
            parameter: list of parameter names (e.g. ["beta_channel"])
            dims: optional dict specifying dimensions and coordinate values to slice over
                e.g. {"geo": ["geo_a", "geo_b"]}
        """
        if dims is None:
            # --- No dims: standard forest plot ---
            fig, ax = plt.subplots(figsize=(9, 6))
            az.plot_forest(
                data=[r.idata["posterior"] for r in results],
                model_names=[f"Iteration {i}" for i in range(len(results))],
                var_names=parameter,
                combined=True,
                ax=ax,
            )
            fig.suptitle(
                f"Parameter Stability: {parameter}",
                fontsize=18,
                fontweight="bold",
                y=1.06,
            )
            plt.show()

        else:
            # --- Plot one forest plot per dim value ---
            for dim_name, coord_values in dims.items():
                for coord in coord_values:
                    fig, ax = plt.subplots(figsize=(9, 6))
                    az.plot_forest(
                        data=[
                            r.idata["posterior"].sel({dim_name: coord}) for r in results
                        ],
                        model_names=[f"Iteration {i}" for i in range(len(results))],
                        var_names=parameter,
                        combined=True,
                        ax=ax,
                    )
                    fig.suptitle(
                        f"Parameter Stability: {parameter} | {dim_name}={coord}",
                        fontsize=18,
                        fontweight="bold",
                        y=1.06,
                    )
                    plt.show()

    def plot_crps(self, results):
        """Plot CRPS for train and test sets across all CV splits."""

        def _pred_matrix_for_rows(idata, rows_df):
            """Build (n_samples, n_rows) prediction matrix for given rows DataFrame.

            For each row in rows_df we select the posterior predictive samples
            corresponding to that row's date (and geo if present in the rows_df).
            This is robust to the ordering of dimensions in the xarray DataArray.
            """
            da = idata.posterior_predictive["y_original_scale"]
            # Stack sample dims (chain, draw) into single 'sample' dim
            da_s = da.stack(sample=("chain", "draw"))

            # Ensure 'sample' is the first axis for easier indexing
            if da_s.dims[0] != "sample":
                try:
                    da_s = da_s.transpose("sample", ...)
                except Exception:
                    dims = list(da_s.dims)
                    order = ["sample"] + [d for d in dims if d != "sample"]
                    da_s = da_s.transpose(*order)

            n_samples = int(da_s.sizes["sample"])
            n_rows = len(rows_df)
            mat = np.empty((n_samples, n_rows))

            for j, (_idx, row) in enumerate(rows_df.iterrows()):
                # select by date
                sel = da_s.sel({self.date_column: row[self.date_column]})
                # if geo is present in the data and in the row, select it
                if "geo" in sel.dims and "geo" in rows_df.columns:
                    sel = sel.sel(geo=str(row["geo"]))

                arr = np.squeeze(getattr(sel, "values", sel))
                if arr.ndim == 0:
                    raise ValueError(
                        "Posterior predictive selection returned a scalar for a row"
                    )
                # ensure shape (n_samples,)
                if arr.ndim > 1:
                    # try to collapse remaining dims, expecting (sample, ...)
                    arr = arr.reshape(n_samples, -1)[:, 0]

                mat[:, j] = arr

            return mat

        crps_results_train = []
        for result in results:
            X_train_rows = result.X_train.reset_index(drop=True)
            y_pred_train = _pred_matrix_for_rows(result.idata, X_train_rows)
            crps_results_train.append(
                crps(y_true=result.y_train.to_numpy(), y_pred=y_pred_train)
            )

        crps_results_test = []
        for result in results:
            # Only consider test rows for this fold (preserves order)
            X_test_rows = result.X_test.reset_index(drop=True)
            y_pred_test = _pred_matrix_for_rows(result.idata, X_test_rows)
            crps_results_test.append(
                crps(y_true=result.y_test.to_numpy(), y_pred=y_pred_test)
            )

        fig, ax = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(12, 7),
            sharex=True,
            sharey=False,
            layout="constrained",
        )

        ax[0].plot(crps_results_train, marker="o", color="C0", label="train")
        ax[0].set(ylabel="CRPS", title="Train CRPS")
        ax[1].plot(crps_results_test, marker="o", color="C1", label="test")
        ax[1].set(xlabel="Iteration", ylabel="CRPS", title="Test CRPS")
        fig.suptitle("CRPS for each iteration", fontsize=18, fontweight="bold", y=1.05)
        plt.show()
