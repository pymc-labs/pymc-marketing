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
        # y_combined is intentionally not used; drop to satisfy linters

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
    ):
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
    def plot_predictions(
        self, results, y, X, dims: dict[str, list[str]] | None = None
    ):  # TODO: add dims argument
        """Plot posterior predictive intervals for all CV slices.

        Args:
            results: list of TimeSliceCrossValidationResult
            y: full observed series (pd.Series) aligned with X
            X: full feature DataFrame containing the date column
            dims: optional dict specifying dimensions and coordinate values to slice over
                e.g. {"geo": ["geo_a", "geo_b"]}
                If a dim key is present but the list of coords is empty or None,
                the function will attempt to read coordinate values from
                ``self.mmm.model.coords[dim]`` (i.e. dims defined in the model).
        """
        n_iterations = len(results)

        def _plot_for_results(
            fig, axes, results, dim_name: str | None = None, coord: str | None = None
        ):
            import numpy as np

            # Normalize axes to a list for consistent indexing
            if isinstance(axes, (np.ndarray, list, tuple)):
                axes_list = np.ravel(axes).tolist()
            else:
                axes_list = [axes]

            for i, result in enumerate(results):
                ax = axes_list[i]

                # helper to select dim if present, with fallback and warning
                def _maybe_select(pred, dim_name, coord):
                    try:
                        if dim_name is not None and dim_name in pred.coords:
                            return pred.sel({dim_name: coord})
                    except Exception:
                        # If selection fails, fall back to pred unmodified
                        msg = (
                            "Warning: could not select "
                            f"{dim_name}={coord} on posterior_predictive; "
                            "using full predictions."
                        )
                        print(msg)
                    return pred

                # Plot in-sample predictions (blue)
                for hdi_prob in [0.94, 0.5]:
                    # Sum over any model dims (e.g. 'product') that are present in the posterior_predictive
                    pred = _maybe_select(
                        result.idata.posterior_predictive["y_original_scale"],
                        dim_name,
                        coord,
                    )
                    # identify dims that represent samples/time and should be preserved
                    sample_time_dims = ("chain", "draw", self.date_column)
                    dims_to_sum = [d for d in pred.dims if d not in sample_time_dims]
                    if dims_to_sum:
                        pred = pred.sum(dim=dims_to_sum)

                    az.plot_hdi(
                        x=result.X_train[self.date_column],
                        y=pred.sel(
                            {
                                self.date_column: slice(
                                    result.X_train[self.date_column].min(),
                                    result.X_train[self.date_column].max(),
                                )
                            }
                        ),
                        color="C0",
                        smooth=False,
                        hdi_prob=hdi_prob,
                        fill_kwargs={
                            "alpha": 0.4,
                            "label": f"{hdi_prob:.0%} HDI (train)",
                        },
                        ax=ax,
                    )

                # Plot out-of-sample predictions (orange)
                for hdi_prob in [0.94, 0.5]:
                    pred = _maybe_select(
                        result.idata.posterior_predictive["y_original_scale"],
                        dim_name,
                        coord,
                    )
                    sample_time_dims = ("chain", "draw", self.date_column)
                    dims_to_sum = [d for d in pred.dims if d not in sample_time_dims]
                    if dims_to_sum:
                        pred = pred.sum(dim=dims_to_sum)

                    az.plot_hdi(
                        x=result.X_test[self.date_column],
                        y=pred.sel(
                            {
                                self.date_column: slice(
                                    result.X_test[self.date_column].min(),
                                    result.X_test[self.date_column].max(),
                                )
                            }
                        ),
                        color="C1",
                        smooth=False,
                        hdi_prob=hdi_prob,
                        fill_kwargs={
                            "alpha": 0.4,
                            "label": f"{hdi_prob:.0%} HDI (test)",
                        },
                        ax=ax,
                    )

                # Plot observed values for combined train+test date range
                n_combined = len(result.X_train) + len(result.X_test)
                combined_dates = pd.concat(
                    [result.X_train[self.date_column], result.X_test[self.date_column]]
                ).sort_values()
                ax.plot(
                    combined_dates,
                    y.iloc[:n_combined],
                    marker="o",
                    markersize=4,
                    color="black",
                    label="observed",
                )
                ax.axvline(
                    result.X_test[self.date_column].iloc[0], color="C2", linestyle="--"
                )

                if i == 0:
                    ax.legend(loc="upper right")

        if dims is None:
            fig, axes = plt.subplots(
                nrows=n_iterations,
                ncols=1,
                figsize=(9, 6),
                sharex=True,
                sharey=True,
                layout="constrained",
            )
            # normalize axes for single-iteration case
            if n_iterations == 1:
                axes = [axes]
            else:
                axes = axes.ravel()

            _plot_for_results(fig, axes, results)
            axes[-1].set(xlim=(X[self.date_column].iloc[self.n_init - 9], None))
            fig.suptitle(
                "Posterior Predictive Check", fontsize=18, fontweight="bold", y=1.02
            )
            plt.show()
            return

        # dims provided: iterate over dims defined in the model if no coords supplied
        for dim_name, coord_values in dims.items():
            # If no coords provided for this dim, try to pull from model coords
            if not coord_values:
                try:
                    # derive coords from the idata stored in the CV results
                    coord_values = list(
                        results[0].idata.posterior.coords[dim_name].values
                    )
                except Exception:
                    raise ValueError(
                        f"Dim '{dim_name}' not found in posterior/model coords and no coord values were provided."
                    )

            for coord in coord_values:
                fig, axes = plt.subplots(
                    nrows=n_iterations,
                    ncols=1,
                    figsize=(9, 6),
                    sharex=True,
                    sharey=True,
                    layout="constrained",
                )
                if n_iterations == 1:
                    axes = [axes]
                else:
                    axes = axes.ravel()

                _plot_for_results(fig, axes, results, dim_name=dim_name, coord=coord)
                axes[-1].set(xlim=(X[self.date_column].iloc[self.n_init - 9], None))
                fig.suptitle(
                    f"Posterior Predictive Check | {dim_name}={coord}",
                    fontsize=18,
                    fontweight="bold",
                    y=1.02,
                )
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
        crps_results_train = [
            crps(
                y_true=result.y_train.to_numpy(),
                y_pred=result.idata.posterior_predictive["y_original_scale"]
                .squeeze()
                .stack(sample=("chain", "draw"))
                .transpose("sample", self.date_column)
                .to_numpy()[
                    :, : result.y_train.shape[0]
                ],  # Ensure matching number of observations
            )
            for result in results
        ]

        crps_results_test = []
        for result in results:
            y_pred_processed = (
                result.idata.posterior_predictive["y_original_scale"]
                .sel(
                    {
                        self.date_column: slice(
                            result.X_test[self.date_column].min(), None
                        )
                    }
                )
                .squeeze()
                .stack(sample=("chain", "draw"))
                .transpose("sample", self.date_column)
                .to_numpy()[:, : result.y_test.shape[0]]
            )  # Ensure matching number of observations

            # Ensure we have a 2D array
            if y_pred_processed.ndim != 2:
                raise ValueError(
                    f"Expected 2D array, got {y_pred_processed.ndim}D with shape {y_pred_processed.shape}"
                )

            crps_results_test.append(
                crps(
                    y_true=result.y_test.to_numpy(),
                    y_pred=y_pred_processed,
                )
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
