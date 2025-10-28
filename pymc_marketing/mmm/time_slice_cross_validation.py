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
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml
from pymc_marketing.mmm.plot import MMMPlotSuite


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

    @property
    def plot(self) -> MMMPlotSuite:
        """Use the MMMPlotSuite to plot the results."""
        self._validate_model_was_built()
        self._validate_idata_exists()
        return MMMPlotSuite(idata=self.idata)

    def _validate_model_was_built(self) -> None:
        """Validate that at least one CV run has produced results.

        Ensures `self._cv_results` exists and is non-empty. If an
        InferenceData is present on the last result, expose it as
        `self.idata` for compatibility with the MMMPlotSuite API.
        """
        if not hasattr(self, "_cv_results") or not self._cv_results:
            raise ValueError(
                "No CV results available. Run `TimeSliceCrossValidator.run(...)` first."
            )
        last_result = self._cv_results[-1]
        if hasattr(last_result, "idata") and last_result.idata is not None:
            # make idata accessible for plotting helpers
            self.idata = last_result.idata

    def _validate_idata_exists(self) -> None:
        """Validate that `self.idata` is present and not None."""
        if not hasattr(self, "idata") or self.idata is None:
            raise ValueError(
                "No InferenceData available on the validator. Run `TimeSliceCrossValidator.run(...)` first."
            )

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
        # Persist results on the instance so plotting helpers can access them
        self._cv_results = results
        # Also expose the last fold's idata (if any) for compatibility with MMMPlotSuite
        if results:
            last = results[-1]
            if hasattr(last, "idata") and last.idata is not None:
                self.idata = last.idata
        return results
