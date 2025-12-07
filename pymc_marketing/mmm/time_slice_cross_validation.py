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
import xarray as xr
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

    def _create_metadata(self, cv_coord: pd.Index) -> xr.Dataset:
        """
        Build a cv_metadata Dataset that stores per-fold metadata.

        The dataset stores per-fold metadata as Python objects (DataFrames/Series)
        under a single DataArray named 'metadata' indexed by the same 'cv' labels.
        Consumers can access fold metadata via `cv_idata.cv_metadata.metadata.sel(cv=...)`.

        cv_coord: pd.Index
            The coordinate index for the 'cv' dimension.

        """
        metadata_list = []
        for r in self._cv_results:
            meta = {
                "X_train": getattr(r, "X_train", None),
                "y_train": getattr(r, "y_train", None),
                "X_test": getattr(r, "X_test", None),
                "y_test": getattr(r, "y_test", None),
            }
            metadata_list.append(meta)

        # Create an object-dtype array so xarray can hold arbitrary Python objects
        meta_arr = np.empty((len(metadata_list),), dtype=object)
        for i, m in enumerate(metadata_list):
            meta_arr[i] = m

        ds_meta = xr.Dataset(
            {"metadata": ("cv", meta_arr)},
            coords={"cv": cv_coord},
        )

        # persist on instance for convenience
        self.cv_metadata = metadata_list

        return ds_meta

    def _combine_idata(self, results, model_names: list[str]) -> az.InferenceData:
        """Combine InferenceData objects from multiple CV results."""
        cv_idata: az.InferenceData | None = None
        if results:
            # try to discover available groups from the first idata
            first_idata = results[0].idata
            try:
                groups = list(first_idata._groups)
            except Exception:
                # fallback to common groups
                groups = [
                    "posterior",
                    "posterior_predictive",
                    "observed_data",
                    "sample_stats",
                    "prior",
                ]

            combined_kwargs: dict = {}
            # Ensure we pass a concrete list[str] into pd.Index to satisfy type checkers
            cv_coord = pd.Index([str(n) for n in model_names], name="cv")

            for group in groups:
                # collect available datasets for this group
                ds_list = []
                for r in results:
                    if r.idata is None:
                        continue
                    try:
                        ds = r.idata[group]
                    except Exception:
                        ds = None
                    if ds is not None:
                        ds_list.append(ds)

                if not ds_list:
                    continue

                # concatenate along new cv coordinate, making sure each dataset
                # gets the cv coordinate labels
                try:
                    combined_ds = xr.concat(ds_list, dim=cv_coord)
                except Exception:
                    # if concat fails, try to align then concat without coords
                    combined_ds = xr.concat(
                        [
                            d.assign_coords({"cv": [n]})
                            for d, n in zip(ds_list, model_names, strict=False)
                        ],
                        dim="cv",
                    )

                combined_kwargs[group] = combined_ds

            # Build a cv_metadata Dataset that stores per-fold metadata
            ds_meta = self._create_metadata(cv_coord)
            combined_kwargs["cv_metadata"] = ds_meta

            if combined_kwargs:
                cv_idata = az.InferenceData(**combined_kwargs)
                # persist for plot helpers
                self.cv_idata = cv_idata
        # Also expose the last fold's idata (if any) for compatibility with MMMPlotSuite
        if results:
            last = results[-1]
            if hasattr(last, "idata") and last.idata is not None:
                self.idata = last.idata
        # Always return the combined arviz.InferenceData. If none could be
        # constructed (e.g. folds did not produce idata), raise an error so the
        # caller knows something went wrong.
        if cv_idata is None:
            raise ValueError(
                "No InferenceData objects were produced during CV; ensure models produce idata."
            )
        return cv_idata

    def _fit_mmm(self, mmm, X, y, sampler_config: dict | None = None):
        """Fit the MMM.

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
        return mmm

    def _time_slice_step(
        self, mmm, X_train, y_train, X_test, y_test, sampler_config: dict | None = None
    ):
        """Run one CV step and return results."""
        # Fit the model for this fold. sampler_config can override the validator-level config.
        mmm = self._fit_mmm(mmm, X_train, y_train, sampler_config=sampler_config)

        # Combine train and test data for posterior predictions
        X_combined = pd.concat([X_train, X_test], ignore_index=True)

        # Remove existing posterior_predictive groups if they exist to avoid conflicts
        # when extending idata with new predictions
        if mmm.idata is not None:
            if "posterior_predictive" in mmm.idata.groups():
                del mmm.idata.posterior_predictive
            if "posterior_predictive_constant_data" in mmm.idata.groups():
                del mmm.idata.posterior_predictive_constant_data

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
        model_names: list[str] | None = None,
    ) -> az.InferenceData:
        """Run the complete time-slice CV loop.

        If `yaml_path` is provided, the validator will rebuild the MMM from the
        YAML for each fold using the training data before calling `_fit_mmm`.

        sampler_config: Optional dict to override the validator-level sampler configuration
                        for all folds in this run. If provided here it takes precedence
                        over the configuration passed at construction time.

        model_names: Optional list of names to assign to each CV fold's model in the
                     combined InferenceData. If provided its length must match the
                     number of splits. If not provided, default names will be generated
                     from each fold's model's `_model_name` attribute (if available)
                     or a generic `Iteration {i}` name.

        This function always returns a combined `arviz.InferenceData` where each
        CV fold is concatenated along a new coordinate named `cv`. If no
        InferenceData objects are present in the folds, a ValueError is raised.
        """
        results = []
        # Preserve the user-provided `model_names` parameter separately so we
        # don't shadow it with the accumulator used to collect generated names.
        user_model_names = model_names
        model_name_labels: list[str] = []
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

            # determine name for this fold
            if user_model_names is not None:
                # user supplied explicit model names: validate length implicitly
                try:
                    fold_name = user_model_names[_i]
                except IndexError:
                    raise ValueError(
                        "`model_names` was provided but its length is shorter than the number of CV splits."
                    )
            else:
                base_name = (
                    getattr(fold_mmm, "_model_name", None)
                    or getattr(mmm, "_model_name", None)
                    or "Iteration"
                )
                # produce human-friendly default
                if base_name == "Iteration":
                    fold_name = f"Iteration {_i}"
                else:
                    fold_name = f"{base_name}_{_i}"
            model_name_labels.append(fold_name)

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
        # Build a combined InferenceData. We combine each fold's
        # datasets along a new coordinate named 'cv' where each label is the
        # fold name determined above.
        cv_idata = self._combine_idata(results, model_name_labels)

        return cv_idata
