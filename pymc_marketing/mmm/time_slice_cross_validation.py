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
"""Time-slice cross-validation utilities for PyMC-Marketing MMM.

This module provides the TimeSliceCrossValidator which can run rolling
time-slice cross-validation for media-mix models built with the library.
The validator does not retain a fitted MMM instance; models may be
constructed per-fold from a YAML configuration or supplied to ``run()``.
"""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml
from pymc_marketing.mmm.plot import MMMPlotSuite
from pymc_marketing.mmm.types import MMMBuilder


@dataclass
class TimeSliceCrossValidationResult:
    """Container for the results of one time-slice CV step.

    Attributes
    ----------
    X_train : pd.DataFrame
        Feature matrix used for training in this fold.
    y_train : pd.Series
        Target variable used for training in this fold.
    X_test : pd.DataFrame
        Feature matrix used for testing in this fold.
    y_test : pd.Series
        Target variable used for testing in this fold.
    idata : az.InferenceData
        ArviZ InferenceData object containing posterior samples and predictions
        from the fitted model for this fold.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    idata: az.InferenceData


class TimeSliceCrossValidator:
    """Time-Slice Cross Validator for Media Mix Models (MMM).

    Provides a scikit-learn-style API for performing rolling time-slice
    cross-validation on media mix models. This is useful for evaluating
    model stability and out-of-sample prediction performance.

    Parameters
    ----------
    n_init : int
        Number of initial time periods to use for the first training fold.
        Must be a positive integer.
    forecast_horizon : int
        Number of time periods to forecast in each fold.
        Must be a positive integer.
    date_column : str
        Name of the column in X containing date values.
    step_size : int, optional
        Number of time periods to step forward between consecutive folds.
        Default is 1. Must be a positive integer.
    sampler_config : dict, optional
        Configuration dictionary for the PyMC sampler. Can include keys like
        'tune', 'draws', 'chains', 'random_seed', 'target_accept', etc.
        Can be overridden per-run via the ``run()`` method.

    Attributes
    ----------
    n_init : int
        Number of initial training periods.
    forecast_horizon : int
        Number of forecast periods per fold.
    date_column : str
        Name of the date column.
    step_size : int
        Step size between folds.
    sampler_config : dict or None
        Sampler configuration dictionary.

    See Also
    --------
    pymc_marketing.mmm.MMM : The Media Mix Model class.
    pymc_marketing.mmm.plot.MMMPlotSuite : Plotting utilities for CV results.

    Notes
    -----
    This validator does not retain a fitted MMM instance; models are
    constructed per-fold from a YAML configuration or supplied to ``run()``.

    Each fold stores its full InferenceData, which can consume significant
    memory for large models with many folds.

    Examples
    --------
    Basic usage with a YAML configuration:

    >>> cv = TimeSliceCrossValidator(
    ...     n_init=100,
    ...     forecast_horizon=10,
    ...     date_column="date",
    ...     step_size=5,
    ... )
    >>> combined_idata = cv.run(X, y, yaml_path="model_config.yml")

    With custom sampler configuration:

    >>> cv = TimeSliceCrossValidator(
    ...     n_init=158,
    ...     forecast_horizon=10,
    ...     date_column="date",
    ...     step_size=50,
    ...     sampler_config={
    ...         "tune": 500,
    ...         "draws": 200,
    ...         "chains": 4,
    ...         "random_seed": 123,
    ...     },
    ... )
    >>> combined_idata = cv.run(X, y, mmm=mmm_builder)
    """

    def __init__(
        self,
        n_init: int,
        forecast_horizon: int,
        date_column: str,
        step_size: int = 1,
        sampler_config: dict[str, Any] | None = None,
    ) -> None:
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
        """Build a cv_metadata Dataset that stores per-fold metadata.

        The dataset stores per-fold metadata as Python objects (DataFrames/Series)
        under a single DataArray named 'metadata' indexed by the same 'cv' labels.
        Consumers can access fold metadata via ``cv_idata.cv_metadata.metadata.sel(cv=...)``.

        Parameters
        ----------
        cv_coord : pd.Index
            The coordinate index for the 'cv' dimension.

        Returns
        -------
        xr.Dataset
            Dataset containing per-fold metadata.
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

    def _combine_idata(
        self,
        results: list[TimeSliceCrossValidationResult],
        model_names: list[str],
    ) -> az.InferenceData:
        """Combine InferenceData objects from multiple CV results.

        Parameters
        ----------
        results : list of TimeSliceCrossValidationResult
            List of CV results from each fold.
        model_names : list of str
            Names for each CV fold.

        Returns
        -------
        az.InferenceData
            Combined InferenceData with folds concatenated along 'cv' coordinate.

        Raises
        ------
        ValueError
            If no InferenceData objects were produced during CV.
        """
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

    def _fit_mmm(
        self,
        mmm: Any,
        X: pd.DataFrame,
        y: pd.Series,
        sampler_config: dict[str, Any] | None = None,
    ) -> Any:
        """Fit the MMM model.

        Parameters
        ----------
        mmm : object
            MMM instance to fit.
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target variable.
        sampler_config : dict, optional
            Sampler configuration to apply before fitting.

        Returns
        -------
        object
            The fitted MMM instance.
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
        self,
        mmm: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sampler_config: dict[str, Any] | None = None,
    ) -> TimeSliceCrossValidationResult:
        """Run one CV step and return results.

        Parameters
        ----------
        mmm : object
            MMM instance to fit.
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training target variable.
        X_test : pd.DataFrame
            Test feature matrix.
        y_test : pd.Series
            Test target variable.
        sampler_config : dict, optional
            Sampler configuration to apply before fitting.

        Returns
        -------
        TimeSliceCrossValidationResult
            Results container with fitted model data and predictions.
        """
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

    def get_n_splits(self, X: pd.DataFrame, y: pd.Series | None = None) -> int:
        """Return the number of possible rolling splits.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix containing the date column.
        y : pd.Series, optional
            Target variable. Not used but included for scikit-learn API
            compatibility.

        Returns
        -------
        int
            Number of cross-validation splits that can be generated.
        """
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

    def split(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices for each time-slice split.

        This implementation selects rows by date masks so that all coordinate
        levels (e.g., multiple geos) for the selected date ranges are included
        in each fold. It returns integer positions suitable for use with
        ``DataFrame.iloc``.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix containing the date column.
        y : pd.Series, optional
            Target variable. Not used but included for scikit-learn API
            compatibility.

        Yields
        ------
        train_idx : np.ndarray
            Integer indices for training rows in this fold.
        test_idx : np.ndarray
            Integer indices for test rows in this fold.

        Raises
        ------
        ValueError
            If no splits are possible with the given parameters.

        Examples
        --------
        >>> cv = TimeSliceCrossValidator(
        ...     n_init=10, forecast_horizon=5, date_column="date"
        ... )
        >>> for train_idx, test_idx in cv.split(X, y):
        ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        ...     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
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

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sampler_config: dict[str, Any] | None = None,
        yaml_path: str | None = None,
        mmm: MMMBuilder | None = None,
        model_names: list[str] | None = None,
    ) -> az.InferenceData:
        """Run the complete time-slice cross-validation loop.

        Executes cross-validation by iterating through all folds, fitting a model
        for each training set, and generating predictions on the combined
        train+test data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix containing the date column and predictor variables.
        y : pd.Series
            Target variable.
        sampler_config : dict, optional
            Sampler configuration to override the validator-level configuration
            for all folds in this run. If provided, takes precedence over the
            configuration passed at construction time.
        yaml_path : str, optional
            Path to a YAML configuration file for building the MMM model per fold.
            Mutually exclusive with ``mmm``.
        mmm : object, optional
            An object with a ``build_model(X, y)`` method that returns a fitted
            MMM instance. Mutually exclusive with ``yaml_path``.
        model_names : list of str, optional
            Names to assign to each CV fold in the combined InferenceData.
            If provided, length must match the number of splits. If not provided,
            names are generated from each model's ``_model_name`` attribute or
            as ``'Iteration {i}'``.

        Returns
        -------
        arviz.InferenceData
            Combined InferenceData where each fold is concatenated along a new
            coordinate named 'cv'. Includes a 'cv_metadata' group with per-fold
            train/test data.

        Raises
        ------
        ValueError
            If neither ``yaml_path`` nor ``mmm`` is provided.
            If ``model_names`` length doesn't match the number of splits.
            If no InferenceData objects are produced during CV.

        See Also
        --------
        split : Generate train/test indices for cross-validation.
        get_n_splits : Return the number of splits.

        Notes
        -----
        Per-fold results are also stored in ``self._cv_results`` after calling
        this method.

        Examples
        --------
        Using a YAML configuration:

        >>> cv = TimeSliceCrossValidator(
        ...     n_init=100, forecast_horizon=10, date_column="date"
        ... )
        >>> combined_idata = cv.run(X, y, yaml_path="model_config.yml")

        Using a model builder object:

        >>> cv = TimeSliceCrossValidator(
        ...     n_init=100, forecast_horizon=10, date_column="date"
        ... )
        >>> combined_idata = cv.run(X, y, mmm=mmm_builder)
        """
        # Upfront validation of model_names length
        n_splits = self.get_n_splits(X, y)
        if model_names is not None and len(model_names) != n_splits:
            raise ValueError(
                f"`model_names` length ({len(model_names)}) must match the number "
                f"of CV splits ({n_splits})."
            )

        results: list[TimeSliceCrossValidationResult] = []
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
                # Length was validated upfront, so direct indexing is safe
                fold_name = user_model_names[_i]
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
