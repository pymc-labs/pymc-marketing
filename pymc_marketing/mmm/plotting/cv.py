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
"""CV plotting namespace — MMMCVPlotSuite for TimeSliceCrossValidator results."""

from __future__ import annotations

from typing import Any

import arviz as az
import arviz_plots as azp  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from arviz_plots import PlotCollection  # noqa: F401
from matplotlib.axes import Axes  # noqa: F401
from matplotlib.figure import Figure  # noqa: F401
from numpy.typing import NDArray  # noqa: F401

from pymc_marketing.metrics import crps as _crps_score  # noqa: F401
from pymc_marketing.mmm.plotting._helpers import (  # noqa: F401
    _extract_matplotlib_result,
    _process_plot_params,
    _select_dims,
)

# ── Shared base validation ────────────────────────────────────────────────────


def _validate_cv_results(cv_data: az.InferenceData) -> None:
    """Raise if cv_data is not a valid CV InferenceData.

    Minimum required: correct type + cv_metadata group present.
    Method-specific checks (e.g. posterior_predictive contents) are
    performed inside each method.
    """
    if not isinstance(cv_data, az.InferenceData):
        raise TypeError(
            f"cv_data must be az.InferenceData, got {type(cv_data).__name__}."
        )
    if not hasattr(cv_data, "cv_metadata"):
        raise ValueError(
            "cv_data must have a 'cv_metadata' group. "
            "Ensure TimeSliceCrossValidator.run() has been called and the "
            "resulting InferenceData is passed here."
        )


def _extract_cv_labels(cv_data: az.InferenceData) -> list[str]:
    """Return the list of CV fold labels from cv_metadata coords."""
    return list(cv_data.cv_metadata.coords["cv"].values)


def _read_fold_meta(
    cv_data: az.InferenceData, cv_label: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_train, y_train, X_test, y_test) for a given fold label."""
    meta = cv_data.cv_metadata["metadata"].sel(cv=cv_label).values.item()
    return meta["X_train"], meta["y_train"], meta["X_test"], meta["y_test"]


def _build_predictions_arrays(
    cv_data: az.InferenceData,
    pp: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Build stacked train/test/observed/train-end arrays across all CV folds.

    Parameters
    ----------
    cv_data : az.InferenceData
        Full CV InferenceData (already validated by the caller).
    pp : xr.DataArray
        ``posterior_predictive["y_original_scale"]`` with dims
        ``(cv, chain, draw, date, ...)``.

    Returns
    -------
    y_train_da : xr.DataArray  — (cv, chain, draw, date, ...) NaN outside train window
    y_test_da  : xr.DataArray  — (cv, chain, draw, date, ...) NaN outside test window
    y_obs_da   : xr.DataArray  — (cv, date) observed actuals aligned to full date coord
    train_end_da : xr.DataArray — (cv,) last training date per fold
    """
    cv_labels = _extract_cv_labels(cv_data)
    full_dates = pp.coords["date"].values

    y_train_list: list[xr.DataArray] = []
    y_test_list: list[xr.DataArray] = []
    y_obs_list: list[xr.DataArray] = []
    train_end_list: list[Any] = []

    for lbl in cv_labels:
        X_train, y_train, X_test, y_test = _read_fold_meta(cv_data, lbl)

        train_dates = pd.DatetimeIndex(X_train["date"].values)
        test_dates = (
            pd.DatetimeIndex(X_test["date"].values)
            if X_test is not None and len(X_test) > 0
            else pd.DatetimeIndex([])
        )

        train_mask = xr.DataArray(
            np.isin(full_dates, train_dates.values),
            dims=["date"],
            coords={"date": full_dates},
        )
        test_mask = xr.DataArray(
            np.isin(full_dates, test_dates.values),
            dims=["date"],
            coords={"date": full_dates},
        )

        pp_fold = pp.sel(cv=lbl)
        y_train_list.append(pp_fold.where(train_mask))
        y_test_list.append(pp_fold.where(test_mask))

        date_to_y: dict[Any, float] = {}
        for d, y in zip(X_train["date"].values, np.asarray(y_train), strict=False):
            date_to_y[d] = float(y)
        if X_test is not None and len(X_test) > 0:
            for d, y in zip(X_test["date"].values, np.asarray(y_test), strict=False):
                date_to_y[d] = float(y)
        y_obs_arr = np.array([date_to_y.get(d, np.nan) for d in full_dates])
        y_obs_list.append(
            xr.DataArray(y_obs_arr, dims=["date"], coords={"date": full_dates})
        )
        train_end_list.append(train_dates.max())

    cv_coord = xr.DataArray(cv_labels, dims=["cv"], name="cv")
    y_train_da = xr.concat(y_train_list, dim=cv_coord).assign_coords(cv=cv_labels)
    y_test_da = xr.concat(y_test_list, dim=cv_coord).assign_coords(cv=cv_labels)
    y_obs_da = xr.concat(y_obs_list, dim=cv_coord).assign_coords(cv=cv_labels)
    train_end_da = xr.DataArray(train_end_list, dims=["cv"], coords={"cv": cv_labels})
    return y_train_da, y_test_da, y_obs_da, train_end_da


# ── Main class ────────────────────────────────────────────────────────────────


class MMMCVPlotSuite:
    """PlotCollection-native plots for TimeSliceCrossValidator results.

    Parameters
    ----------
    cv_data : az.InferenceData
        Combined InferenceData produced by ``TimeSliceCrossValidator.run()``.
        Must contain a ``cv_metadata`` group with per-fold metadata.
    """

    def __init__(self, cv_data: az.InferenceData) -> None:
        _validate_cv_results(cv_data)
        self.cv_data = cv_data

    def predictions(self, *args, **kwargs):
        """Plot posterior-predictive train/test predictions across CV folds."""
        raise NotImplementedError

    def param_stability(self, *args, **kwargs):
        """Plot parameter stability (forest plot) across CV folds."""
        raise NotImplementedError

    def crps(self, *args, **kwargs):
        """Plot CRPS scores per fold for out-of-sample evaluation."""
        raise NotImplementedError
