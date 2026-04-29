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
import arviz_plots as azp
import numpy as np
import pandas as pd
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.metrics import crps as _crps_score  # noqa: F401
from pymc_marketing.mmm.plotting._helpers import (
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
            if len(X_test) > 0
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
        for d, y in zip(X_train["date"].values, np.asarray(y_train), strict=True):
            date_to_y[d] = float(y)
        if len(X_test) > 0:
            for d, y in zip(X_test["date"].values, np.asarray(y_test), strict=True):
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

    def predictions(
        self,
        cv_data: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        hdi_prob: float = 0.94,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        hdi_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Posterior predictive HDI bands per CV fold.

        For each fold: blue HDI band over train dates, orange HDI band over test
        dates, black observed line, and a green dashed vertical boundary at the
        train/test split.

        Parameters
        ----------
        cv_data : az.InferenceData or None
            Override the stored ``self.cv_data`` for this call only.
            ``_validate_cv_results`` is re-run on the override.
        dims : dict or None
            Filter coordinate values before rendering
            (e.g. ``{"geo": ["North"]}``).
        hdi_prob : float
            HDI probability mass (default 0.94).
        figsize : tuple or None
            Figure size in inches; injected into ``figure_kwargs``.
        backend : str or None
            PlotCollection backend (``"matplotlib"`` / ``"plotly"`` / ``"bokeh"``).
            Non-matplotlib requires ``return_as_pc=True``.
        return_as_pc : bool
            Return the raw ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
        hdi_kwargs : dict or None
            Extra kwargs forwarded to ``azp.visuals.fill_between_y``.
        **pc_kwargs
            Forwarded to ``PlotCollection.grid()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        data = cv_data if cv_data is not None else self.cv_data
        if cv_data is not None:
            _validate_cv_results(data)

        if not hasattr(data, "cv_metadata") or "metadata" not in data.cv_metadata:
            raise ValueError(
                "cv_data must have a cv_metadata group containing a 'metadata' variable."
            )
        if (
            not hasattr(data, "posterior_predictive")
            or "y_original_scale" not in data.posterior_predictive
        ):
            raise ValueError(
                "cv_data must have posterior_predictive['y_original_scale']."
            )

        pp = data.posterior_predictive["y_original_scale"]
        y_train_da, y_test_da, y_obs_da, train_end_da = _build_predictions_arrays(
            data, pp
        )

        if dims:
            y_train_da = _select_dims(y_train_da, dims)
            y_test_da = _select_dims(y_test_da, dims)
            y_obs_da = _select_dims(y_obs_da, dims)

        standard_dims = {"cv", "chain", "draw", "date"}
        custom_dims = [d for d in y_train_da.dims if d not in standard_dims]

        split_ds = xr.Dataset({"train": y_train_da, "test": y_test_da})

        pc_kwargs = _process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)
        rows = pc_kwargs.pop("rows", [*custom_dims, "cv"])
        cols = pc_kwargs.pop("cols", [])

        pc = PlotCollection.grid(
            split_ds,
            rows=rows,
            cols=cols,
            aes={"color": ["__variable__"]},
            backend=backend,
            **pc_kwargs,
        )

        hdi_ds = split_ds.azstats.hdi(hdi_prob)
        date_da = split_ds["train"].coords["date"]

        pc.map(
            azp.visuals.fill_between_y,
            x=date_da,
            y_bottom=hdi_ds.sel(ci_bound="lower"),
            y_top=hdi_ds.sel(ci_bound="upper"),
            alpha=0.3,
            **(hdi_kwargs or {}),
        )

        pc.map(azp.visuals.line_xy, x=date_da, y=y_obs_da, color="black", linewidth=1.5)

        azp.add_lines(
            pc,
            train_end_da,
            orientation="vertical",
            visuals={
                "ref_line": {
                    "color": "green",
                    "linestyle": "--",
                    "linewidth": 2,
                    "alpha": 0.9,
                }
            },
        )

        pc.add_legend("__variable__")

        return _extract_matplotlib_result(pc, return_as_pc)

    def param_stability(
        self,
        cv_data: az.InferenceData | None = None,
        var_names: list[str] | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        figure_kwargs: dict[str, Any] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Forest plot comparing parameter posteriors across all CV folds.

        Parameters
        ----------
        cv_data : az.InferenceData or None
            Override the stored ``self.cv_data`` for this call only.
        var_names : list[str] or None
            Variables to include (passed directly to ``azp.plot_forest``).
        dims : dict or None
            Filter coordinate values before plotting
            (e.g. ``{"channel": ["tv"]}``).
        figsize : tuple or None
            Figure size in inches; takes precedence over ``figure_kwargs["figsize"]``.
        figure_kwargs : dict or None
            Extra kwargs for the figure constructor; merged with defaults.
        backend : str or None
            PlotCollection backend.
        return_as_pc : bool
            Return the raw ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
        **pc_kwargs
            Forwarded to ``azp.plot_forest()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        data = cv_data if cv_data is not None else self.cv_data
        if cv_data is not None:
            _validate_cv_results(data)

        if not hasattr(data, "posterior"):
            raise ValueError("cv_data has no 'posterior' group.")
        if "cv" not in data.posterior.coords:
            raise ValueError(
                "No 'cv' coordinate found in cv_data.posterior. "
                "Ensure the InferenceData was produced by TimeSliceCrossValidator.run()."
            )

        posterior = data.posterior
        if dims:
            posterior = _select_dims(posterior, dims)

        # Move labelled dims to the end so the forest plot reads naturally.
        # Guard: only include dims that actually exist after optional filtering.
        dims_to_end = [d for d in ("channel", "cv") if d in posterior.dims]
        if dims_to_end:
            posterior = posterior.transpose(..., *dims_to_end)

        idata_for_plot = az.InferenceData(posterior=posterior)

        fig_kw: dict[str, Any] = {
            "width_ratios": [1, 2],
            "layout": "none",
            **(figure_kwargs or {}),
        }
        if figsize is not None:
            fig_kw["figsize"] = figsize

        pc = azp.plot_forest(
            idata_for_plot.to_datatree(),
            var_names=var_names,
            aes={"color": ["cv"]},
            figure_kwargs=fig_kw,
            combined=True,
            shade_label="channel",
            backend=backend,
            **pc_kwargs,
        )
        return _extract_matplotlib_result(pc, return_as_pc)

    def crps(self, *args, **kwargs):
        """Plot CRPS scores per fold for out-of-sample evaluation."""
        raise NotImplementedError
