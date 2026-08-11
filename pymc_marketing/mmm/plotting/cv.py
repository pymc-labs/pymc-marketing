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

import itertools
from typing import Any

import arviz_plots as azp
import numpy as np
import xarray as xr
from arviz_base.labels import DimCoordLabeller, NoVarLabeller, mix_labellers
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.mmm.plotting._helpers import (
    _extract_matplotlib_result,
    _process_plot_params,
    _select_dims,
)
from pymc_marketing.mmm.summary.cv import (
    _build_predictions_arrays,
    _crps_for_split,
    _extract_cv_labels,
    _read_fold_meta,
    _validate_cv_results,
)


class MMMCVPlotSuite:
    """PlotCollection-native plots for TimeSliceCrossValidator results.

    Parameters
    ----------
    cv_data : xr.DataTree
        Combined DataTree produced by ``TimeSliceCrossValidator.run()``.
        Must contain a ``cv_metadata`` group with per-fold metadata.
    """

    def __init__(self, cv_data: xr.DataTree) -> None:
        _validate_cv_results(cv_data)
        self.cv_data = cv_data

    def predictions(
        self,
        cv_data: xr.DataTree | None = None,
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
        cv_data : xr.DataTree or None
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

        hdi_ds = split_ds.azstats.hdi(prob=hdi_prob)
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

        _vline_kw: dict[str, Any] = {
            "color": "green",
            "linestyle": "--",
            "linewidth": 2,
            "alpha": 0.9,
        }
        _plot_da = pc.viz["plot"]
        _cv_pos = _plot_da.dims.index("cv")
        for _idx in np.ndindex(*_plot_da.shape):
            _ax = _plot_da.values[_idx]
            _cv_lbl = str(_plot_da.coords["cv"].values[_idx[_cv_pos]])
            _ax.axvline(x=float(train_end_da.sel(cv=_cv_lbl).item()), **_vline_kw)

        pc.add_legend("__variable__")
        pc.map(
            azp.visuals.labelled_title,
            subset_info=True,
            labeller=mix_labellers((NoVarLabeller, DimCoordLabeller))(),
            ignore_aes={"color"},
        )

        return _extract_matplotlib_result(pc, return_as_pc)

    def param_stability(
        self,
        cv_data: xr.DataTree | None = None,
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
        cv_data : xr.DataTree or None
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
        if "cv" not in data.posterior.dims:
            raise ValueError(
                "No 'cv' coordinate found in cv_data.posterior. "
                "Ensure the DataTree was produced by TimeSliceCrossValidator.run()."
            )

        posterior = data["/posterior"].dataset
        if dims:
            posterior = _select_dims(posterior, dims)

        # Move labelled dims to the end so the forest plot reads naturally.
        # Guard: only include dims that actually exist after optional filtering.
        dims_to_end = [d for d in ("channel", "cv") if d in posterior.dims]
        if dims_to_end:
            posterior = posterior.transpose(..., *dims_to_end)

        idata_for_plot = xr.DataTree.from_dict({"/posterior": posterior})

        fig_kw: dict[str, Any] = {
            "width_ratios": [1, 2],
            "layout": "none",
            **(figure_kwargs or {}),
        }
        if figsize is not None:
            fig_kw["figsize"] = figsize

        pc = azp.plot_forest(
            idata_for_plot,
            var_names=var_names,
            aes={"color": ["cv"]},
            figure_kwargs=fig_kw,
            combined=True,
            shade_label="channel",
            backend=backend,
            **pc_kwargs,
        )
        return _extract_matplotlib_result(pc, return_as_pc)

    def crps(
        self,
        cv_data: xr.DataTree | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        line_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Line chart of mean CRPS per fold for train and test splits.

        Renders an n×2 grid: left column = train CRPS, right column = test CRPS,
        one row per Cartesian combination of extra dimensions in
        ``y_original_scale`` (e.g. one row per geo). When no extra dimensions
        are present the result is a 1×2 grid.

        Parameters
        ----------
        cv_data : xr.DataTree or None
            Override the stored ``self.cv_data`` for this call only.
        dims : dict or None
            Filters which coordinate values of extra dimensions appear as rows
            (e.g. ``{"geo": ["geo_b"]}`` → only geo_b row).
            Non-extra-dim keys are silently ignored.
        figsize : tuple or None
            Figure size in inches.
        backend : str or None
            PlotCollection backend.
        return_as_pc : bool
            Return the raw ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
        line_kwargs : dict or None
            Extra kwargs forwarded to ``azp.visuals.line_xy``.
        **pc_kwargs
            Forwarded to ``PlotCollection.grid()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        data = cv_data if cv_data is not None else self.cv_data
        if cv_data is not None:
            _validate_cv_results(data)

        if not hasattr(data, "cv_metadata"):
            raise ValueError("cv_data must have a 'cv_metadata' group.")
        if (
            not hasattr(data, "posterior_predictive")
            or "y_original_scale" not in data.posterior_predictive
        ):
            raise ValueError(
                "cv_data must have posterior_predictive['y_original_scale']."
            )

        pp = data.posterior_predictive["y_original_scale"]
        cv_labels = _extract_cv_labels(data)

        base_dims = {"cv", "chain", "draw", "date"}
        extra_dims = [d for d in pp.dims if d not in base_dims]

        combo_coords: dict[str, list[Any]] = {
            d: (list(dims[d]) if dims and d in dims else list(pp.coords[d].values))
            for d in extra_dims
        }
        combos = list(itertools.product(*combo_coords.values()))
        combo_shape = [len(v) for v in combo_coords.values()]
        data_arr = np.full((2, *combo_shape, len(cv_labels)), np.nan)

        for flat_idx, combo in enumerate(combos):
            dim_indexers = dict(zip(extra_dims, combo, strict=True))
            multi_idx = (
                tuple(np.unravel_index(flat_idx, combo_shape)) if combo_shape else ()
            )
            for fold_idx, lbl in enumerate(cv_labels):
                X_train, y_train, X_test, y_test = _read_fold_meta(data, lbl)
                data_arr[(0, *multi_idx, fold_idx)] = _crps_for_split(  # type: ignore[index]
                    data, lbl, X_train, y_train, dim_indexers
                )
                data_arr[(1, *multi_idx, fold_idx)] = _crps_for_split(  # type: ignore[index]
                    data, lbl, X_test, y_test, dim_indexers
                )

        coords: dict[str, Any] = {
            "split": ["train", "test"],
            **combo_coords,
            "cv": cv_labels,
        }
        crps_da = xr.DataArray(
            data_arr, dims=["split", *extra_dims, "cv"], coords=coords
        )
        crps_ds = crps_da.to_dataset(name="crps")

        pc_kwargs = _process_plot_params(figsize, backend, return_as_pc, **pc_kwargs)
        pc = PlotCollection.grid(
            crps_ds,
            rows=[*extra_dims],
            cols=["split"],
            aes={"color": ["split"]},
            backend=backend,
            **pc_kwargs,
        )

        cv_x = xr.DataArray(
            np.arange(len(cv_labels)), dims=["cv"], coords={"cv": cv_labels}
        )
        pc.map(azp.visuals.line_xy, x=cv_x, y=crps_ds["crps"], **(line_kwargs or {}))
        pc.add_legend("split")
        pc.map(
            azp.visuals.labelled_title,
            subset_info=True,
            labeller=mix_labellers((NoVarLabeller, DimCoordLabeller))(),
            ignore_aes={"color"},
        )

        return _extract_matplotlib_result(pc, return_as_pc)
