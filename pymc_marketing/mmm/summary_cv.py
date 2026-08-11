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
"""Summary DataFrame generation for time-slice cross-validation results."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from typing import Any

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr

from pymc_marketing.metrics import crps as _crps_score
from pymc_marketing.mmm.summary_helpers import (
    DataFrameType,
    OutputFormat,
    StatsHelper,
    compute_summary_stats_with_hdi,
    convert_output,
)
from pymc_marketing.mmm.xarray_utils import _select_dims

__all__ = [
    "MMMCVSummaryFactory",
]

_stats = StatsHelper()


def _validate_cv_results(cv_data: xr.DataTree) -> None:
    """Raise if cv_data is not a valid CV DataTree."""
    if not isinstance(cv_data, xr.DataTree):
        raise TypeError(f"cv_data must be xr.DataTree, got {type(cv_data).__name__}.")
    if not hasattr(cv_data, "cv_metadata"):
        raise ValueError(
            "cv_data must have a 'cv_metadata' group. "
            "Ensure TimeSliceCrossValidator.run() has been called and the "
            "resulting DataTree is passed here."
        )


def _extract_cv_labels(cv_data: xr.DataTree) -> list[str]:
    """Return the list of CV fold labels from cv_metadata coords."""
    return list(cv_data.cv_metadata.coords["cv"].values)


def _read_fold_meta(
    cv_data: xr.DataTree, cv_label: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_train, y_train, X_test, y_test) for a given fold label."""
    meta = cv_data.cv_metadata["metadata"].sel(cv=cv_label).values.item()
    return meta["X_train"], meta["y_train"], meta["X_test"], meta["y_test"]


def _build_predictions_arrays(
    cv_data: xr.DataTree,
    pp: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Build stacked train/test/observed/train-end arrays across all CV folds."""
    cv_labels = _extract_cv_labels(cv_data)
    full_dates = pp.coords["date"].values

    y_train_list: list[xr.DataArray] = []
    y_test_list: list[xr.DataArray] = []
    y_obs_list: list[xr.DataArray] = []
    train_end_list: list[float] = []

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
        train_end_list.append(mdates.date2num(train_dates.max()))

    cv_coord = xr.DataArray(cv_labels, dims=["cv"], name="cv")
    y_train_da = xr.concat(y_train_list, dim=cv_coord).assign_coords(cv=cv_labels)
    y_test_da = xr.concat(y_test_list, dim=cv_coord).assign_coords(cv=cv_labels)
    y_obs_da = xr.concat(y_obs_list, dim=cv_coord).assign_coords(cv=cv_labels)
    train_end_da = xr.DataArray(train_end_list, dims=["cv"], coords={"cv": cv_labels})
    return y_train_da, y_test_da, y_obs_da, train_end_da


def _pred_matrix_for_rows(
    cv_data: xr.DataTree,
    cv_label: str,
    rows_df: pd.DataFrame,
) -> np.ndarray:
    """Build (n_samples, n_rows) prediction matrix for CRPS computation."""
    da = cv_data["/posterior_predictive"].dataset["y_original_scale"].sel(cv=cv_label)

    base_dims = {"chain", "draw", "date"}
    extra_dims = [d for d in da.dims if d not in base_dims]

    da_s = da.stack(sample=("chain", "draw"))
    if da_s.dims[0] != "sample":
        da_s = da_s.transpose("sample", ...)

    n_samples = int(da_s.sizes["sample"])
    n_rows = len(rows_df)
    mat = np.empty((n_samples, n_rows))

    for j, (_, row) in enumerate(rows_df.iterrows()):
        sel_kwargs: dict[str, Any] = {"date": row["date"]}
        for dim in extra_dims:
            if dim in row.index:
                sel_kwargs[dim] = row[dim]
        arr = np.squeeze(da_s.sel(**sel_kwargs).values)
        if arr.ndim == 0:
            arr = arr.reshape(n_samples)
        mat[:, j] = arr[:n_samples]

    return mat


def _filter_rows_and_y(
    df: pd.DataFrame | None,
    y: pd.Series | None,
    indexers: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter DataFrame rows by column equality, return aligned y array."""
    if df is None or len(df) == 0:
        return pd.DataFrame(), np.array([])
    mask = np.ones(len(df), dtype=bool)
    for col, val in indexers.items():
        if col in df.columns:
            mask &= df[col] == val
    return df[mask].reset_index(drop=True), np.asarray(y)[mask]


def _crps_for_split(
    cv_data: xr.DataTree,
    cv_label: str,
    X: pd.DataFrame,
    y: pd.Series,
    dim_indexers: dict[str, Any],
) -> float:
    """Compute mean CRPS for one fold/split. Returns np.nan on failure or empty set."""
    try:
        X_filtered, y_arr = _filter_rows_and_y(X, y, dim_indexers)
        if len(X_filtered) == 0:
            return float(np.nan)
        pred_mat = _pred_matrix_for_rows(cv_data, cv_label, X_filtered)
        return float(_crps_score(y_true=y_arr, y_pred=pred_mat))
    except Exception as exc:
        warnings.warn(
            f"CRPS computation failed for fold '{cv_label}': {exc}",
            UserWarning,
            stacklevel=3,
        )
        return float(np.nan)


class MMMCVSummaryFactory:
    """Factory for creating summary DataFrames from CV DataTree results.

    Parameters
    ----------
    cv_data : xr.DataTree
        Combined DataTree produced by ``TimeSliceCrossValidator.run()``.
    """

    def __init__(self, cv_data: xr.DataTree) -> None:
        _validate_cv_results(cv_data)
        self.cv_data = cv_data

    def predictions(
        self,
        hdi_probs: Sequence[float] = (0.94,),
        dims: dict[str, Any] | None = None,
        output_format: OutputFormat = "pandas",
    ) -> DataFrameType:
        """Create posterior predictive summary per CV fold in long format.

        Parameters
        ----------
        hdi_probs : sequence of float, default (0.94,)
            HDI probability levels.
        dims : dict, optional
            Dimension filters applied before summarization.
        output_format : {"pandas", "polars"}, default "pandas"
            Output DataFrame format.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Long-format table with columns ``cv``, ``date``, ``split``,
            ``mean``, ``median``, HDI bounds, and ``observed``.
        """
        _stats.validate_hdi_probs(hdi_probs)

        if (
            not hasattr(self.cv_data, "posterior_predictive")
            or "y_original_scale" not in self.cv_data.posterior_predictive
        ):
            raise ValueError(
                "cv_data must have posterior_predictive['y_original_scale']."
            )

        pp = self.cv_data.posterior_predictive["y_original_scale"]
        y_train_da, y_test_da, y_obs_da, _ = _build_predictions_arrays(self.cv_data, pp)

        if dims:
            y_train_da = _select_dims(y_train_da, dims)
            y_test_da = _select_dims(y_test_da, dims)
            y_obs_da = _select_dims(y_obs_da, dims)

        split_ds = xr.Dataset({"train": y_train_da, "test": y_test_da})
        stacked = split_ds.to_array(dim="split")
        stacked.name = "prediction"

        df = compute_summary_stats_with_hdi(stacked, hdi_probs)

        observed_df = y_obs_da.to_dataframe(name="observed").reset_index()
        merge_keys = [
            "cv",
            "date",
            *[d for d in observed_df.columns if d not in {"cv", "date", "observed"}],
        ]
        df = df.merge(observed_df, on=merge_keys, how="left")

        return convert_output(df, output_format)

    def param_stability(
        self,
        var_names: list[str] | None = None,
        dims: dict[str, Any] | None = None,
        hdi_probs: Sequence[float] = (0.94,),
        output_format: OutputFormat = "pandas",
    ) -> DataFrameType:
        """Summarize parameter posteriors across CV folds.

        Parameters
        ----------
        var_names : list of str or None
            Variables to include. If None, all posterior variables are used.
        dims : dict, optional
            Dimension filters.
        hdi_probs : sequence of float, default (0.94,)
            HDI probability levels.
        output_format : {"pandas", "polars"}, default "pandas"
            Output DataFrame format.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Table with ``cv``, ``variable``, mean, median, and HDI columns.
            Additional coordinate columns are preserved when present.
        """
        _stats.validate_hdi_probs(hdi_probs)

        if not hasattr(self.cv_data, "posterior"):
            raise ValueError("cv_data has no 'posterior' group.")
        if "cv" not in self.cv_data.posterior.dims:
            raise ValueError(
                "No 'cv' coordinate found in cv_data.posterior. "
                "Ensure the DataTree was produced by TimeSliceCrossValidator.run()."
            )

        posterior = self.cv_data["/posterior"].dataset
        if dims:
            posterior = _select_dims(posterior, dims)

        variables = var_names if var_names is not None else list(posterior.data_vars)
        cv_labels = _extract_cv_labels(self.cv_data)

        all_dfs: list[pd.DataFrame] = []
        for cv_label in cv_labels:
            for var_name in variables:
                if var_name not in posterior:
                    continue
                da = posterior[var_name].sel(cv=cv_label)
                df = compute_summary_stats_with_hdi(da, hdi_probs)
                df["cv"] = cv_label
                df["variable"] = var_name
                all_dfs.append(df)

        if not all_dfs:
            return convert_output(
                pd.DataFrame(columns=["cv", "variable", "mean", "median"]),
                output_format,
            )

        result_df = pd.concat(all_dfs, ignore_index=True)
        index_cols = [
            c
            for c in result_df.columns
            if c
            not in {
                "mean",
                "median",
                *[col for col in result_df.columns if col.startswith("abs_error_")],
            }
        ]
        stat_cols = [c for c in result_df.columns if c not in index_cols]
        result_df = result_df[index_cols + stat_cols]

        return convert_output(result_df, output_format)

    def crps(
        self,
        dims: dict[str, Any] | None = None,
        output_format: OutputFormat = "pandas",
    ) -> DataFrameType:
        """Create mean CRPS summary per CV fold and split.

        Parameters
        ----------
        dims : dict, optional
            Filters which coordinate values of extra dimensions appear.
        output_format : {"pandas", "polars"}, default "pandas"
            Output DataFrame format.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Table with ``split``, ``cv``, optional extra-dimension columns,
            and ``mean_crps``.
        """
        if not hasattr(self.cv_data, "cv_metadata"):
            raise ValueError("cv_data must have a 'cv_metadata' group.")
        if (
            not hasattr(self.cv_data, "posterior_predictive")
            or "y_original_scale" not in self.cv_data.posterior_predictive
        ):
            raise ValueError(
                "cv_data must have posterior_predictive['y_original_scale']."
            )

        pp = self.cv_data.posterior_predictive["y_original_scale"]
        cv_labels = _extract_cv_labels(self.cv_data)

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
            if combo_shape:
                multi_idx = tuple(
                    int(i) for i in np.unravel_index(flat_idx, combo_shape)
                )
            else:
                multi_idx = ()
            for fold_idx, lbl in enumerate(cv_labels):
                X_train, y_train, X_test, y_test = _read_fold_meta(self.cv_data, lbl)
                data_arr[(0, *multi_idx, fold_idx)] = _crps_for_split(  # type: ignore[index]
                    self.cv_data, lbl, X_train, y_train, dim_indexers
                )
                data_arr[(1, *multi_idx, fold_idx)] = _crps_for_split(  # type: ignore[index]
                    self.cv_data, lbl, X_test, y_test, dim_indexers
                )

        coords: dict[str, Any] = {
            "split": ["train", "test"],
            **combo_coords,
            "cv": cv_labels,
        }
        crps_da = xr.DataArray(
            data_arr, dims=["split", *extra_dims, "cv"], coords=coords, name="mean_crps"
        )

        df = crps_da.to_dataframe(name="mean_crps").reset_index()
        return convert_output(df, output_format)
