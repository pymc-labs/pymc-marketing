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
"""Data conversion utilities for MMM.

Normalises various input types (``pd.DataFrame``, ``xr.Dataset``,
``xr.DataArray``) into the canonical ``xr.Dataset`` format that the
MMM class uses internally.

Public vs. internal variable names
-----------------------------------
.. list-table::
    :header-rows: 1

    * - Public name
      - Internal name
      - Description
    * - ``media``
      - ``_channel``
      - Media channel spend data
    * - ``target``
      - ``_target``
      - Response/target variable
    * - ``_control``
      - ``_control``
      - Control variable data

The names ``channel`` and ``control`` **cannot** be used as data variable names
because xarray promotes them to dimension coordinates when the variable and
dimension share a name. Use ``media`` instead of ``channel``, and use
``_control`` for control data.

Examples
--------
.. code-block:: python

    import xarray as xr
    import pandas as pd
    from pymc_marketing.mmm._data_conversion import to_mmm_dataset

    # From a DataFrame
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=10, freq="W"),
            "tv": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "digital": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    )
    y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], name="sales")

    ds = to_mmm_dataset(
        df,
        y,
        date_column="date",
        channel_columns=["tv", "digital"],
        target_column="sales",
    )
    # ds has variables _channel and _target
"""

from functools import singledispatch

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Public API — single entry point
# ---------------------------------------------------------------------------


def to_mmm_dataset(
    X: pd.DataFrame | xr.Dataset | xr.DataArray,
    y: pd.Series | xr.DataArray | np.ndarray | None = None,
    *,
    date_column: str,
    dims: tuple[str, ...] = (),
    channel_columns: list[str],
    control_columns: list[str] | None = None,
    target_column: str | None = None,
) -> xr.Dataset:
    """Normalise X (and optionally y) to a single canonical ``xr.Dataset``.

    This is the **sole** entry point for data normalisation in the MMM
    pipeline.  It handles all supported input type combinations and returns
    a dataset with the canonical underscore-prefixed variable names
    (``_channel``, ``_target``, ``_control``).

    Parameters
    ----------
    X
        Feature data.
    y
        Target variable.  When *X* is already an ``xr.Dataset`` that contains
        a ``target`` / ``_target`` data variable this may be omitted.
    date_column
        Name of the date column in a ``pd.DataFrame`` input.
    dims
        Extra dimension names (e.g. ``("geo",)``).
    channel_columns
        Names of the media-channel columns.
    control_columns
        Names of control-variable columns.
    target_column
        Name of the target / response variable.

    Returns
    -------
    xr.Dataset
        Dataset with variables ``_channel``, (optional) ``_target``, and
        (optional) ``_control`` and coordinates for each dimension.
    """
    # 1. Normalise X.
    ds = _to_mmm_xarray(
        X,
        date_column=date_column,
        dims=dims,
        channel_columns=channel_columns,
        control_columns=control_columns,
    )

    # 2. Normalise y and merge into the dataset.
    if y is not None:
        y_da = _to_mmm_target(y)
        if "date" not in y_da.coords and isinstance(X, pd.DataFrame):
            y_da = _reshape_flat_target(
                y_da,
                X,
                date_column=date_column,
                dims=dims,
            )
        _add_target_to_dataset(ds, y_da, date_column=date_column, dims=dims)

    return ds


# ---------------------------------------------------------------------------
# X normalisation (internal singledispatch)
# ---------------------------------------------------------------------------


@singledispatch
def _to_mmm_xarray(X, /, **params) -> xr.Dataset:
    raise TypeError(
        f"Unsupported X type: {type(X)}. "
        f"Supported: xr.Dataset, xr.DataArray, pd.DataFrame."
    )


@_to_mmm_xarray.register(xr.Dataset)
def _(X, /, **params) -> xr.Dataset:
    rename = {}
    for public, internal in [
        ("media", "_channel"),
        ("channel", "_channel"),
        ("target", "_target"),
        ("control", "_control"),
    ]:
        if public in X.data_vars and public != internal:
            rename[public] = internal
    if rename:
        X = X.rename(rename)

    _validate_mmm_structure(X, **params)
    return X


@_to_mmm_xarray.register(xr.DataArray)
def _(X, /, **params) -> xr.Dataset:
    name = X.name or "media"
    return _to_mmm_xarray(X.to_dataset(name=name), **params)


@_to_mmm_xarray.register(pd.DataFrame)
def _(X, /, **params) -> xr.Dataset:
    date_column = params["date_column"]
    dims = params.get("dims", ())
    channel_columns = params["channel_columns"]
    control_columns = params.get("control_columns")

    datasets = []
    channel_ds = _pandas_to_xarray_dataarray(
        X, date_column, dims, channel_columns, "channel"
    )
    datasets.append(channel_ds)

    if control_columns:
        control_ds = _pandas_to_xarray_dataarray(
            X, date_column, dims, control_columns, "control"
        )
        datasets.append(control_ds)

    ds = xr.merge(datasets).fillna(0)
    ds = _reindex_to_user_order(ds, channel_columns, control_columns)
    return ds


# ---------------------------------------------------------------------------
# y normalisation (internal)
# ---------------------------------------------------------------------------


@singledispatch
def _to_mmm_target(y, /) -> xr.DataArray:
    raise TypeError(
        f"Unsupported y type: {type(y)}. "
        f"Supported: xr.DataArray, pd.Series, np.ndarray."
    )


@_to_mmm_target.register(xr.DataArray)
def _(y, /) -> xr.DataArray:
    return y


@_to_mmm_target.register(pd.Series)
def _(y, /) -> xr.DataArray:
    if isinstance(y.index, pd.MultiIndex):
        return y.to_xarray()
    return xr.DataArray(y.values, dims=("date",))


@_to_mmm_target.register(np.ndarray)
def _(y, /) -> xr.DataArray:
    return xr.DataArray(y, dims=("date",))


def _reshape_flat_target(
    y_da: xr.DataArray,
    X: pd.DataFrame,
    *,
    date_column: str,
    dims: tuple[str, ...],
) -> xr.DataArray:
    """Reshape a flat y DataArray using X's dimension columns for proper indexing.

    Uses pandas MultiIndex + sort_index to ensure correct alignment with
    xarray's sorted coordinate order.  The date column is normalised to
    the canonical name ``"date"`` to match ``_to_mmm_xarray``.
    """
    idx_cols = [date_column] + [d for d in dims if d in X.columns]
    y_df = pd.DataFrame(
        {"_target": y_da.values, **{col: X[col].values for col in idx_cols}}
    )
    return (
        y_df.rename(columns={date_column: "date"})
        .set_index(["date"] + [d for d in dims if d in X.columns])
        .sort_index()
        .to_xarray()["_target"]
    )
    return y_df.set_index(idx_cols).sort_index().to_xarray()["_target"]


def _add_target_to_dataset(
    ds: xr.Dataset,
    y_da: xr.DataArray,
    *,
    date_column: str,
    dims: tuple[str, ...],
) -> None:
    """Add *y_da* as ``_target`` to *ds* (in-place)."""
    ds["_target"] = y_da


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def _validate_mmm_structure(X: xr.Dataset, **params) -> None:
    if "_channel" not in X.data_vars:
        raise ValueError(
            "Dataset missing required variable '_channel' (or 'media'). "
            "Provide a 'media' or '_channel' data variable. "
            "Note: 'channel' cannot be used as a data variable name because "
            "xarray promotes it to a dimension coordinate."
        )

    if "date" not in X.coords:
        raise ValueError("Dataset missing required coordinate 'date'.")

    if "channel" not in X.coords:
        raise ValueError(
            "Dataset missing required coordinate 'channel'. "
            "Ensure your media variable has a 'channel' dimension."
        )

    control_columns = params.get("control_columns")
    if control_columns and "_control" not in X.data_vars:
        raise ValueError(
            f"control_columns={control_columns} but dataset is missing "
            f"'_control' data variable. Note: 'control' cannot be used as a "
            f"data variable name because xarray promotes it to a dimension "
            f"coordinate. Use '_control' instead."
        )


# ---------------------------------------------------------------------------
# DataFrame → xarray helpers
# ---------------------------------------------------------------------------


def _validate_dims_in_multiindex(
    index: pd.MultiIndex,
    dims: tuple[str, ...],
    date_column: str,
) -> list[str]:
    if date_column not in index.names:
        raise ValueError(f"date_column '{date_column}' not found in index")
    return [dim for dim in dims if dim in index.names]


def _validate_dims_in_dataframe(
    df: pd.DataFrame,
    dims: tuple[str, ...],
    date_column: str,
) -> list[str]:
    if date_column not in df.columns:
        raise ValueError(f"date_column '{date_column}' not found in DataFrame")
    return [dim for dim in dims if dim in df.columns]


def _validate_metrics(
    data: pd.DataFrame | pd.Series,
    metric_list: list[str],
) -> list[str]:
    if isinstance(data, pd.DataFrame):
        return [m for m in metric_list if m in data.columns]
    else:
        return [m for m in metric_list if m in data.index.names]


def _process_multiindex_series(
    series: pd.Series,
    date_column: str,
    valid_dims: list[str],
    metric_coordinate_name: str,
) -> xr.Dataset:
    df = series.reset_index()
    df_long = pd.DataFrame(
        {
            **{col: df[col] for col in [date_column, *valid_dims]},
            metric_coordinate_name: series.name,
            f"_{metric_coordinate_name}": series.values,
        }
    )
    df_long = df_long.drop_duplicates(
        subset=[date_column, *valid_dims, metric_coordinate_name]
    )
    df_long = df_long.rename(columns={date_column: "date"})
    if valid_dims:
        return df_long.set_index(
            ["date", *valid_dims, metric_coordinate_name]
        ).to_xarray()
    return df_long.set_index(["date", metric_coordinate_name]).to_xarray()


def _process_dataframe(
    df: pd.DataFrame,
    date_column: str,
    valid_dims: list[str],
    valid_metrics: list[str],
    metric_coordinate_name: str,
) -> xr.Dataset:
    df_long = df.melt(
        id_vars=[date_column, *valid_dims],
        value_vars=valid_metrics,
        var_name=metric_coordinate_name,
        value_name=f"_{metric_coordinate_name}",
    )
    df_long = df_long.drop_duplicates(
        subset=[date_column, *valid_dims, metric_coordinate_name]
    )
    df_long = df_long.rename(columns={date_column: "date"})
    if valid_dims:
        ds = df_long.set_index(
            ["date", *valid_dims, metric_coordinate_name]
        ).to_xarray()
    else:
        ds = df_long.set_index(["date", metric_coordinate_name]).to_xarray()
    return ds.reindex({metric_coordinate_name: valid_metrics})


def _reindex_to_user_order(
    dataset: xr.Dataset,
    channel_columns: list[str],
    control_columns: list[str] | None = None,
) -> xr.Dataset:
    dataset = dataset.reindex(channel=channel_columns)
    if control_columns is not None:
        dataset = dataset.reindex(control=control_columns)
    return dataset


def _pandas_to_xarray_dataarray(
    data: pd.DataFrame | pd.Series,
    date_column: str,
    dims: tuple[str, ...],
    metric_list: list[str],
    metric_coordinate_name: str,
) -> xr.Dataset:
    """Standalone version of ``MMM._create_xarray_from_pandas``."""
    if isinstance(data, pd.Series):
        valid_dims = _validate_dims_in_multiindex(data.index, dims, date_column)  # type: ignore[arg-type]
        return _process_multiindex_series(
            data, date_column, valid_dims, metric_coordinate_name
        )
    else:
        valid_dims = _validate_dims_in_dataframe(data, dims, date_column)
        valid_metrics = _validate_metrics(data, metric_list)
        return _process_dataframe(
            data, date_column, valid_dims, valid_metrics, metric_coordinate_name
        )
