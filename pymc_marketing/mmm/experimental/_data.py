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
"""Labeled raw data and fitted scaling for the experimental MMM."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pymc_marketing.mmm.data_conversion import _pandas_columns_to_dataarrays
from pymc_marketing.mmm.scaling import (
    DataDerivedScaling,
    FixedScaling,
    Scaling,
    VariableScaling,
    deserialize_variable_scaling,
    validate_fixed_scaling_keys,
)


def _dates(values: Any) -> pd.DatetimeIndex:
    """Parse dates without accepting missing values or numeric timestamps."""
    if pd.api.types.is_numeric_dtype(np.asarray(values).dtype):
        raise ValueError("Dates must be datetime values or date strings, not numbers.")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError) as error:
        raise ValueError("Dates must contain valid datetime values.") from error
    if dates.hasnans or dates.tz is not None:
        raise ValueError("Dates must be finite, timezone-naive datetime values.")
    return dates


def _align_labels(
    array: xr.DataArray, reference: xr.Dataset | xr.DataArray
) -> xr.DataArray:
    """Require matching labeled dimensions, then restore reference order."""
    indexers = {}
    for dim in array.dims:
        if dim not in reference.dims or dim not in array.coords:
            raise ValueError(f"Dimension {dim!r} must have matching named coordinates.")
        source = array.get_index(dim)
        target = reference.get_index(dim)
        if (
            not source.is_unique
            or len(source) != len(target)
            or not source.isin(target).all()
        ):
            raise ValueError(
                f"Coordinate labels for dimension {dim!r} do not match the data."
            )
        indexers[dim] = reference.coords[dim]
    return array.sel(indexers)


def _target_array(
    y: pd.Series | pd.DataFrame | xr.DataArray | np.ndarray,
    X: pd.DataFrame | xr.Dataset,
    ds: xr.Dataset,
    *,
    date_column: str,
    target_column: str,
    dims: tuple[str, ...],
) -> xr.DataArray:
    """Convert a separate response while retaining its labeled alignment."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("A separate y DataFrame must have exactly one column.")
        y = y.iloc[:, 0]
    if isinstance(y, pd.Series):
        if isinstance(y.index, pd.MultiIndex) or isinstance(y.index, pd.DatetimeIndex):
            if not y.index.is_unique:
                raise ValueError("The y index must be unique.")
            if isinstance(y.index, pd.DatetimeIndex):
                y = y.rename_axis("date")
            y = y.to_xarray()
        elif isinstance(X, pd.DataFrame):
            if not y.index.is_unique or not X.index.is_unique:
                raise ValueError(
                    "X and y row indexes must be unique for labeled alignment."
                )
            if len(y) != len(X) or not y.index.isin(X.index).all():
                raise ValueError("The y row index must match the X row index.")
            y = y.reindex(X.index).to_numpy()
        else:
            raise ValueError(
                "A separate y for an xarray Dataset needs named coordinates."
            )
    if isinstance(y, np.ndarray):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "A separate y for an xarray Dataset needs named coordinates."
            )
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(
                "A separate array y must have one value for each DataFrame row."
            )
        frame = X[[date_column, *dims]].copy()
        frame[date_column] = _dates(frame[date_column])
        frame[target_column] = y
        y = _pandas_columns_to_dataarrays(frame, date_column, dims, [target_column])[
            target_column
        ]
    if not isinstance(y, xr.DataArray):
        raise TypeError(
            "y must be a Series, one-column DataFrame, DataArray, or ndarray."
        )
    if date_column != "date" and date_column in y.dims:
        y = y.rename({date_column: "date"})
    if set(y.dims) != {"date", *dims}:
        raise ValueError(f"y must have dimensions {('date', *dims)!r}.")
    if "date" not in y.coords:
        raise ValueError("A separate y DataArray must have date coordinates.")
    y = y.assign_coords(date=_dates(y.coords["date"].values))
    return _align_labels(y, ds).transpose("date", *dims)


def normalize_data(
    X: pd.DataFrame | xr.Dataset,
    y: pd.Series | pd.DataFrame | xr.DataArray | np.ndarray | None = None,
    *,
    date_column: str,
    target_column: str,
    dims: tuple[str, ...],
) -> xr.Dataset:
    """Normalize raw named variables to a sorted, labeled dataset.

    Parameters
    ----------
    X : pandas.DataFrame or xarray.Dataset
        Raw inputs, optionally including the response and intermediate observations.
    y : pandas.Series, pandas.DataFrame, xarray.DataArray, numpy.ndarray, optional
        A separate response, stored under ``target_column``.
        Arrays follow DataFrame rows; Dataset responses require named coordinates.
    date_column : str
        Input date column or dimension, normalized to ``date``.
    target_column : str
        Name of the response when supplied separately.
    dims : tuple of str
        Panel dimensions; DataFrame rows must cover their complete Cartesian grid.

    Returns
    -------
    xarray.Dataset
        Unscaled named variables and preserved coordinates.
        Canonical ``_channel`` arrays are split into their channel labels.

    Notes
    -----
    Missing values are never filled or imputed.
    Only variables used by the graph must be finite.
    Unused columns do not impose model requirements.
    """
    dims = tuple(dims)
    if len(set(dims)) != len(dims) or "date" in dims or date_column in dims:
        raise ValueError(
            "Panel dimensions must be unique and cannot include the date dimension."
        )
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise ValueError("DataFrame column names must be unique.")
        missing = set((date_column, *dims)) - set(X.columns)
        if missing:
            raise ValueError(f"Missing date or panel columns: {sorted(missing)!r}.")
        frame = X.copy()
        frame[date_column] = _dates(frame[date_column])
        index_columns = [date_column, *dims]
        if frame[index_columns].isna().any().any():
            raise ValueError(
                "Date and panel coordinates cannot contain missing values."
            )
        if frame.duplicated(index_columns).any():
            raise ValueError("Duplicate date or panel coordinate rows are not allowed.")
        expected = int(np.prod([frame[column].nunique() for column in index_columns]))
        if len(frame) != expected:
            raise ValueError("DataFrame contains missing panel cells.")
        variables: list[str] = []
        for column in frame:
            if column not in index_columns:
                if not isinstance(column, str):
                    raise ValueError("Data variable names must be strings.")
                variables.append(column)
        ds = xr.Dataset(
            _pandas_columns_to_dataarrays(frame, date_column, dims, variables)
        )
        coords = {"date": pd.unique(frame[date_column])}
        coords.update({dim: pd.unique(frame[dim]) for dim in dims})
        ds = ds.reindex(coords) if variables else xr.Dataset(coords=coords)
    elif isinstance(X, xr.Dataset):
        ds = X.copy(deep=False)
        if date_column != "date" and date_column in ds.dims:
            if "date" in ds.dims or "date" in ds.variables:
                raise ValueError("Both input and normalized date names are present.")
            ds = ds.rename({date_column: "date"})
        if "date" not in ds.coords or "date" not in ds.dims:
            raise ValueError("The Dataset must have a labeled date dimension.")
        ds = ds.assign_coords(date=_dates(ds.coords["date"].values))
        if "_channel" in ds.data_vars:
            channel = ds["_channel"]
            if "channel" not in channel.dims or "channel" not in channel.coords:
                raise ValueError(
                    "Canonical _channel data must have labeled channel coordinates."
                )
            labels = channel.coords["channel"].values.tolist()
            if len(set(labels)) != len(labels) or not all(
                isinstance(label, str) for label in labels
            ):
                raise ValueError("Canonical channel labels must be unique strings.")
            conflicts = set(labels) & (set(ds.variables) - {"_channel"})
            if conflicts:
                raise ValueError(
                    f"Canonical channel labels conflict with named variables: {sorted(conflicts)!r}."
                )
            ds = ds.drop_vars("_channel")
            for label in labels:
                ds[label] = channel.sel(channel=label, drop=True)
    else:
        raise TypeError("X must be a pandas DataFrame or xarray Dataset.")
    if not ds.sizes.get("date", 0):
        raise ValueError("Data must contain at least one date.")
    for dim in ("date", *dims):
        if dim not in ds.dims or dim not in ds.coords:
            raise ValueError(f"Data must contain the labeled dimension {dim!r}.")
    for dim in ds.dims:
        if dim not in ds.coords:
            raise ValueError(f"Dimension {dim!r} must have named coordinates.")
        index = ds.get_index(dim)
        if not index.is_unique or pd.isna(index).any():
            raise ValueError(f"Coordinate {dim!r} must have unique, nonmissing labels.")
    ds = ds.sortby("date")
    if y is not None:
        if target_column in ds.variables:
            raise ValueError(f"Response {target_column!r} is present in both X and y.")
        ds[target_column] = _target_array(
            y, X, ds, date_column=date_column, target_column=target_column, dims=dims
        )
    return ds


def _scaling_config(
    scaling: Scaling | Mapping[str, Any] | None, kind: str, dims: tuple[str, ...]
) -> VariableScaling:
    """Resolve only the requested scaling recipe without mutating user mappings."""
    if isinstance(scaling, Scaling):
        return getattr(scaling, kind)
    if scaling is not None and not isinstance(scaling, Mapping):
        raise TypeError("scaling must be a Scaling instance or a mapping.")
    value = None if scaling is None else scaling.get(kind)
    if value is None:
        return DataDerivedScaling(method="max", dims=dims)
    if isinstance(value, VariableScaling):
        return value
    if isinstance(value, Mapping):
        return deserialize_variable_scaling(dict(value))
    raise TypeError(f"The {kind} scaling must be a VariableScaling or mapping.")


def _compute_scale(data: xr.DataArray, scaling: VariableScaling) -> xr.DataArray:
    """Apply existing signed reductions or construct a labeled fixed divisor."""
    try:
        finite = np.isfinite(data.values).all()
    except TypeError as error:
        raise ValueError(
            f"Data for scaling {data.name!r} must be numeric and finite."
        ) from error
    if not finite:
        raise ValueError(
            f"Data for scaling {data.name!r} must be finite, with no missing observations."
        )
    reduce_dims = ("date", *scaling.dims)
    if not set(reduce_dims).issubset(data.dims):
        raise ValueError(
            f"Scaling dimensions {reduce_dims!r} are not present in {data.dims!r}."
        )
    if isinstance(scaling, DataDerivedScaling):
        scale = getattr(data, scaling.method)(dim=reduce_dims)
        scale = xr.where(scale == 0, 1.0, scale)
    elif isinstance(scaling, FixedScaling):
        remaining = tuple(dim for dim in data.dims if dim not in reduce_dims)
        value = scaling.value
        if isinstance(value, dict):
            if len(remaining) != 1:
                raise ValueError(
                    "Dict-valued fixed scaling requires exactly one remaining dimension; use a DataArray."
                )
            dim = remaining[0]
            labels = [str(label) for label in data.coords[dim].values]
            validate_fixed_scaling_keys(scaling, labels, str(data.name))
            scale = xr.DataArray(
                [value[label] for label in labels],
                dims=dim,
                coords={dim: data.coords[dim]},
            )
        elif isinstance(value, xr.DataArray):
            if not set(value.dims).issubset(remaining):
                raise ValueError(
                    "Fixed scale dimensions must be a subset of the non-reduced data dimensions."
                )
            scale = _align_labels(value, data).astype(float)
        else:
            scale = xr.DataArray(float(value))
        if (scale <= 0).any():
            raise ValueError("Fixed scaling values must be positive.")
    else:
        raise TypeError(f"Unsupported scaling recipe: {type(scaling).__name__}.")
    if not np.isfinite(scale.values).all():
        raise ValueError("Scaling values must be finite.")
    return scale


def compute_scales(
    ds: xr.Dataset,
    *,
    channels: Sequence[str],
    target_column: str | None,
    dims: tuple[str, ...],
    scaling: Scaling | Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Compute only requested training divisors, preserving signed reductions.

    Parameters
    ----------
    ds : xarray.Dataset
        Normalized raw training data.
    channels : sequence of str
        Channel variables actually used by the recipe; an empty sequence skips media scaling.
    target_column : str, optional
        Response to scale, or ``None`` for a custom equation on its declared raw scale.
    dims : tuple of str
        Panel dimensions, reduced by default along with date.
    scaling : Scaling or mapping, optional
        Existing MMM scaling recipes, with omitted entries defaulting to max scaling.

    Returns
    -------
    xarray.Dataset
        Requested ``channel_scale`` and/or ``target_scale`` variables.

    Notes
    -----
    A data-derived divisor of exactly zero becomes one, so nonzero future data is not silently discarded.
    Nonzero negative max/mean reductions retain the existing signed semantics.
    All divisors must be finite; fixed divisors additionally retain positive-value validation.
    """
    result = {}
    channels = tuple(channels)
    if channels:
        arrays = []
        for channel in channels:
            if channel not in ds.data_vars:
                raise ValueError(f"Missing channel data {channel!r}.")
            if set(ds[channel].dims) != {"date", *dims}:
                raise ValueError(
                    f"Channel {channel!r} must have dimensions {('date', *dims)!r}."
                )
            arrays.append(ds[channel].transpose("date", *dims))
        data = xr.concat(
            arrays, dim=xr.IndexVariable("channel", list(channels))
        ).transpose("date", *dims, "channel")
        data.name = "channel"
        result["channel_scale"] = _compute_scale(
            data, _scaling_config(scaling, "channel", dims)
        )
    if target_column is not None:
        if target_column not in ds.data_vars:
            raise ValueError(f"Missing target data {target_column!r}.")
        result["target_scale"] = _compute_scale(
            ds[target_column], _scaling_config(scaling, "target", dims)
        )
    return xr.Dataset(result)
