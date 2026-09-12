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
"""Labeled dataset validation for the experimental MMM."""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


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


def validate_dataset(data: xr.Dataset) -> xr.Dataset:
    """Check that every dimension is labeled and normalize the ``date`` coordinate.

    Parameters
    ----------
    data : xarray.Dataset
        Model inputs and observations. Each variable keeps its own dimensions.

    Returns
    -------
    xarray.Dataset
        The same variables with ``date`` parsed to a ``DatetimeIndex``.

    Raises
    ------
    TypeError
        If ``data`` is not a Dataset.
    ValueError
        If a dimension is unnamed, empty, unlabeled, or has duplicate or missing labels,
        or if dates are unparseable, duplicated, or out of order.
    """
    if not isinstance(data, xr.Dataset):
        raise TypeError("Data must be an xarray.Dataset with labeled variables.")
    for dim, size in data.sizes.items():
        if not isinstance(dim, str) or not dim:
            raise ValueError("Dimension names must be nonempty strings.")
        if not size or dim not in data.coords or data.coords[dim].dims != (dim,):
            raise ValueError(
                f"Dimension {dim!r} needs nonempty, one-dimensional coordinate labels."
            )
        index = data.get_index(dim)
        if not index.is_unique or pd.isna(index).any():
            raise ValueError(f"Coordinate {dim!r} must have unique, nonmissing labels.")
    if "date" in data.dims:
        dates = _dates(data.get_index("date"))
        if not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError("Dates must be unique and in increasing order.")
        data = data.assign_coords(date=dates)
    return data
