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
"""Data conversion utilities for the Bass diffusion model.

Provides :func:`to_bass_dataset` to convert common data formats (xarray Datasets,
pandas DataFrames/Series, numpy arrays) into a canonical ``xr.Dataset`` that the
:class:`~pymc_marketing.bass.model.BassModel` uses internally.
"""

from functools import singledispatch

import numpy as np
import pandas as pd
import xarray as xr


@singledispatch
def to_bass_dataset(data):
    """Convert common data formats to the canonical ``xr.Dataset`` for the Bass model.

    Parameters
    ----------
    data : xr.Dataset, pd.DataFrame, pd.Series, or np.ndarray
        Input data. See the table below for conversion rules.

    Returns
    -------
    xr.Dataset
        Dataset with at least an ``"observed"`` variable and a ``"T"``
        coordinate. Additional coordinates from the input are preserved.

    Notes
    -----
    **Conversion rules by input type:**

    ==================  =======================================================
    Input               Behaviour
    ==================  =======================================================
    ``xr.Dataset``      Returned as-is.  ``"T"`` coord is auto-generated from
                        ``np.arange(n)`` when missing.
    ``pd.DataFrame``    Column ``"observed"`` is treated as a single-product
                        time series.  Without that column, **every** numeric
                        column is treated as a separate product (wide format).
                        A column named ``"T"`` is used as the time coordinate.
    ``pd.Series``       Single-product time series.
    ``np.ndarray`` 1D   Single-product time series.
    ``np.ndarray`` 2D   Multi-product matrix ``(T, product)`` with
                        auto-generated product labels ``"P0"``, ``"P1"``, …
    ``np.ndarray`` 3D+  Raises ``ValueError``; use ``xr.Dataset`` instead.
    ==================  =======================================================

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from pymc_marketing.bass import to_bass_dataset

        # Single product from a 1-D array
        ds = to_bass_dataset(np.array([10, 25, 50, 80]))
        # <xarray.Dataset>
        # Dimensions:   (T: 4)
        # Coordinates:  T (T) int64 0 1 2 3
        # Data vars:    observed (T) int64 10 25 50 80

        # Multiple products from a wide DataFrame
        df = pd.DataFrame(
            {
                "product_A": [10, 25, 40],
                "product_B": [15, 30, 45],
            }
        )
        ds = to_bass_dataset(df)
        # <xarray.Dataset>
        # Dimensions:   (T: 3, product: 2)
        # Coordinates:  T (T) int64 0 1 2, product (product) object ...
        # Data vars:    observed (T, product) int64

        # From a 2-D matrix
        data = np.random.poisson(lam=100, size=(50, 3))
        ds = to_bass_dataset(data)
        # Dimensions:   (T: 50, product: 3)
    """
    raise TypeError(
        f"Unsupported data type: {type(data)}. "
        "Supported types: xr.Dataset, pd.DataFrame, pd.Series, np.ndarray."
    )


@to_bass_dataset.register(xr.Dataset)
def _from_xarray(data: xr.Dataset) -> xr.Dataset:
    """Preserve an ``xr.Dataset``, adding ``T`` when missing."""
    if "T" not in data.coords:
        n = next(iter(data.dims.values()))
        data = data.assign_coords(T=np.arange(n))
    return data


@to_bass_dataset.register(pd.DataFrame)
def _from_dataframe(data: pd.DataFrame) -> xr.Dataset:
    """Convert a ``pd.DataFrame`` to an ``xr.Dataset``.

    * If an ``"observed"`` column exists → single-product (extra numeric
      columns are kept as data variables).
    * Otherwise → wide multi-product format (column names → product labels).
    """
    if "observed" in data.columns:
        return _dataframe_single_product(data)

    numeric_cols = _get_numeric_columns(data)
    if not numeric_cols:
        raise ValueError(
            "DataFrame has no valid data columns. "
            "Add an ``'observed'`` column or numeric product columns."
        )

    # Wide multi-product format
    t, cols = _extract_t_column(data, numeric_cols)
    matrix = np.column_stack([data[c].to_numpy() for c in cols])

    return xr.Dataset(
        {"observed": (("T", "product"), matrix)},
        coords={"T": t, "product": cols},
    )


def _dataframe_single_product(data: pd.DataFrame) -> xr.Dataset:
    """Convert a DataFrame that has an ``"observed"`` column."""
    obs = data["observed"].to_numpy()
    remaining = data.drop(columns=["observed"])
    t, _ = _extract_t_column(remaining, list(remaining.columns))

    ds = xr.Dataset({"observed": ("T", obs)}, coords={"T": t})

    for col in remaining.columns:
        if col == "T":
            continue
        if not _is_non_numeric(remaining[col]):
            ds[col] = ("T", remaining[col].to_numpy())

    return ds


@to_bass_dataset.register(pd.Series)
def _from_series(data: pd.Series) -> xr.Dataset:
    """Convert a ``pd.Series`` to a single-product Dataset."""
    return xr.Dataset(
        {"observed": ("T", data.to_numpy())},
        coords={"T": np.arange(len(data))},
    )


@to_bass_dataset.register(np.ndarray)
def _from_array(data: np.ndarray) -> xr.Dataset:
    """Convert a ``np.ndarray`` to a Dataset."""
    if data.ndim == 1:
        return xr.Dataset(
            {"observed": ("T", data)},
            coords={"T": np.arange(len(data))},
        )
    elif data.ndim == 2:
        n_products = data.shape[1]
        return xr.Dataset(
            {"observed": (("T", "product"), data)},
            coords={
                "T": np.arange(data.shape[0]),
                "product": [f"P{i}" for i in range(n_products)],
            },
        )
    else:
        raise ValueError(
            f"Arrays with {data.ndim} dimensions require an xr.Dataset "
            "with explicit dimension names."
        )


def _get_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return column names that hold numeric data."""
    return [col for col in data.columns if not _is_non_numeric(data[col])]


def _extract_t_column(
    data: pd.DataFrame, candidates: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Extract the ``"T"`` column (if present) from *candidates*.

    Returns ``(t_values, remaining_column_names)``.
    """
    if "T" in candidates:
        t = data["T"].to_numpy()
        remaining = [c for c in candidates if c != "T"]
    else:
        t = np.arange(len(data))
        remaining = list(candidates)
    return t, remaining


def _is_non_numeric(series: pd.Series) -> bool:
    """Return ``True`` for date/time, string, or other non-numeric columns."""
    return not pd.api.types.is_numeric_dtype(series)
