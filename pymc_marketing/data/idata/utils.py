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
"""Standalone utility functions for DataTree operations."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from numpy.random import Generator, RandomState

from pymc_marketing.data.idata.schema import Frequency


def idata_to_zarr(
    idata: xr.DataTree,
    store: str | Path,
    groups: list[str] | None = None,
) -> None:
    """Save a DataTree to a Zarr store.

    Parameters
    ----------
    idata : xr.DataTree
        The DataTree to save.
    store : str or Path
        Path to the Zarr store directory.
    groups : list of str, optional
        Groups to save. If None, all groups are saved.
    """
    if groups is not None:
        filtered = {
            f"/{g}": idata[f"/{g}"].to_dataset()
            for g in groups
            if f"/{g}" in idata.groups
        }
        idata = xr.DataTree.from_dict(filtered)
        idata.attrs = idata.attrs.copy() if hasattr(idata, "attrs") else {}

    idata.to_zarr(store)


_open_datatree_zarr = functools.partial(xr.open_datatree, engine="zarr")


def idata_from_zarr(store: str | Path) -> xr.DataTree:
    """Load a DataTree from a Zarr store.

    Parameters
    ----------
    store : str or Path
        Path to the Zarr store directory written by :func:`idata_to_zarr`.

    Returns
    -------
    xr.DataTree
    """
    return _open_datatree_zarr(store)


def filter_idata_by_dates(
    idata: xr.DataTree,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> xr.DataTree:
    """Filter DataTree to a date range.

    Parameters
    ----------
    idata : xr.DataTree
        DataTree to filter
    start_date : str or pd.Timestamp, optional
        Start date (inclusive)
    end_date : str or pd.Timestamp, optional
        End date (inclusive)

    Returns
    -------
    xr.DataTree
        New DataTree with filtered groups

    Examples
    --------
    >>> filtered = filter_idata_by_dates(idata, "2024-01-01", "2024-12-31")
    """
    if start_date is None and end_date is None:
        return idata

    date_slice = {"date": slice(start_date, end_date)}

    filtered_groups = {}
    for path in idata.groups:
        if path == "/":
            continue
        group_name = path.lstrip("/")
        ds = idata[path].to_dataset()
        if "date" in ds.dims:
            ds = ds.sel(**date_slice)
        filtered_groups[group_name] = ds

    result = xr.DataTree.from_dict({f"/{k}": v for k, v in filtered_groups.items()})
    result.attrs = idata.attrs.copy()
    return result


def filter_idata_by_dims(
    idata: xr.DataTree,
    **dim_filters,
) -> xr.DataTree:
    """Filter DataTree by dimension values.

    Parameters
    ----------
    idata : xr.DataTree
        DataTree to filter
    **dim_filters
        Dimension filters, e.g., country="US", channel=["TV", "Radio"]
        Note: When filtering to a single value, the dimension is dropped
        (xarray's default behavior). When filtering to multiple values,
        the dimension is preserved.

    Returns
    -------
    xr.DataTree
        New DataTree with filtered groups

    Raises
    ------
    ValueError
        If a dimension in dim_filters doesn't exist in any group

    Examples
    --------
    >>> filtered = filter_idata_by_dims(idata, country="US")
    >>> # Dimension "country" is dropped (single value)
    >>> filtered = filter_idata_by_dims(idata, channel=["TV", "Radio"])
    >>> # Dimension "channel" is preserved (multiple values)
    """
    if not dim_filters:
        return idata

    all_dims = set()
    for path in idata.groups:
        if path == "/":
            continue
        ds = idata[path].to_dataset()
        all_dims.update(ds.dims)

    for dim in dim_filters:
        if dim not in all_dims:
            raise ValueError(
                f"Dimension '{dim}' not found in any group. "
                f"Available dimensions: {sorted(all_dims)}"
            )

    filtered_groups = {}
    for path in idata.groups:
        if path == "/":
            continue
        group_name = path.lstrip("/")
        ds = idata[path].to_dataset()

        group_sel = {}
        for dim, values in dim_filters.items():
            if dim in ds.dims:
                group_sel[dim] = values

        if group_sel:
            filtered = ds.sel(**group_sel)
            filtered_groups[group_name] = filtered
        else:
            filtered_groups[group_name] = ds

    result = xr.DataTree.from_dict({f"/{k}": v for k, v in filtered_groups.items()})
    result.attrs = idata.attrs.copy()
    return result


def aggregate_idata_time(
    idata: xr.DataTree,
    period: Frequency,
    method: Literal["sum", "mean"] = "sum",
) -> xr.DataTree:
    """Aggregate DataTree over time periods.

    Parameters
    ----------
    idata : xr.DataTree
        DataTree to aggregate
    period : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
        Time period to aggregate to. Use "original" for no aggregation (returns
        unchanged), "all_time" to aggregate over the entire time dimension
        (removes the date dimension).
    method : {"sum", "mean"}, default "sum"
        Aggregation method

    Returns
    -------
    xr.DataTree
        New DataTree with aggregated groups (or unchanged if period="original")

    Examples
    --------
    >>> original = aggregate_idata_time(idata, "original")  # No aggregation
    >>> monthly = aggregate_idata_time(idata, "monthly", method="sum")
    >>> total = aggregate_idata_time(idata, "all_time", method="sum")
    """
    if period == "original":
        return idata

    if period == "all_time":
        aggregated_groups = {}
        for path in idata.groups:
            if path == "/":
                continue
            group_name = path.lstrip("/")
            ds = idata[path].to_dataset()

            if "date" not in ds.dims:
                aggregated_groups[group_name] = ds
                continue

            if method == "sum":
                aggregated = ds.sum(dim="date")
            elif method == "mean":
                aggregated = ds.mean(dim="date")
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

            aggregated_groups[group_name] = aggregated

        result = xr.DataTree.from_dict(
            {f"/{k}": v for k, v in aggregated_groups.items()}
        )
        result.attrs = idata.attrs.copy()
        return result

    period_map = {
        "weekly": "W",
        "monthly": "ME",
        "quarterly": "QE",
        "yearly": "YE",
    }
    freq = period_map[period]

    aggregated_groups = {}
    for path in idata.groups:
        if path == "/":
            continue
        group_name = path.lstrip("/")
        ds = idata[path].to_dataset()

        if "date" not in ds.dims:
            aggregated_groups[group_name] = ds
            continue

        if method == "sum":
            aggregated = ds.resample(date=freq).sum(dim="date")
        elif method == "mean":
            aggregated = ds.resample(date=freq).mean(dim="date")
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        aggregated_groups[group_name] = aggregated

    result = xr.DataTree.from_dict({f"/{k}": v for k, v in aggregated_groups.items()})
    result.attrs = idata.attrs.copy()
    return result


def aggregate_idata_dims(
    idata: xr.DataTree,
    dim: str,
    values: list[str],
    new_label: str,
    method: Literal["sum", "mean"] = "sum",
) -> xr.DataTree:
    """Aggregate multiple dimension values into one.

    Parameters
    ----------
    idata : xr.DataTree
        DataTree to aggregate
    dim : str
        Dimension to aggregate (e.g., "channel", "country")
    values : list of str
        Values to aggregate
    new_label : str
        Label for aggregated value
    method : {"sum", "mean"}, default "sum"
        Aggregation method

    Returns
    -------
    xr.DataTree
        New DataTree with aggregated dimension values

    Raises
    ------
    ValueError
        If the dimension doesn't exist in any group, or if new_label
        conflicts with existing coordinate values that aren't being aggregated

    Examples
    --------
    >>> # Combine social channels into one
    >>> combined = aggregate_idata_dims(
    ...     idata,
    ...     dim="channel",
    ...     values=["Facebook", "Instagram", "TikTok"],
    ...     new_label="Social",
    ...     method="sum",
    ... )
    """
    all_dims = set()
    for path in idata.groups:
        if path == "/":
            continue
        ds = idata[path].to_dataset()
        all_dims.update(ds.dims)

    if dim not in all_dims:
        raise ValueError(
            f"Dimension '{dim}' not found in any group. "
            f"Available dimensions: {sorted(all_dims)}"
        )

    for path in idata.groups:
        if path == "/":
            continue
        ds = idata[path].to_dataset()
        if dim in ds.dims:
            existing_coords = set(ds[dim].values)
            non_aggregated = existing_coords - set(values)
            if new_label in non_aggregated:
                raise ValueError(
                    f"new_label '{new_label}' conflicts with existing coordinate value "
                    f"in dimension '{dim}' that is not being aggregated. "
                    f"Existing values not being aggregated: {sorted(non_aggregated)}"
                )
            break

    aggregated_groups = {}
    for path in idata.groups:
        if path == "/":
            continue
        group_name = path.lstrip("/")
        ds = idata[path].to_dataset()

        if dim not in ds.dims:
            aggregated_groups[group_name] = ds
            continue

        all_coords = set(ds[dim].values)
        other_values = list(all_coords - set(values))

        result_vars = {}
        for var_name, var in ds.data_vars.items():
            if dim not in var.dims:
                result_vars[var_name] = var
            else:
                selected_var = var.sel({dim: values})

                if method == "sum":
                    aggregated_var = selected_var.sum(dim=dim)
                elif method == "mean":
                    aggregated_var = selected_var.mean(dim=dim)
                else:
                    raise ValueError(f"Unknown aggregation method: {method}")

                aggregated_var = aggregated_var.expand_dims({dim: [new_label]})

                if other_values:
                    other_var = var.sel({dim: other_values})
                    combined_var = xr.concat([other_var, aggregated_var], dim=dim)
                else:
                    combined_var = aggregated_var

                result_vars[var_name] = combined_var

        combined = xr.Dataset(result_vars)
        aggregated_groups[group_name] = combined

    result = xr.DataTree.from_dict({f"/{k}": v for k, v in aggregated_groups.items()})
    result.attrs = idata.attrs.copy()
    return result


def get_posterior_predictive(idata: xr.DataTree) -> xr.Dataset:
    """Return the posterior_predictive group from *idata*.

    Parameters
    ----------
    idata : xr.DataTree
        DataTree holding the fitted model results.

    Returns
    -------
    xr.Dataset
        The posterior_predictive group.

    Raises
    ------
    ValueError
        If posterior_predictive is absent from idata.
    """
    if "/posterior_predictive" not in idata.groups:
        raise ValueError(
            "No posterior_predictive data found in idata. "
            "Run MMM.sample_posterior_predictive() first."
        )
    return idata["/posterior_predictive"].to_dataset()


def get_prior_predictive(idata: xr.DataTree) -> xr.Dataset:
    """Return the prior_predictive group from *idata*.

    Parameters
    ----------
    idata : xr.DataTree
        DataTree holding the fitted model results.

    Returns
    -------
    xr.Dataset
        The prior_predictive group.

    Raises
    ------
    ValueError
        If prior_predictive is absent from idata.
    """
    if "/prior_predictive" not in idata.groups:
        raise ValueError(
            "No prior_predictive data found in idata. "
            "Run MMM.sample_prior_predictive() first."
        )
    return idata["/prior_predictive"].to_dataset()


def get_prior(idata: xr.DataTree) -> xr.Dataset:
    """Return the prior group from *idata*.

    Parameters
    ----------
    idata : xr.DataTree
        DataTree holding the fitted model results.

    Returns
    -------
    xr.Dataset
        The prior group.

    Raises
    ------
    ValueError
        If prior is absent from idata.
    """
    if "/prior" not in idata.groups:
        raise ValueError(
            "No prior data found in idata. Run MMM.sample_prior_predictive() first."
        )
    return idata["/prior"].to_dataset()


def subsample_draws(
    dataset: xr.Dataset,
    *,
    num_samples: int | None,
    random_state: RandomState | Generator | None = None,
) -> xr.Dataset:
    """Subsample draws from a Dataset with chain and draw dimensions.

    Randomly selects ``num_samples`` draws from the flattened
    chain × draw space and returns a new Dataset with a single
    chain and ``num_samples`` draws.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with ``chain`` and ``draw`` dimensions.
    num_samples : int or None
        Number of draws to keep.  If ``None`` or >= total available
        draws, returns *dataset* unchanged.
    random_state : RandomState, Generator, or None, optional
        Seed or random state for reproducibility.

    Returns
    -------
    xr.Dataset
        When ``num_samples`` is ``None`` or >= total draws, returns *dataset*
        unchanged (preserving its original chain/draw structure).
        When subsampling occurs, returns a new Dataset with shape
        ``(chain=1, draw=num_samples)``.

    Examples
    --------
    >>> sub = subsample_draws(posterior, num_samples=100, random_state=42)
    >>> sub.sizes["draw"]
    100
    """
    if num_samples is None:
        return dataset

    total_samples = dataset.sizes["chain"] * dataset.sizes["draw"]

    if num_samples >= total_samples:
        return dataset

    rng = np.random.default_rng(random_state)
    flat_indices = rng.choice(total_samples, size=num_samples, replace=False)

    stacked = dataset.stack(sample=("chain", "draw"))
    selected = stacked.isel(sample=flat_indices)

    return (
        selected.drop_vars(["chain", "draw", "sample"])
        .rename({"sample": "draw"})
        .expand_dims("chain")
    )
