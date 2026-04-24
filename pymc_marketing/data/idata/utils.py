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
"""Standalone utility functions for InferenceData operations."""

from __future__ import annotations

import functools
import warnings
from pathlib import Path
from typing import Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from numpy.random import Generator, RandomState

from pymc_marketing.data.idata.schema import Frequency


def from_netcdf(filepath: str | Path) -> az.InferenceData:
    """Load InferenceData from a NetCDF file, suppressing ``fit_data`` warnings.

    Parameters
    ----------
    filepath : str or Path
        Path to the NetCDF file.

    Returns
    -------
    az.InferenceData
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"fit_data group is not defined in the InferenceData scheme",
        )
        return az.from_netcdf(filepath)


def idata_to_zarr(idata: az.InferenceData, store: str | Path) -> None:
    """Save an InferenceData to a Zarr store.

    TODO: Remove this shim once we require ``arviz>=1.0``.

    Works with zarr>=3, which is not supported by
    ``arviz.InferenceData.to_zarr()``. Mirrors the approach taken by
    arviz>=1.0: ``idata.to_datatree().to_zarr(store)``.

    Parameters
    ----------
    idata : az.InferenceData
        The inference data to save.
    store : str or Path
        Path to the Zarr store directory.
    """
    idata.to_datatree().to_zarr(store)


_open_datatree_zarr = functools.partial(xr.open_datatree, engine="zarr")


def idata_from_zarr(store: str | Path) -> az.InferenceData:
    """Load an InferenceData from a Zarr store.

    TODO: Remove this shim once we require ``arviz>=1.0``.

    Counterpart to :func:`idata_to_zarr`. Works with zarr>=3. Mirrors the
    approach taken by arviz>=1.0, where ``from_zarr`` is
    ``functools.partial(open_datatree, engine="zarr")``.

    Parameters
    ----------
    store : str or Path
        Path to the Zarr store directory written by :func:`idata_to_zarr`.

    Returns
    -------
    az.InferenceData
    """
    return az.from_datatree(_open_datatree_zarr(store))


def filter_idata_by_dates(
    idata: az.InferenceData,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> az.InferenceData:
    """Filter InferenceData to a date range.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object to filter
    start_date : str or pd.Timestamp, optional
        Start date (inclusive)
    end_date : str or pd.Timestamp, optional
        End date (inclusive)

    Returns
    -------
    az.InferenceData
        New InferenceData with filtered groups

    Examples
    --------
    >>> filtered = filter_idata_by_dates(idata, "2024-01-01", "2024-12-31")
    """
    if start_date is None and end_date is None:
        return idata  # No filtering needed

    date_slice = {"date": slice(start_date, end_date)}

    filtered_groups = {}
    for group_name in idata.groups():
        group = getattr(idata, group_name)
        if "date" in group.dims:
            filtered_groups[group_name] = group.sel(**date_slice)
        else:
            filtered_groups[group_name] = group

    return az.InferenceData(**filtered_groups)


def filter_idata_by_dims(
    idata: az.InferenceData,
    **dim_filters,
) -> az.InferenceData:
    """Filter InferenceData by dimension values.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object to filter
    **dim_filters
        Dimension filters, e.g., country="US", channel=["TV", "Radio"]
        Note: When filtering to a single value, the dimension is dropped
        (xarray's default behavior). When filtering to multiple values,
        the dimension is preserved.

    Returns
    -------
    az.InferenceData
        New InferenceData with filtered groups

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

    # Validate that all dimensions exist in at least one group
    all_dims = set()
    for group_name in idata.groups():
        group = getattr(idata, group_name)
        all_dims.update(group.dims)

    for dim in dim_filters:
        if dim not in all_dims:
            raise ValueError(
                f"Dimension '{dim}' not found in any group. "
                f"Available dimensions: {sorted(all_dims)}"
            )

    # Filter all groups
    filtered_groups = {}
    for group_name in idata.groups():
        group = getattr(idata, group_name)

        # Build selection for this group's dimensions
        group_sel = {}
        for dim, values in dim_filters.items():
            if dim in group.dims:
                group_sel[dim] = values

        if group_sel:
            # Filter the group
            # Note: xarray's sel() drops dimensions when selecting a single value
            # This is the desired behavior - dimensions are dropped when filtered to single coordinate
            filtered = group.sel(**group_sel)
            filtered_groups[group_name] = filtered
        else:
            filtered_groups[group_name] = group

    return az.InferenceData(**filtered_groups)


def aggregate_idata_time(
    idata: az.InferenceData,
    period: Frequency,
    method: Literal["sum", "mean"] = "sum",
) -> az.InferenceData:
    """Aggregate InferenceData over time periods.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object to aggregate
    period : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
        Time period to aggregate to. Use "original" for no aggregation (returns
        unchanged), "all_time" to aggregate over the entire time dimension
        (removes the date dimension).
    method : {"sum", "mean"}, default "sum"
        Aggregation method

    Returns
    -------
    az.InferenceData
        New InferenceData with aggregated groups (or unchanged if period="original")

    Examples
    --------
    >>> original = aggregate_idata_time(idata, "original")  # No aggregation
    >>> monthly = aggregate_idata_time(idata, "monthly", method="sum")
    >>> total = aggregate_idata_time(idata, "all_time", method="sum")
    """
    # Handle "original" - no aggregation, return unchanged
    if period == "original":
        return idata

    # Handle "all_time" aggregation (removes date dimension entirely)
    if period == "all_time":
        aggregated_groups = {}
        for group_name in idata.groups():
            group = getattr(idata, group_name)

            if "date" not in group.dims:
                aggregated_groups[group_name] = group
                continue

            if method == "sum":
                aggregated = group.sum(dim="date")
            elif method == "mean":
                aggregated = group.mean(dim="date")
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

            aggregated_groups[group_name] = aggregated

        return az.InferenceData(**aggregated_groups)

    # Map period to pandas offset for periodic aggregation
    # Note: "ME", "QE", "YE" are the modern pandas 2.2+ aliases
    # (previously "M", "Q", "Y" which are now deprecated)
    period_map = {
        "weekly": "W",
        "monthly": "ME",
        "quarterly": "QE",
        "yearly": "YE",
    }
    freq = period_map[period]

    # Aggregate all groups with date dimension
    aggregated_groups = {}
    for group_name in idata.groups():
        group = getattr(idata, group_name)

        if "date" not in group.dims:
            aggregated_groups[group_name] = group
            continue

        if method == "sum":
            aggregated = group.resample(date=freq).sum(dim="date")
        elif method == "mean":
            aggregated = group.resample(date=freq).mean(dim="date")
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        aggregated_groups[group_name] = aggregated

    return az.InferenceData(**aggregated_groups)


def aggregate_idata_dims(
    idata: az.InferenceData,
    dim: str,
    values: list[str],
    new_label: str,
    method: Literal["sum", "mean"] = "sum",
) -> az.InferenceData:
    """Aggregate multiple dimension values into one.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object to aggregate
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
    az.InferenceData
        New InferenceData with aggregated dimension values

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
    # Validate that dimension exists in at least one group
    all_dims = set()
    for group_name in idata.groups():
        group = getattr(idata, group_name)
        all_dims.update(group.dims)

    if dim not in all_dims:
        raise ValueError(
            f"Dimension '{dim}' not found in any group. "
            f"Available dimensions: {sorted(all_dims)}"
        )

    # Pre-validate that new_label doesn't conflict with non-aggregated values
    # Check across all groups to fail fast with a clear error
    for group_name in idata.groups():
        group = getattr(idata, group_name)
        if dim in group.dims:
            existing_coords = set(group[dim].values)
            non_aggregated = existing_coords - set(values)
            if new_label in non_aggregated:
                raise ValueError(
                    f"new_label '{new_label}' conflicts with existing coordinate value "
                    f"in dimension '{dim}' that is not being aggregated. "
                    f"Existing values not being aggregated: {sorted(non_aggregated)}"
                )
            break  # Only need to check one group since coords are consistent

    aggregated_groups = {}
    for group_name in idata.groups():
        group = getattr(idata, group_name)

        if dim not in group.dims:
            aggregated_groups[group_name] = group
            continue

        # Get other values (not aggregated)
        all_coords = set(group[dim].values)
        other_values = list(all_coords - set(values))

        # Process variables individually to avoid adding dimension to variables
        # that don't have it (e.g., dayofyear with only 'date' dim when
        # aggregating 'country')
        result_vars = {}
        for var_name, var in group.data_vars.items():
            if dim not in var.dims:
                # Variable doesn't have the dimension - preserve as-is
                result_vars[var_name] = var
            else:
                # Variable has the dimension - aggregate it
                selected_var = var.sel({dim: values})

                if method == "sum":
                    aggregated_var = selected_var.sum(dim=dim)
                elif method == "mean":
                    aggregated_var = selected_var.mean(dim=dim)
                else:
                    raise ValueError(f"Unknown aggregation method: {method}")

                # Add the new label
                aggregated_var = aggregated_var.expand_dims({dim: [new_label]})

                if other_values:
                    other_var = var.sel({dim: other_values})
                    combined_var = xr.concat([other_var, aggregated_var], dim=dim)
                else:
                    combined_var = aggregated_var

                result_vars[var_name] = combined_var

        combined = xr.Dataset(result_vars)
        aggregated_groups[group_name] = combined

    return az.InferenceData(**aggregated_groups)


def get_posterior_predictive(idata: az.InferenceData) -> xr.Dataset:
    """Return the posterior_predictive group from *idata*.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object holding the fitted model results.

    Returns
    -------
    xr.Dataset
        The posterior_predictive group.

    Raises
    ------
    ValueError
        If posterior_predictive is absent from idata.
    """
    if not hasattr(idata, "posterior_predictive") or idata.posterior_predictive is None:
        raise ValueError(
            "No posterior_predictive data found in idata. "
            "Run MMM.sample_posterior_predictive() first."
        )
    return idata.posterior_predictive


def get_prior_predictive(idata: az.InferenceData) -> xr.Dataset:
    """Return the prior_predictive group from *idata*.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object holding the fitted model results.

    Returns
    -------
    xr.Dataset
        The prior_predictive group.

    Raises
    ------
    ValueError
        If prior_predictive is absent from idata.
    """
    if not hasattr(idata, "prior_predictive") or idata.prior_predictive is None:
        raise ValueError(
            "No prior_predictive data found in idata. "
            "Run MMM.sample_prior_predictive() first."
        )
    return idata.prior_predictive


def get_prior(idata: az.InferenceData) -> xr.Dataset:
    """Return the prior group from *idata*.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object holding the fitted model results.

    Returns
    -------
    xr.Dataset
        The prior group.

    Raises
    ------
    ValueError
        If prior is absent from idata.
    """
    if not hasattr(idata, "prior") or idata.prior is None:
        raise ValueError(
            "No prior data found in idata. Run MMM.sample_prior_predictive() first."
        )
    return idata.prior


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
