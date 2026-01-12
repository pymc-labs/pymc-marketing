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

from typing import Literal

import arviz as az
import pandas as pd
import xarray as xr


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
    period: Literal["weekly", "monthly", "quarterly", "yearly", "all_time"],
    method: Literal["sum", "mean"] = "sum",
) -> az.InferenceData:
    """Aggregate InferenceData over time periods.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object to aggregate
    period : {"weekly", "monthly", "quarterly", "yearly", "all_time"}
        Time period to aggregate to. Use "all_time" to aggregate over
        the entire time dimension (removes the date dimension).
    method : {"sum", "mean"}, default "sum"
        Aggregation method

    Returns
    -------
    az.InferenceData
        New InferenceData with aggregated groups

    Examples
    --------
    >>> monthly = aggregate_idata_time(idata, "monthly", method="sum")
    >>> total = aggregate_idata_time(idata, "all_time", method="sum")
    """
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
    period_map = {
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
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
        If the dimension doesn't exist in any group

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

    aggregated_groups = {}
    for group_name in idata.groups():
        group = getattr(idata, group_name)

        if dim not in group.dims:
            aggregated_groups[group_name] = group
            continue

        # Select values to aggregate
        selected = group.sel({dim: values})

        # Aggregate
        if method == "sum":
            aggregated_values = selected.sum(dim=dim)
        elif method == "mean":
            aggregated_values = selected.mean(dim=dim)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Get other values (not aggregated)
        all_coords = set(group[dim].values)
        other_values = list(all_coords - set(values))

        if other_values:
            other_data = group.sel({dim: other_values})

            # Assign new label to aggregated
            aggregated_values = aggregated_values.expand_dims({dim: [new_label]})

            # Concatenate
            combined = xr.concat([other_data, aggregated_values], dim=dim)
        else:
            # All values aggregated
            combined = aggregated_values.expand_dims({dim: [new_label]})

        aggregated_groups[group_name] = combined

    return az.InferenceData(**aggregated_groups)
