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
"""
Fivetran data processing functions.

These functions help transform Fivetran's standardized schemas into formats suitable
for PyMC-Marketing models.

Example usage for MMM:

.. code-block:: python

    from pymc_marketing.data.fivetran import (
        process_fivetran_ad_reporting,
        process_fivetran_shopify_unique_orders,
    )
    from pymc_marketing.mmm import MMM

    # Process ad spend data for media channels
    x = process_fivetran_ad_reporting(
        campaign_df, value_columns="spend", rename_date_to="date"
    )
    # Result: date | facebook_ads_spend | google_ads_spend | ...

    # Process conversion data (orders) as target variable
    y = process_fivetran_shopify_unique_orders(orders_df)
    # Result: date | orders

    # Use in MMM model
    mmm = MMM(...)
    mmm.fit(X=x, y=y["orders"])

There are also pandas accessors for these functions which allows calling them from a
pandas DataFrame. These accessors are registered under the ``fivetran`` namespace and
can be accessed after importing pymc_marketing.

.. code-block:: python

    import pandas as pd

    from pymc_marketing.mmm import MMM


    campaign_df: pd.DataFrame = ...
    orders_df: pd.DataFrame = ...

    X: pd.DataFrame = campaign_df.fivetran.process_ad_reporting(value_columns="spend")
    y: pd.DataFrame = orders_df.fivetran.process_shopify_unique_orders()

    # Use in MMM model
    mmm = MMM(...)
    mmm.fit(X=x, y=y["orders"])

"""

from collections.abc import Sequence

import narwhals as nw
import pandas as pd
from narwhals.typing import IntoFrameT

from pymc_marketing.decorators import copy_docstring


def _normalize_and_validate_inputs(
    df: IntoFrameT,
    value_columns: str | Sequence[str],
    date_col: str,
    platform_col: str,
) -> tuple[IntoFrameT, list[str]]:
    """Validate required columns, coerce date, and normalize metrics list.

    Parameters
    ----------
    df : DataFrame-like
        Input dataframe to validate and normalize.
    value_columns : str or Sequence[str]
        Metric column(s) to process.
    date_col : str
        Name of the date column.
    platform_col : str
        Name of the platform column.

    Returns
    -------
    tuple[narwhals.DataFrame, list[str]]
        - A narwhals DataFrame with ``date_col`` coerced to datetime.
        - A normalized list of metric column names.

    Raises
    ------
    ValueError
        If any of the required columns are missing.
    """
    # Convert to narwhals (works with pandas, polars, pyarrow, etc.)
    nw_df = nw.from_native(df)

    # Normalize metrics list
    metrics = [value_columns] if isinstance(value_columns, str) else list(value_columns)

    # Validate columns
    required_columns: list[str] = [date_col, platform_col, *metrics]
    missing_columns = [c for c in required_columns if c not in nw_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. Present columns: {list(nw_df.columns)}"
        )

    # Cast date column to datetime
    df_local = nw_df.with_columns(nw.col(date_col).cast(nw.Datetime))

    return df_local, metrics


def _aggregate_and_pivot(
    df: IntoFrameT,
    date_col: str,
    platform_col: str,
    value_columns: list[str],
    agg: str,
) -> IntoFrameT:
    """Aggregate metrics by date and platform, then pivot and flatten columns.

    Parameters
    ----------
    df : narwhals.DataFrame
        Input dataframe with coerced date column.
    date_col : str
        Date column name.
    platform_col : str
        Platform column name.
    value_columns : list[str]
        Metric columns to aggregate and pivot.
    agg : str
        Aggregation function passed to pandas ``agg``.

    Returns
    -------
    narwhals.DataFrame
        Wide-format dataframe indexed by date with columns named ``{platform}_{metric}``.
    """
    # Group and aggregate
    # Map aggregation string to narwhals aggregation method
    agg_map = {
        "sum": lambda c: nw.col(c).sum(),
        "mean": lambda c: nw.col(c).mean(),
        "min": lambda c: nw.col(c).min(),
        "max": lambda c: nw.col(c).max(),
        "count": lambda c: nw.col(c).count(),
    }
    if agg not in agg_map:
        raise ValueError(
            f"Unsupported aggregation: {agg}. Supported: {list(agg_map.keys())}"
        )

    agg_fn = agg_map[agg]
    agg_exprs = [agg_fn(c) for c in value_columns]
    grouped = (
        df.select([date_col, platform_col, *value_columns])
        .group_by([date_col, platform_col], drop_null_keys=False)
        .agg(agg_exprs)
    )

    # Pivot operation
    wide = grouped.pivot(
        on=platform_col,
        index=date_col,
        values=value_columns if len(value_columns) > 1 else value_columns[0],
        aggregate_function=None,  # Already aggregated
        sort_columns=False,
    )

    # Narwhals pivot flattens MultiIndex automatically
    # Rename columns to ensure {platform}_{metric} format
    # For single metric: columns are `platform_name`, need to add metric suffix
    # For multiple metrics: columns are `metric_platform`, need to swap to `platform_metric`

    if len(value_columns) == 1:
        # Single metric case: columns are platform names, need to add metric suffix
        metric = value_columns[0]
        new_cols = {col: f"{col}_{metric}" for col in wide.columns if col != date_col}
        wide = wide.rename(new_cols)
    else:
        # Multiple metrics case: columns are {metric}_{platform}, need to swap to {platform}_{metric}
        # Strategy: for each metric, find columns starting with that metric prefix and rename
        new_cols = {}
        for col in wide.columns:
            if col == date_col:
                continue
            # Try each metric to see if the column starts with it
            for metric in value_columns:
                if col.startswith(f"{metric}_"):
                    # Extract the platform part after the metric
                    platform = col[len(metric) + 1 :]  # Skip metric and underscore
                    new_cols[col] = f"{platform}_{metric}"
                    break
        wide = wide.rename(new_cols)

    return wide


def _finalize_wide_output(
    wide: IntoFrameT,
    *,
    date_col: str,
    rename_date_to: str | None,
    include_missing_dates: bool,
    freq: str,
    fill_value: float | None,
) -> IntoFrameT:
    """Complete wide output: fill dates, fill values, reset, rename, standardize.

    Parameters
    ----------
    wide : narwhals.DataFrame
        Dataframe with date column.
    date_col : str
        Name of the date column in the input.
    rename_date_to : str or None
        Optional new name for the date column.
    include_missing_dates : bool
        If ``True``, reindex to a continuous date range using ``freq``.
    freq : str
        Frequency for ``pandas.date_range`` when including missing dates.
    fill_value : float or None
        Value used to fill missing values. If ``None``, do not fill.

    Returns
    -------
    narwhals.DataFrame
        Finalized wide-format dataframe with standardized columns and first column as the date.
    """
    if include_missing_dates:
        # FBruzzesi's workaround for reindex: use join with full date range
        # Get min/max dates (may trigger computation for lazy frames)
        min_date = wide.select(nw.col(date_col).min()).item()
        max_date = wide.select(nw.col(date_col).max()).item()

        # Generate full date range using pandas (temporary)
        full_dates_pd = pd.DataFrame(
            {date_col: pd.date_range(start=min_date, end=max_date, freq=freq)}
        )

        # Convert to narwhals using same backend as input
        full_dates = nw.from_native(full_dates_pd)

        # Left join: keep all dates from full range
        wide = full_dates.join(wide, on=date_col, how="left")

    # Fill nulls if specified
    if fill_value is not None:
        # Fill null values in all columns except the date column
        cols_to_fill = [c for c in wide.columns if c != date_col]
        fill_exprs = [nw.col(c).fill_null(fill_value).alias(c) for c in cols_to_fill]
        wide = wide.with_columns(fill_exprs)

    # Sort by date
    wide = wide.sort(date_col)

    # Rename date column if needed
    if rename_date_to is not None and rename_date_to != date_col:
        wide = wide.rename({date_col: rename_date_to})

    # Standardize column names (lowercase, replace spaces)
    current_cols = wide.columns
    new_cols = {col: col.lower().replace(" ", "_") for col in current_cols}
    wide = wide.rename(new_cols)

    # Normalize dates to midnight using dt.truncate (FBruzzesi's suggestion!)
    first_col = (
        (rename_date_to if rename_date_to is not None else date_col)
        .lower()
        .replace(" ", "_")
    )
    wide = wide.with_columns(nw.col(first_col).dt.truncate(every="1d"))

    # Ensure date column is first
    ordered_cols = [first_col] + [c for c in wide.columns if c != first_col]
    wide = wide.select(ordered_cols)

    return wide


def process_fivetran_ad_reporting(
    df: pd.DataFrame,
    value_columns: str | Sequence[str] = "impressions",
    *,
    date_col: str = "date_day",
    platform_col: str = "platform",
    agg: str = "sum",
    fill_value: float | None = 0.0,
    include_missing_dates: bool = False,
    freq: str = "D",
    rename_date_to: str | None = "date",
) -> pd.DataFrame:
    """Process Fivetran Ad Reporting tables into wide, model-ready features.

    Compatible with Fivetran's Ad Reporting schema tables:

    - ad_reporting__account_report: daily metrics by account
    - ad_reporting__campaign_report: daily metrics by campaign and account
    - ad_reporting__ad_group_report: daily metrics by ad group, campaign and account
    - ad_reporting__ad_report: daily metrics by ad, ad group, campaign and account

    The input data must include a date column, a platform column (e.g., vendor name),
    and one or more metric columns such as ``spend`` or ``impressions``. The output is
    a wide dataframe with one row per date and columns named ``{platform}_{metric}``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe in long format with at least the date, platform, and metric columns.
    value_columns : str or Sequence[str], default "impressions"
        Column name(s) to aggregate and pivot. Example: "spend" or ["spend", "impressions"].
    date_col : str, default "date_day"
        Name of the date column.
    platform_col : str, default "platform"
        Name of the platform (vendor) column.
    agg : str, default "sum"
        Aggregation method applied during groupby.
    fill_value : float or None, default 0.0
        Value used to fill missing values in the wide output. If ``None``, missing values are left as NaN.
    include_missing_dates : bool, default False
        If ``True``, include a continuous date range and fill missing dates using ``fill_value``.
    freq : str, default "D"
        Frequency used when ``include_missing_dates`` is ``True``.
    rename_date_to : str or None, default "date"
        If provided, rename the date column in the result to this value. If ``None``, keep ``date_col``.

    Returns
    -------
    pandas.DataFrame
        A wide-format dataframe with one row per date and columns for each
        ``{platform}_{metric}`` combination.
    """
    df_local, metrics = _normalize_and_validate_inputs(
        df=df,
        value_columns=value_columns,
        date_col=date_col,
        platform_col=platform_col,
    )

    wide = _aggregate_and_pivot(
        df_local,
        date_col=date_col,
        platform_col=platform_col,
        value_columns=metrics,
        agg=agg,
    )

    result = _finalize_wide_output(
        wide,
        date_col=date_col,
        rename_date_to=rename_date_to,
        include_missing_dates=include_missing_dates,
        freq=freq,
        fill_value=fill_value,
    )

    # Convert back to native format (pandas in this case)
    result_native = nw.to_native(result)

    # Fix column names: narwhals pivot sets columns.names to [""] but pandas expects [None]
    if hasattr(result_native.columns, "names"):
        result_native.columns.names = [None]

    return result_native


def process_fivetran_shopify_unique_orders(
    df: pd.DataFrame,
    *,
    date_col: str = "processed_timestamp",
    order_key_col: str = "orders_unique_key",
    rename_date_to: str = "date",
) -> pd.DataFrame:
    """Compute daily unique order counts from a (pre-filtered) Shopify dataset.

    This function targets data following the Fivetran Shopify orders schema
    (e.g., ``shopify__orders``). It assumes the input ``df`` is already filtered to
    the desired subset (e.g., non-canceled, US-delivery, new-only orders).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe following the Shopify orders schema.
    date_col : str, default "processed_timestamp"
        Timestamp column from which the daily bucket is derived.
    order_key_col : str, default "orders_unique_key"
        Unique order identifier column.
    rename_date_to : str, default "date"
        Name of the date column in the result.

    Returns
    -------
    pandas.DataFrame
        A dataframe with two columns: ``rename_date_to`` and ``orders``, where
        ``orders`` is the unique order count per day.
    """
    # Convert to narwhals after coercing datetime (narwhals doesn't support errors="coerce")
    # Coerce invalid dates to NaT at pandas level first
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    nw_df = nw.from_native(df_copy)

    # 1) Required columns
    missing = [c for c in (date_col, order_key_col) if c not in nw_df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. Present: {list(nw_df.columns)}"
        )

    # 2) Minimal projection (datetime already coerced above)
    tmp = nw_df.select([order_key_col, date_col])

    # 3) Daily bucket normalized to midnight using dt.truncate
    tmp = tmp.with_columns(_date=nw.col(date_col).dt.truncate(every="1d"))

    # Filter out nulls (equivalent to dropna)
    tmp = tmp.filter(~nw.col("_date").is_null())

    # 4) De-duplicate by (order, day) before counting
    tmp = tmp.unique(subset=[order_key_col, "_date"])

    # 5) Count unique orders per day
    out = (
        tmp.group_by("_date")
        .agg(orders=nw.col(order_key_col).n_unique())
        .rename({"_date": rename_date_to})
        .sort(rename_date_to)
    )

    # Convert back to native format (pandas)
    return nw.to_native(out)


@pd.api.extensions.register_dataframe_accessor("fivetran")
class FivetranAccessor:
    """Accessor for Fivetran data processing functions."""

    def __init__(self, obj: pd.DataFrame) -> None:
        self._obj = obj

    @copy_docstring(process_fivetran_ad_reporting)
    def process_ad_reporting(  # noqa: D102
        self,
        value_columns: str | Sequence[str] = "impressions",
        *,
        date_col: str = "date_day",
        platform_col: str = "platform",
        agg: str = "sum",
        fill_value: float | None = 0.0,
        include_missing_dates: bool = False,
        freq: str = "D",
        rename_date_to: str | None = "date",
    ) -> pd.DataFrame:
        return process_fivetran_ad_reporting(
            self._obj,
            value_columns=value_columns,
            date_col=date_col,
            platform_col=platform_col,
            agg=agg,
            fill_value=fill_value,
            include_missing_dates=include_missing_dates,
            freq=freq,
            rename_date_to=rename_date_to,
        )

    @copy_docstring(process_fivetran_shopify_unique_orders)
    def process_shopify_unique_orders(  # noqa: D102
        self,
        *,
        date_col: str = "processed_timestamp",
        order_key_col: str = "orders_unique_key",
        rename_date_to: str = "date",
    ) -> pd.DataFrame:
        return process_fivetran_shopify_unique_orders(
            self._obj,
            date_col=date_col,
            order_key_col=order_key_col,
            rename_date_to=rename_date_to,
        )
