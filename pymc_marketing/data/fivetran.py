#   Copyright 2022 - 2025 The PyMC Labs Developers
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

    # Process ad spend data for media channels
    x = process_fivetran_ad_reporting(
        campaign_df, value_columns="spend", rename_date_to="date"
    )
    # Result: date | facebook_ads_spend | google_ads_spend | ...

    # Process conversion data (orders) as target variable
    y = process_fivetran_shopify_orders_unique_orders(orders_df)
    # Result: date | orders

    # Use in MMM model
    mmm = MMM(...)
    mmm.fit(X=x, y=y["orders"])
"""

from collections.abc import Sequence

import pandas as pd


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
    """Process Fivetran's Ad Reporting schema's.

    Compatible with Fivetran's Ad Reporting schema:
    - ad_reporting__account_report: Each record represents daily metrics by account
    - ad_reporting__campaign_report: Each record represents daily metrics by campaign and account
    - ad_reporting__ad_group_report: Each record represents daily metrics by ad group, campaign and account
    - ad_reporting__ad_report: Each record represents daily metrics by ad, ad group, campaign and account

    The input data is expected to contain at least the following columns: a date column
    (default: ``date_day``), a platform column (default: ``platform``), and one or more
    metric columns such as ``spend`` or ``impressions``.

    Parameters
    ----------
    df
        Input DataFrame in long format.
    value_columns
        A single column name or a sequence of column names to aggregate and pivot. For
        example: "spend" or ["spend", "impressions"].
    date_col
        Name of the date column. Defaults to "date_day".
    platform_col
        Name of the platform column. Defaults to "platform".
    agg
        Aggregation method to apply during groupby (e.g., "sum", "mean"). Defaults to "sum".
    fill_value
        Value to use to fill missing values in the wide output. If None, missing values
        are left as NaN.
    include_missing_dates
        If True, the output will include a continuous date index from the min to the max
        date found in the input, with missing dates filled (using ``fill_value``).
    freq
        Frequency used when ``include_missing_dates`` is True. Defaults to daily ("D").
    rename_date_to
        If provided, the date column in the result will be renamed to this value (e.g.,
        "date"). If None, the original ``date_col`` name is kept.

    Returns
    -------
    pd.DataFrame
        A wide-format DataFrame with one row per date and columns for each
        ``{platform}_{metric}`` combination.
    """
    if isinstance(value_columns, str):
        value_columns = [value_columns]
    else:
        value_columns = list(value_columns)

    required_columns: list[str] = [date_col, platform_col, *value_columns]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. Present columns: {list(df.columns)}"
        )

    # Ensure date type for robust reindexing and sorting
    df_local = df.copy()
    df_local[date_col] = pd.to_datetime(df_local[date_col])

    # Group and aggregate
    grouped = (
        df_local[[date_col, platform_col, *value_columns]]
        .groupby([date_col, platform_col], dropna=False)
        .agg(agg)
    )

    # Pivot to wide. Using values=value_columns yields MultiIndex columns (metric, platform)
    wide = grouped.reset_index().pivot(
        index=date_col, columns=platform_col, values=value_columns
    )

    # Flatten MultiIndex columns to "{platform}_{metric}"
    if isinstance(wide.columns, pd.MultiIndex):
        flat_cols = [f"{platform}_{metric}" for metric, platform in wide.columns]
    else:
        # Single metric case: columns are platforms; use the only metric name for suffix
        metric = value_columns[0]
        flat_cols = [f"{platform}_{metric}" for platform in wide.columns]
    wide.columns = flat_cols

    # Optionally include continuous date range
    if include_missing_dates:
        full_index = pd.date_range(
            start=wide.index.min(), end=wide.index.max(), freq=freq
        )
        wide = wide.reindex(full_index)

    # Fill missing values if requested
    if fill_value is not None:
        wide = wide.fillna(fill_value)

    # Output with date as a column, optionally renamed
    wide = wide.sort_index().reset_index().rename(columns={"index": date_col})
    if rename_date_to is not None and rename_date_to != date_col:
        wide = wide.rename(columns={date_col: rename_date_to})

    # Ensure date column is the first column
    first_col = rename_date_to if rename_date_to is not None else date_col
    ordered_cols = [first_col] + [c for c in wide.columns if c != first_col]
    return wide[ordered_cols]


def process_fivetran_shopify_orders_unique_orders(
    df: pd.DataFrame,
    *,
    date_col: str = "processed_timestamp",
    order_key_col: str = "orders_unique_key",
    rename_date_to: str = "date",
) -> pd.DataFrame:
    """Compute daily unique order counts from a (pre-filtered) Shopify orders dataset.

    This function is designed for data following the Fivetran Shopify orders schema
    (e.g., ``shopify__orders``). It assumes the input ``df`` is already filtered to
    the desired subset (e.g., non-canceled, US-delivery, new-only orders).

    Parameters
    ----------
    df
        Input DataFrame following the Shopify orders schema.
    date_col
        Column to derive the daily bucket from. Defaults to "processed_timestamp".
    order_key_col
        Unique order identifier column. Defaults to "orders_unique_key".
    rename_date_to
        Name of the date column in the result. Defaults to "date".

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns: ``rename_date_to`` and ``orders``, where
        ``orders`` is the unique order count per day.
    """
    # 1) Required columns
    missing = [c for c in (date_col, order_key_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. Present: {list(df.columns)}"
        )

    # 2) Minimal projection + robust datetime parsing
    tmp = df[[order_key_col, date_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])

    # 3) Daily bucket as date dtype
    tmp["_date"] = tmp[date_col].dt.date

    # 4) De-duplicate by (order, day) before counting
    tmp = tmp.drop_duplicates(subset=[order_key_col, "_date"])

    # 5) Count unique orders per day
    out = (
        tmp.groupby("_date", as_index=False)
        .agg(orders=(order_key_col, "nunique"))
        .rename(columns={"_date": rename_date_to})
        .sort_values(rename_date_to)
        .reset_index(drop=True)
    )
    return out
