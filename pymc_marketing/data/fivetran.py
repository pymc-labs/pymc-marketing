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

import pandas as pd

from pymc_marketing.decorators import copy_docstring


def _normalize_and_validate_inputs(
    df: pd.DataFrame,
    value_columns: str | Sequence[str],
    date_col: str,
    platform_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate required columns, coerce date, and normalize metrics list.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to validate and normalize.
    value_columns : str or Sequence[str]
        Metric column(s) to process.
    date_col : str
        Name of the date column.
    platform_col : str
        Name of the platform column.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        - A copy of ``df`` with ``date_col`` coerced to datetime64[ns].
        - A normalized list of metric column names.

    Raises
    ------
    ValueError
        If any of the required columns are missing.
    """
    metrics = [value_columns] if isinstance(value_columns, str) else list(value_columns)

    required_columns: list[str] = [date_col, platform_col, *metrics]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. Present columns: {list(df.columns)}"
        )

    df_local = df.copy()
    df_local[date_col] = pd.to_datetime(df_local[date_col])
    return df_local, metrics


def _aggregate_and_pivot(
    df: pd.DataFrame,
    date_col: str,
    platform_col: str,
    value_columns: list[str],
    agg: str,
) -> pd.DataFrame:
    """Aggregate metrics by date and platform, then pivot and flatten columns.

    Parameters
    ----------
    df : pandas.DataFrame
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
    pandas.DataFrame
        Wide-format dataframe indexed by date with columns named ``{platform}_{metric}``.
    """
    grouped = (
        df[[date_col, platform_col, *value_columns]]
        .groupby([date_col, platform_col], dropna=False)
        .agg(agg)
    )

    wide = grouped.reset_index().pivot(
        index=date_col, columns=platform_col, values=value_columns
    )

    if isinstance(wide.columns, pd.MultiIndex):
        flat_cols = [f"{platform}_{metric}" for metric, platform in wide.columns]
    else:
        metric = value_columns[0]
        flat_cols = [f"{platform}_{metric}" for platform in wide.columns]
    wide.columns = flat_cols
    return wide


def _finalize_wide_output(
    wide: pd.DataFrame,
    *,
    date_col: str,
    rename_date_to: str | None,
    include_missing_dates: bool,
    freq: str,
    fill_value: float | None,
) -> pd.DataFrame:
    """Complete wide output: fill dates, fill values, reset, rename, standardize.

    Parameters
    ----------
    wide : pandas.DataFrame
        Dataframe indexed by date.
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
    pandas.DataFrame
        Finalized wide-format dataframe with standardized columns and first column as the date.
    """
    if include_missing_dates:
        full_index = pd.date_range(
            start=wide.index.min(), end=wide.index.max(), freq=freq
        )
        wide = wide.reindex(full_index)

    if fill_value is not None:
        wide = wide.fillna(fill_value)

    wide = wide.sort_index()
    wide.index.name = date_col
    wide = wide.reset_index()

    if rename_date_to is not None and rename_date_to != date_col:
        wide = wide.rename(columns={date_col: rename_date_to})

    # standardize column names
    wide.columns = [col.lower().replace(" ", "_") for col in wide.columns]

    # Ensure date column is first and normalized to midnight
    first_col = (
        (rename_date_to if rename_date_to is not None else date_col)
        .lower()
        .replace(" ", "_")
    )
    ordered_cols = [first_col] + [c for c in wide.columns if c != first_col]
    wide[first_col] = pd.to_datetime(wide[first_col]).dt.normalize()
    return wide[ordered_cols]


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

    return _finalize_wide_output(
        wide,
        date_col=date_col,
        rename_date_to=rename_date_to,
        include_missing_dates=include_missing_dates,
        freq=freq,
        fill_value=fill_value,
    )


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

    # 3) Daily bucket normalized to midnight, preserving datetime64[ns] dtype
    tmp["_date"] = tmp[date_col].dt.normalize()

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
