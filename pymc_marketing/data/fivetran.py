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

**Multi-backend support**: All functions accept pandas DataFrames, polars DataFrames
(eager and lazy), and PySpark DataFrames. The output type matches the input type.

Example usage for MMM:

.. code-block:: python

    from pymc_marketing.data.fivetran import (
        process_fivetran_ad_reporting,
        process_fivetran_shopify_unique_orders,
    )
    from pymc_marketing.mmm import MMM

    # Works with pandas DataFrames
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

Example usage with polars:

.. code-block:: python

    import polars as pl
    from pymc_marketing.data.fivetran import process_fivetran_ad_reporting

    # Works with polars DataFrames (returns polars DataFrame)
    campaign_pl = pl.read_parquet("campaigns.parquet")
    x_pl = process_fivetran_ad_reporting(campaign_pl, value_columns="spend")

Example usage with PySpark (Databricks):

.. code-block:: python

    from pyspark.sql import SparkSession
    from pymc_marketing.data.fivetran import process_fivetran_ad_reporting

    spark = SparkSession.builder.getOrCreate()

    # Works with PySpark DataFrames (returns PySpark DataFrame)
    campaign_spark = spark.read.table("campaigns")
    x_spark = process_fivetran_ad_reporting(campaign_spark, value_columns="spend")

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
from typing import Any

import narwhals as nw
import pandas as pd
from narwhals.typing import IntoFrameT

from pymc_marketing.decorators import copy_docstring


def _is_pyspark(df: Any) -> bool:
    """Check if a DataFrame is PySpark.

    Parameters
    ----------
    df : Any
        DataFrame to check (narwhals or native).

    Returns
    -------
    bool
        True if the DataFrame is PySpark, False otherwise.
    """
    # If it's a narwhals frame, get the native object
    if hasattr(df, "__narwhals_dataframe__") or hasattr(df, "__narwhals_lazyframe__"):
        native = nw.to_native(df)
    else:
        native = df

    # Check if it's a PySpark DataFrame
    return hasattr(native, "sparkSession")


def _pivot_pyspark_native(
    df: Any,
    date_col: str,
    platform_col: str,
    value_columns: list[str],
    agg: str,
) -> Any:
    """Pivot PySpark DataFrame using native PySpark operations.

    TEMPORARY: This function provides PySpark pivot support until narwhals adds
    LazyFrame.pivot support. See: https://github.com/narwhals-dev/narwhals/issues/1901

    Parameters
    ----------
    df : narwhals.DataFrame (backed by PySpark)
        Input dataframe with coerced date column.
    date_col : str
        Date column name.
    platform_col : str
        Platform column name.
    value_columns : list[str]
        Metric columns to aggregate and pivot.
    agg : str
        Aggregation function ('sum', 'mean', 'min', 'max', 'count').

    Returns
    -------
    narwhals.DataFrame (backed by PySpark)
        Wide-format dataframe with columns named {platform}_{metric}.

    Raises
    ------
    ValueError
        If aggregation function is not supported.
    ImportError
        If pyspark is not installed.
    """
    try:
        from pyspark.sql import functions as F
    except ImportError as e:
        raise ImportError(
            "PySpark is required for processing PySpark DataFrames. "
            "Install it with: pip install pyspark"
        ) from e

    # Get native PySpark DataFrame
    spark_df = nw.to_native(df)

    # Map aggregation string to PySpark aggregation function
    agg_map = {
        "sum": F.sum,
        "mean": F.mean,
        "min": F.min,
        "max": F.max,
        "count": F.count,
    }
    if agg not in agg_map:
        raise ValueError(
            f"Unsupported aggregation: {agg}. Supported: {list(agg_map.keys())}"
        )

    pyspark_agg_fn = agg_map[agg]

    # For each metric, aggregate and pivot
    # PySpark pivot syntax: groupBy(index).pivot(on).agg(values)
    pivot_exprs = {col: pyspark_agg_fn(col) for col in value_columns}

    pivoted = (
        spark_df.select([date_col, platform_col, *value_columns])
        .groupBy(date_col, platform_col)
        .agg(pivot_exprs)
        .groupBy(date_col)
        .pivot(platform_col)
        .agg(pivot_exprs)
    )

    # Flatten column names from PySpark's default naming
    # PySpark creates columns like: platform_value_sum(metric)
    # We want: platform_metric
    new_columns = []
    for col_name in pivoted.columns:
        if col_name == date_col:
            new_columns.append(col_name)
            continue

        # Parse PySpark's default pivot column naming
        # Format: {platform}_{agg_func}({metric})
        # Example: "facebook_sum(spend)" -> "facebook_spend"
        if "_" in col_name and "(" in col_name:
            parts = col_name.split("_", 1)  # Split on first underscore
            platform = parts[0]
            # Extract metric from agg_func(metric)
            agg_and_metric = parts[1]
            if "(" in agg_and_metric and ")" in agg_and_metric:
                metric = agg_and_metric.split("(")[1].split(")")[0]
                new_columns.append(f"{platform}_{metric}")
            else:
                new_columns.append(col_name)
        else:
            new_columns.append(col_name)

    # Rename columns
    pivoted = pivoted.toDF(*new_columns)

    # Convert back to narwhals
    return nw.from_native(pivoted)


def _is_polars_lazy(df: Any) -> bool:
    """Check if a DataFrame is Polars LazyFrame.

    Detects Polars LazyFrame by checking for the presence of the .collect() method,
    which is only available on lazy frames (not eager frames).

    Parameters
    ----------
    df : Any
        DataFrame to check (narwhals or native).

    Returns
    -------
    bool
        True if the DataFrame is Polars LazyFrame, False otherwise.

    Notes
    -----
    This detection is necessary because narwhals wraps LazyFrame but doesn't
    expose .pivot() method yet. See: https://github.com/narwhals-dev/narwhals/issues/1901

    Detection logic: narwhals LazyFrames have callable .collect() method,
    while eager DataFrames do not.
    """
    # Check if frame has a callable collect method (only LazyFrames have this)
    return callable(getattr(df, "collect", None))


def _format_pivot_columns(
    wide: Any,  # narwhals DataFrame
    date_col: str,
    value_columns: list[str],
) -> Any:
    """Format pivot result columns to {platform}_{metric} naming convention.

    This helper function standardizes column naming after pivot operations,
    regardless of the backend (pandas, polars, PySpark). It can be reused
    when narwhals adds unified LazyFrame.pivot support.

    Parameters
    ----------
    wide : narwhals DataFrame
        Pivoted DataFrame (wide format) with platform-specific columns.
    date_col : str
        Name of the date column (not renamed).
    value_columns : list[str]
        List of metric column names that were pivoted.

    Returns
    -------
    narwhals DataFrame
        DataFrame with standardized column names: {platform}_{metric}

    Notes
    -----
    Handles two cases:
    - Single metric: columns are platform names → add metric suffix
    - Multiple metrics: columns are {metric}_{platform} → swap to {platform}_{metric}

    Migration path: When narwhals adds LazyFrame.pivot() support (issue #1901),
    this helper can be reused with the unified pivot path, making the transition
    from backend-specific handlers seamless.
    """
    if len(value_columns) == 1:
        # Single metric case: columns are platform names, add metric suffix
        metric = value_columns[0]
        new_cols = {col: f"{col}_{metric}" for col in wide.columns if col != date_col}
        return wide.rename(new_cols)

    # Multiple metrics case: swap {metric}_{platform} to {platform}_{metric}
    new_cols = {}
    for col in wide.columns:
        if col == date_col:
            continue
        for metric in value_columns:
            if col.startswith(metric + "_"):
                platform = col[len(metric) + 1 :]
                new_cols[col] = f"{platform}_{metric}"
                break
    if new_cols:
        return wide.rename(new_cols)
    return wide


def _create_date_range(
    frame: Any,  # narwhals DataFrame
    date_col: str,
    freq: str,
) -> Any:
    """Create a date range DataFrame matching the backend of the input frame.

    This function ensures backend alignment when creating full date ranges
    for reindexing operations. It branches based on the DataFrame backend:
    - Polars: Uses polars.date_range()
    - Pandas/PyArrow: Uses pandas.date_range()

    TEMPORARY: This is a workaround until narwhals adds date_range() support.
    See: https://github.com/narwhals-dev/narwhals/discussions/3425
          https://github.com/narwhals-dev/narwhals/issues/2193

    Parameters
    ----------
    frame : narwhals DataFrame
        Input DataFrame whose backend will be matched.
    date_col : str
        Name of the date column.
    freq : str
        Frequency string (e.g., "1d", "1h"). Uses pandas convention.

    Returns
    -------
    narwhals DataFrame
        DataFrame with single date column, backed by same implementation as input.

    Notes
    -----
    The last line uses `.lazy().collect(backend=impl)` to guarantee backend
    alignment for PyArrow (pandas → pyarrow remapping).

    This implementation is based on FBruzzesi's suggestion for handling backend
    alignment when creating date ranges across different DataFrame backends.

    References
    ----------
    - FBruzzesi's suggestion: https://github.com/pymc-labs/pymc-marketing/pull/2224#discussion_r2741140644
    - narwhals date_range discussion: https://github.com/narwhals-dev/narwhals/discussions/3425
    - narwhals backend alignment: https://github.com/narwhals-dev/narwhals/issues/2193
    """
    date_series = frame.get_column(date_col)
    min_date, max_date = date_series.min(), date_series.max()

    impl = frame.implementation
    if impl.is_polars():
        import polars as pl

        # Polars date_range creates datetime by default when given datetime inputs
        # The time_unit parameter ensures we match the input datetime precision
        dates = pl.date_range(start=min_date, end=max_date, interval=freq, eager=True)
        # Ensure it's datetime type (polars date_range may return Date if inputs are Date)
        # We need to cast to datetime to match the input column type
        dates_series = nw.from_native(dates, series_only=True)
        if dates_series.dtype != nw.Datetime:
            # Convert date to datetime at midnight
            dates_series = dates_series.cast(nw.Datetime)
        dates_frame = dates_series.rename(date_col).to_frame()
    else:  # Fallback to pandas (PyArrow doesn't have date_range)
        dates = pd.date_range(start=min_date, end=max_date, freq=freq)
        dates_frame = (
            nw.from_native(dates, series_only=True).rename(date_col).to_frame()
        )

    # Guarantees backend alignment (remaps pandas → pyarrow if needed)
    return dates_frame.lazy().collect(backend=impl)


def _pivot_polars_lazy(
    df: Any,  # narwhals LazyFrame
    date_col: str,
    platform_col: str,
    value_columns: list[str],
    agg: str,
) -> Any:
    """Handle pivot for Polars LazyFrame with lazy execution preservation.

    This function minimizes eager collection by keeping data in lazy mode until
    the pivot operation (which isn't supported on LazyFrames in narwhals yet),
    then converting back to LazyFrame for type preservation.

    TEMPORARY: This is a workaround until narwhals adds LazyFrame.pivot support.
    Once narwhals #1901 is resolved, this function can be removed in favor of
    the unified pivot path.
    See: https://github.com/narwhals-dev/narwhals/issues/1901

    Parameters
    ----------
    df : narwhals LazyFrame
        Input LazyFrame with coerced date column.
    date_col : str
        Name of the date column.
    platform_col : str
        Name of the platform column to pivot on.
    value_columns : list[str]
        Metrics to aggregate and pivot.
    agg : str
        Aggregation function: "sum", "mean", "min", "max", or "count".

    Returns
    -------
    narwhals LazyFrame
        Pivoted data returned as LazyFrame (type preservation).

    Notes
    -----
    Migration path: When narwhals adds LazyFrame.pivot(), replace this function's
    usage with the unified pivot path in _aggregate_and_pivot() and rely on
    _format_pivot_columns() for column naming.
    """
    # Map aggregation string to narwhals aggregation method
    agg_map = {
        "sum": nw.sum,
        "mean": nw.mean,
        "min": nw.min,
        "max": nw.max,
        "count": lambda c: nw.col(c).count(),
    }
    if agg not in agg_map:
        raise ValueError(
            f"Unsupported aggregation: {agg}. Supported: {list(agg_map.keys())}"
        )

    agg_fn = agg_map[agg]
    agg_exprs = agg_fn(value_columns)

    # Keep in lazy mode as long as possible
    grouped_lazy = df.group_by([date_col, platform_col], drop_null_keys=False).agg(
        agg_exprs
    )

    # Must collect here because narwhals LazyFrame doesn't support .pivot() yet
    grouped_eager = grouped_lazy.collect()

    # Pivot operation (now on eager DataFrame)
    wide = grouped_eager.pivot(
        on=platform_col,
        index=date_col,
        values=value_columns if len(value_columns) > 1 else value_columns[0],
        aggregate_function=None,  # Already aggregated
        sort_columns=False,
    )

    # Format columns using extracted helper (easy to reuse when narwhals #1901 resolved)
    wide = _format_pivot_columns(wide, date_col, value_columns)

    # Convert back to LazyFrame to preserve input type
    result_native = nw.to_native(wide)
    result_lazy = result_native.lazy()  # Polars DataFrame -> LazyFrame

    return nw.from_native(result_lazy)


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

    This function handles multiple DataFrame backends with conditional logic:
    - PySpark: Uses native PySpark pivot (narwhals LazyFrame doesn't support pivot yet)
    - Polars LazyFrame: Keeps data lazy until pivot, collects for pivot operation,
      returns as LazyFrame for type preservation
    - pandas/polars eager: Uses narwhals pivot (works on eager DataFrames)

    Column formatting is handled by _format_pivot_columns() helper, which can be
    reused when narwhals adds unified LazyFrame.pivot support.

    TEMPORARY: Once narwhals #1901 is resolved, _pivot_pyspark_native() and
    _pivot_polars_lazy() can be removed in favor of unified df.pivot() path.
    See: https://github.com/narwhals-dev/narwhals/issues/1901

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
        Aggregation function ('sum', 'mean', 'min', 'max', 'count').

    Returns
    -------
    narwhals.DataFrame
        Wide-format dataframe indexed by date with columns named ``{platform}_{metric}``.
    """
    # Check if we're dealing with PySpark
    if _is_pyspark(df):
        # Use native PySpark pivot implementation
        return _pivot_pyspark_native(df, date_col, platform_col, value_columns, agg)

    # Check if we're dealing with Polars LazyFrame
    if _is_polars_lazy(df):
        # Use Polars LazyFrame-specific pivot implementation
        return _pivot_polars_lazy(df, date_col, platform_col, value_columns, agg)

    # For pandas/polars eager: use narwhals pivot (works on eager DataFrames)
    # Map aggregation string to narwhals aggregation method
    # Use top-level narwhals functions where available (FBruzzesi's suggestion)
    agg_map = {
        "sum": nw.sum,
        "mean": nw.mean,
        "min": nw.min,
        "max": nw.max,
        "count": lambda c: nw.col(c).count(),  # No top-level nw.count available
    }
    if agg not in agg_map:
        raise ValueError(
            f"Unsupported aggregation: {agg}. Supported: {list(agg_map.keys())}"
        )

    agg_fn = agg_map[agg]
    agg_exprs = agg_fn(value_columns)
    # Removed .select() to avoid eager evaluation (FBruzzesi's suggestion)
    grouped = df.group_by([date_col, platform_col], drop_null_keys=False).agg(agg_exprs)

    # Pivot operation (narwhals eager DataFrame only)
    wide = grouped.pivot(
        on=platform_col,
        index=date_col,
        values=value_columns if len(value_columns) > 1 else value_columns[0],
        aggregate_function=None,  # Already aggregated
        sort_columns=False,
    )

    # Format columns using extracted helper
    # This makes it easy to adopt narwhals unified pivot when issue #1901 is resolved
    wide = _format_pivot_columns(wide, date_col, value_columns)

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
        # Create date range matching backend of input DataFrame
        # Uses backend-specific date_range (polars or pandas) to ensure alignment
        # FBruzzesi's suggestion: https://github.com/pymc-labs/pymc-marketing/pull/2224#discussion_r2741140644
        full_dates = _create_date_range(wide, date_col, freq)

        # Left join: keep all dates from full range
        wide = full_dates.join(wide, on=date_col, how="left")

    # Fill nulls if specified
    if fill_value is not None:
        # Fill null values in all columns except the date column
        # Use nw.exclude() for cleaner polars-style syntax (FBruzzesi's suggestion)
        fill_expr = nw.exclude(date_col).fill_null(fill_value)
        wide = wide.with_columns(fill_expr)

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
    df: IntoFrameT,
    value_columns: str | Sequence[str] = "impressions",
    *,
    date_col: str = "date_day",
    platform_col: str = "platform",
    agg: str = "sum",
    fill_value: float | None = 0.0,
    include_missing_dates: bool = False,
    freq: str = "D",
    rename_date_to: str | None = "date",
) -> IntoFrameT:
    """Process Fivetran Ad Reporting tables into wide, model-ready features.

    Compatible with Fivetran's Ad Reporting schema tables:

    - ad_reporting__account_report: daily metrics by account
    - ad_reporting__campaign_report: daily metrics by campaign and account
    - ad_reporting__ad_group_report: daily metrics by ad group, campaign and account
    - ad_reporting__ad_report: daily metrics by ad, ad group, campaign and account

    The input data must include a date column, a platform column (e.g., vendor name),
    and one or more metric columns such as ``spend`` or ``impressions``. The output is
    a wide dataframe with one row per date and columns named ``{platform}_{metric}``.

    This function supports multiple DataFrame backends including pandas, polars, and PySpark.
    The output type will match the input type (type preservation).

    Parameters
    ----------
    df : DataFrame-like
        Input dataframe in long format with at least the date, platform, and metric columns.
        Accepts pandas.DataFrame, polars.DataFrame, polars.LazyFrame, or pyspark.DataFrame.
    value_columns : str or Sequence[str], default "impressions"
        Column name(s) to aggregate and pivot. Example: "spend" or ["spend", "impressions"].
    date_col : str, default "date_day"
        Name of the date column.
    platform_col : str, default "platform"
        Name of the platform (vendor) column.
    agg : str, default "sum"
        Aggregation method applied during groupby.
        Supported: 'sum', 'mean', 'min', 'max', 'count'.
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
    DataFrame-like
        A wide-format dataframe with one row per date and columns for each
        ``{platform}_{metric}`` combination. The return type matches the input type.

    Notes
    -----
    **Backend-specific implementation notes:**

    - **PySpark**: Uses native PySpark pivot operations for distributed computing.
      This is a temporary workaround until narwhals adds LazyFrame.pivot support.
      See: https://github.com/narwhals-dev/narwhals/issues/1901

    - **pandas/polars**: Uses narwhals pivot operations (works on eager DataFrames).

    - **polars.LazyFrame**: Input LazyFrames are automatically collected to eager DataFrames
      for the pivot operation, then results are returned as eager DataFrames.
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

    # Convert back to native format (preserves input type)
    result_native = nw.to_native(result)

    # Fix pandas-specific column naming quirk: narwhals pivot sets columns.names to [""]
    # but pandas expects [None]. This is a pandas-only issue.
    if isinstance(result_native, pd.DataFrame) and hasattr(
        result_native.columns, "names"
    ):
        result_native.columns.names = [None]

    return result_native


def process_fivetran_shopify_unique_orders(
    df: IntoFrameT,
    *,
    date_col: str = "processed_timestamp",
    order_key_col: str = "orders_unique_key",
    rename_date_to: str = "date",
) -> IntoFrameT:
    """Compute daily unique order counts from a (pre-filtered) Shopify dataset.

    This function targets data following the Fivetran Shopify orders schema
    (e.g., ``shopify__orders``). It assumes the input ``df`` is already filtered to
    the desired subset (e.g., non-canceled, US-delivery, new-only orders).

    Supports pandas DataFrames, polars DataFrames (eager and lazy), and PySpark DataFrames.
    The output type matches the input type.

    Parameters
    ----------
    df : IntoFrameT
        Input dataframe following the Shopify orders schema.
        Supported types: pandas.DataFrame, polars.DataFrame, polars.LazyFrame, pyspark.sql.DataFrame
    date_col : str, default "processed_timestamp"
        Timestamp column from which the daily bucket is derived.
    order_key_col : str, default "orders_unique_key"
        Unique order identifier column.
    rename_date_to : str, default "date"
        Name of the date column in the result.

    Returns
    -------
    IntoFrameT
        A dataframe with two columns: ``rename_date_to`` and ``orders``, where
        ``orders`` is the unique order count per day. The output type matches the input type.

    Backend-specific implementation notes
    -------------------------------------
    - **pandas**: Datetime coercion with ``errors="coerce"`` is applied before processing
      (narwhals doesn't support this parameter).

    - **polars/PySpark**: Datetime parsing is handled by narwhals' standard datetime operations.

    - **polars.LazyFrame**: Input LazyFrames are automatically collected to eager DataFrames,
      then results are returned as eager DataFrames.
    """
    # Handle pandas-specific datetime coercion (narwhals doesn't support errors="coerce")
    if isinstance(df, pd.DataFrame):
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
        nw_df = nw.from_native(df_copy)
    else:
        nw_df = nw.from_native(df)

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

    # Convert back to native format (preserves input type)
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
