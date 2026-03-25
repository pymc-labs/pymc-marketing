# Multi-Backend Testing Guide

This guide explains how to write tests for functions that support multiple DataFrame backends (pandas, polars, PySpark) in the `pymc_marketing.data` module.

## Overview

The Fivetran data processing functions support multiple DataFrame backends:
- **pandas**: Standard pandas DataFrames
- **polars**: Polars eager DataFrames
- **polars_lazy**: Polars lazy DataFrames
- **pyspark**: PySpark DataFrames (for Databricks/Delta Lake)

All tests automatically run across all supported backends using pytest parametrization.

## Quick Start

### Writing a Multi-Backend Test

```python
def test_my_function(backend_converter):
    """Test my_function across all backends."""
    # 1. Create pandas test data
    df_pandas = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    # 2. Convert to target backend
    df_backend = backend_converter.to_backend(df_pandas)

    # 3. Call function under test
    result = my_function(df_backend)

    # 4. Convert result back to pandas for assertions
    result_pd = backend_converter.to_pandas(result)

    # 5. Assert on pandas DataFrame
    assert result_pd.shape == (3, 2)
    assert list(result_pd.columns) == ["col1", "col2"]
```

This test will automatically run 4 times (once per backend: pandas, polars, polars_lazy, pyspark).

### Writing a Pandas-Only Test (e.g., Accessor Tests)

```python
def test_my_accessor():
    """Test .myaccessor method (pandas only)."""
    df = pd.DataFrame({"col1": [1, 2, 3]})

    # Use accessor (pandas-specific)
    result = df.myaccessor.my_method()

    # Assert directly on pandas DataFrame
    assert isinstance(result, pd.DataFrame)
```

No `backend_converter` fixture needed for pandas-only tests.

## The `backend_converter` Fixture

The `backend_converter` fixture provides utilities for multi-backend testing. It's automatically parametrized across all backends.

### Methods

#### `to_backend(df: pd.DataFrame) -> Any`

Convert a pandas DataFrame to the target backend.

```python
# Returns pandas.DataFrame if backend_name == "pandas"
df_pandas = backend_converter.to_backend(df)

# Returns polars.DataFrame if backend_name == "polars"
df_polars = backend_converter.to_backend(df)

# Returns polars.LazyFrame if backend_name == "polars_lazy"
df_lazy = backend_converter.to_backend(df)

# Returns pyspark.sql.DataFrame if backend_name == "pyspark"
df_spark = backend_converter.to_backend(df)
```

#### `to_pandas(df: Any) -> pd.DataFrame`

Convert any backend DataFrame back to pandas using narwhals, collecting lazy frames first.

```python
# Works with any backend
result_pd = backend_converter.to_pandas(result)
```

Narwhals automatically handles pandas, polars (eager and lazy), and PySpark; lazy Narwhals frames are collected via `LazyFrame.collect()` before calling `.to_pandas()` so tests that return lazy data still materialize properly.

#### `assert_frame_equal(result: Any, expected: pd.DataFrame, **kwargs)`

Assert two DataFrames are equal, handling backend conversions automatically.

```python
expected = pd.DataFrame({"col1": [1, 2, 3]})

# Automatically converts result to pandas before comparison
backend_converter.assert_frame_equal(result, expected, check_dtype=False)
```

### Properties

- `backend_name: str` - Current backend name ("pandas", "polars", "polars_lazy", "pyspark")

```python
if backend_converter.backend_name == "pyspark":
    # Skip certain assertions for PySpark
    pass
```

## Test Organization Pattern

Tests are organized into two categories:

### 1. Multi-Backend Tests (Function Calls)

Test the standalone function across all backends:

```python
def test_process_data_multibackend(example_data_df, backend_converter):
    """Test process_data across all backends."""
    df_backend = backend_converter.to_backend(example_data_df)
    result = process_data(df_backend)
    result_pd = backend_converter.to_pandas(result)

    # Assertions...
```

**Naming convention**: `test_<function>_multibackend`

### 2. Pandas-Only Accessor Tests

Test pandas accessor methods (pandas-only):

```python
def test_process_data_accessor(example_data_df):
    """Test .accessor.process_data (pandas only)."""
    result = example_data_df.accessor.process_data()

    # Assertions...
```

**Naming convention**: `test_<function>_accessor`

## Example: Real Test from `test_fivetran.py`

```python
@pytest.mark.parametrize(
    "value_columns, expected_columns, expected_values",
    [
        (
            "spend",
            ["facebook_ads_spend", "google_ads_spend"],
            [[30.0, 10.0], [10.0, 0.0]],
        ),
        (
            ["spend", "impressions"],
            [
                "facebook_ads_spend",
                "google_ads_spend",
                "facebook_ads_impressions",
                "google_ads_impressions",
            ],
            [[30.0, 10.0, 1500.0, 1000.0], [10.0, 0.0, 500.0, 0.0]],
        ),
    ],
)
def test_ad_report_schema_multibackend(
    example_ad_report_df,
    backend_converter,
    value_columns,
    expected_columns,
    expected_values,
):
    """Test process_fivetran_ad_reporting with ad_report schema across all backends."""
    # Convert to target backend
    df_backend = backend_converter.to_backend(example_ad_report_df)

    # Call function
    result = process_fivetran_ad_reporting(
        df_backend, value_columns=value_columns, rename_date_to="date"
    )

    # Convert result back to pandas for assertions
    result_pd = backend_converter.to_pandas(result)

    # Build expected DataFrame
    expected = (
        pd.DataFrame(
            data=expected_values,
            columns=expected_columns,
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )

    # Ensure required columns exist and match expected values
    assert set(["date", *expected_columns]).issubset(set(result_pd.columns))

    result_subset = result_pd[["date", *expected_columns]]
    pd.testing.assert_frame_equal(result_subset, expected, check_dtype=False)
```

This test runs **8 times total**:
- 4 backends × 2 parametrizations (value_columns) = 8 test cases

## Running Tests

### Run All Tests (All Backends)

```bash
pytest tests/data/test_fivetran.py
```

This runs all tests across all backends (pandas, polars, polars_lazy, pyspark).

### Run Specific Backend Only

```bash
# Run only pandas tests
pytest tests/data/test_fivetran.py -k "pandas"

# Run only polars tests (eager + lazy)
pytest tests/data/test_fivetran.py -k "polars"

# Run only PySpark tests
pytest tests/data/test_fivetran.py -k "pyspark"
```

### Run Specific Test Function

```bash
# Run all backends for a specific test
pytest tests/data/test_fivetran.py::test_ad_report_schema_multibackend

# Run specific backend for specific test
pytest tests/data/test_fivetran.py::test_ad_report_schema_multibackend[pandas-...]
```

### Skip Tests Requiring Optional Dependencies

If polars or pyspark are not installed, the `backend_converter` fixture automatically skips tests for those backends:

```python
# In conftest.py
if backend_name in ["polars", "polars_lazy"]:
    try:
        import polars as pl
        self.pl = pl
    except ImportError:
        pytest.skip(f"polars not installed, skipping {backend_name} tests")
```

## Test Coverage Summary

From `test_fivetran.py`:

| Test Function | Parametrizations | Backends | Total Cases |
|--------------|------------------|----------|-------------|
| `test_ad_report_schema_multibackend` | 2 | 4 | 8 |
| `test_account_report_schema_multibackend` | 2 | 4 | 8 |
| `test_campaign_report_schema_multibackend` | 2 | 4 | 8 |
| `test_shopify_orders_unique_orders_multibackend` | 1 | 4 | 4 |
| `test_ad_report_schema_accessor` | 2 | 1 (pandas) | 2 |
| `test_account_report_schema_accessor` | 2 | 1 (pandas) | 2 |
| `test_campaign_report_schema_accessor` | 2 | 1 (pandas) | 2 |
| `test_shopify_orders_unique_orders_accessor` | 1 | 1 (pandas) | 1 |
| **TOTAL** | | | **35** |

## Backend-Specific Considerations

### PySpark Notes

1. **Pivot Implementation**: PySpark uses native pivot operations (not narwhals) due to `LazyFrame.pivot()` not being supported yet. See [narwhals #1901](https://github.com/narwhals-dev/narwhals/issues/1901).

2. **Column Naming**: PySpark pivot generates columns like `{platform}_{agg}({metric})` which are normalized to `{platform}_{metric}` by the implementation.

3. **SparkSession**: The `backend_converter` automatically creates a local SparkSession when testing PySpark.

4. **Performance**: PySpark tests may be slower due to Spark initialization overhead.

### Polars Notes

1. **Lazy Frames**: `polars_lazy` tests convert input to `LazyFrame` but return eager DataFrames (collected before returning).

2. **DateTime Handling**: Polars has different datetime handling than pandas. The implementation uses narwhals for consistent behavior.

### Pandas Notes

1. **Datetime Coercion**: Pandas-specific `errors="coerce"` parameter is handled specially in the implementation (not supported by narwhals).

2. **Column Names Fix**: Pandas has a quirk where `columns.names = [""]` after pivot, which is fixed to `[None]` in the implementation.

## Pytest Markers

Two markers are available for conditional test execution:

- `@pytest.mark.requires_polars` - Mark test as requiring polars
- `@pytest.mark.requires_pyspark` - Mark test as requiring pyspark

**Note**: Currently not used in `test_fivetran.py` because the `backend_converter` fixture handles skipping automatically.

## Best Practices

1. **Always create test data as pandas DataFrames** - Convert using `backend_converter.to_backend()`

2. **Always convert results back to pandas** for assertions - Use `backend_converter.to_pandas()`

3. **Separate accessor tests from function tests** - Accessor tests are pandas-only

4. **Use descriptive test names** with `_multibackend` or `_accessor` suffix

5. **Test type preservation** - Verify output type matches input type

6. **Document backend-specific behavior** in docstrings

## Troubleshooting

### Import Errors in LSP

LSP may show import errors for `polars` or `pyspark` if they're not in your current environment. This is expected and doesn't affect test execution.

### PySpark Tests Failing Locally

If PySpark tests fail locally but you don't have Spark installed, that's expected. The `backend_converter` will skip those tests automatically.

### Tests Hanging

PySpark tests may hang if SparkSession fails to initialize. Check Java installation and `JAVA_HOME` environment variable.

### Unexpected Type Mismatches

Remember that type preservation is key: input type should equal output type. If a pandas DataFrame input returns a polars DataFrame, there's a bug in the implementation.

## Future Improvements

1. **narwhals LazyFrame.pivot support**: Once [narwhals #1901](https://github.com/narwhals-dev/narwhals/issues/1901) is resolved, the PySpark-specific pivot implementation can be removed in favor of narwhals' unified API.

2. **Dask support**: The `backend_converter` pattern can be extended to support Dask DataFrames.

3. **Performance benchmarks**: Add benchmarks comparing backend performance on large datasets.

## References

- [narwhals Documentation](https://narwhals-dev.github.io/narwhals/)
- [narwhals Issue #1901 - LazyFrame.pivot](https://github.com/narwhals-dev/narwhals/issues/1901)
- [PyMC Marketing Documentation](https://www.pymc-marketing.io/)
- [Fivetran dbt_ad_reporting Schema](https://fivetran.github.io/dbt_ad_reporting/)
