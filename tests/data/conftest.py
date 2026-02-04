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
"""Test fixtures for multi-backend DataFrame testing.

This module provides fixtures for testing functions that support multiple
DataFrame backends (pandas, polars, PySpark).

Note
----
PySpark Java 21 compatibility options are configured in tests/conftest.py
via the pytest_configure hook. This ensures PYSPARK_SUBMIT_ARGS is set
before any PySpark imports occur.
"""

from typing import Any

import pandas as pd
import pytest


class BackendConverter:
    """Helper class for converting DataFrames between backends.

    This class provides utilities for testing functions that should work
    with pandas, polars, and PySpark DataFrames.

    Parameters
    ----------
    backend_name : str
        Name of the backend: "pandas", "polars", "polars_lazy", or "pyspark"
    """

    def __init__(self, backend_name: str):
        self.backend_name = backend_name

        # Import backend libraries conditionally
        if backend_name in ["polars", "polars_lazy"]:
            try:
                import polars as pl

                self.pl = pl
            except ImportError:
                pytest.skip(f"polars not installed, skipping {backend_name} tests")

        elif backend_name == "pyspark":
            try:
                from pyspark.sql import SparkSession

                # Java options are set at module level (top of file) before any imports
                # This ensures they're applied when py4j gateway initializes the JVM
                self.spark = (
                    SparkSession.builder.master("local[1]")
                    .appName("pymc-marketing-tests")
                    .getOrCreate()
                )
            except ImportError:
                pytest.skip("pyspark not installed, skipping pyspark tests")

    def to_backend(self, df: pd.DataFrame) -> Any:
        """Convert pandas DataFrame to the target backend.

        Parameters
        ----------
        df : pd.DataFrame
            Input pandas DataFrame to convert.

        Returns
        -------
        DataFrame
            DataFrame in the target backend format.
        """
        if self.backend_name == "pandas":
            return df

        elif self.backend_name == "polars":
            return self.pl.from_pandas(df)

        elif self.backend_name == "polars_lazy":
            return self.pl.from_pandas(df).lazy()

        elif self.backend_name == "pyspark":
            return self.spark.createDataFrame(df)

        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")

    def to_pandas(self, df: Any) -> pd.DataFrame:
        """Convert DataFrame from any backend to pandas.

        Uses narwhals for unified conversion across all backends.

        Parameters
        ----------
        df : Any
            DataFrame in any backend format.

        Returns
        -------
        pd.DataFrame
            DataFrame converted to pandas format.
        """
        import narwhals as nw

        # Convert to narwhals DataFrame, collect lazy frames, then to pandas
        # This handles pandas, polars (eager and lazy), and PySpark
        nw_df = nw.from_native(df)

        if hasattr(nw_df, "collect") and not hasattr(nw_df, "to_pandas"):
            nw_df = nw_df.collect()

        return nw_df.to_pandas()

    def assert_frame_equal(
        self, result: Any, expected: pd.DataFrame, **kwargs: Any
    ) -> None:
        """Assert that two DataFrames are equal, handling backend conversions.

        Parameters
        ----------
        result : Any
            Result DataFrame (any backend).
        expected : pd.DataFrame
            Expected pandas DataFrame.
        **kwargs : Any
            Additional arguments passed to pd.testing.assert_frame_equal.
        """
        result_pd = self.to_pandas(result)
        pd.testing.assert_frame_equal(result_pd, expected, **kwargs)


@pytest.fixture(params=["pandas", "polars", "polars_lazy", "pyspark"])
def backend_converter(request: pytest.FixtureRequest) -> BackendConverter:
    """Fixture providing a BackendConverter for multi-backend testing.

    This fixture is parametrized across all supported backends:
    - pandas: Standard pandas DataFrames
    - polars: Polars eager DataFrames
    - polars_lazy: Polars lazy DataFrames
    - pyspark: PySpark DataFrames

    Tests using this fixture will automatically run for all backends.

    Example
    -------
    >>> def test_function(backend_converter):
    ...     # Create pandas test data
    ...     df_pandas = pd.DataFrame({"a": [1, 2, 3]})
    ...
    ...     # Convert to target backend
    ...     df_backend = backend_converter.to_backend(df_pandas)
    ...
    ...     # Call function under test
    ...     result = my_function(df_backend)
    ...
    ...     # Convert back to pandas for assertions
    ...     result_pandas = backend_converter.to_pandas(result)
    ...     assert result_pandas.shape == (3, 1)
    """
    return BackendConverter(request.param)


@pytest.fixture(params=["pandas"])
def pandas_only_converter(request: pytest.FixtureRequest) -> BackendConverter:
    """Fixture for pandas-only tests (e.g., accessor tests).

    This is useful for testing pandas-specific features like the
    .fivetran accessor that only works with pandas DataFrames.

    Example
    -------
    >>> def test_pandas_accessor(pandas_only_converter):
    ...     df = pd.DataFrame({"a": [1, 2, 3]})
    ...     result = df.fivetran.some_method()
    ...     assert isinstance(result, pd.DataFrame)
    """
    return BackendConverter(request.param)
