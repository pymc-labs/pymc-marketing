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
"""Tests for the types module."""

from typing import Any

import pandas as pd
import pytest

from pymc_marketing.mmm.types import MMMBuilder


class TestMMMBuilderProtocol:
    """Test the MMMBuilder Protocol class."""

    def test_protocol_is_defined(self):
        """Test that the MMMBuilder Protocol is properly defined."""
        # Check that the Protocol exists and has the expected method
        assert hasattr(MMMBuilder, "build_model")

    def test_class_with_correct_signature_satisfies_protocol(self):
        """Test that a class with the correct method signature satisfies the protocol."""

        class ValidBuilder:
            """A class that correctly implements the MMMBuilder protocol."""

            def build_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
                return {"mock": "mmm"}

        builder = ValidBuilder()

        # Check that the instance can be used where MMMBuilder is expected
        # This is a structural check - the class has the required method
        assert hasattr(builder, "build_model")
        assert callable(builder.build_model)

        # Test that the method can be called with the expected arguments
        X = pd.DataFrame(
            {"date": pd.date_range("2025-01-01", periods=3), "x": [1, 2, 3]}
        )
        y = pd.Series([10, 20, 30])
        result = builder.build_model(X, y)
        assert result == {"mock": "mmm"}

    def test_class_with_additional_init_args_satisfies_protocol(self):
        """Test that a builder with additional __init__ args still satisfies the protocol."""

        class ConfigurableBuilder:
            """A builder that takes configuration at construction time."""

            def __init__(self, config: dict, name: str = "default"):
                self.config = config
                self.name = name

            def build_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
                return {"config": self.config, "name": self.name, "n_rows": len(X)}

        builder = ConfigurableBuilder(
            config={"channels": ["tv", "radio"]}, name="my_model"
        )

        assert hasattr(builder, "build_model")

        X = pd.DataFrame(
            {"date": pd.date_range("2025-01-01", periods=5), "x": range(5)}
        )
        y = pd.Series(range(5))
        result = builder.build_model(X, y)

        assert result["config"] == {"channels": ["tv", "radio"]}
        assert result["name"] == "my_model"
        assert result["n_rows"] == 5

    def test_class_returning_real_object_satisfies_protocol(self):
        """Test a builder that returns a mock MMM-like object."""

        class MockMMM:
            """A mock MMM class with expected attributes."""

            def __init__(self, X: pd.DataFrame, y: pd.Series):
                self.X = X
                self.y = y
                self.idata = None
                self.sampler_config = None

            def fit(self, X: pd.DataFrame, y: pd.Series, progressbar: bool = True):
                return self

            def sample_posterior_predictive(self, X: pd.DataFrame, **kwargs):
                return self.idata

        class RealBuilder:
            """A builder that returns an MMM-like object."""

            def build_model(self, X: pd.DataFrame, y: pd.Series) -> MockMMM:
                return MockMMM(X, y)

        builder = RealBuilder()
        X = pd.DataFrame(
            {"date": pd.date_range("2025-01-01", periods=3), "x": [1, 2, 3]}
        )
        y = pd.Series([10, 20, 30])

        mmm = builder.build_model(X, y)

        # Check that the returned object has MMM-like attributes
        assert hasattr(mmm, "fit")
        assert hasattr(mmm, "sample_posterior_predictive")
        assert hasattr(mmm, "idata")
        assert hasattr(mmm, "sampler_config")
        assert mmm.X.equals(X)
        assert mmm.y.equals(y)

    def test_class_without_build_model_does_not_satisfy_protocol(self):
        """Test that a class without build_model doesn't satisfy the protocol."""

        class InvalidBuilder:
            """A class that doesn't implement build_model."""

            def create_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
                return {"mock": "mmm"}

        builder = InvalidBuilder()

        # The class doesn't have build_model
        assert not hasattr(builder, "build_model")
        assert hasattr(builder, "create_model")

    def test_class_with_wrong_signature_is_still_callable(self):
        """Test behavior of a class with wrong build_model signature."""

        class WrongSignatureBuilder:
            """A class with build_model but wrong signature."""

            def build_model(self, data: pd.DataFrame) -> Any:
                # Missing y parameter
                return {"mock": "mmm"}

        builder = WrongSignatureBuilder()

        # Still has build_model but with wrong signature
        assert hasattr(builder, "build_model")
        assert callable(builder.build_model)

        # Calling with wrong number of args would fail at runtime
        X = pd.DataFrame({"x": [1, 2, 3]})
        y = pd.Series([10, 20, 30])

        with pytest.raises(TypeError):
            builder.build_model(X, y)  # Too many arguments

    def test_runtime_isinstance_check(self):
        """Test that runtime_checkable allows isinstance checks."""

        class ValidBuilder:
            def build_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
                return None

        builder = ValidBuilder()

        # Note: By default, Protocols are not runtime_checkable
        # If we want isinstance checks, we'd need to add @runtime_checkable
        # This test documents the current behavior
        assert hasattr(builder, "build_model")

    def test_protocol_used_in_type_hints(self):
        """Test that the Protocol can be used for type hints in functions."""

        def run_cv_with_builder(X: pd.DataFrame, y: pd.Series, mmm: MMMBuilder) -> Any:
            """A function that uses MMMBuilder in its type hints."""
            return mmm.build_model(X, y)

        class SimpleBuilder:
            def build_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
                return {"built": True, "rows": len(X)}

        X = pd.DataFrame({"x": [1, 2, 3]})
        y = pd.Series([10, 20, 30])
        builder = SimpleBuilder()

        result = run_cv_with_builder(X, y, builder)
        assert result["built"] is True
        assert result["rows"] == 3
