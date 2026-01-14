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
"""Tests for InferenceData schema validation."""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.data.idata.schema import (
    InferenceDataGroupSchema,
    MMMIdataSchema,
    VariableSchema,
)

# Seed for reproducibility
seed = sum(map(ord, "idata_schema_tests"))
rng = np.random.default_rng(seed=seed)


# ============================================================================
# Shared Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def simple_dates() -> pd.DatetimeIndex:
    """Simple date range for testing (52 weeks)."""
    return pd.date_range("2024-01-01", periods=52, freq="W")


@pytest.fixture(scope="module")
def simple_channels() -> list[str]:
    """Simple channel list for testing."""
    return ["TV", "Radio", "Social"]


@pytest.fixture
def basic_schema() -> MMMIdataSchema:
    """Schema for basic MMM (no controls, seasonality, time-varying)."""
    return MMMIdataSchema.from_model_config(
        custom_dims=(),
        has_controls=False,
        has_seasonality=False,
        time_varying=False,
    )


@pytest.fixture
def valid_basic_idata(
    simple_dates: pd.DatetimeIndex, simple_channels: list[str]
) -> az.InferenceData:
    """Complete valid InferenceData for basic MMM schema."""
    return az.InferenceData(
        constant_data=xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    rng.uniform(0, 100, size=(52, 3)),
                    dims=("date", "channel"),
                    coords={"date": simple_dates, "channel": simple_channels},
                ),
                "target_data": xr.DataArray(
                    rng.uniform(100, 1000, size=52),
                    dims=("date",),
                    coords={"date": simple_dates},
                ),
                "channel_scale": xr.DataArray(
                    [100.0, 50.0, 75.0],
                    dims=("channel",),
                    coords={"channel": simple_channels},
                ),
                "target_scale": xr.DataArray(500.0),
            }
        ),
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 52, 3)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": simple_dates, "channel": simple_channels},
                ),
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 52)),
                    dims=("chain", "draw", "date"),
                    coords={"date": simple_dates},
                ),
            }
        ),
        fit_data=xr.Dataset(
            {
                "TV": xr.DataArray(
                    np.ones(52), dims=("date",), coords={"date": simple_dates}
                ),
                "Radio": xr.DataArray(
                    np.ones(52), dims=("date",), coords={"date": simple_dates}
                ),
                "Social": xr.DataArray(
                    np.ones(52), dims=("date",), coords={"date": simple_dates}
                ),
                "target": xr.DataArray(
                    np.ones(52), dims=("date",), coords={"date": simple_dates}
                ),
            }
        ),
    )


# ============================================================================
# Category 1: VariableSchema Tests
# ============================================================================


def test_variable_schema_validates_correct_dims():
    """Test that VariableSchema accepts variable with correct dimensions."""
    # Arrange
    schema = VariableSchema(
        name="channel_contribution",
        dims=("date", "channel"),
        dtype="float64",
        required=True,
    )

    # Create mock DataArray with correct structure
    data_array = xr.DataArray(
        np.ones((52, 3)),
        dims=("date", "channel"),
        coords={
            "date": pd.date_range("2024-01-01", periods=52, freq="W"),
            "channel": ["TV", "Radio", "Social"],
        },
    )

    # Act
    errors = schema.validate_variable(data_array)

    # Assert
    assert errors == [], f"Expected no validation errors, got: {errors}"


@pytest.mark.parametrize(
    "wrong_dims",
    [
        ("date",),  # Missing channel dimension
        ("date", "country"),  # Wrong dimension name
        ("date", "channel", "extra"),  # Extra dimension
    ],
    ids=["missing_dim", "wrong_dim_name", "extra_dim"],
)
def test_variable_schema_detects_wrong_dims(wrong_dims):
    """Test that VariableSchema detects incorrect dimensions."""
    # Arrange
    schema = VariableSchema(
        name="channel_contribution",
        dims=("date", "channel"),
        dtype="float64",
        required=True,
    )

    # Create mock DataArray with wrong dimensions
    shape = tuple([10] * len(wrong_dims))
    data_array = xr.DataArray(
        np.ones(shape),
        dims=wrong_dims,
    )

    # Act
    errors = schema.validate_variable(data_array)

    # Assert
    assert len(errors) > 0, "Expected validation errors for wrong dimensions"
    assert "dims" in errors[0].lower(), (
        f"Error message should mention 'dims': {errors[0]}"
    )
    assert "channel_contribution" in errors[0], (
        f"Error message should mention variable name: {errors[0]}"
    )


@pytest.mark.parametrize(
    "wrong_dtype,expected_dtype",
    [
        ("int64", "float64"),
        ("float32", "float64"),
        ("object", "float64"),
    ],
    ids=["int_instead_of_float", "float32_instead_of_float64", "object_type"],
)
def test_variable_schema_detects_wrong_dtype(wrong_dtype, expected_dtype):
    """Test that VariableSchema detects incorrect data types."""
    # Arrange
    schema = VariableSchema(
        name="channel_contribution",
        dims=("date", "channel"),
        dtype=expected_dtype,
        required=True,
    )

    # Create mock DataArray with wrong dtype
    if wrong_dtype == "object":
        data = np.array([["a", "b"], ["c", "d"]], dtype=object)
    else:
        data = np.ones((2, 2), dtype=wrong_dtype)

    data_array = xr.DataArray(data, dims=("date", "channel"))

    # Act
    errors = schema.validate_variable(data_array)

    # Assert
    assert len(errors) > 0, "Expected validation errors for wrong dtype"
    assert "dtype" in errors[0].lower(), (
        f"Error message should mention 'dtype': {errors[0]}"
    )


def test_variable_schema_accepts_wildcard_dims():
    """Test that dims='*' accepts any dimension structure."""
    # Arrange
    schema = VariableSchema(
        name="channel_scale",
        dims="*",  # Wildcard - any dims acceptable
        dtype="float64",
        required=True,
    )

    # Test with various dimension structures
    test_cases = [
        xr.DataArray([1.0, 2.0], dims=("channel",)),
        xr.DataArray([[1.0, 2.0]], dims=("country", "channel")),
        xr.DataArray(5.0),  # Scalar (no dimensions)
    ]

    for data_array in test_cases:
        # Act
        errors = schema.validate_variable(data_array)

        # Assert
        assert errors == [], (
            f"Wildcard dims should accept {data_array.dims}, got errors: {errors}"
        )


def test_variable_schema_skips_dtype_check_when_none():
    """Test that dtype=None skips dtype validation."""
    # Arrange
    schema = VariableSchema(
        name="some_variable",
        dims=("date",),
        dtype=None,  # Skip dtype check
        required=True,
    )

    # Test with various dtypes
    test_dtypes = ["int64", "float64", "float32", "object"]

    for dtype in test_dtypes:
        # Create data with this dtype
        if dtype == "object":
            data = np.array(["a", "b", "c"], dtype=object)
        else:
            data = np.ones(3, dtype=dtype)

        data_array = xr.DataArray(data, dims=("date",))

        # Act
        errors = schema.validate_variable(data_array)

        # Assert
        assert errors == [], (
            f"dtype=None should skip validation for {dtype}, got errors: {errors}"
        )


# ============================================================================
# Category 2: InferenceDataGroupSchema Tests
# ============================================================================


def test_group_schema_validates_required_group_exists():
    """Test that required group existence is validated."""
    # Arrange
    schema = InferenceDataGroupSchema(
        name="posterior",
        required=True,
        variables={},
    )

    # Create idata without posterior group
    idata = az.from_dict(prior={"a": np.ones((2, 10))})

    # Act
    errors = schema.validate_group(idata)

    # Assert
    assert len(errors) > 0, "Expected error for missing required group"
    assert "posterior" in errors[0], (
        f"Error should mention group name 'posterior': {errors[0]}"
    )
    assert "not found" in errors[0].lower(), (
        f"Error should say 'not found': {errors[0]}"
    )


def test_group_schema_allows_missing_optional_group():
    """Test that optional group can be missing without error."""
    # Arrange
    schema = InferenceDataGroupSchema(
        name="posterior_predictive",
        required=False,  # Optional
        variables={},
    )

    # Create idata without posterior_predictive group
    idata = az.from_dict(posterior={"a": np.ones((2, 10))})

    # Act
    errors = schema.validate_group(idata)

    # Assert
    assert errors == [], (
        f"Optional group missing should not produce errors, got: {errors}"
    )


def test_group_schema_validates_required_variables():
    """Test that required variables are validated."""
    # Arrange
    schema = InferenceDataGroupSchema(
        name="posterior",
        required=True,
        variables={
            "channel_contribution": VariableSchema(
                name="channel_contribution",
                dims=("date", "channel"),
                dtype="float64",
                required=True,
            ),
            "mu": VariableSchema(
                name="mu",
                dims=("date",),
                dtype="float64",
                required=True,
            ),
        },
    )

    # Create idata with only one of the required variables
    dates = pd.date_range("2024-01-01", periods=52, freq="W")
    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    np.ones((2, 10, 52)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
                # Missing: channel_contribution
            }
        )
    )

    # Act
    errors = schema.validate_group(idata)

    # Assert
    assert len(errors) > 0, "Expected error for missing required variable"
    assert "channel_contribution" in errors[0], (
        f"Error should mention missing variable: {errors[0]}"
    )
    assert "not found" in errors[0].lower(), (
        f"Error should say 'not found': {errors[0]}"
    )


def test_group_schema_allows_missing_optional_variables():
    """Test that optional variables can be missing."""
    # Arrange
    schema = InferenceDataGroupSchema(
        name="posterior",
        required=True,
        variables={
            "channel_contribution": VariableSchema(
                name="channel_contribution",
                dims=("chain", "draw", "date", "channel"),
                dtype="float64",
                required=True,
            ),
            "control_contribution": VariableSchema(
                name="control_contribution",
                dims=("chain", "draw", "date", "control"),
                dtype="float64",
                required=False,  # Optional
            ),
        },
    )

    # Create idata without optional variable
    dates = pd.date_range("2024-01-01", periods=52, freq="W")
    channels = ["TV", "Radio"]
    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    np.ones((2, 10, 52, 2)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": dates, "channel": channels},
                ),
                # Missing: control_contribution (but it's optional)
            }
        )
    )

    # Act
    errors = schema.validate_group(idata)

    # Assert
    assert errors == [], (
        f"Optional variable missing should not produce errors, got: {errors}"
    )


def test_group_schema_delegates_variable_validation():
    """Test that variable structure validation is delegated correctly."""
    # Arrange
    schema = InferenceDataGroupSchema(
        name="posterior",
        required=True,
        variables={
            "channel_contribution": VariableSchema(
                name="channel_contribution",
                dims=("date", "channel"),
                dtype="float64",
                required=True,
            ),
        },
    )

    # Create idata with variable but WRONG dimensions
    dates = pd.date_range("2024-01-01", periods=52, freq="W")
    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    np.ones((2, 10, 52)),
                    dims=("chain", "draw", "date"),  # Missing "channel" dim
                    coords={"date": dates},
                ),
            }
        )
    )

    # Act
    errors = schema.validate_group(idata)

    # Assert
    assert len(errors) > 0, "Expected error for wrong variable structure"
    assert "dims" in errors[0].lower(), (
        f"Error should mention dimension problem: {errors[0]}"
    )


# ============================================================================
# Category 3: MMMIdataSchema Factory Tests
# ============================================================================


def test_schema_factory_creates_basic_schema():
    """Test that from_model_config creates basic MMM schema."""
    # Arrange & Act
    schema = MMMIdataSchema.from_model_config(
        custom_dims=(),
        has_controls=False,
        has_seasonality=False,
        time_varying=False,
    )

    # Assert - Check structure
    assert "constant_data" in schema.groups, "Schema should include constant_data group"
    assert "posterior" in schema.groups, "Schema should include posterior group"

    # Check constant_data variables
    constant_data_group = schema.groups["constant_data"]
    assert "channel_data" in constant_data_group.variables, (
        "constant_data should include channel_data"
    )
    assert "target_data" in constant_data_group.variables, (
        "constant_data should include target_data"
    )
    assert "channel_scale" in constant_data_group.variables, (
        "constant_data should include channel_scale"
    )
    assert "target_scale" in constant_data_group.variables, (
        "constant_data should include target_scale"
    )

    # Check posterior variables
    posterior_group = schema.groups["posterior"]
    assert "channel_contribution" in posterior_group.variables, (
        "posterior should include channel_contribution"
    )
    assert "mu" in posterior_group.variables, "posterior should include mu"

    # Verify no control variables (not requested)
    assert "control_data_" not in constant_data_group.variables, (
        "Should not include control variables when has_controls=False"
    )


def test_schema_factory_includes_controls():
    """Test that from_model_config includes control variables."""
    # Arrange & Act
    schema = MMMIdataSchema.from_model_config(
        custom_dims=(),
        has_controls=True,  # Request controls
        has_seasonality=False,
        time_varying=False,
    )

    # Assert
    constant_data_group = schema.groups["constant_data"]
    assert "control_data_" in constant_data_group.variables, (
        "Should include control_data_ when has_controls=True"
    )

    posterior_group = schema.groups["posterior"]
    assert "control_contribution" in posterior_group.variables, (
        "Should include control_contribution when has_controls=True"
    )


def test_schema_factory_includes_seasonality():
    """Test that from_model_config includes seasonality variables."""
    # Arrange & Act
    schema = MMMIdataSchema.from_model_config(
        custom_dims=(),
        has_controls=False,
        has_seasonality=True,  # Request seasonality
        time_varying=False,
    )

    # Assert
    constant_data_group = schema.groups["constant_data"]
    assert "dayofyear" in constant_data_group.variables, (
        "Should include dayofyear when has_seasonality=True"
    )

    posterior_group = schema.groups["posterior"]
    assert "yearly_seasonality_contribution" in posterior_group.variables, (
        "Should include yearly_seasonality_contribution when has_seasonality=True"
    )


def test_schema_factory_includes_time_index():
    """Test that from_model_config includes time_index for time-varying effects."""
    # Arrange & Act
    schema = MMMIdataSchema.from_model_config(
        custom_dims=(),
        has_controls=False,
        has_seasonality=False,
        time_varying=True,  # Request time-varying effects
    )

    # Assert
    constant_data_group = schema.groups["constant_data"]
    assert "time_index" in constant_data_group.variables, (
        "Should include time_index when time_varying=True"
    )


@pytest.mark.parametrize(
    "custom_dims",
    [
        ("country",),
        ("country", "region"),
        ("geo",),
    ],
    ids=["single_dim", "two_dims", "custom_name"],
)
def test_schema_factory_handles_custom_dims(custom_dims):
    """Test that from_model_config handles custom dimensions."""
    # Arrange & Act
    schema = MMMIdataSchema.from_model_config(
        custom_dims=custom_dims,
        has_controls=False,
        has_seasonality=False,
        time_varying=False,
    )

    # Assert
    assert schema.custom_dims == custom_dims, (
        f"Schema should store custom_dims: {custom_dims}"
    )

    # Check that variables have correct dimensions
    constant_data_group = schema.groups["constant_data"]
    channel_data_schema = constant_data_group.variables["channel_data"]

    # channel_data should have dims: ("date", *custom_dims, "channel")
    expected_dims = ("date", *custom_dims, "channel")
    assert channel_data_schema.dims == expected_dims, (
        f"channel_data should have dims {expected_dims}, got {channel_data_schema.dims}"
    )


# ============================================================================
# Category 4: Complete Schema Validation Tests
# ============================================================================


def test_valid_idata_passes_validation(
    basic_schema: MMMIdataSchema, valid_basic_idata: az.InferenceData
) -> None:
    """Test that valid InferenceData passes schema validation."""
    # Act
    errors = basic_schema.validate(valid_basic_idata)

    # Assert
    assert errors == [], (
        f"Valid InferenceData should pass validation, got errors: {errors}"
    )


def test_invalid_idata_collects_all_errors():
    """Test that validation collects all errors."""
    # Arrange
    schema = MMMIdataSchema.from_model_config(
        custom_dims=(),
        has_controls=True,  # Require controls
        has_seasonality=True,  # Require seasonality
        time_varying=False,
    )

    # Create idata with MULTIPLE problems:
    # 1. Missing constant_data group
    # 2. Missing control_contribution in posterior
    # 3. Missing yearly_seasonality_contribution in posterior
    dates = pd.date_range("2024-01-01", periods=52, freq="W")
    channels = ["TV", "Radio"]

    idata = az.InferenceData(
        # Missing: constant_data group
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    np.random.normal(size=(2, 10, 52, 2)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": dates, "channel": channels},
                ),
                "mu": xr.DataArray(
                    np.random.normal(size=(2, 10, 52)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
                # Missing: control_contribution
                # Missing: yearly_seasonality_contribution
            }
        ),
    )

    # Act
    errors = schema.validate(idata)

    # Assert - Should have multiple errors
    assert len(errors) >= 3, (
        f"Expected at least 3 errors (missing group + 2 missing variables), got {len(errors)}"
    )

    # Check that all problems are mentioned
    all_errors_str = " ".join(errors)
    assert "constant_data" in all_errors_str, (
        f"Errors should mention missing constant_data: {errors}"
    )
    assert "control_contribution" in all_errors_str, (
        f"Errors should mention missing control_contribution: {errors}"
    )
    assert "yearly_seasonality_contribution" in all_errors_str, (
        f"Errors should mention missing yearly_seasonality_contribution: {errors}"
    )


def test_validate_or_raise_raises_with_errors(basic_schema: MMMIdataSchema) -> None:
    """Test that validate_or_raise raises ValueError with error details."""
    # Create invalid idata (missing required group)
    idata = az.from_dict(prior={"a": np.ones((2, 10))})  # Only prior, missing posterior

    # Act & Assert
    with pytest.raises(ValueError, match=r"InferenceData validation failed"):
        basic_schema.validate_or_raise(idata)


def test_validate_or_raise_silent_when_valid(
    basic_schema: MMMIdataSchema, valid_basic_idata: az.InferenceData
) -> None:
    """Test that validate_or_raise returns None for valid idata."""
    # Act - should not raise
    result = basic_schema.validate_or_raise(valid_basic_idata)

    # Assert
    assert result is None, "validate_or_raise should return None for valid idata"
