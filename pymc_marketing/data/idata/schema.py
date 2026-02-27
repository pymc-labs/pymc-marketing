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
"""Pydantic schemas for validating InferenceData structure."""

from typing import Literal

import arviz as az
import xarray as xr
from pydantic import BaseModel, Field

# Type aliases for time aggregation
Frequency = Literal["original", "weekly", "monthly", "quarterly", "yearly", "all_time"]


class VariableSchema(BaseModel):
    """Schema for a single variable in InferenceData.

    Validates the structure (dimensions and dtype) of xarray.DataArray
    variables within InferenceData groups.

    Parameters
    ----------
    name : str
        Variable name
    required : bool, default True
        Whether this variable must be present
    dims : tuple of str or "*"
        Expected dimension names. Use "*" to accept any dimensions.
    dtype : str, tuple of str, or None
        Expected numpy dtype(s) (e.g., "float64", "int64", or ("float64", "int64")).
        Use None to skip dtype validation.
    description : str, default ""
        Human-readable description of this variable

    Examples
    --------
    >>> schema = VariableSchema(
    ...     name="channel_contribution",
    ...     dims=("date", "channel"),
    ...     dtype="float64",
    ...     required=True,
    ... )
    >>> data_array = xr.DataArray(...)
    >>> errors = schema.validate_variable(data_array)
    >>> if errors:
    ...     print("Validation errors:", errors)
    """

    name: str
    required: bool = True
    dims: tuple[str, ...] | Literal["*"]
    dtype: str | tuple[str, ...] | None = None
    description: str = ""

    def validate_variable(self, data_array: xr.DataArray) -> list[str]:
        """Validate variable structure.

        Parameters
        ----------
        data_array : xr.DataArray
            The data array to validate

        Returns
        -------
        list of str
            Validation errors (empty if valid)

        Examples
        --------
        >>> errors = schema.validate_variable(data_array)
        >>> assert errors == [], f"Validation failed: {errors}"
        """
        errors = []

        # Check dimensions
        if self.dims != "*":
            if set(data_array.dims) != set(self.dims):
                errors.append(
                    f"Variable '{self.name}' has dims {data_array.dims}, "
                    f"expected {self.dims}"
                )

        # Check dtype
        if self.dtype:
            if isinstance(self.dtype, str):
                allowed_dtypes: tuple[str, ...] = (self.dtype,)
            else:
                allowed_dtypes = self.dtype

            if str(data_array.dtype) not in allowed_dtypes:
                errors.append(
                    f"Variable '{self.name}' has dtype {data_array.dtype}, "
                    f"expected one of {allowed_dtypes}"
                )

        return errors


class InferenceDataGroupSchema(BaseModel):
    """Schema for a single InferenceData group.

    Validates that a group exists (if required) and contains expected
    variables with correct structure.

    Parameters
    ----------
    name : Literal
        Group name (e.g., "posterior", "constant_data")
    required : bool, default True
        Whether this group must be present
    variables : dict of str to VariableSchema
        Expected variables in this group

    Examples
    --------
    >>> schema = InferenceDataGroupSchema(
    ...     name="posterior",
    ...     required=True,
    ...     variables={
    ...         "channel_contribution": VariableSchema(
    ...             name="channel_contribution",
    ...             dims=("date", "channel"),
    ...             dtype="float64",
    ...         ),
    ...     },
    ... )
    >>> errors = schema.validate_group(idata)
    """

    name: Literal[
        "posterior",
        "prior",
        "constant_data",
        "observed_data",
        "fit_data",
        "posterior_predictive",
        "prior_predictive",
        "sample_stats",
        "posterior_predictive_constant_data",
    ]
    required: bool = True
    variables: dict[str, VariableSchema] = Field(default_factory=dict)

    def validate_group(self, idata: az.InferenceData) -> list[str]:
        """Validate group exists and contains expected variables.

        Parameters
        ----------
        idata : az.InferenceData
            InferenceData object to validate

        Returns
        -------
        list of str
            Validation errors (empty if valid)
        """
        errors = []

        # Check group exists
        if self.required and not hasattr(idata, self.name):
            errors.append(f"Required group '{self.name}' not found in InferenceData")
            return errors

        if not hasattr(idata, self.name):
            return errors  # Optional group not present

        group = getattr(idata, self.name)

        # Check variables
        for var_name, var_schema in self.variables.items():
            if var_schema.required and var_name not in group:
                errors.append(
                    f"Required variable '{var_name}' not found in group '{self.name}'"
                )
            elif var_name in group:
                # Validate variable structure
                errors.extend(var_schema.validate_variable(group[var_name]))

        return errors


class MMMIdataSchema(BaseModel):
    """Complete schema for multidimensional MMM InferenceData.

    Defines expected groups and variables for a fitted MMM model,
    with configuration based on model settings.

    Parameters
    ----------
    model_type : Literal["mmm"], default "mmm"
        Model type (currently only MMM supported)
    groups : dict of str to InferenceDataGroupSchema
        Schema for each InferenceData group
    custom_dims : tuple of str, default ()
        Custom dimensions beyond standard (date, channel)

    Examples
    --------
    >>> schema = MMMIdataSchema.from_model_config(
    ...     custom_dims=("country",),
    ...     has_controls=True,
    ...     has_seasonality=False,
    ...     time_varying=False,
    ... )
    >>> errors = schema.validate(mmm.idata)
    >>> if errors:
    ...     print("Validation errors:", errors)
    """

    model_type: Literal["mmm"] = "mmm"
    groups: dict[str, InferenceDataGroupSchema]
    custom_dims: tuple[str, ...] = Field(
        default=(), description="Custom dimensions beyond standard (date, channel)"
    )

    @classmethod
    def from_model_config(
        cls,
        custom_dims: tuple[str, ...] = (),
        has_controls: bool = False,
        has_seasonality: bool = False,
        time_varying: bool = False,
    ) -> "MMMIdataSchema":
        """Create schema based on model configuration.

        Parameters
        ----------
        custom_dims : tuple of str, default ()
            Custom dimensions (e.g., ("country", "region"))
        has_controls : bool, default False
            Whether model includes control variables
        has_seasonality : bool, default False
            Whether model includes yearly seasonality
        time_varying : bool, default False
            Whether model has time-varying effects

        Returns
        -------
        MMMIdataSchema
            Schema configured for this model type

        Examples
        --------
        >>> # Basic MMM with no extras
        >>> schema = MMMIdataSchema.from_model_config()
        >>>
        >>> # MMM with controls and custom dimensions
        >>> schema = MMMIdataSchema.from_model_config(
        ...     custom_dims=("country",),
        ...     has_controls=True,
        ... )
        """
        groups = {}

        # Constant data group
        constant_data_vars = {
            "channel_data": VariableSchema(
                name="channel_data",
                dims=("date", *custom_dims, "channel"),
                dtype=("float64", "float32", "int64", "int32"),
                description="Raw channel spend/impressions data",
                required=True,
            ),
            "target_data": VariableSchema(
                name="target_data",
                dims=("date", *custom_dims),
                dtype=("float64", "float32", "int64", "int32"),
                description="Raw target variable",
                required=True,
            ),
            "channel_scale": VariableSchema(
                name="channel_scale",
                dims="*",  # Varies by scaling config
                dtype=("float64", "float32", "int64", "int32"),
                description="Scaling factors for channels",
                required=True,
            ),
            "target_scale": VariableSchema(
                name="target_scale",
                dims="*",  # Varies by scaling config
                dtype=("float64", "float32", "int64", "int32"),
                description="Scaling factor for target",
                required=True,
            ),
        }

        constant_data_vars["channel_spend"] = VariableSchema(
            name="channel_spend",
            dims=("date", *custom_dims, "channel"),
            dtype=("float64", "float32", "int64", "int32"),
            description=(
                "Channel spend in monetary units. Precomputed as "
                "channel_data * cost_per_unit when cost_per_unit is provided; "
                "otherwise absent (falls back to channel_data)."
            ),
            required=False,
        )

        if has_controls:
            constant_data_vars["control_data_"] = VariableSchema(
                name="control_data_",
                dims=("date", *custom_dims, "control"),
                dtype=("float64", "float32", "int64", "int32"),
                description="Control variable data",
                required=False,
            )

        if time_varying:
            constant_data_vars["time_index"] = VariableSchema(
                name="time_index",
                dims=("date",),
                dtype=("float64", "float32", "int64", "int32"),
                description="Integer time index",
                required=True,
            )

        if has_seasonality:
            constant_data_vars["dayofyear"] = VariableSchema(
                name="dayofyear",
                dims=("date",),
                dtype=("int64", "int32"),
                description="Day of year (1-365)",
                required=True,
            )

        groups["constant_data"] = InferenceDataGroupSchema(
            name="constant_data", required=True, variables=constant_data_vars
        )

        # Posterior group
        # Note: Posterior variables include 'chain' and 'draw' dimensions from MCMC sampling
        posterior_vars = {
            "channel_contribution": VariableSchema(
                name="channel_contribution",
                dims=("chain", "draw", "date", *custom_dims, "channel"),
                dtype="float64",
                description="Channel contributions (scaled)",
                required=True,
            ),
            "mu": VariableSchema(
                name="mu",
                dims=("chain", "draw", "date", *custom_dims),
                dtype="float64",
                description="Total predicted mean (scaled)",
                required=False,
            ),
        }

        if has_controls:
            posterior_vars["control_contribution"] = VariableSchema(
                name="control_contribution",
                dims=("chain", "draw", "date", *custom_dims, "control"),
                dtype="float64",
                description="Control variable contributions",
                required=True,
            )

        if has_seasonality:
            posterior_vars["yearly_seasonality_contribution"] = VariableSchema(
                name="yearly_seasonality_contribution",
                dims=("chain", "draw", "date", *custom_dims),
                dtype="float64",
                description="Yearly seasonality contribution",
                required=True,
            )

        groups["posterior"] = InferenceDataGroupSchema(
            name="posterior", required=True, variables=posterior_vars
        )

        # Fit data group (dynamic variables, just check it exists)
        groups["fit_data"] = InferenceDataGroupSchema(
            name="fit_data",
            required=True,
            variables={},  # Dynamic based on input columns
        )

        # Posterior predictive group (optional)
        groups["posterior_predictive"] = InferenceDataGroupSchema(
            name="posterior_predictive",
            required=False,  # Only after prediction
            variables={
                "y": VariableSchema(
                    name="y",
                    dims=("chain", "draw", "date", *custom_dims),
                    dtype="float64",
                    description="Posterior predictive samples",
                    required=True,
                )
            },
        )

        return cls(groups=groups, custom_dims=custom_dims)

    def validate(self, idata: az.InferenceData) -> list[str]:
        """Validate InferenceData against schema.

        Parameters
        ----------
        idata : az.InferenceData
            InferenceData object to validate

        Returns
        -------
        list of str
            All validation errors (empty if valid)

        Examples
        --------
        >>> errors = schema.validate(mmm.idata)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"  - {error}")
        """
        all_errors = []

        for _group_name, group_schema in self.groups.items():
            errors = group_schema.validate_group(idata)
            all_errors.extend(errors)

        return all_errors

    def validate_or_raise(self, idata: az.InferenceData) -> None:
        """Validate InferenceData, raising detailed exception if invalid.

        Parameters
        ----------
        idata : az.InferenceData
            InferenceData object to validate

        Raises
        ------
        ValueError
            If validation fails, with detailed error messages

        Examples
        --------
        >>> schema.validate_or_raise(mmm.idata)  # Raises if invalid
        """
        errors = self.validate(idata)

        if errors:
            error_msg = "InferenceData validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise ValueError(error_msg)
