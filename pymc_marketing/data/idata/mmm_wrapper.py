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
"""MMMIDataWrapper class for validated access to InferenceData."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from pymc_marketing.data.idata.schema import Frequency

if TYPE_CHECKING:
    from pymc_marketing.mmm.multidimensional import MMM


class MMMIDataWrapper:
    """Codified wrapper around InferenceData for MMM models.

    Provides validated access to data and common transformations.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object from fitted MMM model
    schema : MMMIdataSchema, optional
        Schema to validate against. If None, validation skipped.
    validate_on_init : bool, default True
        Whether to validate idata structure on initialization

    Examples
    --------
    >>> wrapper = MMMIDataWrapper(mmm.idata)
    >>>
    >>> # Access observed data
    >>> observed = wrapper.get_target()
    >>>
    >>> # Get contributions in original scale
    >>> contributions = wrapper.get_contributions(original_scale=True)
    """

    def __init__(
        self,
        idata: az.InferenceData,
        schema: Any | None = None,
        validate_on_init: bool = True,
    ):
        self.idata = idata
        self.schema = schema

        if validate_on_init and schema is not None:
            schema.validate_or_raise(idata)

        # Cache for expensive operations
        self._cache: dict[str, Any] = {}

    @classmethod
    def from_mmm(
        cls,
        mmm: MMM,
        idata: az.InferenceData | None = None,
    ) -> MMMIDataWrapper:
        """Create an MMMIDataWrapper from a fitted MMM model.

        Builds the appropriate schema from the model configuration
        and wraps the provided (or model's own) InferenceData.

        Parameters
        ----------
        mmm : MMM
            Fitted MMM model instance.
        idata : az.InferenceData, optional
            InferenceData to wrap. If None, uses ``mmm.idata``.

        Returns
        -------
        MMMIDataWrapper
            Wrapper with schema derived from the model configuration.

        Raises
        ------
        ValueError
            If no idata is provided and ``mmm.idata`` is not available.

        Examples
        --------
        >>> wrapper = MMMIDataWrapper.from_mmm(mmm)
        >>> wrapper = MMMIDataWrapper.from_mmm(mmm, idata=custom_idata)
        """
        from pymc_marketing.data.idata.schema import MMMIdataSchema

        if idata is None:
            if not hasattr(mmm, "idata") or mmm.idata is None:
                raise ValueError(
                    "No idata provided and mmm.idata is not available. "
                    "Either pass idata explicitly or fit the model first."
                )
            idata = mmm.idata

        schema = MMMIdataSchema.from_model_config(
            custom_dims=mmm.dims if hasattr(mmm, "dims") and mmm.dims else (),
            has_controls=bool(mmm.control_columns),
            has_seasonality=bool(mmm.yearly_seasonality),
            time_varying=(
                getattr(mmm, "time_varying_intercept", False)
                or getattr(mmm, "time_varying_media", False)
            ),
        )

        return cls(idata, schema=schema, validate_on_init=False)

    def compare_coords(
        self,
        mmm: MMM,
        variable: str = "channel_data",
    ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Find coordinate mismatches between idata and the model.

        Compares the coordinate values for each custom dimension (i.e. not
        ``"date"``) of the specified variable between this wrapper's idata
        and the fitted PyMC model.

        Parameters
        ----------
        mmm : MMM
            Fitted MMM model instance whose ``model`` attribute contains the
            PyMC model with coordinate metadata.
        variable : str, default ``"channel_data"``
            Name of the model variable whose dimensions are checked.

        Returns
        -------
        tuple of (dict[str, set[str]], dict[str, set[str]])
            ``(in_model_not_idata, in_idata_not_model)`` where:

            - ``in_model_not_idata``: mapping from dimension name to the
              set of coordinate values present in the model but absent from
              the idata.  When a dimension is entirely missing from the
              idata (e.g. dropped by a scalar ``filter_dims`` call), *all*
              of the model's coordinates for that dimension are included.
            - ``in_idata_not_model``: mapping from dimension name to the
              set of coordinate values present in the idata but absent from
              the model (e.g. new labels introduced by ``aggregate_dims``).

            Only dimensions with at least one mismatched coordinate are
            included in each dict; empty dicts mean full compatibility.
        """
        pymc_model = mmm.model
        variable_dims = pymc_model.named_vars_to_dims.get(variable, ())
        idata_var = self.get_channel_spend()

        in_model_not_idata: dict[str, set[str]] = {}
        in_idata_not_model: dict[str, set[str]] = {}
        for dim_name in variable_dims:
            if dim_name == "date":
                continue
            model_coords = {str(v) for v in pymc_model.coords[dim_name]}
            if dim_name not in idata_var.dims:
                in_model_not_idata[dim_name] = model_coords
                continue
            idata_coords = {str(v) for v in idata_var.coords[dim_name].values}
            missing = model_coords - idata_coords
            if missing:
                in_model_not_idata[dim_name] = missing
            extra = idata_coords - model_coords
            if extra:
                in_idata_not_model[dim_name] = extra

        return in_model_not_idata, in_idata_not_model

    # ==================== Scale Accessor Methods ====================

    def get_channel_scale(self) -> xr.DataArray:
        """Get channel scaling factor used during model fitting.

        Returns
        -------
        xr.DataArray
            Channel scale values with dims matching channel dimensions.
            Typically has dims like ("channel",) for simple models or
            ("country", "channel") for panel models.

        Raises
        ------
        ValueError
            If channel_scale not found in constant_data

        Examples
        --------
        >>> channel_scale = mmm.data.get_channel_scale()
        >>> # Convert original scale value to scaled space
        >>> max_scaled = 1000 / float(channel_scale.mean())
        """
        if not (
            hasattr(self.idata, "constant_data")
            and "channel_scale" in self.idata.constant_data
        ):
            raise ValueError(
                "channel_scale not found in constant_data. "
                "Expected 'channel_scale' variable in idata.constant_data."
            )
        return self.idata.constant_data.channel_scale

    def get_target_scale(self) -> xr.DataArray:
        """Get target scaling factor used during model fitting.

        Returns
        -------
        xr.DataArray
            Target scale values. May have dims for panel models
            (e.g., ("country",)) or be scalar.

        Raises
        ------
        ValueError
            If target_scale not found in constant_data

        Examples
        --------
        >>> target_scale = mmm.data.get_target_scale()
        >>> # Convert scaled contribution to original units
        >>> original = scaled_contribution * target_scale
        """
        if not (
            hasattr(self.idata, "constant_data")
            and "target_scale" in self.idata.constant_data
        ):
            raise ValueError(
                "target_scale not found in constant_data. "
                "Expected 'target_scale' variable in idata.constant_data."
            )
        return self.idata.constant_data.target_scale

    # ==================== Observed Data Access ====================

    def get_target(self, original_scale: bool = True) -> xr.DataArray:
        """Get observed target data with consistent access pattern.

        Parameters
        ----------
        original_scale : bool, default True
            Whether to return data in original scale

        Returns
        -------
        xr.DataArray
            Observed target values

        Raises
        ------
        ValueError
            If target data not found in constant_data, or if
            original_scale=False and target_scale is not found
        """
        if not (
            hasattr(self.idata, "constant_data")
            and "target_data" in self.idata.constant_data
        ):
            raise ValueError(
                "Target data not found in constant_data. "
                "Expected 'target_data' variable in idata.constant_data."
            )

        data = self.idata.constant_data.target_data
        if original_scale:
            return data
        else:
            # Scale down using target_scale
            target_scale = self.get_target_scale()
            return data / target_scale

    def get_channel_spend(self) -> xr.DataArray:
        """Get channel spend data in monetary units.

        If ``channel_spend`` exists in ``constant_data`` (set via
        ``cost_per_unit``), returns it directly.  Otherwise falls back to
        ``channel_data`` (backward compatible — assumed already in spend
        units).

        Returns
        -------
        xr.DataArray
            Channel spend values with dims (date, channel) or
            (date, *custom_dims, channel).

        Raises
        ------
        ValueError
            If neither channel_spend nor channel_data found in constant_data.
        """
        if (
            hasattr(self.idata, "constant_data")
            and "channel_spend" in self.idata.constant_data
        ):
            return self.idata.constant_data.channel_spend

        if not (
            hasattr(self.idata, "constant_data")
            and "channel_data" in self.idata.constant_data
        ):
            raise ValueError(
                "Channel data not found in constant_data. "
                "Expected 'channel_data' or 'channel_spend' variable "
                "in idata.constant_data."
            )

        return self.idata.constant_data.channel_data

    def get_channel_data(self) -> xr.DataArray:
        """Get raw channel data in original units (not spend-converted).

        Always returns ``channel_data`` from ``constant_data``, regardless
        of whether ``cost_per_unit`` / ``channel_spend`` has been set.
        Use this when you need data in the units the model was trained on
        (e.g., impressions, clicks) rather than monetary units.

        Returns
        -------
        xr.DataArray
            Raw channel data with dims (date, channel) or
            (date, *custom_dims, channel).

        Raises
        ------
        ValueError
            If channel_data not found in constant_data.
        """
        if not (
            hasattr(self.idata, "constant_data")
            and "channel_data" in self.idata.constant_data
        ):
            raise ValueError(
                "Channel data not found in constant_data. "
                "Expected 'channel_data' variable in idata.constant_data."
            )
        return self.idata.constant_data.channel_data

    @property
    def cost_per_unit(self) -> xr.DataArray | None:
        """Cost per unit conversion factors, computed from stored data.

        Derived on-the-fly as ``channel_spend / channel_data``.
        Returns None if ``channel_spend`` is not present in
        ``idata.constant_data``.

        Returns
        -------
        xr.DataArray or None
            Cost per unit values with dims ("date", *custom_dims, "channel").
            Returns None if cost_per_unit has not been set.
        """
        if not (
            hasattr(self.idata, "constant_data")
            and "channel_spend" in self.idata.constant_data
        ):
            return None

        channel_spend = self.idata.constant_data.channel_spend
        channel_data = self.idata.constant_data.channel_data
        return xr.where(channel_data == 0, np.nan, channel_spend / channel_data)

    # ==================== Contribution Access ====================

    def get_channel_contributions(self, original_scale: bool = True) -> xr.DataArray:
        """Get channel contribution posterior samples.

        Convenience method that delegates to get_contributions() and
        extracts the channel contribution DataArray.

        Parameters
        ----------
        original_scale : bool, default True
            Whether to return contributions in original scale.
            If True, multiplies by target_scale (or uses pre-computed
            _original_scale variable if available).

        Returns
        -------
        xr.DataArray
            Channel contributions with dims (chain, draw, date, channel)
        """
        contributions = self.get_contributions(
            original_scale=original_scale,
            include_baseline=False,
            include_controls=False,
            include_seasonality=False,
        )
        # Extract from Dataset - xarray preserves coordinate structure
        return contributions["channels"]

    def get_contributions(
        self,
        original_scale: bool = True,
        include_baseline: bool = True,
        include_controls: bool = True,
        include_seasonality: bool = True,
    ) -> xr.Dataset:
        """Get all contribution variables in a single dataset.

        Parameters
        ----------
        original_scale : bool, default True
            Whether to return contributions in original scale
        include_baseline : bool, default True
            Include intercept/baseline contribution
        include_controls : bool, default True
            Include control variable contributions (if present)
        include_seasonality : bool, default True
            Include seasonality contributions (if present)

        Returns
        -------
        xr.Dataset
            Dataset with all contribution variables

        Raises
        ------
        ValueError
            If original_scale=True and target_scale is not found in constant_data
        """
        contributions = {}

        # Channel contributions
        # Channels variables - use "channels" (plural) as key to avoid xarray
        # dimension/key name conflict (a key matching a dimension name gets
        # promoted to a coordinate instead of staying as a data variable)
        if original_scale:
            if "channel_contribution_original_scale" in self.idata.posterior:
                contributions["channels"] = (
                    self.idata.posterior.channel_contribution_original_scale
                )
            else:
                # Compute on-the-fly
                channel_contrib = self.idata.posterior.channel_contribution
                target_scale = self.get_target_scale()
                # xarray automatically handles broadcasting when dimensions match
                contributions["channels"] = channel_contrib * target_scale
        else:
            contributions["channels"] = self.idata.posterior.channel_contribution

        # Baseline/intercept
        if include_baseline:
            for var in ["intercept_contribution", "intercept_baseline"]:
                if var in self.idata.posterior:
                    baseline = self.idata.posterior[var]
                    if original_scale:
                        target_scale = self.get_target_scale()
                        contributions["baseline"] = baseline * target_scale
                    else:
                        contributions["baseline"] = baseline
                    break

        # Control variables - use "controls" (plural) as key to avoid xarray
        # dimension/key name conflict (a key matching a dimension name gets
        # promoted to a coordinate instead of staying as a data variable)
        if include_controls and "control_contribution" in self.idata.posterior:
            control = self.idata.posterior.control_contribution
            if original_scale:
                if "control_contribution_original_scale" in self.idata.posterior:
                    contributions["controls"] = (
                        self.idata.posterior.control_contribution_original_scale
                    )
                else:
                    target_scale = self.get_target_scale()
                    contributions["controls"] = control * target_scale
            else:
                contributions["controls"] = control

        # Seasonality
        if (
            include_seasonality
            and "yearly_seasonality_contribution" in self.idata.posterior
        ):
            seasonality = self.idata.posterior.yearly_seasonality_contribution
            if original_scale:
                if (
                    "yearly_seasonality_contribution_original_scale"
                    in self.idata.posterior
                ):
                    contributions["seasonality"] = (
                        self.idata.posterior.yearly_seasonality_contribution_original_scale
                    )
                else:
                    target_scale = self.get_target_scale()
                    contributions["seasonality"] = seasonality * target_scale
            else:
                contributions["seasonality"] = seasonality

        return xr.Dataset(contributions)

    def get_elementwise_roas(self, original_scale: bool = True) -> xr.DataArray:
        """Compute element-wise ROAS (Return on Ad Spend) for each channel.

        ROAS = contribution / spend for each channel at each time point.
        Does NOT account for adstock carryover effects. For true incremental
        ROAS, use :meth:`pymc_marketing.mmm.incrementality.Incrementality.contribution_over_spend`
        or :meth:`pymc_marketing.mmm.summary.MMMSummaryFactory.roas` with
        ``method="incremental"``.

        Parameters
        ----------
        original_scale : bool, default True
            Whether to return contributions in original scale.

        Returns
        -------
        xr.DataArray
            ROAS values with dims (chain, draw, date, channel) plus any custom dims.
            Zero spend values result in NaN to avoid division by zero.

        Examples
        --------
        >>> roas = mmm.data.get_elementwise_roas()
        >>> roas_mean = roas.mean(dim=["chain", "draw"])
        """
        contributions = self.get_channel_contributions(original_scale=original_scale)
        spend = self.get_channel_spend()

        # Handle zero spend - use xr.where to avoid division by zero
        spend_safe = xr.where(spend == 0, np.nan, spend)

        return contributions / spend_safe

    def get_roas(self, original_scale: bool = True) -> xr.DataArray:
        """Compute ROAS (Return on Ad Spend) for each channel.

        .. deprecated::
            Use :meth:`get_elementwise_roas` for element-wise ROAS, or
            :meth:`pymc_marketing.mmm.summary.MMMSummaryFactory.roas` with
            ``method="incremental"`` for correct carryover-aware ROAS.

        ROAS = contribution / spend for each channel. This method uses
        element-wise division and does NOT account for adstock carryover effects.

        Parameters
        ----------
        original_scale : bool, default True
            Whether to return contributions in original scale.

        Returns
        -------
        xr.DataArray
            ROAS values with dims (chain, draw, date, channel) plus any custom dims.
            Zero spend values result in NaN to avoid division by zero.
        """
        warnings.warn(
            "get_roas() is deprecated. Use get_elementwise_roas() for element-wise "
            "ROAS, or summary.roas(method='incremental') for carryover-aware ROAS.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_elementwise_roas(original_scale=original_scale)

    # ==================== Scaling Operations ====================

    def to_original_scale(self, var: str | xr.DataArray) -> xr.DataArray:
        """Transform variable from scaled to original scale.

        Handles three scenarios based on input type:

        1. **String with corresponding '_original_scale' variable**: If posterior
           contains `{var}_original_scale`, returns it directly (e.g., "mu" returns
           posterior["mu_original_scale"] if it exists).

        2. **String ending with '_original_scale'**: Returns the variable directly
           from posterior (e.g., "mu_original_scale" → posterior["mu_original_scale"]).
           Already in original scale, so no transformation is applied.

        3. **String (scaled variable) or xr.DataArray**: Multiplies by target_scale
           to convert from scaled to original space.

        Parameters
        ----------
        var : str or xr.DataArray
            One of:
            - Variable name with existing '_original_scale' version (returns that)
            - Variable name ending with '_original_scale' (returns as-is)
            - Scaled variable name (multiplies by target_scale)
            - DataArray in scaled space (multiplies by target_scale)

        Returns
        -------
        xr.DataArray
            Variable in original scale

        Raises
        ------
        ValueError
            If string variable is not found in posterior, or if
            target_scale is not found in constant_data when scaling is required

        Examples
        --------
        >>> # Get existing original scale variable
        >>> original = wrapper.to_original_scale("channel_contribution")
        >>>
        >>> # Get variable that's already in original scale
        >>> original = wrapper.to_original_scale("mu_original_scale")
        >>>
        >>> # Convert DataArray from scaled to original space
        >>> scaled_data = posterior["mu"]
        >>> original = wrapper.to_original_scale(scaled_data)
        """
        if isinstance(var, str):
            if f"{var}_original_scale" in self.idata.posterior:
                # Corresponding _original_scale variable exists
                return self.idata.posterior[f"{var}_original_scale"]

            if var.endswith("_original_scale"):
                # Variable is already in original scale - return directly
                if var in self.idata.posterior:
                    return self.idata.posterior[var]
                raise ValueError(f"Variable '{var}' not found in posterior")

            # Scaled variable - compute on-the-fly
            if var in self.idata.posterior:
                data = self.idata.posterior[var]
            else:
                raise ValueError(f"Variable '{var}' not found in posterior")
        else:
            data = var

        target_scale = self.get_target_scale()
        return data * target_scale

    def to_scaled(self, var: str | xr.DataArray) -> xr.DataArray:
        """Transform variable from original to scaled space.

        Handles three scenarios based on input type:

        1. **String ending with '_original_scale'**: Returns the corresponding
           base variable from posterior (e.g., "mu_original_scale" → posterior["mu"]).
           The base variable is already in scaled space.

        2. **String without '_original_scale' suffix**: Returns the variable
           directly from posterior (e.g., "channel_contribution" → posterior["channel_contribution"]).
           These variables are already in scaled space, so no transformation is applied.

        3. **xr.DataArray**: Assumes the data is in original scale and divides
           by target_scale to convert to scaled space.

        Parameters
        ----------
        var : str or xr.DataArray
            One of:
            - Variable name ending with '_original_scale' (returns base scaled variable)
            - Variable name in posterior (returns as-is, already scaled)
            - DataArray in original space (divides by target_scale)

        Returns
        -------
        xr.DataArray
            Variable in scaled space

        Raises
        ------
        ValueError
            If string variable is not found in posterior, or if
            target_scale is not found in constant_data when scaling a DataArray

        Examples
        --------
        >>> # Get scaled version of original scale variable
        >>> scaled = wrapper.to_scaled("mu_original_scale")
        >>>
        >>> # Get already-scaled variable directly
        >>> scaled = wrapper.to_scaled("channel_contribution")
        >>>
        >>> # Convert DataArray from original to scaled space
        >>> original_data = posterior["mu"] * constant_data.target_scale
        >>> scaled = wrapper.to_scaled(original_data)
        """
        if isinstance(var, str):
            if var.endswith("_original_scale"):
                # Get base variable name (which is in scaled space)
                base_name = var.replace("_original_scale", "")
                if base_name in self.idata.posterior:
                    return self.idata.posterior[base_name]
                raise ValueError(f"Variable '{base_name}' not found in posterior")

            # Variable name without _original_scale suffix - return directly (already scaled)
            if var in self.idata.posterior:
                return self.idata.posterior[var]
            else:
                raise ValueError(f"Variable '{var}' not found in posterior")

        # DataArray in original space - convert to scaled
        target_scale = self.get_target_scale()
        return var / target_scale

    # ==================== Filtering Operations ====================

    def filter_dates(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> MMMIDataWrapper:
        """Filter to date range, returning new wrapper.

        Delegates to standalone `filter_idata_by_dates` utility function.

        Parameters
        ----------
        start_date : str or pd.Timestamp, optional
            Start date (inclusive)
        end_date : str or pd.Timestamp, optional
            End date (inclusive)

        Returns
        -------
        MMMIDataWrapper
            New wrapper with filtered idata
        """
        if start_date is None and end_date is None:
            return self

        from pymc_marketing.data.idata.utils import filter_idata_by_dates

        filtered_idata = filter_idata_by_dates(self.idata, start_date, end_date)

        return MMMIDataWrapper(
            filtered_idata, schema=self.schema, validate_on_init=False
        )

    def filter_dims(self, **dim_filters) -> MMMIDataWrapper:
        """Filter by custom dimensions, returning new wrapper.

        Delegates to standalone `filter_idata_by_dims` utility function.

        Parameters
        ----------
        **dim_filters
            Dimension filters, e.g., country="US", channel=["TV", "Radio"]

        Returns
        -------
        MMMIDataWrapper
            New wrapper with filtered idata. Schema is set to None when
            any dimension is dropped (single-value scalar filter), since
            the data no longer conforms to the original schema.

        Examples
        --------
        >>> wrapper.filter_dims(country="US")
        >>> wrapper.filter_dims(channel=["TV", "Radio"])
        """
        if not dim_filters:
            return self

        from pymc_marketing.data.idata.utils import filter_idata_by_dims

        filtered_idata = filter_idata_by_dims(self.idata, **dim_filters)

        # When dimensions are dropped, the data no longer conforms to
        # the original schema, so we set schema=None to prevent
        # downstream validation errors (same pattern as aggregate_time).
        schema = self.schema
        if schema is not None:
            for value in dim_filters.values():
                if not isinstance(value, (list, tuple)):
                    schema = None
                    break

        return MMMIDataWrapper(filtered_idata, schema=schema, validate_on_init=False)

    # ==================== Aggregation Operations ====================

    def aggregate_time(
        self,
        period: Frequency,
        method: Literal["sum", "mean"] = "sum",
    ) -> MMMIDataWrapper:
        """Aggregate data over time periods.

        Delegates to standalone `aggregate_idata_time` utility function.

        Parameters
        ----------
        period : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time period to aggregate to. Use "original" for no aggregation.
        method : {"sum", "mean"}, default "sum"
            Aggregation method

        Returns
        -------
        MMMIDataWrapper
            New wrapper with aggregated idata
        """
        from pymc_marketing.data.idata.utils import aggregate_idata_time

        aggregated_idata = aggregate_idata_time(self.idata, period, method)

        # For "all_time", schema no longer applies (date dimension removed)
        schema = None if period == "all_time" else self.schema

        return MMMIDataWrapper(aggregated_idata, schema=schema, validate_on_init=False)

    def aggregate_dims(
        self,
        dim: str,
        values: list[str],
        new_label: str,
        method: Literal["sum", "mean"] = "sum",
    ) -> MMMIDataWrapper:
        """Aggregate multiple dimension values into one.

        Delegates to standalone `aggregate_idata_dims` utility function.

        Parameters
        ----------
        dim : str
            Dimension to aggregate
        values : list of str
            Values to aggregate
        new_label : str
            Label for aggregated value
        method : {"sum", "mean"}, default "sum"
            Aggregation method

        Returns
        -------
        MMMIDataWrapper
            New wrapper with aggregated idata
        """
        from pymc_marketing.data.idata.utils import aggregate_idata_dims

        aggregated_idata = aggregate_idata_dims(
            self.idata, dim, values, new_label, method
        )

        return MMMIDataWrapper(
            aggregated_idata, schema=self.schema, validate_on_init=False
        )

    # ==================== Summary Statistics ====================

    def compute_posterior_summary(
        self,
        var: str,
        hdi_prob: float = 0.94,
        original_scale: bool = True,
    ) -> pd.DataFrame:
        """Compute summary statistics for a variable.

        Parameters
        ----------
        var : str
            Variable name in posterior
        hdi_prob : float, default 0.94
            Probability for HDI computation
        original_scale : bool, default True
            Whether to compute in original scale

        Returns
        -------
        pd.DataFrame
            Summary statistics (mean, median, std, HDI)
        """
        # Get variable
        if original_scale:
            data = self.to_original_scale(var)
        else:
            if var in self.idata.posterior:
                data = self.idata.posterior[var]
            else:
                raise ValueError(f"Variable '{var}' not found in posterior")

        # Use arviz for summary
        summary = az.summary(data, hdi_prob=hdi_prob, kind="stats")

        return summary

    # ==================== Validation ====================

    def validate(self) -> list[str]:
        """Validate idata structure against schema.

        Returns
        -------
        list of str
            Validation errors (empty if valid)
        """
        if self.schema is None:
            raise ValueError("No schema provided for validation")

        return self.schema.validate(self.idata)

    def validate_or_raise(self) -> None:
        """Validate idata structure, raising detailed exception if invalid.

        Call this after operations that modify idata to ensure structure is valid.

        Raises
        ------
        ValueError
            If schema is None (no validation possible) or validation fails

        Examples
        --------
        >>> mmm.add_original_scale_contribution_variable(["channel_contribution"])
        >>> mmm.data.validate_or_raise()  # Explicitly validate new structure
        """
        if self.schema is None:
            raise ValueError("No schema provided for validation")

        errors = self.schema.validate(self.idata)
        if errors:
            raise ValueError(
                "idata validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    # ==================== Convenience Properties ====================

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Get date coordinate."""
        if hasattr(self.idata, "constant_data"):
            return pd.DatetimeIndex(self.idata.constant_data.coords["date"].values)
        elif hasattr(self.idata, "posterior"):
            return pd.DatetimeIndex(self.idata.posterior.coords["date"].values)
        else:
            raise ValueError("Could not find date coordinate in InferenceData")

    @property
    def channels(self) -> list[str]:
        """Get channel coordinate."""
        if hasattr(self.idata, "constant_data"):
            return self.idata.constant_data.coords["channel"].values.tolist()
        elif hasattr(self.idata, "posterior"):
            return self.idata.posterior.coords["channel"].values.tolist()
        else:
            raise ValueError("Could not find channel coordinate in InferenceData")

    @property
    def custom_dims(self) -> list[str]:
        """Get all custom dimension names."""
        standard_dims = {"date", "channel", "control", "fourier_mode", "chain", "draw"}

        if hasattr(self.idata, "constant_data"):
            return [
                dim for dim in self.idata.constant_data.dims if dim not in standard_dims
            ]

        return []
