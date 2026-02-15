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
"""Media Mix Model class."""

import json
import logging
import warnings
from collections.abc import Sequence
from typing import Annotated, Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
from pydantic import Field, InstanceOf, validate_call
from pymc_extras.prior import Prior
from scipy.optimize import OptimizeResult
from xarray import DataArray, Dataset

from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm.base import BaseValidateMMM
from pymc_marketing.mmm.causal import CausalGraphModel
from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
    adstock_from_dict,
)
from pymc_marketing.mmm.components.saturation import (
    SaturationTransformation,
    saturation_from_dict,
)
from pymc_marketing.mmm.fourier import YearlyFourier
from pymc_marketing.mmm.hsgp import SoftPlusHSGP
from pymc_marketing.mmm.lift_test import (
    add_lift_measurements_to_likelihood_from_saturation,
    scale_lift_measurements,
)
from pymc_marketing.mmm.preprocessing import MaxAbsScaleChannels, MaxAbsScaleTarget
from pymc_marketing.mmm.scaling import Scaling, VariableScaling
from pymc_marketing.mmm.tvp import create_time_varying_gp_multiplier, infer_time_index
from pymc_marketing.mmm.utility import UtilityFunctionType, average_response
from pymc_marketing.mmm.utils import (
    apply_sklearn_transformer_across_dim,
    create_new_spend_data,
)
from pymc_marketing.mmm.validating import ValidateControlColumns
from pymc_marketing.model_builder import _handle_deprecate_pred_argument
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.model_graph import deterministics_to_flat

__all__ = ["MMM", "BaseMMM"]

DEFAULT_HDI_PROB = 0.94


class BaseMMM(BaseValidateMMM):
    """Base class for a media mix model using Delayed Adstock and Logistic Saturation (see [1]_).

    References
    ----------
    .. [1] Jin, Yuxue, et al. "Bayesian methods for media mix modeling with carryover and shape effects." (2017).

    """

    _model_name: str = "BaseMMM"
    _model_type: str = "BaseValidateMMM"
    version: str = "0.0.3"

    @validate_call
    def __init__(
        self,
        date_column: str = Field(..., description="Column name of the date variable."),
        channel_columns: list[str] = Field(
            min_length=1, description="Column names of the media channel variables."
        ),
        adstock: InstanceOf[AdstockTransformation] = Field(
            ..., description="Type of adstock transformation to apply."
        ),
        saturation: InstanceOf[SaturationTransformation] = Field(
            ..., description="Type of saturation transformation to apply."
        ),
        time_varying_intercept: bool = Field(
            False, description="Whether to consider time-varying intercept."
        ),
        time_varying_media: bool = Field(
            False, description="Whether to consider time-varying media contributions."
        ),
        model_config: dict | None = Field(None, description="Model configuration."),
        sampler_config: dict | None = Field(None, description="Sampler configuration."),
        validate_data: bool = Field(
            True, description="Whether to validate the data before fitting to model"
        ),
        control_columns: Annotated[
            list[str],
            Field(
                min_length=1,
                description="Column names of control variables to be added as additional regressors",
            ),
        ]
        | None = None,
        yearly_seasonality: Annotated[
            int,
            Field(
                gt=0, description="Number of Fourier modes to model yearly seasonality."
            ),
        ]
        | None = None,
        adstock_first: bool = Field(
            True, description="Whether to apply adstock first."
        ),
        dag: str | None = Field(
            None,
            description="Optional DAG provided as a string Dot format for causal identification.",
        ),
        treatment_nodes: list[str] | tuple[str] | None = Field(
            None,
            description="Column names of the variables of interest to identify causal effects on outcome.",
        ),
        outcome_node: str | None = Field(
            None, description="Name of the outcome variable."
        ),
        scaling: InstanceOf[Scaling] | dict | None = Field(
            None, description="Scaling configuration for the model."
        ),
    ) -> None:
        """Define the constructor method.

        Parameter
        ---------
        date_column : str
            Column name of the date variable. Must be parsable using ~pandas.to_datetime.
        channel_columns : List[str]
            Column names of the media channel variables.
        adstock : AdstockTransformation
            Type of adstock transformation to apply.
        saturation : SaturationTransformation
            Type of saturation transformation to apply.
        time_varying_intercept : bool, optional
            Whether to consider time-varying intercept, by default False.
            Because the `time-varying` variable is centered around 1 and acts as a multiplier,
            the variable `intercept_baseline` now represents the mean of the time-varying intercept.
        time_varying_media : bool, optional
            Whether to consider time-varying media contributions, by default False.
            The `time-varying-media` creates a time media variable centered around 1,
            this variable acts as a global multiplier (scaling factor) for all channels,
            meaning all media channels share the same latent fluctuation.
        model_config : Dictionary, optional
            Dictionary of parameters that initialise model configuration.
            Class-default defined by the user default_model_config method.
        sampler_config : Dictionary, optional
            Dictionary of parameters that initialise sampler configuration.
            Class-default defined by the user default_sampler_config method.
        validate_data : bool, optional
            Whether to validate the data before fitting to model, by default True.
        control_columns : Optional[List[str]], optional
            Column names of control variables to be added as additional regressors, by default None
        yearly_seasonality : Optional[int], optional
            Number of Fourier modes to model yearly seasonality, by default None.
        adstock_first : bool, optional
            Whether to apply adstock first, by default True.
        dag : Optional[str], optional
            Optional DAG provided as a string Dot format for causal modeling, by default None.
        treatment_nodes : Optional[list[str]], optional
            Column names of the variables of interest to identify causal effects on outcome.
        outcome_node : Optional[str], optional
            Name of the outcome variable, by default None.
        scaling : Scaling | dict | None, optional
            Scaling configuration for the model. If None, defaults to max scaling for both target and channels.
            Can be a Scaling object or a dict that will be converted to a Scaling object.
        """
        warnings.warn(
            """
            The MMM class is deprecated and will be removed in a future version (in version 0.20.0).
            Please use the multidimensional MMM class instead.
            That is, `from pymc_marketing.mmm.multidimensional import MMM`.
            All our documentation has been updated to reflect this change.
            Refer to the migration guide for more details: https://www.pymc-marketing.io/en/latest/notebooks/mmm/mmm_migration_guide.html
            """,
            FutureWarning,
            stacklevel=2,
        )

        self.control_columns = control_columns
        self.time_varying_intercept = time_varying_intercept
        self.time_varying_media = time_varying_media
        self.date_column = date_column
        self.validate_data = validate_data
        self.adstock_first = adstock_first

        # Initialize scaling configuration similar to multidimensional MMM
        if isinstance(scaling, dict):
            scaling = scaling.copy()

            if "channel" not in scaling:
                scaling["channel"] = VariableScaling(method="max", dims=())
            if "target" not in scaling:
                scaling["target"] = VariableScaling(method="max", dims=())

            scaling = Scaling(**scaling)

        self.scaling: Scaling = scaling or Scaling(
            target=VariableScaling(method="max", dims=()),
            channel=VariableScaling(method="max", dims=()),
        )

        model_config = model_config or {}
        model_config = parse_model_config(
            model_config,  # type: ignore
            hsgp_kwargs_fields=["intercept_tvp_config", "media_tvp_config"],
        )

        self.adstock, self.saturation = adstock, saturation
        del adstock, saturation
        if model_config is not None:
            # self.default_model_config accesses self.adstock and self.saturation
            self.adstock = self.adstock.with_updated_priors(
                {**self.default_model_config, **model_config}
            )
            self.saturation = self.saturation.with_updated_priors(
                {**self.default_model_config, **model_config}
            )
        self.adstock = self.adstock.with_default_prior_dims(("channel",))
        self.saturation = self.saturation.with_default_prior_dims(("channel",))

        super().__init__(
            date_column=date_column,
            channel_columns=channel_columns,
            model_config=model_config,
            sampler_config=sampler_config,
        )

        self.yearly_seasonality = yearly_seasonality

        self.dag = dag
        self.treatment_nodes = treatment_nodes
        self.outcome_node = outcome_node

        # Initialize causal graph if provided
        if self.dag is not None and self.outcome_node is not None:
            if self.treatment_nodes is None:
                self.treatment_nodes = self.channel_columns
                warnings.warn(
                    "No treatment nodes provided, using channel columns as treatment nodes.",
                    stacklevel=2,
                )
            self.causal_graphical_model = CausalGraphModel.build_graphical_model(
                graph=self.dag,
                treatment=self.treatment_nodes,
                outcome=self.outcome_node,
            )

            self.control_columns = self.causal_graphical_model.compute_adjustment_sets(
                control_columns=self.control_columns,
                channel_columns=self.channel_columns,
            )

            if "yearly_seasonality" not in self.causal_graphical_model.adjustment_set:
                warnings.warn(
                    "Yearly seasonality excluded as it's not required for adjustment.",
                    stacklevel=2,
                )
                self.yearly_seasonality = None

        if self.yearly_seasonality is not None:
            self.yearly_fourier = YearlyFourier(
                n_order=self.yearly_seasonality,
                prefix="fourier_mode",
                prior=self.model_config["gamma_fourier"],
                variable_name="gamma_fourier",
            )

    def post_sample_model_transformation(self) -> None:
        """Post-sample model transformation in order to store the HSGP state from fit."""
        names = []
        if self.time_varying_intercept:
            names.extend(
                SoftPlusHSGP.deterministics_to_replace("intercept_latent_process")
            )
        if self.time_varying_media:
            names.extend(
                SoftPlusHSGP.deterministics_to_replace(
                    "media_temporal_latent_multiplier"
                )
            )
        if not names:
            return

        self.model = deterministics_to_flat(self.model, names=names)

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration for the model.

        Returns
        -------
        dict
            Empty dictionary.
        """
        return {}

    @property
    def output_var(self) -> Literal["y"]:
        """Define target variable for the model.

        Returns
        -------
        str
            The target variable for the model.
        """
        return "y"

    def _generate_and_preprocess_model_data(  # type: ignore
        self, X: pd.DataFrame | pd.Series, y: pd.Series | np.ndarray
    ) -> None:
        """Apply preprocessing to the data before fitting the model.

        If validate is True, it will check if the data is valid for the model.
        sets self.model_coords based on provided dataset

        Parameters
        ----------
        X : Union[pd.DataFrame, pd.Series], shape (n_obs, n_features)
        y : Union[pd.Series, np.ndarray], shape (n_obs,)

        Sets
        ----
        preprocessed_data : Dict[str, Union[pd.DataFrame, pd.Series]]
            Preprocessed data for the model.
        X : pd.DataFrame
            A filtered version of the input `X`, such that it is guaranteed that
            it contains only the `date_column`, the columns that are specified
            in the `channel_columns` and `control_columns`, and fourier features
            if `yearly_seasonality=True`.
        y : Union[pd.Series, np.ndarray]
            The target variable for the model (as provided).
        _time_index : np.ndarray
            The index of the date column. Used by TVP
        _time_index_mid : int
            The middle index of the date index. Used by TVP.
        _time_resolution: int
            The time resolution of the date index. Used by TVP.

        """
        try:
            date_data = pd.to_datetime(X[self.date_column])
        except Exception as e:
            raise ValueError(
                f"Could not convert {self.date_column} to datetime. Please check the date format."
            ) from e

        coords: dict[str, Any] = {
            "channel": self.channel_columns,
            "date": date_data,
        }

        # Build X_data efficiently by selecting columns once
        columns_to_select = [self.date_column, *self.channel_columns]
        if self.control_columns is not None:
            columns_to_select.extend(self.control_columns)
            coords["control"] = self.control_columns

        # Create X_data with proper date column in one operation
        # Cast to DataFrame to satisfy mypy type checking
        X_data = pd.DataFrame(X[columns_to_select])
        X_data[self.date_column] = date_data

        self.model_coords = coords
        if self.validate_data:
            self.validate("X", X_data)
            self.validate("y", y)
        # Store raw data instead of applying scaling transformations
        # We'll apply scaling within the model graph like multidimensional MMM
        self.preprocessed_data: dict[str, pd.DataFrame | pd.Series | np.ndarray] = {
            "X": X_data,  # Store raw data
            "y": y,  # Store raw data
        }

        # Still fit the transformers for backward compatibility
        # but don't apply the transformations to the data
        try:
            # Fit transformers without transforming data (for backward compatibility)
            if hasattr(self, "max_abs_scale_target_data"):
                self.max_abs_scale_target_data(y)
            if hasattr(self, "max_abs_scale_channel_data"):
                self.max_abs_scale_channel_data(X_data)
        except Exception as e:
            # If transformer fitting fails, continue without them
            import warnings

            warnings.warn(f"Failed to fit transformers: {e}", UserWarning, stacklevel=2)
        self.X: pd.DataFrame = X_data
        self.y: pd.Series | np.ndarray = y

        if self.time_varying_intercept | self.time_varying_media:
            self._time_index = np.arange(0, X.shape[0])
            self._time_index_mid = X.shape[0] // 2
            self._time_resolution = (
                self.X[self.date_column].iloc[1] - self.X[self.date_column].iloc[0]
            ).days

    def _compute_scale_for_data(
        self, data: np.ndarray, method: str, axis: int | None = None
    ) -> np.ndarray | float:
        """Compute scale for data using the specified method.

        Parameters
        ----------
        data : np.ndarray
            Data to compute scale for
        method : str
            Scaling method ('max' or 'mean')
        axis : int | None
            Axis along which to compute scale

        Returns
        -------
        np.ndarray | float
            Computed scale(s)
        """
        if method == "max":
            scale = np.abs(data).max(axis=axis)
        elif method == "mean":
            scale = np.abs(data).mean(axis=axis)
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        # Avoid division by zero using numpy.maximum for efficiency
        # This works for both scalars and arrays
        scale = np.maximum(scale, 1.0)

        return scale

    def _compute_scales(self) -> None:
        """Compute and save scaling factors for channels and target."""
        # Get raw data
        X_data = self.preprocessed_data["X"]
        if not isinstance(X_data, pd.DataFrame):
            raise TypeError("X data must be a DataFrame for scaling computation")

        # Use pandas/numpy efficient operations - avoid redundant .values call
        channel_data = X_data[self.channel_columns].to_numpy()
        target_data = np.atleast_1d(np.asarray(self.preprocessed_data["y"]))

        # Compute scales based on scaling configuration
        self.channel_scale = self._compute_scale_for_data(
            channel_data, self.scaling.channel.method, axis=0
        )
        target_scale = self._compute_scale_for_data(
            target_data, self.scaling.target.method, axis=None
        )
        # Ensure target_scale is a Python float (convert from numpy scalar if needed)
        self.target_scale = float(target_scale)

    def create_idata_attrs(self) -> dict[str, str]:
        """Create attributes for the inference data.

        Returns
        -------
        dict[str, str]
            The attributes for the inference data.

        """
        attrs = super().create_idata_attrs()
        attrs["date_column"] = json.dumps(self.date_column)
        attrs["adstock"] = json.dumps(self.adstock.to_dict())
        attrs["saturation"] = json.dumps(self.saturation.to_dict())
        attrs["adstock_first"] = json.dumps(self.adstock_first)
        attrs["control_columns"] = json.dumps(self.control_columns)
        attrs["channel_columns"] = json.dumps(self.channel_columns)
        attrs["validate_data"] = json.dumps(self.validate_data)
        attrs["yearly_seasonality"] = json.dumps(self.yearly_seasonality)
        attrs["time_varying_intercept"] = json.dumps(self.time_varying_intercept)
        attrs["time_varying_media"] = json.dumps(self.time_varying_media)
        attrs["dag"] = json.dumps(self.dag)
        attrs["treatment_nodes"] = json.dumps(self.treatment_nodes)
        attrs["outcome_node"] = json.dumps(self.outcome_node)

        # Serialize scaling configuration
        if hasattr(self, "scaling") and self.scaling is not None:
            attrs["scaling"] = json.dumps(
                {
                    "target": {
                        "method": self.scaling.target.method,
                        "dims": self.scaling.target.dims,
                    },
                    "channel": {
                        "method": self.scaling.channel.method,
                        "dims": self.scaling.channel.dims,
                    },
                }
            )
        else:
            attrs["scaling"] = json.dumps(None)

        return attrs

    def forward_pass(self, x: pt.TensorVariable | npt.NDArray) -> pt.TensorVariable:
        """Transform channel input into target contributions of each channel.

        This method handles the ordering of the adstock and saturation
        transformations.

        This method must be called from without a pm.Model context but not
        necessarily in the instance's model. A dim named "channel" is required
        associated with the number of columns of `x`.

        Parameters
        ----------
        x : pt.TensorVariable | npt.NDArray
            The channel input which could be spends or impressions

        Returns
        -------
        The contributions associated with the channel input

        """
        first, second = (
            (self.adstock, self.saturation)
            if self.adstock_first
            else (self.saturation, self.adstock)
        )

        return second.apply(x=first.apply(x=x, dims="channel"), dims="channel")

    def _create_scaled_data_variables(
        self, channel_data_value: pd.DataFrame, target_value: pd.Series | np.ndarray
    ) -> tuple[
        pt.TensorVariable, pt.TensorVariable, pt.TensorVariable, pt.TensorVariable
    ]:
        """Create scaled data variables for the model.

        Parameters
        ----------
        channel_data_value : pd.DataFrame
            Raw channel data
        target_value : pd.Series | np.ndarray
            Raw target data

        Returns
        -------
        tuple
            (channel_data_scaled, target_scaled, channel_scale, target_scale)
        """
        # Store scaling factors as pm.Data
        channel_scale_ = pm.Data(
            "channel_scale",
            self.channel_scale,
            dims=("channel",),
        )
        target_scale_ = pm.Data(
            "target_scale",
            self.target_scale,
            dims=(),
        )

        # Store raw data as pm.Data
        channel_data = pm.Data(
            name="channel_data",
            value=channel_data_value,
            dims=("date", "channel"),
        )
        target_data = pm.Data(
            name="target_data",
            value=target_value,
            dims="date",
        )

        # Apply scaling within the model graph
        channel_data_scaled = channel_data / channel_scale_
        channel_data_scaled = pt.switch(
            pt.isnan(channel_data_scaled), 0.0, channel_data_scaled
        )
        channel_data_scaled.name = "channel_data_scaled"

        target_scaled = target_data / target_scale_
        target_scaled.name = "target_scaled"

        return channel_data_scaled, target_scaled, channel_scale_, target_scale_

    def _build_intercept(
        self, time_index: pt.TensorVariable | None = None
    ) -> pt.TensorVariable:
        """Build intercept variable (time-varying or static).

        Parameters
        ----------
        time_index : pt.TensorVariable | None
            Time index for time-varying intercept

        Returns
        -------
        pt.TensorVariable
            Intercept variable
        """
        if not self.time_varying_intercept:
            return self.model_config["intercept"].create_variable(name="intercept")

        intercept_baseline = self.model_config["intercept"].create_variable(
            "intercept_baseline"
        )
        intercept_latent_process = create_time_varying_gp_multiplier(
            name="intercept",
            dims="date",
            time_index=time_index,
            time_index_mid=self._time_index_mid,
            time_resolution=self._time_resolution,
            hsgp_kwargs=self.model_config["intercept_tvp_config"],
        )
        return pm.Deterministic(
            name="intercept",
            var=intercept_baseline * intercept_latent_process,
            dims="date",
        )

    def _build_channel_contribution(
        self,
        channel_data_scaled: pt.TensorVariable,
        time_index: pt.TensorVariable | None = None,
    ) -> pt.TensorVariable:
        """Build channel contribution variable (time-varying or static).

        Parameters
        ----------
        channel_data_scaled : pt.TensorVariable
            Scaled channel data
        time_index : pt.TensorVariable | None
            Time index for time-varying media

        Returns
        -------
        pt.TensorVariable
            Channel contribution variable
        """
        if not self.time_varying_media:
            return pm.Deterministic(
                name="channel_contribution",
                var=self.forward_pass(x=channel_data_scaled),
                dims=("date", "channel"),
            )

        baseline_channel_contribution = pm.Deterministic(
            name="baseline_channel_contribution",
            var=self.forward_pass(x=channel_data_scaled),
            dims=("date", "channel"),
        )
        media_latent_process = create_time_varying_gp_multiplier(
            name="media",
            dims="date",
            time_index=time_index,
            time_index_mid=self._time_index_mid,
            time_resolution=self._time_resolution,
            hsgp_kwargs=self.model_config["media_tvp_config"],
        )
        return pm.Deterministic(
            name="channel_contribution",
            var=baseline_channel_contribution * media_latent_process[:, None],
            dims=("date", "channel"),
        )

    def _build_control_contribution(self) -> pt.TensorVariable | None:
        """Build control contribution variable.

        Returns
        -------
        pt.TensorVariable | None
            Control contribution variable or None if no controls
        """
        if self.control_columns is None or len(self.control_columns) == 0:
            return None

        X_data = self.preprocessed_data["X"]
        if not isinstance(X_data, pd.DataFrame):
            raise TypeError("X data must be a DataFrame for control contribution")

        if not all(column in X_data.columns for column in self.control_columns):
            missing_cols = set(self.control_columns) - set(X_data.columns)
            raise ValueError(f"Control columns {missing_cols} not found in X data")

        if self.model_config["gamma_control"].dims != ("control",):
            self.model_config["gamma_control"].dims = "control"

        gamma_control = self.model_config["gamma_control"].create_variable(
            name="gamma_control"
        )
        control_data_ = pm.Data(
            name="control_data",
            value=X_data[self.control_columns],
            dims=("date", "control"),
        )
        return pm.Deterministic(
            name="control_contribution",
            var=control_data_ * gamma_control,
            dims=("date", "control"),
        )

    def _build_yearly_seasonality_contribution(self) -> pt.TensorVariable | None:
        """Build yearly seasonality contribution variable.

        Returns
        -------
        pt.TensorVariable | None
            Yearly seasonality contribution or None if not enabled
        """
        if self.yearly_seasonality is None:
            return None

        X_data = self.preprocessed_data["X"]
        if not isinstance(X_data, pd.DataFrame):
            raise TypeError("X data must be a DataFrame for yearly seasonality")

        dayofyear_value = X_data[self.date_column].dt.dayofyear.to_numpy()
        dayofyear = pm.Data(name="dayofyear", value=dayofyear_value, dims="date")

        def create_deterministic(x: pt.TensorVariable) -> None:
            pm.Deterministic(
                "fourier_contribution",
                x,
                dims=("date", *self.yearly_fourier.prior.dims),
            )

        return pm.Deterministic(
            name="yearly_seasonality_contribution",
            var=self.yearly_fourier.apply(
                dayofyear, result_callback=create_deterministic
            ),
            dims="date",
        )

    def _add_original_scale_deterministics(
        self,
        channel_contribution: pt.TensorVariable,
        target_scale: pt.TensorVariable,
        control_contribution: pt.TensorVariable | None,
        yearly_seasonality_contribution: pt.TensorVariable | None,
        mu: pt.TensorVariable,
    ) -> None:
        """Add deterministic variables in original scale.

        Parameters
        ----------
        channel_contribution : pt.TensorVariable
            Channel contribution in scaled space
        target_scale : pt.TensorVariable
            Target scaling factor
        control_contribution : pt.TensorVariable | None
            Control contribution in scaled space
        yearly_seasonality_contribution : pt.TensorVariable | None
            Yearly seasonality contribution in scaled space
        mu : pt.TensorVariable
            Model prediction in scaled space
        """
        pm.Deterministic(
            name="channel_contribution_original_scale",
            var=channel_contribution * target_scale,
            dims=("date", "channel"),
        )
        pm.Deterministic(
            name="total_contribution_original_scale",
            var=channel_contribution.sum(axis=-1) * target_scale,
            dims="date",
        )

        if control_contribution is not None:
            pm.Deterministic(
                name="control_contribution_original_scale",
                var=control_contribution * target_scale,
                dims=("date", "control"),
            )

        if yearly_seasonality_contribution is not None:
            pm.Deterministic(
                name="yearly_seasonality_contribution_original_scale",
                var=yearly_seasonality_contribution * target_scale,
                dims="date",
            )

        pm.Deterministic(
            name="y_original_scale",
            var=(mu * target_scale),
            dims="date",
        )

    def build_model(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        **kwargs,
    ) -> None:
        """Build a probabilistic model using PyMC for marketing mix modeling.

        The model incorporates channels, control variables, and Fourier components, applying
        adstock and saturation transformations to the channel data. The final model is
        constructed with multiple factors contributing to the response variable.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for the model, which should include columns for channels,
            control variables (if applicable), and Fourier components (if applicable).

        y : Union[pd.Series, np.ndarray]
            The target/response variable for the modeling.

        **kwargs : dict
            Additional keyword arguments that might be required by underlying methods or utilities.

        Attributes Set
        ---------------
        model : pm.Model
            The PyMC model object containing all the defined stochastic and deterministic variables.

        Examples
        --------
        Initialize model with custom configuration

        .. code-block:: python

            from pymc_marketing.mmm import (
                GeometricAdstock,
                LogisticSaturation
                MMM,
            )
            from pymc_extras.prior import Prior

            custom_config = {
                "intercept": Prior("Normal", mu=0, sigma=2),
                "saturation_beta": Prior("Gamma", mu=1, sigma=3),
                "saturation_lambda": Prior("Beta", alpha=3, beta=1),
                "adstock_alpha": Prior("Beta", alpha=1, beta=3),
                "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
                "gamma_control": Prior("Normal", mu=0, sigma=2, dims="control"),
                "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
            }

            model = MMM(
                date_column="date_week",
                channel_columns=["x1", "x2"],
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                control_columns=[
                    "event_1",
                    "event_2",
                    "t",
                ],
                yearly_seasonality=2,
                model_config=custom_config,
            )

        """
        self._generate_and_preprocess_model_data(X, y)
        self._compute_scales()

        with pm.Model(coords=self.model_coords) as self.model:
            # Prepare data and scaling
            X_data = self.preprocessed_data["X"]
            if not isinstance(X_data, pd.DataFrame):
                raise TypeError("X data must be a DataFrame")

            y_data = self.preprocessed_data["y"]
            if not isinstance(y_data, pd.Series | np.ndarray):
                raise TypeError("y data must be a Series or ndarray")

            channel_data_scaled, target_scaled, _, target_scale_ = (
                self._create_scaled_data_variables(X_data[self.channel_columns], y_data)
            )

            # Create time index if needed
            time_index = None
            if self.time_varying_intercept or self.time_varying_media:
                time_index = pm.Data("time_index", self._time_index, dims="date")

            # Build model components
            intercept = self._build_intercept(time_index)
            channel_contribution = self._build_channel_contribution(
                channel_data_scaled, time_index
            )

            # Total contribution deterministic for optimization
            pm.Deterministic(
                name="total_contribution",
                var=channel_contribution.sum(axis=(-2, -1)),
                dims=(),
            )

            # Build mu starting with intercept and channels
            mu_var = intercept + channel_contribution.sum(axis=-1)

            # Add control contribution if present
            control_contribution = self._build_control_contribution()
            if control_contribution is not None:
                mu_var += control_contribution.sum(axis=-1)

            # Add yearly seasonality if present
            yearly_seasonality_contribution = (
                self._build_yearly_seasonality_contribution()
            )
            if yearly_seasonality_contribution is not None:
                mu_var += yearly_seasonality_contribution

            # Create mu deterministic
            mu = pm.Deterministic(name="mu", var=mu_var, dims="date")

            # Create likelihood
            self.model_config["likelihood"].dims = "date"
            self.model_config["likelihood"].create_likelihood_variable(
                name=self.output_var,
                mu=mu,
                observed=target_scaled,
            )

            # Add original scale deterministics
            self._add_original_scale_deterministics(
                channel_contribution,
                target_scale_,
                control_contribution,
                yearly_seasonality_contribution,
                mu,
            )

    @property
    def default_model_config(self) -> dict:
        """Define the default model configuration."""
        base_config = {
            "intercept": Prior("Normal", mu=0, sigma=2),
            "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
            "gamma_control": Prior("Normal", mu=0, sigma=2, dims="control"),
            "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
        }

        if self.time_varying_intercept:
            base_config["intercept_tvp_config"] = HSGPKwargs(
                m=200,
                L=None,
                eta_lam=1,
                ls_mu=5,
                ls_sigma=10,
                cov_func=None,
            )
        if self.time_varying_media:
            base_config["media_tvp_config"] = HSGPKwargs(
                m=200,
                L=None,
                eta_lam=1,
                ls_mu=5,
                ls_sigma=10,
                cov_func=None,
            )

        return {
            **base_config,
            **self.adstock.model_config,
            **self.saturation.model_config,
        }

    def channel_contribution_forward_pass(
        self,
        channel_data: npt.NDArray,
        disable_logger_stdout: bool | None = False,
    ) -> npt.NDArray:
        """Evaluate the channel contribution for a given channel data and a fitted model, ie. the forward pass.

        Parameters
        ----------
        channel_data : array-like
            Input channel data. Result of all the preprocessing steps.
        disable_logger_stdout : bool, optional
            If True, suppress logger output to stdout

        Returns
        -------
        array-like
            Transformed channel data.

        """
        if disable_logger_stdout:
            logger = logging.getLogger("pymc.sampling.forward")
            logger.propagate = False

        coords = {
            **self.model_coords,
        }
        with pm.Model(coords=coords):
            # Create channel_data as a pm.Data tensor to ensure proper shape handling
            channel_data_tensor = pm.Data(
                name="channel_data",
                value=channel_data,
                dims=("date", "channel"),
            )

            # Apply the same scaling as in the original model - scale by channel_scale
            channel_scale_ = pm.Data(
                "channel_scale",
                self.channel_scale,
                dims=("channel",),
            )

            # Apply scaling within the model graph (same as build_model)
            channel_data_scaled = channel_data_tensor / channel_scale_
            channel_data_scaled = pt.switch(
                pt.isnan(channel_data_scaled), 0.0, channel_data_scaled
            )
            channel_data_scaled.name = "channel_data_scaled"

            pm.Deterministic(
                "channel_contribution",
                self.forward_pass(x=channel_data_scaled),
                dims=("date", "channel"),
            )

            idata = pm.sample_posterior_predictive(
                self.fit_result,
                var_names=["channel_contribution"],
                progressbar=False,
            )

        channel_contribution = idata.posterior_predictive.channel_contribution
        if self.time_varying_media:
            # This is coupled with the name of the
            # latent process Deterministic
            name = "media_temporal_latent_multiplier"
            mutliplier = self.fit_result[name]
            channel_contribution = channel_contribution * mutliplier

        return channel_contribution.to_numpy()

    @property
    def _serializable_model_config(self) -> dict[str, Any]:
        def ndarray_to_list(d: dict) -> dict:
            new_d = d.copy()  # Copy the dictionary to avoid mutating the original one
            for key, value in new_d.items():
                if isinstance(value, np.ndarray):
                    new_d[key] = value.tolist()
                elif isinstance(value, dict):
                    new_d[key] = ndarray_to_list(value)
            return new_d

        serializable_config = self.model_config.copy()
        return ndarray_to_list(serializable_config)

    @classmethod
    def attrs_to_init_kwargs(cls, attrs) -> dict[str, Any]:
        """Convert attributes to initialization kwargs.

        Returns
        -------
        dict[str, Any]
            The initialization kwargs.

        """
        # Batch load JSON attributes for efficiency
        model_config = json.loads(attrs["model_config"])
        adstock_dict = json.loads(attrs["adstock"])
        saturation_dict = json.loads(attrs["saturation"])
        scaling_dict = json.loads(attrs.get("scaling", "null"))

        return {
            "model_config": cls._model_config_formatting(model_config),
            "date_column": json.loads(attrs["date_column"]),
            "control_columns": json.loads(attrs["control_columns"]),
            "channel_columns": json.loads(attrs["channel_columns"]),
            "adstock": adstock_from_dict(adstock_dict),
            "saturation": saturation_from_dict(saturation_dict),
            "adstock_first": json.loads(attrs.get("adstock_first", "true")),
            "yearly_seasonality": json.loads(attrs["yearly_seasonality"]),
            "time_varying_intercept": json.loads(
                attrs.get("time_varying_intercept", "false")
            ),
            "time_varying_media": json.loads(attrs.get("time_varying_media", "false")),
            "validate_data": json.loads(attrs["validate_data"]),
            "sampler_config": json.loads(attrs["sampler_config"]),
            "dag": json.loads(attrs.get("dag", "null")),
            "treatment_nodes": json.loads(attrs.get("treatment_nodes", "null")),
            "outcome_node": json.loads(attrs.get("outcome_node", "null")),
            "scaling": cls._deserialize_scaling(scaling_dict),
        }

    def _has_new_scaling(self) -> bool:
        """Check if model uses new scaling approach."""
        return hasattr(self, "channel_scale") and hasattr(self, "target_scale")

    def _prepare_channel_data(self, X: pd.DataFrame) -> dict[str, Any]:
        """Prepare channel data for model update.

        Parameters
        ----------
        X : pd.DataFrame
            Input data containing channel columns

        Returns
        -------
        dict
            Dictionary with channel data ready for pm.set_data
        """
        try:
            channel_data = X[self.channel_columns].to_numpy()
        except KeyError as e:
            raise RuntimeError("New data must contain channel_data!") from e

        if self._has_new_scaling():
            return {
                "channel_data": channel_data,
                "channel_scale": self.channel_scale,
            }

        # Backward compatibility: use transformer if available
        if hasattr(self, "channel_transformer"):
            channel_data = self.channel_transformer.transform(channel_data)
        return {"channel_data": channel_data}

    def _prepare_control_data(self, X: pd.DataFrame) -> dict[str, Any]:
        """Prepare control data for model update.

        Parameters
        ----------
        X : pd.DataFrame
            Input data containing control columns

        Returns
        -------
        dict
            Dictionary with control data ready for pm.set_data
        """
        if self.control_columns is None:
            return {}

        control_data = X[self.control_columns].to_numpy()

        # Apply transformation if available
        if hasattr(self, "control_transformer"):
            control_data = self.control_transformer.transform(control_data)

        return {"control_data": control_data}

    def _prepare_target_data(
        self, y: np.ndarray | pd.Series | None, n_rows: int
    ) -> dict[str, Any]:
        """Prepare target data for model update.

        Parameters
        ----------
        y : np.ndarray | pd.Series | None
            Target data
        n_rows : int
            Number of rows (for creating zeros if y is None)

        Returns
        -------
        dict
            Dictionary with target data ready for pm.set_data
        """
        if y is None:
            # When y is None, create zeros array matching the type of preprocessed y
            y_preprocessed = self.preprocessed_data["y"]
            if isinstance(y_preprocessed, pd.Series | np.ndarray):
                y_data = np.zeros(n_rows, dtype=np.asarray(y_preprocessed).dtype)
            else:
                # Default to float64 if type is unknown
                y_data = np.zeros(n_rows, dtype="float64")
        elif isinstance(y, pd.Series):
            y_data = y.to_numpy()
        elif isinstance(y, np.ndarray):
            y_data = y
        else:
            raise TypeError("y must be either a pandas Series or a numpy array")

        if self._has_new_scaling():
            return {
                "target_data": y_data,
                "target_scale": self.target_scale,
            }

        # Backward compatibility
        return {"target": y_data}

    def _data_setter(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> None:
        """Set new data in the model.

        This function accepts data in various formats and sets them into the
        model using the PyMC's `set_data` method. The data corresponds to the
        channel data and the target.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Data for the channel. It can be a numpy array or pandas DataFrame.
            If it's a DataFrame, the columns corresponding to self.channel_columns
            are used. If it's an ndarray, it's used directly.
        y : Union[np.ndarray, pd.Series], optional
            Target data. It can be a numpy array or a pandas Series.
            If it's a Series, its values are used. If it's an ndarray, it's used
            directly. The default is None.

        Raises
        ------
        RuntimeError
            If the data for the channel is not provided in `X`.
        TypeError
            If `X` is not a pandas DataFrame or a numpy array, or
            if `y` is not a pandas Series or a numpy array and is not None.

        Returns
        -------
        None

        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame in order to access the columns"
            )

        coords = {"date": X[self.date_column].to_numpy()}

        # Prepare all data components
        data = {}
        data.update(self._prepare_channel_data(X))
        data.update(self._prepare_control_data(X))
        data.update(self._prepare_target_data(y, X.shape[0]))

        # Add additional data if needed
        if self.yearly_seasonality is not None:
            data["dayofyear"] = X[self.date_column].dt.dayofyear.to_numpy()

        if self.time_varying_intercept or self.time_varying_media:
            data["time_index"] = infer_time_index(
                X[self.date_column], self.X[self.date_column], self._time_resolution
            )

        with self.model:
            pm.set_data(data, coords=coords)

    @classmethod
    def _deserialize_scaling(cls, scaling_dict: dict | None) -> Scaling | None:
        """Deserialize scaling configuration from JSON.

        Parameters
        ----------
        scaling_dict : dict | None
            The serialized scaling configuration.

        Returns
        -------
        Scaling | None
            The deserialized Scaling object or None.
        """
        if scaling_dict is None:
            return None

        return Scaling(
            target=VariableScaling(
                method=scaling_dict["target"]["method"],
                dims=tuple(scaling_dict["target"]["dims"]),
            ),
            channel=VariableScaling(
                method=scaling_dict["channel"]["method"],
                dims=tuple(scaling_dict["channel"]["dims"]),
            ),
        )

    @classmethod
    def _model_config_formatting(cls, model_config: dict) -> dict:
        """Format the model configuration.

        Because of json serialization, model_config values that were originally tuples
        or numpy are being encoded as lists. This function converts them back to tuples
        and numpy arrays to ensure correct id encoding.

        Parameters
        ----------
        model_config : dict
            The model configuration to format.

        Returns
        -------
        dict
            The formatted model configuration.

        """

        def format_nested_dict(d: dict) -> dict:
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = format_nested_dict(value)
                elif isinstance(value, list):
                    # Check if the key is "dims" to convert it to tuple
                    if key == "dims":
                        d[key] = tuple(value)
                    # Convert all other lists to numpy arrays
                    else:
                        d[key] = np.array(value)
            return d

        return format_nested_dict(model_config.copy())


class MMM(
    MaxAbsScaleTarget,
    MaxAbsScaleChannels,
    ValidateControlColumns,
    BaseMMM,
):
    r"""Media Mix Model class, Delayed Adstock and logistic saturation as default initialization (see [1]_).

    Given a time series target variable :math:`y_{t}` (e.g. sales on conversions), media variables
    :math:`x_{m, t}` (e.g. impressions, clicks or costs) and a set of control covariates :math:`z_{c, t}` (e.g. holidays, special events)
    we consider a Bayesian linear model of the form:

    .. math::
        y_{t} = \alpha + \sum_{m=1}^{M}\beta_{m}f(x_{m, t}) +  \sum_{c=1}^{C}\gamma_{c}z_{c, t} + \varepsilon_{t},

    where :math:`\alpha` is the intercept, :math:`f` is a media transformation function and :math:`\varepsilon_{t}` is the error therm
    which we assume is normally distributed. The function :math:`f` encodes the contribution of media on the target variable.
    Typically we consider two types of transformation: adstock (carry-over) and saturation effects.

    Notes
    -----
    Here are some important notes about the model:

    1. Before fitting the model, we scale the target variable and the media channels using the maximum absolute value of each variable.
    This enable us to have a more stable model and better convergence. If control variables are present, we do not scale them!
    If needed please do it before passing the data to the model.

    2. We allow to add yearly seasonality controls as Fourier modes.
    You can use the `yearly_seasonality` parameter to specify the number of Fourier modes to include.

    3. This class also allow us to calibrate the model using:

        * Custom priors for the parameters via the `model_config` parameter. You can also set the likelihood distribution.

        * Adding lift tests to the likelihood function via the :meth:`add_lift_test_measurements <pymc_marketing.mmm.mmm.MMM.add_lift_test_measurements>` method.

    For details on a vanilla implementation in PyMC, see [2]_.

    Examples
    --------
    Here is an example of how to instantiate the model with the default configuration:

    .. code-block:: python

        import numpy as np
        import pandas as pd

        from pymc_marketing.mmm import (
            GeometricAdstock,
            LogisticSaturation
            MMM,
        )
        from pymc_marketing.paths import data_dir

        file_path = data_dir / "mmm_example.csv"
        data = pd.read_csv(file_path, parse_dates=["date_week"])

        mmm = MMM(
            date_column="date_week",
            channel_columns=["x1", "x2"],
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            control_columns=[
                "event_1",
                "event_2",
                "t",
            ],
            yearly_seasonality=2,
        )

    Now we can fit the model with the data:

    .. code-block:: python

        # Set features and target
        X = data.drop("y", axis=1)
        y = data["y"]

        # Fit the model
        idata = mmm.fit(X, y)

    We can also define custom priors for the model:

    .. code-block:: python

        import numpy as np

        from pymc_marketing.mmm import (
            GeometricAdstock,
            LogisticSaturation
            MMM,
        )
        from pymc_extras.prior import Prior

        my_model_config = {
            "saturation_beta": Prior("LogNormal", mu=np.array([2, 1]), sigma=1),
            "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
        }

        mmm = MMM(
            date_column="date_week",
            channel_columns=["x1", "x2"],
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            control_columns=[
                "event_1",
                "event_2",
                "t",
            ],
            yearly_seasonality=2,
            model_config=my_model_config,
        )

    As you can see, we can configure all prior and likelihood distributions via the `model_config`.

    The `fit` method accepts keyword arguments that are passed to the PyMC sampling method.
    For example, to change the number of samples and chains, and using a JAX implementation of NUTS we can do:

    .. code-block:: python

        sampler_kwargs = {
            "draws": 2_000,
            "target_accept": 0.9,
            "chains": 5,
            "random_seed": 42,
        }

        idata = mmm.fit(X, y, nuts_sampler="numpyro", **sampler_kwargs)

    References
    ----------
    .. [1] Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017).
    .. [2] Orduz, J. `"Media Effect Estimation with PyMC: Adstock, Saturation & Diminishing Returns" <https://juanitorduz.github.io/pymc_mmm/>`_.

    """  # noqa: E501

    _model_type: str = "MMM"
    version: str = "0.0.3"

    def channel_contribution_forward_pass(
        self,
        channel_data: npt.NDArray,
        disable_logger_stdout: bool | None = False,
    ) -> npt.NDArray:
        """Evaluate the channel contribution for a given channel data and a fitted model, ie. the forward pass.

        We return the contribution in the original scale of the target variable.

        Parameters
        ----------
        channel_data : array-like
            Input channel data. Result of all the preprocessing steps.
        disable_logger_stdout : bool, optional
            If True, suppress logger output to stdout

        Returns
        -------
        array-like
            Transformed channel data.

        """
        channel_contribution_forward_pass = super().channel_contribution_forward_pass(
            channel_data=channel_data, disable_logger_stdout=disable_logger_stdout
        )

        # Convert from scaled space to original scale
        if hasattr(self, "target_scale"):
            # Use the computed scale factor from our new approach
            return channel_contribution_forward_pass * self.target_scale
        else:
            # Fallback to transformer for backward compatibility
            target_transformed_vectorized = np.vectorize(
                self.target_transformer.inverse_transform,
                excluded=[1, 2],
                signature="(m, n) -> (m, n)",
            )
            return target_transformed_vectorized(channel_contribution_forward_pass)

    def get_channel_contribution_forward_pass_grid(
        self, start: float, stop: float, num: int
    ) -> DataArray:
        """Generate a grid of scaled channel contributions for a given grid of shared values.

        Parameters
        ----------
        start : float
            Start of the grid. It must be equal or greater than 0.
        stop : float
            End of the grid. It must be greater than start.
        num : int
            Number of points in the grid.

        Returns
        -------
        DataArray
            Grid of channel contributions.

        """
        if start < 0:
            raise ValueError("start must be greater than or equal to 0.")

        share_grid = np.linspace(start=start, stop=stop, num=num)

        # Extract and validate X_data once outside the loop
        X_data = self.preprocessed_data["X"]
        if not isinstance(X_data, pd.DataFrame):
            raise TypeError("X data must be a DataFrame")

        base_channel_data = X_data[self.channel_columns].to_numpy()

        # Preallocate list for better performance
        channel_contribution = []
        for delta in share_grid:
            # Vectorized scaling - much faster than creating new arrays
            channel_data = delta * base_channel_data
            channel_contribution_forward_pass = self.channel_contribution_forward_pass(
                channel_data=channel_data,
                disable_logger_stdout=True,
            )
            channel_contribution.append(channel_contribution_forward_pass)

        return DataArray(
            data=np.array(channel_contribution),
            dims=("delta", "chain", "draw", "date", "channel"),
            coords={
                "delta": share_grid,
                "date": self.X[self.date_column],
                "channel": self.channel_columns,
            },
        )

    def plot_channel_parameter(self, param_name: str, **plt_kwargs: Any) -> plt.Figure:
        """Plot the posterior distribution of a specific parameter for each channel.

        Parameters
        ----------
        param_name : str
            The name of the parameter to plot.
        **plt_kwargs : Any
            Additional keyword arguments to pass to the `plt.subplots` function.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the plot.

        Raises
        ------
        ValueError
            If the specified parameter name is invalid or not found in the model
            saturation or adstock function.

        """
        saturation: SaturationTransformation = self.saturation
        adstock: AdstockTransformation = self.adstock

        # Use list extension instead of concatenation for better performance
        parameters_to_check = [
            *saturation.variable_mapping.values(),
            *adstock.variable_mapping.values(),
        ]
        if param_name not in parameters_to_check:
            raise ValueError(
                f"Invalid parameter name: {param_name}. Choose from {parameters_to_check}"
            )

        param_samples_df = pd.DataFrame(
            data=az.extract(data=self.fit_result, var_names=[param_name]).T,
            columns=self.channel_columns,
        )

        fig, ax = plt.subplots(**plt_kwargs)
        sns.violinplot(data=param_samples_df, orient="h", ax=ax)
        ax.set(
            title=f"Posterior Distribution: {param_name} Parameter",
            xlabel=param_name,
            ylabel="channel",
        )
        return fig

    def get_ts_contribution_posterior(
        self, var_contribution: str, original_scale: bool = False
    ) -> DataArray:
        """Get the posterior distribution of the time series contributions of a given variable.

        Parameters
        ----------
        var_contribution : str
            The variable for which to get the contributions. It must be a valid variable
            in the `fit_result` attribute.
        original_scale : bool, optional
            Whether to plot in the original scale.

        Returns
        -------
        DataArray
            The posterior distribution of the time series contributions.

        """
        if original_scale:
            # Try to use original scale variables if available
            original_scale_var = f"{var_contribution}_original_scale"
            if hasattr(self, "fit_result") and original_scale_var in self.fit_result:
                contributions = self._format_model_contributions(
                    var_contribution=original_scale_var
                )
                # Rename to expected variable name for consistency with plotting code
                return contributions.rename(var_contribution)
            else:
                # Fallback to transformation for backward compatibility
                contributions = self._format_model_contributions(
                    var_contribution=var_contribution
                )
                return apply_sklearn_transformer_across_dim(
                    data=contributions,
                    func=self.get_target_transformer().inverse_transform,
                    dim_name="date",
                )
        else:
            # Use scaled contributions directly
            return self._format_model_contributions(var_contribution=var_contribution)

    def _get_intercept_for_plot(
        self, original_scale: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get intercept mean and HDI for plotting.

        Parameters
        ----------
        original_scale : bool
            Whether to return in original scale

        Returns
        -------
        tuple
            (intercept_mean, intercept_hdi)
        """
        intercept = az.extract(self.fit_result, var_names=["intercept"], combined=False)

        if not original_scale:
            intercept = apply_sklearn_transformer_across_dim(
                data=intercept,
                func=self.get_target_transformer().transform,
                dim_name="chain",
            )

        intercept_mean = intercept.mean(["chain", "draw"]).data
        hdi_result = az.hdi(intercept).intercept.data

        if intercept.ndim == 2:
            # Stationary intercept - use broadcasting instead of repeat for efficiency
            n_dates = self.X[self.date_column].shape[0]
            intercept_hdi = np.broadcast_to(
                hdi_result[None, :], (n_dates, hdi_result.shape[0])
            )
        else:
            # Time-varying intercept
            intercept_hdi = hdi_result

        return intercept_mean, intercept_hdi

    def _get_target_for_plot(self, original_scale: bool) -> np.ndarray:
        """Get target variable for plotting.

        Parameters
        ----------
        original_scale : bool
            Whether to return in original scale

        Returns
        -------
        np.ndarray
            Target variable for plotting
        """
        y_array = np.asarray(self.preprocessed_data["y"])

        if original_scale:
            return y_array

        return self.get_target_transformer().transform(y_array.reshape(-1, 1))

    def plot_components_contributions(
        self, original_scale: bool = False, **plt_kwargs: Any
    ) -> plt.Figure:
        """Plot the target variable and the posterior predictive model components.

        We can plot the target variable and the posterior predictive model components in
        the scaled space or in the original space.

        Parameters
        ----------
        original_scale : bool, optional
            Whether to plot in the original scale.

        **plt_kwargs
            Additional keyword arguments to pass to `plt.subplots`.

        Returns
        -------
        plt.Figure

        """
        # Collect contributions and their HDIs
        channel_contribution = self.get_ts_contribution_posterior(
            var_contribution="channel_contribution", original_scale=original_scale
        )

        means = [channel_contribution.mean(["chain", "draw"])]
        contribution_vars = [
            az.hdi(channel_contribution, hdi_prob=DEFAULT_HDI_PROB).channel_contribution
        ]

        # Add optional contributions (controls and seasonality)
        component_mapping = [
            ("control_columns", "control_contribution"),
            ("yearly_seasonality", "fourier_contribution"),
        ]

        for attr_name, var_name in component_mapping:
            if getattr(self, attr_name, None):
                contributions = self.get_ts_contribution_posterior(
                    var_contribution=var_name, original_scale=original_scale
                )
                means.append(contributions.mean(["chain", "draw"]))
                contribution_vars.append(
                    az.hdi(contributions, hdi_prob=DEFAULT_HDI_PROB)[var_name]
                )

        # Create plot
        fig, ax = plt.subplots(**plt_kwargs)

        if self.X is None:
            return fig

        dates = self.X[self.date_column]
        contribution_names = [
            "channel_contribution",
            "control_contribution",
            "fourier_contribution",
        ]

        # Plot contributions
        for i, (mean, hdi, var_name) in enumerate(
            zip(means, contribution_vars, contribution_names, strict=False)
        ):
            ax.fill_between(
                x=dates,
                y1=hdi.isel(hdi=0),
                y2=hdi.isel(hdi=1),
                color=f"C{i}",
                alpha=0.25,
                label=f"$94\\%$ HDI ({var_name})",
            )
            ax.plot(dates, np.asarray(mean), color=f"C{i}")

        # Plot intercept
        intercept_mean, intercept_hdi = self._get_intercept_for_plot(original_scale)
        color_idx = len(means)

        # Use scalar intercept if possible, otherwise array
        if np.ndim(intercept_mean) == 0:
            # Scalar intercept - matplotlib handles broadcasting automatically
            ax.axhline(y=intercept_mean, color=f"C{color_idx}")
        else:
            # Time-varying intercept
            ax.plot(dates, intercept_mean, color=f"C{color_idx}")

        ax.fill_between(
            x=dates,
            y1=intercept_hdi[:, 0],
            y2=intercept_hdi[:, 1],
            color=f"C{color_idx}",
            alpha=0.25,
            label="$94\\%$ HDI (intercept)",
        )

        # Plot target
        y_to_plot = self._get_target_for_plot(original_scale)
        ylabel = self.output_var if original_scale else f"{self.output_var} scaled"

        ax.plot(dates, y_to_plot, label=ylabel, color="black")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
        ax.set(
            title="Posterior Predictive Model Components",
            xlabel="date",
            ylabel=ylabel,
        )

        return fig

    def plot_channel_contribution_grid(
        self,
        start: float,
        stop: float,
        num: int,
        absolute_xrange: bool = False,
        **plt_kwargs: Any,
    ) -> plt.Figure:
        """Plot a grid of scaled channel contributions for a given grid of share values.

        Parameters
        ----------
        start : float
            Start of the grid. It must be equal or greater than 0.
        stop : float
            End of the grid. It must be greater than start.
        num : int
            Number of points in the grid.
        absolute_xrange : bool, optional
            If True, the x-axis is in absolute values (input units), otherwise it is in
            relative percentage values, by default False.
        **plt_kwargs
            Keyword arguments to pass to `plt.subplots()`

        Returns
        -------
        plt.Figure
            Plot of grid of channel contributions.

        """
        share_grid = np.linspace(start=start, stop=stop, num=num)
        contributions = self.get_channel_contribution_forward_pass_grid(
            start=start, stop=stop, num=num
        )

        fig, ax = plt.subplots(**plt_kwargs)

        for i, channel in enumerate(self.channel_columns):
            channel_contribution_total = contributions.sel(channel=channel).sum(
                dim="date"
            )

            hdi_contribution = az.hdi(ary=channel_contribution_total).x

            total_channel_input = self.X[channel].sum()
            x_range = (
                total_channel_input * share_grid if absolute_xrange else share_grid
            )

            ax.fill_between(
                x=x_range,
                y1=hdi_contribution[:, 0],
                y2=hdi_contribution[:, 1],
                color=f"C{i}",
                label=f"{channel} $94\\%$ HDI contribution",
                alpha=0.4,
            )

            sns.lineplot(
                x=x_range,
                y=channel_contribution_total.mean(dim=("chain", "draw")),
                color=f"C{i}",
                marker="o",
                label=f"{channel} contribution mean",
                ax=ax,
            )
            if absolute_xrange:
                ax.axvline(
                    x=total_channel_input,
                    color=f"C{i}",
                    linestyle="--",
                    label=f"{channel} current total input",
                )

        if not absolute_xrange:
            ax.axvline(x=1, color="black", linestyle="--", label=r"$\delta = 1$")

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        x_label = "input" if absolute_xrange else r"$\delta$"
        ax.set(
            title="Channel contribution as a function of cost share",
            xlabel=x_label,
            ylabel="contribution",
        )
        return fig

    def new_spend_contributions(
        self,
        spend: np.ndarray | None = None,
        one_time: bool = True,
        spend_leading_up: np.ndarray | None = None,
        prior: bool = False,
        original_scale: bool = True,
        **sample_posterior_predictive_kwargs,
    ) -> DataArray:
        """Return the upcoming contributions for a given spend.

        The spend can be one time or constant over the period. The spend leading up to the
        period can also be specified in order account for the lagged effect of the spend.

        Parameters
        ----------
        spend : np.ndarray, optional
            Array of spend for each channel. If None, the average spend for each channel is used, by default None.
        one_time : bool, optional
            Whether the spends for each channel are only at the start of the period.
            If True, all spends after the initial spend are zero.
            If False, all spends after the initial spend are the same as the initial spend.
            By default True.
        spend_leading_up : np.array, optional
            Array of spend for each channel leading up to the spend, by default None or 0 for each channel.
            Use this parameter to account for the lagged effect of the spend.
        prior : bool, optional
            Whether to use the prior or posterior, by default False (posterior)
        **sample_posterior_predictive_kwargs
            Additional keyword arguments passed to pm.sample_posterior_predictive

        Returns
        -------
        DataArray
            Upcoming contributions for each channel

        Examples
        --------
        Channel contributions from 1 unit on each channel only once.

        .. code-block:: python

            n_channels = len(model.channel_columns)
            spend = np.ones(n_channels)
            new_spend_contributions = model.new_spend_contributions(spend=spend)

        Channel contributions from continuously spending 1 unit on each channel.

        .. code-block:: python

            n_channels = len(model.channel_columns)
            spend = np.ones(n_channels)
            new_spend_contributions = model.new_spend_contributions(
                spend=spend, one_time=False
            )

        Channel contributions from 1 unit on each channel only once but with 1 unit leading up to the spend.

        .. code-block:: python

            n_channels = len(model.channel_columns)
            spend = np.ones(n_channels)
            spend_leading_up = np.ones(n_channels)
            new_spend_contributions = model.new_spend_contributions(
                spend=spend, spend_leading_up=spend_leading_up
            )

        """
        if spend is None:
            spend = self.X.loc[:, self.channel_columns].mean().to_numpy()  # type: ignore

        n_channels = len(self.channel_columns)
        if len(spend) != n_channels:
            raise ValueError("spend must be the same length as the number of channels")

        new_data = create_new_spend_data(
            spend=spend,
            adstock_max_lag=self.adstock.l_max,
            one_time=one_time,
            spend_leading_up=spend_leading_up,
        )

        new_data = (
            self.channel_transformer.transform(new_data) if not prior else new_data
        )

        idata: Dataset = self.fit_result if not prior else self.prior

        coords = {
            "time_since_spend": np.arange(-self.adstock.l_max, self.adstock.l_max + 1),
            "channel": self.channel_columns,
        }
        with pm.Model(coords=coords):
            pm.Deterministic(
                "channel_contribution",
                self.forward_pass(x=new_data),
                dims=("time_since_spend", "channel"),
            )

            samples = pm.sample_posterior_predictive(
                idata,
                var_names=["channel_contribution"],
                **sample_posterior_predictive_kwargs,
            )

        channel_contribution = samples.posterior_predictive["channel_contribution"]

        if not original_scale:
            channel_contribution = apply_sklearn_transformer_across_dim(
                data=channel_contribution,
                func=self.get_target_transformer().transform,
                dim_name="time_since_spend",
            )

        return channel_contribution

    def plot_new_spend_contributions(
        self,
        spend_amount: float,
        one_time: bool = True,
        lower: float = 0.025,
        upper: float = 0.975,
        ylabel: str = "Sales",
        idx: slice | None = None,
        channels: list[str] | None = None,
        prior: bool = False,
        original_scale: bool = True,
        ax: plt.Axes | None = None,
        **sample_posterior_predictive_kwargs,
    ) -> plt.Axes:
        """Plot the upcoming sales for a given spend amount.

        Calls the new_spend_contributions method and plots the results. For more
        control over the plot, use new_spend_contributions directly.

        Parameters
        ----------
        spend_amount : float
            The amount of spend for each channel
        one_time : bool, optional
            Whether the spend are one time (at start of period) or constant (over period), by default True (one time)
        lower : float, optional
            The lower quantile for the confidence interval, by default 0.025
        upper : float, optional
            The upper quantile for the confidence interval, by default 0.975
        ylabel : str, optional
            The label for the y-axis, by default "Sales"
        idx : slice, optional
            The index slice of days to plot, by default None or only the positive days.
            More specifically, slice(0, None, None)
        channels : List[str], optional
            The channels to plot, by default None or all channels
        prior : bool, optional
            Whether to use the prior or posterior, by default False (posterior)
        original_scale : bool, optional
            Whether to plot in the original scale of the target variable, by default True
        ax : plt.Axes, optional
            The axes to plot on, by default None or current axes
        **sample_posterior_predictive_kwargs
            Additional keyword arguments passed to pm.sample_posterior_predictive

        Returns
        -------
        plt.Axes
            The plot of upcoming sales for the given spend amount

        """
        for value in [lower, upper]:
            if value < 0 or value > 1:
                raise ValueError("lower and upper must be between 0 and 1")
        if lower > upper:
            raise ValueError("lower must be less than or equal to upper")

        ax = ax or plt.gca()
        total_channels = len(self.channel_columns)
        contributions = self.new_spend_contributions(
            np.ones(total_channels) * spend_amount,
            one_time=one_time,
            spend_leading_up=np.ones(total_channels) * spend_amount,
            prior=prior,
            original_scale=original_scale,
            **sample_posterior_predictive_kwargs,
        )

        contributions_groupby = contributions.to_series().groupby(
            level=["time_since_spend", "channel"]
        )

        idx = idx or pd.IndexSlice[0:]

        conf = (
            contributions_groupby.quantile([lower, upper])
            .unstack("channel")
            .unstack()
            .loc[idx]
        )

        channels = channels or self.channel_columns  # type: ignore
        for channel in channels:  # type: ignore
            ax.fill_between(
                conf.index,
                conf[channel][lower],
                conf[channel][upper],
                label=f"{channel} {100 * (upper - lower):.0f}% CI",
                alpha=0.5,
            )
        mean = contributions_groupby.mean().unstack("channel").loc[idx, channels]
        color = [f"C{i}" for i in range(len(channels))]  # type: ignore
        mean.add_suffix(" mean").plot(ax=ax, color=color, alpha=0.75)
        ax.legend().set_title("Channel")
        ax.set(
            xlabel="Time since spend",
            ylabel=ylabel,
            title=f"Upcoming sales for {spend_amount:.02f} spend",
        )
        return ax

    def _validate_data(self, X, y=None):
        return X

    @property
    def _channel_scales(self) -> np.ndarray:
        return self.channel_transformer["scaler"].scale_

    def _channel_map_scales(self) -> dict:
        return dict(zip(self.channel_columns, self._channel_scales, strict=True))  # type: ignore

    def format_recovered_transformation_parameters(
        self, quantile: float = 0.5
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Format the recovered transformation parameters for each channel.

        This function retrieves the quantile of the parameters for each channel and formats them into a dictionary
        containing the channel name, the saturation parameters, and the adstock parameters.

        Parameters
        ----------
        quantile : float, optional
            The quantile to retrieve from the posterior distribution of the parameters. Default is 0.5.

        Returns
        -------
        dict
            A dictionary containing the channel names as keys and the corresponding saturation and adstock parameters
            as values.

        Example
        -------
        .. code-block:: python

            self.format_recovered_transformation_parameters(quantile=0.5)
            # Output:
            {
                "x1": {
                    "saturation_params": {
                        "lam": 2.4761893929757077,
                        "beta": 0.360226791880304,
                    },
                    "adstock_params": {"alpha": 0.39910387900504796},
                },
                "x2": {
                    "saturation_params": {
                        "lam": 2.6485978655163436,
                        "beta": 0.2399381337197204,
                    },
                    "adstock_params": {"alpha": 0.18859423763437405},
                },
            }

        """
        # Retrieve channel names
        channels = self.fit_result.channel.values

        # Define the parameter groups for consolidation
        param_groups = {
            "saturation_params": self.saturation.model_config.keys(),
            "adstock_params": self.adstock.model_config.keys(),
        }

        # Pre-compute quantiles for all parameters at once (more efficient)
        quantile_cache = {}
        for group_name, params in param_groups.items():
            prefix = group_name.split("_")[0] + "_"
            for param in params:
                if param in self.fit_result:
                    # Compute quantile once and convert to pandas
                    quantile_cache[param] = (
                        self.fit_result[param]
                        .quantile(quantile, dim=["chain", "draw"])
                        .to_pandas()
                    )

        # Build channel info dictionary efficiently
        channels_info = {}
        for channel in channels:
            channel_info = {}
            for group_name, params in param_groups.items():
                prefix = group_name.split("_")[0] + "_"
                param_dict = {
                    param.replace(prefix, ""): quantile_cache[param].to_dict()[channel]
                    for param in params
                    if param in quantile_cache
                }
                channel_info[group_name] = param_dict
            channels_info[channel] = channel_info

        return channels_info

    def _plot_response_curve_fit(
        self,
        ax: plt.Axes,
        channel: str,
        color_index: int,
        xlim_max: int | None,
        label: str = "Fit Curve",
        quantile_lower: float = 0.05,
        quantile_upper: float = 0.95,
    ) -> None:
        """Plot the curve fit for the given channel based on the estimation of the parameters by the model.

        Parameters
        ----------
        ax : plt.Axes
            The matplotlib axes object where the plot should be drawn.
        channel : str
            The name of the channel for which the curve fit is being plotted.
        color_index : int
            An index used for color selection to ensure distinct colors for multiple plots.
        xlim_max: int
            The maximum value to be plot on the X-axis
        label: str
            The label for the curve being plotted, default is "Fit Curve".
        quantile_lower: float
            The lower quantile for parameter estimation, default is 0.05.
        quantile_upper: float
            The upper quantile for parameter estimation, default is 0.95.

        Returns
        -------
        None
            The function modifies the given axes object in-place and doesn't return any object.

        """
        if self.X is not None:
            x_mean = np.max(self.X[channel])

        # Set x_limit based on the method or xlim_max
        if xlim_max is not None:
            x_limit = xlim_max
        else:
            x_limit = x_mean

        # Generate x_fit and y_fit
        x_fit = np.linspace(0, x_limit, 1000)
        upper_params = self.format_recovered_transformation_parameters(
            quantile=quantile_upper
        )
        lower_params = self.format_recovered_transformation_parameters(
            quantile=quantile_lower
        )
        mid_params = self.format_recovered_transformation_parameters(quantile=0.5)
        y_fit = self.saturation.function(
            x=x_fit, **mid_params[channel]["saturation_params"]
        ).eval()

        y_fit_lower = self.saturation.function(
            x=x_fit, **lower_params[channel]["saturation_params"]
        ).eval()
        y_fit_upper = self.saturation.function(
            x=x_fit, **upper_params[channel]["saturation_params"]
        ).eval()

        # scale all y fit values to the original scale using
        # `mmm.target_transformer.named_steps["scaler"].scale_.item()`
        y_fit = (
            self.get_target_transformer()
            .inverse_transform(y_fit.reshape(-1, 1))
            .flatten()
        )
        y_fit_lower = (
            self.get_target_transformer()
            .inverse_transform(y_fit_lower.reshape(-1, 1))
            .flatten()
        )
        y_fit_upper = (
            self.get_target_transformer()
            .inverse_transform(y_fit_upper.reshape(-1, 1))
            .flatten()
        )

        # scale x fit values
        x_fit = self._channel_map_scales()[channel] * x_fit

        ax.fill_between(
            x_fit, y_fit_lower, y_fit_upper, color=f"C{color_index}", alpha=0.25
        )
        ax.plot(x_fit, y_fit, color=f"C{color_index}", label=label, alpha=0.6)

        ax.set(xlabel="Spent", ylabel="Contribution")
        ax.legend()

    def plot_direct_contribution_curves(
        self,
        show_fit: bool = False,
        same_axes: bool = False,
        xlim_max: int | None = None,
        channels: list[str] | None = None,
        quantile_lower: float = 0.05,
        quantile_upper: float = 0.95,
    ) -> plt.Figure:
        """Plot the direct contribution curves for each marketing channel.

        The term "direct" refers to the fact that we plot costs vs immediate returns and
        we do not take into account the lagged effects of the channels e.g. adstock transformations.

        Parameters
        ----------
        show_fit : bool, optional
            If True, the function will also plot the curve fit. Defaults to False.
        xlim_max : int, optional
            The maximum value to be plot on the X-axis. If not provided, the maximum value in the data will be used.
        channels : List[str], optional
            A list of channels to plot. If not provided, all channels will be plotted.
        same_axes : bool, optional
            If True, all channels will be plotted on the same axes. Defaults to False.

        Returns
        -------
        plt.Figure
            A matplotlib Figure object with the direct contribution curves.

        """
        channels_to_plot = self.channel_columns if channels is None else channels

        if not all(channel in self.channel_columns for channel in channels_to_plot):
            unknown_channels = set(channels_to_plot) - set(self.channel_columns)
            raise ValueError(
                f"The provided channels must be a subset of the available channels. Got {unknown_channels}"
            )

        if len(channels_to_plot) != len(set(channels_to_plot)):
            raise ValueError("The provided channels must be unique.")

        channel_contribution = self.compute_channel_contribution_original_scale().mean(
            ["chain", "draw"]
        )

        if same_axes:
            nrows = 1
            figsize = (12, 4)

            def label_func(channel):
                return f"{channel} Data Points"

            def legend_title_func(channel):
                return "Legend"

        else:
            nrows = len(channels_to_plot)
            figsize = (12, 4 * len(channels_to_plot))

            def label_func(channel):
                return "Data Points"

            def legend_title_func(channel):
                return f"{channel} Legend"

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharex=False,
            sharey=False,
            figsize=figsize,
            layout="constrained",
        )

        if same_axes:
            axes_channels: list[tuple[Any, str]] | Any = [
                (axes, channel) for channel in channels_to_plot
            ]
        else:
            axes_channels = zip(np.ravel(axes), channels_to_plot, strict=False)

        for i, (ax, channel) in enumerate(axes_channels):
            if self.X is not None:
                x = self.X[channel].to_numpy()
                y = channel_contribution.sel(channel=channel).to_numpy()

                label = label_func(channel)
                ax.scatter(x, y, label=label, color=f"C{i}")

                if show_fit:
                    label = f"{channel} Fit Curve" if same_axes else "Fit Curve"
                    self._plot_response_curve_fit(
                        ax=ax,
                        channel=channel,
                        color_index=i,
                        xlim_max=xlim_max,
                        label=label,
                        quantile_lower=quantile_lower,
                        quantile_upper=quantile_upper,
                    )

                title = legend_title_func(channel)
                ax.legend(
                    loc="upper left",
                    facecolor="white",
                    title=title,
                    fontsize="small",
                )

                ax.set(xlabel="Spent", ylabel="Contribution")

        fig.suptitle("Direct response curves", fontsize=16)
        return fig

    def _transform_to_original_scale_new(
        self, samples: DataArray, var_names: list[str] | None = None
    ) -> DataArray:
        """Transform samples to original scale using new scaling approach.

        Parameters
        ----------
        samples : DataArray
            Samples in scaled space
        var_names : list[str] | None
            Variable names requested in sampling.

        Returns
        -------
        DataArray
            Samples in original scale
        """
        vars_to_transform = var_names if var_names is not None else [self.output_var]
        for var_name in (v for v in vars_to_transform if v in samples):
            samples[var_name] *= self.target_scale
        return samples

    def _transform_to_original_scale_legacy(
        self,
        samples: DataArray,
        var_names: list[str] | None,
    ) -> DataArray:
        """Transform samples to original scale using legacy transformer approach.

        Parameters
        ----------
        samples : DataArray
            Samples in scaled space
        var_names : list[str] | None
            Variable names requested in sampling

        Returns
        -------
        DataArray
            Samples in original scale
        """
        # Transform output variable
        if self.output_var in samples:
            samples[self.output_var] = apply_sklearn_transformer_across_dim(
                data=samples[self.output_var],
                func=self.get_target_transformer().inverse_transform,
                dim_name="date",
            )

        # Handle intercept if needed
        intercept_condition = (
            var_names is not None
            and "intercept" in var_names
            and not self.time_varying_intercept
        )

        if intercept_condition:
            # Expand intercept to date dimension for transformation
            samples["intercept"] = (
                samples["intercept"]
                .expand_dims(dim={"date": samples["date"]}, axis=0)
                .rename("intercept")
            )

            samples = apply_sklearn_transformer_across_dim(
                data=samples,
                func=self.get_target_transformer().inverse_transform,
                dim_name="date",
            )

            # Remove date dimension after transformation
            samples["intercept"] = samples["intercept"].isel(date=0)

        return samples

    def sample_posterior_predictive(
        self,
        X=None,
        extend_idata: bool = True,
        combined: bool = True,
        include_last_observations: bool = False,
        original_scale: bool = True,
        **sample_posterior_predictive_kwargs,
    ) -> DataArray:
        """Sample from the model's posterior predictive distribution.

        Parameters
        ----------
        X : array, shape (n_pred, n_features)
            The input data used for prediction.
        extend_idata : bool, optional
            Boolean determining whether the predictions should be added to inference data object. Defaults to True.
        combined: bool, optional
            Combine chain and draw dims into sample. Won't work if a dim named sample already exists. Defaults to True.
        include_last_observations: bool, optional
            Boolean determining whether to include the last observations of the training data in order to carry over
            costs with the adstock transformation. Assumes that X are the next predictions following the
            training data.Defaults to False.
        original_scale: bool, optional
            Boolean determining whether to return the predictions in the original scale of the target variable.
            Defaults to True.
        **sample_posterior_predictive_kwargs
            Additional arguments to pass to pymc.sample_posterior_predictive

        Returns
        -------
        posterior_predictive_samples : DataArray, shape (n_pred, samples)
            Posterior predictive samples for each input X

        """
        X = _handle_deprecate_pred_argument(X, "X", sample_posterior_predictive_kwargs)

        if include_last_observations:
            X = pd.concat(
                [self.X.iloc[-self.adstock.l_max :, :], X], axis=0
            ).sort_values(by=self.date_column)

        self._data_setter(X)

        # Sample predictions
        with self.model:
            post_pred = pm.sample_posterior_predictive(
                self.idata, **sample_posterior_predictive_kwargs
            )
            if extend_idata:
                self.idata.extend(post_pred, join="right")  # type: ignore

        # Extract samples
        group = (
            "predictions"
            if sample_posterior_predictive_kwargs.get("predictions", False)
            else "posterior_predictive"
        )
        posterior_predictive_samples = az.extract(post_pred, group, combined=combined)

        # Remove adstock observations if included
        if include_last_observations:
            posterior_predictive_samples = posterior_predictive_samples.isel(
                date=slice(self.adstock.l_max, None)
            )

        # Transform to original scale if requested
        if original_scale:
            self._posterior_predictive_samples_original_scale = True
            var_names = sample_posterior_predictive_kwargs.get("var_names")
            if self._has_new_scaling():
                posterior_predictive_samples = self._transform_to_original_scale_new(
                    posterior_predictive_samples, var_names
                )
            else:
                posterior_predictive_samples = self._transform_to_original_scale_legacy(
                    posterior_predictive_samples, var_names
                )

        return posterior_predictive_samples

    def add_lift_test_measurements(
        self,
        df_lift_test: pd.DataFrame,
        dist: type[pm.Distribution] = pm.Gamma,
        name: str = "lift_measurements",
    ) -> None:
        """Add lift tests to the model.

        The model difference of a channel's saturation curve is created
        from `x` and `x + delta_x` for each channel. This random variable is
        then conditioned using the empirical lift, `delta_y`, and `sigma` of the lift test
        with the specified distribution `dist`.

        The pseudo-code for the lift test is as follows:

        .. code-block:: python

            model_estimated_lift = saturation_curve(x + delta_x) - saturation_curve(x)
            empirical_lift = delta_y
            dist(abs(model_estimated_lift), sigma=sigma, observed=abs(empirical_lift))


        The model has to be built before adding the lift tests.

        Parameters
        ----------
        df_lift_test : pd.DataFrame
            DataFrame with lift test results with at least the following columns:
                * `channel`: channel name. Must be present in `channel_columns`.
                * `x`: x axis value of the lift test.
                * `delta_x`: change in x axis value of the lift test.
                * `delta_y`: change in y axis value of the lift test.
                * `sigma`: standard deviation of the lift test.
        dist : pm.Distribution, optional
            The distribution to use for the likelihood, by default pm.Gamma
        name : str, optional
            The name of the likelihood of the lift test contribution(s),
            by default "lift_measurements". Name change required if calling
            this method multiple times.

        Raises
        ------
        RuntimeError
            If the model has not been built yet.
        KeyError
            If the 'channel' column is not present in df_lift_test.

        Examples
        --------
        Build the model first then add lift test measurements.

        .. code-block:: python

            import pandas as pd
            import numpy as np

            from pymc_marketing.mmm import (
                GeometricAdstock,
                LogisticSaturation,
                MMM,
            )

            model = MMM(
                date_column="date_week",
                channel_columns=["x1", "x2"],
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                control_columns=[
                    "event_1",
                    "event_2",
                ],
                yearly_seasonality=2,
            )

            X: pd.DataFrame = ...
            y: np.ndarray = ...

            model.build_model(X, y)

            df_lift_test = pd.DataFrame(
                {
                    "channel": ["x1", "x1"],
                    "x": [1, 1],
                    "delta_x": [0.1, 0.2],
                    "delta_y": [0.1, 0.1],
                    "sigma": [0.1, 0.1],
                }
            )

            model.add_lift_test_measurements(df_lift_test)

        """
        if not hasattr(self, "model"):
            raise RuntimeError(
                "The model has not been built yet. Please, build the model first."
            )

        if "channel" not in df_lift_test.columns:
            raise KeyError(
                "The 'channel' column is required to map the lift measurements to the model."
            )

        if self.time_varying_media and "date" not in df_lift_test.columns:
            # `time_varying_media=True` parameter requires the date in the df_lift_test DataFrame.
            # The `add_lift_test_measurements` method itself doesn't need a date
            # We need to make sure the `date` coord is present in model_coords
            # By adding this we make sure the model_coords match
            df_lift_test["date"] = pd.to_datetime(self.model_coords["date"][0])

        # Store df_lift_test for testing purposes
        self._last_lift_test_df = df_lift_test

        df_lift_test_scaled = scale_lift_measurements(
            df_lift_test=df_lift_test,
            channel_col="channel",
            channel_columns=self.channel_columns,  # type: ignore
            channel_transform=self.channel_transformer.transform,
            target_transform=self.target_transformer.transform,
        )
        # This is coupled with the name of the
        # latent process Deterministic
        time_varying_var_name = (
            "media_temporal_latent_multiplier" if self.time_varying_media else None
        )
        add_lift_measurements_to_likelihood_from_saturation(
            df_lift_test=df_lift_test_scaled,
            saturation=self.saturation,
            time_varying_var_name=time_varying_var_name,
            model=self.model,
            dist=dist,
            name=name,
        )

    def _validate_controls_for_synth_dataset(
        self, controls: list[str] | None
    ) -> list[str]:
        """Validate and return controls for synthetic dataset creation.

        Parameters
        ----------
        controls : list[str] | None
            Requested control columns

        Returns
        -------
        list[str]
            Validated control columns

        Raises
        ------
        ValueError
            If controls are requested but model has no controls
        """
        X_data = self.preprocessed_data["X"]
        mmm_has_controls = (
            self.control_columns is not None
            and len(self.control_columns) > 0
            and isinstance(X_data, pd.DataFrame)
            and all(column in X_data.columns for column in self.control_columns)
        )

        if controls is not None and not mmm_has_controls:
            raise ValueError(
                "The model was built without controls and cannot translate the provided controls to contributions. "
                "Remove the controls from the function call and try again."
            )

        return controls if (controls is not None and mmm_has_controls) else []

    def _generate_future_dates(
        self,
        last_date: pd.Timestamp,
        time_granularity: Literal["daily", "weekly", "monthly", "quarterly", "yearly"],
        time_length: int,
    ) -> list[pd.Timestamp]:
        """Generate future dates based on time granularity.

        Parameters
        ----------
        last_date : pd.Timestamp
            The last date in the original dataset
        time_granularity : str
            Time granularity for the dates
        time_length : int
            Number of periods to generate

        Returns
        -------
        list[pd.Timestamp]
            List of future dates
        """
        # Use pandas date_range for efficient date generation
        if time_granularity == "daily":
            freq = "D"
        elif time_granularity == "weekly":
            freq = "W"
        elif time_granularity == "monthly":
            freq = "MS"  # Month start
        elif time_granularity == "quarterly":
            freq = "QS"  # Quarter start
        else:  # yearly
            freq = "YS"  # Year start

        # Generate dates efficiently using pandas
        return pd.date_range(start=last_date, periods=time_length + 1, freq=freq)[
            1:
        ].tolist()

    def _create_synth_dataset(
        self,
        df: pd.DataFrame,
        date_column: str,
        allocation_strategy: DataArray,
        channels: list[str] | tuple[str],
        controls: list[str] | None,
        target_col: str,
        time_granularity: Literal["daily", "weekly", "monthly", "quarterly", "yearly"],
        time_length: int,
        lag: int,
        noise_level: float = 0.01,
    ) -> pd.DataFrame:
        """Create a synthetic dataset based on the given allocation strategy (Budget) and time granularity.

        **Important**: When generating the posterior predicive distribution for the target with the optimized budget,
        we are setting the control variables to zero! This is done because in many situations we do not have all the
        control variables in the future (e.g. outlier control, special events).

        Parameters
        ----------
        df : pd.DataFrame
            The original dataset.
        date_column : str
            The name of the date column in the dataset.
        allocation_strategy : DataArray
            A DataArray mapping channel names to their corresponding allocation values.
        channels : list[str] | tuple[str]
            A list or tuple of channel names.
        controls : list[str] | None
            A list of control column names or None if no controls are present.
        target_col : str
            The name of the target column.
        time_granularity : Literal["daily", "weekly", "monthly", "quarterly", "yearly"]
            The time granularity of the synthetic dataset: 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'.
        time_length : int
            The length of the synthetic dataset in terms of the time granularity.
        lag : int
            The lag value (not used in this function).
        noise_level : int
            The level of noise added to the allocation strategy (by default 1%).

        Returns
        -------
        pd.DataFrame
            A synthetic dataset with the specified allocation strategy and time granularity.

        Raises
        ------
        ValueError
            If the time granularity is not supported.

        """
        valid_granularities = ["daily", "weekly", "monthly", "quarterly", "yearly"]
        if time_granularity not in valid_granularities:
            raise ValueError(
                f"Unsupported time granularity. Choose from {', '.join(valid_granularities)}."
            )

        if allocation_strategy.dims != ("channel",):
            raise ValueError(
                "The allocation strategy DataArray must have a single dimension named 'channel'. "
                f"Got {allocation_strategy.dims}"
            )

        # Validate controls
        _controls = self._validate_controls_for_synth_dataset(controls)

        # Generate future dates
        last_date = pd.to_datetime(df[date_column]).max()
        new_dates = self._generate_future_dates(
            last_date, time_granularity, time_length
        )

        # Vectorized creation of synthetic dataset
        # Extract allocation values once
        channel_allocations = allocation_strategy.to_pandas()

        # Create noise matrix efficiently
        noise = np.random.normal(0, noise_level, size=(time_length, len(channels)))
        channel_values = channel_allocations.values * (1 + noise)

        # Build DataFrame efficiently using dict of arrays
        data_dict: dict[str, Any] = {self.date_column: new_dates}
        data_dict.update(dict(zip(channels, channel_values.T, strict=False)))

        # Add controls efficiently if present (as arrays for proper type consistency)
        if _controls:
            zeros_array = np.zeros(time_length)
            for control in _controls:
                data_dict[control] = zeros_array

        data_dict[target_col] = np.zeros(time_length)

        return pd.DataFrame(data_dict)

    def sample_response_distribution(
        self,
        allocation_strategy: DataArray | dict[str, float],
        time_granularity: Literal["daily", "weekly", "monthly", "quarterly", "yearly"],
        num_periods: int,
        noise_level: float,
    ) -> az.InferenceData:
        """Generate synthetic dataset and sample posterior predictive based on allocation.

        Parameters
        ----------
        allocation_strategy : DataArray or dict[str, float]
            The allocation strategy for the channels.
        time_granularity : Literal["daily", "weekly", "monthly", "quarterly", "yearly"]
            The granularity of the time units (e.g., 'daily', 'weekly', 'monthly').
        num_periods : int
            The number of time periods for prediction.
        noise_level : float
            The level of noise to add to the synthetic data.

        Returns
        -------
        az.InferenceData
            The posterior predictive samples based on the synthetic dataset.
        """
        if isinstance(allocation_strategy, dict):
            # For backward compatibility
            allocation_strategy = DataArray(
                pd.Series(allocation_strategy), dims=("channel",)
            )

        synth_dataset = self._create_synth_dataset(
            df=self.X,
            date_column=self.date_column,
            allocation_strategy=allocation_strategy,
            channels=self.channel_columns,
            controls=self.control_columns,
            target_col=self.output_var,
            time_granularity=time_granularity,
            time_length=num_periods,
            lag=self.adstock.l_max,
            noise_level=noise_level,
        )

        constant_data = allocation_strategy.to_dataset(name="allocation")

        return self.sample_posterior_predictive(
            X=synth_dataset,
            extend_idata=False,
            include_last_observations=True,
            original_scale=False,
            var_names=["y", "channel_contribution"],
            progressbar=False,
        ).merge(constant_data)

    def _set_predictors_for_optimization(self, num_periods: int) -> pm.Model:
        """Return the respective PyMC model with any predictors set for optimization."""
        if "date" not in self.model.coords:
            return self.model

        # Coords aren't used anywhere, so just go with integers
        opt_date_coords = range(num_periods + self.adstock.l_max)

        # Copy the model and update the "date" coordinates to the ones used for optimization (num_periods + l_max)
        model = self.model.copy()

        if "time_index" in model.named_vars:
            # Models with HSGP have a time_index data variable
            # We set them to start after the last date in the training data
            start_date = model["time_index"].get_value(borrow=True)[-1]
            opt_time_index = (
                np.arange(num_periods + self.adstock.l_max) + start_date + 1
            ).astype(start_date.dtype)
            model.set_data(
                "time_index", opt_time_index, coords={"date": opt_date_coords}
            )
        else:
            model.set_dim(
                "date", new_length=len(opt_date_coords), coord_values=opt_date_coords
            )
        return model

    def optimize_budget(
        self,
        budget: float | int,
        num_periods: int,
        budget_bounds: DataArray | dict[str, tuple[float, float]] | None = None,
        response_variable: str = "total_contribution",
        utility_function: UtilityFunctionType = average_response,
        constraints: Sequence[dict[str, Any]] = (),
        default_constraints: bool = True,
        callback: bool = False,
        **minimize_kwargs,
    ) -> (
        tuple[DataArray, OptimizeResult]
        | tuple[DataArray, OptimizeResult, list[dict[str, Any]]]
    ):
        """Optimize the given budget based on the specified utility function over a specified time period.

        .. deprecated:: 0.0.3
            This function optimizes the allocation of a given budget across different channels
           to maximize the response, considering adstock and saturation effects. It scales the
           budget and budget bounds, performs the optimization, and generates a synthetic dataset
           for posterior predictive sampling.

            The function first scales the budget and budget bounds using the maximum scale
            of the channel transformer. It then uses the `BudgetOptimizer` to allocate the
            budget, and creates a synthetic dataset based on the optimal allocation. Finally,
            it performs posterior predictive sampling on the synthetic dataset.

            **Important**: When generating the posterior predicive distribution for the target with the
            optimized budget, we are setting the control variables to zero! This is done because in many
            situations we do not have all the control variables in the future (e.g. outlier control,
            special events).

        Parameters
        ----------
        budget : float or int
            The total budget to be allocated.
        num_periods : int
            The number of time units over which the budget is to be allocated.
        budget_bounds : DataArrayr or dict[str, tuple[float, float]], optional
            An xarray DataArary or dictionary specifying the lower and upper bounds for the budget allocation
            for each channel. If None, no bounds are applied.
        response_variable : str, optional
            The response variable to optimize. Default is "total_contribution".
        utility_function : UtilityFunctionType, optional
            The utility function to maximize. Default is the mean of the response distribution.
        custom_constraints : list[dict[str, Any]], optional
            Custom constraints for the optimization. If None, no custom constraints are applied. Format:
            [{"key":...,"constraint_fun":...,"constraint_type":...}]
        default_constraints : bool, optional
            Whether to add the default sum constraint to the optimizer. Default is True.
        callback : bool, optional
            Whether to return callback information tracking optimization progress. When True, returns a third
            element containing a list of dictionaries with optimization information at each iteration.
            Default is False for backward compatibility.
        **minimize_kwargs
            Additional arguments to pass to the `BudgetOptimizer`.

        Returns
        -------
        az.InferenceData
            The posterior predictive samples generated from the synthetic dataset.

        Raises
        ------
        ValueError
            If the time granularity is not supported.

        ValueError
            If the noise level is not a float.
        """
        warnings.warn(
            "This method is deprecated and will be removed in a future version. "
            "Please migrate to the `Multidimensal.MMM` class.",
            DeprecationWarning,
            stacklevel=2,
        )

        from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

        allocator = BudgetOptimizer(
            num_periods=num_periods,
            utility_function=utility_function,
            response_variable=response_variable,
            custom_constraints=constraints,
            default_constraints=default_constraints,
            model=self,
        )

        return allocator.allocate_budget(
            total_budget=budget,
            budget_bounds=budget_bounds,
            callback=callback,
            **minimize_kwargs,
        )

    def plot_budget_allocation(
        self,
        samples: Dataset,
        figsize: tuple[float, float] = (12, 6),
        ax: plt.Axes | None = None,
        original_scale: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the budget allocation and channel contributions.

        Parameters
        ----------
        samples : Dataset
            The dataset containing the channel contributions.
        figsize : tuple[float, float], optional
            The size of the figure to be created, by default (12, 6).
        ax : plt.Axes, optional
            The axis to plot on. If None, a new figure and axis will be created.
        original_scale : bool, optional
            A boolean flag to determine if the values should be plotted in their original scale,
            by default True.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The matplotlib figure object and axis containing the plot.

        """
        channel_contribution = (
            samples["channel_contribution"].mean(dim=["date", "sample"]).to_numpy()
        )

        if not original_scale:
            channel_contribution /= self.get_target_transformer()["scaler"].scale_

        allocated_spend = samples.allocation.to_numpy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        bar_width = 0.35
        opacity = 0.7

        index = np.arange(len(self.channel_columns))

        bars1 = ax.bar(
            index,
            allocated_spend,
            bar_width,
            color="C0",
            alpha=opacity,
            label="Allocated Spend",
        )

        ax2 = ax.twinx()

        bars2 = ax2.bar(
            index + bar_width,
            channel_contribution,
            bar_width,
            color="C1",
            alpha=opacity,
            label="Channel Contribution",
        )

        ax.set_xlabel("Channels")
        ax.set_ylabel("Allocate Spend", color="C0")
        ax.tick_params(axis="x", rotation=90)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(self.channel_columns)

        ax.set_ylabel("Allocate Spend", color="C0", labelpad=10)
        ax2.set_ylabel("Channel Contributions", color="C1", labelpad=10)

        ax.grid(False)
        ax2.grid(False)

        bars = [bars1, bars2]
        labels = ["Allocated Spend", "Channel Contributions"]
        ax.legend(bars, labels, loc="best")

        return fig, ax

    def plot_allocated_contribution_by_channel(
        self,
        samples: Dataset,
        lower_quantile: float = 0.025,
        upper_quantile: float = 0.975,
        original_scale: bool = True,
    ) -> plt.Figure:
        """Plot the allocated contribution by channel with uncertainty intervals.

        This function visualizes the mean allocated contributions by channel along with
        the uncertainty intervals defined by the lower and upper quantiles. The contributions
        can be plotted on the original scale or the transformed scale.

        Parameters
        ----------
        samples : Dataset
            The dataset containing the samples of channel contributions.
        lower_quantile : float, optional
            The lower quantile for the uncertainty interval. Default is 0.025.
        upper_quantile : float, optional
            The upper quantile for the uncertainty interval. Default is 0.975.
        original_scale : bool, optional
            If True, the contributions are plotted on the original scale. Default is True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the plot.

        """
        if original_scale:
            # Use original scale variables if available
            if "channel_contribution_original_scale" in samples:
                channel_contribution = samples["channel_contribution_original_scale"]
            else:
                # Fallback for backward compatibility
                channel_contribution = samples["channel_contribution"]
        else:
            # Use scaled contributions directly
            channel_contribution = samples["channel_contribution"]

        fig, ax = plt.subplots()
        channel_contribution.mean(dim="sample").plot(hue="channel", ax=ax)

        for channel in self.model_coords["channel"]:
            ax.fill_between(
                x=channel_contribution.date.values,
                y1=channel_contribution.sel(channel=channel).quantile(
                    lower_quantile, dim="sample"
                ),
                y2=channel_contribution.sel(channel=channel).quantile(
                    upper_quantile, dim="sample"
                ),
                alpha=0.1,
            )
        return fig
