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
"""Multidimensional Marketing Mix Model class."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Annotated, Any, Literal

import arviz as az
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pydantic import Field, InstanceOf, validate_call
from pymc.model.fgraph import clone_model as cm
from pymc.util import RandomState
from scipy.optimize import OptimizeResult

from pymc_marketing.mmm import SoftPlusHSGP
from pymc_marketing.mmm.additive_effect import EventAdditiveEffect, MuEffect
from pymc_marketing.mmm.budget_optimizer import OptimizerCompatibleModelWrapper
from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
    adstock_from_dict,
)
from pymc_marketing.mmm.components.saturation import (
    SaturationTransformation,
    saturation_from_dict,
)
from pymc_marketing.mmm.events import EventEffect
from pymc_marketing.mmm.fourier import YearlyFourier
from pymc_marketing.mmm.lift_test import (
    add_lift_measurements_to_likelihood_from_saturation,
    scale_lift_measurements,
)
from pymc_marketing.mmm.plot import MMMPlotSuite
from pymc_marketing.mmm.scaling import Scaling, VariableScaling
from pymc_marketing.mmm.tvp import infer_time_index
from pymc_marketing.mmm.utility import UtilityFunctionType, average_response
from pymc_marketing.mmm.utils import (
    add_noise_to_channel_allocation,
    create_zero_dataset,
)
from pymc_marketing.model_builder import ModelBuilder, _handle_deprecate_pred_argument
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.model_graph import deterministics_to_flat
from pymc_marketing.prior import Prior, create_dim_handler

PYMC_MARKETING_ISSUE = "https://github.com/pymc-labs/pymc-marketing/issues/new"
warning_msg = (
    "This functionality is experimental and subject to change. "
    "If you encounter any issues or have suggestions, please raise them at: "
    f"{PYMC_MARKETING_ISSUE}"
)
warnings.warn(warning_msg, FutureWarning, stacklevel=1)


class MMM(ModelBuilder):
    """Marketing Mix Model class for estimating the impact of marketing channels on a target variable.

    This class implements the core functionality of a Marketing Mix Model (MMM), allowing for the
    specification of various marketing channels, adstock transformations, saturation effects,
    and time-varying parameters. It provides methods for fitting the model to data, making
    predictions, and visualizing the results.

    Attributes
    ----------
    date_column : str
        The name of the column representing the date in the dataset.
    channel_columns : list[str]
        A list of columns representing the marketing channels.
    target_column : str
        The name of the column representing the target variable to be predicted.
    adstock : AdstockTransformation
        The adstock transformation to apply to the channel data.
    saturation : SaturationTransformation
        The saturation transformation to apply to the channel data.
    time_varying_intercept : bool
        Whether to use a time-varying intercept in the model.
    time_varying_media : bool
        Whether to use time-varying effects for media channels.
    dims : tuple | None
        Additional dimensions for the model.
    scaling : Scaling | dict | None
        Scaling methods to be used for the target variable and the marketing channels.
        Defaults to max scaling for both.
    model_config : dict | None
        Configuration settings for the model.
    sampler_config : dict | None
        Configuration settings for the sampler.
    control_columns : list[str] | None
        A list of control variables to include in the model.
    yearly_seasonality : int | None
        The number of yearly seasonalities to include in the model.
    adstock_first : bool
        Whether to apply adstock transformations before saturation.
    """

    _model_type: str = "MMMM (Multi-Dimensional Marketing Mix Model)"
    version: str = "0.0.1"

    @validate_call
    def __init__(
        self,
        date_column: str = Field(..., description="Column name of the date variable."),
        channel_columns: list[str] = Field(
            min_length=1, description="Column names of the media channel variables."
        ),
        target_column: str = Field(..., description="The name of the target column."),
        adstock: InstanceOf[AdstockTransformation] = Field(
            ..., description="Type of adstock transformation to apply."
        ),
        saturation: InstanceOf[SaturationTransformation] = Field(
            ...,
            description="The saturation transformation to apply to the channel data.",
        ),
        time_varying_intercept: Annotated[
            bool,
            Field(strict=True, description="Whether to use a time-varying intercept"),
        ] = False,
        time_varying_media: Annotated[
            bool,
            Field(strict=True, description="Whether to use time-varying media effects"),
        ] = False,
        dims: tuple[str, ...] | None = Field(
            None, description="Additional dimensions for the model."
        ),
        scaling: InstanceOf[Scaling] | dict | None = Field(
            None, description="Scaling configuration for the model."
        ),
        model_config: dict | None = Field(
            None, description="Configuration settings for the model."
        ),
        sampler_config: dict | None = Field(
            None, description="Configuration settings for the sampler."
        ),
        control_columns: Annotated[
            list[str] | None,
            Field(
                min_length=1,
                description="A list of control variables to include in the model.",
            ),
        ] = None,
        yearly_seasonality: Annotated[
            int | None,
            Field(
                gt=0,
                description="The number of yearly seasonalities to include in the model.",
            ),
        ] = None,
        adstock_first: Annotated[
            bool,
            Field(strict=True, description="Apply adstock before saturation?"),
        ] = True,
    ) -> None:
        """Define the constructor method."""
        # Your existing initialization logic
        self.control_columns = control_columns
        self.time_varying_intercept = time_varying_intercept
        self.time_varying_media = time_varying_media
        self.date_column = date_column

        self.adstock = adstock
        self.saturation = saturation
        self.adstock_first = adstock_first

        dims = dims if dims is not None else ()
        self.dims = dims

        if isinstance(scaling, dict):
            scaling = deepcopy(scaling)

            if "channel" not in scaling:
                scaling["channel"] = VariableScaling(method="max", dims=self.dims)
            if "target" not in scaling:
                scaling["target"] = VariableScaling(method="max", dims=self.dims)

            scaling = Scaling(**scaling)

        self.scaling: Scaling = scaling or Scaling(
            target=VariableScaling(method="max", dims=self.dims),
            channel=VariableScaling(method="max", dims=self.dims),
        )

        if set(self.scaling.target.dims).difference([*self.dims, "date"]):
            raise ValueError(
                f"Target scaling dims {self.scaling.target.dims} must contain {self.dims} and 'date'"
            )

        if set(self.scaling.channel.dims).difference([*self.dims, "channel", "date"]):
            raise ValueError(
                f"Channel scaling dims {self.scaling.channel.dims} must contain {self.dims}, 'channel', and 'date'"
            )

        model_config = model_config if model_config is not None else {}
        sampler_config = sampler_config
        model_config = parse_model_config(
            model_config,  # type: ignore
            non_distributions=["intercept_tvp_config", "media_tvp_config"],
        )

        if model_config is not None:
            self.adstock.update_priors({**self.default_model_config, **model_config})
            self.saturation.update_priors({**self.default_model_config, **model_config})

        self._check_compatible_media_dims()

        self.date_column = date_column
        self.target_column = target_column
        self.channel_columns = channel_columns
        self.yearly_seasonality = yearly_seasonality

        super().__init__(model_config=model_config, sampler_config=sampler_config)

        if self.yearly_seasonality is not None:
            self.yearly_fourier = YearlyFourier(
                n_order=self.yearly_seasonality,
                prefix="fourier_mode",
                prior=self.model_config["gamma_fourier"],
                variable_name="gamma_fourier",
            )

        self.mu_effects: list[MuEffect] = []

    def _check_compatible_media_dims(self) -> None:
        allowed_dims = set(self.dims).union({"channel"})

        if not set(self.adstock.combined_dims).issubset(allowed_dims):
            raise ValueError(
                f"Adstock effect dims {self.adstock.combined_dims} must contain {allowed_dims}"
            )

        if not set(self.saturation.combined_dims).issubset(allowed_dims):
            raise ValueError(
                f"Saturation effect dims {self.saturation.combined_dims} must contain {allowed_dims}"
            )

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration."""
        return {}

    def _data_setter(self, X, y=None): ...

    def add_events(
        self,
        df_events: pd.DataFrame,
        prefix: str,
        effect: EventEffect,
    ) -> None:
        """Add event effects to the model.

        This must be called before building the model.

        Parameters
        ----------
        df_events : pd.DataFrame
            The DataFrame containing the event data.
                * `name`: name of the event. Used as the model coordinates.
                * `start_date`: start date of the event
                * `end_date`: end date of the event
        prefix : str
            The prefix to use for the event effect and associated variables.
        effect : EventEffect
            The event effect to apply.

        Raises
        ------
        ValueError
            If the event effect dimensions do not contain the prefix and model dimensions.

        """
        if not set(effect.dims).issubset((prefix, self.dims)):
            raise ValueError(
                f"Event effect dims {effect.dims} must contain {prefix} and {self.dims}"
            )

        event_effect = EventAdditiveEffect(
            df_events=df_events,
            prefix=prefix,
            effect=effect,
        )
        self.mu_effects.append(event_effect)

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

    def create_idata_attrs(self) -> dict[str, str]:
        """Return the idata attributes for the model."""
        attrs = super().create_idata_attrs()
        attrs["dims"] = json.dumps(self.dims)
        attrs["date_column"] = self.date_column
        attrs["adstock"] = json.dumps(self.adstock.to_dict())
        attrs["saturation"] = json.dumps(self.saturation.to_dict())
        attrs["adstock_first"] = json.dumps(self.adstock_first)
        attrs["control_columns"] = json.dumps(self.control_columns)
        attrs["channel_columns"] = json.dumps(self.channel_columns)
        attrs["yearly_seasonality"] = json.dumps(self.yearly_seasonality)
        attrs["time_varying_intercept"] = json.dumps(self.time_varying_intercept)
        attrs["time_varying_media"] = json.dumps(self.time_varying_media)
        attrs["target_column"] = self.target_column
        attrs["scaling"] = json.dumps(self.scaling.model_dump(mode="json"))

        return attrs

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

    @classmethod
    def attrs_to_init_kwargs(cls, attrs: dict[str, str]) -> dict[str, Any]:
        """Convert the idata attributes to the model initialization kwargs."""
        return {
            "model_config": cls._model_config_formatting(
                json.loads(attrs["model_config"])
            ),
            "date_column": attrs["date_column"],
            "control_columns": json.loads(attrs["control_columns"]),
            "channel_columns": json.loads(attrs["channel_columns"]),
            "adstock": adstock_from_dict(json.loads(attrs["adstock"])),
            "saturation": saturation_from_dict(json.loads(attrs["saturation"])),
            "adstock_first": json.loads(attrs.get("adstock_first", "true")),
            "yearly_seasonality": json.loads(attrs["yearly_seasonality"]),
            "time_varying_intercept": json.loads(
                attrs.get("time_varying_intercept", "false")
            ),
            "target_column": attrs["target_column"],
            "time_varying_media": json.loads(attrs.get("time_varying_media", "false")),
            "sampler_config": json.loads(attrs["sampler_config"]),
            "dims": tuple(json.loads(attrs.get("dims", "[]"))),
            "scaling": json.loads(attrs.get("scaling", "null")),
        }

    @property
    def plot(self) -> MMMPlotSuite:
        """Use the MMMPlotSuite to plot the results."""
        self._validate_model_was_built()
        self._validate_idata_exists()
        return MMMPlotSuite(idata=self.idata)

    @property
    def default_model_config(self) -> dict:
        """Define the default model configuration."""
        base_config = {
            "intercept": Prior("Normal", mu=0, sigma=2, dims=self.dims),
            "likelihood": Prior(
                "Normal",
                sigma=Prior("HalfNormal", sigma=2, dims=self.dims),
                dims=self.dims,
            ),
            "gamma_control": Prior("Normal", mu=0, sigma=2, dims="control"),
            "gamma_fourier": Prior(
                "Laplace", mu=0, b=1, dims=(*self.dims, "fourier_mode")
            ),
        }

        if self.time_varying_intercept:
            base_config["intercept_tvp_config"] = {"ls_lower": 0.3, "ls_upper": 2.0}
        if self.time_varying_media:
            base_config["media_tvp_config"] = {"ls_lower": 0.3, "ls_upper": 2.0}

        return {
            **base_config,
            **self.adstock.model_config,
            **self.saturation.model_config,
        }

    @property
    def output_var(self) -> Literal["y"]:
        """Define target variable for the model.

        Returns
        -------
        str
            The target variable for the model.
        """
        return "y"

    def post_sample_model_transformation(self) -> None:
        """Post-sample model transformation in order to store the HSGP state from fit."""
        names = []
        if self.time_varying_intercept:
            names.extend(
                SoftPlusHSGP.deterministics_to_replace(
                    "intercept_temporal_latent_multiplier"
                )
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

    def _validate_idata_exists(self) -> None:
        """Validate that the idata exists."""
        if not hasattr(self, "idata"):
            raise ValueError("idata does not exist. Build the model first and fit.")

    def _validate_dims_in_multiindex(
        self, index: pd.MultiIndex, dims: tuple[str, ...], date_column: str
    ) -> list[str]:
        """Validate that dimensions exist in the MultiIndex.

        Parameters
        ----------
        index : pd.MultiIndex
            The MultiIndex to check
        dims : tuple[str, ...]
            The dimensions to validate
        date_column : str
            The name of the date column

        Returns
        -------
        list[str]
            List of valid dimensions found in the index

        Raises
        ------
        ValueError
            If date_column is not in the index
        """
        if date_column not in index.names:
            raise ValueError(f"date_column '{date_column}' not found in index")

        valid_dims = [dim for dim in dims if dim in index.names]
        return valid_dims

    def _validate_dims_in_dataframe(
        self, df: pd.DataFrame, dims: tuple[str, ...], date_column: str
    ) -> list[str]:
        """Validate that dimensions exist in the DataFrame columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to check
        dims : tuple[str, ...]
            The dimensions to validate
        date_column : str
            The name of the date column

        Returns
        -------
        list[str]
            List of valid dimensions found in the DataFrame

        Raises
        ------
        ValueError
            If date_column is not in the DataFrame
        """
        if date_column not in df.columns:
            raise ValueError(f"date_column '{date_column}' not found in DataFrame")

        valid_dims = [dim for dim in dims if dim in df.columns]
        return valid_dims

    def _validate_metrics(
        self, data: pd.DataFrame | pd.Series, metric_list: list[str]
    ) -> list[str]:
        """Validate that metrics exist in the data.

        Parameters
        ----------
        data : pd.DataFrame | pd.Series
            The data to check
        metric_list : list[str]
            The metrics to validate

        Returns
        -------
        list[str]
            List of valid metrics found in the data
        """
        if isinstance(data, pd.DataFrame):
            return [metric for metric in metric_list if metric in data.columns]
        else:  # pd.Series
            return [metric for metric in metric_list if metric in data.index.names]

    def _process_multiindex_series(
        self,
        series: pd.Series,
        date_column: str,
        valid_dims: list[str],
        metric_coordinate_name: str,
    ) -> xr.Dataset:
        """Process a MultiIndex Series into an xarray Dataset.

        Parameters
        ----------
        series : pd.Series
            The MultiIndex Series to process
        date_column : str
            The name of the date column
        valid_dims : list[str]
            List of valid dimensions
        metric_coordinate_name : str
            Name for the metric coordinate

        Returns
        -------
        xr.Dataset
            The processed xarray Dataset
        """
        # Reset index to get a DataFrame with all index levels as columns
        df = series.reset_index()

        # The series values become the metric values
        df_long = pd.DataFrame(
            {
                **{col: df[col] for col in [date_column, *valid_dims]},
                metric_coordinate_name: series.name,
                f"_{metric_coordinate_name}": series.values,
            }
        )

        # Drop duplicates to avoid non-unique MultiIndex
        df_long = df_long.drop_duplicates(
            subset=[date_column, *valid_dims, metric_coordinate_name]
        )

        # Convert to xarray, renaming date_column to "date" for internal consistency
        if valid_dims:
            df_long = df_long.rename(columns={date_column: "date"})
            return df_long.set_index(
                ["date", *valid_dims, metric_coordinate_name]
            ).to_xarray()
        df_long = df_long.rename(columns={date_column: "date"})
        return df_long.set_index(["date", metric_coordinate_name]).to_xarray()

    def _process_dataframe(
        self,
        df: pd.DataFrame,
        date_column: str,
        valid_dims: list[str],
        valid_metrics: list[str],
        metric_coordinate_name: str,
    ) -> xr.Dataset:
        """Process a DataFrame into an xarray Dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to process
        date_column : str
            The name of the date column
        valid_dims : list[str]
            List of valid dimensions
        valid_metrics : list[str]
            List of valid metrics
        metric_coordinate_name : str
            Name for the metric coordinate

        Returns
        -------
        xr.Dataset
            The processed xarray Dataset
        """
        # Reshape DataFrame to long format
        df_long = df.melt(
            id_vars=[date_column, *valid_dims],
            value_vars=valid_metrics,
            var_name=metric_coordinate_name,
            value_name=f"_{metric_coordinate_name}",
        )

        # Drop duplicates to avoid non-unique MultiIndex
        df_long = df_long.drop_duplicates(
            subset=[date_column, *valid_dims, metric_coordinate_name]
        )

        # Convert to xarray, renaming date_column to "date" for internal consistency
        df_long = df_long.rename(columns={date_column: "date"})
        if valid_dims:
            return df_long.set_index(
                ["date", *valid_dims, metric_coordinate_name]
            ).to_xarray()
        return df_long.set_index(["date", metric_coordinate_name]).to_xarray()

    def _create_xarray_from_pandas(
        self,
        data: pd.DataFrame | pd.Series,
        date_column: str,
        dims: tuple[str, ...],
        metric_list: list[str],
        metric_coordinate_name: str,
    ) -> xr.Dataset:
        """Create an xarray Dataset from a DataFrame or Series.

        This method handles both DataFrame and MultiIndex Series inputs, reshaping them
        into a long format and converting into an xarray Dataset. It validates dimensions
        and metrics, ensuring they exist in the input data.

        Parameters
        ----------
        data : pd.DataFrame | pd.Series
            The input data to transform
        date_column : str
            The name of the date column
        dims : tuple[str, ...]
            The dimensions to include
        metric_list : list[str]
            List of metrics to include
        metric_coordinate_name : str
            Name for the metric coordinate in the output

        Returns
        -------
        xr.Dataset
            The transformed data in xarray format

        Raises
        ------
        ValueError
            If date_column is not found in the data
        """
        # Validate dimensions based on input type
        if isinstance(data, pd.Series):
            valid_dims = self._validate_dims_in_multiindex(
                index=data.index,  # type: ignore
                dims=dims,  # type: ignore
                date_column=date_column,  # type: ignore
            )
            return self._process_multiindex_series(
                series=data,
                date_column=date_column,
                valid_dims=valid_dims,
                metric_coordinate_name=metric_coordinate_name,
            )
        else:  # pd.DataFrame
            valid_dims = self._validate_dims_in_dataframe(
                df=data,
                dims=dims,
                date_column=date_column,  # type: ignore
            )
            valid_metrics = self._validate_metrics(data, metric_list)
            return self._process_dataframe(
                df=data,
                date_column=date_column,
                valid_dims=valid_dims,
                valid_metrics=valid_metrics,
                metric_coordinate_name=metric_coordinate_name,
            )

    def _generate_and_preprocess_model_data(
        self,
        X: pd.DataFrame,  # type: ignore
        y: pd.Series,  # type: ignore
    ):
        self.X = X  # type: ignore
        self.y = y  # type: ignore

        dataarrays = []

        X_dataarray = self._create_xarray_from_pandas(
            data=X,
            date_column=self.date_column,
            dims=self.dims,
            metric_list=self.channel_columns,
            metric_coordinate_name="channel",
        )
        dataarrays.append(X_dataarray)

        # Create a temporary DataFrame to properly handle the y data transformation
        temp_y_df = pd.concat([self.X[[self.date_column, *self.dims]], self.y], axis=1)
        y_dataarray = self._create_xarray_from_pandas(
            data=temp_y_df.set_index([self.date_column, *self.dims])[
                self.target_column
            ],
            date_column=self.date_column,
            dims=self.dims,
            metric_list=[self.target_column],
            metric_coordinate_name="target",
        ).sum("target")
        dataarrays.append(y_dataarray)

        if self.control_columns is not None:
            control_dataarray = self._create_xarray_from_pandas(
                data=X,
                date_column=self.date_column,
                dims=self.dims,
                metric_list=self.control_columns,
                metric_coordinate_name="control",
            )
            dataarrays.append(control_dataarray)

        self.xarray_dataset = xr.merge(dataarrays).fillna(0)

        self.model_coords = {
            dim: self.xarray_dataset.coords[dim].values
            for dim in self.xarray_dataset.coords.dims
        }

        if self.time_varying_intercept | self.time_varying_media:
            self._time_index = np.arange(0, X[self.date_column].unique().shape[0])
            self._time_index_mid = X[self.date_column].unique().shape[0] // 2
            self._time_resolution = (
                X[self.date_column].iloc[1] - X[self.date_column].iloc[0]
            ).days

    def forward_pass(
        self,
        x: pt.TensorVariable | npt.NDArray[np.float64],
        dims: tuple[str, ...],
    ) -> pt.TensorVariable:
        """Transform channel input into target contributions of each channel.

        This method handles the ordering of the adstock and saturation
        transformations.

        This method must be called from without a pm.Model context but not
        necessarily in the instance's model. A dim named "channel" is required
        associated with the number of columns of `x`.

        Parameters
        ----------
        x : pt.TensorVariable | npt.NDArray[np.float64]
            The channel input which could be spends or impressions

        Returns
        -------
        The contributions associated with the channel input

        Examples
        --------
        >>> mmm = MMM(
            date_column="date_week",
            channel_columns=["channel_1", "channel_2"],
            target_column="target",
        )
        """
        first, second = (
            (self.adstock, self.saturation)
            if self.adstock_first
            else (self.saturation, self.adstock)
        )

        return second.apply(x=first.apply(x=x, dims=dims), dims=dims)

    def _compute_scales(self) -> None:
        """Compute and save scaling factors for channels and target."""
        self.scalers = xr.Dataset()

        channel_method = getattr(
            self.xarray_dataset["_channel"],
            self.scaling.channel.method,
        )
        self.scalers["_channel"] = channel_method(
            dim=("date", *self.scaling.channel.dims)
        )

        target_method = getattr(
            self.xarray_dataset["_target"],
            self.scaling.target.method,
        )
        self.scalers["_target"] = target_method(dim=("date", *self.scaling.target.dims))

    def get_scales_as_xarray(self) -> dict[str, xr.DataArray]:
        """Return the saved scaling factors as xarray DataArrays.

        Returns
        -------
        dict[str, xr.DataArray]
            A dictionary containing the scaling factors for channels and target.

        Examples
        --------
        >>> mmm = MMM(
            date_column="date_week",
            channel_columns=["channel_1", "channel_2"],
            target_column="target",
        )
        >>> mmm.build_model(X, y)
        >>> mmm.get_scales_as_xarray()
        """
        if not hasattr(self, "scalers"):
            raise ValueError(
                "Scales have not been computed yet. Build the model first."
            )

        return {
            "channel_scale": self.scalers._channel,
            "target_scale": self.scalers._target,
        }

    def _validate_model_was_built(self) -> None:
        """Validate that the model was built."""
        if not hasattr(self, "model"):
            raise ValueError(
                "Model was not built. Build the model first using MMM.build_model()"
            )

    def _validate_contribution_variable(self, var: str) -> None:
        """Validate that the variable ends with "_contribution" and is in the model."""
        if not (var.endswith("_contribution") or var == "y"):
            raise ValueError(f"Variable {var} must end with '_contribution' or be 'y'")

        if var not in self.model.named_vars:
            raise ValueError(f"Variable {var} is not in the model")

    def add_original_scale_contribution_variable(self, var: list[str]) -> None:
        """Add a pm.Deterministic variable to the model that multiplies by the scaler.

        Restricted to the model parameters. Only make it possible for "_contribution" variables.

        Parameters
        ----------
        var : list[str]
            The variables to add the original scale contribution variable.

        Examples
        --------
        >>> model.add_original_scale_contribution_variable(
        >>>     var=["channel_contribution", "total_media_contribution", "y"]
        >>> )
        """
        self._validate_model_was_built()
        target_dims = self.scalers._target.dims
        with self.model:
            for v in var:
                self._validate_contribution_variable(v)
                var_dims = self.model.named_vars_to_dims[v]
                mmm_dims_order = ("date", *self.dims)

                if v == "channel_contribution":
                    mmm_dims_order += ("channel",)
                elif v == "control_contribution":
                    mmm_dims_order += ("control",)

                deterministic_dims = tuple(
                    [
                        dim
                        for dim in mmm_dims_order
                        if dim in set(target_dims).union(var_dims)
                    ]
                )
                dim_handler = create_dim_handler(deterministic_dims)

                pm.Deterministic(
                    name=v + "_original_scale",
                    var=dim_handler(self.model[v], var_dims)
                    * dim_handler(
                        self.model["target_scale"],
                        target_dims,
                    ),
                    dims=deterministic_dims,
                )

    def build_model(
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

            from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
            from pymc_marketing.mmm.multidimensional import MMM
            from pymc_marketing.prior import Prior

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
        self._generate_and_preprocess_model_data(
            X=X,  # type: ignore
            y=y,  # type: ignore
        )
        # Compute and save scales
        self._compute_scales()

        with pm.Model(
            coords=self.model_coords,
        ) as self.model:
            _channel_scale = pm.Data(
                "channel_scale",
                self.scalers._channel.values,
                dims=self.scalers._channel.dims,
            )
            _target_scale = pm.Data(
                "target_scale",
                self.scalers._target,
                dims=self.scalers._target.dims,
            )

            _channel_data = pm.Data(
                name="channel_data",
                value=self.xarray_dataset._channel.transpose(
                    "date", *self.dims, "channel"
                ).values,
                dims=("date", *self.dims, "channel"),
            )

            _target = pm.Data(
                name="target_data",
                value=(
                    self.xarray_dataset._target.transpose("date", *self.dims).values
                ),
                dims=("date", *self.dims),
            )

            # Scale `channel_data` and `target`
            channel_dim_handler = create_dim_handler(("date", *self.dims, "channel"))
            channel_data_ = _channel_data / channel_dim_handler(
                _channel_scale,
                self.scalers._channel.dims,
            )
            channel_data_ = pt.switch(pt.isnan(channel_data_), 0.0, channel_data_)
            channel_data_.name = "channel_data_scaled"
            channel_data_.dims = ("date", *self.dims, "channel")

            target_dim_handler = create_dim_handler(("date", *self.dims))

            target_data_scaled = _target / target_dim_handler(
                _target_scale, self.scalers._target.dims
            )
            target_data_scaled.name = "target_scaled"
            target_data_scaled.dims = ("date", *self.dims)
            ## TODO: Find a better way to save it or access it in the pytensor graph.
            self.target_data_scaled = target_data_scaled

            for mu_effect in self.mu_effects:
                mu_effect.create_data(self)

            if self.time_varying_intercept | self.time_varying_media:
                time_index = pm.Data(
                    name="time_index",
                    value=self._time_index,
                    dims="date",
                )

            # Add intercept logic
            if self.time_varying_intercept:
                intercept_baseline = self.model_config["intercept"].create_variable(
                    "intercept_baseline"
                )

                intercept_latent_process = SoftPlusHSGP.parameterize_from_data(
                    X=time_index,  # this is
                    dims=("date", *self.dims),
                    **self.model_config["intercept_tvp_config"],
                ).create_variable("intercept_latent_process")

                intercept = pm.Deterministic(
                    name="intercept_contribution",
                    var=intercept_baseline[None, ...] * intercept_latent_process,
                    dims=("date", *self.dims),
                )
            else:
                intercept = self.model_config["intercept"].create_variable(
                    name="intercept_contribution"
                )

            # Add media logic
            if self.time_varying_media:
                baseline_channel_contribution = pm.Deterministic(
                    name="baseline_channel_contribution",
                    var=self.forward_pass(
                        x=channel_data_, dims=(*self.dims, "channel")
                    ),
                    dims=("date", *self.dims, "channel"),
                )

                media_latent_process = SoftPlusHSGP.parameterize_from_data(
                    X=time_index,
                    dims=("date", *self.dims),
                    **self.model_config["media_tvp_config"],
                ).create_variable("media_latent_process")

                channel_contribution = pm.Deterministic(
                    name="channel_contribution",
                    var=baseline_channel_contribution * media_latent_process[..., None],
                    dims=("date", *self.dims, "channel"),
                )
            else:
                channel_contribution = pm.Deterministic(
                    name="channel_contribution",
                    var=self.forward_pass(
                        x=channel_data_, dims=(*self.dims, "channel")
                    ),
                    dims=("date", *self.dims, "channel"),
                )

            dim_handler = create_dim_handler(("date", *self.dims))
            pm.Deterministic(
                name="total_media_contribution_original_scale",
                var=(
                    channel_contribution.sum(axis=-1)
                    * dim_handler(_target_scale, self.scalers._target.dims)
                ).sum(),
                dims=(),
            )

            # Add other contributions and likelihood
            mu_var = intercept + channel_contribution.sum(axis=-1)

            if self.control_columns is not None and len(self.control_columns) > 0:
                gamma_control = self.model_config["gamma_control"].create_variable(
                    name="gamma_control"
                )

                control_data_ = pm.Data(
                    name="control_data",
                    value=self.xarray_dataset._control.transpose(
                        "date", *self.dims, "control"
                    ).values,
                    dims=("date", *self.dims, "control"),
                )

                control_contribution = pm.Deterministic(
                    name="control_contribution",
                    var=control_data_ * gamma_control,
                    dims=("date", *self.dims, "control"),
                )

                mu_var += control_contribution.sum(axis=-1)

            if self.yearly_seasonality is not None:
                dayofyear = pm.Data(
                    name="dayofyear",
                    value=pd.to_datetime(
                        self.model_coords["date"]
                    ).dayofyear.to_numpy(),
                    dims="date",
                )

                def create_deterministic(x: pt.TensorVariable) -> None:
                    pm.Deterministic(
                        "fourier_contribution",
                        x,
                        dims=("date", *self.yearly_fourier.prior.dims),
                    )

                yearly_seasonality_contribution = pm.Deterministic(
                    name="yearly_seasonality_contribution",
                    var=self.yearly_fourier.apply(
                        dayofyear, result_callback=create_deterministic
                    ),
                    dims=("date", *self.dims),
                )
                mu_var += yearly_seasonality_contribution

            for mu_effect in self.mu_effects:
                mu_var += mu_effect.create_effect(self)

            mu_var.name = "mu"
            mu_var.dims = ("date", *self.dims)

            self.model_config["likelihood"].dims = ("date", *self.dims)
            self.model_config["likelihood"].create_likelihood_variable(
                name=self.output_var,
                mu=mu_var,
                observed=target_data_scaled,
            )

    def _validate_date_overlap_with_include_last_observations(
        self, X: pd.DataFrame, include_last_observations: bool
    ) -> None:
        """Validate that include_last_observations is not used with overlapping dates.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for prediction.
        include_last_observations : bool
            Whether to include the last observations of the training data.

        Raises
        ------
        ValueError
            If include_last_observations=True and input dates overlap with training dates.
        """
        if not include_last_observations:
            return

        # Get training dates and input dates
        training_dates = pd.to_datetime(self.model_coords["date"])
        input_dates = pd.to_datetime(X[self.date_column].unique())

        # Check for overlap
        overlapping_dates = set(training_dates).intersection(set(input_dates))

        if overlapping_dates:
            overlapping_dates_str = ", ".join(
                sorted([str(d.date()) for d in overlapping_dates])
            )
            raise ValueError(
                f"Cannot use include_last_observations=True when input dates overlap with training dates. "
                f"Overlapping dates found: {overlapping_dates_str}. "
                f"Either set include_last_observations=False or use input dates that don't overlap with training data."
            )

    def _posterior_predictive_data_transformation(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        include_last_observations: bool = False,
    ) -> xr.Dataset:
        """Transform the data for posterior predictive sampling.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for prediction.
        y : pd.Series, optional
            The target data for prediction.
        include_last_observations : bool, optional
            Whether to include the last observations of the training data for continuity.

        Returns
        -------
        xr.Dataset
            The transformed data in xarray format.
        """
        # Validate that include_last_observations is not used with overlapping dates
        self._validate_date_overlap_with_include_last_observations(
            X, include_last_observations
        )

        dataarrays = []
        if include_last_observations:
            last_obs = self.xarray_dataset.isel(date=slice(-self.adstock.l_max, None))
            dataarrays.append(last_obs)

        # Transform X and y_pred to xarray
        X_xarray = self._create_xarray_from_pandas(
            data=X,
            date_column=self.date_column,
            dims=self.dims,
            metric_list=self.channel_columns,
            metric_coordinate_name="channel",
        ).transpose("date", *self.dims, "channel")
        dataarrays.append(X_xarray)

        if self.control_columns is not None:
            control_dataarray = self._create_xarray_from_pandas(
                data=X,
                date_column=self.date_column,
                dims=self.dims,
                metric_list=self.control_columns,
                metric_coordinate_name="control",
            ).transpose("date", *self.dims, "control")
            dataarrays.append(control_dataarray)

        if y is not None:
            y_xarray = (
                self._create_xarray_from_pandas(
                    data=y,
                    date_column=self.date_column,
                    dims=self.dims,
                    metric_list=[self.target_column],
                    metric_coordinate_name="target",
                )
                .sum("target")
                .transpose("date", *self.dims)
            )
        else:
            # Return empty xarray with same dimensions as the target but full of zeros
            # Use the same dtype as the existing target data to avoid dtype mismatches
            target_dtype = self.xarray_dataset._target.dtype
            y_xarray = xr.DataArray(
                np.zeros(
                    (
                        X[self.date_column].nunique(),
                        *[len(self.xarray_dataset.coords[dim]) for dim in self.dims],
                    ),
                    dtype=target_dtype,
                ),
                dims=("date", *self.dims),
                coords={
                    "date": X[self.date_column].unique(),
                    **{dim: self.xarray_dataset.coords[dim] for dim in self.dims},
                },
                name="_target",
            ).to_dataset()

        dataarrays.append(y_xarray)
        self.dataarrays = dataarrays
        self._new_internal_xarray = xr.merge(dataarrays).fillna(0)

        return xr.merge(dataarrays).fillna(0)

    def _set_xarray_data(
        self,
        dataset_xarray: xr.Dataset,
        clone_model: bool = True,
    ) -> pm.Model:
        """Set xarray data into the model.

        Parameters
        ----------
        dataset_xarray : xr.Dataset
            Input data for channels and other variables.
        clone_model : bool, optional
            Whether to clone the model. Defaults to True.

        Returns
        -------
        None
        """
        model = cm(self.model) if clone_model else self.model

        # Get channel data and handle dtype conversion
        channel_values = dataset_xarray._channel.transpose(
            "date", *self.dims, "channel"
        )
        if "channel_data" in model.named_vars:
            original_dtype = model.named_vars["channel_data"].type.dtype
            channel_values = channel_values.astype(original_dtype)

        data = {"channel_data": channel_values}
        coords = self.model.coords.copy()
        coords["date"] = dataset_xarray["date"].to_numpy()

        if "_control" in dataset_xarray:
            control_values = dataset_xarray["_control"].transpose(
                "date", *self.dims, "control"
            )
            if "control_data" in model.named_vars:
                original_dtype = model.named_vars["control_data"].type.dtype
                control_values = control_values.astype(original_dtype)
            data["control_data"] = control_values
            coords["control"] = dataset_xarray["control"].to_numpy()
        if self.yearly_seasonality is not None:
            data["dayofyear"] = dataset_xarray["date"].dt.dayofyear.to_numpy()

        if self.time_varying_intercept or self.time_varying_media:
            data["time_index"] = infer_time_index(
                pd.Series(dataset_xarray["date"]),
                pd.Series(self.model_coords["date"]),
                self._time_resolution,
            )

        if "_target" in dataset_xarray:
            target_values = dataset_xarray._target.transpose("date", *self.dims)
            # Get the original dtype from the model's shared variable
            if "target_data" in model.named_vars:
                original_dtype = model.named_vars["target_data"].type.dtype
                # Convert to the original dtype to avoid precision loss errors
                data["target_data"] = target_values.astype(original_dtype)
            else:
                data["target_data"] = target_values

        self.new_updated_data = data
        self.new_updated_coords = coords
        self.new_updated_model = model

        with model:
            pm.set_data(data, coords=coords)

        return model

    def sample_posterior_predictive(
        self,
        X: pd.DataFrame | None = None,  # type: ignore
        extend_idata: bool = True,  # type: ignore
        combined: bool = True,  # type: ignore
        include_last_observations: bool = False,  # type: ignore
        clone_model: bool = True,  # type: ignore
        **sample_posterior_predictive_kwargs,  # type: ignore
    ) -> xr.DataArray:
        """Sample from the model's posterior predictive distribution.

        Parameters
        ----------
        X : pd.DataFrame
            Input data for prediction, with the same structure as the training data.
        y : pd.Series, optional
            Optional target data for validation or alignment. Default is None.
        extend_idata : bool, optional
            Whether to add predictions to the inference data object. Defaults to True.
        combined : bool, optional
            Combine chain and draw dimensions into a single sample dimension. Defaults to True.
        include_last_observations : bool, optional
            Whether to include the last observations of the training data for continuity
            (useful for adstock transformations). Defaults to False.
        clone_model : bool, optional
            Whether to clone the model. Defaults to True.
        **sample_posterior_predictive_kwargs
            Additional arguments for `pm.sample_posterior_predictive`.

        Returns
        -------
        xr.DataArray
            Posterior predictive samples.
        """
        X = _handle_deprecate_pred_argument(X, "X", sample_posterior_predictive_kwargs)
        # Update model data with xarray
        if X is None:
            raise ValueError("X values must be provided")
        dataset_xarray = self._posterior_predictive_data_transformation(
            X=X,
            include_last_observations=include_last_observations,
        )
        model = self._set_xarray_data(
            dataset_xarray=dataset_xarray,
            clone_model=clone_model,
        )

        for mu_effect in self.mu_effects:
            mu_effect.set_data(self, model, dataset_xarray)

        with model:
            # Sample from posterior predictive
            post_pred = pm.sample_posterior_predictive(
                self.idata, **sample_posterior_predictive_kwargs
            )

            if extend_idata:
                self.idata.extend(post_pred, join="right")  # type: ignore

        group = "posterior_predictive"
        posterior_predictive_samples = az.extract(post_pred, group, combined=combined)

        if include_last_observations:
            # Remove extra observations used for adstock continuity
            posterior_predictive_samples = posterior_predictive_samples.isel(
                date=slice(self.adstock.l_max, None)
            )

        return posterior_predictive_samples

    def _make_channel_transform(
        self, df_lift_test: pd.DataFrame
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a function for transforming the channel data into the same scale as in the model.

        Parameters
        ----------
        df_lift_test : pd.DataFrame
            Lift test measurements.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            The function for scaling the channel data.
        """
        # The transformer will be passed a np.ndarray of data corresponding to this index.
        index_cols = [*list(self.dims), "channel"]
        # We reconstruct the input dataframe following the transformations performed within
        # `lift_test.scale_channel_lift_measurements()``.
        input_df = (
            df_lift_test.loc[:, [*index_cols, "x", "delta_x"]]
            .set_index(index_cols, append=True)
            .stack()
            .unstack(level=-2)
            .reindex(self.channel_columns, axis=1)  # type: ignore
            .fillna(0)
        )

        def channel_transform(input: np.ndarray) -> np.ndarray:
            """Transform lift test channel data to the same scale as in the model."""
            # reconstruct the df corresponding to the input np.ndarray.
            reconstructed = (
                pd.DataFrame(data=input, index=input_df.index, columns=input_df.columns)
                .stack()
                .unstack(level=-2)
            )
            return (
                (
                    # Scale the data according to the scaler coords.
                    reconstructed.to_xarray() / self.scalers._channel
                )
                .to_dataframe()
                .fillna(0)
                .stack()
                .unstack(level=-2)
                .loc[input_df.index, :]
                .values
            )

        # Finally return the scaled data as a np.ndarray corresponding to the input index order.
        return channel_transform

    def _make_target_transform(
        self, df_lift_test: pd.DataFrame
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a function for transforming the target measurements into the same scale as in the model.

        Parameters
        ----------
        df_lift_test : pd.DataFrame
            Lift test measurements.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            The function for scaling the target data.
        """
        # These are the same order as in the original lift test measurements.
        index_cols = [*list(self.dims), "channel"]
        input_idx = df_lift_test.set_index(index_cols, append=True).index

        def target_transform(input: np.ndarray) -> np.ndarray:
            """Transform lift test measurements and sigma to the same scale as in the model."""
            # Reconstruct the input df column with the correct index.
            reconstructed = pd.DataFrame(
                data=input, index=input_idx, columns=["target"]
            )
            return (
                (
                    # Scale the measurements.
                    reconstructed.to_xarray() / self.scalers._target
                )
                .to_dataframe()
                .loc[input_idx, :]
                .values
            )

        # Finally, return the scaled measurements as a np.ndarray corresponding to
        # the input index order.
        return target_transform

    def add_lift_test_measurements(
        self,
        df_lift_test: pd.DataFrame,
        dist: type[pm.Distribution] = pm.Gamma,
        name: str = "lift_measurements",
    ) -> None:
        """Add lift tests to the model.

        The model for the difference of a channel's saturation curve is created
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
                * `DIM_NAME`: dimension name. One column per dimension in `mmm.dims`.
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
            If the 'channel' column or any of the model dimensions is not present
            in df_lift_test.

        Examples
        --------
        Build the model first then add lift test measurements.

        .. code-block:: python

            import pandas as pd
            import numpy as np

            from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation

            from pymc_marketing.mmm.multidimensional import MMM

            model = MMM(
                date_column="date",
                channel_columns=["x1", "x2"],
                target_column="target",
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                yearly_seasonality=2,
                dims=("geo",),
            )

            X = pd.DataFrame(
                {
                    "date": np.tile(
                        pd.date_range(start="2025-01-01", end="2025-05-01", freq="W"), 2
                    ),
                    "x1": np.random.rand(34),
                    "x2": np.random.rand(34),
                    "target": np.random.rand(34),
                    "geo": 17 * ["FIN"] + 17 * ["SWE"],
                }
            )
            y = X["target"]

            model.build_model(X.drop(columns=["target"]), y)

            df_lift_test = pd.DataFrame(
                {
                    "channel": ["x1", "x1"],
                    "geo": ["FIN", "SWE"],
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

        for dim in self.dims:
            if dim not in df_lift_test.columns:
                raise KeyError(
                    f"The {dim} column is required to map the lift measurements to the model."
                )

        # Function to scale "delta_y", and "sigma" to same scale as target in model.
        target_transform = self._make_target_transform(df_lift_test)

        # Function to scale "x" and "delta_x" to the same scale as their respective channels.
        channel_transform = self._make_channel_transform(df_lift_test)

        df_lift_test_scaled = scale_lift_measurements(
            df_lift_test=df_lift_test,
            channel_col="channel",
            channel_columns=self.channel_columns,  # type: ignore
            channel_transform=channel_transform,
            target_transform=target_transform,
            dim_cols=list(self.dims),
        )
        # This is coupled with the name of the
        # latent process Deterministic
        time_varying_var_name = (
            "media_latent_process" if self.time_varying_media else None
        )
        add_lift_measurements_to_likelihood_from_saturation(
            df_lift_test=df_lift_test_scaled,
            saturation=self.saturation,
            time_varying_var_name=time_varying_var_name,
            model=self.model,
            dist=dist,
            name=name,
        )


def create_sample_kwargs(
    sampler_config: dict[str, Any] | None,
    progressbar: bool | None,
    random_seed: RandomState | None,
    **kwargs,
) -> dict[str, Any]:
    """Create the dictionary of keyword arguments for `pm.sample`.

    Parameters
    ----------
    sampler_config : dict | None
        The configuration dictionary for the sampler. If None, defaults to an empty dict.
    progressbar : bool, optional
        Whether to show the progress bar during sampling. Defaults to True.
    random_seed : RandomState, optional
        The random seed for the sampler.
    **kwargs : Any
        Additional keyword arguments to pass to the sampler.

    Returns
    -------
    dict
        The dictionary of keyword arguments for `pm.sample`.
    """
    # Ensure sampler_config is a dictionary
    sampler_config = sampler_config.copy() if sampler_config is not None else {}

    # Handle progress bar configuration
    sampler_config["progressbar"] = (
        progressbar
        if progressbar is not None
        else sampler_config.get("progressbar", True)
    )

    # Add random seed if provided
    if random_seed is not None:
        sampler_config["random_seed"] = random_seed

    # Update with additional keyword arguments
    sampler_config.update(kwargs)
    return sampler_config


class MultiDimensionalBudgetOptimizerWrapper(OptimizerCompatibleModelWrapper):
    """Wrapper for the BudgetOptimizer to handle multi-dimensional model."""

    def __init__(self, model: MMM, start_date: str, end_date: str):
        self.model_class = model
        self.start_date = start_date
        self.end_date = end_date
        # Compute the number of periods to allocate budget for
        self.zero_data = create_zero_dataset(
            model=self.model_class, start_date=start_date, end_date=end_date
        )
        self.num_periods = len(self.zero_data[self.model_class.date_column].unique())
        # Adding missing dependencies for compatibility with BudgetOptimizer
        self._channel_scales = 1.0

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped MMM model."""
        try:
            # First, try to get the attribute from the wrapper itself
            return object.__getattribute__(self, name)
        except AttributeError:
            # If not found, delegate to the wrapped model
            try:
                return getattr(self.model_class, name)
            except AttributeError as e:
                # Raise an AttributeError if the attribute is not found in either
                raise AttributeError(
                    f"'{type(self).__name__}' object and its wrapped 'MMM' object have no attribute '{name}'"
                ) from e

    def _set_predictors_for_optimization(self, num_periods: int) -> pm.Model:
        """Return the respective PyMC model with any predictors set for optimization."""
        # Use the model's method for transformation
        dataset_xarray = self._posterior_predictive_data_transformation(
            X=self.zero_data,
            include_last_observations=False,
        )

        # Use the model's method to set data
        pymc_model = self._set_xarray_data(
            dataset_xarray=dataset_xarray,
            clone_model=True,  # Ensure we work on a clone
        )

        # Use the model's mu_effects and set data using the model instance
        for mu_effect in self.mu_effects:
            mu_effect.set_data(self, pymc_model, dataset_xarray)

        return pymc_model

    def optimize_budget(
        self,
        budget: float | int,
        budget_bounds: xr.DataArray | None = None,
        response_variable: str = "total_media_contribution_original_scale",
        utility_function: UtilityFunctionType = average_response,
        constraints: Sequence[dict[str, Any]] = (),
        default_constraints: bool = True,
        budgets_to_optimize: xr.DataArray | None = None,
        **minimize_kwargs,
    ) -> tuple[xr.DataArray, OptimizeResult]:
        """Optimize the budget allocation for the model."""
        from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

        allocator = BudgetOptimizer(
            num_periods=self.num_periods,
            utility_function=utility_function,
            response_variable=response_variable,
            custom_constraints=constraints,
            default_constraints=default_constraints,
            budgets_to_optimize=budgets_to_optimize,
            model=self,  # Pass the wrapper instance itself to the BudgetOptimizer
        )

        return allocator.allocate_budget(
            total_budget=budget,
            budget_bounds=budget_bounds,
            **minimize_kwargs,
        )

    def sample_response_distribution(
        self,
        allocation_strategy: xr.DataArray,
        noise_level: float = 0.001,
    ) -> az.InferenceData:
        """Generate synthetic dataset and sample posterior predictive based on allocation.

        Parameters
        ----------
        allocation_strategy : DataArray
            The allocation strategy for the channels.
        noise_level : float
            The relative level of noise to add to the data allocation.

        Returns
        -------
        az.InferenceData
            The posterior predictive samples based on the synthetic dataset.
        """
        data = create_zero_dataset(
            model=self,
            start_date=self.start_date,
            end_date=self.end_date,
            channel_xr=allocation_strategy.to_dataset(dim="channel"),
        )

        data_with_noise = add_noise_to_channel_allocation(
            df=data,
            channels=self.channel_columns,
            rel_std=noise_level,
            seed=42,
        )

        constant_data = allocation_strategy.to_dataset(name="allocation")

        return self.sample_posterior_predictive(
            X=data_with_noise,
            extend_idata=False,
            include_last_observations=True,
            var_names=["y", "channel_contribution_original_scale"],
            progressbar=False,
        ).merge(constant_data)
