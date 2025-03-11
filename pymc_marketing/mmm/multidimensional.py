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
from typing import Any, Literal, Protocol

import arviz as az
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc.model.fgraph import clone_model as cm
from pymc.util import RandomState

from pymc_marketing.mmm import SoftPlusHSGP
from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
    adstock_from_dict,
)
from pymc_marketing.mmm.components.saturation import (
    SaturationTransformation,
    saturation_from_dict,
)
from pymc_marketing.mmm.events import EventEffect, days_from_reference
from pymc_marketing.mmm.fourier import YearlyFourier
from pymc_marketing.mmm.plot import MMMPlotSuite
from pymc_marketing.mmm.tvp import infer_time_index
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


class MuEffect(Protocol):
    """Protocol for arbitrary additive mu effect."""

    def create_data(self, mmm: MMM) -> None:
        """Create the required data in the model."""

    def create_effect(self, mmm: MMM) -> pt.TensorVariable:
        """Create the additive effect in the model."""

    def set_data(self, mmm: MMM, model: pm.Model, X: xr.Dataset) -> None:
        """Set the data for new predictions."""


def create_event_mu_effect(
    df_events: pd.DataFrame,
    prefix: str,
    effect: EventEffect,
) -> MuEffect:
    """Create an event effect for the MMM.

    This class has the ability to create data and mean effects for the MMM model.

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

    Returns
    -------
    MuEffect
        The event effect which is used in the MMM.

    """
    if missing_columns := set(["start_date", "end_date", "name"]).difference(
        df_events.columns,
    ):
        raise ValueError(f"Columns {missing_columns} are missing in df_events.")

    effect.basis.prefix = prefix

    reference_date = "2025-01-01"
    start_dates = pd.to_datetime(df_events["start_date"])
    end_dates = pd.to_datetime(df_events["end_date"])

    class Effect:
        """Event effect class for the MMM."""

        def create_data(self, mmm: MMM) -> None:
            """Create the required data in the model.

            Parameters
            ----------
            mmm : MMM
                The MMM model instance.

            """
            model: pm.Model = mmm.model

            model_dates = pd.to_datetime(model.coords["date"])

            model.add_coord(prefix, df_events["name"].to_numpy())

            if "days" not in model:
                pm.Data(
                    "days",
                    days_from_reference(model_dates, reference_date),
                    dims="date",
                )

            pm.Data(
                f"{prefix}_start_diff",
                days_from_reference(start_dates, reference_date),
                dims=prefix,
            )
            pm.Data(
                f"{prefix}_end_diff",
                days_from_reference(end_dates, reference_date),
                dims=prefix,
            )

        def create_effect(self, mmm: MMM) -> pt.TensorVariable:
            """Create the event effect in the model.

            Parameters
            ----------
            mmm : MMM
                The MMM model instance.

            Returns
            -------
            pt.TensorVariable
                The average event effect in the model.

            """
            model: pm.Model = mmm.model

            s_ref = model["days"][:, None] - model[f"{prefix}_start_diff"]
            e_ref = model["days"][:, None] - model[f"{prefix}_end_diff"]

            def create_basis_matrix(s_ref, e_ref):
                return pt.where(
                    (s_ref >= 0) & (e_ref <= 0),
                    0,
                    pt.where(pt.abs(s_ref) < pt.abs(e_ref), s_ref, e_ref),
                )

            X = create_basis_matrix(s_ref, e_ref)
            event_effect = effect.apply(X, name=prefix)

            total_effect = pm.Deterministic(
                f"{prefix}_total_effect",
                event_effect.sum(axis=1),
                dims="date",
            )

            dim_handler = create_dim_handler(("date", *mmm.dims))
            return dim_handler(total_effect, "date")

        def set_data(self, mmm: MMM, model: pm.Model, X: xr.Dataset) -> None:
            """Set the data for new predictions."""
            new_dates = pd.to_datetime(model.coords["date"])

            new_data = {
                "days": days_from_reference(new_dates, reference_date),
            }
            pm.set_data(new_data=new_data, model=model)

    return Effect()


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

    def __init__(
        self,
        date_column: str,
        channel_columns: list[str],
        target_column: str,
        adstock: AdstockTransformation,
        saturation: SaturationTransformation,
        time_varying_intercept: bool = False,
        time_varying_media: bool = False,
        dims: tuple | None = None,
        model_config: dict | None = None,  # Ensure model_config is a dictionary
        sampler_config: dict | None = None,
        control_columns: list[str] | None = None,
        yearly_seasonality: int | None = None,
        adstock_first: bool = True,
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

        model_config = model_config if model_config is not None else {}
        sampler_config = sampler_config
        model_config = parse_model_config(
            model_config,  # type: ignore
            hsgp_kwargs_fields=["intercept_tvp_config", "media_tvp_config"],
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

        event_effect = create_event_mu_effect(df_events, prefix, effect)
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

        # Convert to xarray
        if valid_dims:
            return df_long.set_index(
                [date_column, *valid_dims, metric_coordinate_name]
            ).to_xarray()
        return df_long.set_index([date_column, metric_coordinate_name]).to_xarray()

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

        # Convert to xarray
        if valid_dims:
            return df_long.set_index(
                [date_column, *valid_dims, metric_coordinate_name]
            ).to_xarray()
        return df_long.set_index([date_column, metric_coordinate_name]).to_xarray()

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

        y_dataarray = self._create_xarray_from_pandas(
            data=pd.concat([self.X, self.y], axis=1).set_index(
                [self.date_column, *self.dims]
            )[self.target_column],
            date_column=self.date_column,
            dims=self.dims,
            metric_list=[self.target_column],
            metric_coordinate_name="target",
        )
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
            date_column="date",
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
        self.scalers = self.xarray_dataset.max(dim=["date", *self.dims])

    def get_scales_as_xarray(self) -> dict[str, xr.DataArray]:
        """Return the saved scaling factors as xarray DataArrays.

        Returns
        -------
        dict[str, xr.DataArray]
            A dictionary containing the scaling factors for channels and target.

        Examples
        --------
        >>> mmm = MMM(
            date_column="date",
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
        if not var.endswith("_contribution"):
            raise ValueError(f"Variable {var} must end with '_contribution'")

        if var not in self.model.named_vars:
            raise ValueError(f"Variable {var} is not in the model")

    def add_original_scale_contribution_variable(self, var: list[str]) -> None:
        """Add a pm.Deterministic variable to the model that multiplies by the scaler.

        Restricted to the model parameters. Only make it possible for "_contirbution" variables.

        Parameters
        ----------
        var : list[str]
            The variables to add the original scale contribution variable.

        Examples
        --------
        >>> model.add_original_scale_contribution_variable(
        >>>     var=["channel_contribution", "total_media_contribution", "likelihood"]
        >>> )
        """
        self._validate_model_was_built()
        with self.model:
            for v in var:
                self._validate_contribution_variable(v)
                pm.Deterministic(
                    name=v + "_original_scale",
                    var=self.model[v] * self.model["target_scale"],
                    dims=self.model.named_vars_to_dims[v],
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
                dims="channel",
            )
            _target_scale = pm.Data(
                "target_scale",
                self.scalers._target.item(),
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
                    self.xarray_dataset._target.sum(dim="target")
                    .transpose("date", *self.dims)
                    .values
                ),
                dims=("date", *self.dims),
            )

            # Scale `channel_data` and `target`
            channel_data_ = _channel_data / _channel_scale
            channel_data_.name = "channel_data_scaled"
            channel_data_.dims = ("date", *self.dims, "channel")

            ## Hot fix for target data meanwhile pymc allows for internal scaling `https://github.com/pymc-devs/pymc/pull/7656`
            target_data_scaled = _target / _target_scale
            target_data_scaled.name = "target_scaled"
            target_data_scaled.dims = ("date", *self.dims)

            target_data_ = pm.Data(
                name="target",
                value=target_data_scaled.eval(),
                dims=("date", *self.dims),
            )

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
                baseline_intercept = self.model_config["intercept"].create_variable(
                    "baseline_intercept"
                )

                intercept_latent_process = SoftPlusHSGP.parameterize_from_data(
                    X=time_index,  # this is
                    dims=("date", *self.dims),
                    **self.model_config["intercept_tvp_config"],
                ).create_variable("intercept_latent_process")

                intercept = pm.Deterministic(
                    name="intercept_contribution",
                    var=baseline_intercept[None, ...] * intercept_latent_process,
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

            pm.Deterministic(
                name="total_media_contribution_original_scale",
                var=channel_contribution.sum() * _target_scale,
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
                observed=target_data_,
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
            y_xarray = self._create_xarray_from_pandas(
                data=y,
                date_column=self.date_column,
                dims=self.dims,
                metric_list=[self.target_column],
                metric_coordinate_name="target",
            ).transpose("date", *self.dims, "target")
        else:
            # Return empty xarray with same dimensions as the target but full of zeros
            y_xarray = xr.DataArray(
                np.zeros(
                    (
                        X[self.date_column].nunique(),
                        *[len(self.xarray_dataset.coords[dim]) for dim in self.dims],
                        1,
                    ),
                    dtype=np.int32,
                ),
                dims=("date", *self.dims, "target"),
                coords={
                    "date": X[self.date_column].unique(),
                    **{dim: self.xarray_dataset.coords[dim] for dim in self.dims},
                    "target": self.xarray_dataset.coords["target"],
                },
                name="_target",
            ).to_dataset()

        dataarrays.append(y_xarray)
        self.dataarrays = dataarrays
        self._new_internal_xarray = xr.merge(dataarrays).fillna(0)

        return xr.merge(dataarrays).fillna(0).astype(np.int32)

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

        Returns
        -------
        None
        """
        model = cm(self.model) if clone_model else self.model

        data = {
            "channel_data": dataset_xarray._channel.transpose(
                "date", *self.dims, "channel"
            )
        }
        coords = {"date": dataset_xarray["date"].to_numpy()}

        if "control_data" in dataset_xarray:
            data["control_data"] = dataset_xarray["control_data"].transpose(
                "date", *self.dims, "control"
            )

        if self.yearly_seasonality is not None:
            data["dayofyear"] = dataset_xarray["date"].dt.dayofyear.to_numpy()

        if self.time_varying_intercept or self.time_varying_media:
            data["time_index"] = infer_time_index(
                pd.Series(dataset_xarray[self.date_column]),
                pd.Series(self.model_coords["date"]),
                self._time_resolution,
            )

        if "target" in dataset_xarray:
            data["target"] = dataset_xarray._target.sum(dim="target").transpose(
                "date", *self.dims
            )

            data["target_data"] = dataset_xarray._target.sum(dim="target").transpose(
                "date", *self.dims
            )

        self.new_updated_data = data
        self.new_updated_coords = coords
        self.new_updated_model = model

        with model:
            pm.set_data(data, coords=coords)

        return model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        progressbar: bool | None = None,
        random_seed: RandomState | None = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Fit a model using the data passed as a parameter.

        Parameters
        ----------
        X : array-like | array, shape (n_obs, n_features)
            The training input samples. If scikit-learn is available, array-like, otherwise array.
        y : array-like | array, shape (n_obs,)
            The target values (real numbers). If scikit-learn is available, array-like, otherwise array.
        progressbar : bool, optional
            Specifies whether the fit progress bar should be displayed. Defaults to True.
        random_seed : RandomState, optional
            Provides the sampler with an initial random seed for reproducible samples.
        **kwargs : dict
            Additional keyword arguments passed to the sampler.

        Returns
        -------
        az.InferenceData
            The inference data from the fitted model.

        Examples
        --------
        >>> model = MyModel()
        >>> idata = model.fit(X, y, progressbar=True)
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series")

        if not hasattr(self, "model"):
            self.build_model(
                X=X,
                y=y,  # type: ignore
            )

        # Ensure sampler_config is initialized as an empty dict if None
        self.sampler_config = self.sampler_config or {}

        sampler_kwargs = create_sample_kwargs(
            self.sampler_config,
            progressbar,
            random_seed,
            **kwargs,
        )  # type: ignore

        with self.model:
            idata = pm.sample(**sampler_kwargs)

        self.idata = idata  # type: ignore

        # (3) Add X,y to a custom group in the InferenceData
        # Combine X and y into one DataFrame then convert to xarray
        df_fit = pd.concat([X, y], axis=1)

        # To xarray:
        fit_data_xr = df_fit.to_xarray()

        # It's possible ArviZ might raise a UserWarning about "fit_data"
        # not matching a recognized group. We'll just ignore that.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )
            self.idata.add_groups({"fit_data": fit_data_xr})

        self.set_idata_attrs(self.idata)
        return self.idata  # type: ignore

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
