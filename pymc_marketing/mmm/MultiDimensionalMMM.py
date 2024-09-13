#   Copyright 2024 The PyMC Labs Developers
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

import itertools
import warnings
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc.util import RandomState

from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
)
from pymc_marketing.mmm.components.saturation import (
    SaturationTransformation,
)
from pymc_marketing.mmm.fourier import YearlyFourier
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.prior import Prior


class PlotMMMMixin:
    """Example docstring."""

    def plot_saturation_curves_scatter(self):
        """Docstring."""
        # Identify additional dimensions beyond 'date' and 'channel'
        additional_dims = [
            dim
            for dim in self.idata.constant_data.channel_data.dims
            if dim not in ("date", "channel")
        ]

        # Get all possible combinations of the additional dimensions
        if additional_dims:
            additional_coords = [
                self.idata.constant_data.coords[dim].values for dim in additional_dims
            ]
            additional_combinations = list(itertools.product(*additional_coords))
        else:
            additional_combinations = [()]

        # Determine the number of rows and columns for subplots
        n_columns = len(additional_combinations)
        n_rows = len(self.idata.constant_data.coords["channel"])

        # Create subplots
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_columns,
            figsize=(5 * n_columns, 4 * n_rows),
            squeeze=False,
        )

        # Loop over each channel (row)
        for row_idx, channel in enumerate(
            self.idata.constant_data.coords["channel"].values
        ):
            # Loop over each combination of additional dimensions (column)
            for col_idx, combo in enumerate(additional_combinations):
                # Build indexers for selecting data
                indexers = dict(zip(additional_dims, combo, strict=False))
                indexers["channel"] = channel

                # Select X data (constant_data)
                x_data = self.idata.constant_data.channel_data.sel(**indexers)

                # Select Y data (posterior contributions)
                y_data = self.idata.posterior.channel_contributions.sel(**indexers)

                # Flatten 'chain' and 'draw' dimensions into a single 'sample' dimension
                y_data = y_data.mean(dim=["chain", "draw"])

                # Ensure X and Y have matching 'date' coordinates
                x_data = x_data.broadcast_like(y_data)
                y_data = y_data.broadcast_like(x_data)

                # Select the appropriate axis
                ax = axes[row_idx][col_idx] if n_columns > 1 else axes[row_idx][0]

                # Plot the scatter plot
                ax.scatter(
                    x_data.values.flatten(),
                    y_data.values.flatten(),
                    alpha=0.8,
                    color=f"C{row_idx}",
                )

                # Set plot title and labels
                title_parts = [
                    f"{dim}={val}"
                    for dim, val in zip(
                        ["channel", *additional_dims],
                        [channel, *list(combo)],
                        strict=False,
                    )
                ]
                ax.set_title(", ".join(title_parts))
                ax.set_xlabel("Channel Data (X)")
                ax.set_ylabel("Channel Contributions (Y)")

        # Adjust layout and display the plot
        # plt.tight_layout()
        return fig, axes


class VanillaMultiDimensionalMMM(PlotMMMMixin):
    """Docstring example."""

    _model_name: str = "BaseMMM"
    _model_type: str = "BaseValidateMMM"
    version: str = "0.0.3"

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
        validate_data: bool = True,
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
        self.validate_data = validate_data

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

        self.date_column = date_column
        self.target_column = target_column
        self.channel_columns = channel_columns
        self.model_config = self.default_model_config
        self.sampler_config = sampler_config
        self.yearly_seasonality = yearly_seasonality

        if self.yearly_seasonality is not None:
            self.yearly_fourier = YearlyFourier(
                n_order=self.yearly_seasonality,
                prefix="fourier_mode",
                prior=self.model_config["gamma_fourier"],
                variable_name="gamma_fourier",
            )

    @property
    def default_model_config(self) -> dict:
        """Define the default model configuration."""
        base_config = {
            "intercept": Prior("Normal", mu=0, sigma=2, dims=self.dims),
            "likelihood": Prior(
                "Normal", sigma=Prior("HalfNormal", sigma=2), dims=self.dims
            ),
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

        for media_transform in [self.adstock, self.saturation]:
            for dist in media_transform.function_priors.values():
                if dist.dims != ("channel",):
                    dist.dims = "channel"

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

    def create_xarray_from_dataframe(
        self, df, date_column, dims, metric_list, metric_coordinate_name
    ):
        """Dosctring Example."""
        # Filter out dimensions that do not exist in the DataFrame
        valid_dims = [dim for dim in dims if dim in df.columns]

        # Ensure metric_list is present in the DataFrame
        valid_metrics = [metric for metric in metric_list if metric in df.columns]

        # Reshape the DataFrame to a long format with 'metric_coordinate_name' as a variable
        df_long = df.melt(
            id_vars=[date_column, *valid_dims],
            value_vars=valid_metrics,
            var_name=metric_coordinate_name,
            value_name="_" + metric_coordinate_name,
        )

        # Drop duplicates to avoid non-unique MultiIndex
        df_long = df_long.drop_duplicates(
            subset=[date_column, *valid_dims, metric_coordinate_name]
        )

        # Convert the long DataFrame to an xarray Dataset
        if valid_dims:
            df_xarray = df_long.set_index(
                [date_column, *valid_dims, metric_coordinate_name]
            ).to_xarray()
        else:
            df_xarray = df_long.set_index(
                [date_column, metric_coordinate_name]
            ).to_xarray()

        return df_xarray

    def _generate_and_preprocess_model_data(self, X, y):
        dataarrays = []
        X[self.date_column] = pd.to_datetime(X[self.date_column])

        X_dataarray = self.create_xarray_from_dataframe(
            df=X,
            date_column=self.date_column,
            dims=self.dims,
            metric_list=self.channel_columns,
            metric_coordinate_name="channel",
        )
        dataarrays.append(X_dataarray)

        y_dataarray = self.create_xarray_from_dataframe(
            df=y,
            date_column=self.date_column,
            dims=self.dims,
            metric_list=[self.target_column],
            metric_coordinate_name="target",
        )
        dataarrays.append(y_dataarray)

        if self.control_columns is not None:
            control_dataarray = self.create_xarray_from_dataframe(
                df=X,
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

    def forward_pass(
        self, x: pt.TensorVariable | npt.NDArray[np.float64]
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

        """
        first, second = (
            (self.adstock, self.saturation)
            if self.adstock_first
            else (self.saturation, self.adstock)
        )

        return second.apply(x=first.apply(x=x, dims="channel"), dims="channel")

    def build_model(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | Any,
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
        self._generate_and_preprocess_model_data(X, y)
        print("-init-")
        with pm.Model(
            coords=self.model_coords,
        ) as self.model:
            channel_data_ = pm.Data(
                name="channel_data",
                value=self.xarray_dataset._channel.transpose(
                    "date", *self.dims, "channel"
                ).values,
                dims=("date", *self.dims, "channel"),
            )

            target_ = pm.Data(
                name="target",
                value=self.xarray_dataset._target.sum(dim="target")
                .transpose("date", *self.dims)
                .values,
                dims=("date", *self.dims),
            )
            if self.time_varying_intercept | self.time_varying_media:
                pass
                # time_index = pm.Data(
                #     "time_index",
                #     self._time_index,
                #     dims="date",
                # )

            if self.time_varying_intercept:
                pass
                # baseline_intercept = self.model_config["intercept"].create_variable(
                #     "baseline_intercept"
                # )

                # intercept_latent_process = create_time_varying_gp_multiplier(
                #     name="intercept",
                #     dims="date",
                #     time_index=time_index,
                #     time_index_mid=self._time_index_mid,
                #     time_resolution=self._time_resolution,
                #     hsgp_kwargs=self.model_config["intercept_tvp_config"],
                # )
                # intercept = pm.Deterministic(
                #     name="intercept",
                #     var=baseline_intercept * intercept_latent_process,
                #     dims="date",
                # )
            else:
                intercept = self.model_config["intercept"].create_variable(
                    name="intercept"
                )

            if self.time_varying_media:
                # baseline_channel_contributions = pm.Deterministic(
                #     name="baseline_channel_contributions",
                #     var=self.forward_pass(x=channel_data_),
                #     dims=("date", *self.dims, "channel"),
                # )

                # media_latent_process = create_time_varying_gp_multiplier(
                #     name="media",
                #     dims="date",
                #     time_index=time_index,
                #     time_index_mid=self._time_index_mid,
                #     time_resolution=self._time_resolution,
                #     hsgp_kwargs=self.model_config["media_tvp_config"],
                # )
                # channel_contributions = pm.Deterministic(
                #     name="channel_contributions",
                #     var=baseline_channel_contributions * media_latent_process[:, None],
                #     dims=("date", *self.dims, "channel"),
                # )
                pass

            else:
                channel_contributions = pm.Deterministic(
                    name="channel_contributions",
                    var=self.forward_pass(x=channel_data_),
                    dims=("date", *self.dims, "channel"),
                )

            mu_var = intercept + channel_contributions.sum(axis=-1)

            if (
                self.control_columns is not None
                and len(self.control_columns) > 0
                and all(column in X.columns for column in self.control_columns)
            ):
                if self.model_config["gamma_control"].dims != ("control",):
                    self.model_config["gamma_control"].dims = "control"

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

                control_contributions = pm.Deterministic(
                    name="control_contributions",
                    var=control_data_ * gamma_control,
                    dims=("date", *self.dims, "control"),
                )

                mu_var += control_contributions.sum(axis=-1)

            # if self.yearly_seasonality is not None:
            # dayofyear = pm.Data(
            #     name="dayofyear",
            #     value=pd.to_datetime(
            #         self.coords_dict["date"]
            #     ).dayofyear.to_numpy(),
            #     dims="date",
            # )

            # def create_deterministic(x: pt.TensorVariable) -> None:
            #     pm.Deterministic(
            #         "fourier_contributions",
            #         x,
            #         dims=("date", *self.yearly_fourier.prior.dims),
            #     )

            # yearly_seasonality_contribution = pm.Deterministic(
            #     name="yearly_seasonality_contribution",
            #     var=self.yearly_fourier.apply(
            #         dayofyear, result_callback=create_deterministic
            #     ),
            #     dims="date",
            # )

            # mu_var += yearly_seasonality_contribution

            mu = pm.Deterministic(name="mu", var=mu_var, dims=("date", *self.dims))

            self.model_config["likelihood"].dims = ("date", *self.dims)
            self.model_config["likelihood"].create_likelihood_variable(
                name=self.output_var,
                mu=mu,
                observed=target_,
            )
            print("-end-")

    def fit(
        self,
        X: pd.DataFrame | Any,
        y: pd.DataFrame | None | Any = None,
        progressbar: bool | None = None,
        predictor_names: list[str] | None = None,
        random_seed: RandomState | None = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Fit a model using the data passed as a parameter.

        Sets attrs to inference data of the model.

        Parameters
        ----------
        X : array-like | array, shape (n_obs, n_features)
            The training input samples. If scikit-learn is available, array-like, otherwise array.
        y : array-like | array, shape (n_obs,)
            The target values (real numbers). If scikit-learn is available, array-like, otherwise array.
        progressbar : bool, optional
            Specifies whether the fit progress bar should be displayed. Defaults to True.
        predictor_names : Optional[List[str]] = None,
            Allows for custom naming of predictors when given in a form of a 2D array.
            Allows for naming of predictors when given in a form of np.ndarray, if not provided
            the predictors will be named like predictor1, predictor2...
        random_seed : Optional[RandomState]
            Provides sampler with initial random seed for obtaining reproducible samples.
        **kwargs : Any
            Custom sampler settings can be provided in form of keyword arguments.

        Returns
        -------
        self : az.InferenceData
            Returns inference data of the fitted model.

        Examples
        --------
        >>> model = MyModel()
        >>> idata = model.fit(X,y)
        Auto-assigning NUTS sampler...
        Initializing NUTS using jitter+adapt_diag...

        """
        if predictor_names is None:
            predictor_names = []

        if not hasattr(self, "model"):
            self.build_model(X, y)

        # sampler_kwargs = create_sample_kwargs(
        #     self.sampler_config,
        #     progressbar,
        #     random_seed,
        #     **kwargs,
        # )
        with self.model:
            idata = pm.sample()  # type: ignore

        if hasattr(self, "idata"):
            self.idata = self.idata.copy()  # type: ignore
            self.idata.extend(idata, join="right")  # type: ignore
        else:
            self.idata = idata  # type: ignore

        if "fit_data" in self.idata:
            del self.idata.fit_data  # type: ignore

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )
            # self.idata.add_groups(fit_data=combined_data.to_xarray())
        # self.set_idata_attrs(self.idata)

        return self.idata  # type: ignore


def create_sample_kwargs(
    sampler_config: dict[str, Any] | Any,
    progressbar: bool | None,
    random_seed,
    **kwargs,
) -> dict[str, Any]:
    """Create the dictionary of keyword arguments for `pm.sample`.

    Parameters
    ----------
    sampler_config : dict
        The configuration dictionary for the sampler.
    progressbar : bool, optional
        Whether to show the progress bar during sampling. Defaults to True.
    random_seed : RandomState
        The random seed for the sampler.
    **kwargs : Any
        Additional keyword arguments to pass to the sampler.

    Returns
    -------
    dict
        The dictionary of keyword arguments for `pm.sample`.

    """
    sampler_config = sampler_config.copy()

    if progressbar is not None:
        sampler_config["progressbar"] = progressbar
    else:
        sampler_config["progressbar"] = sampler_config.get("progressbar", True)

    if random_seed is not None:
        sampler_config["random_seed"] = random_seed

    sampler_config.update(**kwargs)

    return sampler_config
