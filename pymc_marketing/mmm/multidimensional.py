#   Copyright 2025 - 2025 The PyMC Labs Developers
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

from pymc_marketing.mmm import HSGP
from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
)
from pymc_marketing.mmm.components.saturation import (
    SaturationTransformation,
)
from pymc_marketing.mmm.fourier import YearlyFourier
from pymc_marketing.mmm.tvp import infer_time_index
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.prior import Prior


class MMMPlotSuite:
    """Media Mix Model Plot Suite.

    Provides methods for visualizing the posterior predictive distribution,
    contributions over time, and saturation curves for a Media Mix Model.
    """

    def __init__(self, idata: xr.Dataset | az.InferenceData):
        self.idata = idata

    def _init_subplots(
        self,
        n_subplots: int,
        ncols: int = 1,
        width_per_col: float = 10,
        height_per_row: float = 4,
    ):
        """Initialize a grid of subplots.

        Parameters
        ----------
        n_subplots : int
            Number of rows (if ncols=1) or total subplots.
        ncols : int
            Number of columns in the subplot grid.
        width_per_col : float
            Width (in inches) for each column of subplots.
        height_per_row : float
            Height (in inches) for each row of subplots.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created Figure object.
        axes : np.ndarray of matplotlib.axes.Axes
            2D array of axes of shape (n_subplots, ncols).
        """
        fig, axes = plt.subplots(
            nrows=n_subplots,
            ncols=ncols,
            figsize=(width_per_col * ncols, height_per_row * n_subplots),
            squeeze=False,
        )
        return fig, axes

    def _build_subplot_title(
        self,
        dims: list[str],
        combo: tuple,
        fallback_title: str = "Time Series",
    ) -> str:
        """Build a subplot title string from dimension names and their values."""
        if dims:
            title_parts = [f"{d}={v}" for d, v in zip(dims, combo, strict=False)]
            return ", ".join(title_parts)
        return fallback_title

    def _get_additional_dim_combinations(
        self,
        data: xr.Dataset,
        variable: str,
        ignored_dims: set[str],
    ) -> tuple[list[str], list[tuple]]:
        """Identify dimensions to plot over and get their coordinate combinations."""
        if variable not in data:
            raise ValueError(f"Variable '{variable}' not found in the dataset.")

        all_dims = list(data[variable].dims)
        additional_dims = [d for d in all_dims if d not in ignored_dims]

        if additional_dims:
            additional_coords = [data.coords[d].values for d in additional_dims]
            dim_combinations = list(itertools.product(*additional_coords))
        else:
            # If no extra dims, just treat as a single combination
            dim_combinations = [()]

        return additional_dims, dim_combinations

    def _reduce_and_stack(
        self, data: xr.DataArray, dims_to_ignore: set[str] | None = None
    ) -> xr.DataArray:
        """Sum over leftover dims and stack chain+draw into sample if present."""
        if dims_to_ignore is None:
            dims_to_ignore = {"date", "chain", "draw", "sample"}

        leftover_dims = [d for d in data.dims if d not in dims_to_ignore]
        if leftover_dims:
            data = data.sum(dim=leftover_dims)

        # Combine chain+draw into 'sample' if both exist
        if "chain" in data.dims and "draw" in data.dims:
            data = data.stack(sample=("chain", "draw"))

        return data

    def _compute_ci(
        self, data: xr.DataArray, ci: float = 0.85, sample_dim: str = "sample"
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Compute median and lower/upper credible intervals over given sample_dim."""
        lower_q = 0.5 - ci / 2
        upper_q = 0.5 + ci / 2
        data_median = data.quantile(0.5, dim=sample_dim)
        data_lower = data.quantile(lower_q, dim=sample_dim)
        data_upper = data.quantile(upper_q, dim=sample_dim)
        return data_median, data_lower, data_upper

    def _get_posterior_predictive_data(
        self,
        idata: xr.Dataset | None,
    ) -> xr.Dataset:
        """Retrieve the posterior_predictive group from either provided or self.idata."""
        if idata is not None:
            return idata

        # Otherwise, check if self.idata has posterior_predictive
        if (
            not hasattr(self.idata, "posterior_predictive")  # type: ignore
            or self.idata.posterior_predictive is None  # type: ignore
        ):
            raise ValueError(
                "No posterior_predictive data found in 'self.idata'. "
                "Please run 'MMM.sample_posterior_predictive()' or provide "
                "an external 'idata' argument."
            )
        return self.idata.posterior_predictive  # type: ignore

    # ------------------------------------------------------------------------
    #                          Main Plotting Methods
    # ------------------------------------------------------------------------

    def posterior_predictive(
        self, var: list[str] | None = None, idata: xr.Dataset | None = None
    ):
        """Plot time series from the posterior predictive distribution.

        By default, if both `var` and `idata` are not provided, uses
        `self.idata.posterior_predictive` and defaults the variable to `["y"]`.

        Parameters
        ----------
        var : list of str, optional
            A list of variable names to plot. Default is ["y"] if not provided.
        idata : xarray.Dataset, optional
            The posterior predictive dataset to plot. If not provided, tries to
            use `self.idata.posterior_predictive`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the subplots.
        axes : np.ndarray of matplotlib.axes.Axes
            Array of Axes objects corresponding to each subplot row.

        Raises
        ------
        ValueError
            If no `idata` is provided and `self.idata.posterior_predictive` does
            not exist, instructing the user to run `MMM.sample_posterior_predictive()`.
        """
        # 1. Retrieve or validate posterior_predictive data
        pp_data = self._get_posterior_predictive_data(idata)

        # 2. Determine variables to plot
        if var is None:
            var = ["y"]
        main_var = var[0]

        # 3. Identify additional dims & get all combos
        ignored_dims = {"chain", "draw", "date", "sample"}
        additional_dims, dim_combinations = self._get_additional_dim_combinations(
            data=pp_data, variable=main_var, ignored_dims=ignored_dims
        )

        # 4. Prepare subplots
        fig, axes = self._init_subplots(n_subplots=len(dim_combinations), ncols=1)

        # 5. Loop over dimension combinations
        for row_idx, combo in enumerate(dim_combinations):
            ax = axes[row_idx][0]

            # Build indexers
            indexers = (
                dict(zip(additional_dims, combo, strict=False))
                if additional_dims
                else {}
            )

            # 6. Plot each requested variable
            for v in var:
                if v not in pp_data:
                    raise ValueError(
                        f"Variable '{v}' not in the posterior_predictive dataset."
                    )

                data = pp_data[v].sel(**indexers)
                # Sum leftover dims, stack chain+draw if needed
                data = self._reduce_and_stack(data, ignored_dims)
                # Compute median & 85% intervals
                median, lower, upper = self._compute_ci(data, ci=0.85)

                # Extract date coordinate
                if "date" not in data.dims:
                    raise ValueError(
                        f"Expected 'date' dimension in {v}, but none found."
                    )
                dates = data.coords["date"].values

                # Plot
                ax.plot(dates, median, label=v, alpha=0.9)
                ax.fill_between(dates, lower, upper, alpha=0.2)

            # 7. Subplot title & labels
            title = self._build_subplot_title(
                dims=additional_dims,
                combo=combo,
                fallback_title="Posterior Predictive Time Series",
            )
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Posterior Predictive")
            ax.legend(loc="best")

        return fig, axes

    def contributions_over_time(self, var: list[str], ci: float = 0.85):
        """Plot the time-series contributions for each variable in `var`.

        showing the median and the credible interval (default 85%).
        Creates one subplot per combination of non-(chain/draw/date) dimensions
        and places all variables on the same subplot.

        Parameters
        ----------
        var : list of str
            A list of variable names to plot from the posterior.
        ci : float, optional
            Credible interval width. For instance, 0.85 will show the
            7.5th to 92.5th percentile range. The default is 0.85.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the subplots.
        axes : np.ndarray of matplotlib.axes.Axes
            Array of Axes objects corresponding to each subplot row.
        """
        if not hasattr(self.idata, "posterior"):
            raise ValueError(
                "No posterior data found in 'self.idata'. "
                "Please ensure 'self.idata' contains a 'posterior' group."
            )

        main_var = var[0]
        all_dims = list(self.idata.posterior[main_var].dims)  # type: ignore
        ignored_dims = {"chain", "draw", "date"}
        additional_dims = [d for d in all_dims if d not in ignored_dims]

        # Identify combos
        if additional_dims:
            additional_coords = [
                self.idata.posterior.coords[dim].values  # type: ignore
                for dim in additional_dims  # type: ignore
            ]
            dim_combinations = list(itertools.product(*additional_coords))
        else:
            dim_combinations = [()]

        # Prepare subplots
        fig, axes = self._init_subplots(len(dim_combinations), ncols=1)

        # Loop combos
        for row_idx, combo in enumerate(dim_combinations):
            ax = axes[row_idx][0]
            indexers = (
                dict(zip(additional_dims, combo, strict=False))
                if additional_dims
                else {}
            )

            # Plot each var
            for v in var:
                data = self.idata.posterior[v].sel(**indexers)  # type: ignore
                data = self._reduce_and_stack(data, {"date", "chain", "draw", "sample"})

                # Compute median and credible intervals
                median, lower, upper = self._compute_ci(data, ci=ci)

                # Extract dates
                dates = data.coords["date"].values
                ax.plot(dates, median, label=f"{v}", alpha=0.9)
                ax.fill_between(dates, lower, upper, alpha=0.2)

            title = self._build_subplot_title(
                dims=additional_dims, combo=combo, fallback_title="Time Series"
            )
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Posterior Value")
            ax.legend(loc="best")

        return fig, axes

    def saturation_curves_scatter(self):
        """Plot the saturation curves for each channel.

        Creates one subplot per combination of non-(date/channel) dimensions
        and places all channels on the same subplot.
        """
        if not hasattr(self.idata, "constant_data"):
            raise ValueError(
                "No 'constant_data' found in 'self.idata'. "
                "Please ensure 'self.idata' contains the constant_data group."
            )

        # Identify additional dimensions beyond 'date' and 'channel'
        cdims = self.idata.constant_data.channel_data.dims
        additional_dims = [dim for dim in cdims if dim not in ("date", "channel")]

        # Get all possible combinations
        if additional_dims:
            additional_coords = [
                self.idata.constant_data.coords[d].values for d in additional_dims
            ]
            additional_combinations = list(itertools.product(*additional_coords))
        else:
            additional_combinations = [()]

        # Rows = channels, Columns = additional_combinations
        channels = self.idata.constant_data.coords["channel"].values
        n_rows = len(channels)
        n_columns = len(additional_combinations)

        # Create subplots
        fig, axes = self._init_subplots(
            n_subplots=n_rows, ncols=n_columns, width_per_col=5, height_per_row=4
        )

        # Loop channels & combos
        for row_idx, channel in enumerate(channels):
            for col_idx, combo in enumerate(additional_combinations):
                ax = axes[row_idx][col_idx] if n_columns > 1 else axes[row_idx][0]
                indexers = dict(zip(additional_dims, combo, strict=False))
                indexers["channel"] = channel

                # Select X data (constant_data)
                x_data = self.idata.constant_data.channel_data.sel(**indexers)
                # Select Y data (posterior contributions)
                y_data = self.idata.posterior.channel_contribution.sel(**indexers)

                # Flatten chain & draw by taking mean (or sum, up to design)
                y_data = y_data.mean(dim=["chain", "draw"])

                # Ensure X and Y have matching date coords
                x_data = x_data.broadcast_like(y_data)
                y_data = y_data.broadcast_like(x_data)

                # Scatter
                ax.scatter(
                    x_data.values.flatten(),
                    y_data.values.flatten(),
                    alpha=0.8,
                    color=f"C{row_idx}",
                )

                title = self._build_subplot_title(
                    dims=["channel", *additional_dims],
                    combo=(channel, *combo),
                    fallback_title="Channel Saturation Curves",
                )
                ax.set_title(title)
                ax.set_xlabel("Channel Data (X)")
                ax.set_ylabel("Channel Contributions (Y)")

        return fig, axes


class MMM:
    """Media Mix Model class for estimating the impact of marketing channels on a target variable.

    This class implements the core functionality of a Media Mix Model (MMM), allowing for the
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

    _model_name: str = "BaseMMM"
    _model_type: str = "BaseValidateMMM"
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
    def plot(
        self,
    ):
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
                "Normal", sigma=Prior("HalfNormal", sigma=2), dims=self.dims
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

    def _validate_idata_exists(self) -> None:
        """Validate that the idata exists."""
        if not hasattr(self, "idata"):
            raise ValueError("idata does not exist. Build the model first and fit.")

    def _create_xarray_from_dataframe(
        self, df, date_column, dims, metric_list, metric_coordinate_name
    ):
        """Create an xarray Dataset from a DataFrame.

        This method reshapes a DataFrame into a long format and converts it into an xarray Dataset.
        It filters out dimensions that do not exist in the DataFrame and ensures that the metric_list
        is present in the DataFrame.
        """
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

        X_dataarray = self._create_xarray_from_dataframe(
            df=X,
            date_column=self.date_column,
            dims=self.dims,
            metric_list=self.channel_columns,
            metric_coordinate_name="channel",
        )
        dataarrays.append(X_dataarray)

        y_dataarray = self._create_xarray_from_dataframe(
            df=y,
            date_column=self.date_column,
            dims=self.dims,
            metric_list=[self.target_column],
            metric_coordinate_name="target",
        )
        dataarrays.append(y_dataarray)

        if self.control_columns is not None:
            control_dataarray = self._create_xarray_from_dataframe(
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
        self._generate_and_preprocess_model_data(X, y)
        # Compute and save scales
        self._compute_scales()

        with pm.Model(
            coords=self.model_coords,
        ) as self.model:
            _channel_scale = pm.Data(
                "channel_scale",
                self.scalers._channel.values,
                mutable=False,
                dims="channel",
            )
            _target_scale = pm.Data(
                "target_scale",
                self.scalers._target.item(),
                mutable=False,
            )

            _channel_data = pm.Data(
                name="channel_data",
                value=self.xarray_dataset._channel.transpose(
                    "date", *self.dims, "channel"
                ).values,
                dims=("date", *self.dims, "channel"),
            )

            _target = pm.Data(
                name="target",
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

            target_data_ = _target / _target_scale
            target_data_.name = "target_scaled"
            target_data_.dims = ("date", *self.dims)

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

                intercept_latent_process = HSGP.parameterize_from_data(
                    X=time_index,
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

                media_latent_process = HSGP.parameterize_from_data(
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

            mu_var.name = "mu"
            mu_var.dims = ("date", *self.dims)

            self.model_config["likelihood"].dims = ("date", *self.dims)
            self.model_config["likelihood"].create_likelihood_variable(
                name=self.output_var,
                mu=mu_var,
                observed=(_target.eval() / _target_scale.eval()),
            )

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
        predictor_names : list[str], optional
            Allows custom naming of predictors. If `predictor_names` is provided, predictors
            will be named accordingly; otherwise, default names will be used.
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
        if predictor_names is None:
            predictor_names = []

        if not hasattr(self, "model"):
            self.build_model(X, y)

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

        if hasattr(self, "idata"):
            self.idata = self.idata.copy()  # type: ignore
            self.idata.extend(idata, join="right")  # type: ignore
        else:
            self.idata = idata  # type: ignore

        if "fit_data" in self.idata:
            del self.idata.fit_data

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )

        return self.idata  # type: ignore

    def _posterior_predictive_data_transformation(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        include_last_observations: bool = False,
    ):
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

        # Transform X_pred and y_pred to xarray
        X_xarray = self._create_xarray_from_dataframe(
            df=X,
            date_column=self.date_column,
            dims=self.dims,
            metric_list=self.channel_columns,
            metric_coordinate_name="channel",
        ).transpose("date", *self.dims, "channel")
        dataarrays.append(X_xarray)

        if self.control_columns is not None:
            control_dataarray = self._create_xarray_from_dataframe(
                df=X,
                date_column=self.date_column,
                dims=self.dims,
                metric_list=self.control_columns,
                metric_coordinate_name="control",
            ).transpose("date", *self.dims, "control")
            dataarrays.append(control_dataarray)

        if y is not None:
            y_xarray = self._create_xarray_from_dataframe(
                df=y,
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

        return xr.merge(dataarrays).fillna(0)

    def _set_xarray_data(
        self,
        dataset_xarray: xr.Dataset,
    ) -> None:
        """Set xarray data into the model.

        Parameters
        ----------
        dataset_xarray : xr.Dataset
            Input data for channels and other variables.

        Returns
        -------
        None
        """
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

        if "dayofyear" in dataset_xarray:
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

        with self.model:
            pm.set_data(data, coords=coords)

    def sample_posterior_predictive(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        extend_idata: bool = True,
        combined: bool = True,
        include_last_observations: bool = False,
        **sample_posterior_predictive_kwargs,
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
        # Update model data with xarray
        self._set_xarray_data(
            self._posterior_predictive_data_transformation(
                X, y, include_last_observations
            )
        )

        with self.model:
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
