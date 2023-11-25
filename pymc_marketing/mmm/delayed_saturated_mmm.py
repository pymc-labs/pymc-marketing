import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
from xarray import DataArray

from pymc_marketing.mmm.base import MMM
from pymc_marketing.mmm.preprocessing import MaxAbsScaleChannels, MaxAbsScaleTarget
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from pymc_marketing.mmm.utils import generate_fourier_modes
from pymc_marketing.mmm.validating import ValidateControlColumns

__all__ = ["DelayedSaturatedMMM"]


class BaseDelayedSaturatedMMM(MMM):
    _model_type = "DelayedSaturatedMMM"
    version = "0.0.2"

    def __init__(
        self,
        date_column: str,
        channel_columns: List[str],
        adstock_max_lag: int,
        model_config: Optional[Dict] = None,
        sampler_config: Optional[Dict] = None,
        validate_data: bool = True,
        control_columns: Optional[List[str]] = None,
        yearly_seasonality: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Media Mix Model with delayed adstock and logistic saturation class (see [1]_).

        Parameters
        ----------
        date_column : str
            Column name of the date variable.
        channel_columns : List[str]
            Column names of the media channel variables.
        model_config : Dictionary, optional
            dictionary of parameters that initialise model configuration. Class-default defined by the user default_model_config method.
        sampler_config : Dictionary, optional
            dictionary of parameters that initialise sampler configuration. Class-default defined by the user default_sampler_config method.
        validate_data : bool, optional
            Whether to validate the data before fitting to model, by default True.
        control_columns : Optional[List[str]], optional
            Column names of control variables to be added as additional regressors, by default None
        adstock_max_lag : int, optional
            Number of lags to consider in the adstock transformation, by default 4
        yearly_seasonality : Optional[int], optional
            Number of Fourier modes to model yearly seasonality, by default None.

        References
        ----------
        .. [1] Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017).
        """
        self.control_columns = control_columns
        self.adstock_max_lag = adstock_max_lag
        self.yearly_seasonality = yearly_seasonality
        self.date_column = date_column
        self.validate_data = validate_data

        super().__init__(
            date_column=date_column,
            channel_columns=channel_columns,
            model_config=model_config,
            sampler_config=sampler_config,
            adstock_max_lag=adstock_max_lag,
        )

    @property
    def default_sampler_config(self) -> Dict:
        return {}

    @property
    def output_var(self):
        """Defines target variable for the model"""
        return "y"

    def _generate_and_preprocess_model_data(  # type: ignore
        self, X: Union[pd.DataFrame, pd.Series], y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Applies preprocessing to the data before fitting the model.
        if validate is True, it will check if the data is valid for the model.
        sets self.model_coords based on provided dataset

        Parameters
        ----------
        X : Union[pd.DataFrame, pd.Series], shape (n_obs, n_features)
        y : Union[pd.Series, np.ndarray], shape (n_obs,)
        """
        date_data = X[self.date_column]
        channel_data = X[self.channel_columns]
        coords: Dict[str, Any] = {
            "date": date_data,
            "channel": self.channel_columns,
        }

        new_X_dict = {
            self.date_column: date_data,
        }
        X_data = pd.DataFrame.from_dict(new_X_dict)
        X_data = pd.concat([X_data, channel_data], axis=1)
        control_data: Optional[Union[pd.DataFrame, pd.Series]] = None
        if self.control_columns is not None:
            control_data = X[self.control_columns]
            coords["control"] = self.control_columns
            X_data = pd.concat([X_data, control_data], axis=1)

        fourier_features: Optional[pd.DataFrame] = None
        if self.yearly_seasonality is not None:
            fourier_features = self._get_fourier_models_data(X=X)
            self.fourier_columns = fourier_features.columns
            coords["fourier_mode"] = fourier_features.columns.to_numpy()
            X_data = pd.concat([X_data, fourier_features], axis=1)

        self.model_coords = coords
        if self.validate_data:
            self.validate("X", X_data)
            self.validate("y", y)
        self.preprocessed_data: Dict[str, Union[pd.DataFrame, pd.Series]] = {
            "X": self.preprocess("X", X_data),  # type: ignore
            "y": self.preprocess("y", y),  # type: ignore
        }
        self.X: pd.DataFrame = X_data
        self.y: Union[pd.Series, np.ndarray] = y

    def _save_input_params(self, idata) -> None:
        """Saves input parameters to the attrs of idata."""
        idata.attrs["date_column"] = json.dumps(self.date_column)
        idata.attrs["control_columns"] = json.dumps(self.control_columns)
        idata.attrs["channel_columns"] = json.dumps(self.channel_columns)
        idata.attrs["adstock_max_lag"] = json.dumps(self.adstock_max_lag)
        idata.attrs["validate_data"] = json.dumps(self.validate_data)
        idata.attrs["yearly_seasonality"] = json.dumps(self.yearly_seasonality)

    def build_model(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        **kwargs,
    ) -> None:
        """
        Builds a probabilistic model using PyMC for marketing mix modeling.

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
        """
        model_config = self.model_config
        self._generate_and_preprocess_model_data(X, y)
        with pm.Model(coords=self.model_coords) as self.model:
            channel_data_ = pm.MutableData(
                name="channel_data",
                value=self.preprocessed_data["X"][self.channel_columns].to_numpy(),
                dims=("date", "channel"),
            )

            target_ = pm.MutableData(
                name="target",
                value=self.preprocessed_data["y"],
                dims="date",
            )

            #Building the priors
            priors = self.create_priors_from_config(self.model_config)

            #Specifying the variables
            intercept = priors['intercept']
            beta_channel = priors['beta_channel']
            alpha = priors['alpha']
            lam = priors['lam']
            gamma_control = priors['gamma_control']

            channel_adstock = pm.Deterministic(
                name="channel_adstock",
                var=geometric_adstock(
                    x=channel_data_,
                    alpha=alpha,
                    l_max=self.adstock_max_lag,
                    normalize=True,
                    axis=0,
                ),
                dims=("date", "channel"),
            )
            channel_adstock_saturated = pm.Deterministic(
                name="channel_adstock_saturated",
                var=logistic_saturation(x=channel_adstock, lam=lam),
                dims=("date", "channel"),
            )
 
            channel_contributions = pm.Deterministic(
                name="channel_contributions",
                var=channel_adstock_saturated * beta_channel,
                dims=("date", "channel"),
            )

            mu_var = intercept + channel_contributions.sum(axis=-1)
            if (
                self.control_columns is not None
                and len(self.control_columns) > 0
                and all(
                    column in self.preprocessed_data["X"].columns
                    for column in self.control_columns
                )
            ):
                control_data_ = pm.MutableData(
                    name="control_data",
                    value=self.preprocessed_data["X"][self.control_columns],
                    dims=("date", "control"),
                )

                print("Shape of control_data_:", control_data_.eval().shape)
                print("Shape of gamma_control:", gamma_control.eval().shape)

                control_contributions = pm.Deterministic(
                    name="control_contributions",
                    var=control_data_ * gamma_control,
                    dims=("date", "control"),
                )

                mu_var += control_contributions.sum(axis=-1)
            if (
                hasattr(self, "fourier_columns")
                and self.fourier_columns is not None
                and len(self.fourier_columns) > 0
                and all(
                    column in self.preprocessed_data["X"].columns
                    for column in self.fourier_columns
                )
            ):
                fourier_data_ = pm.MutableData(
                    name="fourier_data",
                    value=self.preprocessed_data["X"][self.fourier_columns],
                    dims=("date", "fourier_mode"),
                )

                gamma_fourier = pm.Laplace(
                    name="gamma_fourier",
                    mu=model_config["gamma_fourier"]["mu"],
                    b=model_config["gamma_fourier"]["b"],
                    dims=model_config["gamma_fourier"]["dims"],
                )

                fourier_contribution = pm.Deterministic(
                    name="fourier_contributions",
                    var=fourier_data_ * gamma_fourier,
                    dims=("date", "fourier_mode"),
                )

                mu_var += fourier_contribution.sum(axis=-1)

            mu = pm.Deterministic(
                name="mu", var=mu_var, dims=model_config["mu"]["dims"]
            )

            likelihood = self.create_likelihood(self.model_config, target_, mu)

    def create_priors_from_config(self, model_config):
        priors, dimensions = {}, {"channel": len(self.channel_columns), "control": len(self.control_columns)}
        stacked_priors = {}
    
        positive_params = {"intercept", "beta_channel", "alpha", "lam", "sigma"}  # Set of params that need positive=True

        for param, config in model_config.items():
            if param == "likelihood": continue

            prior_type = config.get("type")
            if prior_type is not None:
               
                # Initial value based on parameter name
                is_positive = param in positive_params

                # Override if the config explicitly sets the 'positive' key
                if 'positive' in config:
                    is_positive = config.get('positive')

                if prior_type == "tvp":
                    if param in ["intercept", "lam", "alpha", "sigma"]:
                        priors[param] = self.gp_wrapper(name=param, X=np.arange(len(self.X[self.date_column]))[:, None], config=config, positive=is_positive)
                        continue

                    length = dimensions.get(config.get("dims", [None, None])[1], 1)
                    priors[param] = self.create_tvp_priors(param, config, length, positive=is_positive)
                    continue

                dist_func = getattr(pm, prior_type, None)
                if not dist_func: raise ValueError(f"Invalid distribution type {prior_type}")
                config_copy = {k: v for k, v in config.items() if k != "type"}
                priors[param] = dist_func(name=param, **config_copy)

        return priors

    def create_likelihood(self, model_config, target_, mu):
        likelihood_config = model_config.get("likelihood", {})
        likelihood_type = likelihood_config.get("type")
        dims = likelihood_config.get("dims")

        if not likelihood_type:
            raise ValueError("Likelihood type must be specified in the model config.")

        likelihood_func = getattr(pm, likelihood_type, None)
        if likelihood_func is None:
            raise ValueError(f"Invalid likelihood type {likelihood_type}")

        # Transform mu if the likelihood type is Lognormal or HurdleLognormal
        if likelihood_type in ['LogNormal', 'HurdleLogNormal']:
            mu = pt.log(mu)

        # Create sub-priors
        sub_priors = {}
        for param, config in likelihood_config.items():
            if param not in ['type', 'dims']:  # Skip 'type' and 'dims'
                if param == 'params':  # Handle nested 'params'
                    for sub_param, sub_config in config.items():
                        sub_priors[sub_param] = self.create_priors_from_config({sub_param: sub_config})[sub_param]
                else:
                    sub_priors[param] = self.create_priors_from_config({param: config})[param]

        return likelihood_func(name="likelihood", mu=mu, observed=target_, dims=dims,  **sub_priors)

    def create_tvp_priors(self, param, config, length, positive=False):
        dims = config.get("dims", None)  # Extracting dims from the config
        print(dims)
        gp_list = [self.gp_wrapper(name=f"{param}_{i}", X=np.arange(len(self.X[self.date_column]))[:, None], config=config, positive=positive) for i in range(length)]
        stacked_gp = pt.stack(gp_list, axis=1)
        return pm.Deterministic(f"{param}", stacked_gp, dims=dims)


    def gp_wrapper(self, name, X, config, positive=False, **kwargs):
        return self.gp_coeff(X, name, config=config, positive=positive, **kwargs)

    def gp_coeff(self, X, name, mean=0.0, positive=False, config=None):
        params = pm.find_constrained_prior(pm.Gamma, 8, 12, init_guess={"alpha": 1, "beta": 1}, mass=0.8)
        ell = pm.Gamma(f"ell_{name}", **params)
        eta = pm.Exponential(f"_eta_{name}", lam=1)
  #      cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ls=ell)

        cov = eta ** 2 * pm.gp.cov.Matern32(1, ls=ell)
    
        gp = pm.gp.HSGP(m=[40], c=4, cov_func=cov)
        f_raw = gp.prior(f"{name}_tvp_raw", X=X)
    

        # Inside your gp_coeff function
        # Offset
        offset_config = config.get('offset', None) if config else None
        if offset_config:
            offset_type = offset_config.get('type')
            offset_params = {k: v for k, v in offset_config.items() if k != 'type'}
            offset_prior = getattr(pm, offset_type)(name=f"{name}_offset", **offset_params)
        else: 
            offset_prior = 0

        if positive:
            f_output = pm.Deterministic(f"{name}", (pt.exp(f_raw)) + offset_prior, dims=("date"))
        else:
            f_output = pm.Deterministic(f"{name}", f_raw + offset_prior, dims=("date"))
    
        return f_output



    @property
    def default_model_config(self) -> Dict:
        model_config: Dict = {
            "intercept": {"type": "Normal", "mu": 0, "sigma": 2},
            "beta_channel": {"type": "HalfNormal", "sigma": 2, "dims": ("channel",)},
            "alpha": {"type": "Beta", "alpha": 1, "beta": 3, "dims": ("channel",)},
            "lam": {"type": "Gamma", "alpha": 3, "beta": 1, "dims": ("channel",)},
            "gamma_control": {'type': 'Gamma', 'alpha': 2, 'beta': 1, 'dims': ('control',)},
            "mu": {"dims": ("date",)},
            "likelihood": {
                "type": "Normal",
                "dims": ("date",),
                "params": {
                    "sigma": {"type": "HalfNormal", "sigma": 1, 'dims': ('date',)},
                }
            },
            "gamma_fourier": {"mu": 0, "b": 1, "dims": "fourier_mode"},
        }
        return model_config



    def _get_fourier_models_data(self, X) -> pd.DataFrame:
        """Generates fourier modes to model seasonality.

        References
        ----------
        https://www.pymc.io/projects/examples/en/latest/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html
        """
        if self.yearly_seasonality is None:
            raise ValueError("yearly_seasonality must be specified.")
        date_data: pd.Series = pd.to_datetime(
            arg=X[self.date_column], format="%Y-%m-%d"
        )
        periods: npt.NDArray[np.float_] = date_data.dt.dayofyear.to_numpy() / 365.25
        return generate_fourier_modes(
            periods=periods,
            n_order=self.yearly_seasonality,
        )

    def channel_contributions_forward_pass(
        self, channel_data: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """Evaluate the channel contribution for a given channel data and a fitted model, ie. the forward pass.
        Parameters
        ----------
        channel_data : array-like
            Input channel data. Result of all the preprocessing steps.
        Returns
        -------
        array-like
            Transformed channel data.
        """
        alpha_posterior = self.fit_result["alpha"].to_numpy()

        lam_posterior = self.fit_result["lam"].to_numpy()
        lam_posterior_expanded = np.expand_dims(a=lam_posterior, axis=2)

        beta_channel_posterior = self.fit_result["beta_channel"].to_numpy()
        beta_channel_posterior_expanded = np.expand_dims(
            a=beta_channel_posterior, axis=2
        )

        geometric_adstock_posterior = geometric_adstock(
            x=channel_data,
            alpha=alpha_posterior,
            l_max=self.adstock_max_lag,
            normalize=True,
            axis=0,
        )

        logistic_saturation_posterior = logistic_saturation(
            x=geometric_adstock_posterior,
            lam=lam_posterior_expanded,
        )

        channel_contribution_forward_pass = (
            beta_channel_posterior_expanded * logistic_saturation_posterior
        )
        return channel_contribution_forward_pass.eval()

    @property
    def _serializable_model_config(self) -> Dict[str, Any]:
        def ndarray_to_list(d: Dict) -> Dict:
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
    def load(cls, fname: str):
        """
        Creates a DelayedSaturatedMMM instance from a file,
        instantiating the model with the saved original input parameters.
        Loads inference data for the model.

        Parameters
        ----------
        fname : string
            This denotes the name with path from where idata should be loaded from.

        Returns
        -------
        Returns an instance of DelayedSaturatedMMM.

        Raises
        ------
        ValueError
            If the inference data that is loaded doesn't match with the model.
        """

        filepath = Path(str(fname))
        idata = az.from_netcdf(filepath)
        model_config = cls._model_config_formatting(
            json.loads(idata.attrs["model_config"])
        )
        model = cls(
            date_column=json.loads(idata.attrs["date_column"]),
            control_columns=json.loads(idata.attrs["control_columns"]),
            channel_columns=json.loads(idata.attrs["channel_columns"]),
            adstock_max_lag=json.loads(idata.attrs["adstock_max_lag"]),
            validate_data=json.loads(idata.attrs["validate_data"]),
            yearly_seasonality=json.loads(idata.attrs["yearly_seasonality"]),
            model_config=model_config,
            sampler_config=json.loads(idata.attrs["sampler_config"]),
        )
        model.idata = idata
        dataset = idata.fit_data.to_dataframe()
        X = dataset.drop(columns=[model.output_var])
        y = dataset[model.output_var].values
        model.build_model(X, y)
        # All previously used data is in idata.
        if model.id != idata.attrs["id"]:
            raise ValueError(
                f"The file '{fname}' does not contain an inference data of the same model or configuration as '{cls._model_type}'"
            )

        return model

    def _data_setter(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        """
        Sets new data in the model.

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
        new_channel_data: Optional[np.ndarray] = None

        if isinstance(X, pd.DataFrame):
            try:
                new_channel_data = X[self.channel_columns].to_numpy()
            except KeyError as e:
                raise RuntimeError("New data must contain channel_data!", e)
        elif isinstance(X, np.ndarray):
            new_channel_data = X
        else:
            raise TypeError("X must be either a pandas DataFrame or a numpy array")

        data: Dict[str, Union[np.ndarray, Any]] = {"channel_data": new_channel_data}

        if y is not None:
            if isinstance(y, pd.Series):
                data[
                    "target"
                ] = y.to_numpy()  # convert Series to numpy array explicitly
            elif isinstance(y, np.ndarray):
                data["target"] = y
            else:
                raise TypeError("y must be either a pandas Series or a numpy array")

        with self.model:
            pm.set_data(data)

    @classmethod
    def _model_config_formatting(cls, model_config: Dict) -> Dict:
        """
        Because of json serialization, model_config values that were originally tuples or numpy are being encoded as lists.
        This function converts them back to tuples and numpy arrays to ensure correct id encoding.
        """

        def format_nested_dict(d: Dict) -> Dict:
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


class DelayedSaturatedMMM(
    MaxAbsScaleTarget,
    MaxAbsScaleChannels,
    ValidateControlColumns,
    BaseDelayedSaturatedMMM,
):
    ...

    def channel_contributions_forward_pass(
        self, channel_data: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """Evaluate the channel contribution for a given channel data and a fitted model, ie. the forward pass.
        We return the contribution in the original scale of the target variable.
        Parameters
        ----------
        channel_data : array-like
            Input channel data. Result of all the preprocessing steps.
        Returns
        -------
        array-like
            Transformed channel data.
        """
        channel_contribution_forward_pass = super().channel_contributions_forward_pass(
            channel_data=channel_data
        )
        target_transformed_vectorized = np.vectorize(
            self.target_transformer.inverse_transform,
            excluded=[1, 2],
            signature="(m, n) -> (m, n)",
        )
        return target_transformed_vectorized(channel_contribution_forward_pass)

    def get_channel_contributions_forward_pass_grid(
        self, start: float, stop: float, num: int
    ) -> DataArray:
        """Generate a grid of scaled channel contributions for a given grid of share values.
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

        channel_contributions = []
        for delta in share_grid:
            channel_data = (
                delta * self.preprocessed_data["X"][self.channel_columns].to_numpy()
            )
            channel_contribution_forward_pass = self.channel_contributions_forward_pass(
                channel_data=channel_data
            )
            channel_contributions.append(channel_contribution_forward_pass)
        return DataArray(
            data=np.array(channel_contributions),
            dims=("delta", "chain", "draw", "date", "channel"),
            coords={
                "delta": share_grid,
                "date": self.X[self.date_column],
                "channel": self.channel_columns,
            },
        )

    def plot_channel_contributions_grid(
        self,
        start: float,
        stop: float,
        num: int,
        absolute_xrange: bool = False,
        **plt_kwargs: Any,
    ) -> plt.Figure:
        """Plots a grid of scaled channel contributions for a given grid of share values.
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
        Returns
        -------
        plt.Figure
            Plot of grid of channel contributions.
        """
        share_grid = np.linspace(start=start, stop=stop, num=num)
        contributions = self.get_channel_contributions_forward_pass_grid(
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
                label=f"{channel} $94%$ HDI contribution",
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
