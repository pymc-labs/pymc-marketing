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
"""Multivariate Interrupted Time Series Analysis for Product Incrementality."""

import json
from typing import Any, cast

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pymc as pm
from matplotlib.axes import Axes
from typing_extensions import Self
from xarray import DataArray

from pymc_marketing.model_builder import ModelBuilder, create_idata_accessor
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.prior import Prior

HDI_ALPHA = 0.5


class MVITS(ModelBuilder):
    """Multivariate Interrupted Time Series class.

    Class to perform a multivariate interrupted time series analysis with the
    specific intent of determining where the sales of a new product came from.

    Parameters
    ----------
    existing_sales : list of str
        The names of the existing products.
    saturated_market : bool, optional
        Whether the market is saturated or not. If True, the sum of the beta's will be
        1. Else, the sum of the beta's will be less than 1 with the remaining sales
        attributed to the new product.
    model_config : dict, optional
        The model configuration. If None, the default model configuration will be used.
    sampler_config : dict, optional
        The sampler configuration. If None, the default sampler configuration will be used.

    """

    _model_type = "Multivariate Interrupted Time Series"
    version = "0.1.0"

    def __init__(
        self,
        existing_sales: list[str],
        saturated_market: bool = True,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        self.existing_sales = existing_sales
        self.saturated_market = saturated_market

        model_config = model_config or {}
        model_config = parse_model_config(model_config)

        super().__init__(model_config=model_config, sampler_config=sampler_config)

        self._distribution_checks()

    def _distribution_checks(self):
        if self.model_config["market_distribution"].distribution != "Dirichlet":
            raise ValueError("market_distribution must be a Dirichlet distribution")  #

        dims = "existing_product" if self.saturated_market else "all_sources"

        if dims not in self.model_config["market_distribution"].dims:
            raise ValueError(
                f"market_distribution must have dims='{dims}', not {self.model_config['market_distribution'].dims}"
            )

    def create_idata_attrs(self) -> dict[str, str]:
        """Create the attributes for the InferenceData object.

        Returns
        -------
        dict[str, str]
            The attributes for the InferenceData object.

        """
        attrs = super().create_idata_attrs()
        attrs["existing_sales"] = json.dumps(self.existing_sales)
        attrs["saturated_market"] = json.dumps(self.saturated_market)

        return attrs

    @classmethod
    def attrs_to_init_kwargs(cls, attrs) -> dict[str, Any]:
        """Convert the attributes of the InferenceData object to the __init__ kwargs.

        Parameters
        ----------
        attrs : dict
            The attributes of the InferenceData object.

        Returns
        -------
        dict
            The __init__ kwargs for the class.

        """
        return {
            "model_config": json.loads(attrs["model_config"]),
            "sampler_config": json.loads(attrs["sampler_config"]),
            "existing_sales": json.loads(attrs["existing_sales"]),
            "saturated_market": json.loads(attrs["saturated_market"]),
        }

    @property
    def default_model_config(self) -> dict:
        """Default model configuration.

        This is TruncatedNormal likelihood with a HalfNormal sigma, Normal intercept,
        and a Dirichlet market distribution

        Returns
        -------
        dict
            The default model configuration.

        """
        if self.saturated_market:
            a = np.full(len(self.existing_sales), 0.5)
            dims = "existing_product"
        else:
            a = np.full(len(self.existing_sales) + 1, 0.5)
            dims = "all_sources"

        market_distribution = Prior("Dirichlet", a=a, dims=dims)

        return {
            "intercept": Prior("Normal", dims="existing_product"),
            "likelihood": Prior(
                "TruncatedNormal",
                lower=0,
                sigma=Prior("HalfNormal", dims="existing_product"),
                dims=("time", "existing_product"),
            ),
            "market_distribution": market_distribution,
        }

    def inform_default_prior(self, data: pd.DataFrame) -> Self:
        """Inform the default prior based on the data.

        This only works with the default prior.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to inform the default prior. This should be the data before
            the treatment.

        Returns
        -------
        Self
            The model instance.

        Examples
        --------
        Use the data before the treatment to inform the prior.

        .. code-block:: python

            data = df.loc[:treatment_time, model.existing_sales]
            model.inform_default_prior(data=data)

        Check the model configuration

        .. code-block:: python

            model.model_config

        """
        intercept = self.model_config["intercept"]
        likelihood_sigma = self.model_config["likelihood"]["sigma"]

        if intercept.distribution != "Normal":
            raise ValueError("intercept must be a Normal distribution")

        if likelihood_sigma.distribution != "HalfNormal":
            raise ValueError("likelihood sigma must be a HalfNormal distribution")

        mean = data.mean()
        std = data.std()

        intercept.parameters = {
            "mu": mean.to_numpy(),
            "sigma": std.to_numpy(),
        }
        likelihood_sigma.parameters["sigma"] = std.mean()
        return self

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration."""
        return {}

    @property
    def output_var(self) -> str:
        """The output variable of the model."""
        return "y"

    @property
    def _serializable_model_config(self) -> dict[str, int | float | dict]:  # type: ignore
        result: dict[str, int | float | dict] = {
            "intercept": self.model_config["intercept"].to_dict(),
            "likelihood": self.model_config["likelihood"].to_dict(),
            "market_distribution": self.model_config["market_distribution"].to_dict(),
        }

        return result

    def _generate_and_preprocess_model_data(
        self,
        X: pd.DataFrame | pd.Series,
        y: np.ndarray,
    ) -> None:
        if isinstance(X, pd.Series):
            raise ValueError("X must be a DataFrame, not a Series")  # pragma: no cover

        self.X = X[self.existing_sales]
        self.y = pd.Series(y, index=X.index, name=self.output_var)

        # note: type hints for coords required for mypy to not get confused
        self.coords: dict[str, list[str]] = {
            "existing_product": list(self.existing_sales),
            "time": list(X.index.values),
            "all_sources": [
                *list(self.existing_sales),
                "new",
            ],
        }

    def build_model(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        **kwargs,
    ) -> None:
        """Build a PyMC model for a multivariate interrupted time series analysis.

        Parameters
        ----------
        X : pd.DataFrame
            The data for the existing products.
        y : np.ndarray | pd.Series
            The data for the new product.

        """
        self._generate_and_preprocess_model_data(X, y)  # type: ignore

        with pm.Model(coords=self.coords) as model:
            # data
            _existing_sales = pm.Data(
                "existing_sales",
                X.values,
                dims=("time", "existing_product"),
            )
            y = pm.Data(
                "treatment_sales",
                y if not isinstance(y, pd.Series) else y.values,
                dims="time",
            )

            # priors
            intercept = self.model_config["intercept"].create_variable(name="intercept")

            if self.saturated_market:
                """We assume the market is saturated. The sum of the beta's will be 1.
                This means that the reduction in sales of existing products will equal
                the increase in sales of the new product, such that the total sales
                remain constant."""
                beta = self.model_config["market_distribution"].create_variable("beta")
            else:
                """We assume the market is not saturated. The sum of the beta's will be
                less than 1. This means that the reduction in sales of existing products
                will be less than the increase in sales of the new product."""
                beta_all = self.model_config["market_distribution"].create_variable(
                    "beta_all",
                )
                beta = pm.Deterministic(
                    "beta",
                    beta_all[:-1],
                    dims="existing_product",
                )
                pm.Deterministic("new sales", beta_all[-1])

            # expectation
            mu = pm.Deterministic(
                "mu",
                intercept[None, :] - y[:, None] * beta[None, :],
                dims=("time", "existing_product"),
            )

            # likelihood
            self.model_config["likelihood"].create_likelihood_variable(
                name=self.output_var,
                mu=mu,
                observed=_existing_sales,
            )

        self.model = model

    def _data_setter(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> None:
        """Set the data.

        Required from the parent class

        """

    def calculate_counterfactual(
        self,
        random_seed: np.random.Generator | int | None = None,
    ) -> None:
        """Calculate the counterfactual scenario of never releasing the new product.

        Extends the InferenceData object

        Parameters
        ----------
        random_seed : np.random.Generator | int, optional
            The random seed for the sampling.

        """
        if not hasattr(self, "model"):
            raise RuntimeError("Call the 'fit' method first.")

        zero_sales = np.zeros_like(self.y, dtype=np.int32)
        self.counterfactual_model = pm.do(self.model, {"treatment_sales": zero_sales})
        with self.counterfactual_model:
            self.idata.extend(  # type: ignore
                pm.sample_posterior_predictive(
                    self.posterior,
                    var_names=["mu", self.output_var],
                    random_seed=random_seed,
                    predictions=True,
                )
            )

    def sample(
        self,
        X,
        y,
        random_seed: np.random.Generator | int | None = None,
        sample_prior_predictive_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        sample_posterior_predictive_kwargs: dict | None = None,
    ) -> Self:
        """Sample all the things.

        Run all of the sample methods in the sequence:

        - :meth:`sample_prior_predictive`
        - :meth:`fit`
        - :meth:`sample_posterior_predictive`
        - :meth:`calculate_counterfactual`

        Parameters
        ----------
        X : pd.DataFrame
            The data for the existing products.
        y : np.ndarray | pd.Series
            The data for the new product.
        random_seed : np.random.Generator | int, optional
            The random seed for each stage of sampling.
        sample_prior_predictive_kwargs : dict, optional
            The keyword arguments for the sample_prior_predictive method.
        fit_kwargs : dict, optional
            The keyword arguments for the fit method.
        sample_posterior_predictive_kwargs : dict, optional
            The keyword arguments for the sample_posterior_predictive method.

        Returns
        -------
        Self
            The model instance.

        """
        sample_prior_predictive_kwargs = sample_prior_predictive_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        sample_posterior_predictive_kwargs = sample_posterior_predictive_kwargs or {}

        self.sample_prior_predictive(
            X,
            y,
            random_seed=random_seed,
            **sample_prior_predictive_kwargs,
        )
        self.fit(X, y, random_seed=random_seed, **fit_kwargs)
        self.sample_posterior_predictive(
            X,
            random_seed=random_seed,
            var_names=[self.output_var, "mu"],
            **sample_posterior_predictive_kwargs,
        )
        self.calculate_counterfactual(random_seed=random_seed)

        return self

    def causal_impact(self, variable: str = "mu") -> DataArray:
        """Calculate the causal impact of the new product on the existing products.

        Note: if we compare "mu" then we are comparing the expected sales, if we compare
        "y" then we are comparing the actual sales

        Parameters
        ----------
        variable : str, optional
            The variable to compare. Either "mu" or "y".

        Returns
        -------
        xr.DataArray
            The causal impact of the new product on the existing products.

        """
        if variable not in ["mu", "y"]:
            raise ValueError(
                f"variable must be either 'mu' or 'y', not {variable}"
            )  # pragma: no cover

        return self.posterior_predictive[variable] - self.predictions[variable]

    def plot_fit(
        self,
        variable: str = "mu",
        plot_total_sales: bool = True,
        ax: Axes | None = None,
    ):
        """Plot the model fit (posterior predictive) of the existing products.

        Parameters
        ----------
        variable : str, optional
            The variable to compare. Either "mu" or "y".
        plot_total_sales : bool, optional
            Whether to plot the total sales or not.
        ax : plt.Axes, optional
            The matplotlib axes.

        Returns
        -------
        plt.Axes
            The new or modified matplotlib axes.

        """
        if ax is None:
            _, ax = plt.subplots()

        ax = cast(Axes, ax)

        if variable not in ["mu", "y"]:
            raise ValueError(
                f"variable must be either 'mu' or 'y', not {variable}"
            )  # pragma: no cover

        # plot data
        self.plot_data(ax=ax, plot_total_sales=plot_total_sales)

        # plot posterior predictive distribution of sales for each of the existing products
        x = self.X.index.values  # type: ignore
        existing_products = self.coords["existing_product"]
        for i, existing_product in enumerate(existing_products):
            az.plot_hdi(
                x,
                self.posterior_predictive[variable]  # type: ignore
                .transpose(..., "time")
                .sel(existing_product=existing_product),
                fill_kwargs={
                    "alpha": HDI_ALPHA,
                    "color": f"C{i}",
                    "label": "Posterior predictive (HDI)",
                },
                smooth=False,
            )

        # formatting
        ax.legend()
        ax.set(title="Model fit of sales of existing products", ylabel="Sales")
        return ax

    def plot_counterfactual(
        self,
        variable: str = "mu",
        plot_total_sales: bool = True,
        ax: Axes | None = None,
    ):
        """Plot counterfactual scenario.

        Plot the predicted sales of the existing products under the counterfactual
        scenario of never releasing the new product.

        Parameters
        ----------
        variable : str, optional
            The variable to compare. Either "mu" or "y".
        plot_total_sales : bool, optional
            Whether to plot the total sales or not.
        axes : plt.Axes, optional
            The matplotlib axes.

        Returns
        -------
        plt.Axes
            The new or modified matplotlib axes.

        """
        if ax is None:
            _, ax = plt.subplots()

        ax = cast(Axes, ax)

        if variable not in ["mu", "y"]:
            raise ValueError(
                f"variable must be either 'mu' or 'y', not {variable}"
            )  # pragma: no cover

        # plot data
        self.plot_data(ax=ax, plot_total_sales=plot_total_sales)

        # plot posterior predictive distribution of sales for each of the existing products
        x = cast(pd.DataFrame, self.X).index.values
        existing_products = self.coords["existing_product"]
        for i, existing_product in enumerate(existing_products):
            az.plot_hdi(
                x,
                self.predictions[variable]  # type: ignore
                .transpose(..., "time")
                .sel(existing_product=existing_product),
                fill_kwargs={
                    "alpha": HDI_ALPHA,
                    "color": f"C{i}",
                    "label": "Posterior predictive (HDI)",
                },
                smooth=False,
            )

        # formatting
        ax.legend()
        ax.set(
            title="Model predictions under the counterfactual scenario", ylabel="Sales"
        )
        return ax

    def plot_causal_impact_sales(self, variable: str = "mu", ax: Axes | None = None):
        """Plot causal impact of sales.

        Plot the inferred causal impact of the new product on the sales of the
        existing products.

        Note: if we compare "mu" then we are comparing the expected sales, if we compare
        "y" then we are comparing the actual sales

        Parameters
        ----------
        variable : str, optional
            The variable to compare. Either "mu" or "y".
        ax : plt.Axes, optional
            The matplotlib axes.

        Returns
        -------
        plt.Axes
            The new or modified matplotlib axes.

        """
        if ax is None:
            _, ax = plt.subplots()

        ax = cast(Axes, ax)

        # plot posterior predictive distribution of sales for each of the existing products
        x = self.X.index.values  # type: ignore
        existing_products = self.coords["existing_product"]

        for i, existing_product in enumerate(existing_products):
            az.plot_hdi(
                x,
                self.causal_impact(variable=variable)
                .transpose(..., "time")
                .sel(existing_product=existing_product),
                fill_kwargs={
                    "alpha": HDI_ALPHA,
                    "color": f"C{i}",
                    "label": "Posterior predictive (HDI)",
                },
                smooth=False,
            )
        ax.set(ylabel="Change in sales caused by new product")

        # formatting
        ax.legend()
        ax.set(title="Estimated causal impact of new product upon existing products")
        return ax

    def plot_causal_impact_market_share(
        self, variable: str = "mu", ax: Axes | None = None
    ):
        """Plot the inferred causal impact of the new product on the existing products.

        Note: if we compare "mu" then we are comparing the expected sales, if we compare
        "y" then we are comparing the actual sales

        Parameters
        ----------
        variable : str, optional
            The variable to compare. Either "mu" or "y".
        ax : plt.Axes, optional
            The matplotlib axes.

        Returns
        -------
        plt.Axes
            The new or modified matplotlib axes.

        """
        if ax is None:
            _, ax = plt.subplots()

        ax = cast(Axes, ax)

        # plot posterior predictive distribution of sales for each of the existing products
        x = self.X.index.values  # type: ignore
        existing_products = list(self.idata.observed_data.existing_product.data)  # type: ignore

        # divide the causal impact change in sales by the counterfactual predicted sales
        variable = "mu"
        for i, existing_product in enumerate(existing_products):
            causal_impact = (
                self.causal_impact(variable=variable)
                .transpose(..., "time")
                .sel(existing_product=existing_product)
            )
            total_sales = (
                self.predictions[variable]  # type: ignore
                .transpose(..., "time")
                .sum(dim="existing_product")
            )
            causal_impact_market_share = (causal_impact / total_sales) * 100

            az.plot_hdi(
                x,
                causal_impact_market_share,
                fill_kwargs={
                    "alpha": HDI_ALPHA,
                    "color": f"C{i}",
                    "label": f"{existing_product} - Posterior predictive (HDI)",
                },
                smooth=False,
            )
        ax.set(ylabel="Change in market share caused by new product")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        # formatting
        ax.legend()
        ax.set(title="Estimated causal impact of new product upon existing products")
        return ax

    def plot_data(self, plot_total_sales: bool = True, ax: Axes | None = None):
        """Plot the observed data.

        Wrapper around the plot_product function.

        Parameters
        ----------
        plot_total_sales : bool, optional
            Whether to plot the total sales or not.
        ax : plt.Axes, optional
            The matplotlib axes.

        Returns
        -------
        plt.Axes
            The new or modified matplotlib axes.

        """
        data = pd.concat([self.X, self.y], axis=1)  # type: ignore

        return plot_product(data=data, ax=ax, plot_total_sales=plot_total_sales)

    predictions = create_idata_accessor(
        "predictions",
        "Call the 'calculate_counterfactual' method first.",
    )


def plot_product(
    data: pd.DataFrame,
    plot_total_sales: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Plot the sales of a single product.

    Parameters
    ----------
    data : pd.DataFrame
        The sales data.
    plot_total_sales : bool, optional
        Whether to plot the total sales or not.
    ax : plt.Axes, optional
        The matplotlib axes.

    Returns
    -------
    plt.Axes
        The new or modified matplotlib axes.

    """
    if ax is None:
        _, ax = plt.subplots()

    ax = cast(Axes, ax)

    data.plot(ax=ax)
    if plot_total_sales:
        data.sum(axis=1).plot(label="total sales", color="black", ax=ax)
    ax.set_ylim(bottom=0)
    ax.set(ylabel="Sales")
    return ax
