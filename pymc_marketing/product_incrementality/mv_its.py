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
"""Multivariate Interrupted Time Series Analysis for Product Incrementality."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

HDI_ALPHA = 0.5


class MVITS:
    """Class to perform a multivariate interrupted time series analysis with the
    specific intent of determining where the sales of a new product came from."""

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time,
        background_sales: list[str],
        innovation_sales: str,
        rng=42,
        sample_kwargs: dict | None = None,
    ):
        self.data = data
        self.treatment_time = treatment_time
        self.background_sales = background_sales
        self.innovation_sales = innovation_sales
        self.rng = rng
        self.sample_kwargs = sample_kwargs if sample_kwargs is not None else {}

        # build the model
        self.model = self.build_model(
            self.data[self.background_sales],
            self.data[self.innovation_sales],
            treatment_time=self.treatment_time,
        )

        # sample from prior, posterior, posterior predictive
        with self.model:
            self.idata = pm.sample_prior_predictive(random_seed=self.rng)
            self.idata.extend(pm.sample(**self.sample_kwargs, random_seed=self.rng))
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata,
                    var_names=["mu", "y"],
                    random_seed=self.rng,
                )
            )

        # Calculate the counterfactual background sales, if the new product had not been introduced
        zero_sales = np.zeros(self.data[self.innovation_sales].shape, dtype=np.int32)
        self.counterfactual_model = pm.do(self.model, {"innovation_sales": zero_sales})
        with self.counterfactual_model:
            self.idata_counterfactual = pm.sample_posterior_predictive(
                self.idata, var_names=["mu", "y"], random_seed=self.rng
            )

        return

    @staticmethod
    def build_model(
        background_sales: pd.DataFrame,
        innovation_sales: pd.Series,
        treatment_time,
        *,
        alpha_background=0.5,
    ):
        """Return a PyMC model for a multivariate interrupted time series analysis."""

        if not background_sales.index.equals(innovation_sales.index):
            raise ValueError(
                "Index of background_sales and innovation_sales must match."
            )

        # note: type hints for coords required for mypi to not get confused
        coords: dict[str, list[str]] = {
            "background_product": list(background_sales.columns),
            "time": list(background_sales.index.values),
        }

        print(coords["background_product"])
        print(type(coords["background_product"]))
        print(len(coords["background_product"]))

        with pm.Model(coords=coords) as model:
            # data
            _background_sales = pm.Data(
                "background_sales",
                background_sales.values,
                dims=("time", "background_product"),
            )
            innovation_sales = pm.Data(
                "innovation_sales", innovation_sales.values, dims=("time",)
            )

            # priors
            intercept = pm.Normal(
                "intercept",
                mu=pm.math.mean(background_sales[:treatment_time], axis=0),
                sigma=np.std(background_sales[:treatment_time], axis=0),
                # sigma=20,
                dims="background_product",
            )

            sigma = pm.HalfNormal(
                "background_product_sigma", sigma=10, dims="background_product"
            )

            alpha = np.full(len(coords["background_product"]), alpha_background)
            beta = pm.Dirichlet("beta", a=alpha, dims="background_product")

            # expectation
            mu = pm.Deterministic(
                "mu",
                intercept[None, :] - innovation_sales[:, None] * beta[None, :],
                dims=("time", "background_product"),
            )

            # likelihood
            pm.Normal(
                "y",
                mu=mu,
                sigma=sigma,
                observed=_background_sales,
                dims=("time", "background_product"),
            )
        return model

    @property
    def causal_impact(self, variable="mu"):
        """Calculates the causal impact of the new product on the background products."""
        # Note: if we compare "mu" then we are comparing the expected sales,
        # if we compare "y" then we are comparing the actual sales
        if variable not in ["mu", "y"]:
            raise ValueError(f"variable must be either 'mu' or 'y', not {variable}")

        return (
            self.idata.posterior_predictive["mu"]
            - self.idata_counterfactual.posterior_predictive["mu"]
        )

    def plot_fit(self, variable="mu"):
        """Plots the model fit (posterior predictive) of the background products."""

        if variable not in ["mu", "y"]:
            raise ValueError(f"variable must be either 'mu' or 'y', not {variable}")

        fig, ax = plt.subplots()

        # plot data
        self.plot_data(self.data, ax)

        # plot posterior predictive distribution of sales for each of the background products
        x = self.data.index.values
        background_products = list(self.idata.observed_data.background_product.data)
        for i, background_product in enumerate(background_products):
            az.plot_hdi(
                x,
                self.idata.posterior_predictive[variable]
                .transpose(..., "time")
                .sel(background_product=background_product),
                fill_kwargs={
                    "alpha": HDI_ALPHA,
                    "color": f"C{i}",
                    "label": "Posterior predictive (HDI)",
                },
                smooth=False,
            )

        # formatting
        ax.legend()
        ax.set(title="Model fit of sales of background products", ylabel="Sales")
        return ax

    def plot_counterfactual(self, variable="mu"):
        """Plots the predicted sales of the background products under the counterfactual
        scenario of never releasing the new product."""
        fig, ax = plt.subplots()

        if variable not in ["mu", "y"]:
            raise ValueError(f"variable must be either 'mu' or 'y', not {variable}")

        # plot data
        self.plot_data(self.data, ax)

        # plot posterior predictive distribution of sales for each of the background products
        x = self.data.index.values
        background_products = list(self.idata.observed_data.background_product.data)
        for i, background_product in enumerate(background_products):
            az.plot_hdi(
                x,
                self.idata_counterfactual.posterior_predictive[variable]
                .transpose(..., "time")
                .sel(background_product=background_product),
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

    def plot_causal_impact(self, type="sales"):
        """Plot the inferred causal impact of the new product on the background products."""
        fig, ax = plt.subplots()

        # plot posterior predictive distribution of sales for each of the background products
        x = self.data.index.values
        background_products = list(self.idata.observed_data.background_product.data)

        if type == "sales":
            for i, background_product in enumerate(background_products):
                az.plot_hdi(
                    x,
                    self.causal_impact.transpose(..., "time").sel(
                        background_product=background_product
                    ),
                    fill_kwargs={
                        "alpha": HDI_ALPHA,
                        "color": f"C{i}",
                        "label": "Posterior predictive (HDI)",
                    },
                    smooth=False,
                )
            ax.set(ylabel="Change in sales caused by new product")

        elif type == "market_share":
            """change in terms of market share in percent is given by:
            (causal_impact / total_sales) * 100
            """
            import matplotlib.ticker as mtick

            # divide the causal impact change in sales by the counterfactual predicted sales
            variable = "mu"
            for i, background_product in enumerate(background_products):
                causal_impact = self.causal_impact.transpose(..., "time").sel(
                    background_product=background_product
                )
                total_sales = (
                    self.idata_counterfactual.posterior_predictive[variable]
                    .transpose(..., "time")
                    .sum(dim="background_product")
                )
                causal_impact_market_share = (causal_impact / total_sales) * 100

                az.plot_hdi(
                    x,
                    causal_impact_market_share,
                    fill_kwargs={
                        "alpha": HDI_ALPHA,
                        "color": f"C{i}",
                        "label": f"{background_product} - Posterior predictive (HDI)",
                    },
                    smooth=False,
                )
            ax.set(ylabel="Change in market share caused by new product")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        else:
            raise ValueError("`type` must be either 'sales' or 'market_share'.")

        # formatting
        ax.legend()
        ax.set(title="Estimated causal impact of new product upon existing products")
        return ax

    @staticmethod
    def plot_data(data, ax=None):
        """Plot the observed data."""
        if ax is None:
            fig, ax = plt.subplots()
        data.plot(ax=ax)
        data.sum(axis=1).plot(label="total sales", color="black", ax=ax)
        ax.set_ylim(bottom=0)
        ax.set(ylabel="Sales")
        ax.legend()
        return ax
