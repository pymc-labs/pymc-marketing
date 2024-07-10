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
        market_saturated: bool = True,
        rng=42,
        sample_kwargs: dict | None = None,
    ):
        self.data = data
        self.treatment_time = treatment_time
        self.background_sales = background_sales
        self.innovation_sales = innovation_sales
        self.rng = rng
        self.sample_kwargs = sample_kwargs if sample_kwargs is not None else {}
        self.market_saturated = market_saturated

        # build the model
        self.model = self.build_model(
            self.data[self.background_sales],
            self.data[self.innovation_sales],
            self.market_saturated,
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
        market_saturated: bool,
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
            "all_sources": [
                *list(background_sales.columns),
                "new",
            ],  # for non-saturated market only
            # "all_sources": list(background_sales.columns)
            # + ["new"],  # for non-saturated market only
        }

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

            if market_saturated:
                """We assume the market is saturated. The sum of the beta's will be 1.
                This means that the reduction in sales of existing products will equal
                the increase in sales of the new product, such that the total sales
                remain constant."""
                alpha = np.full(len(coords["background_product"]), alpha_background)
                beta = pm.Dirichlet("beta", a=alpha, dims="background_product")
            else:
                """We assume the market is not saturated. The sum of the beta's will be
                less than 1. This means that the reduction in sales of existing products
                will be less than the increase in sales of the new product."""
                alpha_all = np.full(len(coords["all_sources"]), alpha_background)
                beta_all = pm.Dirichlet("beta_all", a=alpha_all, dims="all_sources")
                beta = pm.Deterministic(
                    "beta", beta_all[:-1], dims="background_product"
                )
                pm.Deterministic("new sales", beta_all[-1])

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


def generate_constrained_data(
    total_sales_mu: int,
    total_sales_sigma: float,
    treatment_time: int,
    n_observations: int,
    market_shares_before,
    market_shares_after,
    market_share_labels,
    rng: np.random.Generator,
):
    rates = np.array(
        treatment_time * market_shares_before
        + (n_observations - treatment_time) * market_shares_after
    )

    # Generate total demand (sales) as normally distributed around some average level of sales
    total = (
        rng.normal(loc=total_sales_mu, scale=total_sales_sigma, size=n_observations)
    ).astype(int)

    # Ensure total sales are never negative
    total[total < 0] = 0

    # Generate sales counts
    counts = rng.multinomial(total, rates)

    # Convert to DataFrame
    data = pd.DataFrame(counts)
    data.columns = market_share_labels
    data.columns.name = "product"
    data.index.name = "day"
    data["pre"] = data.index < treatment_time
    return data


def generate_unconstrained_data(
    total_sales_before: list[int],
    total_sales_after: list[int],
    total_sales_sigma: float,
    treatment_time: int,
    n_observations: int,
    market_shares_before: list[list[float]],
    market_shares_after: list[list[float]],
    market_share_labels: list[str],
    rng: np.random.Generator,
):
    """This function generates synthetic data for the MVITS model. Notably, we can
    define different total sales levels before and after the introduction of the new
    model"""

    rates = np.array(
        treatment_time * market_shares_before
        + (n_observations - treatment_time) * market_shares_after
    )

    total_sales_mu = np.array(
        treatment_time * total_sales_before
        + (n_observations - treatment_time) * total_sales_after
    )

    total = (
        rng.normal(loc=total_sales_mu, scale=total_sales_sigma, size=n_observations)
    ).astype(int)

    # Ensure total sales are never negative
    total[total < 0] = 0

    # Generate sales counts
    counts = rng.multinomial(total, rates)

    # Convert to DataFrame
    data = pd.DataFrame(counts)
    data.columns = pd.Index(market_share_labels)
    data.columns.name = "product"
    data.index.name = "day"
    data["pre"] = data.index < treatment_time
    return data
