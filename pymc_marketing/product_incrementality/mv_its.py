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
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pymc as pm

HDI_ALPHA = 0.5


class MVITS:
    """Multivariate Interrupted Time Series class.

    Class to perform a multivariate interrupted time series analysis with the
    specific intent of determining where the sales of a new product came from.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time,
        existing_sales: list[str],
        treatment_sales: str,
        market_saturated: bool = True,
        rng=42,
        sample_kwargs: dict | None = None,
    ):
        self.data = data
        self.treatment_time = treatment_time
        self.existing_sales = existing_sales
        self.treatment_sales = treatment_sales
        self.rng = rng
        self.sample_kwargs = sample_kwargs if sample_kwargs is not None else {}
        self.market_saturated = market_saturated

        self.model = self.build_model(
            self.data[self.existing_sales],
            self.data[self.treatment_sales],
            self.market_saturated,
            treatment_time=self.treatment_time,
        )
        self.sample_prior_predictive()
        self.fit()
        self.sample_posterior_predictive()
        self.calculate_counterfactual()
        return

    @staticmethod
    def build_model(
        existing_sales: pd.DataFrame,
        treatment_sales: pd.Series,
        market_saturated: bool,
        treatment_time,
        *,
        alpha_background=0.5,
    ):
        """Return a PyMC model for a multivariate interrupted time series analysis."""
        if not existing_sales.index.equals(treatment_sales.index):
            raise ValueError(  # pragma: no cover
                "Index of existing_sales and treatment_sales must match."
            )

        # note: type hints for coords required for mypi to not get confused
        coords: dict[str, list[str]] = {
            "background_product": list(existing_sales.columns),
            "time": list(existing_sales.index.values),
            "all_sources": [
                *list(existing_sales.columns),
                "new",
            ],
        }

        with pm.Model(coords=coords) as model:
            # data
            _existing_sales = pm.Data(
                "existing_sales",
                existing_sales.values,
                dims=("time", "background_product"),
            )
            treatment_sales = pm.Data(
                "treatment_sales", treatment_sales.values, dims=("time",)
            )

            # priors
            intercept = pm.Normal(
                "intercept",
                mu=pm.math.mean(existing_sales[:treatment_time], axis=0),
                sigma=np.std(existing_sales[:treatment_time], axis=0),
                dims="background_product",
            )

            sigma = pm.HalfNormal(
                "background_product_sigma",
                sigma=pm.math.mean(existing_sales.std().values),
                dims="background_product",
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
                intercept[None, :] - treatment_sales[:, None] * beta[None, :],
                dims=("time", "background_product"),
            )

            # likelihood
            normal_dist = pm.Normal.dist(mu=mu, sigma=sigma)
            pm.Truncated(
                "y",
                normal_dist,
                lower=0,
                observed=_existing_sales,
                dims=("time", "background_product"),
            )

        return model

    def sample_prior_predictive(self):
        """Sample from the prior predictive distribution."""
        with self.model:
            self.idata = pm.sample_prior_predictive(random_seed=self.rng)

    def fit(self):
        """Fit the model to the data."""
        with self.model:
            self.idata.extend(pm.sample(**self.sample_kwargs, random_seed=self.rng))

    def sample_posterior_predictive(self):
        """Sample from the posterior predictive distribution."""
        with self.model:
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata,
                    var_names=["mu", "y"],
                    random_seed=self.rng,
                )
            )

    def calculate_counterfactual(self):
        """Calculate the counterfactual scenario of never releasing the new product."""
        zero_sales = np.zeros(self.data[self.treatment_sales].shape, dtype=np.int32)
        self.counterfactual_model = pm.do(self.model, {"treatment_sales": zero_sales})
        with self.counterfactual_model:
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata,
                    var_names=["mu", "y"],
                    random_seed=self.rng,
                    predictions=True,
                )
            )

    def causal_impact(self, variable="mu"):
        """Calculate the causal impact of the new product on the background products.

        Note: if we compare "mu" then we are comparing the expected sales, if we compare
        "y" then we are comparing the actual sales
        """
        if variable not in ["mu", "y"]:
            raise ValueError(
                f"variable must be either 'mu' or 'y', not {variable}"
            )  # pragma: no cover

        return (
            self.idata.posterior_predictive[variable] - self.idata.predictions[variable]
        )

    def plot_fit(self, variable="mu"):
        """Plot the model fit (posterior predictive) of the background products."""
        if variable not in ["mu", "y"]:
            raise ValueError(
                f"variable must be either 'mu' or 'y', not {variable}"
            )  # pragma: no cover

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
        """Plot counterfactual scenario.

        Plot the predicted sales of the background products under the counterfactual
        scenario of never releasing the new product.
        """
        fig, ax = plt.subplots()

        if variable not in ["mu", "y"]:
            raise ValueError(
                f"variable must be either 'mu' or 'y', not {variable}"
            )  # pragma: no cover

        # plot data
        self.plot_data(self.data, ax)

        # plot posterior predictive distribution of sales for each of the background products
        x = self.data.index.values
        background_products = list(self.idata.observed_data.background_product.data)
        for i, background_product in enumerate(background_products):
            az.plot_hdi(
                x,
                self.idata.predictions[variable]
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

    def plot_causal_impact_sales(self, variable="mu"):
        """Plot causal impact of sales.

        Plot the inferred causal impact of the new product on the sales of the
        background products.

        Note: if we compare "mu" then we are comparing the expected sales, if we compare
        "y" then we are comparing the actual sales
        """
        fig, ax = plt.subplots()

        # plot posterior predictive distribution of sales for each of the background products
        x = self.data.index.values
        background_products = list(self.idata.observed_data.background_product.data)

        for i, background_product in enumerate(background_products):
            az.plot_hdi(
                x,
                self.causal_impact(variable=variable)
                .transpose(..., "time")
                .sel(background_product=background_product),
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

    def plot_causal_impact_market_share(self, variable="mu"):
        """Plot the inferred causal impact of the new product on the background products.

        Note: if we compare "mu" then we are comparing the expected sales, if we compare
        "y" then we are comparing the actual sales
        """
        fig, ax = plt.subplots()

        # plot posterior predictive distribution of sales for each of the background products
        x = self.data.index.values
        background_products = list(self.idata.observed_data.background_product.data)

        # divide the causal impact change in sales by the counterfactual predicted sales
        variable = "mu"
        for i, background_product in enumerate(background_products):
            causal_impact = (
                self.causal_impact(variable=variable)
                .transpose(..., "time")
                .sel(background_product=background_product)
            )
            total_sales = (
                self.idata.predictions[variable]
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


def generate_saturated_data(
    total_sales_mu: int,
    total_sales_sigma: float,
    treatment_time: int,
    n_observations: int,
    market_shares_before,
    market_shares_after,
    market_share_labels,
    rng: np.random.Generator,
):
    """Generate synthetic data for the MVITS model, assuming market is saturated."""
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


def generate_unsaturated_data(
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
    """Generate synthetic data for the MVITS model.

    Notably, we can define different total sales levels before and after the
    introduction of the new model.
    """
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
