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

"""Beta-Geometric/Beta-Binomial Model."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray
from numpy import exp
from pymc.util import RandomState
from pytensor.graph import vectorize_graph
from scipy.special import betaln, gammaln

from pymc_marketing.clv.distributions import BetaGeoBetaBinom
from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.clv.utils import to_xarray
from pymc_marketing.model_config import ModelConfig
from pymc_marketing.prior import Prior


class BetaGeoBetaBinomModel(CLVModel):
    """Beta-Geometric/Beta-Binomial Model (BG/BB).

    CLV model for non-contractual, discrete purchase opportunities, introduced by Fadel et al. [1]_.

    The BG/BB model assumes the probability a customer will become inactive follows a Beta distribution,
    and the probability of making a purchase is also Beta-distributed while customers are still active.

    This model requires data to be summarized by *recency*, *frequency*, and *T* for each customer.
    *T* should be the same value across all customers.

    Parameters
    ----------
    data : ~pandas.DataFrame
        DataFrame containing the following columns:

        * `customer_id`: Unique customer identifier
        * `frequency`: Number of repeat purchases
        * `recency`: Purchase opportunities between the first and the last purchase
        * `T`: Total purchase opportunities.
          Model assumptions require *T >= recency* and all customers share the same value for *T.

    model_config : dict, optional
        Dictionary containing model parameters:

        * `alpha`: Shape parameter of purchase process; defaults to `phi_purchase` * `kappa_purchase`
        * `beta`: Shape parameter of purchase process; defaults to `1-phi_purchase` * `kappa_purchase`
        * `gamma`: Shape parameter of dropout process; defaults to `phi_purchase` * `kappa_purchase`
        * `delta`: Shape parameter of dropout process; defaults to `1-phi_dropout` * `kappa_dropout`
        * `phi_purchase`: Nested prior for alpha and beta priors; defaults to `Prior("Uniform", lower=0, upper=1)`
        * `kappa_purchase`: Nested prior for alpha and beta priors; defaults to `Prior("Pareto", alpha=1, m=1)`
        * `phi_dropout`: Nested prior for gamma and delta priors; defaults to `Prior("Uniform", lower=0, upper=1)`
        * `kappa_dropout`: Nested prior for gamma and delta priors; defaults to `Prior("Pareto", alpha=1, m=1)`

        If not provided, the model will use default priors specified in the `default_model_config` class attribute.
    sampler_config : dict, optional
        Dictionary of sampler parameters. Defaults to None.

    Examples
    --------

    .. code-block:: python

        import pymc as pm

        from pymc_marketing.prior import Prior
        from pymc_marketing.clv import BetaGeoBetaBinomModel

        rfm_df = rfm_summary(raw_data, "id_col_name", "date_col_name")

        # Initialize model with customer data; `model_config` parameter is optional
        model = BetaGeoBetaBinomModel(
            data=rfm_df,
            model_config={
                "alpha": Prior("HalfFlat"),
                "beta": Prior("HalfFlat"),
                "gamma": Prior("HalfFlat"),
                "delta": Prior("HalfFlat"),
            },
        )

        # Fit model quickly to large datasets via Maximum a Posteriori
        model.fit(fit_method="map")
        print(model.fit_summary())

        # Fit with the default 'mcmc' for more informative predictions and reliable performance on smaller datasets
        model.fit(fit_method="mcmc")
        print(model.fit_summary())

        # Predict number of purchases for customers over the next 10 time periods
        expected_purchases = model.expected_purchases(
            data=rfm_df,
            future_t=10,
        )

        # Predict probability of customer making 'n' purchases over 't' time periods
        # Data parameter is omitted here because predictions are ran on original dataset
        expected_num_purchases = model.expected_purchase_probability(
            n=[0, 1, 2, 3],
            future_t=[10, 20, 30, 40],
        )

        new_data = pd.DataFrame(
            data={
                "customer_id": [0, 1, 2, 3],
                "frequency": [5, 2, 1, 8],
                "recency": [7, 4, 2.5, 11],
                "T": [10, 8, 10, 22],
            }
        )

        # Predict probability customers will still be active in 'future_t' time periods
        probability_alive = model.expected_probability_alive(
            data=new_data,
            future_t=[0, 3, 6, 9],
        )

        # Predict number of purchases for a new customer over 't' time periods.
        expected_purchases_new_customer = model.expected_purchases_new_customer(
            t=[2, 5, 7, 10],
        )

    References
    ----------
    .. [1] Peter Fader, Bruce Hardie, and Jen Shang.
           "Customer-Base Analysis in a Discrete-Time Noncontractual Setting".
           Marketing Science, Vol. 29, No. 6 (Nov-Dec, 2010), pp. 1086-1108.
           https://www.brucehardie.com/papers/020/fader_et_al_mksc_10.pdf
    """

    _model_type = "BG/BB"  # Beta-Geometric, Beta-Binomial Distribution

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        model_config: ModelConfig | None = None,
        sampler_config: dict | None = None,
    ):
        super().__init__(
            data=data,
            model_config=model_config,
            sampler_config=sampler_config,
            non_distributions=None,
        )
        self._validate_cols(
            data,
            required_cols=[
                "customer_id",
                "frequency",
                "recency",
                "T",
            ],
            must_be_unique=["customer_id"],
            must_be_homogenous=["T"],
        )

    @property
    def default_model_config(self) -> ModelConfig:
        """Default model configuration."""
        return {
            "phi_purchase": Prior("Uniform", lower=0, upper=1),
            "kappa_purchase": Prior("Pareto", alpha=1, m=1),
            "phi_dropout": Prior("Uniform", lower=0, upper=1),
            "kappa_dropout": Prior("Pareto", alpha=1, m=1),
        }

    def build_model(self) -> None:  # type: ignore[override]
        """Build the model."""
        coords = {
            "obs_var": ["recency", "frequency"],
            "customer_id": self.data["customer_id"],
        }
        with pm.Model(coords=coords) as self.model:
            # purchase rate priors
            if "alpha" in self.model_config and "beta" in self.model_config:
                alpha = self.model_config["alpha"].create_variable("alpha")
                beta = self.model_config["beta"].create_variable("beta")
            else:
                # hierarchical pooling of purchase rate priors
                phi_purchase = self.model_config["phi_purchase"].create_variable(
                    "phi_purchase"
                )
                kappa_purchase = self.model_config["kappa_purchase"].create_variable(
                    "kappa_purchase"
                )

                alpha = pm.Deterministic("alpha", phi_purchase * kappa_purchase)
                beta = pm.Deterministic("beta", (1.0 - phi_purchase) * kappa_purchase)

            # dropout priors
            if "gamma" in self.model_config and "delta" in self.model_config:
                gamma = self.model_config["gamma"].create_variable("gamma")
                delta = self.model_config["delta"].create_variable("delta")
            else:
                # hierarchical pooling of dropout rate priors
                phi_dropout = self.model_config["phi_dropout"].create_variable(
                    "phi_dropout"
                )
                kappa_dropout = self.model_config["kappa_dropout"].create_variable(
                    "kappa_dropout"
                )

                gamma = pm.Deterministic("gamma", phi_dropout * kappa_dropout)
                delta = pm.Deterministic("delta", (1.0 - phi_dropout) * kappa_dropout)

            BetaGeoBetaBinom(
                name="recency_frequency",
                alpha=alpha,
                beta=beta,
                delta=delta,
                gamma=gamma,
                T=self.data["T"],
                observed=np.stack(
                    (self.data["recency"], self.data["frequency"]), axis=1
                ),
                dims=["customer_id", "obs_var"],
            )

    # TODO: cache this as a property
    @staticmethod
    def _logp(
        alpha: xarray.DataArray,
        beta: xarray.DataArray,
        gamma: xarray.DataArray,
        delta: xarray.DataArray,
        x: xarray.DataArray,
        t_x: xarray.DataArray,
        T: xarray.DataArray,
    ) -> xarray.DataArray:
        """Log-likelihood of the BG/NBD model.

        Utility function for using BG/BB log-likelihood in predictive methods.
        """
        # The BetaGeoBetaBinom distribution only works with vector parameters
        # We stack the chain/draw dimensions in a long vector and use vectorize
        # to broadcast along each customer `T`
        dummy_T = pt.tensor(shape=(1,), dtype=int)
        dummy_values = pt.tensor(shape=(1, 2), dtype=int)
        bgbb_dist = BetaGeoBetaBinom.dist(
            alpha=alpha.stack(sample=("chain", "draw")).values,
            beta=beta.stack(sample=("chain", "draw")).values,
            gamma=gamma.stack(sample=("chain", "draw")).values,
            delta=delta.stack(sample=("chain", "draw")).values,
            T=dummy_T,
        )
        dummy_logp = pm.logp(bgbb_dist, dummy_values)
        values = pt.constant(np.stack((t_x.values, x.values), axis=-1))
        loglike = vectorize_graph(
            dummy_logp,
            replace={dummy_T: T.values[:, None], dummy_values: values[:, None, :]},
        ).eval()
        # Unstack chain/draw and put customer in last axis
        loglike = np.moveaxis(
            loglike.reshape((-1, alpha.sizes["chain"], alpha.sizes["draw"])), 0, -1
        )
        return xarray.DataArray(data=loglike, dims=("chain", "draw", "customer_id"))

    # TODO: move this into BaseModel
    def _extract_predictive_variables(
        self,
        data: pd.DataFrame,
        customer_varnames: Sequence[str] = (),
    ) -> xarray.Dataset:
        """
        Extract predictive variables from the data.

        Utility function assigning default customer arguments
        for predictive methods and converting to xarrays.
        """
        self._validate_cols(
            data,
            required_cols=[
                "customer_id",
                *customer_varnames,
            ],
            must_be_unique=["customer_id"],
            must_be_homogenous=["T"],
        )

        alpha = self.fit_result["alpha"]
        beta = self.fit_result["beta"]
        gamma = self.fit_result["gamma"]
        delta = self.fit_result["delta"]

        customer_vars = to_xarray(
            data["customer_id"],
            *[data[customer_varname] for customer_varname in customer_varnames],
        )
        if len(customer_varnames) == 1:
            customer_vars = [customer_vars]

        return xarray.combine_by_coords(
            (
                alpha,
                beta,
                gamma,
                delta,
                *customer_vars,
            )
        )

    def expected_purchases(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        """
        Predict expected number of future purchases.

        Given *recency*, *frequency*, and *T* for an individual customer, this method estimates the
        expected number of future purchases across *future_t* time periods.

        Adapted from equation (13) in Bruce Hardie's notes [1]_, and `lifetimes` library:
        https://github.com/CamDavidsonPilon/lifetimes/blob/master/lifetimes/fitters/beta_geo_beta_binom_fitter.py#L179

        Parameters
        ----------
        data : ~pandas.DataFrame, optional
        Dataframe containing the following columns:

        * `customer_id`: Unique customer identifier
        * `frequency`: Number of repeat purchases
        * `recency`: Purchase opportunities between the first and the last purchase
        * `T`: Total purchase opportunities.
          Model assumptions require *T >= recency* and all customers share the same value for *T.
        * `future_t`: Optional column for *future_t* parametrization.

        If not provided, predictions will be ran with data used to fit model.
        future_t : array_like
            Number of time periods to predict expected purchases.
            Not required if `data` Dataframe contains a *future_t* column.

        References
        ----------
        .. [1] Peter Fader, Bruce Hardie, and Jen Shang.
               "Customer-Base Analysis in a Discrete-Time Noncontractual Setting".
               Marketing Science, Vol. 29, No. 6 (Nov-Dec, 2010), pp. 1086-1108.
               https://www.brucehardie.com/papers/020/fader_et_al_mksc_10.pdf
        """
        if data is None:
            data = self.data

        if future_t is not None:
            data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["frequency", "recency", "T", "future_t"]
        )
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        gamma = dataset["gamma"]
        delta = dataset["delta"]
        x = dataset["frequency"]
        t_x = dataset["recency"]
        T = dataset["T"]
        future_t = dataset["future_t"]

        loglike = self._logp(alpha, beta, gamma, delta, x, t_x, T)

        first_term = 1 / exp(loglike)
        second_term = exp(betaln(alpha + x + 1, beta + T - x) - betaln(alpha, beta))
        third_term = (
            delta / (gamma - 1) * exp(gammaln(gamma + delta) - gammaln(1 + delta))
        )
        fourth_term = exp(gammaln(1 + delta + T) - gammaln(gamma + delta + T))
        fifth_term = exp(
            gammaln(1 + delta + T + future_t) - gammaln(gamma + delta + T + future_t)
        )

        exp_purchases = (
            first_term * second_term * third_term * (fourth_term - fifth_term)
        )

        return exp_purchases.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_probability_alive(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        """
        Predict expected probability of being alive.

        Estimate the probability that a customer with history *frequency*, *recency*, and *T*
        is currently active. Can also estimate alive probability for *future_t* periods into the future.

        Adapted from equation (11) in Bruce Hardie's notes [1]_ and lifetimes library:
        https://github.com/CamDavidsonPilon/lifetimes/blob/master/lifetimes/fitters/beta_geo_beta_binom_fitter.py#L217

        Parameters
        ----------
        data : ~pandas.DataFrame, optional
            Dataframe containing the following columns:

            * `customer_id`: Unique customer identifier
            * `frequency`: Number of repeat purchases
            * `recency`: Purchase opportunities between the first and the last purchase
            * `T`: Total purchase opportunities.
              Model assumptions require *T >= recency* and all customers share the same value for *T.
            * `future_t`: Optional column for *future_t* parametrization.

            If not provided, predictions will be ran with data used to fit model.
        future_t : array_like
            Number of time periods to predict expected purchases.
            Not required if `data` Dataframe contains a *future_t* column.

        References
        ----------
        .. [1] Peter Fader, Bruce Hardie, and Jen Shang.
               "Customer-Base Analysis in a Discrete-Time Noncontractual Setting".
               Marketing Science, Vol. 29, No. 6 (Nov-Dec, 2010), pp. 1086-1108.
               https://www.brucehardie.com/papers/020/fader_et_al_mksc_10.pdf
        """
        if data is None:
            data = self.data

        if future_t is not None:
            data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["frequency", "recency", "T", "future_t"]
        )
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        gamma = dataset["gamma"]
        delta = dataset["delta"]
        x = dataset["frequency"]
        t_x = dataset["recency"]
        T = dataset["T"]
        future_t = dataset["future_t"]

        loglike = self._logp(alpha, beta, gamma, delta, x, t_x, T)

        term1 = betaln(alpha + x, beta + T - x) - betaln(alpha, beta)
        term2 = betaln(gamma, delta + T + future_t) - betaln(gamma, delta)

        prob_alive = exp(term1 + term2) / exp(loglike)

        return prob_alive.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_purchases_new_customer(
        self,
        data: pd.DataFrame | None = None,
        *,
        t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        """Predict the expected number of purchases for a new customer across *t* time periods.

        Adapted from equation (8) in Bruce Hardie's notes [1]:

        Parameters
        ----------
        t : array_like
            Number of time periods over which to estimate purchases.

        References
        ----------
        .. [1] Peter Fader, Bruce Hardie, and Jen Shang.
               "Customer-Base Analysis in a Discrete-Time Noncontractual Setting".
               Marketing Science, Vol. 29, No. 6 (Nov-Dec, 2010), pp. 1086-1108.
               https://www.brucehardie.com/papers/020/fader_et_al_mksc_10.pdf
        """
        if data is None:
            data = self.data

        if t is not None:
            data = data.assign(t=t)

        dataset = self._extract_predictive_variables(data, customer_varnames=["t"])
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        gamma = dataset["gamma"]
        delta = dataset["delta"]
        t = dataset["t"]

        first_term = alpha / (alpha + beta) * delta / (gamma - 1)
        second_term = 1 - exp(
            gammaln(gamma + delta)
            + gammaln(1 + delta + t)
            - gammaln(gamma + delta + t)
            - gammaln(1 + delta)
        )

        return (first_term * second_term).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def _distribution_new_customers(
        self,
        data: pd.DataFrame | None = None,
        *,
        T: int | np.ndarray | pd.Series | None = None,
        random_seed: RandomState | None = None,
        var_names: Sequence[
            Literal["dropout", "purchase_rate", "recency_frequency"]
        ] = (
            "dropout",
            "purchase_rate",
            "recency_frequency",
        ),
        n_samples: int = 1000,
    ) -> xarray.Dataset:
        """Compute posterior predictive samples of dropout, purchase rate and frequency/recency of new customers.

        Parameters
        ----------
        data : ~pandas.DataFrame, Optional
            DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier
            * `T`: Total number of purchase opportunities

            If not provided, predictions will be ran with data used to fit model.
        T : array_like, optional
            Total number of purchase opportunities. Not needed if `data` parameter is provided with a `T` column.
        random_seed : ~numpy.random.RandomState, optional
            Random state to use for sampling.
        var_names : sequence of str, optional
            Names of the variables to sample from. Defaults to ["dropout", "purchase_rate", "recency_frequency"].
        n_samples : int, optional
            Number of posterior predictive samples to generate. Defaults to 1000

        """
        if data is None:
            data = self.data

        if T is not None:
            data = data.assign(T=T)

        dataset = self._extract_predictive_variables(data, customer_varnames=["T"])
        T = dataset["T"].values
        # Delete "T" so we can pass dataset directly to `sample_posterior_predictive`
        del dataset["T"]

        if dataset.sizes["chain"] == 1 and dataset.sizes["draw"] == 1:
            # For map fit add a dummy draw dimension
            dataset = dataset.squeeze("draw").expand_dims(draw=range(n_samples))

        coords = self.model.coords.copy()  # type: ignore
        coords["customer_id"] = data["customer_id"]
        with pm.Model(coords=coords):
            alpha = pm.Flat("alpha")
            beta = pm.Flat("beta")
            gamma = pm.Flat("gamma")
            delta = pm.Flat("delta")

            pm.Beta(
                "purchase_rate",
                alpha=alpha,
                beta=beta,
            )
            pm.Beta(
                "dropout",
                alpha=gamma,
                beta=delta,
            )

            BetaGeoBetaBinom(
                name="recency_frequency",
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                T=T,
                dims=["customer_id", "obs_var"],
            )

            return pm.sample_posterior_predictive(
                dataset,
                var_names=var_names,
                random_seed=random_seed,
                predictions=True,
            ).predictions

    # TODO: Move into BaseModel?
    def distribution_new_customer_dropout(
        self,
        data: pd.DataFrame | None = None,
        *,
        random_seed: RandomState | None = None,
    ) -> xarray.Dataset:
        """Sample from the Beta distribution representing dropout probabilities for new customers.

        This is the probability a new customer will drop out after making a purchase.

        Parameters
        ----------
        data : ~pandas.DataFrame, optional
            DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier

            If not provided, predictions will be ran with data used to fit model.
        random_seed : ~numpy.random.RandomState, optional
            Random state to use for sampling.

        Returns
        -------
        ~xarray.Dataset
            Dataset containing the posterior samples for the population-level dropout rate.
        """
        return self._distribution_new_customers(
            data=data,
            random_seed=random_seed,
            var_names=["dropout"],
        )["dropout"]

    # TODO: Move into BaseModel?
    def distribution_new_customer_purchase_rate(
        self,
        data: pd.DataFrame | None = None,
        *,
        random_seed: RandomState | None = None,
    ) -> xarray.Dataset:
        """Sample from the Beta distribution representing purchase probabilities for new customers.

        This is the probability a new customer will make a purchase given they are still active.

        Parameters
        ----------
        data : ~pandas.DataFrame, optional
            DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier

            If not provided, predictions will be ran with data used to fit model.
        random_seed : ~numpy.random.RandomState, optional
            Random state to use for sampling.

        Returns
        -------
        ~xarray.Dataset
            Dataset containing the posterior samples for the population-level purchase rate.
        """
        return self._distribution_new_customers(
            data=data,
            random_seed=random_seed,
            var_names=["purchase_rate"],
        )["purchase_rate"]

    # TODO: This is quite slow due to sampling far more than necessary. Same for other CLV models.
    def distribution_new_customer_recency_frequency(
        self,
        data: pd.DataFrame | None = None,
        *,
        T: int | np.ndarray | pd.Series | None = None,
        random_seed: RandomState | None = None,
        n_samples: int = 1,
    ) -> xarray.Dataset:
        """BG/BB process representing purchases across the customer population.

        This is the distribution of purchase frequencies given 'T' observation periods for each customer.

        Parameters
        ----------
        data : ~pandas.DataFrame, optional
            DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier
            * `T`: Total purchase opportunities.

            If not provided, the method will use the fit dataset.
        T : array_like, optional
            Total purchase opportunities. If not provided, T values from fit dataset will be used.
            Not required if `data` Dataframe contains a `T` column.
        random_seed : ~numpy.random.RandomState, optional
            Random state to use for sampling.
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        ~xarray.Dataset
            Dataset containing the posterior samples for the customer population.
        """
        return self._distribution_new_customers(
            data=data,
            T=T,
            random_seed=random_seed,
            var_names=["recency_frequency"],
            n_samples=n_samples,
        )["recency_frequency"]
