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
"""Modified Beta-Geometric Negative Binomial Distribution (MBG/NBD) model for a non-contractual customer population across continuous time."""  # noqa: E501

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import pymc as pm
import xarray
from pymc.util import RandomState
from scipy.special import hyp2f1

from pymc_marketing.clv.distributions import ModifiedBetaGeoNBD
from pymc_marketing.clv.models import BetaGeoModel


class ModifiedBetaGeoModel(BetaGeoModel):
    r"""Modified Beta-Geometric Negative Binomial Distribution (MBG/NBD) model for a non-contractual customer population across continuous time.

    Based on proposed modifications to the BG/NBD model by Battislam, et al. in [1]_, and Wagner & Hoppe in[2]_,
    which remove the BG/NBD assumption that all non-repeat customers are still active.

    The MBG/NBD model assumes dropout probabilities for the customer population are Beta distributed,
    and time between transactions follows a Gamma distribution while the customer is still active.

    This model requires data to be summarized by *recency*, *frequency*, and *T* for each customer,
    using `clv.utils.rfm_summary()` or equivalent. Modeling assumptions require *T >= recency*.

    Predictive methods have been adapted from the *ModifiedBetaGeoFitter* class in the legacy *lifetimes* library
    (see https://github.com/CamDavidsonPilon/lifetimes/).

    Parameters
    ----------
    data : ~pandas.DataFrame
        DataFrame containing the following columns:
            * `customer_id`: Unique customer identifier
            * `frequency`: Number of repeat purchases
            * `recency`: Time between the first and the last purchase
            * `T`: Time between the first purchase and the end of the observation period
    model_config : dict, optional
        Dictionary of model prior parameters:
            * `alpha`: Scale parameter for time between purchases; defaults to `Prior("HalfFlat")`
            * `r`: Shape parameter for time between purchases; defaults to `Prior("HalfFlat")`
            * `a`: Shape parameter of dropout process; defaults to `phi_purchase` * `kappa_purchase`
            * `b`: Shape parameter of dropout process; defaults to `1-phi_dropout` * `kappa_dropout`
            * `phi_dropout`: Nested prior for a and b priors; defaults to `Prior("Uniform", lower=0, upper=1)`
            * `kappa_dropout`: Nested prior for a and b priors; defaults to `Prior("Pareto", alpha=1, m=1)`
            * `purchase_covariates`: Coefficients for purchase rate covariates; defaults to `Normal(0, 1)`
            * `dropout_covariates`: Coefficients for dropout covariates; defaults to `Normal.dist(0, 1)`
            * `purchase_covariate_cols`: List containing column names of covariates for customer purchase rates.
            * `dropout_covariate_cols`: List containing column names of covariates for customer dropouts.
    sampler_config : dict, optional
        Dictionary of sampler parameters. Defaults to *None*.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.prior import Prior
        from pymc_marketing.clv import ModifiedBetaGeoModel, rfm_summary

        # customer identifiers and purchase datetimes
        # are all that's needed to start modeling
        data = [
            [1, "2024-01-01"],
            [1, "2024-02-06"],
            [2, "2024-01-01"],
            [3, "2024-01-02"],
            [3, "2024-01-05"],
            [4, "2024-01-16"],
            [4, "2024-02-05"],
            [5, "2024-01-17"],
            [5, "2024-01-18"],
            [5, "2024-01-19"],
        ]
        raw_data = pd.DataFrame(data, columns=["id", "date"]

        # preprocess data
        rfm_df = rfm_summary(raw_data,'id','date')

        # model_config and sampler_configs are optional
        model = ModifiedBetaGeoModel(
            data=data,
            model_config={
                "r": Prior("HalfFlat"),
                "alpha": Prior("HalfFlat"),
                "a": Prior("HalfFlat"),
                "b": Prior("HalfFlat),
            },
            sampler_config={
                "draws": 1000,
                "tune": 1000,
                "chains": 2,
                "cores": 2,
            },
        )

        # The default 'mcmc' fit_method provides informative predictions
        # and reliable performance on small datasets
        model.fit()
        print(model.fit_summary())

        # Maximum a Posteriori can quickly fit a model to large datasets,
        # but will give limited insights into predictive uncertainty.
        model.fit(fit_method='map')
        print(model.fit_summary())

        # Predict number of purchases for current customers
        # over the next 10 time periods
        expected_purchases = model.expected_purchases(future_t=10)

        # Predict probability customers are still active
        probability_alive = model.expected_probability_alive()

        # Predict number of purchases for a new customer over 't' time periods
        expected_purchases_new_customer = model.expected_purchases_new_customer(t=10)

    References
    ----------
    .. [1] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
       "Empirical validation and comparison of models for customer base
       analysis." International Journal of Research in Marketing, 24 (3), 201-209.
       https://works.bepress.com/meltem-denizel/2/download/
    .. [2] Wagner, U. and Hoppe D. (2008), "Erratum on the MBG/NBD Model,"
       International Journal of Research in Marketing, 25 (3), 225-226.
        https://www.researchgate.net/profile/Udo-Wagner/publication/274894157_Customer_Base_Analysis_The_Case_for_a_Central_Variant_of_the_BetageometricBND_Model/links/55c3728608aeca747d5f6658/Customer-Base-Analysis-The-Case-for-a-Central-Variant-of-the-Betageometric-BND-Model.pdf
    """  # noqa: E501

    _model_type = "MBG/NBD"

    def build_model(self) -> None:  # type: ignore[override]
        """Build the model."""
        coords = {
            "purchase_covariate": self.purchase_covariate_cols,
            "dropout_covariate": self.dropout_covariate_cols,
            "customer_id": self.data["customer_id"],
            "obs_var": ["recency", "frequency"],
        }
        with pm.Model(coords=coords) as self.model:
            # purchase rate priors
            if self.purchase_covariate_cols:
                purchase_data = pm.Data(
                    "purchase_data",
                    self.data[self.purchase_covariate_cols],
                    dims=["customer_id", "purchase_covariate"],
                )
                self.model_config["purchase_coefficient"].dims = "purchase_covariate"
                purchase_coefficient_alpha = self.model_config[
                    "purchase_coefficient"
                ].create_variable("purchase_coefficient_alpha")

                alpha_scale = self.model_config["alpha"].create_variable("alpha_scale")
                alpha = pm.Deterministic(
                    "alpha",
                    (
                        alpha_scale
                        * pm.math.exp(
                            -pm.math.dot(purchase_data, purchase_coefficient_alpha)
                        )
                    ),
                    dims="customer_id",
                )
            else:
                alpha = self.model_config["alpha"].create_variable("alpha")

            # dropout priors
            if "a" in self.model_config and "b" in self.model_config:
                if self.dropout_covariate_cols:
                    dropout_data = pm.Data(
                        "dropout_data",
                        self.data[self.dropout_covariate_cols],
                        dims=["customer_id", "dropout_covariate"],
                    )

                    self.model_config["dropout_coefficient"].dims = "dropout_covariate"
                    dropout_coefficient_a = self.model_config[
                        "dropout_coefficient"
                    ].create_variable("dropout_coefficient_a")
                    dropout_coefficient_b = self.model_config[
                        "dropout_coefficient"
                    ].create_variable("dropout_coefficient_b")

                    a_scale = self.model_config["a"].create_variable("a_scale")
                    b_scale = self.model_config["b"].create_variable("b_scale")
                    a = pm.Deterministic(
                        "a",
                        a_scale
                        * pm.math.exp(pm.math.dot(dropout_data, dropout_coefficient_a)),
                        dims="customer_id",
                    )
                    b = pm.Deterministic(
                        "b",
                        b_scale
                        * pm.math.exp(pm.math.dot(dropout_data, dropout_coefficient_b)),
                        dims="customer_id",
                    )
                else:
                    a = self.model_config["a"].create_variable("a")
                    b = self.model_config["b"].create_variable("b")
            else:
                # hierarchical pooling of dropout rate priors
                if self.dropout_covariate_cols:
                    dropout_data = pm.Data(
                        "dropout_data",
                        self.data[self.dropout_covariate_cols],
                        dims=["customer_id", "dropout_covariate"],
                    )

                    self.model_config["dropout_coefficient"].dims = "dropout_covariate"
                    dropout_coefficient_a = self.model_config[
                        "dropout_coefficient"
                    ].create_variable("dropout_coefficient_a")
                    dropout_coefficient_b = self.model_config[
                        "dropout_coefficient"
                    ].create_variable("dropout_coefficient_b")

                    phi_dropout = self.model_config["phi_dropout"].create_variable(
                        "phi_dropout"
                    )
                    kappa_dropout = self.model_config["kappa_dropout"].create_variable(
                        "kappa_dropout"
                    )

                    a_scale = pm.Deterministic(
                        "a_scale",
                        phi_dropout * kappa_dropout,
                    )
                    b_scale = pm.Deterministic(
                        "b_scale",
                        (1.0 - phi_dropout) * kappa_dropout,
                    )

                    a = pm.Deterministic(
                        "a",
                        a_scale
                        * pm.math.exp(pm.math.dot(dropout_data, dropout_coefficient_a)),
                        dims="customer_id",
                    )
                    b = pm.Deterministic(
                        "b",
                        b_scale
                        * pm.math.exp(pm.math.dot(dropout_data, dropout_coefficient_b)),
                        dims="customer_id",
                    )

                else:
                    phi_dropout = self.model_config["phi_dropout"].create_variable(
                        "phi_dropout"
                    )
                    kappa_dropout = self.model_config["kappa_dropout"].create_variable(
                        "kappa_dropout"
                    )

                    a = pm.Deterministic("a", phi_dropout * kappa_dropout)
                    b = pm.Deterministic("b", (1.0 - phi_dropout) * kappa_dropout)

            # r remains unchanged with or without covariates
            r = self.model_config["r"].create_variable("r")

            ModifiedBetaGeoNBD(
                name="recency_frequency",
                a=a,
                b=b,
                r=r,
                alpha=alpha,
                T=self.data["T"],
                observed=np.stack(
                    (self.data["recency"], self.data["frequency"]), axis=1
                ),
                dims=["customer_id", "obs_var"],
            )

    def expected_purchases(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        r"""Compute the expected number of future purchases across *future_t* time periods given *recency*, *frequency*, and *T* for each customer.

        The *data* parameter is only required for out-of-sample customers.

        Adapted from equation (6) in [1]_, and *lifetimes* package:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/modified_beta_geo_fitter.py#L151

        Parameters
        ----------
        future_t : int, array_like
            Number of time periods to predict expected purchases.
        data : ~pandas.DataFrame
            Optional dataframe containing the following columns:

            * `customer_id`: Unique customer identifier
            * `frequency`: Number of repeat purchases
            * `recency`: Time between the first and the last purchase
            * `T`: Time between first purchase and end of observation period; model assumptions require T >= recency

        References
        ----------
        .. [1] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
        "Empirical validation and comparison of models for customer base
        analysis,"
        International Journal of Research in Marketing, 24 (3), 201-209.
        https://works.bepress.com/meltem-denizel/2/download/
        """  # noqa: E501
        if data is None:
            data = self.data

        if future_t is not None:
            data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["frequency", "recency", "T", "future_t"]
        )
        a = dataset["a"]
        b = dataset["b"]
        alpha = dataset["alpha"]
        r = dataset["r"]
        x = dataset["frequency"]
        t_x = dataset["recency"]
        T = dataset["T"]
        t = dataset["future_t"]

        hyp_term = hyp2f1(r + x, b + x + 1, a + b + x, t / (alpha + T + t))
        first_term = (a + b + x) / (a - 1)
        second_term = 1 - hyp_term * ((alpha + T) / (alpha + t + T)) ** (r + x)
        numerator = first_term * second_term
        denominator = 1 + (a / (b + x)) * ((alpha + T) / (alpha + t_x)) ** (r + x)

        return (numerator / denominator).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_purchases_new_customer(
        self,
        data: pd.DataFrame | None = None,
        *,
        t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        r"""Compute the expected number of purchases for a new customer across *t* time periods.

        Adapted from equation (4) in [1]_, and `lifetimes` library:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/modified_beta_geo_fitter.py#L130

        Parameters
        ----------
        t : array_like
            Number of time periods over which to estimate purchases.

        References
        ----------
        .. [1] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
        "Empirical validation and comparison of models for customer base
        analysis." International Journal of Research in Marketing, 24 (3), 201-209.
        https://works.bepress.com/meltem-denizel/2/download/
        """
        # TODO: This is extraneous now, but needed for future covariate support.
        if data is None:
            data = self.data

        if t is not None:
            data = data.assign(t=t)

        dataset = self._extract_predictive_variables(data, customer_varnames=["t"])
        a = dataset["a"]
        b = dataset["b"]
        alpha = dataset["alpha"]
        r = dataset["r"]
        t = dataset["t"]

        hyp_term = hyp2f1(r, b + 1, a + b, t / (alpha + t))
        first_term = b / (a - 1)
        second_term = 1 - hyp_term * (alpha / (alpha + t)) ** (r)

        return (first_term * second_term).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_probability_alive(
        self,
        data: pd.DataFrame | None = None,
    ) -> xarray.DataArray:
        r"""Compute the probability a customer with history *frequency*, *recency*, and *T* is currently active.

        The *data* parameter is only required for out-of-sample customers.

        Adapted from equation (5) in [1]_, and `lifetimes` library:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/modified_beta_geo_fitter.py#L188

        Parameters
        ----------
        data : *pandas.DataFrame
            Optional dataframe containing the following columns:

            * `customer_id`: Unique customer identifier
            * `frequency`: Number of repeat purchases
            * `recency`: Time between the first and the last purchase
            * `T`: Time between first purchase and end of observation period, model assumptions require T >= recency

        References
        ----------
        .. [1] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
        "Empirical validation and comparison of models for customer base
        analysis." International Journal of Research in Marketing, 24 (3), 201-209.
        https://works.bepress.com/meltem-denizel/2/download/
        """
        if data is None:
            data = self.data

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["frequency", "recency", "T"]
        )

        a = dataset["a"]
        b = dataset["b"]
        alpha = dataset["alpha"]
        r = dataset["r"]
        x = dataset["frequency"]
        t_x = dataset["recency"]
        T = dataset["T"]

        proba = 1.0 / (1 + (a / (b + x)) * ((alpha + T) / (alpha + t_x)) ** (r + x))

        return proba.transpose("chain", "draw", "customer_id", missing_dims="ignore")

    def expected_probability_no_purchase(
        self,
        t: int,
        data: pd.DataFrame | None = None,
    ) -> xarray.DataArray:
        r"""Probability a customer with frequency, recency, and T will have 0 purchases in the period (T, T+t].

        The MBG/NBD model does not support this method.
        """
        raise NotImplementedError("The MBG/NBD model does not support this method.")

    def distribution_new_customer(
        self,
        data: pd.DataFrame | None = None,
        *,
        T: int | np.ndarray | pd.Series | None = None,
        random_seed: RandomState | None = None,
        var_names: Sequence[
            Literal["dropout", "purchase_rate", "recency_frequency"]
        ] = ("dropout", "purchase_rate", "recency_frequency"),
        n_samples: int = 1000,
    ) -> xarray.Dataset:
        """Compute posterior predictive samples of dropout, purchase rate and frequency/recency of new customers."""
        if data is None:
            data = self.data

        if T is not None:
            dataset = data.assign(T=T)

        dataset = self._extract_predictive_variables(data, customer_varnames=["T"])
        T = dataset["T"].values  # type: ignore
        # Delete "T" so we can pass dataset directly to `sample_posterior_predictive`
        del dataset["T"]

        if dataset.sizes["chain"] == 1 and dataset.sizes["draw"] == 1:
            # For map fit add a dummy draw dimension
            dataset = dataset.squeeze("draw").expand_dims(draw=range(n_samples))  # type: ignore

        coords = self.model.coords.copy()  # type: ignore
        coords["customer_id"] = data["customer_id"]

        with pm.Model(coords=coords) as pred_model:
            if self.purchase_covariate_cols:
                alpha = pm.Flat("alpha", dims=["customer_id"])
            else:
                alpha = pm.Flat("alpha")

            if self.dropout_covariate_cols:
                a = pm.Flat("a", dims=["customer_id"])
                b = pm.Flat("b", dims=["customer_id"])
            else:
                a = pm.Flat("a")
                b = pm.Flat("b")

            r = pm.Flat("r")

            pm.Beta(
                "dropout", alpha=a, beta=b, dims=pred_model.named_vars_to_dims.get("a")
            )
            pm.Gamma(
                "purchase_rate",
                alpha=r,
                beta=alpha,
                dims=pred_model.named_vars_to_dims.get("alpha"),
            )

            ModifiedBetaGeoNBD(
                name="recency_frequency",
                a=a,
                b=b,
                r=r,
                alpha=alpha,
                T=T,
                dims=["customer_id", "obs_var"],
            )

            return pm.sample_posterior_predictive(
                dataset,
                var_names=var_names,
                random_seed=random_seed,
                predictions=True,
            ).predictions
