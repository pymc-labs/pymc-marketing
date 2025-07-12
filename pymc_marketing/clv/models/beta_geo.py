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
"""Beta-Geometric Negative Binomial Distribution (BG/NBD) model for a non-contractual customer population across continuous time."""  # noqa: E501

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import pymc as pm
import xarray
from pymc.util import RandomState
from scipy.special import betaln, expit, hyp2f1

from pymc_marketing.clv.distributions import BetaGeoNBD
from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.clv.utils import to_xarray
from pymc_marketing.model_config import ModelConfig
from pymc_marketing.prior import Prior


class BetaGeoModel(CLVModel):
    r"""Beta-Geometric Negative Binomial Distribution (BG/NBD) model for a non-contractual customer population across continuous time.

    First introduced by Fader, Hardie & Lee [1]_, with additional predictive methods
    and enhancements in [2]_,[3]_, [4]_ and [5]_

    The BG/NBD model assumes dropout probabilities for the customer population are Beta distributed,
    and time between transactions follows a Gamma distribution while the customer is still active.

    This model requires data to be summarized by *recency*, *frequency*, and *T* for each customer,
    using `clv.utils.rfm_summary()` or equivalent. Modeling assumptions require *T >= recency*.

    Predictive methods have been adapted from the *BetaGeoFitter* class in the legacy *lifetimes* library
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
            * `alpha`: Scale parameter for time between purchases; defaults to `Prior("Weibull", alpha=2, beta=10)`
            * `r`: Shape parameter for time between purchases; defaults to `Prior("Weibull", alpha=2, beta=1)`
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
        from pymc_marketing.clv import BetaGeoModel, rfm_summary

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
        model = BetaGeoModel(
            data=data,
            model_config={
                "r": Prior("Weibull", alpha=2, beta=1),
                "alpha": Prior("HalfFlat"),
                "a": Prior("Beta", alpha=2, beta=3),
                "b": Prior("Beta", alpha=3, beta=2),
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
    .. [1] Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). “Counting your customers
           the easy way: An alternative to the Pareto/NBD model." Marketing science,
           24(2), 275-284. http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf
    .. [2] Fader, P. S., Hardie, B. G., & Lee, K. L. (2008). "Computing
           P (alive) using the BG/NBD model." http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf.
    .. [3] Fader, P. S. & Hardie, B. G. (2013) "Overcoming the BG/NBD Model's #NUM!
           Error Problem." http://brucehardie.com/notes/027/bgnbd_num_error.pdf.
    .. [4] Fader, P. S. & Hardie, B. G. (2019) "A Step-by-Step Derivation of the BG/NBD
           Model." https://www.brucehardie.com/notes/039/bgnbd_derivation__2019-11-06.pdf
    .. [5] Fader, Peter & G. S. Hardie, Bruce (2007).
           "Incorporating Time-Invariant Covariates into the Pareto/NBD and BG/NBD Models".
           https://www.brucehardie.com/notes/019/time_invariant_covariates.pdf

    """  # noqa: E501

    _model_type = "BG/NBD"  # Beta-Geometric Negative Binomial Distribution

    def __init__(
        self,
        data: pd.DataFrame,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        super().__init__(
            data=data,
            model_config=model_config,
            sampler_config=sampler_config,
            non_distributions=["purchase_covariate_cols", "dropout_covariate_cols"],
        )
        self.purchase_covariate_cols = list(
            self.model_config["purchase_covariate_cols"]
        )
        self.dropout_covariate_cols = list(self.model_config["dropout_covariate_cols"])
        self.covariate_cols = self.purchase_covariate_cols + self.dropout_covariate_cols
        self._validate_cols(
            data,
            required_cols=[
                "customer_id",
                "frequency",
                "recency",
                "T",
                *self.covariate_cols,
            ],
            must_be_unique=["customer_id"],
        )

    @property
    def default_model_config(self) -> ModelConfig:
        """Default model configuration."""
        return {
            "alpha": Prior("Weibull", alpha=2, beta=10),
            "r": Prior("Weibull", alpha=2, beta=1),
            "phi_dropout": Prior("Uniform", lower=0, upper=1),
            "kappa_dropout": Prior("Pareto", alpha=1, m=1),
            "purchase_coefficient": Prior("Normal", mu=0, sigma=1),
            "dropout_coefficient": Prior("Normal", mu=0, sigma=1),
            "purchase_covariate_cols": [],
            "dropout_covariate_cols": [],
        }

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

            BetaGeoNBD(
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

    # TODO: delete this utility after API standardization is completed
    def _unload_params(self):
        trace = self.idata.posterior
        a = trace["a"]
        b = trace["b"]
        alpha = trace["alpha"]
        r = trace["r"]

        return a, b, alpha, r

    def _extract_predictive_variables(
        self,
        data: pd.DataFrame,
        customer_varnames: Sequence[str] = (),
    ) -> xarray.Dataset:
        """
        Extract predictive variables from the data.

        Utility function assigning default customer arguments for predictive methods and converting to xarrays.
        """
        self._validate_cols(
            data,
            required_cols=[
                "customer_id",
                *customer_varnames,
                *self.purchase_covariate_cols,
                *self.dropout_covariate_cols,
            ],
            must_be_unique=["customer_id"],
        )

        customer_id = data["customer_id"]
        model_coords = self.model.coords
        if self.purchase_covariate_cols:
            purchase_xarray = xarray.DataArray(
                data[self.purchase_covariate_cols],
                dims=["customer_id", "purchase_covariate"],
                coords=[customer_id, list(model_coords["purchase_covariate"])],
            )
            alpha_scale = self.fit_result["alpha_scale"]
            purchase_coefficient_alpha = self.fit_result["purchase_coefficient_alpha"]
            alpha = alpha_scale * np.exp(
                -xarray.dot(
                    purchase_xarray,
                    purchase_coefficient_alpha,
                    dim="purchase_covariate",
                )
            )
            alpha.name = "alpha"
        else:
            alpha = self.fit_result["alpha"]

        if self.dropout_covariate_cols:
            dropout_xarray = xarray.DataArray(
                data[self.dropout_covariate_cols],
                dims=["customer_id", "dropout_covariate"],
                coords=[customer_id, list(model_coords["dropout_covariate"])],
            )
            a_scale = self.fit_result["a_scale"]
            dropout_coefficient_a = self.fit_result["dropout_coefficient_a"]
            a = a_scale * np.exp(
                xarray.dot(
                    dropout_xarray, dropout_coefficient_a, dim="dropout_covariate"
                )
            )
            a.name = "a"

            dropout_coefficient_b = self.fit_result["dropout_coefficient_b"]
            b_scale = self.fit_result["b_scale"]
            b = b_scale * np.exp(
                xarray.dot(
                    dropout_xarray, dropout_coefficient_b, dim="dropout_covariate"
                )
            )
            b.name = "b"
        else:
            a = self.fit_result["a"]
            b = self.fit_result["b"]

        r = self.fit_result["r"]

        customer_vars = to_xarray(
            data["customer_id"],
            *[data[customer_varname] for customer_varname in customer_varnames],
        )
        if len(customer_varnames) == 1:
            customer_vars = [customer_vars]

        return xarray.combine_by_coords(
            (
                a,
                b,
                alpha,
                r,
                *customer_vars,
            )
        )

    def expected_purchases(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        r"""Compute the expected number of future purchases across *future_t* time periods given *recency*, *frequency*, and *T* for each customer.

        The *data* parameter is only required for out-of-sample customers.

        Adapted from equation (10) in [1]_, and *lifetimes* package:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/beta_geo_fitter.py#L201

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
        .. [1] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
            "Counting Your Customers the Easy Way: An Alternative to the
            Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
            https://www.brucehardie.com/papers/bgnbd_2004-04-20.pdf

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

        numerator = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp2f1(
            r + x,
            b + x,
            a + b + x - 1,
            t / (alpha + T + t),
        )
        numerator *= (a + b + x - 1) / (a - 1)
        denominator = 1 + (x > 0) * (a / (b + x - 1)) * (
            (alpha + T) / (alpha + t_x)
        ) ** (r + x)

        return (numerator / denominator).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_probability_alive(
        self,
        data: pd.DataFrame | None = None,
    ) -> xarray.DataArray:
        r"""Compute the probability a customer with history *frequency*, *recency*, and *T* is currently active.

        The *data* parameter is only required for out-of-sample customers.

        Adapted from page (2) in Bruce Hardie's notes [1]_, and *lifetimes* package:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/beta_geo_fitter.py#L260

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
        .. [1] Fader, P. S., Hardie, B. G., & Lee, K. L. (2008). Computing
               P (alive) using the BG/NBD model. http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf.

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

        log_div = (r + x) * np.log((alpha + T) / (alpha + t_x)) + np.log(
            a / (b + np.maximum(x, 1) - 1)
        )

        return xarray.where(x == 0, 1.0, expit(-log_div)).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_probability_no_purchase(
        self,
        t: int,
        data: pd.DataFrame | None = None,
    ) -> xarray.DataArray:
        r"""Compute the probability a customer with history frequency, recency, and T
        will have 0 purchases in the period (T, T+t].

        The data parameter is only required for out-of-sample customers.

        Adapted from Section 5.3, Equation 34 in Bruce Hardie's notes [1]_.

        Parameters
        ----------
        data : *pandas.DataFrame
            Optional dataframe containing the following columns:

            * `customer_id`: Unique customer identifier
            * `frequency`: Number of repeat purchases
            * `recency`: Time between the first and the last purchase
            * `T`: Time between first purchase and end of observation period, model assumptions require T >= recency

        t : int
            Days after T which defines the range (T, T+t].

        References
        ----------
        .. [1] Fader, P. S. & Hardie, B. G. (2019) "A Step-by-Step Derivation of the
                BG/NBD Model." https://www.brucehardie.com/notes/039/bgnbd_derivation__2019-11-06.pdf
        """  # noqa: D205
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

        E = alpha + t_x
        F = alpha + T + t
        M = alpha + T

        beta_rep = betaln(a, b + x)
        K_E = betaln(a + 1, b + x - 1) - (r + x) * np.log(E)
        K_F = beta_rep - (r + x) * np.log(F)
        K_M = beta_rep - (r + x) * np.log(M)

        K1 = np.maximum(K_E, K_F)
        K2 = np.maximum(K_E, K_M)

        numer = np.exp(K_E - K1) + np.exp(K_F - K1)
        denom = np.exp(K_E - K2) + np.exp(K_M - K2)

        prob_no_deposits = np.exp(K1 - K2) * numer / denom

        return prob_no_deposits.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_purchases_new_customer(
        self,
        data: pd.DataFrame | None = None,
        *,
        t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        r"""Compute the expected number of purchases for a new customer across *t* time periods.

        Adapted from equation (9) in [1]_, and `lifetimes` library:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/beta_geo_fitter.py#L328

        Parameters
        ----------
        t : array_like
            Number of time periods over which to estimate purchases.

        References
        ----------
        .. [1] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
            "Counting Your Customers the Easy Way: An Alternative to the
            Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
            http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf

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

        first_term = (a + b - 1) / (a - 1)
        second_term = 1 - (alpha / (alpha + t)) ** r * hyp2f1(
            r, b, a + b - 1, t / (alpha + t)
        )

        return (first_term * second_term).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

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
        """Compute posterior predictive samples of dropout, purchase rate and frequency/recency of new customers.

        In a model with covariates, if `data` is not specified, the dataset used for fitting will be used and
        a prediction will be computed for a *new customer* with each set of covariates.
        *This is not a conditional prediction for observed customers!*

        Parameters
        ----------
        data : ~pandas.DataFrame, Optional
            DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier
            * `T`: Time between the first purchase and the end of the observation period

            If not provided, predictions will be ran with data used to fit model.
        T : array_like, optional
            time between the first purchase and the end of the observation period.
            Not needed if `data` parameter is provided with a `T` column.
        random_seed : ~numpy.random.RandomState, optional
            Random state to use for sampling.
        var_names : sequence of str, optional
            Names of the variables to sample from. Defaults to ["dropout", "purchase_rate", "recency_frequency"].
        n_samples : int, optional
            Number of samples to generate. Defaults to 1000

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

            BetaGeoNBD(
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

    def distribution_new_customer_dropout(
        self,
        data: pd.DataFrame | None = None,
        *,
        random_seed: RandomState | None = None,
    ) -> xarray.Dataset:
        """Sample the Beta distribution for the population-level dropout rate.

        This is the probability that a new customer will "drop out" and make no further purchases.

        Parameters
        ----------
        random_seed : RandomState, optional
            Random state to use for sampling.

        Returns
        -------
        xarray.Dataset
            Dataset containing the posterior samples for the population-level dropout rate.

        """
        return self.distribution_new_customer(
            data=data,
            random_seed=random_seed,
            var_names=["dropout"],
        )["dropout"]

    def distribution_new_customer_purchase_rate(
        self,
        data: pd.DataFrame | None = None,
        *,
        random_seed: RandomState | None = None,
    ) -> xarray.Dataset:
        """Sample the Gamma distribution for the population-level purchase rate.

        This is the purchase rate for a new customer and determines the time between
        purchases for any new customer.

        Parameters
        ----------
        random_seed : RandomState, optional
            Random state to use for sampling.

        Returns
        -------
        xarray.Dataset
            Dataset containing the posterior samples for the population-level purchase rate.

        """
        return self.distribution_new_customer(
            data=data,
            random_seed=random_seed,
            var_names=["purchase_rate"],
        )["purchase_rate"]

    def distribution_new_customer_recency_frequency(
        self,
        data: pd.DataFrame | None = None,
        *,
        T: int | np.ndarray | pd.Series | None = None,
        random_seed: RandomState | None = None,
        n_samples: int = 1000,
    ) -> xarray.Dataset:
        """BG/NBD process representing purchases across the customer population.

        This is the distribution of purchase frequencies given 'T' observation periods for each customer.

        Parameters
        ----------
        data : ~pandas.DataFrame, optional
            DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier
            * `T`: Time between the first purchase and the end of the observation period.
            * All covariate columns specified when model was initialized.

            If not provided, the method will use the fit dataset.
        T : array_like, optional
            Number of observation periods for each customer. If not provided, T values from fit dataset will be used.
            Not required if `data` Dataframe contains a `T` column.
        random_seed : ~numpy.random.RandomState, optional
            Random state to use for sampling.
        n_samples : int, optional
            Number of samples to generate. Defaults to 1000.

        Returns
        -------
        ~xarray.Dataset
            Dataset containing the posterior samples for the customer population.

        """
        return self.distribution_new_customer(
            data=data,
            T=T,
            random_seed=random_seed,
            var_names=["recency_frequency"],
            n_samples=n_samples,
        )["recency_frequency"]
