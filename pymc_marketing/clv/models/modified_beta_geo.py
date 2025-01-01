#   Copyright 2025 The PyMC Labs Developers
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

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray
from pymc.distributions.dist_math import check_parameters
from scipy.special import hyp2f1

from pymc_marketing.clv.models import BetaGeoModel
from pymc_marketing.clv.utils import to_xarray


class ModifiedBetaGeoModel(BetaGeoModel):
    r"""Also known as the MBG/NBD model.

    Based on [1]_, [2]_

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
            * `alpha_prior`: Scale parameter for time between purchases; defaults to `Prior("HalfFlat")`
            * `r_prior`: Shape parameter for time between purchases; defaults to `Prior("HalfFlat")`
            * `a_prior`: Shape parameter of dropout process; defaults to `phi_purchase_prior` * `kappa_purchase_prior`
            * `b_prior`: Shape parameter of dropout process; defaults to `1-phi_dropout_prior` * `kappa_dropout_prior`
            * `phi_dropout_prior`: Nested prior for a and b priors; defaults to `Prior("Uniform", lower=0, upper=1)`
            * `kappa_dropout_prior`: Nested prior for a and b priors; defaults to `Prior("Pareto", alpha=1, m=1)`
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
                "r_prior": Prior("HalfFlat"),
                "alpha_prior": Prior("HalfFlat"),
                "a_prior": Prior("HalfFlat"),
                "b_prior": Prior("HalfFlat),
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
       analysis,"
       International Journal of Research in Marketing, 24 (3), 201-209.
    .. [2] Wagner, U. and Hoppe D. (2008), "Erratum on the MBG/NBD Model,"
       International Journal of Research in Marketing, 25 (3), 225-226.
    """

    _model_type = "MBG/NBD"

    def build_model(self) -> None:  # type: ignore[override]
        """Build the model."""
        coords = {"customer_id": self.data["customer_id"]}
        with pm.Model(coords=coords) as self.model:
            # purchase rate priors
            alpha = self.model_config["alpha_prior"].create_variable("alpha")
            r = self.model_config["r_prior"].create_variable("r")

            # dropout priors
            if "a_prior" in self.model_config and "b_prior" in self.model_config:
                a = self.model_config["a_prior"].create_variable("a")
                b = self.model_config["b_prior"].create_variable("b")
            else:
                # hierarchical pooling of dropout rate priors
                phi_dropout = self.model_config["phi_dropout_prior"].create_variable(
                    "phi_dropout"
                )
                kappa_dropout = self.model_config[
                    "kappa_dropout_prior"
                ].create_variable("kappa_dropout")

                a = pm.Deterministic("a", phi_dropout * kappa_dropout)
                b = pm.Deterministic("b", (1.0 - phi_dropout) * kappa_dropout)

            def logp(t_x, x, a, b, r, alpha, T):
                """Compute the log-likelihood of the MBG/NBD model."""
                a1 = pt.gammaln(r + x) - pt.gammaln(r) + r * pt.log(alpha)
                a2 = (
                    pt.gammaln(a + b)
                    + pt.gammaln(b + x + 1)
                    - pt.gammaln(b)
                    - pt.gammaln(a + b + x + 1)
                )
                a3 = -(r + x) * pt.log(alpha + T)
                a4 = (
                    pt.log(a)
                    - pt.log(b + x)
                    + (r + x) * (pt.log(alpha + T) - pt.log(alpha + t_x))
                )

                logp = a1 + a2 + a3 + pt.logaddexp(a4, 0)

                return check_parameters(
                    logp,
                    a > 0,
                    b > 0,
                    alpha > 0,
                    r > 0,
                    msg="a, b, alpha, r > 0",
                )

            pm.Potential(
                "likelihood",
                logp(
                    x=self.data["frequency"],
                    t_x=self.data["recency"],
                    a=a,
                    b=b,
                    alpha=alpha,
                    r=r,
                    T=self.data["T"],
                ),
            )

    def expected_num_purchases(
        self,
        customer_id: np.ndarray | pd.Series,
        t: np.ndarray | pd.Series | pt.TensorVariable,
        frequency: np.ndarray | pd.Series | pt.TensorVariable,
        recency: np.ndarray | pd.Series | pt.TensorVariable,
        T: np.ndarray | pd.Series | pt.TensorVariable,
    ) -> xarray.DataArray:
        r"""Compute the expected number of purchases for a customer.

        This is a deprecated method and will be removed in a future release.
        Please use `BetaGeoModel.expected_purchases` instead.
        """
        warnings.warn(
            "Deprecated method. Use 'expected_purchases' instead.",
            FutureWarning,
            stacklevel=1,
        )

        t = np.asarray(t)
        if t.size != 1:
            t = to_xarray(customer_id, t)

        T = np.asarray(T)
        if T.size != 1:
            T = to_xarray(customer_id, T)

        # print(customer_id)
        x, t_x = to_xarray(customer_id, frequency, recency)

        a, b, alpha, r = self._unload_params()

        hyp_term = hyp2f1(r + x, b + x + 1, a + b + x, t / (alpha + T + t))
        first_term = (a + b + x) / (a - 1)
        second_term = 1 - hyp_term * ((alpha + T) / (alpha + t + T)) ** (r + x)
        numerator = first_term * second_term

        denominator = 1 + (a / (b + x)) * ((alpha + T) / (alpha + t_x)) ** (r + x)

        return (numerator / denominator).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_purchases(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        r"""Compute the expected number of future purchases across *future_t* time periods given *recency*, *frequency*, and *T* for each customer.

        The *data* parameter is only required for out-of-sample customers.

        Adapted from equation (3) in [1]_, and *lifetimes* package:
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

        Adapted from equation (3) in [1]_, and `lifetimes` library:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/modified_beta_geo_fitter.py#L130

        Parameters
        ----------
        t : array_like
            Number of time periods over which to estimate purchases.

        References
        ----------
        .. [1] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
        "Empirical validation and comparison of models for customer base
        analysis,"

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

        Adapted from *lifetimes* package:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/modified_beta_geo_fitter.py#L188

        Parameters
        ----------
        data : *pandas.DataFrame
            Optional dataframe containing the following columns:

            * `customer_id`: Unique customer identifier
            * `frequency`: Number of repeat purchases
            * `recency`: Time between the first and the last purchase
            * `T`: Time between first purchase and end of observation period, model assumptions require T >= recency
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

        The MBG/NBD model does not support this feature at the moment.
        """
        raise NotImplementedError(
            "The MBG/NBD model does not support this feature at the moment."
        )
