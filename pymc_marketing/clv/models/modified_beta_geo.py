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

    Based on [5]_, [6]_, this model has the following assumptions:
    1) Each individual, ``i``, has a hidden ``lambda_i`` and ``p_i`` parameter
    2) These come from a population wide Gamma and a Beta distribution
       respectively.
    3) Individuals purchases follow a Poisson process with rate :math:`\lambda_i*t` .
    4) At the beginning of their lifetime and after each purchase, an
       individual has a p_i probability of dieing (never buying again).

    References
    ----------
    .. [5] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
       "Empirical validation and comparison of models for customer base
       analysis,"
       International Journal of Research in Marketing, 24 (3), 201-209.
    .. [6] Wagner, U. and Hoppe D. (2008), "Erratum on the MBG/NBD Model,"
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

        proba = 1.0 / (1 + (a / (b + x)) * ((alpha + T) / (alpha + t_x)) ** (r + x))

        return proba.transpose("chain", "draw", "customer_id", missing_dims="ignore")
