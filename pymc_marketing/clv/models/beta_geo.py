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
"""Beta-Geometric Negative Binomial Distribution (BG/NBD) model for a non-contractual customer population across continuous time."""  # noqa: E501

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray
from pymc.distributions.dist_math import check_parameters
from pymc.util import RandomState
from pytensor.tensor import TensorVariable
from scipy.special import expit, hyp2f1

from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.clv.utils import to_xarray
from pymc_marketing.model_config import ModelConfig
from pymc_marketing.prior import Prior


class BetaGeoModel(CLVModel):
    r"""Beta-Geometric Negative Binomial Distribution (BG/NBD) model for a non-contractual customer population across continuous time.

    First introduced by Fader, Hardie & Lee [1]_, with additional predictive methods
    and enhancements in [2]_ and [3]_.

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
            * `a_prior`: Shape parameter for time until dropout; defaults to `pymc.HalfFlat()`
            * `b_prior`: Shape parameter for time until dropout; defaults to `pymc.HalfFlat()`
            * `alpha_prior`: Scale parameter for time between purchases; defaults to `pymc.HalfFlat()`
            * `r_prior`: Scale parameter for time between purchases; defaults to `pymc.HalfFlat()`
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
                "r_prior": Prior("Gamma", alpha=0.1, beta=1),
                "alpha_prior": Prior("Gamma", alpha=0.1, beta=1),
                "a_prior": Prior("Gamma", alpha=0.1, beta=1),
                "b_prior": Prior("Gamma", alpha=0.1, beta=1),
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

    """  # noqa: E501

    _model_type = "BG/NBD"  # Beta-Geometric Negative Binomial Distribution

    def __init__(
        self,
        data: pd.DataFrame,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        self._validate_cols(
            data,
            required_cols=["customer_id", "frequency", "recency", "T"],
            must_be_unique=["customer_id"],
        )
        super().__init__(
            data=data,
            model_config=model_config,
            sampler_config=sampler_config,
        )

    @property
    def default_model_config(self) -> ModelConfig:
        """Default model configuration."""
        return {
            "a_prior": Prior("HalfFlat"),
            "b_prior": Prior("HalfFlat"),
            "alpha_prior": Prior("HalfFlat"),
            "r_prior": Prior("HalfFlat"),
        }

    def build_model(self) -> None:  # type: ignore[override]
        """Build the model."""
        coords = {"customer_id": self.data["customer_id"]}
        with pm.Model(coords=coords) as self.model:
            a = self.model_config["a_prior"].create_variable("a")
            b = self.model_config["b_prior"].create_variable("b")
            alpha = self.model_config["alpha_prior"].create_variable("alpha")
            r = self.model_config["r_prior"].create_variable("r")

            def logp(t_x, x, a, b, r, alpha, T):
                """
                Compute the log-likelihood of the BG/NBD model.

                The log-likelihood expression here aligns with expression (4) from [3]
                due to the possible numerical instability of expression (3).
                """
                x_non_zero = x > 0

                # Refactored for numerical error
                d1 = (
                    pt.gammaln(r + x)
                    - pt.gammaln(r)
                    + pt.gammaln(a + b)
                    + pt.gammaln(b + x)
                    - pt.gammaln(b)
                    - pt.gammaln(a + b + x)
                )

                d2 = r * pt.log(alpha) - (r + x) * pt.log(alpha + t_x)
                c3 = ((alpha + t_x) / (alpha + T)) ** (r + x)
                c4 = a / (b + x - 1)

                logp = d1 + d2 + pt.log(c3 + pt.switch(x_non_zero, c4, 0))

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
            ],
            must_be_unique=["customer_id"],
        )

        a = self.fit_result["a"]
        b = self.fit_result["b"]
        alpha = self.fit_result["alpha"]
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

    def expected_num_purchases(
        self,
        customer_id: np.ndarray | pd.Series,
        t: np.ndarray | pd.Series | TensorVariable,
        frequency: np.ndarray | pd.Series | TensorVariable,
        recency: np.ndarray | pd.Series | TensorVariable,
        T: np.ndarray | pd.Series | TensorVariable,
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
        frequency, recency = to_xarray(customer_id, frequency, recency)

        a, b, alpha, r = self._unload_params()

        numerator = 1 - ((alpha + T) / (alpha + T + t)) ** (r + frequency) * hyp2f1(
            r + frequency,
            b + frequency,
            a + b + frequency - 1,
            t / (alpha + T + t),
        )
        numerator *= (a + b + frequency - 1) / (a - 1)
        denominator = 1 + (frequency > 0) * (a / (b + frequency - 1)) * (
            (alpha + T) / (alpha + recency)
        ) ** (r + frequency)

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

    def expected_num_purchases_new_customer(self, *args, **kwargs) -> xarray.DataArray:
        """Compute the expected number of purchases for a new customer.

        This is a deprecated method and will be removed in a future release.
        Please use `BetaGeoModel.expected_purchases_new_customer` instead.
        """
        warnings.warn(
            "Deprecated method. Use 'expected_purchases_new_customer' instead.",
            FutureWarning,
            stacklevel=1,
        )
        self.expected_purchases_new_customer(*args, **kwargs)

    def expected_purchases_new_customer(
        self,
        data: pd.DataFrame | None = None,
        *,
        t: np.ndarray | pd.Series,
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

    def _distribution_new_customers(
        self,
        random_seed: RandomState | None = None,
        var_names: Sequence[str] = ("population_dropout", "population_purchase_rate"),
    ) -> xarray.Dataset:
        with pm.Model():
            a = pm.HalfFlat("a")
            b = pm.HalfFlat("b")
            alpha = pm.HalfFlat("alpha")
            r = pm.HalfFlat("r")

            fit_result = self.fit_result
            if fit_result.sizes["chain"] == 1 and fit_result.sizes["draw"] == 1:
                # For map fit add a dummy draw dimension
                fit_result = self.fit_result.squeeze("draw").expand_dims(
                    draw=range(1000)
                )

            pm.Beta("population_dropout", alpha=a, beta=b)
            pm.Gamma("population_purchase_rate", alpha=r, beta=alpha)

            return pm.sample_posterior_predictive(
                fit_result,
                var_names=var_names,
                random_seed=random_seed,
            ).posterior_predictive

    def distribution_new_customer_dropout(
        self,
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
        return self._distribution_new_customers(
            random_seed=random_seed,
            var_names=["population_dropout"],
        )["population_dropout"]

    def distribution_new_customer_purchase_rate(
        self,
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
        return self._distribution_new_customers(
            random_seed=random_seed,
            var_names=["population_purchase_rate"],
        )["population_purchase_rate"]
