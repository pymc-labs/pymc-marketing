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


class BetaGeoModel(CLVModel):
    r"""Beta-Geo Negative Binomial Distribution (BG/NBD) model

    In the BG/NBD model, the frequency of customer purchases is modelled as the time
    of each purchase has an instantaneous probability of occurrence (hazard) and, at
    every purchase, a probability of "dropout", i.e. no longer being a customer.

    Customer-specific data needed for statistical inference include 1) the total
    number of purchases (:math:`x`) and 2) the time of the last, i.e. xth, purchase. The
    omission of purchase times :math:`t_1, ..., t_x` is due to a telescoping sum in the
    exponential function of the joint likelihood; see Section 4.1 of [1] for more
    details.

    Methods below are adapted from the BetaGeoFitter class from the lifetimes package
    (see https://github.com/CamDavidsonPilon/lifetimes/).

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the following columns:
            * `frequency`: number of repeat purchases (with possible values 0, 1, 2, ...)
            * `recency`: time between the first and the last purchase (with possible values 0, 1, 2, ...)
            * `T`: time between the first purchase and the end of the observation
                period (with possible values 0, 1, 2, ...)
            * `customer_id`: unique customer identifier
    model_config: dict, optional
        Dictionary of model prior parameters. If not provided, the model will use default priors specified in
        the `default_model_config` class attribute.
    sampler_config: dict, optional
        Dictionary of sampler parameters. Defaults to None.

    Examples
    --------
    BG/NBD model for customer

    .. code-block:: python

        import pandas as pd

        import pymc as pm
        from pymc_marketing.clv import BetaGeoModel

        data = pd.DataFrame({
            "frequency": [4, 0, 6, 3],
            "recency": [30.73, 1.72, 0., 0.],
            "T": [38.86, 38.86, 38.86, 38.86],
        })
        data["customer_id"] = data.index

        prior_distribution = {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 0.1}}
        model = BetaGeoModel(
            data=data,
            model_config={
                "r_prior": prior_distribution,
                "alpha_prior": prior_distribution,
                "a_prior": prior_distribution,
                "b_prior": prior_distribution,
            },
            sampler_config={
                "draws": 1000,
                "tune": 1000,
                "chains": 2,
                "cores": 2,
            },
        )
        model.build_model()
        model.fit()
        print(model.fit_summary())

    Predicting the customer-specific number of purchases for a future time
    interval of length t given their previous frequency, recency, and T

    .. code-block:: python

        future_t = [2, 5, 7, 10]

        expected_num_purchases = model.expected_num_purchases(
            t=future_t,
            customer_id=data["customer_id"],
            frequency=data["frequenc"y],
            recency=data["recency"],
            T=data["T"],
        )

    Estimating the expected number of purchases for a randomly chosen
    individual in a future time period of length t

    .. code-block:: python

        expected_num_purchases_new_customer = model.expected_num_purchases_new_customer(
            t=future_t,
        )

    References
    ----------
    .. [1] Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). “Counting your customers”
           the easy way: An alternative to the Pareto/NBD model. Marketing science,
           24(2), 275-284. http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf
    .. [2] Fader, P. S., Hardie, B. G., & Lee, K. L. (2008). Computing
           P (alive) using the BG/NBD model. Research Note available via
           http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf.
    .. [3] Fader, P. S. & Hardie, B. G. (2013) Overcoming the BG/NBD Model's #NUM!
           Error Problem. Research Note available via
           http://brucehardie.com/notes/027/bgnbd_num_error.pdf.
    """

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
    def default_model_config(self) -> dict[str, dict]:
        return {
            "a_prior": {"dist": "HalfFlat", "kwargs": {}},
            "b_prior": {"dist": "HalfFlat", "kwargs": {}},
            "alpha_prior": {"dist": "HalfFlat", "kwargs": {}},
            "r_prior": {"dist": "HalfFlat", "kwargs": {}},
        }

    def build_model(self) -> None:  # type: ignore[override]
        a_prior = self._create_distribution(self.model_config["a_prior"])
        b_prior = self._create_distribution(self.model_config["b_prior"])
        alpha_prior = self._create_distribution(self.model_config["alpha_prior"])
        r_prior = self._create_distribution(self.model_config["r_prior"])

        coords = {"customer_id": self.data["customer_id"]}
        with pm.Model(coords=coords) as self.model:
            a = self.model.register_rv(a_prior, name="a")
            b = self.model.register_rv(b_prior, name="b")

            alpha = self.model.register_rv(alpha_prior, name="alpha")
            r = self.model.register_rv(r_prior, name="r")

            def logp(t_x, x, a, b, r, alpha, T):
                """
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
        """Utility function assigning default customer arguments
        for predictive methods and converting to xarrays.
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

    # taken from https://lifetimes.readthedocs.io/en/latest/lifetimes.fitters.html
    def expected_num_purchases(
        self,
        customer_id: np.ndarray | pd.Series,
        t: np.ndarray | pd.Series | TensorVariable,
        frequency: np.ndarray | pd.Series | TensorVariable,
        recency: np.ndarray | pd.Series | TensorVariable,
        T: np.ndarray | pd.Series | TensorVariable,
    ) -> xarray.DataArray:
        r"""
        Given a purchase history/profile of :math:`x` and :math:`t_x` for an individual
        customer, this method returns the expected number of future purchases in the
        next time interval of length :math:`t`, i.e. :math:`(T, T + t]`. The closed form
        solution for this equation is available as (10) from [1] linked above. With
        :math:`\text{hyp2f1}` being the Gaussian hypergeometric function, the
        expectation is defined as below.

        .. math::

           \mathbb{E}\left[Y(t) \mid x, t_x, T, r, \alpha, a, b\right] =
           \frac
           {
            \frac{a + b + x - 1}{a - 1}\left[
                1 - \left(\frac{\alpha + T}{\alpha + T + t}\right)^{r + x}
                \text{hyp2f1}\left(
                    r + x, b + x; a + b + x, \frac{1}{\alpha + T + t}
                \right)
            \right]
           }
           {
            1 + \delta_{x > 0} \frac{a}{b + x - 1} \left(
                \frac{\alpha + T}{\alpha + T + t}
            \right)^{r + x}
           }

        Parameters
        ----------
        customer_id : np.ndarray | pd.Series
            Customer identifier. Not required to be from data.
        t : np.ndarray | pd.Series | TensorVariable
            Time interval for which to predict the number of future purchases.
        frequency : np.ndarray | pd.Series | TensorVariable
            Number of repeat purchases.
        recency : np.ndarray | pd.Series | TensorVariable
            Time between the first and the last purchase.

        Returns
        -------
        xarray.DataArray
            Expected number of purchases in the next time interval of length :math:`t`.
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

    def expected_probability_alive(
        self,
        data: pd.DataFrame | None = None,
    ) -> xarray.DataArray:
        r"""
        Estimate probability a customer with history *frequency*, *recency*, and *T*
        is currently active. `data` parameter is only required for out-of-sample customers.

        Adapted from page (2) in Bruce Hardie's notes [3]_ and `lifetimes` package:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/beta_geo_fitter.py#L260

        Parameters
        ----------
        data: pd.DataFrame
            Optional dataframe containing the following columns:
                * `frequency`: number of repeat purchases
                * `recency`: time between the first and the last purchase
                * `T`: time between first purchase and end of observation period, model assumptions require T >= recency
                * `customer_id`: unique customer identifier

        References
        ----------
        .. [2] Fader, P. S., Hardie, B. G., & Lee, K. L. (2008). Computing
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
        """
        Deprecated method. Use `expected_purchases_new_customer` instead.
        """
        warnings.warn(
            "Deprecated method. Use 'expected_purchases_new_customer' instead.",
            FutureWarning,
            stacklevel=1,
        )
        self.expected_purchases_new_customer(*args, **kwargs)

    def expected_purchases_new_customer(
        self,
        t: np.ndarray | pd.Series,
    ) -> xarray.DataArray:
        """
        Expected number of purchases for a new customer across *t* time periods.

        Adapted from equation (9) in [1]_, and `lifetimes` library:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/beta_geo_fitter.py#L328

        Parameters
        ----------
        t: array_like
            Number of time periods over which to estimate purchases.
        References
        ----------
        .. [2] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
            "Counting Your Customers the Easy Way: An Alternative to the
            Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
            http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf
        """
        if t is not None:
            data = self.data.assign(t=t)

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

        This is the probability that a new customer will not make another purchase ("drops out")
        immediately after any previous purchase.

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
