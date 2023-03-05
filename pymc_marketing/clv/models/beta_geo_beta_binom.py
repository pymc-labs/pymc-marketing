from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pymc as pm
import xarray
from numpy import exp
from pytensor.tensor import TensorVariable
from scipy.special import betaln
from scipy.special import gamma as gammaf
from scipy.special import gammaln

from pymc_marketing.clv.distributions import BetaGeoBetaBinom
from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.clv.utils import to_xarray


# TODO: Proofread docstrings
class BetaGeoBetaBinomModel(CLVModel):
    r"""Beta-Geometric Beta Binomial Distribution (BG/BB) model for discrete, non-contractual transactions.

    In the BG/BB model, the frequency of customer purchases is modelled as the time
    of each purchase has an instantaneous probability of occurrence (hazard) and, at
    every purchase, a probability of "dropout", i.e. no longer being a customer.

    Customer-specific data needed for statistical inference include 1) the total
    number of purchases (:math:`x`) and 2) the time of the last, i.e. xth, purchase. The
    omission of purchase times :math:`t_1, ..., t_x` is due to a telescoping sum in the
    exponential function of the joint likelihood; see Section 4.1 of [1] for more
    details.

    Parameters
    ----------
    customer_id: array_like
        Customer labels. Must not repeat.
    frequency: array_like
        The number of purchases of customers.
    recency: array_like
        The time of the last, i.e. xth, purchase.
    T: int
        The number of purchase opportunities.
    purchase_hyperprior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`
    churn_hyperprior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`

    Examples
    --------
    BG/BB model for customer

    .. code-block:: python
        import pymc as pm
        from pymc_marketing.clv import BetaGeoModel

        model = BetaGeoBetaBinom(
            frequency=[4, 0, 6, 3, ...],
            recency=[30.73, 1.72, 0., 0., ...]
            p_prior=pm.HalfNormal.dist(10),
            q_prior=pm.HalfNormal.dist(10),
            v_prior=pm.HalfNormal.dist(10),
        )

        model.fit()
        print(model.fit_summary())

        # Estimating the expected number of purchases for a randomly chosen
        # individual in a future time period of length t
        expected_num_purchases = model.expected_num_purchases(
            t=[2, 5, 7, 10],
        )

        # Predicting the customer-specific number of purchases for a future
        # time interval of length t given their previous frequency and recency
        expected_num_purchases_new_customer = model.expected_num_purchases_new_customer(
            t=[5, 5, 5, 5],
            frequency=[5, 2, 1, 8],
            recency=[7, 4, 2.5, 11],
            T=[10, 8, 10, 22],
        )

    References
    ----------
    .. [1] Fader, Peter S., Bruce G.S. Hardie, and Jen Shang (2010),
       "Customer-Base Analysis in a Discrete-Time Noncontractual Setting,"
       Marketing Science, 29 (6), 1086-1108. https://www.brucehardie.com/papers/020/fader_et_al_mksc_10.pdf
    """

    _model_name = "BG/BB"  # Beta-Geometric Beta-Binomial Model
    _params = ["alpha", "beta", "gamma", "delta"]

    def __init__(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        frequency: Union[np.ndarray, pd.Series, TensorVariable],
        recency: Union[np.ndarray, pd.Series, TensorVariable],
        T: Union[int, np.ndarray, pd.Series, TensorVariable],
        purchase_hyperprior: Optional[TensorVariable] = None,
        churn_hyperprior: Optional[TensorVariable] = None,
    ):
        # Validate inputs before initializing model:

        t_array = np.array(T)
        if np.unique(t_array).shape[0] != 1:
            raise ValueError(
                "The BG/BB Model requires a static T value for all customers,"
                " but a heterogeneous T array was provided."
                " If using clv_summary(), try df.assign(T=df['T'].max())"
            )

        if len(np.unique(customer_id)) != len(customer_id):
            raise ValueError(
                "The BG/BB expects exactly one entry per customer. More than"
                " one entry is currently provided per customer id."
            )

        super().__init__()

        self._customer_id = customer_id
        self._frequency = frequency
        self._recency = recency
        self._T = T

        purchase_hyperprior, churn_hyperprior = self._process_priors(
            purchase_hyperprior, churn_hyperprior
        )

        coords = {"customer_id": self._customer_id}
        with pm.Model(coords=coords) as self.model:
            purchase_hyper = self.model.register_rv(
                purchase_hyperprior, name="purchase_hyperprior"
            )
            churn_hyper = self.model.register_rv(
                churn_hyperprior, name="churn_hyperprior"
            )

            # Heirarchical pooling of hyperparams for beta distribution parameters.

            purchase_pool = pm.Uniform(
                "purchase_pool",
                lower=0,
                upper=1,
            )
            churn_pool = pm.Uniform(
                "churn_pool",
                lower=0,
                upper=1,
            )

            alpha = pm.Deterministic("alpha", purchase_pool * purchase_hyper)
            beta = pm.Deterministic("beta", (1.0 - purchase_pool) * purchase_hyper)
            gamma = pm.Deterministic("gamma", churn_pool * churn_hyper)
            delta = pm.Deterministic("delta", (1.0 - churn_pool) * churn_hyper)

            # TODO: This is for plotting the population posterior, but will fail flake8
            purchase_heterogeniety = pm.Beta(
                "purchase_heterogeneity", alpha=alpha, beta=beta
            )
            # TODO: This is for plotting the population posterior, but will fail flake8
            churn_heterogeniety = pm.Beta(
                "churn_heterogeneity", alpha=gamma, beta=delta
            )

            T = pm.ConstantData(name="T", value=self._T)

            self.llike = BetaGeoBetaBinom(
                name="llike",
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                T=T,
                observed=np.stack((self._recency, self._frequency), axis=1),
            )

    # TODO: Add type hinting
    def _process_priors(self, purchase_hyperprior, churn_hyperprior):
        # hyper priors for the Gamma params
        if purchase_hyperprior is None:
            purchase_hyperprior = pm.HalfFlat.dist()
        else:
            self._check_prior_ndim(purchase_hyperprior)
        if churn_hyperprior is None:
            churn_hyperprior = pm.HalfFlat.dist()
        else:
            self._check_prior_ndim(churn_hyperprior)

        return super()._process_priors(purchase_hyperprior, churn_hyperprior)

    # TODO: Is this return type hint correct?
    def _unload_params(self) -> Tuple[np.ndarray]:
        # trace = self.fit_result.posterior
        #
        # alpha = trace["alpha"]
        # beta = trace["beta"]
        # gamma = trace["gamma"]
        # delta = trace["delta"]
        # return alpha, beta, gamma, delta

        # TODO: will .get(param) work here?
        return tuple([self.fit_result.posterior[param] for param in self._params])

    # TODO: Edit docstrings
    def expected_purchases_new_customer(
        self,
        t: Union[int, np.ndarray, pd.Series, TensorVariable],
    ) -> xarray.DataArray:
        r"""
        Expected number of transactions for a new customer with `t` purchase opportunities.

        .. math:: E(X(n) | \alpha, \beta, \gamma, \delta)
        See (8) in Fader & Hardie 2010.

        Parameters
        ----------
        t: int
            number of transaction opportunities
        Returns
        -------
        Xarray
        """

        t = np.asarray(t)
        if t.size != 1:
            t = to_xarray(range(len(t)), t, dim="t")

        alpha, beta, gamma, delta = self._unload_params()

        term1 = alpha / (alpha + beta) * delta / (gamma - 1)
        term2 = 1 - (gammaf(gamma + delta)) / gammaf(gamma + delta + t) * (
            gammaf(1 + delta + t)
        ) / gammaf(1 + delta)

        return (term1 * term2).transpose("chain", "draw", "t", missing_dims="ignore")

    # TODO: Edit docstrings
    def probability_alive(
        self,
        future_t: Union[int, np.ndarray, pd.Series, TensorVariable],
        customer_id: Union[np.ndarray, pd.Series] = None,
        frequency: Union[np.ndarray, pd.Series, TensorVariable] = None,
        recency: Union[np.ndarray, pd.Series, TensorVariable] = None,
        T: Union[int, np.ndarray, pd.Series, TensorVariable] = None,
    ) -> xarray.DataArray:
        r"""
        Conditional probability alive.
        Conditional probability customer is alive at transaction opportunity
        n_periods + m_periods_in_future.
        .. math:: P(alive at n_periods + m_periods_in_future|alpha, beta, gamma, delta, frequency, recency, n_periods)
        See (11) in Fader & Hardie 2010.

        This method is adapted from the lifetimes package
        (see https://raw.githubusercontent.com/CamDavidsonPilon/lifetimes/master/lifetimes/fitters/beta_geo_beta_binom_fitter.py).

        Parameters
        ----------
        future_t: array_like
            transaction opportunities
        Returns
        -------
        array_like
            alive probabilities
        """

        if customer_id is None:
            customer_id = self._customer_id
        if frequency is None:
            x = self._frequency
        else:
            x = frequency
        if recency is None:
            tx = self._recency
        else:
            tx = recency
        if T is None:
            T = self._T

        t = np.asarray(future_t)
        if t.size != 1:
            t = to_xarray(customer_id, t)

        T = np.asarray(T)
        if T.size != 1:
            T = to_xarray(customer_id, T)

        alpha, beta, gamma, delta = self._unload_params()

        p1 = betaln(alpha + x, beta + T - x) - betaln(alpha, beta)
        p2 = betaln(gamma, delta + T + t) - betaln(gamma, delta)
        p3 = pm.logp(self.llike, np.stack((tx, x), axis=1)).eval()

        prob_alive = exp(p1 + p2) / exp(p3)

        return prob_alive.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    # TODO: Edit docstrings
    def expected_purchases(
        self,
        future_t: Union[int, np.ndarray, pd.Series, TensorVariable],
        customer_id: Union[np.ndarray, pd.Series] = None,
        frequency: Union[np.ndarray, pd.Series, TensorVariable] = None,
        recency: Union[np.ndarray, pd.Series, TensorVariable] = None,
        T: Union[int, np.ndarray, pd.Series, TensorVariable] = None,
    ) -> xarray.DataArray:
        r"""
                Conditional expected purchases in future time period.
                The  expected  number  of  future  transactions across the next m_periods_in_future
                transaction opportunities by a customer with purchase history
                (x, tx, n).
                .. math:: E(X(n_{periods}, n_{periods}+m_{periods_in_future})| \alpha, \beta, \gamma, \delta, frequency, recency, n_{periods})
                See (13) in Fader & Hardie 2010.

                This method is adapted from the lifetimes package
                (see https://raw.githubusercontent.com/CamDavidsonPilon/lifetimes/master/lifetimes/fitters/beta_geo_beta_binom_fitter.py).
        .

                Parameters
                ----------
                t: array_like
                    time n_periods (n+t)
                Returns
                -------
                array_like
                    predicted transactions
        """

        if customer_id is None:
            customer_id = self._customer_id
        if frequency is None:
            x = self._frequency
        else:
            x = frequency
        if recency is None:
            tx = self._recency
        else:
            tx = recency
        if T is None:
            T = self._T

        t = np.asarray(future_t)
        if t.size != 1:
            t = to_xarray(customer_id, t)

        T = np.asarray(T)
        if T.size != 1:
            T = to_xarray(customer_id, T)

        params = self._unload_params("alpha", "beta", "gamma", "delta")
        alpha, beta, gamma, delta = params

        p1 = 1 / exp(pm.logp(self.llike, np.stack((tx, x), axis=1)).eval())
        p2 = exp(betaln(alpha + x + 1, beta + T - x) - betaln(alpha, beta))
        p3 = delta / (gamma - 1) * exp(gammaln(gamma + delta) - gammaln(1 + delta))
        p4 = exp(gammaln(1 + delta + T) - gammaln(gamma + delta + T))
        p5 = exp(gammaln(1 + delta + T + t) - gammaln(gamma + delta + T + t))

        purchases = p1 * p2 * p3 * (p4 - p5)

        return purchases.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )
