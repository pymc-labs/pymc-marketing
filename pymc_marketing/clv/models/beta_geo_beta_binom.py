from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc.distributions.dist_math import betaln
from pytensor.tensor import TensorVariable
from scipy.special import betaln, binom
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

        self.customer_id = customer_id
        self.frequency = frequency
        self.recency = recency
        self.T = T

        purchase_hyperprior, churn_hyperprior = self._process_priors(
            purchase_hyperprior, churn_hyperprior
        )

        coords = {"customer_id": customer_id}
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

            purchase_heterogeniety = pm.Beta(
                "purchase_heterogeneity", alpha=alpha, beta=beta
            )
            churn_heterogeniety = pm.Beta(
                "churn_heterogeneity", alpha=gamma, beta=delta
            )

            llike = BetaGeoBetaBinom(
                name="llike",
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                T=self.T,
                observed=np.stack((self.recency, self.frequency), axis=1),
            )

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

    # TODO: Add xarray call.
    # TODO: This is just copy-pasted from lifetimes; revise where needed.
    # TODO: Add type hinting
    def conditional_expected_number_of_purchases_up_to_time(
        self, m_periods_in_future, frequency, recency, n_periods
    ) -> xarray.DataArray:
        r"""
        Conditional expected purchases in future time period.
        The  expected  number  of  future  transactions across the next m_periods_in_future
        transaction opportunities by a customer with purchase history
        (x, tx, n).
        .. math:: E(X(n_{periods}, n_{periods}+m_{periods_in_future})| \alpha, \beta, \gamma, \delta, frequency, recency, n_{periods})
        See (13) in Fader & Hardie 2010.

        Methods below are adapted from the BetaGeoFitter class from the lifetimes package
        (see https://github.com/CamDavidsonPilon/lifetimes/).

        Parameters
        ----------
        t: array_like
            time n_periods (n+t)
        Returns
        -------
        array_like
            predicted transactions
        """
        x = frequency
        tx = recency
        n = n_periods

        params = self._unload_params("alpha", "beta", "gamma", "delta")
        alpha, beta, gamma, delta = params

        p1 = 1 / exp(self._loglikelihood(params, x, tx, n))
        p2 = exp(betaln(alpha + x + 1, beta + n - x) - betaln(alpha, beta))
        p3 = delta / (gamma - 1) * exp(gammaln(gamma + delta) - gammaln(1 + delta))
        p4 = exp(gammaln(1 + delta + n) - gammaln(gamma + delta + n))
        p5 = exp(
            gammaln(1 + delta + n + m_periods_in_future)
            - gammaln(gamma + delta + n + m_periods_in_future)
        )

        return p1 * p2 * p3 * (p4 - p5)

    # TODO: Add xarray call.
    # TODO: This is just copy-pasted from lifetimes; revise where needed.
    # TODO: Add type hinting
    def probability_alive(
        self, m_periods_in_future, frequency, recency, n_periods
    ) -> xarray.DataArray:
        """
        Conditional probability alive.
        Conditional probability customer is alive at transaction opportunity
        n_periods + m_periods_in_future.
        .. math:: P(alive at n_periods + m_periods_in_future|alpha, beta, gamma, delta, frequency, recency, n_periods)
        See (11) in [1].

        Methods below are adapted from the BetaGeoFitter class from the lifetimes package
        (see https://github.com/CamDavidsonPilon/lifetimes/).

        Parameters
        ----------
        m: array_like
            transaction opportunities
        Returns
        -------
        array_like
            alive probabilities
        """
        params = self._unload_params("alpha", "beta", "gamma", "delta")
        alpha, beta, gamma, delta = params

        p1 = betaln(alpha + frequency, beta + n_periods - frequency) - betaln(
            alpha, beta
        )
        p2 = betaln(gamma, delta + n_periods + m_periods_in_future) - betaln(
            gamma, delta
        )
        p3 = self._loglikelihood(params, frequency, recency, n_periods)

        return exp(p1 + p2) / exp(p3)

    def expected_purchases_new_customer(
        self,
        t=Union[int, np.ndarray, pd.Series, TensorVariable],
    ) -> xarray.DataArray:
        r"""
        Return expected number of transactions in first n n_periods.
        Expected number of transactions occurring across first n transaction
        opportunities.
        Used by Fader and Hardie to assess in-sample fit.
        .. math:: E(X(n) = x| \alpha, \beta, \gamma, \delta)
        See (8) in Fader & Hardie 2010.
        Parameters
        ----------
        t: int
            number of transaction opportunities
        Returns
        -------
        DataFrame:
            Predicted values, indexed by x
        """

        t = np.asarray(t)
        if t.size != 1:
            t = to_xarray(range(len(t)), t, dim="t")

        alpha, beta, gamma, delta = self._unload_params()

        term1 = alpha / (alpha + beta) * delta / (gamma - 1)
        term2 = 1 - (gammaf(gamma + delta)) / gammaf(gamma + delta + t) * (
            gammaf(1 + delta + t)
        ) / gammaf(1 + delta)

        return (term1 * term2).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )
