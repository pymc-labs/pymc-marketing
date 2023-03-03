import types
from typing import Optional, Union

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc import str_for_dist
from pymc.distributions.dist_math import betaln, check_parameters
from pytensor.tensor import TensorVariable
from scipy.special import expit, hyp2f1

from pymc_marketing.clv.distributions import ParetoNBD
from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.clv.utils import to_xarray


class ParetoNBDModel(CLVModel):
    # TODO: EDIT ME
    r"""Pareto Negative Binomial Distribution (Pareto/NBD) model

    In the Pareto/NBD model, the frequency of customer purchases is modelled as the time
    of each purchase has an instantaneous probability of occurrence (hazard) and, at
    every purchase, a probability of "dropout", i.e. no longer being a customer.

    Customer-specific data needed for statistical inference include 1) the total
    number of purchases (:math:`x`) and 2) the time of the last, i.e. xth, purchase. The
    omission of purchase times :math:`t_1, ..., t_x` is due to a telescoping sum in the
    exponential function of the joint likelihood; see Section 4.1 of [2] for more
    details.

    Methods below are adapted from the ParetoFitter class from the lifetimes package
    (see https://github.com/CamDavidsonPilon/lifetimes/).


    Parameters
    ----------
    customer_id: array_like
        Customer labels. Must not repeat.
    frequency: array_like
        The number of purchases of customers.
    recency: array_like
        The time of the last, i.e. xth, purchase.
    T: array_like
        The time of a customer's period under which they are under observation. By
        construction of the model, T > t_x.
    a_prior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`
    b_prior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`
    alpha_prior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`
    r: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`

    Examples
    --------
    Pareto/NBD model for customer

    .. code-block:: python
        import pymc as pm
        from pymc_marketing.clv import ParetoNBDModel, clv_summary

        summary = clv_summary(raw_data,'id_col_name','date_col_name')

        model = ParetoNBDModel(
            customer_id=summary['id_col'],
            frequency=summary['frequency']
            recency=[summary['recency'],
            T=summary['T'],
            r_prior=pm.HalfNormal.dist(10),
            alpha_prior=pm.HalfNormal.dist(10),
            s_prior=pm.HalfNormal.dist(10),
            beta_prior=pm.HalfNormal.dist(10),
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
    .. [1] David C. Schmittlein, Donald G. Morrison and Richard Colombo.
           "Counting Your Customers: Who Are They and What Will They Do Next."
           Management Science,Vol. 33, No. 1 (Jan., 1987), pp. 1-24.
    .. [2] Fader, Peter & G. S. Hardie, Bruce (2005).
           "A Note on Deriving the Pareto/NBD Model and Related Expressions."
           http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf
    """

    _model_name = "Pareto/NBD"  # Pareto Negative-Binomial Distribution
    _params = ["r", "alpha", "s", "beta"]

    def __init__(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        frequency: Union[np.ndarray, pd.Series, TensorVariable],
        recency: Union[np.ndarray, pd.Series, TensorVariable],
        T: Union[np.ndarray, pd.Series, TensorVariable],
        r_prior: Optional[TensorVariable] = None,
        alpha_prior: Optional[TensorVariable] = None,
        s_prior: Optional[TensorVariable] = None,
        beta_prior: Optional[TensorVariable] = None,
    ):
        super().__init__()

        self.customer_id = customer_id
        self.frequency = frequency
        self.recency = recency
        self.T = T

        r_prior, alpha_prior, s_prior, beta_prior = self._process_priors(
            r_prior, alpha_prior, s_prior, beta_prior
        )

        # each customer's information should be encapsulated by a single data entry
        # TODO: Create a separate _check_inputs() utility function for this and other constraints
        if len(np.unique(customer_id)) != len(customer_id):
            raise ValueError(
                "ParetoNBD  expects exactly one entry per customer. More than"
                " one entry is currently provided per customer id."
            )

        coords = {"customer_id": customer_id}
        with pm.Model(coords=coords) as self.model:
            # purchase rate hyperpriors
            r = self.model.register_rv(a_prior, name="r")
            alpha = self.model.register_rv(b_prior, name="alpha")

            # churn hyperpriors
            s = self.model.register_rv(alpha_prior, name="s")
            beta = self.model.register_rv(r_prior, name="beta")

            purchase_rate_prior = pm.Gamma(
                name="purchase_rate", alpha=r, beta=alpha, size=None
            )
            churn_prior = pm.Gamma(name="churn", alpha=s, beta=beta, size=None)

            T_ = pm.MutableData(name="T", value=self.T)

            llike = ParetoNBD(
                name="llike",
                purchase_rate=purchase_rate_prior,
                churn=churn_prior,
                T=T_,
                size=None,
                observed=np.stack((self.recency, self.frequency), axis=1),
            )

    # TODO: Edit per https://github.com/pymc-labs/pymc-marketing/pull/133
    def _process_priors(self, r_prior, alpha_prior, s_prior, beta_prior):
        # hyper priors for the transaction rate
        if r_prior is None:
            r_prior = pm.HalfFlat.dist()
        else:
            assert r_prior.eval().shape == ()
        if alpha_prior is None:
            alpha_prior = pm.HalfFlat.dist()
        else:
            assert alpha_prior.eval().shape == ()

        # hyper priors for the dropout rate
        if s_prior is None:
            s_prior = pm.HalfFlat.dist()
        else:
            assert s_prior.eval().shape == ()
        if beta_prior is None:
            beta_prior = pm.HalfFlat.dist()
        else:
            assert beta_prior.eval().shape == ()

        r_prior.str_repr = types.MethodType(str_for_dist, r_prior)
        alpha_prior.str_repr = types.MethodType(str_for_dist, alpha_prior)
        s_prior.str_repr = types.MethodType(str_for_dist, s_prior)
        beta_prior.str_repr = types.MethodType(str_for_dist, beta_prior)

        return r_prior, alpha_prior, s_prior, beta_prior

    # TODO: Move to CLVModel in future PR
    def _unload_params(self):
        trace = self.fit_result.posterior

        r = trace["r"]
        alpha = trace["alpha"]
        a = trace["s"]
        beta = trace["beta"]

        return r, alpha, s, beta, purchase_rate, churn

    # TODO: clv.utils.to_xarray needs to be called here
    # TODO: Add type hinting for returned value
    def expected_num_purchases(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        t: Union[np.ndarray, pd.Series, TensorVariable],
        frequency: Union[np.ndarray, pd.Series, TensorVariable],
        recency: Union[np.ndarray, pd.Series, TensorVariable],
        T: Union[np.ndarray, pd.Series, TensorVariable],
    ):
        # TODO: EDIT ME
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

        Conditional expected number of purchases up to time.

        Calculate the expected number of repeat purchases up to time t for a
        randomly choose individual from the population, given they have
        purchase history (frequency, recency, T).

        This is equation (41) from:
        http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf

        Parameters
        ----------
        t: array_like
            times to calculate the expectation for.
        frequency: array_like
            historical frequency of customer.
        recency: array_like
            historical recency of customer.
        T: array_like
            age of the customer.

        Returns
        -------
        array_like
        """

        x, t_x = frequency, recency
        params = self._unload_params("r", "alpha", "s", "beta")
        r, alpha, s, beta = params

        likelihood = self._conditional_log_likelihood(params, x, t_x, T)
        first_term = (
            gammaln(r + x)
            - gammaln(r)
            + r * log(alpha)
            + s * log(beta)
            - (r + x) * log(alpha + T)
            - s * log(beta + T)
        )
        second_term = log(r + x) + log(beta + T) - log(alpha + T)
        third_term = log((1 - ((beta + T) / (beta + T + t)) ** (s - 1)) / (s - 1))

        return exp(first_term + second_term + third_term - likelihood)

    # TODO: clv.utils.to_xarray needs to be called here
    # TODO: Add type hinting for returned value
    # TODO: rename to expected_avg_purchases_all_customers?
    def expected_num_purchases_new_customer(
        self,
        t: Union[np.ndarray, pd.Series],
    ):
        # TODO: EDIT ME
        r"""
        Posterior expected number of purchases for any interval of length :math:`t`. See
        equation (9) of [1].

        .. math::
            \mathbb{E}\left(X(t) \mid r, \alpha, a, b \right)
            = \frac{a + b - 1}{a - 1}
            \left[
                1 - \left(\frac{\alpha}{\alpha + t}\right)^r
                \text{hyp2f1}\left(r, b; a + b - 1; \frac{t}{\alpha + t}\right)
            \right]

        Return expected number of repeat purchases up to time t.

        Calculate the expected number of repeat purchases up to time t for a
        randomly choose individual from the population.

        Equation (27) from:
        http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf

        Parameters
        ----------
        t: array_like
            times to calculate the expectation for.

        Returns
        -------
        array_like
        """
        r, alpha, s, beta = self._unload_params("r", "alpha", "s", "beta")
        first_term = r * beta / alpha / (s - 1)
        second_term = 1 - (beta / (beta + t)) ** (s - 1)

        return first_term * second_term

    # TODO: clv.utils.to_xarray needs to be called here
    # TODO: Add type hinting for returned value
    # TODO: The equation cited in the docstrings is wrong; it's actually (35)
    #       The very next equation at the bottom of page 12 is a more stable equivalent,
    #       and very similar to conditional_probability_of_being_alive_up_to_time()
    def expected_probability_alive(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        frequency: Union[np.ndarray, pd.Series],
        recency: Union[np.ndarray, pd.Series],
        T: Union[np.ndarray, pd.Series],
    ):
        # TODO: EDIT ME
        r"""
        Posterior expected value of the probability of being alive at time T. The
        derivation of the closed form solution is available in [2].

        .. math::
            P\left\text{alive} \mid x, t_x, T, r, \alpha, a, b\right)
            = 1 \Big/
                \left\{
                    1 + \delta_{x>0} \frac{a}{b + x - 1}
                        \left(
                            \frac{\alpha + T}{\alpha + t_x}^{r + x}
                        \right)^{r x}
                \right\}

        Conditional probability alive.

        Compute the probability that a customer with history
        (frequency, recency, T) is currently alive.

        Section 5.1 from (equations (36) and (37)):
        http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf

        Parameters
        ----------
        frequency: float
            historical frequency of customer.
        recency: float
            historical recency of customer.
        T: float
            age of the customer.

        Returns
        -------
        float
            value representing a probability
        """
        x, t_x = frequency, recency
        r, alpha, s, beta = self._unload_params("r", "alpha", "s", "beta")
        A_0 = self._log_A_0([r, alpha, s, beta], x, t_x, T)

        return 1.0 / (
            1.0
            + exp(
                log(s)
                - log(r + s + x)
                + (r + x) * log(alpha + T)
                + s * log(beta + T)
                + A_0
            )
        )

    # TODO: clv.utils.to_xarray needs to be called here
    # TODO: Add type hinting
    # TODO: This method omits the case of x=0, and is ridiculously complex.
    def expected_purchases_probability(self, n, t, frequency, recency, T):
        """
        Return conditional probability of n purchases up to time t.

        Calculate the probability of n purchases up to time t for an individual
        with history frequency, recency and T (age).

        The main equation being implemented is (16) from:
        http://www.brucehardie.com/notes/028/pareto_nbd_conditional_pmf.pdf

        Parameters
        ----------
        n: int
            number of purchases.
        t: a scalar
            time up to which probability should be calculated.
        frequency: float
            historical frequency of customer.
        recency: float
            historical recency of customer.
        T: float
            age of the customer.

        Returns
        -------
        array_like
        """

        if t <= 0:
            return 0

        x, t_x = frequency, recency
        params = self._unload_params("r", "alpha", "s", "beta")
        r, alpha, s, beta = params

        if alpha < beta:
            min_of_alpha_beta, max_of_alpha_beta, p, _, _ = (
                alpha,
                beta,
                r + x + n,
                r + x,
                r + x + 1,
            )
        else:
            min_of_alpha_beta, max_of_alpha_beta, p, _, _ = (
                beta,
                alpha,
                s + 1,
                s + 1,
                s,
            )
        abs_alpha_beta = max_of_alpha_beta - min_of_alpha_beta

        log_l = self._conditional_log_likelihood(params, x, t_x, T)
        log_p_zero = (
            gammaln(r + x)
            + r * log(alpha)
            + s * log(beta)
            - (gammaln(r) + (r + x) * log(alpha + T) + s * log(beta + T) + log_l)
        )
        log_B_one = (
            gammaln(r + x + n)
            + r * log(alpha)
            + s * log(beta)
            - (gammaln(r) + (r + x + n) * log(alpha + T + t) + s * log(beta + T + t))
        )
        log_B_two = (
            r * log(alpha)
            + s * log(beta)
            + gammaln(r + s + x)
            + betaln(r + x + n, s + 1)
            + log(
                hyp2f1(
                    r + s + x,
                    p,
                    r + s + x + n + 1,
                    abs_alpha_beta / (max_of_alpha_beta + T),
                )
            )
            - (gammaln(r) + gammaln(s) + (r + s + x) * log(max_of_alpha_beta + T))
        )

        def _log_B_three(i):
            return (
                r * log(alpha)
                + s * log(beta)
                + gammaln(r + s + x + i)
                + betaln(r + x + n, s + 1)
                + log(
                    hyp2f1(
                        r + s + x + i,
                        p,
                        r + s + x + n + 1,
                        abs_alpha_beta / (max_of_alpha_beta + T + t),
                    )
                )
                - (
                    gammaln(r)
                    + gammaln(s)
                    + (r + s + x + i) * log(max_of_alpha_beta + T + t)
                )
            )

        zeroth_term = (n == 0) * (1 - exp(log_p_zero))
        first_term = n * log(t) - gammaln(n + 1) + log_B_one - log_l
        second_term = log_B_two - log_l
        third_term = logsumexp(
            [
                i * log(t) - gammaln(i + 1) + _log_B_three(i) - log_l
                for i in range(n + 1)
            ],
            axis=0,
        )

        try:
            size = len(x)
            sign = np.ones(size)
        except TypeError:
            sign = 1

        # In some scenarios (e.g. large n) tiny numerical errors in the calculation of second_term and third_term
        # cause sumexp to be ever so slightly negative and logsumexp throws an error. Hence we ignore the sign here.
        return zeroth_term + exp(
            logsumexp(
                [first_term, second_term, third_term],
                b=[sign, sign, -sign],
                axis=0,
                return_sign=True,
            )[0]
        )

    # TODO: clv.utils.to_xarray needs to be called here
    # TODO: Add type hinting
    # TODO: This function looks nothing like (18) in the paper.
    #       It is also almost identical to (34) for prob_alive,
    #       and should be refactored accordingly.
    def expected_future_probability_alive(self, t, frequency, recency, T):
        """
        Conditional probability of being alive up to time T+t.

        Compute the probability that a customer with history
        (frequency, recency, T) is still alive up to time T+t, given they have
            purchase history (frequency, recency, T).
        From paper:
        http://www.brucehardie.com/notes/015/additional_pareto_nbd_results.pdf

        Parameters
        ----------
        t: int
            time up to which probability should be calculated.
        frequency: float
            historical frequency of customer.
        recency: float
            historical recency of customer.
        T: float
            age of the customer.
        Returns
        -------
        float
            value representing a probability
        """

        x, t_x = frequency, recency
        r, alpha, s, beta = self._unload_params("r", "alpha", "s", "beta")
        A_0 = self._log_A_0([r, alpha, s, beta], x, t_x, T)
        K_1 = s * log(beta + T + t) - s * log(beta + T)
        K_2 = (
            log(s)
            - log(r + s + x)
            + (r + x) * log(alpha + T)
            + s * log(beta + T + t)
            + A_0
        )

        return 1.0 / (exp(K_1) + exp(K_2))
