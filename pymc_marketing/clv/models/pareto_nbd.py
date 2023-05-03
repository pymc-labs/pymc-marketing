import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pymc as pm
import xarray
from numpy import exp, log
from pytensor.tensor import TensorVariable
from scipy.special import betaln, gammaln, hyp2f1
from xarray_einstats.stats import logsumexp as xr_logsumexp

from pymc_marketing.clv.distributions import ParetoNBD
from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.clv.utils import to_xarray


class ParetoNBDModel(CLVModel):
    r"""Pareto Negative Binomial Distribution (Pareto/NBD) population model for continuous, non-contractual scenarios,
    based on Schmittlein, et al. in [1]_.

    The Pareto/NBD model assumes the population of customer lifetime lengths follows a Gamma distribution,
    and time between customer purchases are also Gamma-distributed while a given customer is active.

    This model requires customer data to be aggregated in RFM format via `clv.rfm_summary()` or equivalent.

    Please note this model is still experimental. See code examples in documentation if encountering fitting issues.

    Parameters
    ----------
    customer_id: array_like
        Customer labels; must be unique.
    recency: array_like
        Number of time periods between the customer's first and most recent purchases.
    frequency: array_like
        Number of repeat purchases per customer.
    T: array_like
        Number of time periods since the customer's first purchase.
        Model assumptions require T >= recency.
    r_prior: scalar PyMC distribution, optional
        Shape parameter of time between purchases for customer population.
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.Weibull.dist(alpha=2, beta=1)`
    alpha_prior: scalar PyMC distribution, optional
        Scale parameter of time between purchases for customer population.
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.Weibull.dist(alpha=2, beta=10)`
    s_prior: scalar PyMC distribution, optional
        Shape parameter of time until churn for customer population.
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.Weibull.dist(alpha=2, beta=1)`
    beta_prior: scalar PyMC distribution, optional
        Scale parameter of time until churn for customer population.
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.Weibull.dist(alpha=2, beta=10)`

    Examples
    --------

        .. code-block:: python
            import pymc as pm
            from pymc_marketing.clv import ParetoNBDModel, clv_summary

            summary = clv_summary(raw_data,'id_col_name','date_col_name')

            model = ParetoNBDModel(
                customer_id=summary['id_col'],
                frequency=summary['frequency']
                recency=[summary['recency'],
                T=summary['T'],
                r_prior=pm.Weibull.dist(alpha=2,beta=1),
                alpha_prior=pm.Weibull.dist(alpha=2,beta=10),
                s_prior=pm.Weibull.dist(alpha=2,beta=1),
                beta_prior=pm.Weibull.dist(alpha=2,beta=10),,
            )

            # Fit model via Maximum a Posteriori:
            model.fit(fit_method='map')

            # Full posterior estimation while model is still in experimental status:
            with model.model:
                model.fit(step=pm.Slice(), draws=2000, tune=1000)

            print(model.fit_summary())

            # Predict number of purchases for a specific customer
            # over future_t time periods given their current frequency, recency, T
            expected_purchases = model.expected_purchases(
                future_t=[5, 5, 5, 5],
                frequency=[5, 2, 1, 8],
                recency=[7, 4, 2.5, 11],
                T=[10, 8, 10, 22],
            )

            # Predict probability a customer will still be active
            # in 't' time periods given current frequency, recency, T
            probability_alive = model.probability_alive(
                future_t=[0, 3, 6, 9],
                frequency=[5, 2, 1, 8],
                recency=[7, 4, 2.5, 11],
                T=[10, 8, 10, 22],
            )

            # Predict probability of customer making 'n' purchases over 't' time periods
            # given current frequency, recency, T
            expected_num_purchases = model.purchase_probability(
                n=[0, 1, 2, 3],
                future_t=[10,20,30,40],
                frequency=[5, 2, 1, 8],
                recency=[7, 4, 2.5, 11],
                T=[10, 8, 10, 22],
            )

            # Estimate expected number of purchases for a new customer
            # over 't' time periods
            expected_purchases_new_customer = model.expected_purchases_new_customer(
                t=[2, 5, 7, 10],
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
    .. [3] Fader, Peter & G. S. Hardie, Bruce (2014).
           "Additional Results for the Pareto/NBD Model."
           https://www.brucehardie.com/notes/015/additional_pareto_nbd_results.pdf
    .. [4] Fader, Peter & G. S. Hardie, Bruce (2014).
           "Deriving the Conditional PMF of the Pareto/NBD Model."
           https://www.brucehardie.com/notes/028/pareto_nbd_conditional_pmf.pdf
    """

    _model_name = "Pareto/NBD"  # Pareto Negative-Binomial Distribution
    _params = ["r", "alpha", "s", "beta"]

    def __init__(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        recency: Union[np.ndarray, pd.Series, TensorVariable],
        frequency: Union[np.ndarray, pd.Series, TensorVariable],
        T: Union[np.ndarray, pd.Series, TensorVariable],
        r_prior: Optional[TensorVariable] = None,
        alpha_prior: Optional[TensorVariable] = None,
        s_prior: Optional[TensorVariable] = None,
        beta_prior: Optional[TensorVariable] = None,
    ):
        warnings.warn(
            "The Pareto/NBD model is still experimental. Please see code examples in\
                        documentation if model fitting issues are encountered.",
            UserWarning,
        )

        if len(np.unique(customer_id)) != len(customer_id):
            raise ValueError("Customers must have unique ID labels.")

        super().__init__()

        self._customer_id = customer_id
        self._frequency = frequency
        self._recency = recency
        self._T = T

        r_prior, alpha_prior, s_prior, beta_prior = self._process_priors(
            r_prior, alpha_prior, s_prior, beta_prior
        )
        # TODO: rename hyperpriors to purchase_shape, purchase_scale, churn_shape, churn_scale?
        coords = {"customer_id": self._customer_id}
        with pm.Model(coords=coords) as self.model:
            # purchase rate priors
            r = self.model.register_rv(r_prior, name="r")
            alpha = self.model.register_rv(alpha_prior, name="alpha")

            # churn priors
            s = self.model.register_rv(s_prior, name="s")
            beta = self.model.register_rv(beta_prior, name="beta")

            ParetoNBD(
                name="likelihood",
                r=r,
                alpha=alpha,
                s=s,
                beta=beta,
                T=self._T,
                observed=np.stack((self._recency, self._frequency), axis=1),
            )

    def _process_priors(self, r_prior, alpha_prior, s_prior, beta_prior):
        # priors for purchase rate
        if r_prior is None:
            r_prior = pm.Weibull.dist(alpha=10, beta=1)
        else:
            self._check_prior_ndim(r_prior)
        if alpha_prior is None:
            alpha_prior = pm.Weibull.dist(alpha=10, beta=10)
        else:
            self._check_prior_ndim(alpha_prior)

        # hyper priors for churn rate
        if s_prior is None:
            s_prior = pm.Weibull.dist(alpha=10, beta=1)
        else:
            self._check_prior_ndim(s_prior)
        if beta_prior is None:
            beta_prior = pm.Weibull.dist(alpha=10, beta=10)
        else:
            self._check_prior_ndim(beta_prior)

        return super()._process_priors(r_prior, alpha_prior, s_prior, beta_prior)

    def _unload_params(self) -> Tuple[xarray.DataArray]:
        """Utility function retrieving posterior parameters for predictive methods"""
        return tuple([self.fit_result.posterior[param] for param in self._params])

    # TODO: Convert to list comprehension to support covariates?
    def _process_customers(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        frequency: Union[np.ndarray, pd.Series, TensorVariable],
        recency: Union[np.ndarray, pd.Series, TensorVariable],
        T: Union[np.ndarray, pd.Series, TensorVariable],
    ) -> Tuple[xarray.DataArray]:
        """Utility function assigning default customer arguments
        for predictive methods and converting to xarrays.
        """
        if customer_id is None:
            customer_id = self._customer_id
        if frequency is None:
            frequency = self._frequency
        if recency is None:
            recency = self._recency
        if T is None:
            T = self._T

        return to_xarray(customer_id, frequency, recency, T)

    @staticmethod
    def _logp(
        r: np.ndarray,
        alpha: np.ndarray,
        s: np.ndarray,
        beta: np.ndarray,
        x: xarray.DataArray,
        t_x: xarray.DataArray,
        T: xarray.DataArray,
    ) -> xarray.DataArray:
        """
        Utility function for using ParetoNBD log-likelihood in predictive methods.
        """
        # Add one dummy dimension to the right of the scalar parameters, so they broadcast with the `T` vector
        pareto_dist = ParetoNBD.dist(
            r=r.values[..., None],
            alpha=alpha.values[..., None],
            s=s.values[..., None],
            beta=beta.values[..., None],
            T=T.values,
        )
        values = np.vstack((t_x.values, x.values)).T
        loglike = pm.logp(pareto_dist, values).eval()
        return xarray.DataArray(data=loglike, dims=("chain", "draw", "customer_id"))

    def expected_purchases(
        self,
        future_t: Union[float, np.ndarray, pd.Series, TensorVariable],
        customer_id: Union[np.ndarray, pd.Series] = None,
        recency: Union[np.ndarray, pd.Series, TensorVariable] = None,
        frequency: Union[np.ndarray, pd.Series, TensorVariable] = None,
        T: Union[np.ndarray, pd.Series, TensorVariable] = None,
    ) -> xarray.DataArray:
        r"""
            Given :math:`recency`, :math:`frequency`, and :math:`T` for an individual customer, this method returns the
            expected number of future purchases across :math:`future_t` time periods.

            If no customer data is provided, probabilities for all customers in model fit dataset are returned.

            Calculate the expected number of repeat purchases up to time t for a
            randomly choose individual from the population, given they have
            purchase history (frequency, recency, T).

            See equation (41) from [2]_.

            Adapted from lifetimes package
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/pareto_nbd_fitter.py#L242
            Parameters
            ----------
            future_t: array_like
                times to calculate the expectation for.
            customer_id: array_like
                Customer labels.
            recency: array_like
                Number of time periods between the customer's first and most recent purchases.
            frequency: array_like
                Number of repeat purchases per customer.
            T: array_like
                Number of time periods since the customer's first purchase.
                Model assumptions require T >= recency.
        """

        x, t_x, T = self._process_customers(customer_id, frequency, recency, T)
        r, alpha, s, beta = self._unload_params()
        loglike = self._logp(r, alpha, s, beta, x, t_x, T)

        first_term = (
            gammaln(r + x)
            - gammaln(r)
            + r * log(alpha)
            + s * log(beta)
            - (r + x) * log(alpha + T)
            - s * log(beta + T)
        )
        second_term = log(r + x) + log(beta + T) - log(alpha + T)
        third_term = log(
            (1 - ((beta + T) / (beta + T + future_t)) ** (s - 1)) / (s - 1)
        )

        exp_purchases = exp(first_term + second_term + third_term - loglike)

        return exp_purchases.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_purchases_new_customer(
        self,
        t: Union[np.ndarray, pd.Series],
    ) -> xarray.DataArray:
        r"""
            Expected number of purchases for a new customer across :math:`t` time periods. See
            equation (27) of [2]_.

            http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf

            Adapted from lifetimes package
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/pareto_nbd_fitter.py#L359

            Parameters
            ----------
            t: array_like
                Number of time periods over which to estimate purchases.
        """
        r, alpha, s, beta = self._unload_params()

        t = np.asarray(t)

        r, alpha, s, beta = self._unload_params()  # type: ignore [has-type]
        first_term = r * beta / alpha / (s - 1)
        second_term = 1 - (beta / (beta + t)) ** (s - 1)

        return (first_term * second_term).transpose(
            "chain", "draw", "t", missing_dims="ignore"
        )

    def probability_alive(
        self,
        future_t: Union[int, float] = 0,
        customer_id: Union[np.ndarray, pd.Series] = None,
        recency: Union[np.ndarray, pd.Series, TensorVariable] = None,
        frequency: Union[np.ndarray, pd.Series, TensorVariable] = None,
        T: Union[np.ndarray, pd.Series, TensorVariable] = None,
    ) -> xarray.DataArray:
        r"""
        Compute the probability that a customer with history :math:`frequency`, :math:`recency`, and :math:`T`
        is currently alive. Can also estimate alive probability :math:`future_t` periods in the future.

        If no customer data is provided, probabilities for all customers in model fit dataset are returned.

        See equation (18) from [3]_.

        Parameters
        ----------
        future_t: scalar
            Number of time periods in the future to estimate alive probability; defaults to 0.
        customer_id: array_like
            Customer labels.
        recency: array_like
            Number of time periods between the customer's first and most recent purchases.
        frequency: array_like
            Number of repeat purchases per customer.
        T: array_like
            Number of time periods since the customer's first purchase.
            Model assumptions require T >= recency.
        """
        x, t_x, T = self._process_customers(customer_id, frequency, recency, T)
        r, alpha, s, beta = self._unload_params()  # type: ignore [has-type]
        loglike = self._logp(r, alpha, s, beta, x, t_x, T)

        term1 = gammaln(r + x) - gammaln(r)
        term2 = r * log(alpha / (alpha + T))
        term3 = -x * log(alpha + T)
        term4 = s * log(beta / (beta + T + future_t))

        prob_alive = exp(term1 + term2 + term3 + term4 - loglike)

        return prob_alive.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def purchase_probability(
        self,
        n_purchases: Union[int, np.ndarray, pd.Series, TensorVariable],
        future_t: Union[float, np.ndarray, pd.Series, TensorVariable],
        customer_id: Union[np.ndarray, pd.Series] = None,
        recency: Union[np.ndarray, pd.Series, TensorVariable] = None,
        frequency: Union[np.ndarray, pd.Series, TensorVariable] = None,
        T: Union[np.ndarray, pd.Series, TensorVariable] = None,
    ) -> xarray.DataArray:
        """
            Estimate probability of :math:`n_purchases` over :math:`future_t` time periods,
            given an individual customer's current :math:`frequency`, :math:`recency`, and :math:`T`.
            If no customer data is provided, probabilities for all customers in model fit dataset are returned.

            See equation (16) from [4]_.

            Adapted from lifetimes package
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/pareto_nbd_fitter.py#L388

            Parameters
            ----------
            n_purchases: int
                number of purchases predicted.
            future_t: a scalar
                time periods over which the probability should be calculated.
            customer_id: array_like
                Customer labels.
            recency: array_like
                Number of time periods between the customer's first and most recent purchases.
            frequency: array_like
                Number of repeat purchases per customer.
            T: array_like
                Number of time periods since the customer's first purchase.
                Model assumptions require T >= recency.
        """

        x, t_x, T = self._process_customers(customer_id, frequency, recency, T)  # type: ignore [has-type]
        r, alpha, s, beta = self._unload_params()  # type: ignore [has-type]
        loglike = self._logp(r, alpha, s, beta, x, t_x, T)

        _alpha_less_than_beta = alpha < beta
        min_of_alpha_beta = xarray.where(_alpha_less_than_beta, alpha, beta)
        max_of_alpha_beta = xarray.where(_alpha_less_than_beta, beta, alpha)
        p = xarray.where(_alpha_less_than_beta, r + x + n_purchases, s + 1)

        abs_alpha_beta = max_of_alpha_beta - min_of_alpha_beta

        log_p_zero = (
            gammaln(r + x)
            + r * log(alpha)
            + s * log(beta)
            - (gammaln(r) + (r + x) * log(alpha + T) + s * log(beta + T) + loglike)
        )
        log_B_one = (
            gammaln(r + x + n_purchases)
            + r * log(alpha)
            + s * log(beta)
            - (
                gammaln(r)
                + (r + x + n_purchases) * log(alpha + T + future_t)
                + s * log(beta + T + future_t)
            )
        )
        log_B_two = (
            r * log(alpha)
            + s * log(beta)
            + gammaln(r + s + x)
            + betaln(r + x + n_purchases, s + 1)
            + log(
                hyp2f1(
                    r + s + x,
                    p,
                    r + s + x + n_purchases + 1,
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
                + betaln(r + x + n_purchases, s + 1)
                + log(
                    hyp2f1(
                        r + s + x + i,
                        p,
                        r + s + x + n_purchases + 1,
                        abs_alpha_beta / (max_of_alpha_beta + T + future_t),
                    )
                )
                - (
                    gammaln(r)
                    + gammaln(s)
                    + (r + s + x + i) * log(max_of_alpha_beta + T + future_t)
                )
            )

        zeroth_term = (n_purchases == 0) * (1 - exp(log_p_zero))
        first_term = (
            n_purchases * log(future_t) - gammaln(n_purchases + 1) + log_B_one - loglike
        )
        second_term = log_B_two - loglike
        third_term = xr_logsumexp(
            xarray.concat(
                [
                    i * log(future_t) - gammaln(i + 1) + _log_B_three(i) - loglike
                    for i in range(n_purchases + 1)
                ],
                dim="concat_dim_",
            ),
            dims="concat_dim_",
        )

        purchase_prob = zeroth_term + exp(
            xr_logsumexp(
                xarray.concat([first_term, second_term, third_term], dim="_concat_dim"),
                b=xarray.DataArray(data=[1, 1, -1], dims="_concat_dim"),
                dims="_concat_dim",
            )
        )

        # TODO: Can this be done prior to performing the above calculations?
        if future_t <= 0:
            purchase_prob = purchase_prob.fillna(0)

        return purchase_prob.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )
