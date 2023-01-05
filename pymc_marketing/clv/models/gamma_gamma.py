import types
from typing import Optional, Union

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc import str_for_dist
from pytensor.tensor import TensorVariable

from pymc_marketing.clv.models.basic import CLVModel


class BaseGammaGammaModel(CLVModel):
    def _process_priors(self, p_prior, q_prior, v_prior):
        if p_prior is None:
            p_prior = pm.HalfFlat.dist()
        else:
            assert p_prior.eval().shape == ()
        if q_prior is None:
            q_prior = pm.HalfFlat.dist()
        else:
            assert q_prior.eval().shape == ()
        if v_prior is None:
            v_prior = pm.HalfFlat.dist()
        else:
            assert v_prior.eval().shape == ()

        assert p_prior is not q_prior
        assert p_prior is not v_prior
        assert q_prior is not v_prior

        # Who had the idea of attaching methods to pre-existing objects?
        p_prior.str_repr = types.MethodType(str_for_dist, p_prior)
        q_prior.str_repr = types.MethodType(str_for_dist, q_prior)
        v_prior.str_repr = types.MethodType(str_for_dist, v_prior)

        return p_prior, q_prior, v_prior

    def expected_customer_spend(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        mean_transaction_value: Union[np.ndarray, pd.Series, TensorVariable],
        frequency: Union[np.ndarray, pd.Series, TensorVariable],
        random_seed=None,
    ):
        """Return distribution of expected transaction value per customer, based on
        observed individual or mean transaction values"""

        x = frequency
        z_mean = mean_transaction_value

        coords = {"customer_id": np.unique(customer_id)}
        with pm.Model(coords=coords):
            p = pm.HalfFlat("p")
            q = pm.HalfFlat("q")
            v = pm.HalfFlat("v")

            # Closed form solution to the posterior of nu
            # Eq 5 from [1], p.3
            nu = pm.Gamma("nu", p * x + q, v + x * z_mean, dims=("customer_id",))
            pm.Deterministic("mean_spend", p / nu, dims=("customer_id",))

            return pm.sample_posterior_predictive(
                self.fit_result,
                var_names=["nu", "mean_spend"],
                random_seed=random_seed,
            ).posterior_predictive["mean_spend"]

    def expected_new_customer_spend(self, n=1, random_seed=None):
        """Predict expected transaction value of new unobserved customers"""
        coords = {"new_customer_id": range(n)}
        with pm.Model(coords=coords):
            p = pm.HalfFlat("p")
            q = pm.HalfFlat("q")
            v = pm.HalfFlat("v")

            nu = pm.Gamma("nu", q, v, dims=("new_customer_id",))
            pm.Deterministic("mean_spend", p / nu, dims=("new_customer_id",))

            return pm.sample_posterior_predictive(
                self.fit_result,
                var_names=["nu", "mean_spend"],
                random_seed=random_seed,
            ).posterior_predictive["mean_spend"]


class GammaGammaModel(BaseGammaGammaModel):
    """Gamma-Gamma model

    Estimate the average monetary value of customer transactions.

    The model is conditioned on the mean transaction value of each user, and is based
    on [1]_, [2]_.

    TODO: Explain assumptions of model

    Check `GammaGammaModelIndividual` for an equivalent model conditioned
    on individual transaction values.

    Parameters
    ----------
    customer_id: array_like
        Customer labels. Must not repeat.
    mean_transaction_value: array_like
        Mean transaction value of each customer.
    frequency: array_like
        Number of transactions observed for each customer.
    p_prior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`
    q_prior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`
    v_prior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`

    Examples
    --------
        Gamma-Gamma model condioned on mean transaction value

        .. code-block:: python

            import pymc as pm
            from pymc_marketing.clv import GammaGammaModel

            model = GammaGammaModel(
                customer_id=[0, 1, 2, 3, ...],
                mean_transactionn_value=[23.5, 19.3, 11.2, 100.5, ...],
                frequency=[6, 8, 2, 1, ...],
                p_prior=pm.HalfNormal.dist(10),
                q_prior=pm.HalfNormal.dist(10),
                v_prior=pm.HalfNormal.dist(10),
            )

            model.fit()
            print(model.fit_summary())

            # Predict spend of customers for which we know transaction history,
            # conditioned on data. May include customers not included in fitting
            expected_customer_spend = model.expected_customer_spend(
                customer_id=[4, 5, 6, 7, ...],
                mean_transactionn_value=[2.3, 5.9, 221, 3.0, ...],
                frequency=[3, 4, 6, 6, ...],
            )
            print(expected_customer_spend.mean("customer_id"))

            # Predict spend of 10 new customers, conditioned on data
            new_customer_spend = model.expected_new_customer_spend(n=10)
            print(new_customer_spend.mean("customer_id"))

    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2013). The Gamma-Gamma model of monetary
           value. February, 2, 1-9. http://www.brucehardie.com/notes/025/gamma_gamma.pdf
    .. [2] Peter S. Fader, Bruce G. S. Hardie, and Ka Lok Lee (2005), “RFM and CLV:
           Using iso-value curves for customer base analysis”, Journal of Marketing
           Research, 42 (November), 415-430.
           https://journals.sagepub.com/doi/pdf/10.1509/jmkr.2005.42.4.415
    """

    _model_name = "Gamma-Gamma Model (Mean Transactions)"

    def __init__(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        mean_transaction_value: Union[np.ndarray, pd.Series, TensorVariable],
        frequency: Union[np.ndarray, pd.Series, TensorVariable],
        p_prior: Optional[TensorVariable] = None,
        q_prior: Optional[TensorVariable] = None,
        v_prior: Optional[TensorVariable] = None,
    ):
        super().__init__()

        p_prior, q_prior, v_prior = self._process_priors(p_prior, q_prior, v_prior)

        z_mean = pt.as_tensor_variable(mean_transaction_value)
        x = pt.as_tensor_variable(frequency)

        coords = {"customer_id": np.unique(customer_id)}
        with pm.Model(coords=coords) as self.model:
            p = self.model.register_rv(p_prior, name="p")
            q = self.model.register_rv(q_prior, name="q")
            v = self.model.register_rv(v_prior, name="v")

            # Likelihood for mean_spend, marginalizing over nu
            # Eq 1a from [1], p.2
            pm.Potential(
                "likelihood",
                (
                    pt.gammaln(p * x + q)
                    - pt.gammaln(p * x)
                    - pt.gammaln(q)
                    + q * pt.log(v)
                    + (p * x - 1) * pt.log(z_mean)
                    + (p * x) * pt.log(x)
                    - (p * x + q) * pt.log(x * z_mean + v)
                ),
            )


class GammaGammaModelIndividual(BaseGammaGammaModel):
    """Gamma-Gamma model

    Estimate the average monetary value of customer transactions.

    The model is conditioned on the individual transaction values per user, and is based
    on [1]_, [2]_.

    TODO: Explain assumptions of model

    Check `GammaGammaModel` for an equivalent model conditioned on the average
    transaction value of each user.

    Parameters
    ----------
    customer_id: array_like
        Customer labels. The same value should be used for each observation
        coming from the same customer.
    individual_transaction_value: array_like
        Value of individual transactions.
    p_prior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`
    q_prior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`
    v_prior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`


    Examples
    --------

        Gamma-Gamma model conditioned on individual customer spend

        .. code-block:: python

            import pymc as pm
            from pymc_marketing.clv import GammaGammaModel

            model = GammaGammaModel(
                customer_id=[0, 0, 0, 1, 1, 2, ...],
                individual_transaction_value=[5.3. 5.7, 6.9, 13.5, 0.3, 19.2 ...],
            )

            model.fit()
            print(model.fit_summary())

            # Predict spend of customers for which we know transaction history,
            # conditioned on data. May include customers not included in fitting
            expected_customer_spend = model.expected_customer_spend(
                customer_id=[0, 0, 0, 1, 1, 2, ...],
                individual_transaction_value=[5.3. 5.7, 6.9, 13.5, 0.3, 19.2 ...],
            )
            print(expected_customer_spend.mean("customer_id"))

            # Predict spend of 10 new customers, conditioned on data
            new_customer_spend = model.expected_new_customer_spend(n=10)
            print(new_customer_spend.mean("customer_id"))


    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2013). The Gamma-Gamma model of monetary
           value. February, 2, 1-9. http://www.brucehardie.com/notes/025/gamma_gamma.pdf
    .. [2] Peter S. Fader, Bruce G. S. Hardie, and Ka Lok Lee (2005), “RFM and CLV:
           Using iso-value curves for customer base analysis”, Journal of Marketing
           Research, 42 (November), 415-430.
           https://journals.sagepub.com/doi/pdf/10.1509/jmkr.2005.42.4.415
    """

    _model_name = "Gamma-Gamma Model (Individual Transactions)"

    def __init__(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        individual_transaction_value: Union[np.ndarray, pd.Series, TensorVariable],
        p_prior: Optional[TensorVariable] = None,
        q_prior: Optional[TensorVariable] = None,
        v_prior: Optional[TensorVariable] = None,
    ):
        super().__init__()

        p_prior, q_prior, v_prior = self._process_priors(p_prior, q_prior, v_prior)
        z = individual_transaction_value

        coords = {
            "customer_id": np.unique(customer_id),
            "obs": range(len(customer_id)),
        }
        with pm.Model(coords=coords) as self.model:
            p = self.model.register_rv(p_prior, name="p")
            q = self.model.register_rv(q_prior, name="q")
            v = self.model.register_rv(v_prior, name="v")

            nu = pm.Gamma("nu", q, v, dims=("customer_id",))
            pm.Gamma("spend", p, nu[customer_id], observed=z, dims=("obs",))

    def expected_customer_spend(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        individual_transaction_value: Union[np.ndarray, pd.Series, TensorVariable],
        random_seed=None,
    ):
        """Return distribution of expected transaction value per customer, based on
        observed individual or mean transaction values"""

        df = pd.DataFrame(
            {
                "customer_id": customer_id,
                "individual_transaction_value": individual_transaction_value,
            }
        )
        gdf = df.groupby("customer_id")["individual_transaction_value"].aggregate(
            ("count", "mean")
        )
        customer_id = gdf.index
        x = gdf["count"]
        z_mean = gdf["mean"]

        return super().expected_customer_spend(
            customer_id=customer_id,
            mean_transaction_value=z_mean,
            frequency=x,
            random_seed=random_seed,
        )
