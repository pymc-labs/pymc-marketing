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
"""Shifted Beta Geometric model."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pymc as pm
import xarray
from pymc.util import RandomState
from pymc_extras.prior import Prior
from scipy.special import gammaln, hyp2f1

from pymc_marketing.clv.distributions import ShiftedBetaGeometric
from pymc_marketing.clv.models import CLVModel
from pymc_marketing.clv.utils import to_xarray
from pymc_marketing.model_config import ModelConfig


class ShiftedBetaGeoModel(CLVModel):
    """Shifted Beta Geometric (sBG) model for cohorts of customers renewing contracts across discrete time periods.

    The sBG model has the following assumptions:
      * At the end of each time period, each cohort has a probability `theta` of contract cancellation.
      * Cohort `theta` probabilities are Beta distributed with hyperparameters `alpha` and `beta`.
      * Cohort retention rates increase over time due to customer heterogeneity.

    This model requires data to be summarized by *recency*, and *T* for each customer,
    using `clv.utils.rfm_summary()` or equivalent. Modeling assumptions require *0 <= recency <= T*.
    If cohorts are not specified, the model will assume a single cohort,
    in which all customers began their contract in the same time period.

    First introduced by Fader & Hardie in [1]_, with additional predictive methods in [2]_.

    Parameters
    ----------
    data : ~pandas.DataFrame
        DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier
            * `recency`: Time period of last contract renewal.
              It should  equal T for users that have not cancelled by the end of the
              observation period.
            * `T`: Maximum observed time period.
              Model assumptions require *T >= recency* and all customers in the same cohort share the same value for *T.
            * `cohort`: Optional Customer cohort label. This is usually the date the customer first signed up.
    model_config : dict, optional
        Dictionary of model prior parameters:
            * `a`: Shape parameter of dropout process; defaults to `phi_purchase` * `kappa_purchase`
            * `b`: Shape parameter of dropout process; defaults to `1-phi_dropout` * `kappa_dropout`
            * `phi_dropout`: Nested prior for a and b priors; defaults to `Prior("Uniform", lower=0, upper=1)`
            * `kappa_dropout`: Nested prior for a and b priors; defaults to `Prior("Pareto", alpha=1, m=1)`
    sampler_config : dict, optional
        Dictionary of sampler parameters. Defaults to *None*.

    Examples
    --------
        .. code-block:: python

            import pymc as pm

            from pymc_extras.prior import Prior
            from pymc_marketing.clv import ShiftedBetaGeoModel

            model = ShiftedBetaGeoModel(
                data=pd.DataFrame({
                    customer_id=[1, 2, 3, ...],
                    recency=[1, 4, 8, ...],
                    T=[5, 5, 8, ...],
                    cohort= ["2025-04-01", "2025-04-01", "2025-02-01", ...],
                }),
                model_config={
                    "alpha": Prior("HalfNormal", sigma=10),
                    "beta": Prior("HalfStudentT", nu=4, sigma=10),
                },
                sampler_config={
                    "draws": 1000,
                    "tune": 1000,
                    "chains": 4,
                    "cores": 4,
                    "nuts_kwargs": {"target_accept": 0.95},
                },
            )

            model.fit()
            print(model.fit_summary())

            # Predict how many periods in the future are existing customers
            likely to cancel (ignoring that some may already have cancelled)
            expected_churn_time = model.expected_probability_alive(
                customer_id=[0, 1, 2, 3, ...],
            )
            print(expected_churn_time.mean("customer_id"))

    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2007). How to project customer retention.
           Journal of Interactive Marketing, 21(1), 76-90.
           https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
    .. [2] Fader, P. S., & Hardie, B. G. (2010). Customer-Base Valuation in a Contractual Setting:
           The Perils of Ignoring Heterogeneity. Marketing Science, 29(1), 85-93.
           https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_contractual_mksc_10.pdf
    """

    _model_type = "Shifted Beta-Geometric"

    def __init__(
        self,
        data: pd.DataFrame,
        model_config: ModelConfig | None = None,
        sampler_config: dict | None = None,
    ):
        self._validate_cols(
            data,
            required_cols=["customer_id", "recency", "T", "cohort"],
            must_be_unique=["customer_id"],
        )
        # TODO: Move into _validate_cols; this is true for all CLV models
        if np.any(
            (data["recency"] < 0)
            | (data["recency"] > data["T"])
            | np.isnan(data["recency"])
        ):
            raise ValueError(
                "recency must respect 0 < recency <= T.\n",
                "Customers that are still alive should have recency = T",
            )

        super().__init__(
            data=data, model_config=model_config, sampler_config=sampler_config
        )

        # Validate that provided Priors specify dims="cohort"
        # This ensures consistency with the cohort dimension handling across CLV models
        for key in ("alpha", "beta"):
            prior = self.model_config.get(key)
            if isinstance(prior, Prior):
                # Normalize dims to a tuple of strings for comparison
                dims = prior.dims
                if isinstance(dims, str):
                    dims_tuple = (dims,)
                else:
                    dims_tuple = tuple(dims) if dims is not None else tuple()

                if "cohort" not in dims_tuple:
                    raise ValueError(
                        f"ModelConfig Prior for '{key}' must include dims=\"cohort\". "
                        f'Got dims={prior.dims!r}. Example: Prior("HalfFlat", dims="cohort").'
                    )
        # TODO: Add a check for the "cohort" dimension, or just a check for the column?
        if "cohort" in self.data.columns:
            # create cohort dim & coords
            self.cohorts = self.data["cohort"].unique()
            self.cohort_idx = pd.Categorical(
                self.data["cohort"], categories=self.cohorts
            ).codes
        else:
            # create single cohort to preserve dimension handling
            self.cohorts = np.ones(1, dtype=np.int8)
            self.cohort_idx = np.ones(len(self.data), dtype=np.int8)

    @property
    def default_model_config(self) -> ModelConfig:
        """Default model configuration."""
        return {
            "phi": Prior("Uniform", lower=0, upper=1, dims="cohort"),
            "kappa": Prior("Pareto", alpha=1, m=1, dims="cohort"),
        }

    def build_model(self) -> None:  # type: ignore[override]
        """Build the model."""
        coords = {
            "customer_id": self.data["customer_id"],
            "cohort": self.cohorts,
        }
        with pm.Model(coords=coords) as self.model:
            if "alpha" in self.model_config and "beta" in self.model_config:
                alpha = self.model_config["alpha"].create_variable("alpha")
                beta = self.model_config["beta"].create_variable("beta")
            else:
                # hierarchical pooling of purchase rate priors
                phi = self.model_config["phi"].create_variable("phi")
                kappa = self.model_config["kappa"].create_variable("kappa")

                alpha = pm.Deterministic("alpha", phi * kappa, dims="cohort")
                beta = pm.Deterministic("beta", (1.0 - phi) * kappa, dims="cohort")

            dropout = ShiftedBetaGeometric.dist(
                alpha[self.cohort_idx],
                beta[self.cohort_idx],
            )

            pm.Censored(
                "churn_censored",
                dropout,
                lower=None,
                upper=self.data["T"],
                observed=self.data["recency"],
                dims=("customer_id",),
            )

    # TODO: move this into BaseModel
    def _extract_predictive_variables(
        self,
        data: pd.DataFrame,
        customer_varnames: Sequence[str] = (),
    ) -> xarray.Dataset:
        """
        Extract predictive variables from the data.

        Utility function assigning default customer arguments
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
        # TODO: Move into _validate_cols; this is true for all CLV models
        if np.any(
            (data["recency"] < 0)
            | (data["recency"] > data["T"])
            | np.isnan(data["recency"])
        ):
            raise ValueError(
                "recency must respect 0 < recency <= T.\n",
                "Customers that are still alive should have recency = T",
            )

        alpha = self.fit_result["alpha"]
        beta = self.fit_result["beta"]

        customer_vars = to_xarray(
            data["customer_id"],
            *[data[customer_varname] for customer_varname in customer_varnames],
        )
        if len(customer_varnames) == 1:
            customer_vars = [customer_vars]

        return xarray.combine_by_coords(
            (
                alpha,
                beta,
                *customer_vars,
            )
        )

    def expected_retention_rate(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        """Compute expected retention rate.

        Adapted from equation (8) in [1]_.

        References
        ----------
        .. [1] Fader, P. S., & Hardie, B. G. (2007). How to project customer retention.
            Journal of Interactive Marketing, 21(1), 76-90.
            https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
        """
        if data is None:
            data = self.data

        if future_t is not None:
            data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["T", "future_t"]
        )
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        T = dataset["T"]
        future_t = dataset["future_t"]

        # TODO add numpy.arange(T) for loop by cohort
        retention_rate = (beta + T - 1) / (alpha + beta + T - 1)

        return retention_rate.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_probability_alive(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        """Compute expected probability of being alive.

        Adapted from equation (6) in [1]_.

        References
        ----------
        .. [1] Fader, P. S., & Hardie, B. G. (2007). How to project customer retention.
            Journal of Interactive Marketing, 21(1), 76-90.
            https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
        """
        if data is None:
            data = self.data

        if future_t is not None:
            data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["recency", "T", "future_t"]
        )
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        T = dataset["T"]
        future_t = dataset["future_t"]

        # Rewrite beta functions from paper in terms of gamma functions on log scale
        logS = (
            gammaln(beta + T)
            - gammaln(beta)
            + gammaln(alpha + beta)
            - gammaln(alpha + beta + T)
        )
        survival = np.exp(logS)

        return survival.transpose("chain", "draw", "customer_id", missing_dims="ignore")

    def expected_retention_elasticity(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
        discount_rate: float = 0.0,
    ) -> xarray.DataArray:
        """Compute expected retention elasticity.

        Adapted from equation (8) in [1]_.

        References
        ----------
        .. [1] Fader, P. S., & Hardie, B. G. (2010). Customer-Base Valuation in a Contractual Setting:
               The Perils of Ignoring Heterogeneity. Marketing Science, 29(1), 85-93.
               https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_contractual_mksc_10.pdf
        """
        if data is None:
            data = self.data

        if future_t is not None:
            data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["recency", "T", "future_t"]
        )
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        T = dataset["T"]
        future_t = dataset["future_t"]

        retention_elasticity = hyp2f1(
            1, beta + T - 1, alpha + beta + 1, 1 / (1 + discount_rate)
        )

        return retention_elasticity.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_lifetime_purchases(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
        discount_rate: float = 0.0,
    ) -> xarray.DataArray:
        """Compute expected lifetime purchases.

        Adapted from equation (6) in [1]_.

        References
        ----------
        .. [1] Fader, P. S., & Hardie, B. G. (2010). Customer-Base Valuation in a Contractual Setting:
               The Perils of Ignoring Heterogeneity. Marketing Science, 29(1), 85-93.
               https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_contractual_mksc_10.pdf
        """
        if data is None:
            data = self.data

        if future_t is not None:
            data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["recency", "T", "future_t"]
        )
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        T = dataset["T"]
        future_t = dataset["future_t"]

        retention_rate = (beta + T - 1) / (alpha + beta + T - 1)
        retention_elasticity = hyp2f1(
            1, beta + T, alpha + beta, 1 / (1 + discount_rate)
        )
        expected_lifetime_purchases = retention_rate * retention_elasticity

        return expected_lifetime_purchases.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    # TODO: Update to support both prior and posterior distributions
    def distribution_cohort_churn(
        self, customer_id: np.ndarray | pd.Series, random_seed: RandomState = None
    ) -> xarray.DataArray:
        """Sample distribution of dropout process for existing customers.

        The draws represent the distribution of dropout probabilities by cohort.

        """
        # create cohort indices from data
        cohorts = self.data["cohort"].unique()
        cohort_idx = pd.Categorical(self.data["cohort"], categories=cohorts).codes

        coords = {
            "customer_id": self.data["customer_id"],
            "cohort": cohorts,
        }
        with pm.Model(coords=coords):
            alpha = pm.HalfFlat("alpha", dims="cohort")
            beta = pm.HalfFlat("beta", dims="cohort")

            # TODO: This is the distribution of interest; resolve mypy complaints
            # theta = pm.Beta("theta", alpha[cohort_idx], beta[cohort_idx], dims="cohort")

            # TODO: This may not be needed here as we are sampling theta
            ShiftedBetaGeometric(
                "churn", alpha[cohort_idx], beta[cohort_idx], dims=("customer_id",)
            )

            return pm.sample_posterior_predictive(
                self.idata,
                var_names=["theta"],
                random_seed=random_seed,
            ).posterior_predictive["theta"]


class ShiftedBetaGeoModelIndividual(CLVModel):
    """Shifted Beta Geometric model for individual customers.

    Model for customer behavior in a discrete contractual setting. It assumes that:
      * At the end of each period, a customer has a probability `theta` of renewing the contract
        and `1-theta` of cancelling
      * The probability `theta` does not change over time for a given customer
      * The probability `theta` varies across customers according to a Beta prior distribution
        with hyperparameters `alpha` and `beta`.

    based on [1]_.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the following columns:
            * `customer_id`: Customer labels. There should be one unique label for each customer
            * `t_churn`: Time at which the customer cancelled the contract (starting at 0).
        It should  equal T for users that have not cancelled by the end of the
        observation period
            * `T`: Maximum observed time period (starting at 0)
    model_config: dict, optional
        Dictionary of model prior parameters. If not provided, the model will use default priors specified in the
        `default_model_config` class attribute.
    sampler_config: dict, optional
        Dictionary of sampler parameters. Defaults to None.


    Examples
    --------
        .. code-block:: python

            import pymc as pm

            from pymc_extras.prior import Prior
            from pymc_marketing.clv import ShiftedBetaGeoModelIndividual

            model = ShiftedBetaGeoModelIndividual(
                data=pd.DataFrame({
                    customer_id=[0, 1, 2, 3, ...],
                    t_churn=[1, 2, 8, 4, 8 ...],
                    T=[8 for x in range(len(customer_id))],
                }),
                model_config={
                    "alpha": Prior("HalfNormal", sigma=10),
                    "beta": Prior("HalfStudentT", nu=4, sigma=10),
                },
                sampler_config={
                    "draws": 1000,
                    "tune": 1000,
                    "chains": 2,
                    "cores": 2,
                    "nuts_kwargs": {"target_accept": 0.95},
                },
            )

            model.fit()
            print(model.fit_summary())

            # Predict how many periods in the future are existing customers
            likely to cancel (ignoring that some may already have cancelled)
            expected_churn_time = model.distribution_customer_churn_time(
                customer_id=[0, 1, 2, 3, ...],
            )
            print(expected_churn_time.mean("customer_id"))

            # Predict churn time for 10 new customers, conditioned on data
            new_customers_churn_time = model.distribution_new_customer_churn_time(n=10)
            print(new_customers_churn_time.mean("new_customer_id"))


    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2007). How to project customer retention.
           Journal of Interactive Marketing, 21(1), 76-90.
           https://journals.sagepub.com/doi/pdf/10.1002/dir.20074

    """

    _model_type = "Shifted-Beta-Geometric Model (Individual Customers)"

    def __init__(
        self,
        data: pd.DataFrame,
        model_config: ModelConfig | None = None,
        sampler_config: dict | None = None,
    ):
        self._validate_cols(
            data,
            required_cols=["customer_id", "t_churn", "T"],
            must_be_unique=["customer_id"],
        )

        if np.any(
            (data["t_churn"] < 0)
            | (data["t_churn"] > data["T"])
            | np.isnan(data["t_churn"])
        ):
            raise ValueError(
                "t_churn must respect 0 < t_churn <= T.\n",
                "Customers that are still alive should have t_churn = T",
            )
        super().__init__(
            data=data, model_config=model_config, sampler_config=sampler_config
        )

    @property
    def default_model_config(self) -> dict:
        """Default model configuration."""
        return {
            "alpha": Prior("HalfFlat"),
            "beta": Prior("HalfFlat"),
        }

    def build_model(self) -> None:  # type: ignore[override]
        """Build the model."""
        coords = {"customer_id": self.data["customer_id"]}
        with pm.Model(coords=coords) as self.model:
            alpha = self.model_config["alpha"].create_variable("alpha")
            beta = self.model_config["beta"].create_variable("beta")

            theta = pm.Beta("theta", alpha, beta, dims=("customer_id",))

            churn_raw = pm.Geometric.dist(theta)
            pm.Censored(
                "churn_censored",
                churn_raw,
                lower=None,
                upper=self.data["T"],
                observed=self.data["t_churn"],
                dims=("customer_id",),
            )

    def distribution_customer_churn_time(
        self, customer_id: np.ndarray | pd.Series, random_seed: RandomState = None
    ) -> xarray.DataArray:
        """Sample distribution of churn time for existing customers.

        The draws represent the number of periods into the future after which
        a customer cancels their contract.

        It ignores that some customers may have already cancelled.
        """
        coords = {"customer_id": customer_id}
        with pm.Model(coords=coords):
            alpha = pm.HalfFlat("alpha")
            beta = pm.HalfFlat("beta")

            theta = pm.Beta("theta", alpha, beta, dims=("customer_id",))
            pm.Geometric("churn", theta, dims=("customer_id",))

            return pm.sample_posterior_predictive(
                self.idata,
                var_names=["churn"],
                random_seed=random_seed,
            ).posterior_predictive["churn"]

    def _distribution_new_customer(
        self,
        n: int = 1,
        random_seed: RandomState = None,
        var_names: Sequence[str] = ("theta", "churn"),
    ) -> xarray.Dataset:
        coords = {"new_customer_id": np.arange(n)}
        with pm.Model(coords=coords):
            alpha = pm.HalfFlat("alpha")
            beta = pm.HalfFlat("beta")

            theta = pm.Beta("theta", alpha, beta, dims=("new_customer_id",))
            pm.Geometric("churn", theta, dims=("new_customer_id",))

            return pm.sample_posterior_predictive(
                self.idata,
                var_names=var_names,
                random_seed=random_seed,
            ).posterior_predictive

    def distribution_new_customer_churn_time(
        self, n: int = 1, random_seed: RandomState = None
    ) -> xarray.DataArray:
        """Sample distribution of churn time for new customers.

        The draws represent the number of periods into the future after which
        a customer cancels their contract.

        Use `n > 1` to simulate multiple identically distributed users.
        """
        return self._distribution_new_customer(
            n=n, random_seed=random_seed, var_names=["churn"]
        )["churn"]

    def distribution_new_customer_theta(
        self, n: int = 1, random_seed: RandomState = None
    ) -> xarray.DataArray:
        """Sample distribution of theta parameter for new customers.

        Use `n > 1` to simulate multiple identically distributed users.
        """
        return self._distribution_new_customer(
            n=n, random_seed=random_seed, var_names=["theta"]
        )["theta"]
