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
from scipy.special import gammaln

from pymc_marketing.clv.distributions import ShiftedBetaGeometric
from pymc_marketing.clv.models import CLVModel
from pymc_marketing.clv.utils import to_xarray
from pymc_marketing.model_config import ModelConfig

__all__ = ["ShiftedBetaGeoModel", "ShiftedBetaGeoModelIndividual"]


class ShiftedBetaGeoModel(CLVModel):
    """Shifted Beta Geometric (sBG) model for cohorts of customers renewing contracts across discrete time periods.

    The sBG model has the following assumptions:
      * At the end of each time period, each cohort has a probability `theta` of contract cancellation.
      * Cohort `theta` probabilities are Beta distributed with hyperparameters `alpha` and `beta`.
      * Cohort retention rates increase over time due to customer heterogeneity.

    This model requires data to be summarized by *recency*, and *T* for each customer.
    Modeling assumptions require *1 <= recency <= T*.
    If cohorts are not specified, the model will assume a single cohort,
    in which all customers began their contract in the same time period.

    First introduced by Fader & Hardie in [1]_.

    Parameters
    ----------
    data : ~pandas.DataFrame
        DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier
            * `recency`: Time period of last contract renewal. It should equal *T* for active customers.
            * `T`: Maximum observed time period in the cohort. Model assumptions require *T >= recency >= 1*.
            All customers in a given cohort share the same value for *T*.
            * `cohort`: Customer cohort label. This is usually the month or year the customer first signed up.
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
                data=pd.DataFrame(
                    customer_id=[1, 2, 3, ...],
                    recency=[8, 1, 4, ...],
                    T=[8, 5, 5, ...],
                    cohort=["2025-02-01", "2025-04-01", "2025-04-01", ...],
                ),
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

            # Fit model quickly to large datasets via the default Maximum a Posteriori method
            model.fit(method="map")
            model.fit_summary()

            # Use 'mcmc' for more informative predictions and reliable performance on smaller datasets
            model.fit(method="mcmc")
            model.fit_summary()

            # Predict likelihood active customers will renew in the next time period
            expected_alive_probability = model.expected_probability_alive(
                active_customers  # ADD CREATE DATAFRAME HERE!
            )
            print(expected_churn_time.mean("customer_id"))

    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2007). How to project customer retention.
           Journal of Interactive Marketing, 21(1), 76-90.
           https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
    """

    _model_type = "Shifted Beta-Geometric"

    def __init__(
        self,
        data: pd.DataFrame,
        model_config: ModelConfig | None = None,
        sampler_config: dict | None = None,
    ):
        # TODO: Should add homogeneity check for T per cohort, but groupings not supported in this method
        self._validate_cols(
            data,
            required_cols=["customer_id", "recency", "T", "cohort"],
            must_be_unique=["customer_id"],
        )
        # TODO: Create another internal validation method in CLVBasic containing this logic
        # TODO: This function should also check fit data contains both active and inactive customers
        if np.any(
            (data["recency"] < 1)
            | (data["recency"] > data["T"])
            | np.isnan(data["recency"])
        ):
            raise ValueError(
                "recency must respect 0 < recency <= T.\n",
                "Customers still active should have recency = T",
            )

        super().__init__(
            data=data,
            model_config=model_config,
            sampler_config=sampler_config,
            non_distributions=[
                "alpha_covariate_cols",
                "beta_covariate_cols",
            ],
        )

        # Parse covariate configuration (defaults handled by parse_model_config)
        self.alpha_covariate_cols = list(
            self.model_config.get("alpha_covariate_cols", [])
        )
        self.beta_covariate_cols = list(
            self.model_config.get("beta_covariate_cols", [])
        )

        # Only validate prior dims when not using covariates
        if not (self.alpha_covariate_cols or self.beta_covariate_cols):
            # TODO: Could this be cleaned up? Or a separate _validate_prior_dims() method created?
            # Validate provided Priors specify dims="cohort"
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

        # create cohort dim & coords
        self.cohorts = self.data["cohort"].unique()
        self.cohort_idx = pd.Categorical(
            self.data["cohort"], categories=self.cohorts
        ).codes

    @property
    def default_model_config(self) -> ModelConfig:
        """Default model configuration."""
        return {
            # Cohort-level hierarchical defaults (no covariates)
            "phi": Prior("Uniform", lower=0, upper=1, dims="cohort"),
            "kappa": Prior("Pareto", alpha=1, m=1, dims="cohort"),
            # Customer-level covariates (optional)
            "alpha_covariate_cols": [],
            "beta_covariate_cols": [],
            "alpha_coefficient": Prior("Normal", mu=0, sigma=1),
            "beta_coefficient": Prior("Normal", mu=0, sigma=1),
            "alpha_scale": Prior("HalfFlat"),
            "beta_scale": Prior("HalfFlat"),
        }

    def build_model(self) -> None:  # type: ignore[override]
        """Build the model."""
        coords = {
            "customer_id": self.data["customer_id"],
            "cohort": self.cohorts,
        }
        if self.alpha_covariate_cols:
            coords["alpha_covariate"] = self.alpha_covariate_cols
        if self.beta_covariate_cols:
            coords["beta_covariate"] = self.beta_covariate_cols
        with pm.Model(coords=coords) as self.model:
            # Customer-level covariates path (alpha, beta directly)
            if self.alpha_covariate_cols or self.beta_covariate_cols:
                if self.alpha_covariate_cols:
                    alpha_data = pm.Data(
                        "alpha_data",
                        self.data[self.alpha_covariate_cols],
                        dims=["customer_id", "alpha_covariate"],
                    )
                    self.model_config["alpha_coefficient"].dims = "alpha_covariate"
                    alpha_coefficient = self.model_config[
                        "alpha_coefficient"
                    ].create_variable("alpha_coefficient")
                    alpha_scale = self.model_config["alpha_scale"].create_variable(
                        "alpha_scale"
                    )
                    alpha = pm.Deterministic(
                        "alpha",
                        alpha_scale
                        * pm.math.exp(pm.math.dot(alpha_data, alpha_coefficient)),
                        dims="customer_id",
                    )
                else:
                    # No alpha covariates: allow alpha to be cohort-level or direct prior if provided
                    if "alpha" in self.model_config:
                        alpha = self.model_config["alpha"].create_variable("alpha")
                    else:
                        phi = self.model_config["phi"].create_variable("phi")
                        kappa = self.model_config["kappa"].create_variable("kappa")
                        alpha = pm.Deterministic("alpha", phi * kappa, dims="cohort")

                if self.beta_covariate_cols:
                    beta_data = pm.Data(
                        "beta_data",
                        self.data[self.beta_covariate_cols],
                        dims=["customer_id", "beta_covariate"],
                    )
                    self.model_config["beta_coefficient"].dims = "beta_covariate"
                    beta_coefficient = self.model_config[
                        "beta_coefficient"
                    ].create_variable("beta_coefficient")
                    beta_scale = self.model_config["beta_scale"].create_variable(
                        "beta_scale"
                    )
                    beta = pm.Deterministic(
                        "beta",
                        beta_scale
                        * pm.math.exp(pm.math.dot(beta_data, beta_coefficient)),
                        dims="customer_id",
                    )
                else:
                    if "beta" in self.model_config:
                        beta = self.model_config["beta"].create_variable("beta")
                    else:
                        phi = self.model_config["phi"].create_variable("phi")
                        kappa = self.model_config["kappa"].create_variable("kappa")
                        beta = pm.Deterministic(
                            "beta", (1.0 - phi) * kappa, dims="cohort"
                        )

                # Broadcast alpha/beta to customer dimension if they are cohort-dimensioned
                if "cohort" in self.model.named_vars_to_dims.get("alpha", []):
                    alpha = alpha[self.cohort_idx]
                if "cohort" in self.model.named_vars_to_dims.get("beta", []):
                    beta = beta[self.cohort_idx]

                dropout = ShiftedBetaGeometric.dist(
                    alpha,
                    beta,
                )
            else:
                # Original cohort-level behavior
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
                "dropout",
                dropout,
                lower=None,
                upper=self.data["T"],
                observed=self.data["recency"],
                dims=("customer_id",),
            )

    # TODO: can this be generalized and moved into BaseModel?
    def _extract_predictive_variables(
        self,
        pred_data: pd.DataFrame | None = None,
        customer_varnames: Sequence[str] = (),
    ) -> xarray.Dataset:
        """
        Extract predictive variables from the data.

        Utility function assigning default customer arguments
        for predictive methods and converting to xarrays.
        """
        if pred_data is None:
            # Filter to active customers only (recency == T)
            pred_data = self.data.loc[self.data["recency"] == self.data["T"]].copy()
        else:
            self._validate_cols(
                pred_data,
                required_cols=[
                    "customer_id",
                    "cohort",
                    *customer_varnames,
                ],
                must_be_unique=["customer_id"],
            )
            # Validate recency only if provided in the input data
            if "recency" in pred_data.columns:
                # Base validity check
                if np.any(
                    (pred_data["recency"] < 0)
                    | (pred_data["recency"] > pred_data["T"])
                    | np.isnan(pred_data["recency"])
                ):
                    raise ValueError(
                        "recency must respect 0 < recency <= T.\n",
                        "Customers still active should have recency = T",
                    )
                # External data must be active customers only
                if np.any(pred_data["recency"] != pred_data["T"]):
                    raise ValueError(
                        "Predictions require active customers: recency must equal T."
                    )

        alpha = self.fit_result["alpha"]
        beta = self.fit_result["beta"]

        # Covariate path: rebuild alpha, beta for provided data
        if self.alpha_covariate_cols or self.beta_covariate_cols:
            model_coords = self.model.coords
            if self.alpha_covariate_cols:
                alpha_x = xarray.DataArray(
                    pred_data[self.alpha_covariate_cols],
                    dims=["customer_id", "alpha_covariate"],
                    coords=[
                        pred_data["customer_id"],
                        list(model_coords.get("alpha_covariate", [])),
                    ],
                )
                alpha_scale = self.fit_result.get("alpha_scale", None)
                alpha_coef = self.fit_result.get("alpha_coefficient", None)
                if (alpha_scale is not None) and (alpha_coef is not None):
                    alpha = alpha_scale * np.exp(
                        xarray.dot(alpha_x, alpha_coef, dim="alpha_covariate")
                    )
                    alpha.name = "alpha"
            if self.beta_covariate_cols:
                beta_x = xarray.DataArray(
                    pred_data[self.beta_covariate_cols],
                    dims=["customer_id", "beta_covariate"],
                    coords=[
                        pred_data["customer_id"],
                        list(model_coords.get("beta_covariate", [])),
                    ],
                )
                beta_scale = self.fit_result.get("beta_scale", None)
                beta_coef = self.fit_result.get("beta_coefficient", None)
                if (beta_scale is not None) and (beta_coef is not None):
                    beta = beta_scale * np.exp(
                        xarray.dot(beta_x, beta_coef, dim="beta_covariate")
                    )
                    beta.name = "beta"

        # Map from training data using customer_id if cohorts not passed explicitly
        # TODO: customer_ids are required for static covariate broadcasting, which is not yet supported
        # TODO: also need to check if external data cohorts and build_model cohorts match
        if len(self.cohorts) > 1:
            cohort_map = pred_data.set_index("customer_id")["cohort"]
            customer_cohorts = pred_data["customer_id"].map(cohort_map)
        else:
            # Single cohort case: broadcast the first cohort label
            single_label = self.cohorts[0] if hasattr(self, "cohort") else 0
            customer_cohorts = pd.Series(
                [single_label] * len(pred_data["customer_id"]), index=pred_data.index
            )

        customer_cohorts_xr = xarray.DataArray(
            customer_cohorts.values,
            dims=("customer_id",),
            coords={"customer_id": pred_data["customer_id"].values},
        )

        # Vectorized selection depends on whether alpha/beta are cohort- or customer-level
        if "cohort" in alpha.dims:
            alpha_customer = alpha.sel(cohorts=customer_cohorts_xr)
        else:
            alpha_customer = alpha
        if "cohort" in beta.dims:
            beta_customer = beta.sel(cohorts=customer_cohorts_xr)
        else:
            beta_customer = beta

        # Convert additional customer variables to xarray
        customer_vars = to_xarray(
            pred_data["customer_id"],
            *[pred_data[customer_varname] for customer_varname in customer_varnames],
        )
        if len(customer_varnames) == 1:
            customer_vars = [customer_vars]

        return xarray.combine_by_coords(
            (
                alpha_customer,
                beta_customer,
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

        retention_rate = (beta + T + future_t - 1) / (alpha + beta + T + future_t - 1)
        # TODO: "cohort" dim instead of "customer_id"?
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
            data, customer_varnames=["T", "future_t"]
        )
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        T = dataset["T"]
        future_t = dataset["future_t"]

        # Rewrite beta functions from paper in terms of gamma functions on log scale
        logS = (
            gammaln(beta + T + future_t)
            - gammaln(beta)
            + gammaln(alpha + beta)
            - gammaln(alpha + beta + T + future_t)
        )
        survival = np.exp(logS)
        # TODO: "cohort" dim instead of "customer_id"?
        return survival.transpose("chain", "draw", "customer_id", missing_dims="ignore")


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
