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

__all__ = ["ShiftedBetaGeoModel", "ShiftedBetaGeoModelIndividual"]


class ShiftedBetaGeoModel(CLVModel):
    """Shifted Beta Geometric (sBG) model for customers renewing contracts over discrete time periods.

    The sBG model has the following assumptions:
      * Dropout probabilities for each cohortare Beta-distributed with hyperparameters `alpha` and `beta`.
      * Cohort retention rates change over time due to customer heterogeneity.
      * Customers in the same cohort began their contract in the same time period.

    This model requires data to be summarized by *recency*, *T*, and *cohort* for each customer.
    Modeling assumptions require *1 <= recency <= T*, and *T >= 2*.

    First introduced by Fader & Hardie in [1]_, with additional expressions and enhancements described in [2]_ and [3].

    Parameters
    ----------
    data : ~pandas.DataFrame
        DataFrame containing the following columns:
            * `customer_id`: Unique customer identifier
            * `recency`: Time period of last contract renewal. It should equal *T* for active customers.
            * `T`: Max observed time period in the cohort. All customers in a given cohort share the same value for *T*.
            * `cohort`: Customer cohort label
            * Additional columns for static covariates
    model_config : dict, optional
        Dictionary of model prior parameters:
            * `alpha`: Shape parameter of dropout process (cohort-level);
              defaults to `phi` * `kappa`
            * `beta`: Shape parameter of dropout process (cohort-level);
              defaults to `(1-phi)` * `kappa`
            * `phi`: Pooling prior if `alpha` and `beta` are not provided;
              defaults to `Prior("Uniform", lower=0, upper=1, dims="cohort")`
            * `kappa`: Pooling prior if `alpha` and `beta` are not provided;
              defaults to `Prior("Pareto", alpha=1, m=1, dims="cohort")`
            * `dropout_coefficient`: Prior for covariate coefficients; defaults to `Prior("Normal", mu=0, sigma=1)`
            * `dropout_covariate_cols`: List of column names for customer-level covariates.
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

            # Fit model quickly to large datasets via Maximum a Posteriori
            model.fit(method="map")
            model.fit_summary()

            # Use 'mcmc' for more informative predictions and reliable performance on smaller datasets
            model.fit(method="mcmc")
            model.fit_summary()


            # Predict probability customers are still active
            expected_alive_probability = model.expected_probability_alive(
                active_customers,
                future_t=0,
            )

            # Predict retention rate for a specific cohort
            cohort_name = "2025-02-01"

            expected_alive_probability = model.expected_retention_rate(
                future_t=0,
            ).sel(cohort=cohort_name)

            # Predict expected remaining lifetime for all customers with a 5% discount rate
            expected_alive_probability = model.expected_residual_lifetime(
                discount_rate=0.05,
            )

            # Predict expected retention elasticity for all customers in a specific cohort
            expected_alive_probability = model.expected_retention_elasticity(
                discount_rate=0.05,
            ).sel(cohort=cohort_name)

            # Example with customer-level covariates
            model_with_covariates = ShiftedBetaGeoModel(
                data=pd.DataFrame(
                    {
                        "customer_id": [1, 2, 3, ...],
                        "recency": [8, 1, 4, ...],
                        "T": [8, 5, 5, ...],
                        "cohort": ["2025-02", "2025-04", "2025-04", ...],
                        "channel_covariate": [1, 0, 1, ...],
                        "rating_covariate": [
                            2.172,
                            1.234,
                            2.345,
                            ...,
                        ],  # time-invariant
                    }
                ),
                model_config={
                    "dropout_coefficient": Prior("Normal", mu=0, sigma=2),
                    "dropout_covariate_cols": ["channel_covariate", "rating_covariate"],
                },
            )
            model_with_covariates.fit()

            # Predictions with covariates require covariate columns in prediction data
            pred_data = pd.DataFrame(
                {
                    "customer_id": [...],
                    "T": [...],
                    "cohort": [...],
                    "channel_covariate": [...],
                    "rating_covariate": [...],
                }
            )
            retention_with_covariates = model_with_covariates.expected_retention_rate(
                data=pred_data, future_t=1
            )


    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2007). "How to project customer retention."
        Journal of Interactive Marketing, 21(1), 76-90.
        https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
    .. [2] Fader, P. S., & Hardie, B. G. (2010). "Customer-Base Valuation in a Contractual Setting:
        The Perils of Ignoring Heterogeneity." Marketing Science, 29(1), 85-93.
    https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_contractual_mksc_10.pdf
    .. [3] Fader, Peter & G. S. Hardie, Bruce (2007).
        "Incorporating Time-Invariant Covariates into the Pareto/NBD and BG/NBD Models".
        https://www.brucehardie.com/notes/019/time_invariant_covariates.pdf
    """

    _model_type = "Shifted Beta-Geometric"

    def __init__(
        self,
        data: pd.DataFrame,
        model_config: ModelConfig | None = None,
        sampler_config: dict | None = None,
    ):
        super().__init__(
            data=data,
            model_config=model_config,
            sampler_config=sampler_config,
            non_distributions=["dropout_covariate_cols"],
        )

        # Extract covariate columns from model_config
        self.dropout_covariate_cols = list(self.model_config["dropout_covariate_cols"])

        self._validate_cols(
            data,
            required_cols=[
                "customer_id",
                "recency",
                "T",
                "cohort",
                *self.dropout_covariate_cols,
            ],
            must_be_unique=["customer_id"],
        )

        if np.any(
            (data["recency"] < 1) | (data["recency"] > data["T"]) | (data["T"] < 2)
        ):
            raise ValueError("Model fitting requires 1 <= recency <= T, and T >= 2.")

        self._validate_cohorts(self.data, check_param_dims=("alpha", "beta"))

        # Create cohort dim & coords
        self.cohorts = self.data["cohort"].unique()
        self.cohort_idx = pd.Categorical(
            self.data["cohort"], categories=self.cohorts
        ).codes

    def _validate_cohorts(
        self,
        data,
        check_pred_data=False,
        check_param_dims=None,
    ):
        """Validate cohort parameter dims, T homogeneity, and if provided in external data."""
        if check_pred_data:
            # Validate cohorts in external prediction data match any or all cohorts used to fix model.
            cohorts_present = pd.Index(data["cohort"].unique())
            cohorts_present = cohorts_present.intersection(pd.Index(self.cohorts))
            if len(cohorts_present) == 0:
                raise ValueError(
                    "Cohorts in prediction data do not match cohorts used to fit the model."
                )
            return cohorts_present
        else:
            # Validate T is homogeneous within each cohort
            t_per_cohort = data.groupby("cohort")["T"].nunique()
            non_homogeneous_cohorts = t_per_cohort[t_per_cohort > 1]
            if len(non_homogeneous_cohorts) > 0:
                cohort_names = ", ".join(
                    map(str, non_homogeneous_cohorts.index.tolist())
                )
                raise ValueError(
                    f"T must be homogeneous within each cohort. "
                    f"The following cohorts have multiple T values: {cohort_names}"
                )
            if check_param_dims is not None:
                # Validate provided Priors specify dims="cohort"
                for key in check_param_dims:
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

    @property
    def default_model_config(self) -> ModelConfig:
        """Default model configuration."""
        return {
            # Cohort-level hierarchical defaults (no covariates)
            "phi": Prior("Uniform", lower=0, upper=1, dims="cohort"),
            "kappa": Prior("Pareto", alpha=1, m=1, dims="cohort"),
            "dropout_coefficient": Prior("Normal", mu=0, sigma=1),
            "dropout_covariate_cols": [],
        }

    def build_model(self) -> None:  # type: ignore[override]
        """Build the model."""
        coords = {
            "customer_id": self.data["customer_id"],
            "cohort": self.cohorts,
            "dropout_covariate": self.dropout_covariate_cols,
        }
        with pm.Model(coords=coords) as self.model:
            if self.dropout_covariate_cols:
                # Customer-level behavior with covariates
                dropout_data = pm.Data(
                    "dropout_data",
                    self.data[self.dropout_covariate_cols],
                    dims=["customer_id", "dropout_covariate"],
                )

                # Get scale parameters (cohort-level baseline)
                if "alpha" in self.model_config and "beta" in self.model_config:
                    alpha_scale = self.model_config["alpha"].create_variable(
                        "alpha_scale"
                    )
                    beta_scale = self.model_config["beta"].create_variable("beta_scale")
                else:
                    # hierarchical pooling of dropout rate priors
                    phi = self.model_config["phi"].create_variable("phi")
                    kappa = self.model_config["kappa"].create_variable("kappa")

                    alpha_scale = pm.Deterministic(
                        "alpha_scale", phi * kappa, dims="cohort"
                    )
                    beta_scale = pm.Deterministic(
                        "beta_scale", (1.0 - phi) * kappa, dims="cohort"
                    )

                # Get covariate coefficients
                self.model_config["dropout_coefficient"].dims = "dropout_covariate"
                dropout_coefficient_alpha = self.model_config[
                    "dropout_coefficient"
                ].create_variable("dropout_coefficient_alpha")
                dropout_coefficient_beta = self.model_config[
                    "dropout_coefficient"
                ].create_variable("dropout_coefficient_beta")

                # Apply covariate effects to get customer-level parameters
                # expressions adapted from BG/NBD covariate extensions on p2 of [3]_:
                # https://www.brucehardie.com/notes/019/time_invariant_covariates.pdf
                alpha = pm.Deterministic(
                    "alpha",
                    alpha_scale[self.cohort_idx]
                    * pm.math.exp(
                        -pm.math.dot(dropout_data, dropout_coefficient_alpha)
                    ),
                    dims="customer_id",
                )
                beta = pm.Deterministic(
                    "beta",
                    beta_scale[self.cohort_idx]
                    * pm.math.exp(-pm.math.dot(dropout_data, dropout_coefficient_beta)),
                    dims="customer_id",
                )

                dropout = ShiftedBetaGeometric.dist(alpha, beta)
            else:
                # Cohort-level behavior only, no covariates
                if "alpha" in self.model_config and "beta" in self.model_config:
                    alpha = self.model_config["alpha"].create_variable("alpha")
                    beta = self.model_config["beta"].create_variable("beta")
                else:
                    # hierarchical pooling of dropout rate priors
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

    def _extract_predictive_variables(
        self,
        pred_data: pd.DataFrame,
        customer_varnames: Sequence[str] = (),
    ) -> xarray.Dataset:
        """
        Extract predictive variables from the data.

        Utility function assigning default customer arguments
        for predictive methods and converting to xarrays.
        """
        self._validate_cols(
            pred_data,
            required_cols=[
                "customer_id",
                *customer_varnames,
                *self.dropout_covariate_cols,
            ],
            must_be_unique=["customer_id"],
        )

        # Validate T requirements for predictions (T>=2 only required for fit data)
        if np.any(pred_data["T"] <= 0):
            raise ValueError(
                "T must be a non-zero, positive whole number.",
            )

        # Validate cohorts in prediction data match any or all cohorts used to fit model
        cohorts_present = self._validate_cohorts(pred_data, check_pred_data=True)

        # Use cohorts in prediction data to extract only cohort-level parameters
        pred_cohorts = xarray.DataArray(
            cohorts_present.values,
            dims=("cohort",),
            coords={"cohort": cohorts_present.values},
        )

        # Create a cohort-by-customer array to map cohort parameters to each customer
        customer_cohort_map = pred_data.set_index("customer_id")["cohort"]

        if self.dropout_covariate_cols:
            # Get alpha and beta scale parameters for each cohort
            alpha_cohort = self.fit_result["alpha_scale"].sel(cohort=pred_cohorts)
            beta_cohort = self.fit_result["beta_scale"].sel(cohort=pred_cohorts)
            # Get dropout covariate coefficients
            dropout_coefficient_alpha = self.fit_result["dropout_coefficient_alpha"]
            dropout_coefficient_beta = self.fit_result["dropout_coefficient_beta"]

            # Reconstruct customer-level alpha and beta with covariates
            # Create covariate xarray
            dropout_xarray = xarray.DataArray(
                pred_data[self.dropout_covariate_cols].values,
                dims=["customer_id", "dropout_covariate"],
                coords={
                    "customer_id": pred_data["customer_id"],
                    "dropout_covariate": self.dropout_covariate_cols,
                },
            )

            # Map cohort indices for each customer
            pred_cohort_idx = pd.Categorical(
                customer_cohort_map.values, categories=self.cohorts
            ).codes

            # Reconstruct customer-level parameters
            alpha_pred = alpha_cohort.isel(
                cohort=xarray.DataArray(pred_cohort_idx, dims="customer_id")
            ) * np.exp(
                -xarray.dot(
                    dropout_xarray, dropout_coefficient_alpha, dim="dropout_covariate"
                )
            )
            alpha_pred.name = "alpha"

            beta_pred = beta_cohort.isel(
                cohort=xarray.DataArray(pred_cohort_idx, dims="customer_id")
            ) * np.exp(
                -xarray.dot(
                    dropout_xarray, dropout_coefficient_beta, dim="dropout_covariate"
                )
            )
            beta_pred.name = "beta"

        else:
            # Get alpha and beta parameters for each cohort
            alpha_cohort = self.fit_result["alpha"].sel(cohort=pred_cohorts)
            beta_cohort = self.fit_result["beta"].sel(cohort=pred_cohorts)

            # Map cohorts to customer_id for alpha and beta
            customer_cohort_mapping = xarray.DataArray(
                customer_cohort_map.values,
                dims=("customer_id",),
                coords={"customer_id": customer_cohort_map.index},
                name="customer_cohort_mapping",
            )
            alpha_pred = alpha_cohort.sel(cohort=customer_cohort_mapping)
            beta_pred = beta_cohort.sel(cohort=customer_cohort_mapping)

        # Add cohorts as non-dimensional coordinates to merge with predictive variables
        alpha_pred = alpha_pred.assign_coords(
            cohort=("customer_id", customer_cohort_map.values)
        )
        beta_pred = beta_pred.assign_coords(
            cohort=("customer_id", customer_cohort_map.values)
        )

        # Filter out cohort from customer_varnames to avoid merge conflict
        # (it's already added as a coordinate above)
        customer_varnames_filtered = [v for v in customer_varnames if v != "cohort"]

        if customer_varnames_filtered:
            customer_vars = to_xarray(
                pred_data["customer_id"],
                *[
                    pred_data[customer_varname]
                    for customer_varname in customer_varnames_filtered
                ],
            )
            if len(customer_varnames_filtered) == 1:
                customer_vars = [customer_vars]
        else:
            customer_vars = []

        return xarray.combine_by_coords(
            (
                alpha_pred,
                beta_pred,
                *customer_vars,
            )
        ).swap_dims(
            {"customer_id": "cohort"}
        )  # swap dims to enable cohort selection for predictions

    def expected_retention_rate(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        """Compute expected retention rate for each customer.

        This is the percentage of customers who were active in the previous time period
        and are still active in the current period. Retention rates are expected to increase over time.

        The *data* parameter is only required for out-of-sample customers.

        Adapted from equation (8) in [1]_.

        Parameters
        ----------
        future_t : int, array_like
            Number of time periods in the future to predict retention rate.
        data : ~pandas.DataFrame
            Optional dataframe containing the following columns:
            * `customer_id`: Unique customer identifier
            * `T`: Number of time periods customer has been active
            * `cohort`: Customer cohort label
            * Covariate columns specified in `dropout_covariate_cols` (if using covariates)

        References
        ----------
        .. [1] Fader, P. S., & Hardie, B. G. (2007). "How to project customer retention."
            Journal of Interactive Marketing, 21(1), 76-90.
            https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
        """
        if data is None:
            data = self.data.query("recency == T").copy()

        if future_t is not None:
            data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["T", "future_t", "cohort"]
        )

        alpha = dataset["alpha"]
        beta = dataset["beta"]
        T = dataset["T"]
        t = dataset["future_t"]

        retention_rate = (beta + T + t - 1) / (alpha + beta + T + t - 1)
        return retention_rate.transpose(
            "chain", "draw", "customer_id", "cohort", missing_dims="ignore"
        )

    def expected_probability_alive(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        """Compute expected probability of contract renewal for each customer.

        The *data* parameter is only required for out-of-sample customers.

        Adapted from equation (6) in [1]_.

        Parameters
        ----------
        future_t : int, array_like
            Number of time periods in the future to predict probability of being active.
        data : ~pandas.DataFrame
            Optional dataframe containing the following columns:
            * `customer_id`: Unique customer identifier
            * `T`: Number of time periods customer has been active
            * `cohort`: Customer cohort label
            * Covariate columns specified in `dropout_covariate_cols` (if using covariates)

        References
        ----------
        .. [1] Fader, P. S., & Hardie, B. G. (2007). "How to project customer retention."
            Journal of Interactive Marketing, 21(1), 76-90.
            https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
        """
        if data is None:
            data = self.data.query("recency == T").copy()

        if future_t is not None:
            data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["T", "future_t", "cohort"]
        )

        alpha = dataset["alpha"]
        beta = dataset["beta"]
        T = dataset["T"]
        t = dataset["future_t"]

        logS = (
            gammaln(beta + T + t)
            - gammaln(beta)
            + gammaln(alpha + beta)
            - gammaln(alpha + beta + T + t)
        )
        survival = np.exp(logS)

        return survival.transpose(
            "chain", "draw", "customer_id", "cohort", missing_dims="ignore"
        )

    def expected_residual_lifetime(
        self,
        data: pd.DataFrame | None = None,
        *,
        discount_rate: float | np.ndarray | pd.Series | None = 0.0,
    ) -> xarray.DataArray:
        """Compute expected residual lifetime of each customer.

        This is the expected number of periods a customer will remain active after the current time period,
        subject to a discount rate for net present value (NPV) calculations.
        It is recommended to set a discount rate > 0 to avoid infinite lifetime estimates.

        Adapted from equation (6) in [1]_.

        Parameters
        ----------
        discount_rate : float
            Discount rate to apply for net present value estimations.
        data : ~pandas.DataFrame
            Optional dataframe containing the following columns:
            * `customer_id`: Unique customer identifier
            * `T`: Number of time periods customer has been active
            * `cohort`: Customer cohort label
            * Covariate columns specified in `dropout_covariate_cols` (if using covariates)

        References
        ----------
        .. [1] Fader, P. S., & Hardie, B. G. (2010). "Customer-Base Valuation in a Contractual Setting:
            The Perils of Ignoring Heterogeneity". Marketing Science, 29(1), 85-93.
            https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_contractual_mksc_10.pdf
        """
        if data is None:
            data = self.data

        if discount_rate is not None:
            data = data.assign(discount_rate=discount_rate)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["T", "discount_rate"]
        )
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        T = dataset["T"]
        d = dataset["discount_rate"]

        retention_rate = (beta + T - 1) / (alpha + beta + T - 1)
        retention_elasticity = hyp2f1(1, beta + T, alpha + beta + T, 1 / (1 + d))
        expected_lifetime_purchases = retention_rate * retention_elasticity

        return expected_lifetime_purchases.transpose(
            "chain", "draw", "customer_id", "cohort", missing_dims="ignore"
        )

    def expected_retention_elasticity(
        self,
        data: pd.DataFrame | None = None,
        *,
        discount_rate: float | np.ndarray | pd.Series | None = 0.0,
    ) -> xarray.DataArray:
        """Compute expected retention elasticity for each customer.

        This is the percent increase in expected residual lifetime given a 1% increase in the retention rate,
        subject to a discount rate for net present value (NPV) calculations.
        It is recommended to set a discount rate > 0 to avoid infinite retention elasticity estimates.

        Adapted from equation (8) in [1]_.

        Parameters
        ----------
        discount_rate : float
            Discount rate to apply for net present value estimations.
        data : ~pandas.DataFrame
            Optional dataframe containing the following columns:
            * `customer_id`: Unique customer identifier
            * `T`: Number of time periods customer has been active
            * `cohort`: Customer cohort label
            * Covariate columns specified in `dropout_covariate_cols` (if using covariates)

        References
        ----------
        .. [1] Fader, P. S., & Hardie, B. G. (2010). "Customer-Base Valuation in a Contractual Setting:
            The Perils of Ignoring Heterogeneity". Marketing Science, 29(1), 85-93.
            https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_contractual_mksc_10.pdf
        """
        if data is None:
            data = self.data

        if discount_rate is not None:
            data = data.assign(discount_rate=discount_rate)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["T", "discount_rate"]
        )

        alpha = dataset["alpha"]
        beta = dataset["beta"]
        T = dataset["T"]
        d = dataset["discount_rate"]

        retention_elasticity = hyp2f1(
            1, beta + T - 1, alpha + beta + T - 1, 1 / (1 + d)
        )
        return retention_elasticity.transpose(
            "chain", "draw", "customer_id", "cohort", missing_dims="ignore"
        )


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
