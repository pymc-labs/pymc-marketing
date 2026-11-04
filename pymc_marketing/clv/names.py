#   Copyright 2022 - 2026 The PyMC Labs Developers
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
"""Names of the CLV models: data columns, dimensions and model variables.

Each name is a member of a :class:`~pymc_marketing.names.Name` enum, so ``mypy``
reports a typo as ``attr-defined``; see :class:`~pymc_marketing.names.Name` for
how members behave when saved and loaded. The values are public API: users pass
DataFrames with these columns and saved models store them, so they must not
change.

Parameter names that are math symbols (``a``, ``b``, ``r``, ``alpha``, ...) are
not listed: they mean different things in each model, and each model's
``default_model_config`` is their single source of truth.
"""

from enum import unique
from typing import Final, Literal

from pymc_marketing.names import Name


@unique
class Col(Name):
    """DataFrame columns the CLV models and utilities read or create."""

    CUSTOMER_ID = "customer_id"
    FREQUENCY = "frequency"
    RECENCY = "recency"
    T = "T"
    MONETARY_VALUE = "monetary_value"
    FUTURE_T = "future_t"
    FUTURE_SPEND = "future_spend"
    NEW_CUSTOMER_T = "t"
    COHORT = "cohort"
    N_PURCHASES = "n_purchases"
    DISCOUNT_RATE = "discount_rate"
    INDIVIDUAL_TRANSACTION_VALUE = "individual_transaction_value"


@unique
class Dim(Name):
    """Dimension names of the CLV models."""

    CUSTOMER_ID = "customer_id"
    OBS_VAR = "obs_var"
    COHORT = "cohort"
    PURCHASE_COVARIATE = "purchase_covariate"
    DROPOUT_COVARIATE = "dropout_covariate"
    OBS = "obs"


@unique
class Var(Name):
    """Model variable names that the CLV models create and read back."""

    RECENCY_FREQUENCY = "recency_frequency"
    DROPOUT = "dropout"
    PURCHASE_RATE = "purchase_rate"
    ALPHA_SCALE = "alpha_scale"
    BETA_SCALE = "beta_scale"
    A_SCALE = "a_scale"
    B_SCALE = "b_scale"
    PURCHASE_COEFFICIENT = "purchase_coefficient"
    PURCHASE_COEFFICIENT_ALPHA = "purchase_coefficient_alpha"
    DROPOUT_COEFFICIENT = "dropout_coefficient"
    DROPOUT_COEFFICIENT_A = "dropout_coefficient_a"
    DROPOUT_COEFFICIENT_B = "dropout_coefficient_b"
    DROPOUT_COEFFICIENT_ALPHA = "dropout_coefficient_alpha"
    DROPOUT_COEFFICIENT_BETA = "dropout_coefficient_beta"


NewCustomerVar = Literal["dropout", "purchase_rate", "recency_frequency"]
"""Variables ``distribution_new_customer`` can sample.

A ``Literal`` of strings rather than of :class:`Var` members, so that users can
pass plain strings. ``mypy`` reports a value of :data:`NEW_CUSTOMER_VARS` that
this ``Literal`` lacks; a value added here but not there is not caught.
"""

NEW_CUSTOMER_VARS: Final[tuple[NewCustomerVar, ...]] = (
    Var.DROPOUT.value,
    Var.PURCHASE_RATE.value,
    Var.RECENCY_FREQUENCY.value,
)
"""Default ``var_names`` of ``distribution_new_customer``: all of :data:`NewCustomerVar`."""
