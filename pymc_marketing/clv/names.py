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

Each name is a ``Final`` string attribute, so ``mypy`` reports a typo as
``attr-defined`` and the runtime value stays a plain ``str``. The values are
public API: users pass DataFrames with these columns and saved models store
them, so they must not change.

Parameter names that are math symbols (``a``, ``b``, ``r``, ``alpha``, ...) are
not listed: they mean different things in each model, and each model's
``default_model_config`` is their single source of truth.
"""

from typing import Final, Literal


class Col:
    """DataFrame columns the CLV models and utilities read or create."""

    CUSTOMER_ID: Final = "customer_id"
    FREQUENCY: Final = "frequency"
    RECENCY: Final = "recency"
    T: Final = "T"
    MONETARY_VALUE: Final = "monetary_value"
    FUTURE_T: Final = "future_t"
    FUTURE_SPEND: Final = "future_spend"
    NEW_CUSTOMER_T: Final = "t"
    COHORT: Final = "cohort"
    N_PURCHASES: Final = "n_purchases"
    DISCOUNT_RATE: Final = "discount_rate"
    INDIVIDUAL_TRANSACTION_VALUE: Final = "individual_transaction_value"


class Dim:
    """Dimension names of the CLV models."""

    CUSTOMER_ID: Final = "customer_id"
    OBS_VAR: Final = "obs_var"
    COHORT: Final = "cohort"
    PURCHASE_COVARIATE: Final = "purchase_covariate"
    DROPOUT_COVARIATE: Final = "dropout_covariate"
    OBS: Final = "obs"


class Var:
    """Model variable names that the CLV models create and read back."""

    RECENCY_FREQUENCY: Final = "recency_frequency"
    DROPOUT: Final = "dropout"
    PURCHASE_RATE: Final = "purchase_rate"
    ALPHA_SCALE: Final = "alpha_scale"
    BETA_SCALE: Final = "beta_scale"
    A_SCALE: Final = "a_scale"
    B_SCALE: Final = "b_scale"
    PURCHASE_COEFFICIENT: Final = "purchase_coefficient"
    PURCHASE_COEFFICIENT_ALPHA: Final = "purchase_coefficient_alpha"
    DROPOUT_COEFFICIENT: Final = "dropout_coefficient"
    DROPOUT_COEFFICIENT_A: Final = "dropout_coefficient_a"
    DROPOUT_COEFFICIENT_B: Final = "dropout_coefficient_b"
    DROPOUT_COEFFICIENT_ALPHA: Final = "dropout_coefficient_alpha"
    DROPOUT_COEFFICIENT_BETA: Final = "dropout_coefficient_beta"


NewCustomerVar = Literal["dropout", "purchase_rate", "recency_frequency"]
"""Variables ``distribution_new_customer`` can sample."""

NEW_CUSTOMER_VARS: Final[tuple[NewCustomerVar, ...]] = (
    Var.DROPOUT,
    Var.PURCHASE_RATE,
    Var.RECENCY_FREQUENCY,
)
"""Default ``var_names`` of ``distribution_new_customer``: all of :data:`NewCustomerVar`."""
