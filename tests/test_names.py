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
"""Pin the names the library owns.

Saved models store these strings and users type them, so a changed value is a
breaking change. The expected values are literals on purpose: a test that read
the constant back would pass after a rename.
"""

import pkgutil
from typing import get_args

import pytest

from pymc_marketing import names
from pymc_marketing.names import bass, clv, customer_choice, mmm

MODULES = [names, mmm, clv, customer_choice, bass]

# Public module-level names that are not classes.
MODULE_LEVEL = {
    names: set(),
    mmm: {
        "RESERVED_DIMS",
        "ORIGINAL_SCALE_SUFFIX",
        "EFFECT_CONTRIBUTION_SUFFIX",
        "original_scale_name",
        "effect_contribution_name",
    },
    clv: {"NewCustomerVar", "NEW_CUSTOMER_VARS"},
    customer_choice: set(),
    bass: set(),
}
IMPORTED = {"Final", "Literal", "names", "bass", "clv", "customer_choice", "mmm"}

SHARED_ATTR = {
    "ID": "id",
    "MODEL_TYPE": "model_type",
    "VERSION": "version",
    "SAMPLER_CONFIG": "sampler_config",
    "MODEL_CONFIG": "model_config",
    "SERIALIZATION_VERSION": "__serialization_version__",
}
SHARED_GROUP = {"FIT_DATA": "fit_data"}

EXPECTED = {
    (names, "Group"): SHARED_GROUP,
    (names, "Attr"): SHARED_ATTR,
    (names, "Key"): {
        "TYPE": "__type__",
        "DEFERRED": "__deferred__",
        "LITERAL": "__literal__",
    },
    (mmm, "Dim"): {
        "DATE": "date",
        "CHANNEL": "channel",
        "CONTROL": "control",
        "FOURIER_MODE": "fourier_mode",
        "TIME_SINCE_EXPOSURE": "time since exposure",
        "CV": "cv",
        "SWEEP": "sweep",
    },
    (mmm, "Var"): {
        "CHANNEL_DATA": "channel_data",
        "TARGET_DATA": "target_data",
        "CONTROL_DATA": "control_data",
        "CHANNEL_SCALE": "channel_scale",
        "TARGET_SCALE": "target_scale",
        "TIME_INDEX": "time_index",
        "DAYOFYEAR": "dayofyear",
        "MU": "mu",
        "MU_ORIGINAL_SCALE": "mu_original_scale",
        "Y": "y",
        "Y_ORIGINAL_SCALE": "y_original_scale",
        "CHANNEL_CONTRIBUTION": "channel_contribution",
        "CHANNEL_CONTRIBUTION_ORIGINAL_SCALE": "channel_contribution_original_scale",
        "CHANNEL_INCREMENTAL_CONTRIBUTION": "channel_incremental_contribution",
        "CONTROL_CONTRIBUTION": "control_contribution",
        "CONTROL_CONTRIBUTION_ORIGINAL_SCALE": "control_contribution_original_scale",
        "INTERCEPT_BASELINE": "intercept_baseline",
        "INTERCEPT_CONTRIBUTION": "intercept_contribution",
        "INTERCEPT_CONTRIBUTION_ORIGINAL_SCALE": "intercept_contribution_original_scale",
        "YEARLY_SEASONALITY_CONTRIBUTION": "yearly_seasonality_contribution",
        "YEARLY_SEASONALITY_CONTRIBUTION_ORIGINAL_SCALE": "yearly_seasonality_contribution_original_scale",
        "TOTAL_MEDIA_CONTRIBUTION_ORIGINAL_SCALE": "total_media_contribution_original_scale",
        "TOTAL_RESPONSE_ORIGINAL_SCALE": "total_response_original_scale",
        "CHANNEL_SPEND": "channel_spend",
        "ALLOCATION": "allocation",
        "TOTAL_ALLOCATION": "total_allocation",
        "METADATA": "metadata",
    },
    (mmm, "DatasetVar"): {
        "CHANNEL": "_channel",
        "TARGET": "_target",
        "CONTROL": "_control",
    },
    (mmm, "Attr"): {
        **SHARED_ATTR,
        "LINK": "link",
        "COST_PER_UNIT": "cost_per_unit",
        "MU_EFFECTS": "mu_effects",
        "ORIGINAL_SCALE_VARS": "original_scale_vars",
        "TIME_AGGREGATION": "time_aggregation",
        "LIFT_TEST_CALIBRATIONS": "lift_test_calibrations",
        "COST_PER_TARGET_CALIBRATIONS": "cost_per_target_calibrations",
        "ADSTOCK": "adstock",
        "SATURATION": "saturation",
        "TIME_VARYING_INTERCEPT": "time_varying_intercept",
        "TIME_VARYING_MEDIA": "time_varying_media",
    },
    (mmm, "Group"): {
        **SHARED_GROUP,
        "SENSITIVITY_ANALYSIS": "sensitivity_analysis",
        "CV_METADATA": "cv_metadata",
    },
    (clv, "Col"): {
        "CUSTOMER_ID": "customer_id",
        "FREQUENCY": "frequency",
        "RECENCY": "recency",
        "T": "T",
        "MONETARY_VALUE": "monetary_value",
        "FUTURE_T": "future_t",
        "FUTURE_SPEND": "future_spend",
        "NEW_CUSTOMER_T": "t",
        "COHORT": "cohort",
        "N_PURCHASES": "n_purchases",
        "DISCOUNT_RATE": "discount_rate",
        "INDIVIDUAL_TRANSACTION_VALUE": "individual_transaction_value",
        "FIRST": "first",
        "R_QUARTILE": "r_quartile",
        "F_QUARTILE": "f_quartile",
        "M_QUARTILE": "m_quartile",
        "RFM_SCORE": "rfm_score",
        "SEGMENT": "segment",
    },
    (clv, "Dim"): {
        "CUSTOMER_ID": "customer_id",
        "OBS_VAR": "obs_var",
        "COHORT": "cohort",
        "PURCHASE_COVARIATE": "purchase_covariate",
        "DROPOUT_COVARIATE": "dropout_covariate",
        "OBS": "obs",
        "NEW_CUSTOMER_ID": "new_customer_id",
    },
    (clv, "Var"): {
        "RECENCY_FREQUENCY": "recency_frequency",
        "DROPOUT": "dropout",
        "PURCHASE_RATE": "purchase_rate",
        "MEAN_SPEND": "mean_spend",
        "ALPHA_SCALE": "alpha_scale",
        "BETA_SCALE": "beta_scale",
        "A_SCALE": "a_scale",
        "B_SCALE": "b_scale",
        "PURCHASE_COEFFICIENT": "purchase_coefficient",
        "PURCHASE_COEFFICIENT_ALPHA": "purchase_coefficient_alpha",
        "DROPOUT_COEFFICIENT": "dropout_coefficient",
        "DROPOUT_COEFFICIENT_A": "dropout_coefficient_a",
        "DROPOUT_COEFFICIENT_B": "dropout_coefficient_b",
        "DROPOUT_COEFFICIENT_ALPHA": "dropout_coefficient_alpha",
        "DROPOUT_COEFFICIENT_BETA": "dropout_coefficient_beta",
    },
    (customer_choice, "Dim"): {
        "ALTS": "alts",
        "ALTS_PROBS": "alts_probs",
        "OBS": "obs",
        "ALT_COVARIATES": "alt_covariates",
        "FIXED_COVARIATES": "fixed_covariates",
        "COVARIATES": "covariates",
        "INDIVIDUALS": "individuals",
    },
    (customer_choice, "Var"): {
        "P": "p",
        "U": "U",
        "LIKELIHOOD": "likelihood",
        "X": "X",
        "Y": "y",
        "F": "F",
        "W": "W",
        "ALPHAS": "alphas",
        "BETAS_FIXED": "betas_fixed",
        "GRP_IDX": "grp_idx",
    },
    (customer_choice, "Attr"): {
        **SHARED_ATTR,
        "UTILITY_EQUATIONS": "utility_equations",
        "DEPVAR": "depvar",
        "COVARIATES": "covariates",
    },
    (bass, "Dim"): {"T": "T"},
    (bass, "Col"): {"OBSERVED": "observed", "T": "T"},
    (bass, "Var"): {
        "OBSERVED": "observed",
        "TIME": "t",
        "ADOPTERS": "adopters",
        "INNOVATORS": "innovators",
        "IMITATORS": "imitators",
        "PEAK": "peak",
        "Y": "y",
        "Y_OBS": "y_obs",
    },
}

CASES = [
    pytest.param(module, class_name, id=f"{module.__name__}.{class_name}")
    for module, class_name in EXPECTED
]


def _values(cls: type) -> dict[str, str]:
    return {name: getattr(cls, name) for name in dir(cls) if not name.startswith("_")}


@pytest.mark.parametrize("module, class_name", CASES)
def test_class_values_are_pinned(module, class_name) -> None:
    assert _values(getattr(module, class_name)) == EXPECTED[(module, class_name)]


@pytest.mark.parametrize("module, class_name", CASES)
def test_values_are_unique_within_a_class(module, class_name) -> None:
    values = list(_values(getattr(module, class_name)).values())
    assert len(values) == len(set(values))


def test_every_names_module_is_pinned() -> None:
    found = {module.name for module in pkgutil.iter_modules(names.__path__)}
    assert found == {"bass", "clv", "customer_choice", "mmm"}


@pytest.mark.parametrize("module", MODULES)
def test_module_level_names_are_pinned(module) -> None:
    public = {
        name
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and not isinstance(obj, type)
        and name not in IMPORTED
    }
    assert public == MODULE_LEVEL[module]


@pytest.mark.parametrize("module", MODULES)
def test_every_names_class_is_pinned(module) -> None:
    defined = {
        name
        for name, obj in vars(module).items()
        if isinstance(obj, type) and obj.__module__ == module.__name__
    }
    pinned = {
        class_name for pinned_module, class_name in EXPECTED if pinned_module is module
    }
    assert defined == pinned


def test_mmm_module_level_names() -> None:
    assert mmm.RESERVED_DIMS == {"date", "channel", "control", "fourier_mode"}
    assert mmm.ORIGINAL_SCALE_SUFFIX == "_original_scale"
    assert mmm.EFFECT_CONTRIBUTION_SUFFIX == "_effect_contribution"
    assert mmm.original_scale_name("mu") == "mu_original_scale"
    assert mmm.effect_contribution_name("trend") == "trend_effect_contribution"


def test_clv_new_customer_vars() -> None:
    expected = ("dropout", "purchase_rate", "recency_frequency")
    assert clv.NEW_CUSTOMER_VARS == expected
    assert get_args(clv.NewCustomerVar) == expected


def test_kept_aliases_keep_their_values() -> None:
    from pymc_marketing.mmm import budget_optimizer, incrementality, link, spend_reach
    from pymc_marketing.mmm.counterfactual import CounterfactualEvaluator

    assert spend_reach.CHANNEL_CONTRIBUTION == "channel_contribution"
    assert spend_reach.LINEAR_PREDICTOR == "mu"
    assert (
        budget_optimizer.DEFAULT_RESPONSE_VARIABLE
        == "total_media_contribution_original_scale"
    )
    assert budget_optimizer.MMM_CHANNEL_DATA_VAR == "channel_data"
    assert link.MU_ORIGINAL_SCALE == "mu_original_scale"
    assert incrementality.LAG_DIM == "time since exposure"
    assert CounterfactualEvaluator.CHANNEL_DATA == "channel_data"
    assert CounterfactualEvaluator.TIME_INDEX == "time_index"
