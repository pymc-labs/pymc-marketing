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
"""Marketing Mix Models (MMM)."""

from pymc_marketing.mmm import base, mmm, preprocessing, validating
from pymc_marketing.mmm.base import BaseValidateMMM, MMMModelBuilder
from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
    DelayedAdstock,
    GeometricAdstock,
    NoAdstock,
    WeibullCDFAdstock,
    WeibullPDFAdstock,
    adstock_from_dict,
)
from pymc_marketing.mmm.components.saturation import (
    HillSaturation,
    HillSaturationSigmoid,
    InverseScaledLogisticSaturation,
    LogisticSaturation,
    MichaelisMentenSaturation,
    NoSaturation,
    RootSaturation,
    SaturationTransformation,
    TanhSaturation,
    TanhSaturationBaselined,
    saturation_from_dict,
)
from pymc_marketing.mmm.fourier import MonthlyFourier, WeeklyFourier, YearlyFourier
from pymc_marketing.mmm.hsgp import (
    HSGP,
    CovFunc,
    HSGPPeriodic,
    PeriodicCovFunc,
    SoftPlusHSGP,
    approx_hsgp_hyperparams,
    create_complexity_penalizing_prior,
    create_constrained_inverse_gamma_prior,
    create_eta_prior,
    create_m_and_L_recommendations,
)
from pymc_marketing.mmm.linear_regression import FancyLinearRegression
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.mmm.media_transformation import (
    MediaConfig,
    MediaConfigList,
    MediaTransformation,
)
from pymc_marketing.mmm.mmm import MMM
from pymc_marketing.mmm.preprocessing import (
    preprocessing_method_X,
    preprocessing_method_y,
)
from pymc_marketing.mmm.validating import validation_method_X, validation_method_y

__all__ = [
    "HSGP",
    "MMM",
    "AdstockTransformation",
    "BaseValidateMMM",
    "CovFunc",
    "DelayedAdstock",
    "FancyLinearRegression",
    "GeometricAdstock",
    "HSGPPeriodic",
    "HillSaturation",
    "HillSaturationSigmoid",
    "InverseScaledLogisticSaturation",
    "LinearTrend",
    "LogisticSaturation",
    "MMMModelBuilder",
    "MediaConfig",
    "MediaConfigList",
    "MediaTransformation",
    "MichaelisMentenSaturation",
    "MonthlyFourier",
    "NoAdstock",
    "NoSaturation",
    "PeriodicCovFunc",
    "RootSaturation",
    "SaturationTransformation",
    "SoftPlusHSGP",
    "TanhSaturation",
    "TanhSaturationBaselined",
    "WeeklyFourier",
    "WeibullCDFAdstock",
    "WeibullPDFAdstock",
    "YearlyFourier",
    "adstock_from_dict",
    "approx_hsgp_hyperparams",
    "base",
    "create_complexity_penalizing_prior",
    "create_constrained_inverse_gamma_prior",
    "create_eta_prior",
    "create_m_and_L_recommendations",
    "mmm",
    "preprocessing",
    "preprocessing_method_X",
    "preprocessing_method_y",
    "saturation_from_dict",
    "validating",
    "validation_method_X",
    "validation_method_y",
]
