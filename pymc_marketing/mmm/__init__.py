#   Copyright 2024 The PyMC Labs Developers
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
    WeibullCDFAdstock,
    WeibullPDFAdstock,
    adstock_from_dict,
    register_adstock_transformation,
)
from pymc_marketing.mmm.components.saturation import (
    HillSaturation,
    HillSaturationSigmoid,
    InverseScaledLogisticSaturation,
    LogisticSaturation,
    MichaelisMentenSaturation,
    RootSaturation,
    SaturationTransformation,
    TanhSaturation,
    TanhSaturationBaselined,
    register_saturation_transformation,
    saturation_from_dict,
)
from pymc_marketing.mmm.fourier import MonthlyFourier, YearlyFourier
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
    "MediaTransformation",
    "MediaConfigList",
    "MediaConfig",
    "AdstockTransformation",
    "BaseValidateMMM",
    "DelayedAdstock",
    "GeometricAdstock",
    "HillSaturation",
    "HillSaturationSigmoid",
    "LogisticSaturation",
    "InverseScaledLogisticSaturation",
    "MMM",
    "MMMModelBuilder",
    "MichaelisMentenSaturation",
    "MonthlyFourier",
    "RootSaturation",
    "SaturationTransformation",
    "TanhSaturation",
    "TanhSaturationBaselined",
    "saturation_from_dict",
    "register_saturation_transformation",
    "WeibullCDFAdstock",
    "WeibullPDFAdstock",
    "adstock_from_dict",
    "register_adstock_transformation",
    "YearlyFourier",
    "base",
    "mmm",
    "preprocessing",
    "preprocessing_method_X",
    "preprocessing_method_y",
    "validating",
    "validation_method_X",
    "validation_method_y",
    "LinearTrend",
]
