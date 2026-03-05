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
"""InferenceData utilities and wrappers for PyMC-Marketing models."""

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
from pymc_marketing.data.idata.schema import (
    InferenceDataGroupSchema,
    MMMIdataSchema,
    VariableSchema,
)
from pymc_marketing.data.idata.utils import (
    aggregate_idata_dims,
    aggregate_idata_time,
    filter_idata_by_dates,
    filter_idata_by_dims,
    subsample_draws,
)

__all__ = [
    "InferenceDataGroupSchema",
    "MMMIDataWrapper",
    "MMMIdataSchema",
    "VariableSchema",
    "aggregate_idata_dims",
    "aggregate_idata_time",
    "filter_idata_by_dates",
    "filter_idata_by_dims",
    "subsample_draws",
]
