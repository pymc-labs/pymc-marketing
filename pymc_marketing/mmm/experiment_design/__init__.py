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

"""Posterior-aware experiment design for marketing lift tests.

Public API
----------
ExperimentDesigner
    Main class for recommending experiments based on a fitted MMM.
ExperimentRecommendation
    Dataclass representing a single recommended experiment design.
generate_experiment_fixture
    Utility for generating realistic InferenceData test fixtures.
"""

from pymc_marketing.mmm.experiment_design.designer import ExperimentDesigner
from pymc_marketing.mmm.experiment_design.fixture import (
    generate_experiment_fixture,
)
from pymc_marketing.mmm.experiment_design.recommendation import (
    ExperimentRecommendation,
)

__all__ = [
    "ExperimentDesigner",
    "ExperimentRecommendation",
    "generate_experiment_fixture",
]
