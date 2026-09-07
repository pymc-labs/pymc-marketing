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
from unittest.mock import MagicMock

from pymc_marketing.mmm.plotting.decomposition import DecompositionPlots
from pymc_marketing.mmm.plotting.diagnostics import DiagnosticsPlots
from pymc_marketing.mmm.plotting.sensitivity import SensitivityPlots
from pymc_marketing.mmm.plotting.suite import MMMPlotSuiteFacade
from pymc_marketing.mmm.plotting.transformations import TransformationPlots


def test_facade_creates_namespace_attributes():
    data = MagicMock()
    facade = MMMPlotSuiteFacade(data=data)
    assert isinstance(facade.decomposition, DecompositionPlots)
    assert isinstance(facade.diagnostics, DiagnosticsPlots)
    assert isinstance(facade.sensitivity, SensitivityPlots)
    assert isinstance(facade.transformation, TransformationPlots)


def test_facade_importable_from_plotting_package():
    from pymc_marketing.mmm.plotting import MMMPlotSuiteFacade as F

    assert F is MMMPlotSuiteFacade


def test_facade_importable_from_mmm_package():
    from pymc_marketing.mmm import MMMPlotSuiteFacade as F

    assert F is MMMPlotSuiteFacade
