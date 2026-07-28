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
"""MMMPlotSuiteFacade — namespace container for v2 plotting API."""

from __future__ import annotations

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
from pymc_marketing.mmm.plotting.decomposition import DecompositionPlots
from pymc_marketing.mmm.plotting.diagnostics import DiagnosticsPlots
from pymc_marketing.mmm.plotting.sensitivity import SensitivityPlots
from pymc_marketing.mmm.plotting.transformations import TransformationPlots

__all__ = ["MMMPlotSuiteFacade"]


class MMMPlotSuiteFacade:
    """Namespace container for the v2 MMMPlotSuite API.

    Access via ``mmm.plot`` when ``plot_suite='new'``.

    Parameters
    ----------
    data : MMMIDataWrapper
        Validated wrapper around the fitted model's InferenceData.

    Attributes
    ----------
    decomposition : DecompositionPlots
    diagnostics : DiagnosticsPlots
    sensitivity : SensitivityPlots
    transformation : TransformationPlots
    """

    def __init__(self, data: MMMIDataWrapper) -> None:
        self.decomposition = DecompositionPlots(data)
        self.diagnostics = DiagnosticsPlots(data)
        self.sensitivity = SensitivityPlots(data)
        self.transformation = TransformationPlots(data)
