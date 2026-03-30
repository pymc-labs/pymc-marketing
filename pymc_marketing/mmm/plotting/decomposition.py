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
"""Decomposition namespace — contribution waterfall and time-series plots."""

from __future__ import annotations

from typing import Any, Literal

import arviz as az
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.data.idata import MMMIDataWrapper


class DecompositionPlots:
    """Decomposition plots for fitted MMM models.

    Provides three methods to visualize how the model decomposes the target:

    - ``contributions_over_time`` — Time-series of each contribution type with HDI.
    - ``waterfall``              — Horizontal waterfall chart of mean contributions.
    - ``channel_share_hdi``     — Forest plot of each channel's share of total response.

    Parameters
    ----------
    data : MMMIDataWrapper
        Validated wrapper around the fitted model's InferenceData.
    """

    def __init__(self, data: MMMIDataWrapper) -> None:
        self._data = data

    def contributions_over_time(
        self,
        include: list[Literal["channels", "baseline", "controls", "seasonality"]]
        | None = None,
        hdi_prob: float = 0.94,
        original_scale: bool = True,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        line_kwargs: dict[str, Any] | None = None,
        hdi_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Plot time-series contributions for selected contribution types with HDI bands."""
        raise NotImplementedError

    def waterfall(
        self,
        original_scale: bool = True,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        bar_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure, NDArray[Axes]]:
        """Horizontal waterfall chart showing mean contribution per component."""
        raise NotImplementedError

    def channel_share_hdi(
        self,
        hdi_prob: float = 0.94,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Forest plot of each channel's share of total channel contribution."""
        raise NotImplementedError
