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
"""Contracts (data models) for MMM report generation.

This module defines the Pydantic models that carry configuration, metadata,
section payloads, and the final assembled report data through the report
pipeline.  All models are frozen (immutable after creation) so they can be
safely shared across exporters without defensive copies.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

ReportPointEstimate = Literal["mean", "median"]
ReportRoasMethod = Literal["elementwise", "incremental"]


class ReportConfig(BaseModel):
    """Configuration for MMM report generation.

    Controls which summaries, uncertainty intervals, and visualisations are
    included in the report output.

    Parameters
    ----------
    hdi_probs : tuple of float
        HDI probability levels for uncertainty intervals.  Duplicates are
        removed and the values are sorted in ascending order.  Every value
        must lie in the open interval (0, 1).
    point_estimate : {"mean", "median"}
        Which point estimate to highlight in tables and figures.
    frequency : str
        Time-aggregation frequency forwarded to summary methods
        (e.g. ``"all_time"``, ``"weekly"``).
    roas_methods : tuple of {"elementwise", "incremental"}
        ROAS computation methods to include.
    dims : dict or None
        Optional dimension filters applied to every summary DataFrame.
        Keys are dimension names; values are single coordinates or sequences
        of coordinates to keep.
    sensitivity_sweep_values : tuple of float or None
        Multipliers for the sensitivity-analysis sweep.  Must contain at
        least two values when provided.
    include_interactive : bool
        Whether to generate Plotly interactive figures alongside static
        matplotlib ones.
    num_samples : int or None
        Number of posterior samples used by stochastic summaries (e.g.
        incremental ROAS).  Must be positive when provided.
    random_state : int, RandomState, or None
        Random state forwarded to stochastic summary methods for
        reproducibility.

    Raises
    ------
    ValueError
        If ``hdi_probs`` is empty, contains values outside (0, 1),
        ``roas_methods`` is invalid, ``num_samples`` is non-positive, or
        ``sensitivity_sweep_values`` has fewer than two entries.
    """

    model_config = ConfigDict(frozen=True)

    hdi_probs: tuple[float, ...] = Field(
        (0.94,),
        description="HDI probability levels for uncertainty intervals.",
    )
    point_estimate: ReportPointEstimate = Field(
        "mean",
        description="Point estimate to highlight: 'mean' or 'median'.",
    )
    frequency: str = Field(
        "all_time",
        description="Time-aggregation frequency forwarded to summary methods.",
    )
    roas_methods: tuple[ReportRoasMethod, ...] = Field(
        ("elementwise", "incremental"),
        description="ROAS computation methods to include.",
    )
    dims: dict[str, str | int | tuple[str | int, ...]] | None = Field(
        None,
        description="Optional dimension filters applied to summary DataFrames.",
    )
    sensitivity_sweep_values: tuple[float, ...] | None = Field(
        None,
        description="Multipliers for the sensitivity-analysis sweep.",
    )
    include_interactive: bool = Field(
        True,
        description="Whether to generate Plotly interactive figures.",
    )
    num_samples: int | None = Field(
        None,
        description="Number of posterior samples for stochastic summaries.",
    )
    random_state: Any | None = Field(
        None,
        description="Random state for reproducibility of stochastic summaries.",
    )

    @field_validator("hdi_probs", mode="before")
    @classmethod
    def _normalize_hdi_probs(cls, v: Any) -> tuple[float, ...]:
        values = tuple(sorted({float(x) for x in v}))
        if not values:
            raise ValueError("hdi_probs must contain at least one probability")
        if any(x <= 0 or x >= 1 for x in values):
            raise ValueError("All hdi_probs must be in the open interval (0, 1)")
        return values

    @field_validator("roas_methods", mode="before")
    @classmethod
    def _normalize_roas_methods(cls, v: Any) -> tuple[str, ...]:
        methods = tuple(dict.fromkeys(v))
        valid = {"elementwise", "incremental"}
        if not methods or not set(methods).issubset(valid):
            raise ValueError(
                "roas_methods must be a non-empty subset of "
                "{'elementwise', 'incremental'}"
            )
        return methods

    @field_validator("num_samples")
    @classmethod
    def _check_num_samples(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("num_samples must be positive when provided")
        return v

    @field_validator("dims", mode="before")
    @classmethod
    def _normalize_dims(
        cls, v: Mapping[str, Any] | None
    ) -> dict[str, str | int | tuple[str | int, ...]] | None:
        if v is None:
            return None
        normalized: dict[str, str | int | tuple[str | int, ...]] = {}
        for key, value in v.items():
            if isinstance(value, (str, int)):
                normalized[key] = value
            else:
                normalized[key] = tuple(value)
        return normalized

    @field_validator("sensitivity_sweep_values", mode="before")
    @classmethod
    def _normalize_sweep_values(cls, v: Any) -> tuple[float, ...] | None:
        if v is None:
            return None
        values = tuple(float(x) for x in v)
        if len(values) < 2:
            raise ValueError(
                "sensitivity_sweep_values must contain at least two values"
            )
        return values


class ReportMetadata(BaseModel):
    """Metadata displayed in report headers.

    Captures the provenance information (timestamps, package version, model
    configuration) that appears on the cover page of every exported report.

    Parameters
    ----------
    created_at : datetime
        UTC timestamp of report creation.
    package_version : str
        Version string of the ``pymc-marketing`` package.
    model_name : str
        Class name of the fitted MMM.
    date_column : str
        Name of the date column in the model's training data.
    start_date : str or None
        Earliest date in the training data, as an ISO-formatted string.
    end_date : str or None
        Latest date in the training data, as an ISO-formatted string.
    chains : int
        Number of MCMC chains in the posterior.
    draws : int
        Number of draws per chain in the posterior.
    channels : tuple of str
        Channel column names used in the model.
    controls : tuple of str
        Control column names used in the model.
    """

    model_config = ConfigDict(frozen=True)

    created_at: datetime = Field(..., description="UTC timestamp of report creation.")
    package_version: str = Field(
        ..., description="Version string of the pymc-marketing package."
    )
    model_name: str = Field(..., description="Class name of the fitted MMM.")
    date_column: str = Field(
        ..., description="Name of the date column in the training data."
    )
    start_date: str | None = Field(
        ..., description="Earliest date in training data (ISO string)."
    )
    end_date: str | None = Field(
        ..., description="Latest date in training data (ISO string)."
    )
    chains: int = Field(..., description="Number of MCMC chains in the posterior.")
    draws: int = Field(..., description="Number of draws per chain in the posterior.")
    channels: tuple[str, ...] = Field(
        ..., description="Channel column names used in the model."
    )
    controls: tuple[str, ...] = Field(
        ..., description="Control column names used in the model."
    )

    @classmethod
    def now(
        cls,
        *,
        package_version: str,
        model_name: str,
        date_column: str,
        start_date: str | None,
        end_date: str | None,
        chains: int,
        draws: int,
        channels: Sequence[str],
        controls: Sequence[str],
    ) -> ReportMetadata:
        """Create metadata with the current UTC timestamp.

        Parameters
        ----------
        package_version : str
            Version string of the ``pymc-marketing`` package.
        model_name : str
            Class name of the fitted MMM.
        date_column : str
            Name of the date column in the model's training data.
        start_date : str or None
            Earliest date in the training data.
        end_date : str or None
            Latest date in the training data.
        chains : int
            Number of MCMC chains.
        draws : int
            Number of draws per chain.
        channels : sequence of str
            Channel column names.
        controls : sequence of str
            Control column names.

        Returns
        -------
        ReportMetadata
            A new metadata instance stamped with ``datetime.now(UTC)``.
        """
        return cls(
            created_at=datetime.now(UTC),
            package_version=package_version,
            model_name=model_name,
            date_column=date_column,
            start_date=start_date,
            end_date=end_date,
            chains=chains,
            draws=draws,
            channels=tuple(channels),
            controls=tuple(controls),
        )


class ReportSection(BaseModel):
    """A single report section with tabular and visual outputs.

    Each section corresponds to one analytical block (e.g. "Diagnostics",
    "ROAS") and bundles together the DataFrames, static matplotlib figures,
    and optional interactive Plotly figures produced by that block.

    Parameters
    ----------
    title : str
        Human-readable section heading.
    description : str
        Short prose description shown beneath the heading.
    source_code : str
        Illustrative source snippet shown in collapsed notebook cells.
    dataframes : dict of str to DataFrame
        Keyed collection of **all** result tables (used by Excel export and
        ``to_dataframe()``).
    display_dataframes : dict of str to DataFrame
        Subset of compact tables rendered in the HTML/PDF notebook.  When
        empty the notebook shows no tables for this section (plots only).
    static_figures : dict of str to matplotlib Figure
        Keyed collection of static (raster) figures.
    interactive_figures : dict of str to plotly Figure
        Keyed collection of interactive (Plotly) figures.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    title: str = Field(..., description="Human-readable section heading.")
    description: str = Field(
        ..., description="Short prose description shown beneath the heading."
    )
    source_code: str = Field(
        ...,
        description="Illustrative source snippet for collapsed notebook cells.",
    )
    dataframes: dict[str, pd.DataFrame] = Field(
        default_factory=dict,
        description="Keyed collection of all result tables.",
    )
    display_dataframes: dict[str, pd.DataFrame] = Field(
        default_factory=dict,
        description=(
            "Subset of compact tables rendered in the HTML/PDF notebook. "
            "When empty the notebook shows no tables for this section."
        ),
    )
    static_figures: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyed collection of static matplotlib figures.",
    )
    interactive_figures: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyed collection of interactive Plotly figures.",
    )


class ReportData(BaseModel):
    """Computed report payload consumed by renderers.

    This is the top-level container that holds all metadata and section
    outputs.  Exporters (HTML, PDF, Excel) receive a single ``ReportData``
    instance and iterate over its sections to produce the final artefact.

    Parameters
    ----------
    metadata : ReportMetadata
        Provenance information for the cover page.
    sections : OrderedDict of str to ReportSection
        Ordered mapping of section keys to their computed payloads.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    metadata: ReportMetadata = Field(
        ..., description="Provenance information for the cover page."
    )
    sections: OrderedDict[str, ReportSection] = Field(
        ...,
        description="Ordered mapping of section keys to computed payloads.",
    )
