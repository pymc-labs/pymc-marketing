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
"""Contracts for MMM report generation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd

ReportPointEstimate = Literal["mean", "median"]
ReportRoasMethod = Literal["elementwise", "incremental"]


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for report generation."""

    hdi_probs: Sequence[float] = (0.94,)
    point_estimate: ReportPointEstimate = "mean"
    frequency: str = "all_time"
    roas_methods: Sequence[ReportRoasMethod] = ("elementwise", "incremental")
    dims: Mapping[str, str | int | Sequence[str | int]] | None = None
    sensitivity_sweep_values: Sequence[float] | None = None
    include_interactive: bool = True
    num_samples: int | None = None
    random_state: Any | None = None

    def __post_init__(self) -> None:
        hdi_probs = tuple(sorted({float(v) for v in self.hdi_probs}))
        if not hdi_probs:
            raise ValueError("hdi_probs must contain at least one probability")
        if any(v <= 0 or v >= 1 for v in hdi_probs):
            raise ValueError("All hdi_probs must be in the open interval (0, 1)")
        object.__setattr__(self, "hdi_probs", hdi_probs)

        roas_methods = tuple(dict.fromkeys(self.roas_methods))
        valid_methods = {"elementwise", "incremental"}
        if not roas_methods or not set(roas_methods).issubset(valid_methods):
            raise ValueError(
                "roas_methods must be a non-empty subset of {'elementwise', 'incremental'}"
            )
        object.__setattr__(self, "roas_methods", roas_methods)

        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError("num_samples must be positive when provided")

        if self.dims is not None:
            normalized_dims: dict[str, str | int | tuple[str | int, ...]] = {}
            for key, value in self.dims.items():
                if isinstance(value, (str, int)):
                    normalized_dims[key] = value
                elif isinstance(value, (list, tuple)):
                    normalized_dims[key] = tuple(value)
                else:
                    normalized_dims[key] = tuple(value)
            object.__setattr__(self, "dims", normalized_dims)

        if self.sensitivity_sweep_values is not None:
            values = tuple(float(v) for v in self.sensitivity_sweep_values)
            if len(values) < 2:
                raise ValueError(
                    "sensitivity_sweep_values must contain at least two values"
                )
            object.__setattr__(self, "sensitivity_sweep_values", values)


@dataclass(frozen=True)
class ReportMetadata:
    """Metadata displayed in report headers."""

    created_at: datetime
    package_version: str
    model_name: str
    date_column: str
    start_date: str | None
    end_date: str | None
    chains: int
    draws: int
    channels: tuple[str, ...]
    controls: tuple[str, ...]

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
        """Create metadata with current UTC timestamp."""
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


@dataclass(frozen=True)
class ReportSection:
    """A report section with tabular and visual outputs."""

    title: str
    description: str
    source_code: str
    dataframes: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    static_figures: Mapping[str, Any] = field(default_factory=dict)
    interactive_figures: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReportData:
    """Computed report payload consumed by renderers."""

    metadata: ReportMetadata
    sections: OrderedDict[str, ReportSection]
    parity_matrix: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
