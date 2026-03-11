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
"""Schemas for MMM benchmark task specifications and outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SamplerPolicy(BaseModel):
    """Pinned sampler settings for reproducible benchmark runs."""

    model_config = ConfigDict(extra="forbid")

    nuts_sampler: str = "nutpie"
    chains: int = 14
    cores: int = 14
    draws: int = 500


class TimeSliceCVConfig(BaseModel):
    """Time-slice CV configuration for a single task."""

    model_config = ConfigDict(extra="forbid")

    n_init: int
    forecast_horizon: int
    step_size: int = 1
    n_folds: int = 5

    @field_validator("n_init", "forecast_horizon", "step_size")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("CV settings must be positive integers.")
        return value

    @field_validator("n_folds")
    @classmethod
    def _minimum_folds(cls, value: int) -> int:
        if value < 2:
            raise ValueError("Each task must have at least 2 time-slice CV folds.")
        return value


class GroundTruthSpec(BaseModel):
    """Optional ground truth data for recovery evaluation."""

    model_config = ConfigDict(extra="forbid")

    parameters: dict[str, Any] = Field(default_factory=dict)
    roas: dict[str, float] = Field(default_factory=dict)


class BenchmarkTaskSpec(BaseModel):
    """Task definition consumed by the benchmark runner."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_type: Literal["mmm_1d", "mmm_multidimensional", "mmm_roas_confounding"]
    dataset_path: str
    date_column: str
    target_column: str
    channel_columns: list[str]
    cv: TimeSliceCVConfig
    sampler: SamplerPolicy = Field(default_factory=SamplerPolicy)
    ground_truth: GroundTruthSpec | None = None
    notes: str | None = None


def load_task_spec(data: dict[str, Any]) -> BenchmarkTaskSpec:
    """Validate and load a task specification from a Python dictionary."""
    return BenchmarkTaskSpec.model_validate(data)


def load_task_spec_from_yaml(path: str | Path) -> BenchmarkTaskSpec:
    """Load and validate a task specification from a YAML file."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return load_task_spec(payload)


def load_task_specs_from_directory(directory: str | Path) -> list[BenchmarkTaskSpec]:
    """Load all benchmark task specifications in a directory."""
    root = Path(directory)
    task_files = sorted(root.glob("*.yaml"))
    return [load_task_spec_from_yaml(path) for path in task_files]
