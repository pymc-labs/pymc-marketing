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
"""Agent execution interface for benchmark backends."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmark.schemas import BenchmarkTaskSpec


class AgentModeProfile(BaseModel):
    """Prompt behavior profile per benchmark mode."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(...)
    mode_instruction: str = Field(...)


class AgentExecutionConfig(BaseModel):
    """Configuration for command-line agent execution."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(default="claude")
    max_turns: int = Field(default=10, gt=0)
    timeout_sec: int = Field(default=900, gt=0)
    working_directory: str = Field(default=".")
    skip_permissions: bool = Field(default=True)


class AgentExecutionResult(BaseModel):
    """Raw execution result from a single agent run."""

    model_config = ConfigDict(extra="forbid")

    stdout: str = Field(...)
    stderr: str = Field(...)
    exit_code: int = Field(...)
    runtime_sec: float = Field(..., ge=0.0)


class FoldMetric(BaseModel):
    """Fold-level metric record returned by an agent."""

    model_config = ConfigDict(extra="forbid")

    fold_idx: int = Field(..., ge=0)
    crps: float = Field(...)


class CVParameterEstimateFold(BaseModel):
    """Fold-level parameter estimates for stability diagnostics."""

    model_config = ConfigDict(extra="forbid")

    fold_idx: int = Field(..., ge=0)
    parameter_estimates: dict[str, Any] = Field(default_factory=dict)


class CVFoldDiagnostic(BaseModel):
    """Optional fold-level fit diagnostics returned by the backend."""

    model_config = ConfigDict(extra="forbid")

    fold_idx: int = Field(..., ge=0)
    divergence_count: int | None = Field(default=None, ge=0)
    rhat_max: float | None = Field(default=None)
    ess_bulk_min: float | None = Field(default=None)
    ess_tail_min: float | None = Field(default=None)
    bfmi_min: float | None = Field(default=None)
    max_treedepth_hits: int | None = Field(default=None, ge=0)


class AgentBackendPayload(BaseModel):
    """Strict payload schema expected from agent outputs."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    status: Literal["success", "failure", "timeout"] = Field(...)
    metrics: dict[str, float] = Field(default_factory=dict)
    sample_stats_diverging: list[int] = Field(default_factory=list)
    fold_metrics: list[FoldMetric] = Field(default_factory=list)
    parameter_estimates: dict[str, Any] = Field(default_factory=dict)
    roas_estimates: dict[str, float] = Field(default_factory=dict)
    cv_parameter_estimates: list[CVParameterEstimateFold] = Field(default_factory=list)
    cv_fold_diagnostics: list[CVFoldDiagnostic] = Field(default_factory=list)
    fit_diagnostics: dict[str, float] = Field(default_factory=dict)
    model: Any | None = Field(default=None)


def build_task_prompt(
    task: BenchmarkTaskSpec, seed: int, profile: AgentModeProfile
) -> str:
    """Construct a deterministic task prompt for agent evaluation."""
    return f"""
You are running a benchmark MMM task.

Mode profile: {profile.name}
Mode instruction:
{profile.mode_instruction}

Task id: {task.task_id}
Task type: {task.task_type}
Dataset: {task.dataset_path}
Date column: {task.date_column}
Target column: {task.target_column}
Channels: {", ".join(task.channel_columns)}
Random seed: {seed}

Sampling policy (must be used):
- nuts_sampler={task.sampler.nuts_sampler}
- chains={task.sampler.chains}
- cores={task.sampler.cores}
- draws={task.sampler.draws}

Time-slice CV requirements:
- n_init={task.cv.n_init}
- forecast_horizon={task.cv.forecast_horizon}
- step_size={task.cv.step_size}
- n_folds={task.cv.n_folds}
- Use `pymc_marketing.mmm.time_slice_cross_validation.TimeSliceCrossValidator`
  to generate CV folds (do not create fold splits manually).

Return ONLY a JSON object with keys:
- status
- metrics
- sample_stats_diverging
- fold_metrics
- parameter_estimates
- roas_estimates
- cv_parameter_estimates (optional)
- cv_fold_diagnostics (optional)
- fit_diagnostics (optional)

If execution fails, still return JSON with status='failure' and empty metric payloads.
""".strip()


class ClaudeCliExecutionInterface(BaseModel):
    """Executes tasks via a Claude-compatible CLI binary."""

    model_config = ConfigDict(extra="forbid")

    config: AgentExecutionConfig = Field(...)

    def run_task(
        self,
        task: BenchmarkTaskSpec,
        seed: int,
        profile: AgentModeProfile,
    ) -> AgentExecutionResult:
        """Execute one benchmark task using a local Claude-compatible CLI binary."""
        prompt = build_task_prompt(task=task, seed=seed, profile=profile)
        command = [self.config.command, "-p", "--max-turns", str(self.config.max_turns)]
        if self.config.skip_permissions:
            command.append("--dangerously-skip-permissions")
        command.append(prompt)

        start = time.monotonic()
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=Path(self.config.working_directory),
            capture_output=True,
            text=True,
            timeout=self.config.timeout_sec,
            check=False,
        )
        elapsed = time.monotonic() - start
        return AgentExecutionResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            runtime_sec=elapsed,
        )


def parse_json_payload_from_stdout(stdout: str) -> dict[str, Any]:
    """Parse the first valid JSON object in stdout."""
    text = stdout.strip()
    if not text:
        raise ValueError("Agent stdout is empty.")

    # Fast path for plain JSON output.
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    # Fallback for markdown fenced JSON output.
    fenced_start = text.find("```json")
    fenced_end = text.find("```", fenced_start + 7) if fenced_start >= 0 else -1
    if fenced_start >= 0 and fenced_end > fenced_start:
        blob = text[fenced_start + 7 : fenced_end].strip()
        payload = json.loads(blob)
        if isinstance(payload, dict):
            return payload

    raise ValueError("Could not parse JSON payload from agent stdout.")
