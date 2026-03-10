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
"""Execution backends for baseline and skill-enabled benchmark modes."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from benchmark.agent_interface import (
    AgentBackendPayload,
    AgentModeProfile,
    parse_json_payload_from_stdout,
)
from benchmark.runner import BenchmarkMode, BenchmarkResult
from benchmark.schemas import BenchmarkTaskSpec

AgentRunnerCallable = Callable[[BenchmarkTaskSpec, int], dict[str, Any]]
logger = logging.getLogger("benchmark.backends")


class ModeExecutors(BaseModel):
    """Container for baseline and skill-enabled task executors."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    baseline: AgentRunnerCallable = Field(...)
    skilled: AgentRunnerCallable = Field(...)


class BaselineVsSkilledBackend(BaseModel):
    """Backend that executes task functions for baseline and skilled modes."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    executors: ModeExecutors = Field(...)
    run_timeout_sec: float | None = Field(default=None, ge=0.0)

    def run(
        self, task: BenchmarkTaskSpec, mode: BenchmarkMode, seed: int
    ) -> BenchmarkResult:
        """Execute one benchmark task using function executors by mode."""
        start = time.monotonic()
        if mode == "baseline":
            payload = self.executors.baseline(task, seed)
        else:
            payload = self.executors.skilled(task, seed)
        elapsed = time.monotonic() - start
        status = (
            "timeout" if self._timed_out(elapsed) else payload.get("status", "success")
        )
        return BenchmarkResult(
            task_id=task.task_id,
            mode=mode,
            seed=seed,
            status=status,
            runtime_sec=elapsed,
            metrics=payload.get("metrics", {}),
            sample_stats_diverging=payload.get("sample_stats_diverging", []),
            fold_metrics=payload.get("fold_metrics", []),
            cv_fold_crps=payload.get("cv_fold_crps", []),
            parameter_estimates=payload.get("parameter_estimates", {}),
            roas_estimates=payload.get("roas_estimates", {}),
            cv_parameter_estimates=payload.get("cv_parameter_estimates", []),
            cv_fold_diagnostics=payload.get("cv_fold_diagnostics", []),
            cv_posterior_artifacts=payload.get("cv_posterior_artifacts", []),
            cv_fold_posteriors=payload.get("cv_fold_posteriors", []),
            fit_diagnostics=payload.get("fit_diagnostics", {}),
            model=payload.get("model"),
        )

    def _timed_out(self, elapsed_sec: float) -> bool:
        return self.run_timeout_sec is not None and elapsed_sec > self.run_timeout_sec


class AgentInterfaceBackend(BaseModel):
    """Backend that executes tasks through an agent command-line interface."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    agent_interface: Any = Field(...)
    baseline_profile: AgentModeProfile = Field(
        default_factory=lambda: AgentModeProfile(
            name="baseline",
            mode_instruction=(
                "Solve the MMM task without relying on repository skills or helper skill files."
            ),
        )
    )
    skilled_profile: AgentModeProfile = Field(
        default_factory=lambda: AgentModeProfile(
            name="skilled",
            mode_instruction=(
                "Use repository skills under .cursor/skills when relevant to solve the MMM task."
            ),
        )
    )
    payload_parser: Callable[[str], dict[str, Any]] = Field(
        default=parse_json_payload_from_stdout
    )

    def run(
        self,
        task: BenchmarkTaskSpec,
        mode: BenchmarkMode,
        seed: int,
    ) -> BenchmarkResult:
        """Execute one benchmark task through the configured agent interface."""
        logger.info(
            "backend_run_start task_id=%s mode=%s seed=%s", task.task_id, mode, seed
        )
        profile = self.baseline_profile if mode == "baseline" else self.skilled_profile
        execution = self.agent_interface.run_task(task=task, seed=seed, profile=profile)

        if execution.exit_code != 0:
            logger.warning(
                "backend_nonzero_exit task_id=%s mode=%s seed=%s exit_code=%s runtime_sec=%.3f",
                task.task_id,
                mode,
                seed,
                execution.exit_code,
                execution.runtime_sec,
            )
            return BenchmarkResult(
                task_id=task.task_id,
                mode=mode,
                seed=seed,
                status="failure",
                runtime_sec=execution.runtime_sec,
                metrics={},
                agent_events=execution.stream_events,
            )

        try:
            payload = self.payload_parser(execution.stdout)
        except Exception as error:
            logger.warning(
                "backend_json_parse_failure task_id=%s mode=%s seed=%s error=%s",
                task.task_id,
                mode,
                seed,
                error,
            )
            return BenchmarkResult(
                task_id=task.task_id,
                mode=mode,
                seed=seed,
                status="failure",
                runtime_sec=execution.runtime_sec,
                metrics={},
                agent_events=execution.stream_events,
                payload_validation_errors=[
                    {
                        "type": "json_parse_error",
                        "message": str(error),
                    }
                ],
            )

        try:
            validated_payload = AgentBackendPayload.model_validate(payload)
        except PydanticValidationError as error:
            logger.warning(
                "backend_payload_validation_failure task_id=%s mode=%s seed=%s error_count=%s",
                task.task_id,
                mode,
                seed,
                len(error.errors()),
            )
            return BenchmarkResult(
                task_id=task.task_id,
                mode=mode,
                seed=seed,
                status="failure",
                runtime_sec=execution.runtime_sec,
                metrics={},
                agent_events=execution.stream_events,
                payload_validation_errors=error.errors(),
            )

        interface_config = getattr(self.agent_interface, "config", None)
        isolation_enabled = bool(
            getattr(interface_config, "enable_baseline_path_isolation", False)
        )
        fit_diagnostics = dict(validated_payload.fit_diagnostics)
        fit_diagnostics["baseline_isolation_enabled"] = float(isolation_enabled)
        fit_diagnostics["mode_isolated"] = float(
            mode == "baseline" and isolation_enabled
        )
        logger.info(
            "backend_run_success task_id=%s mode=%s seed=%s status=%s crps_oos=%s fold_count=%s runtime_sec=%.3f",
            task.task_id,
            mode,
            seed,
            validated_payload.status,
            validated_payload.metrics.get("crps_oos"),
            len(validated_payload.fold_metrics),
            execution.runtime_sec,
        )

        return BenchmarkResult(
            task_id=task.task_id,
            mode=mode,
            seed=seed,
            status=validated_payload.status,
            runtime_sec=execution.runtime_sec,
            metrics=validated_payload.metrics,
            sample_stats_diverging=validated_payload.sample_stats_diverging,
            fold_metrics=[
                item.model_dump(mode="python")
                for item in validated_payload.fold_metrics
            ],
            cv_fold_crps=[
                item.model_dump(mode="python")
                for item in validated_payload.cv_fold_crps
            ],
            parameter_estimates=validated_payload.parameter_estimates,
            roas_estimates=validated_payload.roas_estimates,
            cv_parameter_estimates=[
                item.model_dump(mode="python")
                for item in validated_payload.cv_parameter_estimates
            ],
            cv_fold_diagnostics=[
                item.model_dump(mode="python")
                for item in validated_payload.cv_fold_diagnostics
            ],
            cv_posterior_artifacts=[
                item.model_dump(mode="python")
                for item in validated_payload.cv_posterior_artifacts
            ],
            agent_events=execution.stream_events,
            fit_diagnostics=fit_diagnostics,
            model=validated_payload.model,
        )
