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
import logging
import select
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmark.schemas import BenchmarkTaskSpec

logger = logging.getLogger("benchmark.agent_interface")


class AgentModeProfile(BaseModel):
    """Prompt behavior profile per benchmark mode."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(...)
    mode_instruction: str = Field(...)


class AgentExecutionConfig(BaseModel):
    """Configuration for command-line agent execution."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(default="cursor-agent")
    max_turns: int = Field(default=10, gt=0)
    timeout_sec: int = Field(default=900, gt=0)
    working_directory: str = Field(default=".")
    baseline_working_directory: str | None = Field(default=None)
    skilled_working_directory: str | None = Field(default=None)
    enable_baseline_path_isolation: bool = Field(default=False)
    baseline_denied_paths: list[str] = Field(default_factory=list)
    skip_permissions: bool = Field(default=True)
    debug_stream: bool = Field(default=False)
    debug_stream_partial_output: bool = Field(default=True)


class AgentExecutionResult(BaseModel):
    """Raw execution result from a single agent run."""

    model_config = ConfigDict(extra="forbid")

    stdout: str = Field(...)
    stderr: str = Field(...)
    exit_code: int = Field(...)
    runtime_sec: float = Field(..., ge=0.0)
    stream_events: list[dict[str, Any]] = Field(default_factory=list)


class FoldMetric(BaseModel):
    """Fold-level metric record returned by an agent."""

    model_config = ConfigDict(extra="forbid")

    fold_idx: int = Field(..., ge=0)
    crps: float = Field(...)


class FoldCRPSMetric(BaseModel):
    """Fold-level train/test CRPS payload used for richer diagnostics."""

    model_config = ConfigDict(extra="forbid")

    fold_idx: int = Field(..., ge=0)
    crps_train: float | None = Field(default=None)
    crps_test: float | None = Field(default=None)


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


class CVPosteriorArtifactReference(BaseModel):
    """Optional pointer to fold-level posterior artifact."""

    model_config = ConfigDict(extra="forbid")

    fold_idx: int = Field(..., ge=0)
    path: str = Field(...)
    variables: list[str] = Field(default_factory=list)


class AgentBackendPayload(BaseModel):
    """Strict payload schema expected from agent outputs."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    status: Literal["success", "failure", "timeout"] = Field(...)
    metrics: dict[str, float] = Field(default_factory=dict)
    sample_stats_diverging: list[int] = Field(default_factory=list)
    fold_metrics: list[FoldMetric] = Field(default_factory=list)
    cv_fold_crps: list[FoldCRPSMetric] = Field(default_factory=list)
    parameter_estimates: dict[str, Any] = Field(default_factory=dict)
    roas_estimates: dict[str, float] = Field(default_factory=dict)
    cv_parameter_estimates: list[CVParameterEstimateFold] = Field(default_factory=list)
    cv_fold_diagnostics: list[CVFoldDiagnostic] = Field(default_factory=list)
    cv_posterior_artifacts: list[CVPosteriorArtifactReference] = Field(
        default_factory=list
    )
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
- cv_fold_crps (optional)
- parameter_estimates
- roas_estimates
- cv_parameter_estimates (optional)
- cv_fold_diagnostics (optional)
- cv_posterior_artifacts (optional)
- fit_diagnostics (optional)

If execution fails, still return JSON with status='failure' and empty metric payloads.
""".strip()


class CursorAgentExecutionInterface(BaseModel):
    """Executes tasks via a Cursor-compatible CLI binary."""

    model_config = ConfigDict(extra="forbid")

    config: AgentExecutionConfig = Field(...)

    def _build_command(self, prompt: str) -> list[str]:
        command = [
            self.config.command,
            "--print",
            "--output-format",
            "stream-json" if self.config.debug_stream else "json",
        ]
        if self.config.debug_stream and self.config.debug_stream_partial_output:
            command.append("--stream-partial-output")
        if self.config.skip_permissions:
            command.append("--force")
        command.append(prompt)
        return command

    def _run_task_streaming(
        self,
        command: list[str],
        cwd: Path,
        task: BenchmarkTaskSpec,
        seed: int,
        profile: AgentModeProfile,
    ) -> AgentExecutionResult:
        """Run agent command in stream-json mode and log incremental events."""
        start = time.monotonic()
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        stream_events: list[dict[str, Any]] = []
        stdout_lines: list[str] = []
        result_payload_text: str | None = None
        event_count = 0
        timed_out = False

        if process.stdout is None:
            return AgentExecutionResult(
                stdout="",
                stderr="Agent streaming stdout pipe is unavailable.",
                exit_code=1,
                runtime_sec=0.0,
            )
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= self.config.timeout_sec:
                timed_out = True
                process.kill()
                break

            ready, _, _ = select.select([process.stdout], [], [], 0.5)
            if ready:
                line = process.stdout.readline()
                if line == "":
                    if process.poll() is not None:
                        break
                    continue
                stdout_lines.append(line)
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.debug(
                        "agent_stream_non_json task_id=%s mode=%s seed=%s line=%s",
                        task.task_id,
                        profile.name,
                        seed,
                        stripped[:200],
                    )
                    continue

                if not isinstance(event, dict):
                    continue
                stream_events.append(event)
                event_count += 1
                event_type = str(event.get("type", "unknown"))
                if event_count == 1 or event_count % 25 == 0 or event_type == "result":
                    logger.info(
                        "agent_stream_event task_id=%s mode=%s seed=%s event_count=%s event_type=%s",
                        task.task_id,
                        profile.name,
                        seed,
                        event_count,
                        event_type,
                    )
                if (
                    event_type == "result"
                    and event.get("subtype") == "success"
                    and isinstance(event.get("result"), str)
                ):
                    result_payload_text = event["result"]
            elif process.poll() is not None:
                break

        elapsed = time.monotonic() - start
        stdout_text = "".join(stdout_lines)
        if timed_out:
            logger.warning(
                "agent_process_timeout task_id=%s mode=%s seed=%s runtime_sec=%.3f timeout_sec=%s",
                task.task_id,
                profile.name,
                seed,
                elapsed,
                self.config.timeout_sec,
            )
            return AgentExecutionResult(
                stdout=stdout_text,
                stderr=f"Agent execution timed out after {self.config.timeout_sec} seconds.",
                exit_code=124,
                runtime_sec=elapsed,
                stream_events=stream_events,
            )

        exit_code = process.wait()
        payload_stdout = (
            result_payload_text if result_payload_text is not None else stdout_text
        )
        logger.info(
            "agent_process_completed task_id=%s mode=%s seed=%s exit_code=%s "
            "runtime_sec=%.3f stdout_bytes=%s stderr_bytes=%s stream_events=%s",
            task.task_id,
            profile.name,
            seed,
            exit_code,
            elapsed,
            len(payload_stdout),
            0,
            len(stream_events),
        )
        return AgentExecutionResult(
            stdout=payload_stdout,
            stderr="",
            exit_code=exit_code,
            runtime_sec=elapsed,
            stream_events=stream_events,
        )

    def _resolve_working_directory(self, profile: AgentModeProfile) -> Path:
        if (
            profile.name == "baseline"
            and self.config.baseline_working_directory is not None
        ):
            return Path(self.config.baseline_working_directory)
        if (
            profile.name == "skilled"
            and self.config.skilled_working_directory is not None
        ):
            return Path(self.config.skilled_working_directory)
        return Path(self.config.working_directory)

    def _resolve_denied_paths(self, cwd: Path) -> list[str]:
        raw_paths = self.config.baseline_denied_paths or [
            str(cwd / ".cursor" / "skills")
        ]
        denied_paths: list[str] = []
        for path in raw_paths:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = (cwd / candidate).resolve()
            else:
                candidate = candidate.resolve()
            denied_paths.append(str(candidate))
        return denied_paths

    def _build_sandbox_profile(self, denied_paths: list[str]) -> str:
        profile_lines = [
            "(version 1)",
            "(allow default)",
        ]
        for denied_path in denied_paths:
            escaped_path = denied_path.replace("\\", "\\\\").replace('"', '\\"')
            profile_lines.append(f'(deny file-read* (subpath "{escaped_path}"))')
        return "\n".join(profile_lines)

    def run_task(
        self,
        task: BenchmarkTaskSpec,
        seed: int,
        profile: AgentModeProfile,
    ) -> AgentExecutionResult:
        """Execute one benchmark task using a local Cursor-compatible CLI binary."""
        prompt = build_task_prompt(task=task, seed=seed, profile=profile)
        command = self._build_command(prompt=prompt)
        cwd = self._resolve_working_directory(profile)

        is_baseline = profile.name == "baseline"
        if is_baseline and self.config.enable_baseline_path_isolation:
            sandbox_exec = shutil.which("sandbox-exec")
            if sandbox_exec is None:
                return AgentExecutionResult(
                    stdout="",
                    stderr=(
                        "Baseline isolation requested but `sandbox-exec` is not available "
                        "on this machine."
                    ),
                    exit_code=1,
                    runtime_sec=0.0,
                )
            denied_paths = self._resolve_denied_paths(cwd=cwd)
            sandbox_profile = self._build_sandbox_profile(denied_paths=denied_paths)
            logger.info(
                "baseline_isolation_enabled task_id=%s mode=%s seed=%s denied_paths=%s",
                task.task_id,
                profile.name,
                seed,
                denied_paths,
            )
            command = [sandbox_exec, "-p", sandbox_profile, *command]

        log_command = [*command[:-1], "<prompt_redacted>"]
        logger.info(
            "sampling_started task_id=%s mode=%s seed=%s timeout_sec=%s cwd=%s command=%s",
            task.task_id,
            profile.name,
            seed,
            self.config.timeout_sec,
            cwd,
            log_command,
        )

        if self.config.debug_stream:
            return self._run_task_streaming(
                command=command,
                cwd=cwd,
                task=task,
                seed=seed,
                profile=profile,
            )

        start = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_sec,
                check=False,
            )
            elapsed = time.monotonic() - start
            logger.info(
                "agent_process_completed task_id=%s mode=%s seed=%s exit_code=%s "
                "runtime_sec=%.3f stdout_bytes=%s stderr_bytes=%s",
                task.task_id,
                profile.name,
                seed,
                completed.returncode,
                elapsed,
                len(completed.stdout),
                len(completed.stderr),
            )
        except subprocess.TimeoutExpired as error:
            elapsed = time.monotonic() - start
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            timeout_msg = (
                f"Agent execution timed out after {self.config.timeout_sec} seconds."
            )
            stderr_text = f"{stderr}\n{timeout_msg}".strip() if stderr else timeout_msg
            logger.warning(
                "agent_process_timeout task_id=%s mode=%s seed=%s runtime_sec=%.3f timeout_sec=%s",
                task.task_id,
                profile.name,
                seed,
                elapsed,
                self.config.timeout_sec,
            )
            return AgentExecutionResult(
                stdout=stdout,
                stderr=stderr_text,
                exit_code=124,
                runtime_sec=elapsed,
            )
        return AgentExecutionResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            runtime_sec=elapsed,
        )


# Backward-compatible alias for previous interface name.
ClaudeCliExecutionInterface = CursorAgentExecutionInterface


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
