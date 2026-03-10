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

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from benchmark.agent_interface import (
    AgentBackendPayload,
    AgentModeProfile,
    parse_json_payload_from_stdout,
)
from benchmark.ground_truth import (
    compute_true_parameters_from_roas_dataset,
    compute_true_roas_from_roas_dataset,
)
from benchmark.runner import BenchmarkMode, BenchmarkResult
from benchmark.schemas import BenchmarkTaskSpec
from pymc_marketing.mmm.time_slice_cross_validation import TimeSliceCrossValidator

AgentRunnerCallable = Callable[[BenchmarkTaskSpec, int], dict[str, Any]]


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


class DummyDeterministicBackend(BaseModel):
    """Deterministic placeholder backend for local harness validation."""

    model_config = ConfigDict(extra="forbid")

    def _build_fold_posterior_dataset(
        self,
        fold_idx: int,
        mode: BenchmarkMode,
        rng: np.random.Generator,
        channels: list[str],
    ) -> xr.Dataset:
        """Create synthetic posterior samples for HDI stability plots."""
        chain = np.arange(2)
        draw = np.arange(150)
        channel = np.asarray(channels, dtype=object)
        mode_shift = 0.01 if mode == "skilled" else 0.0
        center_alpha = 0.18 + 0.01 * fold_idx + mode_shift
        center_beta = 0.35 + 0.01 * fold_idx + mode_shift
        alpha = rng.normal(loc=center_alpha, scale=0.015, size=(2, 150, len(channel)))
        beta = rng.normal(loc=center_beta, scale=0.02, size=(2, 150, len(channel)))
        return xr.Dataset(
            data_vars={
                "adstock_alpha": (("chain", "draw", "channel"), alpha),
                "saturation_beta": (("chain", "draw", "channel"), beta),
            },
            coords={"chain": chain, "draw": draw, "channel": channel},
        )

    def _load_dataset_for_cv(self, task: BenchmarkTaskSpec) -> pd.DataFrame:
        """Load task dataset and ensure a valid date column for CV splitting."""
        try:
            frame = pd.read_csv(task.dataset_path)
        except Exception:
            # Fallback synthetic frame used only if dataset loading fails.
            n_rows = (
                task.cv.n_init
                + task.cv.forecast_horizon
                + task.cv.step_size * max(task.cv.n_folds, 1)
                + 5
            )
            frame = pd.DataFrame(
                {
                    task.date_column: pd.date_range(
                        "2021-01-01", periods=n_rows, freq="W"
                    )
                }
            )

        if task.date_column not in frame.columns:
            frame = frame.copy()
            frame[task.date_column] = pd.date_range(
                "2021-01-01",
                periods=len(frame),
                freq="W",
            )

        frame = frame.copy()
        frame[task.date_column] = pd.to_datetime(frame[task.date_column])
        return frame

    def _build_roas_task_outputs(
        self,
        task: BenchmarkTaskSpec,
        mode: BenchmarkMode,
        rng: np.random.Generator,
        frame: pd.DataFrame,
    ) -> tuple[dict[str, float], dict[str, float], float]:
        """Build task-3 specific outputs with explicit calibration gap."""
        if mode == "skilled":
            # Skilled mode emulates calibrated behavior from the ROAS notebook.
            roas_estimates = compute_true_roas_from_roas_dataset(
                dataset_path=task.dataset_path,
                channel_columns=task.channel_columns,
            )
            parameter_estimates = compute_true_parameters_from_roas_dataset(
                dataset_path=task.dataset_path,
                channel_columns=task.channel_columns,
            )
            roas_estimates = {
                key: float(value + 0.02 * rng.normal())
                for key, value in roas_estimates.items()
            }
            parameter_estimates = {
                key: float(value + 0.02 * rng.normal())
                for key, value in parameter_estimates.items()
            }
            calibration_flag = 1.0
            return roas_estimates, parameter_estimates, calibration_flag

        # Baseline mode emulates naive (uncalibrated) ROAS from observed outcome.
        roas_estimates: dict[str, float] = {}
        for channel in task.channel_columns:
            spend_sum = float(frame[channel].sum()) if channel in frame.columns else 0.0
            if spend_sum <= 0:
                roas_estimates[channel] = float("nan")
            else:
                roas_estimates[channel] = float(
                    frame[task.target_column].sum() / spend_sum
                )

        parameter_estimates = {
            f"beta_{channel}_adstock_saturated": float(0.5 + 0.2 * rng.random())
            for channel in task.channel_columns
        }
        calibration_flag = 0.0
        return roas_estimates, parameter_estimates, calibration_flag

    def run(
        self, task: BenchmarkTaskSpec, mode: BenchmarkMode, seed: int
    ) -> BenchmarkResult:
        """Generate deterministic synthetic benchmark outputs for local validation."""
        rng = np.random.default_rng(seed + (1000 if mode == "skilled" else 0))
        base_crps = 0.2 if mode == "baseline" else 0.16
        frame = self._load_dataset_for_cv(task)
        target = (
            frame[task.target_column]
            if task.target_column in frame.columns
            else pd.Series(np.zeros(len(frame)))
        )

        cv = TimeSliceCrossValidator(
            n_init=task.cv.n_init,
            forecast_horizon=task.cv.forecast_horizon,
            date_column=task.date_column,
            step_size=task.cv.step_size,
        )

        splits = list(cv.split(frame, target))
        selected_splits = splits[: task.cv.n_folds]
        fold_metrics = []
        cv_fold_crps = []
        cv_parameter_estimates = []
        cv_fold_posteriors: list[xr.Dataset] = []
        for fold_idx, (train_idx, test_idx) in enumerate(selected_splits):
            fold_scale = len(test_idx) / max(len(train_idx), 1)
            crps_train = float(
                base_crps * 0.9 + 0.005 * fold_scale + 0.005 * rng.random()
            )
            crps_test = float(base_crps + 0.01 * fold_scale + 0.01 * rng.random())
            fold_metrics.append(
                {
                    "fold_idx": fold_idx,
                    "crps": crps_test,
                    "crps_train": crps_train,
                    "crps_test": crps_test,
                }
            )
            cv_fold_crps.append(
                {
                    "fold_idx": fold_idx,
                    "crps_train": crps_train,
                    "crps_test": crps_test,
                }
            )
            cv_parameter_estimates.append(
                {
                    "fold_idx": fold_idx,
                    "parameter_estimates": {
                        "x1": float(0.20 + 0.01 * fold_idx + 0.005 * rng.normal()),
                        "x2": float(0.30 + 0.01 * fold_idx + 0.005 * rng.normal()),
                    },
                }
            )
            cv_fold_posteriors.append(
                self._build_fold_posterior_dataset(
                    fold_idx=fold_idx,
                    mode=mode,
                    rng=rng,
                    channels=task.channel_columns,
                )
            )
        roas_estimates = {
            "x1": float(1.1 + 0.05 * rng.random()),
            "x2": float(0.9 + 0.05 * rng.random()),
        }
        parameter_estimates: dict[str, float] = {
            "x1": float(0.20 + 0.01 * rng.random()),
            "x2": float(0.30 + 0.01 * rng.random()),
            "beta_x1_adstock_saturated": float(2.0 + 0.1 * rng.random()),
            "beta_x2_adstock_saturated": float(3.0 + 0.1 * rng.random()),
        }
        calibration_applied = 0.0
        if task.task_type == "mmm_roas_confounding":
            (
                roas_estimates,
                parameter_estimates,
                calibration_applied,
            ) = self._build_roas_task_outputs(
                task=task,
                mode=mode,
                rng=rng,
                frame=frame,
            )

        return BenchmarkResult(
            task_id=task.task_id,
            mode=mode,
            seed=seed,
            status="success",
            runtime_sec=float(5.0 + rng.random()),
            metrics={
                "crps_oos": float(np.mean([m["crps"] for m in fold_metrics])),
                "parameter_stability": float(0.05 + 0.02 * rng.random()),
                "calibration_applied": calibration_applied,
            },
            sample_stats_diverging=[0] * (task.sampler.chains * task.sampler.draws),
            fold_metrics=fold_metrics,
            cv_fold_crps=cv_fold_crps,
            parameter_estimates=parameter_estimates,
            roas_estimates=roas_estimates,
            cv_parameter_estimates=cv_parameter_estimates,
            cv_fold_diagnostics=[
                {"fold_idx": fold_idx, "divergence_count": 0}
                for fold_idx in range(len(fold_metrics))
            ],
            cv_fold_posteriors=cv_fold_posteriors,
            fit_diagnostics={
                "mean_runtime_proxy": float(5.0 + rng.random()),
                "stability_proxy": float(0.1 + 0.01 * rng.random()),
                "calibration_applied": calibration_applied,
            },
            model=None,
        )


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
        profile = self.baseline_profile if mode == "baseline" else self.skilled_profile
        execution = self.agent_interface.run_task(task=task, seed=seed, profile=profile)

        if execution.exit_code != 0:
            return BenchmarkResult(
                task_id=task.task_id,
                mode=mode,
                seed=seed,
                status="failure",
                runtime_sec=execution.runtime_sec,
                metrics={},
            )

        try:
            payload = self.payload_parser(execution.stdout)
        except Exception as error:
            return BenchmarkResult(
                task_id=task.task_id,
                mode=mode,
                seed=seed,
                status="failure",
                runtime_sec=execution.runtime_sec,
                metrics={},
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
            return BenchmarkResult(
                task_id=task.task_id,
                mode=mode,
                seed=seed,
                status="failure",
                runtime_sec=execution.runtime_sec,
                metrics={},
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
            fit_diagnostics=fit_diagnostics,
            model=validated_payload.model,
        )
