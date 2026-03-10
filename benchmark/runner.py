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
"""Benchmark runner that compares baseline and skill-enabled modes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from benchmark.ground_truth import resolve_task_ground_truth
from benchmark.schemas import BenchmarkTaskSpec
from benchmark.scoring import (
    aggregate_cv_crps,
    aggregate_cv_train_test_crps,
    compute_cv_parameter_stability,
    compute_generalization_gap,
    compute_parameter_recovery_details,
    compute_roas_recovery_details,
    convergence_from_sample_stats,
    paired_delta,
    runtime_efficiency_metrics,
    summarize_cv_fold_diagnostics,
)

BenchmarkMode = Literal["baseline", "skilled"]
logger = logging.getLogger("benchmark.runner")


class BenchmarkResult(BaseModel):
    """Raw result returned by a benchmark backend."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_id: str = Field(...)
    mode: BenchmarkMode = Field(...)
    seed: int = Field(...)
    status: str = Field(...)
    runtime_sec: float = Field(..., ge=0.0)
    metrics: dict[str, float] = Field(default_factory=dict)
    sample_stats_diverging: list[int] = Field(default_factory=list)
    fold_metrics: list[dict[str, float]] = Field(default_factory=list)
    cv_fold_crps: list[dict[str, float]] = Field(default_factory=list)
    parameter_estimates: dict[str, Any] = Field(default_factory=dict)
    roas_estimates: dict[str, float] = Field(default_factory=dict)
    cv_parameter_estimates: list[dict[str, Any]] = Field(default_factory=list)
    cv_fold_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    cv_posterior_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    cv_fold_posteriors: list[Any] = Field(default_factory=list)
    agent_events: list[dict[str, Any]] = Field(default_factory=list)
    fit_diagnostics: dict[str, float] = Field(default_factory=dict)
    payload_validation_errors: list[dict[str, Any]] = Field(default_factory=list)
    model: Any | None = Field(default=None)


class BenchmarkOutput(BaseModel):
    """Collected output records after full benchmark execution."""

    model_config = ConfigDict(extra="forbid")

    run_records: list[dict[str, Any]] = Field(default_factory=list)
    paired_records: list[dict[str, Any]] = Field(default_factory=list)


class BenchmarkRunner(BaseModel):
    """Coordinator for paired benchmark runs and report exports."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    backend: Any = Field(...)
    output_dir: Path = Field(...)
    artifact_root: Path | None = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Prepare output and artifact directories after model initialization."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_root = self.output_dir / "artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        tasks: list[BenchmarkTaskSpec],
        seeds: list[int],
        modes: list[BenchmarkMode] | None = None,
    ) -> BenchmarkOutput:
        """Run all tasks under all seeds and modes, then export benchmark outputs."""
        modes = modes or ["baseline", "skilled"]
        run_records: list[dict[str, Any]] = []

        for task in tasks:
            for seed in seeds:
                for mode in modes:
                    self._log_dataset_status(task=task, mode=mode, seed=seed)
                    logger.info(
                        "runner_run_start task_id=%s mode=%s seed=%s",
                        task.task_id,
                        mode,
                        seed,
                    )
                    raw: BenchmarkResult = self.backend.run(
                        task=task, mode=mode, seed=seed
                    )
                    run_record, diagnostic_artifacts = self._build_run_record(
                        task=task, raw=raw
                    )
                    run_records.append(run_record)
                    self._persist_artifacts(
                        task=task,
                        raw=raw,
                        run_record=run_record,
                        diagnostic_artifacts=diagnostic_artifacts,
                    )
                    logger.info(
                        "runner_run_complete task_id=%s mode=%s seed=%s status=%s runtime_sec=%.3f",
                        task.task_id,
                        mode,
                        seed,
                        run_record["status"],
                        run_record["runtime_sec"],
                    )

        paired_records = self._build_paired_records(run_records=run_records)
        self._export(run_records=run_records, paired_records=paired_records)
        success_count = sum(1 for row in run_records if row.get("status") == "success")
        failure_count = sum(1 for row in run_records if row.get("status") == "failure")
        timeout_count = sum(1 for row in run_records if row.get("status") == "timeout")
        logger.info(
            "runner_all_complete run_count=%s success=%s failure=%s timeout=%s",
            len(run_records),
            success_count,
            failure_count,
            timeout_count,
        )
        return BenchmarkOutput(run_records=run_records, paired_records=paired_records)

    def _log_dataset_status(
        self,
        task: BenchmarkTaskSpec,
        mode: BenchmarkMode,
        seed: int,
    ) -> None:
        parsed = urlparse(task.dataset_path)
        if parsed.scheme in {"http", "https"}:
            logger.info(
                "dataset_check task_id=%s mode=%s seed=%s dataset=%s status=remote_dataset",
                task.task_id,
                mode,
                seed,
                task.dataset_path,
            )
            return
        dataset_path = Path(task.dataset_path).expanduser()
        if dataset_path.exists() and dataset_path.is_file():
            logger.info(
                "dataset_check task_id=%s mode=%s seed=%s dataset=%s status=ok",
                task.task_id,
                mode,
                seed,
                dataset_path,
            )
            return
        logger.warning(
            "dataset_check task_id=%s mode=%s seed=%s dataset=%s status=missing",
            task.task_id,
            mode,
            seed,
            dataset_path,
        )

    def _build_run_record(
        self,
        task: BenchmarkTaskSpec,
        raw: BenchmarkResult,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        divergences = xr.Dataset(
            data_vars={"diverging": (("draw",), raw.sample_stats_diverging)}
        )
        convergence = convergence_from_sample_stats(divergences)

        record = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "mode": raw.mode,
            "seed": raw.seed,
            "status": raw.status,
            "runtime_sec": raw.runtime_sec,
            "dataset_path": task.dataset_path,
            "sampler_nuts_sampler": task.sampler.nuts_sampler,
            "sampler_chains": task.sampler.chains,
            "sampler_cores": task.sampler.cores,
            "sampler_draws": task.sampler.draws,
            "cv_n_folds": task.cv.n_folds,
            "convergence_divergence_count": convergence["divergence_count"],
            "convergence_is_converged": convergence["is_converged"],
            "fold_count_observed": len(raw.fold_metrics),
            "folds_valid": len(raw.fold_metrics) >= task.cv.n_folds,
            "metric_crps_oos": float(raw.metrics.get("crps_oos", float("nan"))),
            "payload_validation_error_count": len(raw.payload_validation_errors),
            "metric_convergence_divergence_count": float(
                convergence["divergence_count"]
            ),
        }
        diagnostic_artifacts: dict[str, Any] = {}
        if raw.fold_metrics:
            cv_summary = aggregate_cv_crps(raw.fold_metrics)
            if pd.isna(record["metric_crps_oos"]):
                record["metric_crps_oos"] = cv_summary["crps_mean"]
            record["metric_crps_cv_mean"] = cv_summary["crps_mean"]
            record["metric_crps_cv_std"] = cv_summary["crps_std"]
            record["metric_cv_n_folds"] = cv_summary["n_folds"]
            train_test_crps = aggregate_cv_train_test_crps(raw.fold_metrics)
            record["metric_crps_train_cv_mean"] = train_test_crps["train_crps_mean"]
            record["metric_crps_train_cv_std"] = train_test_crps["train_crps_std"]
            record["metric_crps_test_cv_mean"] = train_test_crps["test_crps_mean"]
            record["metric_crps_test_cv_std"] = train_test_crps["test_crps_std"]
            gap = compute_generalization_gap(raw.fold_metrics)
            record["metric_generalization_gap_mean"] = gap["gap_mean"]
            record["metric_generalization_gap_std"] = gap["gap_std"]
            record["metric_generalization_gap_max"] = gap["gap_max"]
            diagnostic_artifacts["cv_generalization_gap"] = gap
        record["pass_no_divergence"] = bool(record["convergence_divergence_count"] == 0)
        record["pass_cv_fold_count"] = bool(
            record["fold_count_observed"] >= task.cv.n_folds
        )

        for metric_name, metric_value in raw.metrics.items():
            record[f"metric_{metric_name}"] = float(metric_value)

        cv_param_stability = compute_cv_parameter_stability(raw.cv_parameter_estimates)
        record["metric_cv_param_std_mean"] = cv_param_stability["param_std_mean"]
        record["metric_cv_param_iqr_mean"] = cv_param_stability["param_iqr_mean"]
        record["metric_cv_param_cv_mean"] = cv_param_stability["param_cv_mean"]
        record["metric_cv_param_range_mean"] = cv_param_stability["param_range_mean"]
        record["metric_cv_param_count"] = float(cv_param_stability["parameter_count"])
        diagnostic_artifacts["cv_parameter_stability"] = cv_param_stability
        cv_diag_summary = summarize_cv_fold_diagnostics(raw.cv_fold_diagnostics)
        record["metric_cv_divergence_mean"] = cv_diag_summary["divergence_mean"]
        record["metric_cv_divergence_max"] = cv_diag_summary["divergence_max"]
        record["metric_cv_rhat_max"] = cv_diag_summary["rhat_max"]
        record["metric_cv_ess_bulk_min"] = cv_diag_summary["ess_bulk_min"]
        record["metric_cv_bfmi_min"] = cv_diag_summary["bfmi_min"]
        diagnostic_artifacts["cv_fold_diagnostics_summary"] = cv_diag_summary

        runtime_metrics = runtime_efficiency_metrics(
            runtime_sec=raw.runtime_sec, n_folds=len(raw.fold_metrics)
        )
        record["metric_runtime_sec"] = runtime_metrics["runtime_sec"]
        record["metric_runtime_per_fold_sec"] = runtime_metrics["runtime_per_fold_sec"]
        record["metric_folds_per_second"] = runtime_metrics["folds_per_second"]
        diagnostic_artifacts["runtime_efficiency"] = runtime_metrics

        if task.ground_truth is not None:
            truth_parameters, truth_roas = resolve_task_ground_truth(task)
            parameter_recovery_details = compute_parameter_recovery_details(
                estimate=raw.parameter_estimates,
                truth=truth_parameters,
            )
            roas_recovery_details = compute_roas_recovery_details(
                estimate=raw.roas_estimates,
                truth=truth_roas,
            )
            record["metric_parameter_recovery_mae"] = parameter_recovery_details["mae"]
            record["metric_parameter_recovery_rmse"] = parameter_recovery_details[
                "rmse"
            ]
            record["metric_parameter_recovery_median_ae"] = parameter_recovery_details[
                "median_ae"
            ]
            record["metric_parameter_recovery_max_ae"] = parameter_recovery_details[
                "max_ae"
            ]
            record["metric_parameter_recovery_shared_key_count"] = float(
                parameter_recovery_details["shared_key_count"]
            )
            record["metric_parameter_recovery_missing_estimate_count"] = float(
                parameter_recovery_details["missing_estimate_key_count"]
            )
            record["metric_parameter_recovery_missing_truth_count"] = float(
                parameter_recovery_details["missing_truth_key_count"]
            )

            record["metric_roas_recovery_mae"] = roas_recovery_details["mae"]
            record["metric_roas_recovery_rmse"] = roas_recovery_details["rmse"]
            record["metric_roas_recovery_median_ae"] = roas_recovery_details[
                "median_ae"
            ]
            record["metric_roas_recovery_max_ae"] = roas_recovery_details["max_ae"]
            record["metric_roas_recovery_shared_key_count"] = float(
                roas_recovery_details["shared_key_count"]
            )
            record["metric_roas_recovery_missing_estimate_count"] = float(
                roas_recovery_details["missing_estimate_key_count"]
            )
            record["metric_roas_recovery_missing_truth_count"] = float(
                roas_recovery_details["missing_truth_key_count"]
            )

            diagnostic_artifacts["parameter_recovery_details"] = (
                parameter_recovery_details
            )
            diagnostic_artifacts["roas_recovery_details"] = roas_recovery_details

        diagnostic_artifacts["fit_difference_context"] = {
            "task_id": task.task_id,
            "mode": raw.mode,
            "seed": raw.seed,
            "status": raw.status,
            "runtime_sec": raw.runtime_sec,
            "convergence": convergence,
            "metrics": raw.metrics,
            "cv_fold_count": len(raw.fold_metrics),
            "cv_fold_crps": raw.cv_fold_crps,
            "cv_fold_diagnostics": raw.cv_fold_diagnostics,
            "cv_posterior_artifacts": raw.cv_posterior_artifacts,
            "fit_diagnostics": raw.fit_diagnostics,
            "payload_validation_error_count": len(raw.payload_validation_errors),
        }

        return record, diagnostic_artifacts

    def _persist_artifacts(
        self,
        task: BenchmarkTaskSpec,
        raw: BenchmarkResult,
        run_record: dict[str, Any],
        diagnostic_artifacts: dict[str, Any],
    ) -> None:
        artifact_dir = self.artifact_root / task.task_id / raw.mode / f"seed_{raw.seed}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "artifact_dir_ready task_id=%s mode=%s seed=%s path=%s",
            task.task_id,
            raw.mode,
            raw.seed,
            artifact_dir,
        )

        run_record["artifacts_path"] = str(artifact_dir)

        # Persist fold-level metrics for detailed reporting.
        fold_metrics_path = artifact_dir / "fold_metrics.json"
        fold_metrics_path.write_text(
            json.dumps(raw.fold_metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if raw.cv_fold_crps:
            fold_crps_path = artifact_dir / "cv_fold_crps.json"
            fold_crps_path.write_text(
                json.dumps(raw.cv_fold_crps, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if raw.agent_events:
            events_path = artifact_dir / "agent_events.jsonl"
            events_blob = "\n".join(
                json.dumps(event, sort_keys=True) for event in raw.agent_events
            )
            events_path.write_text(f"{events_blob}\n", encoding="utf-8")
            run_record["agent_events_path"] = str(events_path)

        manifest_entries: list[dict[str, Any]] = []
        cv_folds_dir = artifact_dir / "cv_folds"
        if raw.cv_fold_posteriors:
            cv_folds_dir.mkdir(parents=True, exist_ok=True)
            for idx, posterior in enumerate(raw.cv_fold_posteriors):
                if not isinstance(posterior, xr.Dataset):
                    continue
                posterior_path = cv_folds_dir / f"fold_{idx}.nc"
                posterior.to_netcdf(posterior_path)
                manifest_entries.append(
                    {
                        "fold_idx": idx,
                        "path": str(posterior_path),
                        "variables": list(posterior.data_vars.keys()),
                        "dims": {
                            key: int(value) for key, value in posterior.sizes.items()
                        },
                    }
                )

        for payload in raw.cv_posterior_artifacts:
            if isinstance(payload, dict):
                manifest_entries.append(payload)

        if manifest_entries:
            manifest_path = artifact_dir / "cv_posterior_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest_entries, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            run_record["cv_posterior_manifest_path"] = str(manifest_path)

        for artifact_name, payload in diagnostic_artifacts.items():
            artifact_path = artifact_dir / f"{artifact_name}.json"
            artifact_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        if raw.model is not None and hasattr(raw.model, "save"):
            raw.model.save(str(artifact_dir / "model.nc"))

    def _build_paired_records(
        self, run_records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        for record in run_records:
            key = (record["task_id"], int(record["seed"]))
            by_key.setdefault(key, {})
            by_key[key][record["mode"]] = record

        paired_records: list[dict[str, Any]] = []
        for (task_id, seed), mode_records in by_key.items():
            if "baseline" not in mode_records or "skilled" not in mode_records:
                continue
            baseline = mode_records["baseline"]
            skilled = mode_records["skilled"]
            row: dict[str, Any] = {
                "task_id": task_id,
                "seed": seed,
            }
            metric_cols = sorted(
                col
                for col in baseline
                if col.startswith("metric_")
                and col in skilled
                and pd.notna(baseline[col])
                and pd.notna(skilled[col])
            )
            for metric_col in metric_cols:
                metric_name = metric_col.removeprefix("metric_")
                row[f"delta_{metric_name}"] = paired_delta(
                    baseline_value=float(baseline[metric_col]),
                    skilled_value=float(skilled[metric_col]),
                )
            paired_records.append(row)
        return paired_records

    def _export(
        self, run_records: list[dict[str, Any]], paired_records: list[dict[str, Any]]
    ) -> None:
        run_df = pd.DataFrame(run_records)
        paired_df = pd.DataFrame(paired_records)

        run_df.to_csv(self.output_dir / "run_results.csv", index=False)
        paired_df.to_csv(self.output_dir / "paired_deltas.csv", index=False)
        run_df.to_json(
            self.output_dir / "run_results.jsonl", orient="records", lines=True
        )

        task_summary = self._summarize_by_task(run_df)
        task_summary.to_csv(self.output_dir / "task_summary.csv", index=False)

        benchmark_summary = self._summarize_benchmark(task_summary)
        benchmark_summary.to_csv(self.output_dir / "benchmark_summary.csv", index=False)
        logger.info(
            "exports_written output_dir=%s run_results=%s paired_deltas=%s task_summary=%s benchmark_summary=%s",
            self.output_dir,
            self.output_dir / "run_results.csv",
            self.output_dir / "paired_deltas.csv",
            self.output_dir / "task_summary.csv",
            self.output_dir / "benchmark_summary.csv",
        )

    def _summarize_by_task(self, run_df: pd.DataFrame) -> pd.DataFrame:
        metric_cols = [col for col in run_df.columns if col.startswith("metric_")]
        if not metric_cols:
            return pd.DataFrame(columns=["task_id", "mode", "run_count"])
        grouped = (
            run_df.groupby(["task_id", "mode"], dropna=False)[metric_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        grouped["run_count"] = run_df.groupby(["task_id", "mode"]).size().values
        return grouped

    def _summarize_benchmark(self, task_summary: pd.DataFrame) -> pd.DataFrame:
        if task_summary.empty:
            return pd.DataFrame(columns=["mode", "task_count"])
        metric_cols = [col for col in task_summary.columns if col.startswith("metric_")]
        summary = (
            task_summary.groupby("mode", dropna=False)[metric_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        summary["task_count"] = task_summary.groupby("mode").size().values
        return summary


def serialize_result(result: BenchmarkResult) -> dict[str, Any]:
    """Convert result to a JSON-serializable dictionary."""
    payload = result.model_dump(mode="json")
    if payload.get("model") is not None:
        payload["model"] = str(type(result.model).__name__)
    return payload
