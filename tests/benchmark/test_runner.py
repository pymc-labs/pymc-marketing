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
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from benchmark.runner import BenchmarkResult, BenchmarkRunner
from benchmark.schemas import load_task_spec


class _FakeModel:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def save(self, fname: str, **kwargs) -> None:
        Path(fname).write_text(f"model:{self.marker}", encoding="utf-8")


class _FakeBackend:
    def run(self, task, mode, seed):
        posterior = xr.Dataset(
            data_vars={
                "adstock_alpha": (
                    ("chain", "draw", "channel"),
                    np.ones((1, 5, 1), dtype=np.float64) * 0.2,
                )
            },
            coords={"chain": [0], "draw": [0, 1, 2, 3, 4], "channel": ["x1"]},
        )
        return BenchmarkResult(
            task_id=task.task_id,
            mode=mode,
            seed=seed,
            status="success",
            runtime_sec=1.0,
            metrics={"crps_oos": 0.1 if mode == "skilled" else 0.2, "pass_flag": 1.0},
            sample_stats_diverging=[0, 0, 0, 0],
            fold_metrics=[
                {
                    "fold_idx": i,
                    "crps": 0.1 + i * 0.01,
                    "crps_train": 0.08 + i * 0.01,
                    "crps_test": 0.1 + i * 0.01,
                }
                for i in range(5)
            ],
            parameter_estimates={"x1": 0.2},
            roas_estimates={"x1": 1.1},
            cv_parameter_estimates=[
                {"fold_idx": i, "parameter_estimates": {"x1": 0.2 + i * 0.01}}
                for i in range(5)
            ],
            cv_fold_diagnostics=[
                {
                    "fold_idx": i,
                    "divergence_count": 0,
                    "rhat_max": 1.01,
                    "ess_bulk_min": 100.0,
                    "bfmi_min": 0.6,
                }
                for i in range(5)
            ],
            cv_fold_posteriors=[posterior for _ in range(5)],
            fit_diagnostics={"rhat_max": 1.01},
            model=_FakeModel(marker=f"{task.task_id}-{mode}-{seed}"),
        )


def test_runner_exports_csv_and_jsonl(tmp_path: Path) -> None:
    task = load_task_spec(
        {
            "task_id": "task_a",
            "task_type": "mmm_1d",
            "dataset_path": "data/unused.csv",
            "date_column": "date_week",
            "target_column": "y",
            "channel_columns": ["x1"],
            "cv": {"n_init": 12, "forecast_horizon": 2, "step_size": 1, "n_folds": 5},
        }
    )
    runner = BenchmarkRunner(backend=_FakeBackend(), output_dir=tmp_path)
    output = runner.run(tasks=[task], seeds=[11], modes=["baseline", "skilled"])

    assert (tmp_path / "run_results.csv").exists()
    assert (tmp_path / "paired_deltas.csv").exists()
    assert (tmp_path / "task_summary.csv").exists()
    assert (tmp_path / "benchmark_summary.csv").exists()
    assert (tmp_path / "run_results.jsonl").exists()
    assert len(output.run_records) == 2

    df = pd.read_csv(tmp_path / "paired_deltas.csv")
    assert "delta_crps_oos" in df.columns
    assert float(df.loc[0, "delta_crps_oos"]) < 0.0

    run_df = pd.read_csv(tmp_path / "run_results.csv")
    assert "metric_parameter_recovery_mae" not in run_df.columns
    assert "metric_cv_param_std_mean" in run_df.columns
    assert "metric_generalization_gap_mean" in run_df.columns
    assert "metric_runtime_per_fold_sec" in run_df.columns
