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

from benchmark.runner import BenchmarkResult, BenchmarkRunner
from benchmark.schemas import load_task_spec


class _FakeModel:
    def save(self, fname: str, **kwargs) -> None:
        Path(fname).write_text("saved", encoding="utf-8")


class _FakeBackend:
    def run(self, task, mode, seed):
        return BenchmarkResult(
            task_id=task.task_id,
            mode=mode,
            seed=seed,
            status="success",
            runtime_sec=1.0,
            metrics={"crps_oos": 0.2},
            sample_stats_diverging=[0, 0],
            fold_metrics=[{"fold_idx": i, "crps": 0.2} for i in range(5)],
            parameter_estimates={},
            roas_estimates={},
            cv_parameter_estimates=[
                {"fold_idx": i, "parameter_estimates": {"x1": 0.2}} for i in range(5)
            ],
            fit_diagnostics={"rhat_max": 1.01},
            model=_FakeModel(),
        )


def test_runner_persists_model_artifacts(tmp_path: Path) -> None:
    task = load_task_spec(
        {
            "task_id": "task_artifact",
            "task_type": "mmm_1d",
            "dataset_path": "data/unused.csv",
            "date_column": "date_week",
            "target_column": "y",
            "channel_columns": ["x1"],
            "cv": {"n_init": 12, "forecast_horizon": 2, "step_size": 1, "n_folds": 5},
        }
    )
    runner = BenchmarkRunner(backend=_FakeBackend(), output_dir=tmp_path)
    runner.run(tasks=[task], seeds=[5], modes=["baseline"])

    saved = list(tmp_path.glob("artifacts/**/model.nc"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == "saved"
    assert list(tmp_path.glob("artifacts/**/cv_parameter_stability.json"))
    assert list(tmp_path.glob("artifacts/**/fit_difference_context.json"))
