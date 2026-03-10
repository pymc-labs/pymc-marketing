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
import pytest

from benchmark.schemas import BenchmarkTaskSpec, load_task_spec


def test_load_task_spec_enforces_minimum_cv_folds() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        load_task_spec(
            {
                "task_id": "bad_task",
                "task_type": "mmm_1d",
                "dataset_path": "data/something.csv",
                "date_column": "date_week",
                "target_column": "y",
                "channel_columns": ["x1", "x2"],
                "cv": {
                    "n_init": 10,
                    "forecast_horizon": 2,
                    "step_size": 1,
                    "n_folds": 4,
                },
            }
        )


def test_load_task_spec_applies_default_sampler_policy() -> None:
    task = load_task_spec(
        {
            "task_id": "good_task",
            "task_type": "mmm_1d",
            "dataset_path": "data/something.csv",
            "date_column": "date_week",
            "target_column": "y",
            "channel_columns": ["x1", "x2"],
            "cv": {"n_init": 10, "forecast_horizon": 2, "step_size": 1, "n_folds": 5},
        }
    )

    assert isinstance(task, BenchmarkTaskSpec)
    assert task.sampler.nuts_sampler == "nutpie"
    assert task.sampler.chains == 14
    assert task.sampler.cores == 14
    assert task.sampler.draws == 500


def test_load_task_spec_supports_ground_truth_for_parameter_and_roas_recovery() -> None:
    task = load_task_spec(
        {
            "task_id": "task_b",
            "task_type": "mmm_multidimensional",
            "dataset_path": "data/mmm_multidimensional_example.csv",
            "date_column": "date_week",
            "target_column": "y",
            "channel_columns": ["x1", "x2"],
            "cv": {"n_init": 40, "forecast_horizon": 4, "step_size": 2, "n_folds": 5},
            "ground_truth": {
                "parameters": {"beta_channel": {"x1": 0.2, "x2": 0.3}},
                "roas": {"x1": 1.2, "x2": 0.9},
            },
        }
    )

    assert task.ground_truth is not None
    assert "beta_channel" in task.ground_truth.parameters
    assert task.ground_truth.roas["x1"] == pytest.approx(1.2)
