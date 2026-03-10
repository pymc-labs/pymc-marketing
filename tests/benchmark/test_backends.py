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
from dataclasses import dataclass

from benchmark.agent_interface import (
    AgentExecutionResult,
    AgentModeProfile,
    parse_json_payload_from_stdout,
)
from benchmark.backends import AgentInterfaceBackend
from benchmark.schemas import load_task_spec


@dataclass
class _FakeAgentInterface:
    output: str

    def run_task(self, task, seed, profile):
        return AgentExecutionResult(
            stdout=self.output,
            stderr="",
            exit_code=0,
            runtime_sec=1.23,
        )


def test_parse_json_payload_from_stdout_plain_json() -> None:
    payload = parse_json_payload_from_stdout(
        '{"status":"success","metrics":{"crps_oos":0.1}}'
    )
    assert payload["status"] == "success"
    assert payload["metrics"]["crps_oos"] == 0.1


def test_agent_interface_backend_uses_mode_profiles(caplog) -> None:
    task = load_task_spec(
        {
            "task_id": "task_agent",
            "task_type": "mmm_1d",
            "dataset_path": "data/unused.csv",
            "date_column": "date",
            "target_column": "y",
            "channel_columns": ["x1", "x2"],
            "cv": {"n_init": 10, "forecast_horizon": 2, "step_size": 1, "n_folds": 5},
        }
    )
    interface = _FakeAgentInterface(
        output=(
            '{"status":"success","metrics":{"crps_oos":0.11},'
            '"sample_stats_diverging":[0,0],"fold_metrics":[{"fold_idx":0,"crps":0.11}],'
            '"cv_fold_crps":[{"fold_idx":0,"crps_train":0.10,"crps_test":0.11}],'
            '"parameter_estimates":{"x1":0.2},"roas_estimates":{"x1":1.1},'
            '"cv_parameter_estimates":[{"fold_idx":0,"parameter_estimates":{"x1":0.2}}],'
            '"cv_fold_diagnostics":[{"fold_idx":0,"divergence_count":0}],'
            '"cv_posterior_artifacts":[{"fold_idx":0,"path":"dummy.nc","variables":["adstock_alpha"]}],'
            '"fit_diagnostics":{"rhat_max":1.01}}'
        )
    )
    backend = AgentInterfaceBackend(
        agent_interface=interface,
        baseline_profile=AgentModeProfile(
            name="baseline", mode_instruction="no skills"
        ),
        skilled_profile=AgentModeProfile(name="skilled", mode_instruction="use skills"),
    )
    caplog.set_level("INFO", logger="benchmark.backends")

    result = backend.run(task=task, mode="baseline", seed=1)
    assert result.status == "success"
    assert result.metrics["crps_oos"] == 0.11
    assert result.cv_parameter_estimates
    assert result.cv_fold_crps
    assert "backend_run_success" in caplog.text


def test_agent_interface_backend_returns_structured_validation_errors() -> None:
    task = load_task_spec(
        {
            "task_id": "task_agent_invalid",
            "task_type": "mmm_1d",
            "dataset_path": "data/unused.csv",
            "date_column": "date",
            "target_column": "y",
            "channel_columns": ["x1", "x2"],
            "cv": {"n_init": 10, "forecast_horizon": 2, "step_size": 1, "n_folds": 5},
        }
    )
    interface = _FakeAgentInterface(
        # Invalid payload: missing "status", wrong metrics type
        output='{"metrics":{"crps_oos":"bad_type"},"fold_metrics":[{"fold_idx":0,"crps":0.11}]}'
    )
    backend = AgentInterfaceBackend(agent_interface=interface)

    result = backend.run(task=task, mode="baseline", seed=1)
    assert result.status == "failure"
    assert result.payload_validation_errors


def test_agent_interface_backend_seed_deterministic_and_mode_consistent() -> None:
    task = load_task_spec(
        {
            "task_id": "task_agent_deterministic",
            "task_type": "mmm_1d",
            "dataset_path": "data/unused.csv",
            "date_column": "date",
            "target_column": "y",
            "channel_columns": ["x1", "x2"],
            "cv": {"n_init": 8, "forecast_horizon": 2, "step_size": 1, "n_folds": 5},
        }
    )
    interface = _FakeAgentInterface(
        output=(
            '{"status":"success","metrics":{"crps_oos":0.21},'
            '"sample_stats_diverging":[0,0],"fold_metrics":[{"fold_idx":0,"crps":0.21}],'
            '"cv_fold_crps":[{"fold_idx":0,"crps_train":0.20,"crps_test":0.21}],'
            '"parameter_estimates":{"x1":0.3},"roas_estimates":{"x1":1.2},'
            '"cv_parameter_estimates":[{"fold_idx":0,"parameter_estimates":{"x1":0.3}}],'
            '"cv_fold_diagnostics":[{"fold_idx":0,"divergence_count":0}],'
            '"fit_diagnostics":{"rhat_max":1.01}}'
        )
    )
    backend = AgentInterfaceBackend(agent_interface=interface)

    baseline = backend.run(task=task, mode="baseline", seed=13)
    skilled = backend.run(task=task, mode="skilled", seed=13)
    baseline_repeat = backend.run(task=task, mode="baseline", seed=13)

    assert baseline.status == "success"
    assert skilled.status == "success"
    assert baseline.metrics == skilled.metrics
    assert baseline.roas_estimates == skilled.roas_estimates
    assert baseline.parameter_estimates == skilled.parameter_estimates
    assert baseline.metrics == baseline_repeat.metrics
