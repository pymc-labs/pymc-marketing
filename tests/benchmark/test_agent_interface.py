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
import subprocess
from pathlib import Path

from benchmark.agent_interface import (
    AgentExecutionConfig,
    AgentModeProfile,
    CursorAgentExecutionInterface,
)
from benchmark.schemas import load_task_spec


def _task():
    return load_task_spec(
        {
            "task_id": "task_cli_isolation",
            "task_type": "mmm_1d",
            "dataset_path": "data/unused.csv",
            "date_column": "date",
            "target_column": "y",
            "channel_columns": ["x1", "x2"],
            "cv": {"n_init": 10, "forecast_horizon": 2, "step_size": 1, "n_folds": 5},
        }
    )


def test_baseline_isolation_wraps_with_sandbox_exec(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, cwd, capture_output, text, timeout, check):
        captured["command"] = command
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout='{"status":"success"}', stderr=""
        )

    monkeypatch.setattr("benchmark.agent_interface.subprocess.run", fake_run)
    monkeypatch.setattr(
        "benchmark.agent_interface.shutil.which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )

    config = AgentExecutionConfig(
        command="cursor-agent",
        working_directory=str(tmp_path),
        enable_baseline_path_isolation=True,
        baseline_denied_paths=[str(tmp_path / ".cursor" / "skills")],
    )
    interface = CursorAgentExecutionInterface(config=config)

    result = interface.run_task(
        task=_task(),
        seed=1,
        profile=AgentModeProfile(name="baseline", mode_instruction="no skills"),
    )

    assert result.exit_code == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == "/usr/bin/sandbox-exec"
    assert command[1] == "-p"
    assert "deny file-read*" in command[2]
    assert "--print" in command
    assert "--output-format" in command
    assert "json" in command
    assert captured["cwd"] == Path(tmp_path)


def test_skilled_mode_does_not_use_sandbox_when_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, cwd, capture_output, text, timeout, check):
        captured["command"] = command
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout='{"status":"success"}', stderr=""
        )

    monkeypatch.setattr("benchmark.agent_interface.subprocess.run", fake_run)
    monkeypatch.setattr(
        "benchmark.agent_interface.shutil.which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )

    config = AgentExecutionConfig(
        command="cursor-agent",
        working_directory=str(tmp_path),
        enable_baseline_path_isolation=True,
    )
    interface = CursorAgentExecutionInterface(config=config)

    result = interface.run_task(
        task=_task(),
        seed=1,
        profile=AgentModeProfile(name="skilled", mode_instruction="use skills"),
    )

    assert result.exit_code == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == "cursor-agent"
    assert "--print" in command
    assert "--output-format" in command
    assert "json" in command
    assert captured["cwd"] == Path(tmp_path)


def test_baseline_isolation_fails_when_sandbox_exec_missing(
    monkeypatch, tmp_path: Path
) -> None:
    def should_not_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called without sandbox-exec")

    # Force unsupported isolation runtime.
    monkeypatch.setattr("benchmark.agent_interface.shutil.which", lambda _name: None)
    monkeypatch.setattr("benchmark.agent_interface.subprocess.run", should_not_run)

    config = AgentExecutionConfig(
        command="cursor-agent",
        working_directory=str(tmp_path),
        enable_baseline_path_isolation=True,
    )
    interface = CursorAgentExecutionInterface(config=config)

    result = interface.run_task(
        task=_task(),
        seed=1,
        profile=AgentModeProfile(name="baseline", mode_instruction="no skills"),
    )

    assert result.exit_code == 1
    assert "sandbox-exec" in result.stderr


def test_mode_specific_working_directories(monkeypatch, tmp_path: Path) -> None:
    cwd_values: list[Path] = []

    def fake_run(command, cwd, capture_output, text, timeout, check):
        cwd_values.append(Path(cwd))
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout='{"status":"success"}', stderr=""
        )

    monkeypatch.setattr("benchmark.agent_interface.subprocess.run", fake_run)

    baseline_cwd = tmp_path / "baseline"
    skilled_cwd = tmp_path / "skilled"
    baseline_cwd.mkdir(parents=True, exist_ok=True)
    skilled_cwd.mkdir(parents=True, exist_ok=True)

    config = AgentExecutionConfig(
        command="cursor-agent",
        working_directory=str(tmp_path),
        baseline_working_directory=str(baseline_cwd),
        skilled_working_directory=str(skilled_cwd),
    )
    interface = CursorAgentExecutionInterface(config=config)

    _ = interface.run_task(
        task=_task(),
        seed=1,
        profile=AgentModeProfile(name="baseline", mode_instruction="no skills"),
    )
    _ = interface.run_task(
        task=_task(),
        seed=1,
        profile=AgentModeProfile(name="skilled", mode_instruction="use skills"),
    )

    assert cwd_values == [baseline_cwd, skilled_cwd]


def test_timeout_returns_failure_execution_result(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, cwd, capture_output, text, timeout, check):
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    monkeypatch.setattr("benchmark.agent_interface.subprocess.run", fake_run)

    config = AgentExecutionConfig(
        command="cursor-agent",
        working_directory=str(tmp_path),
        timeout_sec=1,
    )
    interface = CursorAgentExecutionInterface(config=config)

    result = interface.run_task(
        task=_task(),
        seed=1,
        profile=AgentModeProfile(name="baseline", mode_instruction="no skills"),
    )

    assert result.exit_code == 124
    assert "timed out" in result.stderr.lower()


def test_logs_sampling_started(monkeypatch, tmp_path: Path, caplog) -> None:
    def fake_run(command, cwd, capture_output, text, timeout, check):
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout='{"status":"success"}', stderr=""
        )

    monkeypatch.setattr("benchmark.agent_interface.subprocess.run", fake_run)
    caplog.set_level("INFO", logger="benchmark.agent_interface")

    config = AgentExecutionConfig(
        command="cursor-agent",
        working_directory=str(tmp_path),
        timeout_sec=5,
    )
    interface = CursorAgentExecutionInterface(config=config)
    _ = interface.run_task(
        task=_task(),
        seed=7,
        profile=AgentModeProfile(name="baseline", mode_instruction="no skills"),
    )

    assert "sampling_started" in caplog.text


def test_streaming_mode_captures_events_and_result(tmp_path: Path) -> None:
    config = AgentExecutionConfig(
        command="cursor-agent",
        working_directory=str(tmp_path),
        timeout_sec=10,
        debug_stream=True,
    )
    interface = CursorAgentExecutionInterface(config=config)
    command = interface._build_command("demo prompt")

    assert "--output-format" in command
    assert "stream-json" in command

    result = interface._run_task_streaming(
        command=[
            "python",
            "-c",
            (
                "import json; "
                "print(json.dumps({'type':'system','subtype':'init'})); "
                'print(json.dumps({\'type\':\'result\',\'subtype\':\'success\',\'result\':\'{"status":"success","metrics":{},"sample_stats_diverging":[],"fold_metrics":[],"parameter_estimates":{},"roas_estimates":{}}\'}))'
            ),
        ],
        cwd=tmp_path,
        task=_task(),
        seed=1,
        profile=AgentModeProfile(name="baseline", mode_instruction="no skills"),
    )

    assert result.exit_code == 0
    assert result.stream_events
    assert '"status":"success"' in result.stdout
