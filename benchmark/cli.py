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
"""CLI helpers for running MMM benchmarks locally."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from rich.logging import RichHandler

from benchmark.agent_interface import (
    AgentExecutionConfig,
    CursorAgentExecutionInterface,
)
from benchmark.backends import AgentInterfaceBackend
from benchmark.runner import BenchmarkRunner
from benchmark.schemas import load_task_specs_from_directory

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
RICH_LOG_FORMAT = "%(message)s"


def configure_logging(log_level: str, log_file: str | None) -> None:
    """Configure benchmark logging to stdout and optional file."""
    root_logger = logging.getLogger("benchmark")
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.handlers.clear()
    root_logger.propagate = False

    stream_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
    )
    stream_handler.setFormatter(logging.Formatter(RICH_LOG_FORMAT))
    root_logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(file_handler)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for benchmark execution options."""
    parser = argparse.ArgumentParser(description="Run MMM benchmark suite.")
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="benchmark/tasks",
        help="Directory containing task YAML specs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/results/latest",
        help="Directory where benchmark outputs are written.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds used for paired baseline/skilled runs.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["agent-cli"],
        default="agent-cli",
        help="Backend to execute benchmark runs.",
    )
    parser.add_argument(
        "--agent-command",
        type=str,
        default="cursor-agent",
        help="CLI command used when backend=agent-cli.",
    )
    parser.add_argument(
        "--agent-max-turns",
        type=int,
        default=10,
        help="Maximum turns for agent CLI calls when backend=agent-cli.",
    )
    parser.add_argument(
        "--agent-timeout-sec",
        type=int,
        default=900,
        help="Timeout in seconds per agent run when backend=agent-cli.",
    )
    parser.add_argument(
        "--agent-debug-stream",
        action="store_true",
        help=(
            "Enable Cursor stream-json mode for live event logs and "
            "persisted agent_events.jsonl artifacts."
        ),
    )
    parser.add_argument(
        "--baseline-working-dir",
        type=str,
        default=None,
        help="Optional working directory override for baseline mode.",
    )
    parser.add_argument(
        "--skilled-working-dir",
        type=str,
        default=None,
        help="Optional working directory override for skilled mode.",
    )
    parser.add_argument(
        "--enable-baseline-path-isolation",
        action="store_true",
        help="Enable runtime path isolation for baseline runs.",
    )
    parser.add_argument(
        "--baseline-denied-path",
        action="append",
        default=[],
        help=(
            "Path to deny for baseline runs when isolation is enabled. "
            "Repeat flag to deny multiple paths."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for benchmark runtime logs.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional file path for benchmark logs.",
    )
    return parser


def main() -> None:
    """Run benchmark workflow from CLI arguments."""
    args = build_parser().parse_args()
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    logger = logging.getLogger("benchmark.cli")

    tasks = load_task_specs_from_directory(args.tasks_dir)
    output_dir = Path(args.output_dir)
    logger.info(
        "benchmark_start tasks_dir=%s output_dir=%s seeds=%s agent_command=%s timeout_sec=%s debug_stream=%s",
        args.tasks_dir,
        args.output_dir,
        args.seeds,
        args.agent_command,
        args.agent_timeout_sec,
        args.agent_debug_stream,
    )
    if shutil.which(args.agent_command) is None:
        raise RuntimeError(
            f"Agent command '{args.agent_command}' is not available on PATH."
        )

    agent_config = AgentExecutionConfig(
        command=args.agent_command,
        max_turns=args.agent_max_turns,
        timeout_sec=args.agent_timeout_sec,
        working_directory=".",
        baseline_working_directory=args.baseline_working_dir,
        skilled_working_directory=args.skilled_working_dir,
        enable_baseline_path_isolation=args.enable_baseline_path_isolation,
        baseline_denied_paths=list(args.baseline_denied_path),
        debug_stream=args.agent_debug_stream,
    )
    agent_interface = CursorAgentExecutionInterface(config=agent_config)
    backend = AgentInterfaceBackend(agent_interface=agent_interface)
    runner = BenchmarkRunner(backend=backend, output_dir=output_dir)
    output = runner.run(tasks=tasks, seeds=args.seeds, modes=["baseline", "skilled"])
    success_count = sum(
        1 for row in output.run_records if row.get("status") == "success"
    )
    failure_count = sum(
        1 for row in output.run_records if row.get("status") == "failure"
    )
    timeout_count = sum(
        1 for row in output.run_records if row.get("status") == "timeout"
    )
    logger.info(
        "benchmark_complete run_count=%s success=%s failure=%s timeout=%s output_dir=%s",
        len(output.run_records),
        success_count,
        failure_count,
        timeout_count,
        output_dir,
    )
    print(f"Benchmark finished. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
