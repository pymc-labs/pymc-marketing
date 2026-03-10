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
from pathlib import Path

from benchmark.agent_interface import AgentExecutionConfig, ClaudeCliExecutionInterface
from benchmark.backends import AgentInterfaceBackend, DummyDeterministicBackend
from benchmark.runner import BenchmarkRunner
from benchmark.schemas import load_task_specs_from_directory


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
        choices=["dummy", "agent-cli"],
        default="dummy",
        help="Backend to execute benchmark runs.",
    )
    parser.add_argument(
        "--agent-command",
        type=str,
        default="claude",
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
    return parser


def main() -> None:
    """Run benchmark workflow from CLI arguments."""
    args = build_parser().parse_args()
    tasks = load_task_specs_from_directory(args.tasks_dir)
    output_dir = Path(args.output_dir)
    if args.backend == "agent-cli":
        agent_config = AgentExecutionConfig(
            command=args.agent_command,
            max_turns=args.agent_max_turns,
            timeout_sec=args.agent_timeout_sec,
            working_directory=".",
        )
        agent_interface = ClaudeCliExecutionInterface(config=agent_config)
        backend = AgentInterfaceBackend(agent_interface=agent_interface)
    else:
        backend = DummyDeterministicBackend()
    runner = BenchmarkRunner(backend=backend, output_dir=output_dir)
    runner.run(tasks=tasks, seeds=args.seeds, modes=["baseline", "skilled"])
    print(f"Benchmark finished. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
