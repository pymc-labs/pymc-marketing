# MMM Agent Benchmarks

## Objective

This benchmark suite measures whether a **skill-enabled agent** performs better than a **baseline agent** on core MMM workflows in `pymc-marketing`.

The benchmark is designed to answer:

1. Does skill access improve model quality and reliability?
2. Does skill access improve parameter recovery and ROAS estimation when ground truth is known?
3. Does skill access improve calibration-aware decisions under confounding?

## Non-objectives

- Replacing notebook examples as canonical MMM references.
- Defining new MMM model classes.
- Enforcing one universal scoring function for all future tasks.

## Benchmark Tasks

Task specs live in `benchmark/tasks/`.

1. **`mmm_case_study_1d`**
   - Source: `docs/source/notebooks/mmm/mmm_case_study.ipynb`
   - Data: `https://raw.githubusercontent.com/sibylhe/mmm_stan/main/data.csv`
   - Focus: convergence, out-of-sample CRPS, parameter stability.

2. **`mmm_multidimensional_recovery`**
   - Source: `docs/source/notebooks/mmm/mmm_multidimensional_example.ipynb`
   - Data: `data/mmm_multidimensional_example.csv`
   - Ground truth reference: `data/mmm_multidimensional_example_true_parameters.nc`
   - Focus: parameter recovery + ROAS estimation error + CV behavior.

3. **`mmm_roas_confounding_calibration`**
   - Source: `docs/source/notebooks/mmm/mmm_roas.ipynb`
   - Data: `data/mmm_roas_data.csv`
   - Focus: confounding-aware modeling, calibration-aware decisions, parameter/ROAS recovery.

## Pinned Runtime Policy

All benchmarked MMM fits must use the same sampler policy:

- Environment: `micromamba activate pymc-marketing-dev`
- Sampler: `nuts_sampler="nutpie"`
- Chains: `14`
- Cores: `14`
- Draws: `500`

These settings are encoded in each task spec and written into run records.

## Core Components

- `benchmark/schemas.py`
  - Validates task specs.
  - Enforces `n_folds >= 5` for time-slice CV.
  - Applies default sampler settings.

- `benchmark/backends.py`
  - `BaselineVsSkilledBackend`: mode-aware backend wrapper.
  - `AgentInterfaceBackend`: executes baseline/skilled profiles through a real agent CLI interface.
  - `DummyDeterministicBackend`: deterministic local harness backend for fast validation.

- `benchmark/agent_interface.py`
  - `ClaudeCliExecutionInterface`: command-line adapter for agent execution.
  - `AgentModeProfile`: mode-specific behavior policy (baseline vs skilled).
  - Structured prompt builder with pinned sampler and CV requirements.
  - JSON payload parser for agent outputs.

- `benchmark/runner.py`
  - Orchestrates `(task, mode, seed)` runs.
  - Persists model artifacts via `mmm.save(...)` when backend returns a model object.
  - Exports CSV/JSONL outputs and paired deltas.
  - Computes convergence and fold-validity pass flags.

- `benchmark/scoring.py`
  - Convergence diagnostics (divergence count).
  - Parameter/ROAS recovery diagnostics (MAE, RMSE, median AE, max AE, key coverage).
  - CV CRPS aggregation and CV parameter stability statistics.
  - Paired deltas for all `metric_*` fields.

- `benchmark/ground_truth.py`
  - Task B parameter recovery extraction from `mmm_multidimensional_example_true_parameters.nc`.
  - Task B ROAS proxy extraction from multidimensional data + true adstock/saturation parameters.
  - Task C true ROAS extraction using notebook equations:
    - `true_roas_x1 = sum(y - y01) / sum(x1)`
    - `true_roas_x2 = sum(y - y02) / sum(x2)`
  - Task C parameter extraction from generated columns:
    - `beta_channel ~= median(channel_effect / channel_adstock_saturated)`

- `benchmark/report.ipynb`
  - Loads benchmark CSV outputs.
  - Produces per-task and aggregate comparison plots/tables.

## Evaluation Logic

### 1) Run generation

For each task and seed:

- Run baseline mode.
- Run skill-enabled mode.
- Keep same seed and task config for paired comparability.

### 2) Convergence gate

Each run records:

- `convergence_divergence_count`
- `pass_no_divergence` (must be true for strict pass)

### 3) Time-slice cross validation

- Minimum 5 folds per task (validated by schema).
- Fold-level CRPS is aggregated to:
  - `metric_crps_cv_mean`
  - `metric_crps_cv_std`
  - `metric_cv_n_folds`

### 4) Task-specific scoring

- Task A: fit quality + convergence + CV behavior.
- Tasks B/C:
  - parameter recovery (MAE, RMSE, median AE, max AE)
  - ROAS recovery (MAE, RMSE, median AE, max AE)
- Task C additionally tracks calibration-aware decision quality (to be provided by backend rubric output).

### 5) Ground-truth extraction details for Tasks B/C

- **Task B (multidimensional)**
  - Parameter truth is loaded from `data/mmm_multidimensional_example_true_parameters.nc`.
  - ROAS truth is computed per channel by:
    1. applying geometric adstock (`l_max=8`) with true `adstock_alpha`,
    2. applying logistic saturation with true `saturation_lam`,
    3. scaling by true `saturation_beta`,
    4. dividing total channel contribution by total channel spend.

- **Task C (ROAS confounding)**
  - True ROAS follows notebook equations (`mmm_roas.ipynb`):
    - `(y - y01) / x1` aggregated over the full period for `x1`,
    - `(y - y02) / x2` aggregated over the full period for `x2`.
  - True channel effect coefficients are estimated directly from generated columns:
    - `x*_effect / x*_adstock_saturated` (robust median).

### 6) Paired uplift

Runs are paired by `(task_id, seed)`, and the benchmark computes:

- `delta_<metric> = skilled - baseline`

For error metrics (e.g., CRPS), negative deltas indicate improvement.

## Output Files

Outputs are written under the selected output directory (default `benchmark/results/latest/`):

- `run_results.csv`
  - One row per `(task, mode, seed)` run.
- `paired_deltas.csv`
  - One row per `(task, seed)` pair with `skilled - baseline` deltas.
- `task_summary.csv`
  - Aggregated metrics by `(task_id, mode)`.
- `benchmark_summary.csv`
  - Overall metric summary by mode.
- `run_results.jsonl`
  - Machine-readable per-run records.
- `artifacts/<task>/<mode>/seed_<seed>/`
  - `model.nc` (when `model.save()` available)
  - `fold_metrics.json`
  - `parameter_recovery_details.json`
  - `roas_recovery_details.json`
  - `cv_parameter_stability.json`
  - `fit_difference_context.json`

## Local Execution

### Step 1: activate environment

```bash
micromamba activate pymc-marketing-dev
```

### Step 2: run benchmark

From repository root:

```bash
python -m benchmark.cli --tasks-dir benchmark/tasks --output-dir benchmark/results/latest --seeds 42 314
```

This runs baseline and skilled modes for all task specs using the configured backend (default local dummy backend).

To run with the actual agent interface:

```bash
python -m benchmark.cli \
  --backend agent-cli \
  --agent-command claude \
  --agent-max-turns 10 \
  --agent-timeout-sec 900 \
  --tasks-dir benchmark/tasks \
  --output-dir benchmark/results/latest \
  --seeds 42
```

### Step 3: inspect outputs

```bash
ls benchmark/results/latest
```

Expected files:

- `run_results.csv`
- `paired_deltas.csv`
- `task_summary.csv`
- `benchmark_summary.csv`
- `run_results.jsonl`
- `artifacts/`

### Step 4: open report notebook

Open `benchmark/report.ipynb` and execute cells to generate:

- CRPS by task and mode
- Parameter recovery comparisons for Tasks 2/3
- CV parameter stability comparisons
- Fit-difference decomposition and metric-driven explanations
- Aggregate summary tables

## Agent output contract

When using `AgentInterfaceBackend`, each agent run must return JSON with:

- `status` (e.g., `"success"`)
- `metrics` (dict of scalar metrics)
- `sample_stats_diverging` (list/array of divergence indicators)
- `fold_metrics` (list of dicts; include `crps`)
- `parameter_estimates` (dict)
- `roas_estimates` (dict)
- `cv_parameter_estimates` (optional fold-level parameter estimates)
- `cv_fold_diagnostics` (optional fold-level diagnostics)
- `fit_diagnostics` (optional run-level diagnostics)
- `model` (optional fitted MMM object exposing `.save(...)`)

The CLI backend currently parses either plain JSON stdout or fenced ` ```json ... ``` ` output.

The parsed JSON is then validated against a strict Pydantic schema (`AgentBackendPayload`):

- unknown keys are rejected (`extra="forbid"`),
- required keys (including `status`) must be present,
- value types must match expected schema (for example, metric values must be numeric).

On validation failure, the run is marked as `status="failure"` and structured details are emitted via:

- `BenchmarkResult.payload_validation_errors`
- `run_results.csv` column `payload_validation_error_count`

## Troubleshooting

- **`ModuleNotFoundError: benchmark`**
  - Run commands from repository root so `benchmark/` is importable.

- **No `model.nc` artifacts**
  - Backend must return a `model` object with `save(path)` implemented.

- **Agent backend returns failure with empty metrics**
  - Ensure the agent stdout is valid JSON and includes required keys listed in the output contract.

- **Fold count gate failing**
  - Ensure each task emits at least 5 CV folds and each fold has CRPS data.

- **Divergence gate failing**
  - Check sampler diagnostics from `sample_stats_diverging`; tune priors/model specification as needed.

- **Inconsistent paired deltas**
  - Verify baseline and skilled runs share exactly the same task spec and seed.

- **Parameter recovery still NaN**
  - Check `parameter_recovery_details.json` for key-overlap counts and verify estimate/truth naming alignment.
