# Multiplicative MMM Hardening Report

This report documents a principal-level hardening plan for multiplicative MMM behavior in PyMC-Marketing, with emphasis on:

- connectivity between graph-time and post-fit decomposition logic,
- maintainability and modular ownership boundaries,
- long-term sustainability and safe extension.

Scope excludes notebooks and focuses on code paths compared to `main`.

## Report A: Safety Contracts

### A1. Contribution API contract matrix

| API | Link | Scale | Guaranteed keys | Optional keys | Notes |
|---|---|---|---|---|---|
| `MMMIDataWrapper.get_contributions()` | `identity` | `original_scale=True` | `channels` | `baseline`, `controls`, `seasonality` | Standard additive decomposition semantics. |
| `MMMIDataWrapper.get_contributions()` | `identity` | `original_scale=False` | `channels` | `baseline`, `controls`, `seasonality` | Returns scaled-space contributions. |
| `MMMIDataWrapper.get_contributions()` | `log` | `original_scale=True` | `channels` | `baseline` | `controls` and `seasonality` are embedded in `baseline` for this path. |
| `MMMIDataWrapper.get_contributions()` | `log` | `original_scale=False` | falls back to identity path behavior | same as identity | Behavior remains shape-compatible but interpretability differs from log original-scale path. |
| `MMM.compute_counterfactual_contributions_dataset()` | `identity` | original scale | one variable per component | n/a | Full posterior dims retained. |
| `MMM.compute_counterfactual_contributions_dataset()` | `log` | original scale | one variable per component | n/a | Counterfactual components are per-draw and non-additive in aggregate. |
| `MMM.compute_mean_contributions_over_time()` | `identity` | original scale | tabular component columns | n/a | Mean-first summary optimization path. |
| `MMM.compute_mean_contributions_over_time()` | `log` | original scale | tabular component columns | n/a | Uses dataset method then averages over `(chain, draw)`. |

### A2. Strictness policy

Current behavior uses warnings for unsupported toggle semantics in log-link wrapper decomposition. To make production behavior robust:

- Keep warning mode as backward-compatible default.
- Add optional strict mode (future) to raise errors for unsupported argument combinations.
- Keep docs explicit that controls/seasonality are not independently returned in log-link original-scale wrapper path.

### A3. Compatibility policy

- Treat serialized payloads using legacy scaling type keys as compatibility inputs.
- Preserve migration behavior in deserialization without weakening validation of malformed payloads.

## Report B: Math Unification and Parity

### B1. Single source of truth map

Core formulas should be centralized in shared helpers and reused by all runtime surfaces:

- `pymc_marketing/mmm/decomposition.py`
  - `original_scale_prediction_from_mu`
  - `log_counterfactual_remove_component`
  - `identity_counterfactual_component`
  - `safe_proportional_share`

Consumers:

- `pymc_marketing/mmm/multidimensional.py`
  - `compute_counterfactual_contributions_dataset`
  - `compute_mean_contributions_over_time`
- `pymc_marketing/data/idata/mmm_wrapper.py`
  - `_get_contributions_log_link`

### B2. Parity assertions to enforce

1. Wrapper conservation identity in log-link path:
   - `channels.sum("channel") + baseline == exp(mu) * target_scale` per draw.
2. Baseline formula identity:
   - `baseline == exp(mu - media_total_log) * target_scale`.
3. Mean/DataFrame parity:
   - `compute_counterfactual_contributions_dataset().mean(("chain","draw"))` equals `compute_mean_contributions_over_time()` (component columns).
4. Finite share behavior near zero denominators:
   - no `NaN` or `inf` in channel shares/contributions from proportional split.

### B3. Drift prevention

- Add cross-reference comments/docstrings where formulas are used.
- Keep formula changes gated behind tests in wrapper and multidimensional suites.

## Report C: Modularity Blueprint

### C1. Immediate boundaries (already low risk)

- Keep decomposition math in `pymc_marketing/mmm/decomposition.py`.
- Keep orchestrator classes (`MMM`, `MMMIDataWrapper`) thin around shared math/utilities.

### C2. Next extraction targets

1. `multidimensional.py`
   - isolate decomposition assembly helpers,
   - isolate scaling-resolution utilities,
   - preserve public API at `MMM`.
2. `scaling.py`
   - keep domain models (`VariableScaling`, `DataDerivedScaling`, `FixedScaling`) together,
   - split serialization payload adapters and geometry/dim helpers into internal modules,
   - re-export stable symbols from `scaling.py` to avoid external import breakage.

### C3. Rollout strategy

- Refactor in compatibility-preserving increments:
  1. extract internals,
  2. rewire callsites,
  3. keep imports stable,
  4. enforce parity tests after each extraction.

## Report D: Test Sustainability Backlog

### D1. Atomic invariant suite (highest value)

- Pure decomposition helper tests (formula-level, deterministic tensors).
- Explicit finite-output tests for near-zero denominator scenarios.
- Dataset/DataFrame parity tests across both link modes.

### D2. Integration invariants

- `multidimensional + log + scaling` interaction tests:
  - idata attrs integrity,
  - dimensional alignment,
  - decomposition output schema stability.

### D3. Brittle test retirement

Replace threshold heuristics and weak negative assertions with deterministic algebraic checks, especially for log-link baseline and non-additivity behavior.

### D4. Layer ownership cleanup

Align test headers and documented test layers with actual implemented tests to avoid stale quality signals.

## Suggested Execution Sequence

1. Contract and strictness hardening.
2. Math parity enforcement and helper reuse.
3. Module extraction with compatibility shims.
4. Test-invariant expansion and brittle-test retirement.

## Success Criteria

- No duplicated multiplicative formulas outside shared decomposition helpers.
- Explicit and test-enforced link-mode contracts for contribution APIs.
- Stable modular boundaries with lower regression risk in high-change files.
- Deterministic parity between posterior dataset and summary DataFrame outputs.
