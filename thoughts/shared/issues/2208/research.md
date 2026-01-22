---
date: 2026-01-22T00:00:00Z
researcher: Claude Sonnet 4.5
git_commit: 00563ba0791e7151535fb2db6bf99f3c356ab2d0
branch: work-issue-2208
repository: pymc-marketing/pymc-marketing
topic: "Experimental warnings appearing when importing pymc_marketing.mmm"
tags: [research, codebase, mmm, warnings, experimental, builders]
status: complete
last_updated: 2026-01-22
last_updated_by: Claude Sonnet 4.5
issue_number: 2208
---

# Research: Experimental warnings appearing when importing pymc_marketing.mmm

**Date**: 2026-01-22T00:00:00Z
**Researcher**: Claude Sonnet 4.5
**Git Commit**: 00563ba0791e7151535fb2db6bf99f3c356ab2d0
**Branch**: work-issue-2208
**Repository**: pymc-marketing/pymc-marketing
**Issue**: #2208

## Research Question

Why does importing `pymc_marketing.mmm` trigger experimental warnings, and how should internal usage (like in `time_slice_cross_validation`) avoid showing these warnings to end users?

## Summary

The issue stems from the fact that **`pymc_marketing.mmm.builders` issues a module-level experimental warning at import time** (pymc_marketing/mmm/builders/__init__.py:23-27). When internal code like `TimeSliceCrossValidator` imports from `builders.yaml` (time_slice_cross_validation.py:32), Python executes the builders module's `__init__.py`, which triggers the warning.

Currently, there is NO mechanism to suppress this warning for internal usage. The warning is issued with `stacklevel=2`, meaning it will appear to come from the immediate caller. When a user does `import pymc_marketing.mmm` or uses `TimeSliceCrossValidator`, they see warnings about experimental APIs even though they're using stable, public functionality.

**Key Finding**: The `builders` module is NOT exposed in `pymc_marketing.mmm.__all__`, which suggests it's intended to be internal/experimental. However, the module-level warning in `builders/__init__.py` fires whenever ANY code imports from it, including internal pymc_marketing code.

## Detailed Findings

### Module-Level Experimental Warnings

Three modules issue warnings at import time (when the module's `__init__.py` is executed):

1. **`pymc_marketing/mmm/builders/__init__.py:23-27`** - The problematic one
   ```python
   warnings.warn(
       "The pymc_marketing.mmm.builders module is experimental and its API may change without warning.",
       UserWarning,
       stacklevel=2,
   )
   ```
   - Warning type: `UserWarning`
   - Fires when: The module is imported (either directly or indirectly)
   - Stacklevel: 2 (appears to come from the caller)

2. **`pymc_marketing/mmm/multidimensional.py:213-218`**
   ```python
   warning_msg = (
       "This functionality is experimental and subject to change. "
       "If you encounter any issues or have suggestions, please raise them at: "
       f"{PYMC_MARKETING_ISSUE}"
   )
   warnings.warn(warning_msg, FutureWarning, stacklevel=1)
   ```
   - Warning type: `FutureWarning`
   - Stacklevel: 1

3. **`pymc_marketing/mlflow.py:181-186`**
   - Same pattern as multidimensional.py
   - Warning type: `FutureWarning`
   - Stacklevel: 1

### Internal Usage of `builders`

**`pymc_marketing/mmm/time_slice_cross_validation.py`** is the key internal consumer:

- **Line 32**: `from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml`
  - This import triggers the warning in `builders/__init__.py` because importing `builders.yaml` requires loading the `builders` package first

- **Line 625**: Uses `build_mmm_from_yaml()` to optionally rebuild models from YAML during cross-validation
  ```python
  fold_mmm = build_mmm_from_yaml(config_path=yaml_path, X=X_train, y=y_train)
  ```

This means **any user who uses `TimeSliceCrossValidator` with a YAML config will see the experimental builders warning**, even though:
- `TimeSliceCrossValidator` is exposed in `pymc_marketing.mmm.__all__` (line 107)
- `TimeSliceCrossValidator` is part of the stable public API
- Users have no direct interaction with the `builders` module

### MMM Module Structure

**`pymc_marketing/mmm/__init__.py`** exports 51 items but notably:
- **DOES export**: `TimeSliceCrossValidator`, `TimeSliceCrossValidationResult`
- **DOES NOT export**: `builders` (not in `__all__`)
- **DOES NOT import**: `builders` module at the top level

This means:
- `builders` is accessible as `pymc_marketing.mmm.builders` (Python allows this)
- But it's not "officially" exposed (not in `__all__`)
- The warning suggests it's experimental and internal

However, the import chain shows:
```
User imports: pymc_marketing.mmm
  → imports: TimeSliceCrossValidator (exposed publicly)
    → imports: from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml
      → triggers: builders/__init__.py execution
        → warns: "experimental and its API may change without warning"
```

### Other Experimental Warnings

For completeness, there are also class-level experimental warnings:

1. **`pymc_marketing/special_priors.py:509-512`** - `MaskedPrior.__init__`
   - Issues warning when the class is instantiated (not at import time)

2. **`pymc_marketing/mmm/causal.py:693-696`** - `TBFPC.__init__`
   - Issues warning when the class is instantiated
   - Has a test: `tests/mmm/test_causal.py:850-852`

These are less problematic because they only warn when users explicitly use those classes.

## Code References

- `pymc_marketing/mmm/builders/__init__.py:23-27` - The experimental warning that fires on import
- `pymc_marketing/mmm/time_slice_cross_validation.py:32` - Internal import that triggers the warning
- `pymc_marketing/mmm/time_slice_cross_validation.py:625` - Where `build_mmm_from_yaml` is used internally
- `pymc_marketing/mmm/__init__.py:67-70` - `TimeSliceCrossValidator` publicly exposed
- `pymc_marketing/mmm/__init__.py:74-127` - `__all__` list (builders not included)

## Architecture Insights

### Warning System Design

The codebase uses standard Python `warnings.warn()` - there's no custom warning framework or suppression mechanism.

**Warning Types Used**:
- `FutureWarning` - For module-level warnings about experimental modules
- `UserWarning` - For class-level warnings and the builders module

**Stacklevel Strategy**:
- Module-level warnings use `stacklevel=2` to point to the importer
- Class-level warnings use `stacklevel=2` to point to the instantiator

### Import Structure Problem

The issue is a classic Python import side-effect problem:

1. **Module-level code executes on first import** - When Python imports a module for the first time, it executes all top-level code in `__init__.py`
2. **Transitive imports trigger this too** - Even if you import `builders.yaml`, Python loads `builders/__init__.py` first
3. **No way to distinguish internal vs external callers** - The warning can't tell if it's being imported by internal code or user code

### Possible Solutions (Not Implemented)

Several approaches could fix this:

1. **Remove the warning entirely** - If builders is only used internally, remove the warning
2. **Move warning to public functions** - Only warn when `build()` or `build_mmm_from_yaml()` are called
3. **Context manager for suppression** - Internal code could suppress warnings explicitly
4. **Separate internal/public entry points** - Have internal code import from a private module that doesn't warn
5. **Lazy imports** - Import `build_mmm_from_yaml` only when needed (inside the function that uses it)

## Related Components

### TimeSliceCrossValidator Usage Pattern

The validator has two modes for building models:
1. **YAML config** (uses builders): `cv.run(X, y, yaml_path="config.yml")`
2. **Model builder object** (doesn't use builders): `cv.run(X, y, mmm=mmm_builder)`

Only the YAML path triggers the builders warning because it imports `build_mmm_from_yaml`.

### Builders Module Structure

```
pymc_marketing/mmm/builders/
├── __init__.py          # Issues the warning, exports build & build_mmm_from_yaml
├── factories.py         # Core factory function: build()
└── yaml.py              # YAML functions: build_mmm_from_yaml()
```

The module provides:
- `build()` - Generic recursive factory for building objects from specs
- `build_mmm_from_yaml()` - Specialized function for MMM models from YAML

## Open Questions

1. **Is builders truly experimental, or is it stable enough for internal use?**
   - If experimental: Should `TimeSliceCrossValidator` use it?
   - If stable for internal use: Should the warning be removed or moved?

2. **Should TimeSliceCrossValidator issue its own experimental warning when using YAML configs?**
   - The issue description asks: "unless they are also deemed experimental, in which case they should issue their own warning"
   - Currently `TimeSliceCrossValidator` is in `__all__` with no experimental warning

3. **What is the intended public API for builders?**
   - Not in `__all__`, suggesting it's not public
   - But accessible as `pymc_marketing.mmm.builders`
   - Used by public-facing `TimeSliceCrossValidator`

4. **Are there other internal usages of builders that would trigger warnings?**
   - Current research only found `time_slice_cross_validation.py`
   - Need to verify no other internal consumers exist

## Recommendations

Based on this analysis, the maintainers need to decide:

1. **Option A: Remove builders warning entirely**
   - Delete lines 23-27 from `pymc_marketing/mmm/builders/__init__.py`
   - Rationale: If it's used by public API (`TimeSliceCrossValidator`), it can't be that experimental

2. **Option B: Move warning to function call sites**
   - Remove module-level warning
   - Add warnings to `build()` and `build_mmm_from_yaml()` functions
   - Internal code can suppress with `warnings.catch_warnings()`

3. **Option C: Make TimeSliceCrossValidator explicitly experimental**
   - Keep builders warning
   - Add experimental warning to `TimeSliceCrossValidator.__init__`
   - Remove from `mmm.__all__` or document as experimental

4. **Option D: Lazy import in TimeSliceCrossValidator**
   - Move the import from module level (line 32) to inside the `run()` method where it's used
   - Wrap in warning suppression for internal usage
   - User-facing code that directly imports builders still gets warned

The comment by @juanitorduz ("We must remove these indeed") suggests **Option A** is the intended path.
