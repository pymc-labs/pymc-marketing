---
date: 2026-02-10T10:02:51+0000
researcher: Claude Sonnet 4.5
git_commit: a6aea1f804f25e58c504a50081349c37364ee671
branch: work-issue-2275
repository: pymc-labs/pymc-marketing
topic: "Tqdm not in env - CI failure investigation"
tags: [research, codebase, dependencies, ci, backcompat, tqdm]
status: complete
last_updated: 2026-02-10
last_updated_by: Claude Sonnet 4.5
issue_number: 2275
---

# Research: Tqdm not in env - CI failure investigation

**Date**: 2026-02-10T10:02:51+0000
**Researcher**: Claude Sonnet 4.5
**Git Commit**: a6aea1f804f25e58c504a50081349c37364ee671
**Branch**: work-issue-2275
**Repository**: pymc-labs/pymc-marketing
**Issue**: #2275

## Research Question

The CI job "Backcompat matrix (basic_mmm)" is failing with what appears to be a tqdm import error. The question is: Is tqdm missing from the environment.yml or is there another configuration issue?

Reference: https://github.com/pymc-labs/pymc-marketing/actions/runs/21843841175/job/63034844604#step:4:1

## Summary

**Key Finding: tqdm IS properly declared in pyproject.toml but is MISSING from environment.yml**

The backcompat workflow (`.github/workflows/backcompat.yml`) exists on the `backwards-compat` branch but not on `main`. This workflow uses `environment.yml` to set up the conda environment via micromamba. However, `environment.yml` does not include tqdm as a dependency, while the codebase uses tqdm in `pymc_marketing/mmm/time_slice_cross_validation.py:30`.

The discrepancy exists because:
1. **pyproject.toml** (line 39) correctly lists `"tqdm"` in the core dependencies
2. **environment.yml** does NOT include tqdm in its dependency list
3. The backcompat workflow uses conda/micromamba with environment.yml, not pip with pyproject.toml

## Detailed Findings

### Dependency Configuration

#### pyproject.toml - HAS tqdm
Location: `pyproject.toml:39`

```python
dependencies = [
    "arviz>=0.13.0",
    "matplotlib>=3.5.1",
    "narwhals",
    "numpy>=2.0",
    "pandas",
    "pydantic>=2.1.0",
    "pymc>=5.27.0",
    "pytensor>=2.36.3",
    "scikit-learn>=1.1.1",
    "seaborn>=0.12.2",
    "tqdm",  # <-- Line 39: tqdm is declared here
    "xarray>=2024.1.0",
    "xarray-einstats>=0.5.1",
    "pyprojroot",
    "pymc-extras>=0.4.0",
    "preliz>=0.20.0",
    "pyyaml",
]
```

#### environment.yml - MISSING tqdm
Location: `environment.yml:1-69`

The file contains 69 lines of dependencies including many packages like:
- Python 3.12
- arviz, matplotlib, numpy, scipy, pandas
- pymc>=5.23.0, pytensor>=2.31.3
- Many testing and documentation packages

**BUT tqdm is NOT listed anywhere in this file.**

#### environment-dev.yml
Location: `scripts/docker/environment-dev.yml`

This is a minimal file that just installs `pymc-marketing` from conda-forge, so it would inherit dependencies from the published package.

### Tqdm Usage in Codebase

**Primary Usage**: `pymc_marketing/mmm/time_slice_cross_validation.py:30`

```python
from tqdm.auto import tqdm
```

This import is used in the `TimeSliceCrossValidator.run()` method at line 619:
```python
for _i, (train_idx, test_idx) in enumerate(tqdm(self.split(X, y))):
```

The tqdm library provides progress bars for the cross-validation iterations.

### CI Workflow Analysis

#### Backcompat Workflow
Location: `.github/workflows/backcompat.yml` (exists on `backwards-compat` branch, commit d6cc95e5)

**Key Details:**
- Job name: "Backcompat matrix"
- Tests multiple models including: `basic_mmm`, `beta_geo`, `beta_geo_beta_binom`, `gamma_gamma`, etc.
- Uses `mamba-org/setup-micromamba@v1` with `environment-file: environment.yml`
- The workflow file was added in commit fc273b6e and last modified in d6cc95e5

**Critical Section:**
```yaml
- name: Set up micromamba
  uses: mamba-org/setup-micromamba@v1
  with:
    micromamba-version: latest
    environment-file: environment.yml  # <-- Uses environment.yml, NOT pyproject.toml
    environment-name: pymc-marketing-dev
    cache-environment: true
    cache-downloads: true
```

The workflow then runs:
```bash
micromamba run -n pymc-marketing-dev python -m scripts.backcompat.capture ${{ matrix.model }} ...
```

When the backcompat scripts eventually import code from `pymc_marketing.mmm`, it will try to import tqdm, which will fail because environment.yml doesn't include it.

#### Test Workflow (for comparison)
Location: `.github/workflows/test.yml`

This workflow uses pip installation:
```yaml
- name: Run tests
  run: |
    sudo apt-get install graphviz graphviz-dev
    pip install -e .[test]  # <-- Uses pyproject.toml via pip
    pytest ${{ matrix.split }}
```

This works fine because pip reads from pyproject.toml which includes tqdm.

#### Test Notebook Workflow (for comparison)
Location: `.github/workflows/test_notebook.yml`

This workflow uses uv (modern pip replacement):
```yaml
- name: Install dependencies
  run: |
    sudo apt-get install graphviz graphviz-dev
    uv venv --python 3.12
    uv pip install --upgrade pip
    uv pip install -e ".[docs,test,dag]"  # <-- Also uses pyproject.toml
```

This also works because it reads from pyproject.toml.

## Code References

- `pyproject.toml:39` - tqdm declared in core dependencies
- `environment.yml:1-69` - Complete file, no tqdm entry
- `pymc_marketing/mmm/time_slice_cross_validation.py:30` - tqdm import
- `pymc_marketing/mmm/time_slice_cross_validation.py:619` - tqdm usage

## Architecture Insights

### Dual Dependency Management
The project maintains two parallel dependency specifications:
1. **pyproject.toml** - Modern Python packaging standard, used by pip/uv
2. **environment.yml** - Conda/Mamba environment specification

This dual maintenance creates opportunity for drift, which is exactly what happened with tqdm.

### Workflow Patterns
- **Modern workflows** (test.yml, test_notebook.yml): Use pip/uv with pyproject.toml
- **Conda-based workflows** (backcompat.yml): Use micromamba with environment.yml
- The backcompat workflow is the only one using environment.yml exclusively

### Version Synchronization Concern
Looking at pyproject.toml and environment.yml:
- pyproject.toml: `"pymc>=5.27.0"` (line 35)
- environment.yml: `- pymc>=5.23.0` (line 22)

There's a comment in environment.yml at line 21:
```yaml
# NOTE: Keep minimum pymc version in sync with ci.yml `OLDEST_PYMC_VERSION`
```

But there's also drift in the PyMC version requirement, suggesting environment.yml isn't being kept in sync with pyproject.toml.

## Historical Context

### Git History
The backcompat workflow and associated scripts were added in a series of commits on the `backwards-compat` branch:
- fc273b6e - Initial "some backward compat logic" commit adding workflow and scripts
- f675f1a3 - Fix to use latest micromamba version
- a9ea60e2 - Add environment caching
- 033ec832 - Cancel in-progress runs on new commits
- 6d803eb7 - Copy __init__.py files for imports
- 1a7ddb0e - Create backcompat directory and copy scripts
- d6cc95e5 - Handle pytensor 2.31+ compatibility

These files exist on `remotes/origin/backwards-compat` but not on `main` or the current `work-issue-2275` branch.

### Recent Main Branch Changes
Recent commits on main show active development:
- a6aea1f8 - Create notebook to display plot_interactive abilities (#2272)
- 8bde31af - Update UML Diagrams (#2265)
- 21675914 - Update version.py (#2276)

The tqdm import was added in the time_slice_cross_validation.py file which is part of the main codebase.

## Root Cause Analysis

The failure occurs because:

1. The `backwards-compat` branch has a backcompat workflow that uses `environment.yml`
2. The main branch (and this current branch) has code that imports tqdm
3. When the backcompat workflow runs, it:
   - Sets up environment from `environment.yml` (which lacks tqdm)
   - Checks out and runs code from main (which imports tqdm)
   - Results in ImportError: No module named 'tqdm'

The workflow is essentially testing old baselines from main against new code, but the environment setup doesn't include all dependencies needed by the new code.

## Solution Options

### Option 1: Add tqdm to environment.yml (Recommended)
Add `- tqdm` to the dependencies list in environment.yml. This ensures parity between conda and pip installation methods.

```yaml
dependencies:
# Base dependencies
- python=3.12
- arviz>=0.13.0
- matplotlib>=3.5.1
# ... other deps ...
- tqdm  # <-- Add this
- xarray
```

**Pros:**
- Simple, direct fix
- Maintains both installation methods
- Prevents future similar issues

**Cons:**
- Requires maintaining two dependency lists (ongoing maintenance burden)

### Option 2: Sync environment.yml with pyproject.toml
Create a systematic process to keep environment.yml synchronized with pyproject.toml, perhaps via a script or CI check.

**Pros:**
- Prevents future drift
- Catches all missing dependencies

**Cons:**
- More complex implementation
- Requires new tooling/process

### Option 3: Migrate backcompat workflow to use pip/uv
Change the backcompat workflow to use pip or uv installation instead of micromamba.

```yaml
- name: Set up Python
  uses: actions/setup-python@v6
  with:
    python-version: "3.12"
- run: pip install -e .[test]
```

**Pros:**
- Single source of truth for dependencies
- Aligns with other workflows

**Cons:**
- Changes the environment setup method
- May have implications for conda-specific packages

## Open Questions

1. **Why does backcompat use conda/micromamba instead of pip?**
   - Are there conda-specific dependencies required?
   - Was this a deliberate choice for reproducibility?

2. **Is environment.yml still actively maintained?**
   - The version drift (pymc 5.23.0 vs 5.27.0) suggests it may be falling behind
   - Should there be a deprecation plan?

3. **Are there other missing dependencies in environment.yml?**
   - A full audit comparing pyproject.toml to environment.yml would be valuable
   - Should include transitive dependencies

4. **What is the status of merging the backwards-compat branch?**
   - The backcompat workflow exists only on that branch
   - Is it planned to merge to main?
   - If so, this tqdm issue needs to be resolved first

## Recommended Action

**Immediate Fix:** Add `- tqdm` to `environment.yml` on both the main branch and the backwards-compat branch.

**Follow-up:** Consider implementing a CI check or script that validates environment.yml contains all dependencies from pyproject.toml to prevent future drift.
