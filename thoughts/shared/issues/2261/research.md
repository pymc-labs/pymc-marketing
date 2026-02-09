---
date: 2026-02-09T11:32:32+00:00
researcher: Claude Sonnet 4.5
git_commit: b7c1bb6c311e181d103ee9d74bb5200815df0dc4
branch: work-issue-2261
repository: pymc-labs/pymc-marketing
topic: "Development Version Suffix Implementation"
tags: [research, codebase, versioning, packaging, release-management]
status: complete
last_updated: 2026-02-09
last_updated_by: Claude Sonnet 4.5
issue_number: 2261
---

# Research: Development Version Suffix Implementation

**Date**: 2026-02-09T11:32:32+00:00
**Researcher**: Claude Sonnet 4.5
**Git Commit**: b7c1bb6c311e181d103ee9d74bb5200815df0dc4
**Branch**: work-issue-2261
**Repository**: pymc-labs/pymc-marketing
**Issue**: #2261

## Research Question

How should pymc-marketing implement a development version suffix (e.g., `+dev`) to differentiate the main branch from the last released version (0.17.1), following Python packaging best practices?

Related to issue #860.

## Summary

The current version string in `pymc_marketing/version.py` is hardcoded as `"0.17.1"`, matching the last release. However, the main branch has diverged with new developments, creating ambiguity about whether an installed version is the official release or a development build.

**Key Findings**:
1. The project uses **Hatchling** as its build backend with dynamic version reading from `version.py`
2. Python ecosystem standard is **`.dev0`** suffix per PEP 440 for development versions
3. Three implementation approaches available:
   - **Manual**: Update `version.py` to `"0.17.2.dev0"` or `"0.17.1+dev"` after each release
   - **Semi-automated**: Use `hatch-vcs` (Hatch's setuptools-scm equivalent) for git-based versioning
   - **Fully automated**: Migrate to setuptools-scm (most common in scientific Python ecosystem)
4. PyMC (the parent project) uses **versioneer** with automatic `.dev` suffix generation
5. The release workflow (`.github/workflows/pypi.yml`) validates that version matches git tags

## Detailed Findings

### Current Version Management System

**Version Source**: `pymc_marketing/version.py:16`
```python
__version__ = "0.17.1"
```

**Package Configuration**: `pyproject.toml:10,107-108`
```toml
[project]
dynamic = ["version", "readme"]

[tool.hatch.version]
path = "pymc_marketing/version.py"
```

**Build System**:
- Backend: `hatchling` with `hatch-fancy-pypi-readme`
- Version read dynamically from `version.py` at build time
- No git-based version generation currently implemented

**Version Export**: `pymc_marketing/__init__.py:22-24`
```python
from pymc_marketing.version import __version__

__all__ = ["__version__", "bass", "clv", "customer_choice", "mmm"]
```

**Usage Throughout Codebase**:
- InferenceData metadata: Version stored in `posterior`, `prior`, and `prior_predictive` group attributes
- MLflow logging: Version logged as experiment parameter
- Documentation: Sphinx uses `pymc_marketing.__version__` for release number
- CI/CD validation: PyPI workflow asserts version matches git tag on releases

### PEP 440 Standards

**Development Release Format**: `X.Y.Z.devN` (canonical form)

**Version Ordering**:
```
1.0.dev0 < 1.0a1 < 1.0b1 < 1.0rc1 < 1.0 < 1.0.post1
```

**Key Rules**:
- `.devN` sorts **before** the release (indicates "approaching" the release)
- `+dev` (local version identifier) sorts **after** the release (indicates "based on" the release)
- `.dev0` is standard when N is omitted
- Development releases should precede their corresponding final release

**Examples**:
- `0.17.1.dev0` → Development toward 0.17.1 (before release)
- `0.17.2.dev0` → Development toward next patch release
- `0.18.0.dev0` → Development toward next minor release
- `0.17.1+dev` → Local development based on 0.17.1 (after release)

### Major Python Projects' Approaches

#### NumPy
- **Pattern**: `X.Y.Z.dev0` on main branch
- **Workflow**: After release, immediately tag `vX.Y+1.0.dev0` and update version
- **Tool**: Manual version management in `pyproject.toml`
- **Example**: Current main is `v2.5.dev0`

#### SciPy
- **Pattern**: `.dev0+<git-hash>`
- **Attributes**: Multiple version attributes (`__version__`, `short_version`, `release`, `git_revision`)
- **Tool**: Automated version generation from git

#### pandas
- **Pattern**: `X.Y.Z.dev0+NNN.gHASH`
- **Tool**: **versioneer** with PEP 440 style
- **Config**: `versionfile_source = "pandas/_version.py"`

#### PyMC
- **Pattern**: `X.Y.Z+NNN.gHASH` (development builds)
- **Tool**: **versioneer** version 0.29
- **Config**:
  ```toml
  [tool.versioneer]
  VCS = "git"
  style = "pep440"
  versionfile_source = "pymc/_version.py"
  versionfile_build = "pymc/_version.py"
  tag_prefix = "v"
  ```

**Common Pattern**: All use `.dev0` suffix, with optional git metadata after `+`

### Implementation Options

#### Option 1: Manual Version Management (Minimal Change)

**Approach**: Update `version.py` after each release with `.dev0` suffix

**After 0.17.1 release**:
```python
__version__ = "0.17.2.dev0"  # or "0.18.0.dev0" for minor bump
```

**Process**:
1. Release 0.17.1 → version file contains `"0.17.1"`
2. Immediately after release → update to `"0.17.2.dev0"`
3. Continue development with dev suffix visible
4. Before next release → update to `"0.17.2"`
5. Repeat

**Pros**:
- No build system changes required
- Complete control over version string
- Works with existing Hatchling setup
- Simple to understand

**Cons**:
- Requires manual version bumps
- Easy to forget after releases
- No git metadata in version string
- Doesn't indicate how far from release (commit distance)

**Documentation Pattern**:
The `docs/source/conf.py:54-64` already handles dev versions:
```python
if os.environ.get("READTHEDOCS", False):
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if rtd_version.lower() == "stable":
        version = release.split("+")[0]  # Removes local version identifier
    elif rtd_version.lower() == "latest":
        version = "dev"
    else:
        version = rtd_version
```

#### Option 2: hatch-vcs (Hatch-Native Solution)

**Approach**: Use `hatch-vcs` plugin for git-based version generation

**Configuration**:
```toml
[build-system]
requires = ["hatchling", "hatch-vcs", "hatch-fancy-pypi-readme>=24.0.0"]
build-backend = "hatchling.build"

[project]
dynamic = ["version", "readme"]

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "pymc_marketing/_version.py"
```

**Version Generation Logic**:
- On tag: `X.Y.Z` (clean)
- After tag: `X.Y.(Z+1).devN+gHASH` (where N is commit distance)
- Dirty working directory: appends timestamp

**Example**:
- Tag `v0.17.1` → version `0.17.1`
- 5 commits after tag → version `0.17.2.dev5+g2b9e661`

**Pros**:
- Automatic version generation from git
- Shows commit distance and hash
- Native Hatch integration
- No manual version bumps needed
- PEP 440 compliant

**Cons**:
- Requires `hatch-vcs` dependency
- Changes version generation mechanism
- Needs migration from current approach
- Requires version file change (`version.py` → `_version.py`)

**Migration Steps**:
1. Add `hatch-vcs` to build dependencies
2. Update `[tool.hatch.version]` configuration
3. Generate initial `_version.py` file
4. Update imports from `version.py` to `_version.py`
5. Remove old `version.py` (or keep for backwards compatibility)

#### Option 3: setuptools-scm (Ecosystem Standard)

**Approach**: Migrate from Hatchling to setuptools with setuptools-scm

**Configuration**:
```toml
[build-system]
requires = ["setuptools>=80", "setuptools-scm>=8"]
build-backend = "setuptools.build_meta"

[project]
dynamic = ["version"]

[tool.setuptools_scm]
version_file = "pymc_marketing/_version.py"
```

**Version Generation Logic**: Same as hatch-vcs (based on setuptools-scm)

**Pros**:
- Industry standard for scientific Python
- Most mature and widely used
- Consistent with PyMC ecosystem
- Extensive documentation and community support
- Battle-tested in NumPy, SciPy, pandas

**Cons**:
- Requires build backend change (Hatchling → setuptools)
- More invasive migration
- May affect fancy PyPI readme generation
- Larger scope of testing needed

**Migration Complexity**: Higher than hatch-vcs due to build backend change

### Release Workflow Integration

**Current Workflow**: `.github/workflows/pypi.yml`

**Version Validation** (lines 32, 41, 71):
```bash
python -c "import pymc_marketing as pmm; assert pmm.__version__ == '${{ github.ref_name }}' if '${{ github.ref_type }}' == 'tag' else pmm.__version__; print(pmm.__version__)"
```

**Compatibility**:
- Manual approach: ✅ Works (requires version update before tagging)
- hatch-vcs: ✅ Works (version auto-generated from tag)
- setuptools-scm: ✅ Works (version auto-generated from tag)

**Key Consideration**: Automated tools remove the manual version bump step:
1. Current: Update `version.py` → commit → tag → release
2. With automation: Commit → tag → release (version generated from tag)

### Recommendation Matrix

| Criterion | Manual | hatch-vcs | setuptools-scm |
|-----------|--------|-----------|----------------|
| Ease of implementation | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Maintenance burden | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Ecosystem alignment | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Git metadata | ❌ | ✅ | ✅ |
| Build system changes | ❌ | Minor | Major |
| Risk level | Low | Medium | Medium-High |

**Recommended Approach**: **Manual version management with `.dev0` suffix**

**Rationale**:
1. **Immediate solution**: Can be implemented in a single commit
2. **Low risk**: No build system changes or dependency additions
3. **Maintains current workflow**: Minimal disruption to release process
4. **Sufficient for the issue**: Solves the core problem (version ambiguity)
5. **Future migration path**: Can later migrate to hatch-vcs if automation is desired

**Recommended Version String**: `"0.17.2.dev0"` (assumes next release is a patch)

**Next Steps**:
1. Update `pymc_marketing/version.py` to `"0.17.2.dev0"`
2. Document version bump process in CONTRIBUTING.md
3. Optionally add automation in future milestone

## Code References

- `pymc_marketing/version.py:16` - Current version definition
- `pymc_marketing/__init__.py:22-24` - Version export
- `pyproject.toml:10,107-108` - Hatchling version configuration
- `.github/workflows/pypi.yml:32,41,71` - Version validation in CI
- `docs/source/conf.py:51,54-64` - Documentation version handling
- `pymc_marketing/model_builder.py:35,1044,1207,1273-1274` - Version metadata in InferenceData
- `pymc_marketing/mlflow.py:176,984` - Version logging in MLflow

## Architecture Insights

**Single Source of Truth**: The project maintains a clean separation between version definition (`version.py`) and version reading (`pyproject.toml`), following Python packaging best practices.

**Version Propagation**:
```
version.py (__version__)
    ↓
__init__.py (re-export)
    ↓
├─→ model_builder.py (InferenceData attributes)
├─→ mlflow.py (experiment logging)
├─→ docs/conf.py (Sphinx documentation)
├─→ CI/CD (validation and PyPI publishing)
└─→ Public API (pymc_marketing.__version__)
```

**Build-Time vs Runtime**:
- Build-time: Hatchling reads from `version.py` to populate package metadata
- Runtime: Python imports directly from `version.py` module
- This dual-access pattern would work seamlessly with automated version generation tools

**Reproducibility Focus**: The extensive use of version metadata in InferenceData attributes and MLflow logging indicates the project prioritizes experiment reproducibility. Development versions should maintain this property.

## Related Research

- PEP 440 - Version Identification and Dependency Specification
- setuptools-scm Documentation
- NumPy Release Process
- PyMC versioning with versioneer

## Open Questions

1. **Next version target**: Should the dev version be `0.17.2.dev0` (patch) or `0.18.0.dev0` (minor)?
   - **Context**: Comment indicates "a couple of PR to merge" before next release
   - **Recommendation**: Check milestone #12 to determine planned version bump

2. **Issue #860**: What is the content of related issue #860?
   - Unable to access via gh CLI in current environment
   - May contain additional context or historical discussion

3. **Automation timeline**: Should automation (hatch-vcs or setuptools-scm) be planned for a future milestone?
   - Depends on team preference and maintenance burden assessment
   - Could be added incrementally after manual approach is validated

4. **Post-release process**: Should version bumping be part of the release checklist?
   - Current process: Manual bump before release
   - Proposed: Manual bump immediately after release (to add .dev0)
   - Consider adding to CONTRIBUTING.md or release documentation

## Implementation Steps (Recommended)

### Phase 1: Immediate Fix (This PR)
1. Determine next version number (check milestone #12)
2. Update `pymc_marketing/version.py` to `"0.X.Y.dev0"`
3. Commit with message: "Add .dev0 suffix to version after 0.17.1 release"
4. Verify all tests pass and version is correctly displayed
5. Document in PR description why this approach was chosen

### Phase 2: Process Documentation (Follow-up)
1. Update CONTRIBUTING.md with version management guidelines
2. Add release checklist item: "Update version.py to next .dev0 after release"
3. Consider adding pre-commit hook to warn if version doesn't contain .dev0 on main

### Phase 3: Consider Automation (Future Milestone)
1. Evaluate hatch-vcs for automated version management
2. Assess migration effort and risks
3. Create implementation plan if benefits justify the change
4. Keep setuptools-scm as backup option for major refactoring

## Sources

- [PEP 440 – Version Identification and Dependency Specification](https://peps.python.org/pep-0440/)
- [setuptools-scm Documentation](https://setuptools-scm.readthedocs.io/)
- [hatch-vcs PyPI](https://pypi.org/project/hatch-vcs/)
- [NumPy Release Process](https://numpy.org/devdocs/dev/releasing.html)
- [PyMC pyproject.toml](https://github.com/pymc-devs/pymc/blob/main/pyproject.toml)
- [Python Packaging User Guide - Single-sourcing Version](https://packaging.python.org/guides/single-sourcing-package-version/)
- [Inter-Release Versioning Recommendations](https://michaelgoerz.net/notes/inter-release-versioning-recommendations.html)
