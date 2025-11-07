---
date: 2025-11-07T00:00:00Z
researcher: Claude
git_commit: f1b929f2c21526a9babafa5b9547d302e586281a
branch: work-issue-2071
repository: pymc-labs/pymc-marketing
topic: "LogNormalPrior DeserializableError with build_mmm_from_yaml"
tags: [research, codebase, deserialization, yaml, special_priors, LogNormalPrior]
status: complete
last_updated: 2025-11-07
last_updated_by: Claude
issue_number: 2071
---

# Research: LogNormalPrior DeserializableError with build_mmm_from_yaml

**Date**: 2025-11-07T00:00:00Z
**Researcher**: Claude
**Git Commit**: f1b929f2c21526a9babafa5b9547d302e586281a
**Branch**: work-issue-2071
**Repository**: pymc-labs/pymc-marketing
**Issue**: #2071

## Research Question

Why does `LogNormalPrior` cause a `DeserializableError` when used with `build_mmm_from_yaml`? The user encountered this error when trying to use the syntax:

```yaml
beta:
  class: pymc_marketing.special_priors.LogNormalPrior
  kwargs:
    mean: {...}
    std: {...}
```

The workaround discovered was to use `special_prior: LogNormalPrior` instead of `class:` syntax, and to add an empty import of `pymc_marketing.special_priors` when loading the saved model.

## Summary

The root cause is a **dual-system architecture** where the YAML builder uses two different mechanisms for object instantiation:

1. **`build()` system** (`factories.py`) - Uses the `"class"` key to dynamically import and instantiate classes
2. **`deserialize()` system** (`pymc_extras`) - Uses structure-based identification (duck typing) to deserialize registered types

The critical issue: When processing `"priors"` kwargs, the code calls `deserialize()` directly (line 104 in `factories.py`), which only recognizes objects with registered type markers like `"special_prior": "LogNormalPrior"`, NOT objects with only a `"class"` key.

**Why the import is needed**: The `register_deserialization()` call that makes `LogNormalPrior` recognizable happens as a **side effect** when `pymc_marketing.special_priors` is imported. Without this import, the deserialization registry doesn't know how to handle `LogNormalPrior` objects.

## Detailed Findings

### 1. Deserialization Registration Mechanism

#### How Registration Works

The deserialization system uses a **registry pattern** where custom classes register themselves with a global `DESERIALIZERS` list by calling `register_deserialization()` at module import time.

**Registration Code** (`pymc_marketing/special_priors.py:236-246`):
```python
def _is_LogNormalPrior_type(data: dict) -> bool:
    if "special_prior" in data:
        return data["special_prior"] == "LogNormalPrior"
    else:
        return False

register_deserialization(
    is_type=_is_LogNormalPrior_type,
    deserialize=LogNormalPrior.from_dict,
)
```

This registration happens **at module import time** as a side effect. The `register_deserialization()` function appends a tuple `(is_type, deserialize)` to the global `DESERIALIZERS` list in `pymc_extras`.

**Serialization Format** (`pymc_marketing/special_priors.py:159-190`):
```python
def to_dict(self):
    """Convert the prior distribution to a dictionary."""
    data = {
        "special_prior": "LogNormalPrior",  # <-- Type marker
    }
    if self.parameters:
        data["kwargs"] = {
            param: handle_value(value)
            for param, value in self.parameters.items()
        }
    if not self.centered:
        data["centered"] = False
    if self.dims:
        data["dims"] = self.dims
    return data
```

The key insight: `LogNormalPrior` uses `"special_prior": "LogNormalPrior"` as its type identifier, not a `"class"` key.

#### Why Import is Required

**Evidence from test file** (`tests/test_prior.py:19-21`):
```python
from pymc_marketing import (
    prior,  # noqa: F401 - import needed to register custom deserializers
)
```

The comment explicitly states: "import needed to register custom deserializers"

**The mechanism:**
1. Python executes all top-level code when importing a module
2. The `register_deserialization()` call at line 243 in `special_priors.py` modifies the global `DESERIALIZERS` list
3. If the module is never imported, `LogNormalPrior` cannot be deserialized
4. The `deserialize()` function iterates through `DESERIALIZERS` to find a matching type checker

### 2. YAML Building vs Deserialization

#### The `build()` Function Path

**Entry point** (`pymc_marketing/mmm/builders/yaml.py:149`):
```python
model = build(cfg["model"])
```

**The build() function** (`pymc_marketing/mmm/builders/factories.py:63-120`):

Key steps:
1. **Class resolution** (line 74-79): Uses `locate()` to dynamically import the class from the string in `spec["class"]`
2. **Special processing for priors** (line 85-116):
   ```python
   special_processing_keys = ["priors", "prior"]

   if k in special_processing_keys:
       if isinstance(v, dict):
           if k == "priors":
               # Create a dictionary of priors
               priors_dict = {}
               for prior_key, prior_value in v.items():
                   if isinstance(prior_value, dict):
                       priors_dict[prior_key] = deserialize(prior_value)  # LINE 104
   ```
3. **The critical call**: Line 104 calls `deserialize(prior_value)` directly, **not** the `resolve()` or `build()` function

#### Why `class:` Syntax Doesn't Work

When the YAML contains:
```yaml
beta:
  class: pymc_marketing.special_priors.LogNormalPrior
  kwargs: {...}
```

The flow is:
1. `build()` detects `k == "priors"` at line 96
2. Calls `deserialize(prior_value)` at line 104
3. `deserialize()` iterates through `DESERIALIZERS`
4. Each type checker is called (e.g., `_is_LogNormalPrior_type(prior_value)`)
5. **All return False** because they check for `"special_prior"` or `"distribution"` keys, not `"class"`
6. `deserialize()` raises `DeserializableError` - no matching deserializer found

#### Why `special_prior:` Syntax Works

When the YAML contains:
```yaml
beta:
  special_prior: LogNormalPrior
  kwargs: {...}
```

The flow is:
1. `build()` detects `k == "priors"` at line 96
2. Calls `deserialize(prior_value)` at line 104
3. `deserialize()` iterates through `DESERIALIZERS`
4. `_is_LogNormalPrior_type()` checks for `"special_prior"` key
5. **Returns True** - match found!
6. `LogNormalPrior.from_dict()` is called at line 192 in `special_priors.py`
7. Returns a properly constructed `LogNormalPrior` instance

### 3. Model Loading and Serialization Roundtrip

#### Saving the Model

When a model with `LogNormalPrior` is saved, the saturation component is serialized in `MMM.attrs_to_init_kwargs()`:

**Code path** (`pymc_marketing/mmm/multidimensional.py:586`):
```python
"saturation": saturation_from_dict(json.loads(attrs["saturation"])),
```

**Saturation serialization** (`pymc_marketing/mmm/components/saturation.py:499-508`):
```python
def saturation_from_dict(data: dict) -> SaturationTransformation:
    """Get a saturation function from a dictionary."""
    data = data.copy()
    cls = SATURATION_TRANSFORMATIONS[data.pop("lookup_name")]

    if "priors" in data:
        data["priors"] = {
            key: deserialize(value) for key, value in data["priors"].items()
        }
    return cls(**data)
```

The saved JSON contains the serialized `LogNormalPrior` with `"special_prior": "LogNormalPrior"` key.

#### Loading the Model

When loading, `deserialize()` is called again (line 501), which requires the registration to be active.

**Without the import**: The registration at `special_priors.py:243` has never executed, so `deserialize()` fails with `DeserializableError`.

**With the import**: `import pymc_marketing.special_priors` triggers the registration as a side effect, making deserialization work.

### 4. Comparison with Standard Prior Class

The standard `Prior` class from `pymc_extras` has its own registration within pymc_extras itself. However, pymc-marketing provides an **alternative deserializer** for a flatter YAML syntax.

**Alternative Prior deserializer** (`pymc_marketing/prior.py:112-162`):
```python
def is_alternative_prior(data: Any) -> bool:
    """Check if the data is a dictionary representing a Prior."""
    return isinstance(data, dict) and "distribution" in data

def deserialize_alternative_prior(data: dict[str, Any]) -> prior.Prior:
    """Alternative deserializer that recursively handles all nested parameters."""
    data = copy.deepcopy(data)

    distribution = data.pop("distribution")
    dims = data.pop("dims", None)
    centered = data.pop("centered", True)
    transform = data.pop("transform", None)
    parameters = data

    # Recursively deserialize any nested parameters
    parameters = {
        key: value if not isinstance(value, dict) else deserialize(value)
        for key, value in parameters.items()
    }

    return prior.Prior(
        distribution,
        transform=transform,
        centered=centered,
        dims=dims,
        **parameters,
    )

register_deserialization(is_alternative_prior, deserialize_alternative_prior)
```

This is why `distribution: Gamma` syntax works for `Prior` objects in YAML.

### 5. Missing Import in Package Init

The user comment pointed to the `__init__.py` file, suggesting that `special_priors` should be imported there.

**Current state** (`pymc_marketing/__init__.py:16-21`):
```python
# Load the data accessor
import pymc_marketing.data.fivetran  # noqa: F401
from pymc_marketing import clv, customer_choice, mmm
from pymc_marketing.version import __version__

__all__ = ["__version__", "clv", "customer_choice", "mmm"]
```

**Key observation**: `special_priors` is NOT imported in the package init, so the registration side effect doesn't happen automatically when users do `import pymc_marketing`.

**Compare with** (`pymc_marketing/mmm/builders/factories.py:24-25`):
```python
# In order to register custom deserializers
import pymc_marketing.prior  # noqa: F401
```

This ensures that when the YAML builder is used, the alternative `Prior` deserializer is registered. However, there's **no similar import for `special_priors`**.

## Code References

### Key Implementation Files
- `pymc_marketing/special_priors.py:35-246` - LogNormalPrior implementation and registration
- `pymc_marketing/special_priors.py:159-190` - `to_dict()` method with `"special_prior"` key
- `pymc_marketing/special_priors.py:192-219` - `from_dict()` classmethod
- `pymc_marketing/special_priors.py:236-240` - Type checker function
- `pymc_marketing/special_priors.py:243-246` - Registration call

### YAML Building Logic
- `pymc_marketing/mmm/builders/yaml.py:149` - Entry point for building model from YAML
- `pymc_marketing/mmm/builders/factories.py:63-120` - `build()` function
- `pymc_marketing/mmm/builders/factories.py:85-116` - Special processing for priors
- `pymc_marketing/mmm/builders/factories.py:104` - **Critical line** where `deserialize()` is called
- `pymc_marketing/mmm/builders/factories.py:123-141` - `resolve()` helper function

### Model Loading Logic
- `pymc_marketing/mmm/multidimensional.py:586` - Calls `saturation_from_dict()` on load
- `pymc_marketing/mmm/components/saturation.py:499-508` - `saturation_from_dict()` deserializes priors
- `pymc_marketing/mmm/components/saturation.py:501` - Calls `deserialize()` for each prior value

### Alternative Prior Handling
- `pymc_marketing/prior.py:112-114` - Type checker for `"distribution"` key
- `pymc_marketing/prior.py:117-158` - Alternative Prior deserializer
- `pymc_marketing/prior.py:162` - Registration call

### Package Init
- `pymc_marketing/__init__.py:16-21` - Does not import `special_priors`
- `pymc_marketing/mmm/builders/factories.py:24-25` - Imports `prior` for registration

## Architecture Insights

### Dual-System Design Pattern

The codebase uses two complementary but separate systems:

1. **Factory System** (`build()` + `resolve()`)
   - Uses `"class"` key with fully-qualified import paths
   - Dynamically imports and instantiates classes
   - Handles top-level objects (MMM, Effects, Transformations)

2. **Deserialization System** (`deserialize()` + registration)
   - Uses structure-based type identification
   - Requires explicit registration via `register_deserialization()`
   - Handles domain objects (Prior, LogNormalPrior, etc.)

The systems intersect at the "special_processing_keys" in `build()`, where `"priors"` and `"prior"` kwargs are routed to `deserialize()` instead of `resolve()`.

### Registry Pattern with Side Effects

**Key characteristics:**
- Global `DESERIALIZERS` list acts as registry
- Registration happens at module import time (side effect)
- Chain of responsibility: `deserialize()` tries each registered deserializer
- Type discrimination via structural checking (duck typing)
- Recursive deserialization for nested structures

**Trade-offs:**
- ✓ Extensibility: New types can register without modifying core code
- ✓ Separation of concerns: Each class knows how to serialize/deserialize itself
- ✗ Hidden dependencies: Must import modules for registration (not obvious)
- ✗ Import order: Registration depends on module import order
- ✗ Global state: `DESERIALIZERS` is mutable global state

### Type Identification Patterns

Different classes use different structural markers:

1. **LogNormalPrior**: `{"special_prior": "LogNormalPrior", "kwargs": {...}}`
2. **MaskedPrior**: `{"class": "MaskedPrior", "data": {...}}`
3. **Prior** (alternative): `{"distribution": "Normal", "mu": 0, ...}`
4. **Saturation**: `{"lookup_name": "logistic", "priors": {...}}`
5. **Fourier**: `{"class": "YearlyFourier", "data": {...}}`

There's no unified pattern - each class chooses its own structural marker.

## The Gap: Why Users Get Confused

### User Mental Model

Users naturally expect that if a class can be imported with a string like `"pymc_marketing.special_priors.LogNormalPrior"`, then this syntax should work in YAML:

```yaml
beta:
  class: pymc_marketing.special_priors.LogNormalPrior
  kwargs: {...}
```

This works for top-level objects (like `pymc_marketing.mmm.LogisticSaturation`), so it should work for priors too, right?

### Actual Architecture

The architecture has a **special case** for priors: they bypass the normal `resolve()` → `build()` path and go directly to `deserialize()`, which uses a completely different identification mechanism.

**The missing piece**: No deserializer is registered to handle the `{"class": "..."}` format for special priors. The `resolve()` function could call `build()` for "class" keys, but that logic doesn't run because priors take a shortcut to `deserialize()`.

### Documentation Gap

**Key finding**: No YAML examples exist for `LogNormalPrior`. All examples use programmatic Python:

- `docs/source/notebooks/mmm/mmm_gam_options.ipynb` - Python only
- `docs/source/notebooks/mmm/mmm_multidimensional_example.ipynb` - Python only
- No test files with LogNormalPrior in YAML configs

Users must discover the `special_prior:` syntax by reading the serialization code or through trial and error.

## Open Questions

### 1. Should `special_priors` be imported in `__init__.py`?

**Pro**: Would make deserialization work automatically when `import pymc_marketing` is used.

**Con**: Adds import overhead for users who don't need special priors. However, the module is relatively small.

**Precedent**: The `factories.py` already imports `pymc_marketing.prior` for registration side effects (line 25).

### 2. Should the `build()` function handle the "class" key for priors?

Currently, line 104 in `factories.py` always calls `deserialize()` for priors. Should it check for a "class" key first and call `build()` in that case?

**Pseudo-code:**
```python
if isinstance(prior_value, dict):
    if "class" in prior_value:
        priors_dict[prior_key] = build(prior_value)
    else:
        priors_dict[prior_key] = deserialize(prior_value)
```

This would make the `class:` syntax work without requiring users to know about `special_prior:`.

### 3. Should there be a fallback in `deserialize()` to call `build()`?

If no registered deserializer matches, should `deserialize()` check for a "class" key and attempt `build()` as a fallback?

This would require coordination with the `pymc_extras` library since `deserialize()` is defined there.

### 4. Should the serialization be standardized?

Different classes use different structural markers (`special_prior`, `class`, `distribution`, `lookup_name`). Should there be a standard format?

This is a bigger architectural question that might require deprecation cycles.

### 5. How should YAML documentation be improved?

- Add YAML examples for LogNormalPrior to docs
- Add YAML config tests for LogNormalPrior
- Document the `special_prior:` syntax explicitly
- Add a troubleshooting section about import requirements

## Related Research

This research builds on understanding of:
- PyMC deserialization system from `pymc_extras`
- YAML configuration patterns in pymc-marketing
- Prior class architecture and hierarchy
- MMM component serialization

## Recommendations

### Short-term (User Workaround)

**Document the current behavior**:
1. Use `special_prior: LogNormalPrior` syntax in YAML
2. Add `import pymc_marketing.special_priors` when loading models
3. Add YAML examples to documentation

### Medium-term (Code Fix)

**Option A - Import in factories.py**:
Add to `pymc_marketing/mmm/builders/factories.py:25`:
```python
# In order to register custom deserializers
import pymc_marketing.prior  # noqa: F401
import pymc_marketing.special_priors  # noqa: F401  # <-- ADD THIS
```

This ensures special priors are registered when the YAML builder is used.

**Option B - Import in __init__.py**:
Add to `pymc_marketing/__init__.py:17`:
```python
import pymc_marketing.data.fivetran  # noqa: F401
import pymc_marketing.special_priors  # noqa: F401  # <-- ADD THIS
```

This ensures special priors are registered whenever pymc_marketing is imported.

### Long-term (Architectural Improvement)

**Unify the build() and deserialize() systems**:
1. Make `resolve()` check for registered deserializers before calling `build()`
2. Or make `build()` check for registered deserializers for special kwargs
3. Or register a generic "class" deserializer that calls `build()`

This would make both syntaxes work seamlessly:
- `class: pymc_marketing.special_priors.LogNormalPrior`
- `special_prior: LogNormalPrior`

## Conclusion

The `LogNormalPrior` deserialization issue stems from a fundamental architectural split between the factory system (`build()`) and the deserialization system (`deserialize()`). The YAML builder routes priors directly to `deserialize()`, which doesn't recognize the `"class"` key format.

The immediate fix is to add `import pymc_marketing.special_priors` to `factories.py` or `__init__.py`, ensuring the registration side effect happens automatically. The long-term solution involves architectural unification to make both syntaxes work seamlessly.

Users should be advised to:
1. Use `special_prior: LogNormalPrior` syntax in YAML configs
2. Ensure `pymc_marketing.special_priors` is imported when loading models
3. Refer to the serialization format in `LogNormalPrior.to_dict()` for the correct structure

The documentation should be updated with explicit YAML examples and troubleshooting guidance for this pattern.
