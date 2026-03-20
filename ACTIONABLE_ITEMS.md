# Actionable Items for PR #2391

This document contains concrete, implementable fixes for issues identified in the code review.

---

## Priority 1: Critical for Merge

### None identified
All critical functionality is working correctly. The items below are improvements.

---

## Priority 2: Should Fix Before Merge

### Issue #1: Remove `lookup_name` from Basis classes

**Problem**: The `lookup_name` field still exists in `Basis` and its subclasses even though the PR aims to eliminate it in favor of `__type__`.

**Files affected**:
- `pymc_marketing/mmm/events.py`

**Current code** (line 121-125):
```python
class Basis(Transformation):
    """Basis transformation associated with an event model."""
    
    prefix: str = "basis"
    lookup_name: str  # ← Should be removed
```

**Concrete subclass examples** (lines 253-256, 308):
```python
@registry.register
class GaussianBasis(Basis):
    """Gaussian basis transformation."""
    
    lookup_name = "gaussian"  # ← Should be removed
```

**Recommended fix**:

1. Remove `lookup_name: str` from the `Basis` class definition (line 125)
2. Remove all `lookup_name = "..."` assignments from concrete subclasses:
   - `GaussianBasis` (line 256)
   - `HalfGaussianBasis` (line 308)
   - `AsymmetricGaussianBasis` (search for its lookup_name)

The `from_dict()` method already handles this gracefully by popping `lookup_name` (line 132), so removal won't break deserialization of old files.

**Verification**: 
- Run `pytest tests/mmm/test_events.py -v`
- Verify that serialization roundtrips still work
- Check migration tests still pass

---

## Priority 3: Should Fix Soon (Post-Merge Acceptable)

### Issue #2: Add validation for EventAdditiveEffect.name

**Problem**: The custom deserializer constructs idata group names via string concatenation, which could fail if `name` contains special characters.

**File**: `pymc_marketing/mmm/additive_effect.py`

**Current code** (lines 280-299):
```python
def _deserialize_event_additive_effect(data: dict[str, Any], context) -> EventAdditiveEffect:
    """Custom deserializer that reconstructs df_events from idata."""
    # ...
    component_id = data.get("name", "event_additive")
    group_name = f"{component_id}_df_events"  # ← No validation
    
    if group_name not in context.idata:
        raise KeyError(...)
```

**Recommended fix**:

Add validation in `EventAdditiveEffect.__init__` (around line 389):

```python
@model_validator(mode="after")
def _validate_name(self):
    """Ensure name is a valid Python identifier for idata group names."""
    if not self.name.isidentifier():
        raise ValueError(
            f"EventAdditiveEffect name must be a valid Python identifier, "
            f"got {self.name!r}. Use only letters, numbers, and underscores, "
            "and don't start with a number."
        )
    return self
```

**Why this helps**:
- Prevents runtime errors during deserialization
- Makes the naming convention explicit
- Fails fast at construction time rather than save/load time

---

### Issue #3: Document model ID invalidation after migration

**Problem**: When users migrate a v0 model to v1, the model `id` hash changes (because serialization format changed). Loading with `check=True` will fail even though the model is functionally identical.

**Files to update**:
1. `pymc_marketing/serialization_migration.py` - module docstring
2. CLI output in `main()` function (line 167-174)

**Recommended addition to module docstring** (after line 19):

```python
"""Migration tool for old-format serialized models.

Converts v0 (pre-TypeRegistry) idata attrs to v1 (__type__-based) format.

**Important**: After migration, the model's `id` attribute will change because the
serialization format has changed. This means:

- If you try to load a migrated model with `MMM.load(..., check=True)`, you may get
  a `DifferentModelError` about mismatched IDs, even though the model is identical.
- **Workaround**: Use `MMM.load(..., check=False)` for migrated models, or re-save
  the model after loading to update the ID.
- The migration tool automatically removes the old `id` to prevent false errors.

Usage:
    python -m pymc_marketing.serialization_migration model.nc
"""
```

**Recommended addition to CLI output** (line 173-174):

```python
shutil.move(tmp_path, fname)
print(f"Done. Saved migrated model to {fname}")
print("\nNote: The model ID has been removed because the serialization format changed.")
print("When loading, consider using check=False if you encounter ID mismatch errors.")
```

---

## Priority 4: Nice to Have (Future Work)

### Enhancement #1: Add factory namespace whitelist for security

**Problem**: `DeferredFactory.resolve()` can import and call arbitrary functions, which could be exploited if an attacker controls a `.nc` file.

**File**: `pymc_marketing/serialization.py`

**Current code** (lines 92-95):
```python
def resolve(self) -> Any:
    """Import the factory function and call it with kwargs."""
    fn = _import_from_dotted_path(self.factory)
    return fn(**self.kwargs)
```

**Recommended enhancement**:

```python
# Add at module level (after line 36)
ALLOWED_FACTORY_NAMESPACES = (
    "pymc_marketing.",
    "pymc_extras.prior.",
    "pymc.distributions.",
    "builtins.dict",  # Explicit whitelist for builtins
)

# Update resolve() method
def resolve(self) -> Any:
    """Import the factory function and call it with kwargs.
    
    Raises
    ------
    SerializationError
        If the factory function is not in an allowed namespace.
    """
    if not any(self.factory.startswith(ns) for ns in ALLOWED_FACTORY_NAMESPACES):
        raise SerializationError(
            f"Factory {self.factory!r} is not in an allowed namespace. "
            f"Only factories from these namespaces are permitted: "
            f"{ALLOWED_FACTORY_NAMESPACES}. "
            f"This restriction prevents arbitrary code execution from untrusted files."
        )
    fn = _import_from_dotted_path(self.factory)
    return fn(**self.kwargs)
```

**Rationale**: Defense in depth. While the TypeRegistry already restricts deserialization to registered types, this adds an extra layer for the DeferredFactory path.

---

### Enhancement #2: Improve registry error messages

**File**: `pymc_marketing/serialization.py`

**Current code** (lines 220-227):
```python
if type_key not in self._registry:
    raise SerializationError(
        f"Unknown type {type_key!r}. The class may not have been "
        f"registered with @registry.register, or the module defining "
        f"it may not have been imported. "
        f"Registered types: {sorted(self._registry.keys())}"
    )
```

**Recommended enhancement**:

```python
if type_key not in self._registry:
    module_path, _, class_name = type_key.rpartition(".")
    raise SerializationError(
        f"Unknown type {type_key!r}.\n\n"
        f"Possible fixes:\n"
        f"1. Import the class before calling load():\n"
        f"   from {module_path} import {class_name}\n"
        f"2. Ensure the class is decorated with @registry.register\n"
        f"3. Check if you're using a custom component that needs registration\n\n"
        f"Currently registered types:\n" +
        "\n".join(f"  - {t}" for t in sorted(self._registry.keys())[:10]) +
        ("\n  ... and more" if len(self._registry) > 10 else "")
    )
```

**Rationale**: More actionable error messages improve developer experience, especially for users creating custom components.

---

### Enhancement #3: Add docstring examples for extension points

**File**: `pymc_marketing/serialization.py`

**Location**: Module docstring (after line 22)

**Recommended addition**:

```python
"""Unified serialization infrastructure for pymc-marketing.

This module provides the ``TypeRegistry``, ``Serializable`` protocol,
``SerializableMixin``, ``DeferredFactory``, and ``DeserializationContext``
that replace the scattered serialization patterns across MMM components.

Every serializable object produces a JSON-safe dict with a ``__type__`` key
(fully-qualified class path). The ``TypeRegistry`` dispatches deserialization
from that key alone.

Creating Custom Serializable Components
----------------------------------------

**Basic pattern** (for simple transformations):

.. code-block:: python

    from pymc_marketing.serialization import registry
    from pymc_marketing.mmm.components.base import Transformation
    
    @registry.register
    class MyTransformation(Transformation):
        def __init__(self, param1, param2, **kwargs):
            super().__init__(**kwargs)
            self.param1 = param1
            self.param2 = param2
        
        def to_dict(self):
            data = super().to_dict()
            data["param1"] = self.param1
            data["param2"] = self.param2
            return data
        
        @classmethod
        def from_dict(cls, data):
            data = data.copy()
            data.pop("__type__", None)
            return cls(**data)
        
        def function(self, x, **params):
            # Your transformation logic
            return x * self.param1 + self.param2

**Advanced pattern** (with DeserializationContext):

.. code-block:: python

    from pymc_marketing.serialization import registry, DeserializationContext
    
    def my_custom_deserializer(data: dict, context: DeserializationContext):
        \"\"\"Custom deserializer that reads from idata.\"\"\"
        if context is None or context.idata is None:
            raise ValueError("MyComponent requires idata context")
        
        # Read supplementary data from idata
        extra_data = context.idata["my_component_data"].to_dataframe()
        
        return MyComponent(
            param1=data["param1"],
            extra_data=extra_data,
        )
    
    registry.register(
        "my_package.MyComponent",
        MyComponent,
        deserializer=my_custom_deserializer,
    )

**Common pitfalls**:

1. **Forgetting to import before load()**: The module defining your custom
   component must be imported before calling ``MMM.load()``, or the registry
   won't know about it.

2. **Non-serializable state**: If your component holds live PyTensor
   variables or DataFrames, use ``DeferredFactory`` or store them in
   supplementary idata groups (like ``EventAdditiveEffect`` does).

3. **Not calling super().to_dict()**: Forgetting this loses the ``__type__``
   key, breaking deserialization.
"""
```

---

## Testing Checklist

After implementing fixes, run these tests to verify:

```bash
# Core serialization tests
pytest tests/test_serialization.py -v

# Migration tests
pytest tests/test_serialization_migration.py -v

# Component-specific roundtrip tests
pytest tests/mmm/test_serialization_roundtrips.py -v

# Events tests (after removing lookup_name)
pytest tests/mmm/test_events.py -v

# Full MMM integration tests
pytest tests/mmm/test_multidimensional.py::test_save_load -v
```

---

## Summary

**Must-fix before merge**: Issue #1 (remove lookup_name)

**Should-fix before merge**: Issues #2 and #3 (validation and documentation)

**Nice-to-have**: Enhancements #1, #2, #3 (security, error messages, docs)

All issues are low-risk and straightforward to implement. The core architecture is solid.
