# Serialization Overhaul Design

**Date:** 2026-03-05
**Status:** Approved
**Scope:** Multidimensional MMM serialization — unify patterns, fix all known issues, future-proof for custom components

---

## Table of Contents

- [Problem Statement](#problem-statement)
  - [Known Failures](#known-failures)
  - [Current Serialization Patterns](#current-serialization-patterns-4-inconsistent)
- [Design Decisions](#design-decisions)
- [Scope Boundaries](#scope-boundaries)
- [Serialization Policy](#serialization-policy)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
  - [1. Core Serialization Infrastructure](#1-core-serialization-infrastructure)
  - [2. DeferredFactory — Solving the Non-Serializable State Problem](#2-deferredfactory--solving-the-non-serializable-state-problem)
  - [3. EventAdditiveEffect & Supplementary Data Storage](#3-eventadditiveeffect--supplementary-data-storage)
  - [4. Registry Consolidation & MMM Save/Load](#4-registry-consolidation--mmm-saveload)
  - [5. Migration Tool](#5-migration-tool)
- [End State Summary](#end-state-summary)
  - [One Registry](#one-registry)
  - [One Serialization Contract](#one-serialization-contract)
  - [How Each Class Category Serializes](#how-each-class-category-serializes)
  - [What Gets Removed](#what-gets-removed)
  - [What Stays Unchanged](#what-stays-unchanged)
  - [What Gets Added](#what-gets-added)
- [File Layout](#file-layout)
- [Error Handling](#error-handling)
- [Implementation Order](#implementation-order)
- [Testing Strategy](#testing-strategy)
- [Tickets Resolved by This Design](#tickets-resolved-by-this-design)

---

## Problem Statement

PyMC-Marketing cannot reliably save and load models that use HSGP-based time-varying
parameters or certain MuEffect-based additive effects. Beyond these known failures,
the codebase uses at least four different serialization patterns with no unified
contract, making it difficult for users to create custom serializable components.

### Known Failures

| Component | Root Cause | When It Affects Users | References |
|---|---|---|---|
| **EventAdditiveEffect** | Deserialization intentionally raises `ValueError` — both `df_events` DataFrame **and** the `effect` field (`EventEffect` config with adstock/basis) are excluded from serialization. Only `event_names` (a list of strings) is preserved. | When saving and loading any model that uses event-based additive effects. The model cannot be loaded — deserialization always raises. | [exclusion](../../pymc_marketing/mmm/multidimensional.py#L344), [deser raises](../../pymc_marketing/mmm/multidimensional.py#L349), [test](../../tests/mmm/test_serialization_issues.py#L176), [#1921](https://github.com/pymc-labs/pymc-marketing/issues/1921) |
| **Custom MuEffects** | `_MUEFFECT_DESERIALIZERS` has only 3 hardcoded entries (`FourierEffect`, `LinearTrendEffect`, `EventAdditiveEffect`); no public API for user registration. Serialization works (singledispatch fallback uses `model_dump`), but deserialization raises `ValueError("Unknown MuEffect class")`. The `Transformation` base class (adstock/saturation/basis) is a separate issue — it is not Pydantic-based and uses a different metaclass-based registration system. | When a user creates a custom `MuEffect` subclass, saves the model, and later tries to load it. Saving succeeds, but loading always fails with "Unknown MuEffect class". | [registry](../../pymc_marketing/mmm/multidimensional.py#L266), [test](../../tests/mmm/test_serialization_issues.py#L242), [#1921](https://github.com/pymc-labs/pymc-marketing/issues/1921) |
| **Default `_serialize_mu_effect` with Prior fields** | The singledispatch fallback calls `model_dump(mode="json")`, which fails for any `MuEffect` subclass containing `Prior` or `InstanceOf[...]` fields. Affects custom MuEffects that embed Priors. Built-in types avoid this via custom singledispatch handlers. | When saving a model that contains a custom `MuEffect` subclass with `Prior` or `InstanceOf` fields. The save itself fails with `PydanticSerializationError`. | [fallback](../../pymc_marketing/mmm/multidimensional.py#L233), [test](../../tests/mmm/test_serialization_issues.py#L291) |
| **HSGP/HSGPPeriodic/SoftPlusHSGP `dims` tuple→list** | JSON has no tuple type: `("time",)` → JSON array → `["time"]`. `from_dict()` and `_dim_is_at_least_one` validator only handle `str→tuple`, not `list→tuple`. Affects all three HSGP classes. `LinearTrend.dims` and `VariableScaling.dims` are unaffected — Pydantic's strict `str \| tuple[str, ...]` annotation coerces list→tuple automatically during `model_validate()`. | When saving and loading any model that includes HSGP, HSGPPeriodic, or SoftPlusHSGP components. The loaded model fails to reconstruct these components due to the type mismatch. | [validator](../../pymc_marketing/mmm/hsgp.py#L324), [test](../../tests/mmm/test_serialization_issues.py#L80), [#2087](https://github.com/pymc-labs/pymc-marketing/issues/2087) |
| **HSGPKwargs.cov_func** | Can hold a live `pm.gp.cov.Covariance` PyMC object (`InstanceOf[pm.gp.cov.Covariance]`). `model_dump(mode="json")` raises `PydanticSerializationError`. Only triggers when users explicitly pass a covariance object (default is `None`). | When saving a model where the user explicitly passed a custom `cov_func` to `HSGPKwargs`. Models using the default `cov_func=None` are unaffected. | [field](../../pymc_marketing/hsgp_kwargs.py#L81), [test](../../tests/mmm/test_serialization_issues.py#L149), [#2087](https://github.com/pymc-labs/pymc-marketing/issues/2087) |
| **LinearTrendEffect** | Two issues: (1) `linear_trend_first_date` is a runtime-only attribute set via `model_config={"extra": "allow"}` + `__init__`, not a Pydantic field — re-derived in `create_data()` during normal load flow but fragile. (2) `model_dump(mode="json")` fails with `PydanticSerializationError` because `LinearTrend` contains `Prior` fields that Pydantic can't JSON-serialize. The custom `_serialize_mu_effect` handler avoids this by serializing `trend` separately. | When serializing a `LinearTrendEffect` through the generic `model_dump` path (e.g., custom code calling `model_dump(mode="json")` directly). The built-in save/load flow works around this via a custom singledispatch handler, so standard save/load is unaffected. | [runtime attr](../../pymc_marketing/mmm/additive_effect.py#L461), [test](../../tests/mmm/test_serialization_issues.py#L271) |
| **HSGP Prior equality** (SoftPlusHSGP, HSGPPeriodic) | `Prior` objects store PyTensor symbolic expressions (`lam=-pt.log(mass)/upper`). `Prior.to_dict()` handles this correctly via `.eval()`, but after roundtrip the parameter type changes from `TensorVariable` to `float`, breaking `Prior.__eq__` (TypeError on cross-type comparison). Dict-level roundtrip works. | When comparing a loaded model's HSGP priors against original objects (e.g., asserting config equality after save/load). Does not affect model fitting or predictions — only equality checks break. | [source](../../pymc_marketing/mmm/hsgp.py#L204), [test](../../tests/mmm/test_serialization_issues.py#L61), [#2087](https://github.com/pymc-labs/pymc-marketing/issues/2087) |
| **Silent failures** | `build_from_idata()` catches all MuEffect deserialization errors with a broad `except Exception` and only warns — model loads but with missing effects. Effects dropped with only a warning and no recovery path. | When loading any model where a MuEffect fails to deserialize. A warning is emitted but the model appears to load successfully with missing effects — predictions and contributions will be wrong, and the warning is easy to miss. | [catch-all](../../pymc_marketing/mmm/multidimensional.py#L3301), [test](../../tests/mmm/test_serialization_issues.py#L195), [#1921](https://github.com/pymc-labs/pymc-marketing/issues/1921) |

### Current Serialization Patterns

These patterns fall into two conceptual layers:

**Component layer** — how individual components serialize themselves into dicts:

1. **`to_dict()`/`from_dict()` + `register_deserialization`** — adstock, saturation, HSGP, fourier, events/Basis
2. **Pydantic `model_dump()`/`model_validate()`** — HSGPKwargs, Scaling, LinearTrend, MuEffect subclasses
3. **`singledispatch` serializers** — per-type handlers for MuEffects in `multidimensional.py`

**Orchestrator layer** — how the MMM packs/unpacks those component dicts into idata attrs:

4. **Manual JSON encoding** — `create_idata_attrs()` / `attrs_to_init_kwargs()` on MMM

## Design Decisions

- **Backward compatibility:** Can break old `.nc` format; provide a migration tool to convert old files
- **PyTensor expressions:** Store a "recipe" (factory function + scalar args) instead of the live tensor result
- **External data (df_events):** Store inside InferenceData to make `.nc` files self-contained
- **Unified pattern:** Hybrid — Pydantic for BaseModel classes, `Serializable` protocol for non-Pydantic classes (e.g., `Transformation`)
- **pymc_extras dependency:** No changes required to `pymc_extras`; new infrastructure lives entirely in pymc-marketing

## Scope Boundaries

This overhaul targets **Multidimensional MMM serialization only**. The legacy
`MMMModelBuilder` (being deprecated) is not addressed. CLV models, Customer
Choice models (MNLogit, NestedLogit, MixedLogit), and MVITS are **out of scope**
and must continue to work unchanged.

**Why non-MMM models are excluded:**

- CLV and Customer Choice models use a much simpler serialization path — their
  only serialized objects are `Prior` instances in `model_config`. They have no
  HSGP, MuEffects, adstock, saturation, or other complex component trees.
- They have **no known serialization failures**.
- They do not use any of the four inconsistent patterns this plan targets
  (no `singledispatch`, no `_MUEFFECT_DESERIALIZERS`, no `RegistrationMeta`,
  no per-family lookup dicts).
- Migrating them adds risk and scope with no user-facing benefit.

**Shared infrastructure constraints:**

- `parse_model_config()` (in `model_config.py`) calls `pymc_extras.deserialize()`
  to reconstruct `Prior` objects from dicts. This flow is used by **all** model
  families and must remain untouched.
- The 5 `register_deserialization` calls in `prior.py` (1 call) and
  `special_priors.py` (4 calls) feed the `pymc_extras` global deserializer
  registry. These calls are **retained** — removing them would break Prior
  deserialization for CLV, Customer Choice, and MMM alike.
- Changes to `_serializable_model_config` and `registry.serialize()` are scoped
  to the **MMM subclass override** in `multidimensional.py`, not the
  `ModelBuilder` base class.

**MVITS note:** MVITS imports `MuEffect` but does not currently serialize
MuEffects (its `mu_effects` defaults to an empty list and is not stored in
idata attrs). If MVITS ever starts serializing MuEffects, it should adopt
the new `TypeRegistry` system at that time.

## Serialization Policy

This section defines what the library guarantees and what extension authors
are responsible for. It serves as a contract for current and future
developers.

### What the library guarantees

1. **Registered types roundtrip correctly.** Any type registered with
   `TypeRegistry` via `@registry.register` or `SerializableMixin` will
   serialize to a JSON-safe dict and deserialize back to an equivalent
   object. This is verified by roundtrip tests for every built-in type.
2. **One mechanism for all types.** The `TypeRegistry` is the single
   dispatch point for serialization and deserialization. There are no
   parallel systems to learn.
3. **Explicit failures.** If deserialization fails (unregistered type,
   corrupt data, missing supplementary data), a `SerializationError` is
   raised with an actionable message. No silent degradation.

### What extension authors must do

1. **Register custom types.** Use `@registry.register` on any class that
   will be saved as part of a model. `MuEffect` subclasses that inherit
   `SerializableMixin` are auto-registered; `Transformation` subclasses
   and other non-mixin classes need the explicit decorator.
2. **Import before load.** The module defining a custom type must be
   imported before calling `load()`. The registry does not dynamically
   import from `__type__` paths.
3. **Ensure `to_dict()` output is self-contained.** The dict returned by
   `to_dict()` must contain everything needed to reconstruct the object
   via `from_dict()`. If auxiliary data (e.g., a DataFrame) is required,
   use the supplementary data mechanism — do not rely on external files
   or side-channel state.

### What is explicitly not supported

1. **Arbitrary live objects.** The library will not attempt to serialize
   PyTensor tensors, PyMC model objects,
   or other non-JSON-safe runtime state. Use `DeferredFactory` to store
   a recipe (factory function + scalar args) instead.
2. **Unregistered types.** If a type is not in the `TypeRegistry`, it
   cannot be deserialized. This is by design — it prevents arbitrary
   code execution from `.nc` files (unlike pickle).
3. **Cross-version object equality.** After a save/load roundtrip,
   `loaded_obj == original_obj` is a goal but not a hard guarantee for
   all types. The guarantee is functional equivalence: the loaded model
   produces the same predictions. Object-level equality depends on each
   class's `__eq__` implementation.

## Solution Overview

The proposed solution addresses both layers of the serialization problem —
the component layer (patterns 1–3: how individual components serialize
themselves) and the orchestrator layer (pattern 4: how the MMM packs those
into idata attrs). The component layer is the priority since that's where
all known failures live; the orchestrator simplification follows
mechanically once components round-trip correctly.

The unified system is built on three pillars:

**1. One registry, one contract.** A `TypeRegistry` singleton (in a new `pymc_marketing/serialization.py` module) replaces all scattered registries, lookup dicts, and `singledispatch` handlers. Every serializable object produces a JSON dict with a `__type__` key containing its fully-qualified class path. The registry dispatches deserialization from that key alone — no more `"class"`, `"hsgp_class"`, `"lookup_name"`, or key-set heuristics. Classes opt in via a `@registry.register` decorator or by inheriting `SerializableMixin` (which auto-registers subclasses). User-defined types follow the same single mechanism.

**2. Deferred factories for non-serializable state.** PyTensor symbolic expressions and live PyMC objects (e.g., HSGP priors, covariance functions) cannot be JSON-encoded. Instead of attempting to serialize them, we store a `DeferredFactory` — a lightweight Pydantic model holding a factory function path and its scalar arguments. The actual object is re-created at `build_model()` time, so non-serializable state is never persisted.

**3. Self-contained model files.** Auxiliary DataFrames that certain components depend on (e.g., `df_events` in `EventAdditiveEffect`) are stored as named groups inside the InferenceData `.nc` file. This makes saved models fully portable — no external files required to reload.

A **migration tool** (`pymc_marketing/migrate.py`) handles backward compatibility by rewriting old-format `.nc` attrs into the new `__type__`-based schema, invoked automatically on load with a deprecation warning.

## Architecture

### 1. Core Serialization Infrastructure

New module: `pymc_marketing/serialization.py`

#### `Serializable` Protocol

```python
from typing import Protocol, runtime_checkable, Any, Self

@runtime_checkable
class Serializable(Protocol):
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation including a __type__ key."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct the object from a dict."""
        ...
```

Structural typing — existing classes with `to_dict()`/`from_dict()` (HSGP, Fourier,
EventEffect, MediaTransformation, SpecialPrior) satisfy it automatically.
Transformation subclasses (adstock, saturation, basis) currently lack `from_dict()` —
their deserialization logic is migrated from standalone functions into classmethods.
No inheritance changes needed.

#### `SerializableMixin`

Auto-implements `Serializable` for Pydantic BaseModel classes. Has three roles:

1. **`to_dict()` / `from_dict()`** — default serialization via `model_dump(mode="json")` / `model_validate()`.
2. **`__init_subclass__()`** — auto-registers every concrete subclass in `TypeRegistry` at class-definition time (no decorator needed). This is a standard Python hook, not a metaclass, so it is compatible with the existing `ABC` + Pydantic `ModelMetaclass` MRO.
3. **Structural `Serializable` compliance** — any class that inherits the mixin satisfies the `Serializable` protocol without additional code (unless it needs custom overrides, e.g. `FourierEffect`, `EventAdditiveEffect`).

```python
class SerializableMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        type_key = f"{cls.__module__}.{cls.__qualname__}"
        registry.register(type_key, cls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            **self.model_dump(mode="json"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data = {k: v for k, v in data.items() if k != "__type__"}
        return cls.model_validate(data)
```

#### `TypeRegistry`

Centralized registry replacing scattered `register_deserialization` calls:

```python
class TypeRegistry:
    def register(self, cls_or_key: type | str | None = None,
                 cls: type | None = None, *,
                 serializer: Callable | None = None,
                 deserializer: Callable | None = None):
        """Register a class. Usable as bare decorator or direct call.

        Decorator form (type_key auto-derived as f"{cls.__module__}.{cls.__qualname__}"):
            @registry.register
            class MyClass: ...

        Direct call (explicit type_key, for custom deserializers):
            registry.register("mod.MyClass", MyClass, deserializer=fn)
        """
        ...

    def serialize(self, obj: Serializable) -> dict[str, Any]: ...

    def deserialize(self, data: dict[str, Any],
                    context: DeserializationContext | None = None) -> Any: ...

registry = TypeRegistry()  # module-level singleton
```

**Type dispatch:** The `__type__` field in every serialized dict uses fully-qualified
module paths (`"pymc_marketing.mmm.hsgp.HSGP"`) to avoid name collisions.
All types must be registered with the `TypeRegistry` before deserialization —
the registry does **not** dynamically import from `__type__` paths (unlike pickle,
this avoids arbitrary code execution from `.nc` files). Backward compatibility with
old serialization formats (e.g., `"lookup_name"`, `"class"`, `"hsgp_class"` keys) is
handled **exclusively** by the migration tool (`migrate.py`) — the `TypeRegistry`
itself only understands `__type__` keys. This keeps the registry simple and the
migration path explicit.

**How built-in types get registered:** Types that gain
`SerializableMixin` (MuEffects, Scaling, LinearTrend, etc.) are
auto-registered in `TypeRegistry` via the mixin. All other MMM-specific
built-in types (adstock, saturation, HSGP, HSGPKwargs, fourier,
MediaTransformation, etc.) have their `to_dict()` updated to include
`__type__` and are registered with `@registry.register`. The 11
MMM-specific `register_deserialization()` calls are removed.

**Prior/SpecialPrior registrations are retained:** The 5
`register_deserialization()` calls in `prior.py` (1) and
`special_priors.py` (4) feed the `pymc_extras` global deserializer
registry. They are **kept as-is** because `parse_model_config()` — used
by all model families (CLV, Customer Choice, MVITS, and MMM) — calls
`pymc_extras.deserialize()` to reconstruct `Prior` objects from dicts.
These types are additionally registered in the `TypeRegistry` via
`@registry.register` so that MMM's new serialization path can also
deserialize them. This dual registration is intentional and safe — the
two registries serve different call sites.

**How user-defined types get registered:** Custom types must use the
`@registry.register` decorator. This is one line per class. If a user creates
a `MyCustomEffect(MuEffect)`, saves a model, then later `load()`s it, the
module defining `MyCustomEffect` must be imported (and thus registered) before
calling `load()`.

**Deserialization dispatch (three-tier):**

1. If `data` contains `"__deferred__": True`, return `DeferredFactory.from_dict(data)`
   immediately — **without** resolving the factory. The caller (e.g., HSGP's
   `create_variable()`) is responsible for calling `.resolve()` later at
   `build_model()` time. This keeps `DeferredFactory` out of the `TypeRegistry`
   mapping and avoids premature creation of non-serializable objects.
2. If a custom `deserializer(data, context)` was registered for the type, call it.
   This is the only path that receives `DeserializationContext` — e.g.,
   `EventAdditiveEffect` registers a deserializer that reads supplementary data
   from `context.idata`.
3. Otherwise, resolve `__type__` to a class and call `cls.from_dict(data)`.
   Context is **not** passed — this keeps the `Serializable` protocol simple and
   avoids `TypeError` on classes with `from_dict(cls, data)` signatures.

```python
def deserialize(self, data: dict[str, Any],
                context: DeserializationContext | None = None) -> Any:
    # Tier 1: deferred factories — return unresolved
    if data.get("__deferred__"):
        return DeferredFactory.from_dict(data)

    # Tier 2: custom deserializer (receives context)
    type_key = data["__type__"]
    entry = self._registry[type_key]
    if entry.deserializer is not None:
        return entry.deserializer(data, context)

    # Tier 3: standard from_dict (no context)
    return entry.cls.from_dict(data)
```

**Auto-registration:** Classes using `@registry.register` decorator or inheriting from
`SerializableMixin` are auto-registered. The mixin uses `__init_subclass__()`
to register each concrete subclass in `TypeRegistry` at class-definition time, which
is compatible with the existing `ABC` + Pydantic `ModelMetaclass` MRO (no additional
metaclass needed). No manual `register_deserialization` calls needed for new code.

#### `DeserializationContext`

Passes runtime state (like InferenceData) to deserializers that need supplementary data:

```python
@dataclass
class DeserializationContext:
    idata: az.InferenceData | None = None
```

### 2. DeferredFactory — Solving the Non-Serializable State Problem

Some objects cannot be JSON-encoded: PyTensor symbolic expressions, live PyMC
covariance objects, and any future state that only makes sense at runtime.
Instead of attempting to serialize these, store a recipe — the factory function
path and its scalar arguments. For example, instead of storing the result of
`create_eta_prior(upper=5.0, mass=0.95)` (a `Prior` with tensor params), store:

```python
class DeferredFactory(BaseModel):
    """Serializable recipe for creating objects with non-serializable state."""

    factory: str            # "pymc_marketing.mmm.hsgp.create_eta_prior"
    kwargs: dict[str, Any]  # {"upper": 5.0, "mass": 0.95}

    def resolve(self) -> Any:
        """Import the factory function and call it with kwargs."""
        fn = import_from_dotted_path(self.factory)
        return fn(**self.kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {"__deferred__": True, "factory": self.factory, "kwargs": self.kwargs}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeferredFactory":
        return cls(factory=data["factory"], kwargs=data["kwargs"])
```

**Integration with HSGP:**

- `HSGP.eta` and `HSGP.ls` fields accept `Prior | DeferredFactory | float`
- `SoftPlusHSGP.parameterize_from_data()` stores `DeferredFactory` instead of the live Prior
- During `build_model()`, a resolution step calls `deferred.resolve()` to produce
  the actual Prior with tensor expressions — right when needed, never stored persistently
- `HSGPKwargs.cov_func` uses the same pattern for custom covariance objects

**Security:** `resolve()` dynamically imports and calls arbitrary dotted paths from data stored in .nc files. This create a security concern similar to pickle.load(). I believe we should document and leave it at that.

**Pydantic integration:** `DeferredFactory` is itself a Pydantic BaseModel, so it
participates in `model_dump(mode="json")`/`model_validate()` naturally. Fields typed as
`Prior | DeferredFactory | float` use a `BeforeValidator` to discriminate:

```python
from pydantic import BeforeValidator
from typing import Annotated

def _maybe_deferred(v: Any) -> Any:
    """If v is a dict with __deferred__, wrap it as DeferredFactory."""
    if isinstance(v, dict) and v.get("__deferred__"):
        return DeferredFactory.from_dict(v)
    return v

PriorOrDeferred = Annotated[
    Prior | DeferredFactory | float,
    BeforeValidator(_maybe_deferred),
]
```

HSGP fields use this annotated type so that `model_validate()` correctly
reconstructs `DeferredFactory` instances from serialized dicts without
confusing them with `Prior` dicts. Resolution to the actual `Prior` (with
tensor expressions) happens later, when the owning class calls
`deferred.resolve()` inside its model-building method (e.g.,
`HSGP.create_variable()`).

### 3. EventAdditiveEffect & Supplementary Data Storage

#### Supplementary Data in InferenceData

ArviZ supports arbitrary named groups. We store auxiliary DataFrames under a
`supplementary_data` convention:

```python
# During save:
idata.supplementary_data = xr.Dataset.from_dataframe(
    effect.df_events.set_index("name")
)
```

**Serialized dict for EventAdditiveEffect:**

```python
{
    "__type__": "pymc_marketing.mmm.additive_effect.EventAdditiveEffect",
    "prefix": "events",
    "reference_date": "2024-01-01",
    "date_dim_name": "date",
    "effect": {... EventEffect.to_dict() ...},       # already works today
    "df_events_group": "supplementary_data/events"    # pointer into idata
}
```

On load, `EventAdditiveEffect` registers a custom deserializer with the `TypeRegistry`
that receives `(data, context)`. It uses `context.idata` to read the DataFrame from
the InferenceData group and reconstruct the full object:

```python
def _deserialize_event_additive_effect(
    data: dict[str, Any], context: DeserializationContext
) -> EventAdditiveEffect:
    group_name = data["df_events_group"]
    df_events = context.idata[group_name].to_dataframe().reset_index()
    effect = EventEffect.from_dict(data["effect"])
    return EventAdditiveEffect(
        df_events=df_events, effect=effect,
        prefix=data["prefix"], reference_date=data["reference_date"],
        date_dim_name=data["date_dim_name"],
    )

registry.register(
    "pymc_marketing.mmm.additive_effect.EventAdditiveEffect",
    EventAdditiveEffect,
    deserializer=_deserialize_event_additive_effect,
)
```

**Naming convention:** Each MuEffect stores supplementary data under
`supplementary_data/{prefix}` to avoid collisions.

#### LinearTrendEffect Fix

Make `linear_trend_first_date` a proper optional Pydantic field:

```python
class LinearTrendEffect(MuEffect):
    linear_trend_first_date: pd.Timestamp | None = Field(default=None, exclude=True)
```

This is cleaner than the current `model_config = {"extra": "allow"}` approach while
remaining functionally equivalent (re-populated in `create_data()` during `build_model()`).

### 4. Registry Consolidation & MMM Save/Load

#### Save Side (`create_idata_attrs`)

Related: [#880](https://github.com/pymc-labs/pymc-marketing/issues/880) — remove unnecessary attrs serialization

1. Each component object (adstock, saturation, HSGP, scaling, mu_effects, etc.) serialized via `registry.serialize(obj)` — one call, consistent output. The MMM subclass's `_serializable_model_config` property in `multidimensional.py` (which currently uses duck-typing with `hasattr(value, "to_dict")` / `hasattr(value, "model_dump")`) is updated to use `registry.serialize()` for consistency. The base `ModelBuilder._json_default` and `ModelBuilder.create_idata_attrs` are **not modified** — CLV, Customer Choice, and MVITS continue to use the existing duck-typing path. Plain JSON-safe scalars (`date_column`, `adstock_first`, `channel_columns`, `dims`, …) remain as direct `json.dumps()` / string assignment.
2. Supplementary data written to idata groups
3. `__serialization_version__` attr added to idata for migration support

#### Load Side (`attrs_to_init_kwargs` + `build_from_idata`)

1. Check `__serialization_version__` — if old format, run migration
2. Each component deserialized via `registry.deserialize(data, context=ctx)`
3. Supplementary data read from idata groups via `DeserializationContext`
4. MuEffect deserialization raises explicit `SerializationError` instead of silently dropping

#### Custom User Types

Users register custom types via decorator:

```python
from pymc_marketing.serialization import registry

@registry.register
class MyCustomEffect(MuEffect):
    my_param: float
    # to_dict/from_dict auto-provided by SerializableMixin on MuEffect

@registry.register
class MyTransform(Transformation):
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "MyTransform": ...
```

### 5. Migration Tool

`migrate_idata(idata) -> InferenceData` function that:

1. Reads `__serialization_version__` attr (absent = v0, the old format)
2. Applies sequential migration steps: v0 -> v1 -> ... -> current
3. Each step rewrites attrs (adding `__type__` keys, renaming `"hsgp_class"`,
   converting MuEffect `"class"` keys to fully-qualified `"__type__"`)
4. Drops the stale `id` attr (the hash changes when `to_dict()` output changes; the next `save()` writes the correct one)
5. Runnable standalone: `python -m pymc_marketing.migrate model.nc`
6. Called automatically by `load()` on old versions with a deprecation warning

## End State Summary

### One Registry

For MMM serialization, exactly **one** active registry: the `TypeRegistry`
singleton in `pymc_marketing/serialization.py`. Of the 16 existing
`register_deserialization()` calls, the **11 MMM-specific calls** (adstock,
saturation, HSGP, HSGPKwargs, fourier ×3, events ×2, media_transformation ×2)
are replaced with `@registry.register` decorators. Every MMM component class's
`to_dict()` is updated to include `__type__`.

The **5 Prior/SpecialPrior calls** in `prior.py` (1) and `special_priors.py` (4)
are **retained** — they feed the `pymc_extras` global deserializer registry that
`parse_model_config()` depends on for all model families (CLV, Customer Choice,
MVITS, and MMM). These types are additionally registered in `TypeRegistry` so
that MMM's new serialization path can also resolve them.


### One Serialization Contract

Every serializable object produces and consumes a JSON dict with a `__type__` key
(fully-qualified class path). The `TypeRegistry` uses `__type__` to dispatch
deserialization — no more `"class"`, `"hsgp_class"`, `"lookup_name"`, key-set
heuristics, or per-family lookup dicts.

### How Each Class Category Serializes

| Category | Classes (examples) | Serialization mechanism | Registration |
|---|---|---|---|
| **Transformation subclasses** (non-Pydantic) | `GeometricAdstock`, `LogisticSaturation`, `GaussianBasis` (~18 classes: 6 adstock, 9 saturation, 3 basis) | `to_dict()` already exists. `from_dict()` classmethod **added** by migrating logic from the standalone `adstock_from_dict()`/`saturation_from_dict()`/`basis_from_dict()` functions into classmethods on `AdstockTransformation`/`SaturationTransformation`/`Basis` respectively (concrete subclasses inherit them). `to_dict()` updated to emit `__type__` instead of `lookup_name`. `RegistrationMeta` metaclass removed; classes use plain inheritance. | Registered with `@registry.register`. |
| **Composite transformations** | `MediaTransformation`, `MediaConfig`, `MediaConfigList` (3 classes) | Existing `to_dict()`/`from_dict()`. `to_dict()` updated to emit `__type__`. | Registered with `@registry.register`. |
| **Pydantic BaseModel with custom `to_dict()`** | `HSGP`, `HSGPPeriodic`, `SoftPlusHSGP`, `FourierBase`/`YearlyFourier`/etc., `EventEffect`, `HSGPKwargs` (~9 classes) | Keep existing `to_dict()`/`from_dict()` (they handle nested `Prior` objects). `to_dict()` updated to include `__type__`. `HSGPKwargs` gets a custom `to_dict()`/`from_dict()` that handles `cov_func` via `DeferredFactory` serialization (the mixin's `model_dump` cannot serialize a live `pm.gp.cov.Covariance`). The `hsgp_class` key currently injected externally in `create_idata_attrs()` is replaced by `__type__` emitted from within `HSGPBase.to_dict()` itself (the external injection in `multidimensional.py` is removed). | Registered with `@registry.register`. |
| **Pure Pydantic BaseModel** | `LinearTrend`, `Scaling`, `VariableScaling` (3 classes) | Gain `SerializableMixin` → auto-provided `to_dict()`/`from_dict()` via `model_dump(mode="json")`/`model_validate()`. | Auto-registered by mixin. |
| **MuEffect subclasses** (Pydantic + ABC) | `FourierEffect`, `LinearTrendEffect`, `EventAdditiveEffect` (3 classes) | `MuEffect` base gains `SerializableMixin` → provides default `to_dict()`/`from_dict()` as fallback (`LinearTrendEffect` uses the default). `FourierEffect` **overrides** `to_dict()`/`from_dict()` to delegate to `FourierBase.to_dict()`/`from_dict()` for subclass dispatch (the mixin's `model_dump` cannot preserve the `FourierBase` subclass discriminator). `EventAdditiveEffect` **overrides** `to_dict()` to store `df_events` as a supplementary-data group pointer and serialize `effect` via `EventEffect.to_dict()` (the mixin's `model_dump` cannot serialize `InstanceOf[pd.DataFrame]`). Replaces the `singledispatch` serializers entirely. | Auto-registered by mixin. `EventAdditiveEffect` additionally registers a custom deserializer for supplementary data. |
| **SpecialPrior hierarchy** (non-Pydantic) | `LogNormalPrior`, `LaplacePrior`, `MaskedPrior` (3 classes) | Existing `to_dict()`/`from_dict()`. `to_dict()` updated to include `__type__`. | Registered with `@registry.register`. |
| **Non-serializable state** | `HSGP.eta`/`ls` (Prior with tensors), `HSGPKwargs.cov_func` | Wrapped in `DeferredFactory` — stores factory function path + scalar args instead of live objects. Resolved lazily at `build_model()` time inside each class's model-building method (e.g., `HSGP.create_variable()`). | `DeferredFactory` is a Pydantic BaseModel; detected by `__deferred__` flag in `TypeRegistry.deserialize()` (tier 1) and via `BeforeValidator` in Pydantic fields. Not registered in `TypeRegistry` as a named type. |
| **User-defined custom types** | `MyCustomEffect(MuEffect)`, `MyTransform(Transformation)` | Implement `to_dict()`/`from_dict()` (manually or inherited from mixin). | Must use `@registry.register` decorator. Module must be imported before `load()`. |

### What Gets Removed

- `singledispatch` `_serialize_mu_effect` and all its registered handlers
- `_MUEFFECT_DESERIALIZERS` dict and `_deserialize_mu_effect` function
- `_register_mu_effect_handlers()` in `multidimensional.py`
- The broad `except Exception` + warning fallback in `build_from_idata()`
- `pymc_marketing/deserialize.py`
- Per-type lookup dicts (`ADSTOCK_TRANSFORMATIONS`, `SATURATION_TRANSFORMATIONS`, `BASIS_TRANSFORMATIONS`)
- `create_registration_meta()` and `RegistrationMeta` metaclass in `base.py`
- `lookup_name` attribute on `Transformation` subclasses (replaced by `__type__`)
- `adstock_from_dict()`, `saturation_from_dict()`, `basis_from_dict()` standalone deserializers
- 11 MMM-specific `register_deserialization()` calls (adstock, saturation, HSGP, HSGPKwargs, fourier ×3, events ×2, media_transformation ×2)

### What Stays Unchanged

- **`prior.py`**: the 1 `register_deserialization()` call for `Prior` is retained (feeds `pymc_extras` global registry used by `parse_model_config` across all model families)
- **`special_priors.py`**: the 4 `register_deserialization()` calls for `LogNormalPrior`, `LaplacePrior`, `MaskedPrior`, `SpecialPrior` are retained (same reason)
- **`model_builder.py`**: `ModelBuilder.create_idata_attrs()`, `_json_default`, `attrs_to_init_kwargs()`, and `_model_config_formatting()` are not modified — CLV, Customer Choice, and MVITS models continue to use the existing serialization path
- **`model_config.py`**: `parse_model_config()` continues to use `pymc_extras.deserialize()` — no changes
- **CLV, Customer Choice, MVITS models**: no serialization changes at all

### What Gets Added

- `pymc_marketing/serialization.py` — `Serializable`, `SerializableMixin`, `TypeRegistry`, `DeferredFactory`, `DeserializationContext`, `SerializationError`
- `pymc_marketing/migrate.py` — version-aware migration for old `.nc` files
- `supplementary_data` groups in InferenceData for auxiliary DataFrames
- `__serialization_version__` attr in saved InferenceData files

## File Layout

```
pymc_marketing/
  serialization.py          # NEW: Serializable, SerializableMixin,
                             #      TypeRegistry, DeferredFactory,
                             #      DeserializationContext, SerializationError
  migrate.py                # NEW: migrate_idata(), version migration steps,
                             #      CLI entry point
  model_builder.py           # MODIFIED: ModelIO uses registry
  deserialize.py             # REMOVED
  special_priors.py          # MODIFIED: 4 register_deserialization calls replaced
                             #           with @registry.register on LogNormalPrior,
                             #           LaplacePrior, MaskedPrior, SpecialPrior
  prior.py                   # MODIFIED: register_deserialization call replaced
                             #           with @registry.register
  mmm/
    multidimensional.py      # MODIFIED: create_idata_attrs/attrs_to_init_kwargs
                             #           simplified, singledispatch removed,
                             #           _serializable_model_config updated to
                             #           use registry.serialize()
    additive_effect.py       # MODIFIED: MuEffect gains SerializableMixin,
                             #           LinearTrendEffect field fix,
                             #           EventAdditiveEffect full serialization
    scaling.py               # MODIFIED: Scaling, VariableScaling gain
                             #           SerializableMixin
    linear_trend.py          # MODIFIED: LinearTrend gains
                             #           SerializableMixin
    hsgp.py                  # MODIFIED: uses DeferredFactory for ls/eta,
                             #           to_dict/from_dict updated
    hsgp_kwargs.py           # MODIFIED: HSGPKwargs gains custom to_dict()
                             #           /from_dict(), cov_func uses
                             #           DeferredFactory
    events.py                # MODIFIED: supplementary data storage,
                             #           BASIS_TRANSFORMATIONS and
                             #           basis_from_dict removed
    fourier.py               # MODIFIED: uses __type__ key
    media_transformation.py  # MODIFIED: register_deserialization calls replaced
                             #           with @registry.register on
                             #           MediaTransformation and MediaConfigList
    components/
      base.py                # MODIFIED: RegistrationMeta, create_registration_meta,
                             #           lookup_name removed; to_dict emits __type__
      adstock.py             # MODIFIED: uses __type__ key, ADSTOCK_TRANSFORMATIONS
                             #           and adstock_from_dict removed
      saturation.py          # MODIFIED: uses __type__ key, SATURATION_TRANSFORMATIONS
                             #           and saturation_from_dict removed
```

## Error Handling

Replace silent `except Exception` + warning with explicit failures:

```python
# Before:
try:
    effect = _deserialize_mu_effect(effect_data)
except Exception as e:
    warnings.warn(f"Could not deserialize mu_effect: {e}")

# After:
effect = registry.deserialize(effect_data, context=ctx)
# Raises SerializationError with actionable message
```


## How the Proposed Design Solves Each Known Issue

1. **EventAdditiveEffect** — The `df_events` DataFrame and the `effect` field are no longer excluded from serialization. `df_events` is stored as a named group (`supplementary_data/events`) inside the InferenceData `.nc` file, and `effect` is serialized via `EventEffect.to_dict()`. A custom deserializer registered with the `TypeRegistry` reads the DataFrame back from `DeserializationContext.idata` on load, fully reconstructing the object.

2. **Custom MuEffects** — The hardcoded `_MUEFFECT_DESERIALIZERS` dict is replaced by the open `TypeRegistry`. `MuEffect` gains `SerializableMixin`, which auto-registers every subclass (including user-defined ones) at class-definition time. Users can also use `@registry.register` explicitly. Any registered type can be deserialized — no more "Unknown MuEffect class".

3. **Default `_serialize_mu_effect` with Prior fields** — The `singledispatch` fallback that calls `model_dump(mode="json")` is removed entirely. All MuEffects serialize through `to_dict()` (provided by `SerializableMixin` or overridden per class), which handles `Prior` and `InstanceOf` fields correctly. No more `PydanticSerializationError` on save.

4. **HSGP/HSGPPeriodic/SoftPlusHSGP `dims` tuple→list** — HSGP classes' `from_dict()` methods are updated to normalize `list→tuple` during deserialization. Since `to_dict()` and `from_dict()` are kept (with `__type__` added), the fix is localized to the `from_dict()` input normalization. The JSON array→list roundtrip is handled explicitly.

5. **HSGPKwargs.cov_func** — `HSGPKwargs` gains a custom `to_dict()`/`from_dict()` that wraps `cov_func` in a `DeferredFactory` (storing the factory function path + scalar args) instead of attempting to serialize the live `pm.gp.cov.Covariance` object. The actual covariance object is re-created lazily at `build_model()` time via `deferred.resolve()`.

6. **LinearTrendEffect** — `linear_trend_first_date` becomes a proper optional Pydantic field (`pd.Timestamp | None = Field(default=None, exclude=True)`) instead of a fragile `model_config={"extra": "allow"}` runtime attribute. `LinearTrendEffect` inherits the mixin's default `to_dict()`/`from_dict()`, which delegates to `model_dump(mode="json")`/`model_validate()` — but `LinearTrend`'s `Prior` fields are now handled correctly because `LinearTrend` also gains `SerializableMixin` with proper `to_dict()` support.

7. **HSGP Prior equality** — HSGP priors that contain PyTensor symbolic expressions (e.g., `lam=-pt.log(mass)/upper`) are replaced by `DeferredFactory` instances that store the factory function and scalar args. After a save/load roundtrip, `deferred.resolve()` re-creates an identical `Prior` with fresh tensor expressions, so the equality problem disappears — there are no stale tensor-vs-float type mismatches to compare.

8. **Silent failures** — The broad `except Exception` catch-all in `build_from_idata()` is removed. All MuEffect deserialization goes through `registry.deserialize()`, which raises an explicit `SerializationError` with an actionable message. If deserialization fails, the user knows immediately.

## Implementation Order

The work is sequenced in two phases matching the two conceptual layers:

**Phase 1 — Component layer (patterns 1–3).** Fix how individual components
serialize themselves. This is where all known failures live and where the
design has the most impact:

1. `pymc_marketing/serialization.py` — `TypeRegistry`, `Serializable` protocol,
   `SerializableMixin`, `DeferredFactory`, `DeserializationContext`, `SerializationError`
2. Transformation subclasses (adstock, saturation, basis) — add `from_dict()`
   classmethods, emit `__type__`, remove `RegistrationMeta`/`lookup_name`,
   register with `@registry.register`
3. HSGP classes — `DeferredFactory` for `eta`/`ls`, `list→tuple` normalization
   in `from_dict()`, `__type__` key, `@registry.register`
4. HSGPKwargs — custom `to_dict()`/`from_dict()` with `DeferredFactory` for `cov_func`
5. Fourier, EventEffect, MediaTransformation — `__type__` key, `@registry.register`
6. MuEffect subclasses — `SerializableMixin` on `MuEffect` base, custom overrides
   for `FourierEffect` and `EventAdditiveEffect` (including supplementary data storage)
7. Remove `singledispatch` serializers, `_MUEFFECT_DESERIALIZERS`, per-type lookup dicts

**Phase 2 — Orchestrator layer (pattern 4).** Once every component can correctly
round-trip via `TypeRegistry`, simplify how the MMM stores and retrieves them:

1. Update `_serializable_model_config` in `multidimensional.py` to use
   `registry.serialize()` instead of duck-typing
2. Simplify `create_idata_attrs()` / `attrs_to_init_kwargs()` — all component
   attrs go through `registry.serialize()`/`registry.deserialize()`
3. Add `__serialization_version__` attr to idata
4. Remove the broad `except Exception` catch-all in `build_from_idata()`
5. `pymc_marketing/migrate.py` — version-aware migration for old `.nc` files

## Testing Strategy

Related: [#988](https://github.com/pymc-labs/pymc-marketing/issues/988) — increase `media_transformation` module coverage for `to_dict`/`from_dict`/equality

- **Unit tests** for `TypeRegistry`, `DeferredFactory`, `SerializableMixin`
- **Round-trip tests** for every serializable component: serialize → deserialize → compare
- **HSGP round-trip**: save model with `SoftPlusHSGP.parameterize_from_data()`, load, verify predictions match
- **EventAdditiveEffect round-trip**: save model with events, load, verify `df_events` intact
- **Migration tests**: load v0 `.nc` file, verify auto-migration produces working model
- **Custom type tests**: register a user-defined MuEffect, save/load, verify round-trip
- **Error tests**: verify `SerializationError` raised with actionable messages for known failure modes

## Tickets Resolved by This Design

The following open tickets will be resolved when this design is fully implemented:

| Ticket | Title | How It's Resolved |
|---|---|---|
| [#2379](https://github.com/pymc-labs/pymc-marketing/issues/2379) | Update the serialization process | Parent tracking issue — the unified `TypeRegistry`, `Serializable` protocol, and `DeferredFactory` replace the 4 inconsistent patterns and fix all 8 known failures listed there. |
| [#1921](https://github.com/pymc-labs/pymc-marketing/issues/1921) | Lack of `idata.attrs` serialization for MuEffects | `MuEffect` gains `SerializableMixin` → all MuEffects (including `EventAdditiveEffect`) are fully serialized to `idata.attrs` via `registry.serialize()`. The broad `except Exception` catch-all is removed. |
| [#2087](https://github.com/pymc-labs/pymc-marketing/issues/2087) | Build out HSGP serialization | HSGP classes get `@registry.register`, `__type__`-based `to_dict()`/`from_dict()`, `DeferredFactory` for tensor priors, and `list→tuple` normalization in `from_dict()`. The external `hsgp_class` key injection is removed. |
| [#880](https://github.com/pymc-labs/pymc-marketing/issues/880) | Remove unnecessary attrs serialization | The MMM subclass's `_serializable_model_config` and `create_idata_attrs` are simplified — all components go through `registry.serialize()` producing clean JSON dicts, eliminating ad-hoc `json.dumps`/`json.loads` workarounds for NetCDF attribute limitations. |
| [#988](https://github.com/pymc-labs/pymc-marketing/issues/988) | Increase `media_transformation` module coverage | Round-trip tests for every serializable component (including `MediaTransformation`, `MediaConfig`, `MediaConfigList`) cover `to_dict`, `from_dict`, and equality. |
| [#1664](https://github.com/pymc-labs/pymc-marketing/issues/1664) | Show off the `to_dict` and `from_dict` structure for our classes | The unified `__type__`-based contract and `@registry.register` decorator provide a single, documentable pattern. Follow-up documentation can reference the new `Serializable` protocol and `TypeRegistry` API. |
