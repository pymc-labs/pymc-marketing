# Serialization Overhaul: Scope Options for Team Review

**Date:** 2026-03-13
**Context:** The [serialization overhaul design](2026-03-05-serialization-overhaul-design.md) proposes a `TypeRegistry`-based solution scoped to Multidimensional MMM. Before finalizing, we want to evaluate whether the scope should extend beyond MMM to cover all model families in pymc-marketing.

---

## Table of Contents

- [Background](#background)
- [Current State by Model Family](#current-state-by-model-family)
- [The Four Options](#the-four-options)
  - [Option A: MMM-Only (Current Plan, No Extension)](#option-a-mmm-only-current-plan-no-extension)
  - [Option B: Convention-First with Base-Class Hooks](#option-b-convention-first-with-base-class-hooks)
  - [Option C: Full Unification Now](#option-c-full-unification-now)
  - [Option D: Phased Full Unification (MMM First, Then Everything)](#option-d-phased-full-unification-mmm-first-then-everything)
- [Comparison Matrix](#comparison-matrix)
- [Decision Framework](#decision-framework)
- [Recommendation](#recommendation)

---

## Background

The proposed serialization design introduces three core primitives:

1. **`TypeRegistry`** — a singleton that maps `__type__` keys to classes, replacing scattered `register_deserialization` calls, `RegistrationMeta` metaclasses, `singledispatch` handlers, and lookup dicts.
2. **`SerializableMixin`** — auto-registers Pydantic BaseModel subclasses and provides default `to_dict()`/`from_dict()`.
3. **`DeferredFactory`** — stores a "recipe" (factory function + scalar args) for objects that can't be JSON-encoded (PyTensor expressions, live PyMC objects).

The design targets 7 known serialization failures, all in the MMM family. Before proceeding, we're evaluating: should this pattern extend to CLV, Customer Choice (MNLogit, NestedLogit, MixedLogit), and MVITS?

## Current State by Model Family

### Serialization complexity

| Feature | MMM | CLV (8 models) | CustomerChoice (3 models) | MVITS |
|---|---|---|---|---|
| Complex components (HSGP, adstock, saturation, fourier) | Yes (~30 classes) | No | No | No |
| MuEffect serialization | Yes | No | No | Imports, doesn't serialize |
| `register_deserialization` calls | 11 MMM-specific | 0 | 0 | 0 |
| `RegistrationMeta` metaclass usage | 3 (adstock, saturation, basis) | 0 | 0 | 0 |
| `singledispatch` serializers | Yes | No | No | No |
| Model config contents | Prior + HSGPKwargs + components | Prior only | Prior only | Prior only |
| Known serialization failures | 7 | 0 | 0 | 0 |

### How each family serializes model_config today

| Family | `_serializable_model_config` | Deserialization path |
|---|---|---|
| **MMM** (multidimensional) | Recursive `serialize_value()`: tries `to_dict()`, `model_dump()`, identity | `attrs_to_init_kwargs` → per-type deserializers (`adstock_from_dict`, `hsgp_from_dict`, etc.) + `parse_model_config()` for Priors |
| **MMM** (legacy) | `ndarray_to_list()` recursive conversion | `_model_config_formatting()` heuristic + `parse_model_config()` |
| **CLV** | Returns `self.model_config` directly (relies on `_json_default` for Prior serialization) | Base `_model_config_formatting()` heuristic + `parse_model_config()` |
| **CustomerChoice** | Manual `Prior.to_dict()` calls per key (4 implementations, 5–15 lines each) | Base `_model_config_formatting()` heuristic + `parse_model_config()` |
| **MVITS** | Manual `Prior.to_dict()` calls per key | Custom `attrs_to_init_kwargs` + `parse_model_config()` |

### Shared infrastructure (used by ALL families)

- **`parse_model_config()`** — calls `pymc_extras.deserialize()` to reconstruct `Prior` objects from dicts. Used by every model's `__init__`.
- **`_json_default()`** — duck-typing cascade in `ModelBuilder.create_idata_attrs()`: tries `to_dict()` → `model_dump()` → `__dict__` → `str()`.
- **`_model_config_formatting()`** — post-JSON heuristic: converts lists to tuples for `dims` keys, lists to numpy arrays for everything else. Fragile; operates on key names, not types.
- **5 `register_deserialization` calls** for `Prior`/`SpecialPrior` (in `prior.py` and `special_priors.py`) — feed the `pymc_extras` global registry that `parse_model_config()` depends on.

---

## The Four Options

### Option A: MMM-Only (Current Plan, No Extension)

**Summary:** Ship the proposed design exactly as specified. `TypeRegistry` lives in `serialization.py`, used exclusively by MMM code. `ModelBuilder` base class is not modified. Non-MMM models are completely untouched.

**What changes:**
- New `pymc_marketing/serialization.py` module (TypeRegistry, SerializableMixin, DeferredFactory)
- New `pymc_marketing/migrate.py` module
- MMM component classes get `@registry.register` / `SerializableMixin`
- `multidimensional.py`: `_serializable_model_config`, `create_idata_attrs`, `attrs_to_init_kwargs` use registry
- 11 MMM-specific `register_deserialization` calls removed
- 5 Prior/SpecialPrior `register_deserialization` calls retained

**What does NOT change:**
- `ModelBuilder._json_default()` — duck-typing cascade remains
- `ModelBuilder._model_config_formatting()` — fragile heuristic remains
- `ModelBuilder.create_idata_attrs()` — unchanged
- `ModelBuilder.attrs_to_init_kwargs()` — unchanged
- All CLV, CustomerChoice, MVITS code — zero modifications
- `parse_model_config()` — unchanged

**Pros:**
- Smallest scope — focused entirely on fixing known failures
- Zero risk to non-MMM models
- Fastest to ship
- No migration needed for non-MMM saved models

**Cons:**
- No future-proofing for non-MMM families: a developer creating a custom CLV component wouldn't naturally discover or use TypeRegistry
- The duck-typing cascade and fragile `_model_config_formatting` remain as the primary mechanism for non-MMM models
- TypeRegistry exists as an MMM-only pattern — new contributors may not realize it's intended as the universal approach
- If MVITS starts serializing MuEffects, it would need a separate effort to adopt TypeRegistry

---

### Option B: Convention-First with Base-Class Hooks

**Summary:** Ship the MMM overhaul as proposed. Additionally, make `ModelBuilder` base class **TypeRegistry-aware** with two small, backward-compatible changes. Non-MMM models continue working identically but gain a natural path to TypeRegistry.

**What changes (beyond Option A):**

1. **`_json_default()` tries registry first:**
```python
def _json_default(obj):
    from pymc_marketing.serialization import registry
    try:
        return registry.serialize(obj)
    except (KeyError, TypeError):
        pass
    # existing duck-typing cascade follows unchanged
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    ...
```

2. **`_model_config_formatting()` checks for `__type__` keys:**
```python
@classmethod
def _model_config_formatting(cls, model_config: dict) -> dict:
    from pymc_marketing.serialization import registry
    for key, value in model_config.items():
        if isinstance(value, dict) and "__type__" in value:
            model_config[key] = registry.deserialize(value)
        elif isinstance(value, dict):
            # existing heuristic for legacy format
            for sub_key in value:
                if isinstance(value[sub_key], list):
                    if sub_key == "dims":
                        value[sub_key] = tuple(value[sub_key])
                    else:
                        value[sub_key] = np.array(value[sub_key])
    return model_config
```

**What does NOT change:**
- All CLV, CustomerChoice, MVITS model code — zero modifications
- `parse_model_config()` — unchanged
- Non-MMM save/load behavior — identical output, identical loading
- No migration needed for non-MMM saved models

**How it enables future adoption:** When a developer creates a new component for any model family and registers it with `@registry.register`, it automatically serializes correctly through the base `_json_default` (which now tries the registry first) and deserializes correctly through `_model_config_formatting` (which recognizes `__type__` keys). No overrides needed.

**Pros:**
- Very low incremental cost over Option A (~10 lines changed in `model_builder.py`)
- Future-proofs all model families — single pattern for new code
- Zero behavioral change for existing models (registry-first check only activates for registered types)
- Non-MMM models can adopt TypeRegistry incrementally (one class at a time, no big bang)
- New contributors see TypeRegistry as the project-wide standard, not an MMM-only pattern

**Cons:**
- Two systems coexist: TypeRegistry (new) + duck-typing fallback (legacy). The fallback may persist indefinitely since non-MMM models have no motivation to migrate.
- Subtle ordering dependency: `_json_default` tries registry before `to_dict()`, which could theoretically change behavior if a class is registered in TypeRegistry but also has a different `to_dict()` (unlikely in practice — the registry calls `to_dict()` internally).
- The `_model_config_formatting` heuristic remains for non-`__type__` dicts.

---

### Option C: Full Unification Now

**Summary:** Migrate all model families to TypeRegistry in one release. The duck-typing cascade and `_model_config_formatting` heuristic are completely removed. Every serializable value in every model goes through the registry.

**What changes (beyond Option A):**

| Change | Scope | Effort |
|---|---|---|
| Replace `_json_default()` in `ModelBuilder` with `registry.serialize()` | Base class | Small |
| Remove `_model_config_formatting()` entirely | Base class + 2 MMM overrides | Medium |
| Update `ModelBuilder.attrs_to_init_kwargs()` to use `registry.deserialize()` | Base class | Medium |
| Delete 4 CustomerChoice `_serializable_model_config` overrides (MNLogit, NestedLogit, MixedLogit, MVITS) | 4 files, ~50 lines removed | Small |
| CLV `_serializable_model_config` passthrough unchanged (but `_json_default` now routes through registry) | No change | None |
| Register `Prior` in TypeRegistry with wrapper around `pymc_extras.deserialize()` | `serialization.py` | Small |
| Update `parse_model_config()` to try TypeRegistry before `pymc_extras.deserialize()` | `model_config.py` | Medium |
| Add `__serialization_version__` to ALL saved models | All families | Small |
| Extend migration tool to handle CLV/CustomerChoice old format | `migrate.py` | Medium |
| Tests: roundtrip save/load for all 15 non-MMM model classes | Tests | Large |

**Non-MMM models affected:** 8 CLV models, 3 CustomerChoice models, 1 MVITS model (12 total).

**Files modified beyond Option A:**
- `model_builder.py` — `_json_default`, `_model_config_formatting`, `attrs_to_init_kwargs`
- `model_config.py` — `parse_model_config` updated
- `customer_choice/mnl_logit.py` — `_serializable_model_config` simplified
- `customer_choice/nested_logit.py` — `_serializable_model_config` simplified
- `customer_choice/mixed_logit.py` — `_serializable_model_config` simplified
- `customer_choice/mv_its.py` — `_serializable_model_config` simplified
- `migrate.py` — migration rules for non-MMM formats

**Model ID impact:** Adding `__type__` keys to serialized Prior dicts changes the JSON output of `_serializable_model_config`, which changes the SHA256 hash used as the model `id`. All previously saved models from all families would produce a different `id` on load, triggering a mismatch warning. The migration tool must either recompute the id or suppress the check for migrated models.

**Pros:**
- True unification — one serialization path for all models
- Eliminates the duck-typing cascade and `_model_config_formatting` heuristic entirely
- No legacy code paths to maintain
- Simplest mental model for contributors: "everything uses TypeRegistry"

**Cons:**
- Large blast radius — 12 non-MMM models touched with no user-facing benefit (they have zero serialization bugs)
- `parse_model_config()` changes affect all models simultaneously — a subtle bug here is a project-wide regression
- Model ID hash changes for all families — every existing saved model needs migration or the id check must be handled
- Migration tool scope roughly doubles (must handle CLV + CustomerChoice old format `.nc` files)
- Testing surface area: need roundtrip tests for all 15 non-MMM model classes, multiplied by old-format and new-format
- Risk of introducing regressions in models that currently work perfectly

---

### Option D: Phased Full Unification (MMM First, Then Everything)

**Summary:** Execute in two phases. Phase 1 is the approved MMM overhaul (same as Option A). Phase 2, in a later release, migrates all remaining model families to TypeRegistry (the delta between Option A and Option C).

**Phase 1 (same as Option A):**
- Ship TypeRegistry, SerializableMixin, DeferredFactory
- Fix the 8 MMM serialization failures
- MMM uses TypeRegistry exclusively
- Non-MMM models completely untouched

**Phase 2 (later release):**
- Replace `_json_default` with registry dispatch
- Remove `_model_config_formatting` heuristic
- Update `parse_model_config()` to use TypeRegistry
- Simplify/delete non-MMM `_serializable_model_config` overrides
- Add `__serialization_version__` to non-MMM models
- Extend migration tool for non-MMM formats
- Full roundtrip tests for all 12 non-MMM models

**What happens between phases:** Two serialization systems coexist. MMM uses TypeRegistry; everything else uses the legacy duck-typing path. This is identical to Option A's end state, but with a commitment to eventually reach Option C's end state.

**Pros:**
- MMM fixes ship without delay
- Phase 2 benefits from real-world experience with TypeRegistry in MMM
- Smaller PRs, easier to review
- If Phase 2 reveals unexpected issues, it can be adjusted without blocking MMM fixes
- Clear roadmap communicates intent to the team

**Cons:**
- `ModelBuilder` base class may need modification twice: once for Phase 1 (to not break non-MMM models while MMM uses the new system), and again for Phase 2 (to make base class use TypeRegistry). This increases the total effort compared to doing it once.
- Phase 2 has the same low ROI as Option C — it's removing working code and re-testing things that pass. The phasing doesn't change the cost/benefit math.
- Classic "Phase 2" risk: if Phase 2 never ships (because non-MMM models never develop serialization problems), you're left with Option A's end state — which works fine, but the planned unification remains incomplete.
- Two rounds of migration tool updates, two rounds of user communication about format changes.
- Model ID hash changes still apply in Phase 2 — users of CLV/CustomerChoice models would face migration warnings in a later release.

---

## Comparison Matrix

| Dimension | A: MMM-Only | B: Convention-First | C: Full Unification | D: Phased Unification |
|---|---|---|---|---|
| **Scope** | MMM only | MMM + ~10 lines in base | All 16 models | MMM now, all 16 later |
| **MMM fixes shipped** | Yes | Yes | Yes | Yes (Phase 1) |
| **Non-MMM risk** | None | Negligible | Medium-High | Medium-High (Phase 2) |
| **Future-proofing** | None | High | Complete | Complete (eventually) |
| **Incremental cost over A** | — | ~10 lines | ~200 lines + tests | ~200 lines + tests (deferred) |
| **Migration tool scope** | MMM only | MMM only | All families | MMM now, all later |
| **Model ID impact** | MMM only | MMM only | All families | MMM now, all later |
| **Duck-typing cascade removed** | No | No (but bypassed for registered types) | Yes | Yes (Phase 2) |
| **`_model_config_formatting` removed** | No | No (but bypassed for `__type__` dicts) | Yes | Yes (Phase 2) |
| **`parse_model_config` modified** | No | No | Yes | Yes (Phase 2) |
| **Time to ship MMM fixes** | Fastest | Fastest | Slower | Fastest (Phase 1) |
| **Total engineering effort** | Lowest | Low | High | Highest (two rounds) |
| **Legacy code remaining** | Most | Some (fallbacks) | None | None (eventually) |

---

## Decision Framework

The right option depends on which of these you value most. Use this framework to guide the team discussion:

### 1. "Do non-MMM models have serialization problems today?"

**No.** CLV, CustomerChoice, and MVITS have zero known serialization failures. Their only serialized components are `Prior` objects, which round-trip correctly through `pymc_extras.deserialize()`.

- If this is decisive → **Option A** (don't fix what isn't broken)
- If this is necessary but not sufficient → continue to question 2

### 2. "How likely is it that non-MMM models will gain complex serializable components?"

**CLV:** Low. The model architecture is mature — model_config contains only `Prior` objects and covariate column lists. No planned features require component trees.

**CustomerChoice:** Low-Medium. The architecture is stable, but `MixedLogit` has some complexity (random coefficients, instrumental variables) that could evolve.

**MVITS:** Medium. Already imports `MuEffect` but doesn't serialize it. If time-varying parameters or additive effects are added, it would need TypeRegistry support.

- If you believe complex components are coming soon → **Option C or D**
- If this is speculative → continue to question 3

### 3. "How important is a single pattern for contributor onboarding?"

If new contributors should learn *one* serialization approach that works everywhere, the duck-typing fallback is confusing — it's implicit, undocumented, and produces output without `__type__` tags.

- If one-pattern-everywhere matters but migration risk is unacceptable → **Option B**
- If one-pattern-everywhere matters and you're willing to accept migration scope → **Option C or D**

### 4. "Is the 'Phase 2' commitment credible?"

Option D only differs from Option A if Phase 2 actually ships. If non-MMM models never develop serialization problems, Phase 2 has no user-facing motivation.

- If Phase 2 would be deprioritized indefinitely → **Option B** (you get the future-proofing hooks now, no Phase 2 needed)
- If the team will commit to Phase 2 on a roadmap → **Option D**

---

## Recommendation

**Option B (Convention-First with Base-Class Hooks)** provides the best value-to-risk ratio:

1. It solves the stated goal (future-proofing: single pattern for all new code) at near-zero incremental cost over Option A (~10 lines in `model_builder.py`).
2. It carries no risk to non-MMM models (the base-class hooks only activate for TypeRegistry-registered types; unregistered types follow the existing path unchanged).
3. It doesn't require a "Phase 2" commitment — the hooks are already in place, so any model family can adopt TypeRegistry incrementally whenever there's a reason to.
4. It avoids the model ID hash change for non-MMM models (their serialized output is unchanged).

Options C and D are appropriate only if the team has a concrete, near-term reason to unify non-MMM serialization (e.g., planned features for CLV/CustomerChoice that introduce complex components). Without that motivation, the engineering cost and migration risk outweigh the benefit of removing legacy code that works correctly.

Option A is appropriate if the team decides that future-proofing is not a priority and prefers to keep the MMM overhaul maximally scoped. The 10-line delta between A and B is small enough that this distinction is minor.
