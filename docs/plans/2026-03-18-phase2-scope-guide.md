# Phase 2 Scope Guide — Serialization Overhaul

**Date:** 2026-03-18
**Purpose:** One-page guide for the agent writing the Phase 2 implementation plan.
**Parent design:** [Serialization Overhaul Design](2026-03-05-serialization-overhaul-design.md)

---

## Phase 1 Recap

Phase 1 implemented the **component layer** additively — the new `TypeRegistry`, `SerializableMixin`, `DeferredFactory`, and `@registry.register` decorators were added to all ~40 serializable classes while keeping every piece of old serialization machinery intact. All components now emit `__type__` keys and have `from_dict()` classmethods, but the MMM orchestrator still uses the old code paths to actually save and load models.

Phase 2 completes the overhaul: switch the orchestrator to use the new registry, add base-class hooks, remove all old code, build the migration tool, and clean up tests.

---

## Work Streams

### 1. Orchestrator Switch (`multidimensional.py`)

The MMM's save/load flow in `multidimensional.py` must switch from the old duck-typing/singledispatch paths to `registry.serialize()` / `registry.deserialize()`.

**What to change:**

- `**_serializable_model_config`** — replace duck-typing (`hasattr(value, "to_dict")` / `hasattr(value, "model_dump")`) with `registry.serialize()`.
- `**create_idata_attrs()**` — all component attrs go through `registry.serialize()` producing clean JSON dicts. Add a `__serialization_version__` attr to idata (integer, starting at `1`).
- `**attrs_to_init_kwargs()**` — deserialize component attrs via `registry.deserialize(data, context=ctx)` where `ctx` carries the `InferenceData` for supplementary data access.
- **Supplementary data write** — wire up writing `df_events` (and any future auxiliary DataFrames) as named groups inside InferenceData during `save()`. The EventAdditiveEffect custom deserializer that reads from `context.idata` is already implemented; the write side is missing.
- `**build_from_idata()`** — replace the broad `except Exception` catch-all (line ~3191) with explicit `SerializationError` propagation. Deserialization failures should raise, not silently drop effects.
- **Remove** `_serialize_mu_effect` singledispatch and all its registered handlers.
- **Remove** `_MUEFFECT_DESERIALIZERS` dict, `_deserialize_mu_effect()`, and `_register_mu_effect_handlers()`.
- **Remove** the `_model_config_formatting()` override in the MMM subclass (the base-class version with `__type__` detection replaces it).

### 2. Base-Class Hooks (`model_builder.py`)

Two small, backward-compatible changes to `ModelBuilder` so the `TypeRegistry` is available project-wide.

- `**_json_default()`** — try `registry.serialize(obj)` before the existing duck-typing cascade (`to_dict()` → `model_dump()` → `__dict__` → `str()`). If the type is not registered, fall through unchanged.
- `**_model_config_formatting()**` — if a value is a dict with a `__type__` key, route through `registry.deserialize()` instead of the existing `list→tuple` / `list→np.array` heuristic. Dicts without `__type__` follow the existing path.
- **Tests** — verify the hooks are transparent to non-MMM models (CLV, Customer Choice, MVITS) and that registered types serialize/deserialize correctly through the base-class path.

### 3. Old Code Removal

Phase 1 was additive. Phase 2 removes the legacy machinery that the new registry replaces.


| What to remove                                                                   | Where                                                                                                                                        |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `RegistrationMeta`, `create_registration_meta()`                                 | `components/base.py`                                                                                                                         |
| `lookup_name` attribute on Transformation subclasses                             | `components/base.py`, `adstock.py`, `saturation.py`                                                                                          |
| `ADSTOCK_TRANSFORMATIONS` dict + `adstock_from_dict()`                           | `adstock.py`                                                                                                                                 |
| `SATURATION_TRANSFORMATIONS` dict + `saturation_from_dict()`                     | `saturation.py`                                                                                                                              |
| `BASIS_TRANSFORMATIONS` dict + `basis_from_dict()`                               | `events.py`                                                                                                                                  |
| `AdstockRegistrationMeta`, `SaturationRegistrationMeta`, `BasisMeta`             | `adstock.py`, `saturation.py`, `events.py`                                                                                                   |
| 11 MMM-specific `register_deserialization()` calls                               | `adstock.py` (2), `saturation.py` (2), `hsgp.py` (2), `fourier.py` (4), `events.py` (3), `media_transformation.py` (3), `hsgp_kwargs.py` (2) |
| `hsgp_from_dict()` standalone function                                           | `hsgp.py`                                                                                                                                    |
| `pymc_marketing/deserialize.py` (deprecated shim)                                | top-level                                                                                                                                    |
| `singledispatch` handlers + `_MUEFFECT_DESERIALIZERS` + `_deserialize_mu_effect` | `multidimensional.py`                                                                                                                        |
| `_model_config_formatting()` override                                            | `multidimensional.py`                                                                                                                        |


**Keep:** the 5 `register_deserialization()` calls in `prior.py` (1) and `special_priors.py` (4) — these feed the `pymc_extras` global deserializer used by `parse_model_config()` across all model families.

### 4. Migration Tool (`migrate.py`)

New module `pymc_marketing/migrate.py`:

- `migrate_idata(idata) -> InferenceData` — reads `__serialization_version__` (absent = v0), applies sequential migration steps v0 → v1.
- v0 → v1 migration: rewrite attrs to add `__type__` keys, rename `"hsgp_class"` → `"__type__"`, convert MuEffect `"class"` keys to fully-qualified `"__type__"`, drop stale `id` attr.
- CLI entry point: `python -m pymc_marketing.migrate model.nc`.
- Called automatically by `load()` on old-format files with a deprecation warning.
- Tests: load a v0 `.nc` file fixture, verify migration produces a working model.

### 5. Test Cleanup

**Remove `test_serialization_issues.py`:** This file is explicitly marked "THIS FILE SHOULD NOT BE MERGED TO THE MAIN REPO." It was a diagnostic tool during Phase 1 to verify known bugs exist. Once the orchestrator switch is complete, these issues are resolved by design. Any behaviors worth asserting are already covered by the roundtrip tests.

**Consolidate redundant roundtrip tests:** The following test files each contain a `TestXxxTypeRegistry` class with roundtrip tests that overlap with `test_serialization_roundtrips.py`:


| Component test file            | Overlapping test class                                                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_adstock.py`              | `TestAdstockTypeRegistry`                                                                                                                  |
| `test_saturation.py`           | `TestSaturationTypeRegistry`                                                                                                               |
| `test_hsgp.py`                 | `TestHSGPTypeRegistry`, `TestDeferredFactoryInHSGP`, `TestHSGPKwargsTypeRegistry`                                                          |
| `test_fourier.py`              | `TestFourierTypeRegistry`                                                                                                                  |
| `test_events.py`               | `TestEventsTypeRegistry`                                                                                                                   |
| `test_media_transformation.py` | `TestMediaTransformationTypeRegistry`                                                                                                      |
| `test_additive_effect.py`      | `TestMuEffectTypeRegistry`, `TestLinearTrendEffectSerialization`, `TestFourierEffectSerialization`, `TestEventAdditiveEffectSerialization` |


**Recommended approach:** Keep `test_serialization_roundtrips.py` as the canonical roundtrip suite. Remove the `TestXxxTypeRegistry` classes from per-component test files. Keep only the component-specific assertions in those files (e.g., `to_dict` output includes expected keys, `from_dict` handles legacy formats).

**Add integration tests:** `test_mmm.py` currently has no serialization-related changes. Add end-to-end save/load tests that exercise the full orchestrator path (create model → fit → save → load → verify predictions match).

---

## Success Criteria

1. **Full roundtrip:** A MultidimensionalMMM with all component types (adstock, saturation, HSGP, fourier, events, linear trend, scaling, media transformation) can be saved and loaded with identical predictions.
2. **EventAdditiveEffect roundtrip:** Model with events saves/loads with `df_events` intact in the `.nc` file.
3. **Custom MuEffect roundtrip:** A user-defined MuEffect subclass saves and loads correctly.
4. **Migration:** An old-format `.nc` file (v0) loads successfully with a deprecation warning.
5. **No silent failures:** Deserialization errors raise `SerializationError`, not warnings.
6. **No old code:** All items in the "Old Code Removal" table are gone. `grep` for `RegistrationMeta`, `lookup_name`, `ADSTOCK_TRANSFORMATIONS`, `_serialize_mu_effect`, `_MUEFFECT_DESERIALIZERS` returns zero hits.
7. **Non-MMM models unaffected:** CLV, Customer Choice, and MVITS test suites pass without changes.
8. **No redundant tests:** `test_serialization_issues.py` removed; per-component TypeRegistry test classes consolidated into `test_serialization_roundtrips.py`.

---

## Risk Notes

- **Backward compatibility:** The migration tool is the critical path for users with existing `.nc` files. It must handle all old-format variations (missing `__type__`, `hsgp_class` keys, MuEffect `class` keys, missing `__serialization_version__`).
- **Non-MMM model safety:** The base-class hooks (`_json_default`, `_model_config_formatting`) must be transparent — verify with CLV and Customer Choice test suites. The hooks only activate for TypeRegistry-registered types.
- `**register_deserialization` in `prior.py` / `special_priors.py`:** These 5 calls must stay — `parse_model_config()` depends on them for all model families.
- **Supplementary data naming:** The `supplementary_data/{prefix}` convention must avoid collisions when multiple MuEffects store auxiliary data. Verify with a model containing multiple EventAdditiveEffects.
