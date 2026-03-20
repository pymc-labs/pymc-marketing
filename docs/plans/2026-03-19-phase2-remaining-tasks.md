# Phase 2: Serialization Overhaul — Remaining Tasks (7–12)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the serialization overhaul by removing all legacy serialization code, consolidating tests, and running final verification.

**Architecture:** Phase 1 added the `TypeRegistry`, `SerializableMixin`, and `DeferredFactory`, and updated all ~40 serializable classes to emit `__type__` keys and have `from_dict()` classmethods. Phase 2 wires the MMM orchestrator to use this new system for save/load, replaces silent failures with explicit errors, provides backward compatibility via a migration tool, and removes all old code paths.

**Tech Stack:** Python 3.10+, PyMC, ArviZ (InferenceData), Pydantic v2, xarray, pytest

**Parent design:** [Serialization Overhaul Design](2026-03-05-serialization-overhaul-design.md)
**Phase 1 recap:** [Phase 2 Scope Guide](2026-03-18-phase2-scope-guide.md)
**Full original plan:** [Phase 2 Full Plan](2026-03-18-phase2-orchestrator-migration-cleanup.md)

**Python executable:** `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python`

---

> ### Important: `MMM` Class Locations
>
> There are **two** `MMM` classes in the codebase:
> - **Old/deprecated:** `pymc_marketing.mmm.mmm.MMM` (aka `BaseMMM`) — tested by `tests/mmm/test_mmm.py` using the `mmm_fitted` fixture.
> - **Current/multidimensional:** `pymc_marketing.mmm.multidimensional.MMM` — tested by `tests/mmm/test_multidimensional.py` using the `simple_fitted_mmm` fixture from `tests/mmm/conftest.py`.
>
> All orchestrator serialization code lives in `multidimensional.py`. **New serialization tests must use the multidimensional `MMM`** via the `simple_fitted_mmm` fixture (or similar fixtures from `tests/mmm/conftest.py`), not the `mmm_fitted` fixture from `test_mmm.py`.

---

## Summary of Completed Tasks (1–6)

Tasks 1–6 have been implemented and committed. Here is what was done and key learnings:

### Task 1: Base-Class Hooks — `_json_default()` registry-first dispatch ✅

Updated `_json_default` in `pymc_marketing/model_builder.py` (`create_idata_attrs`) to try `registry.serialize()` before the duck-typing fallback (`to_dict()`, `model_dump()`, `__dict__`). Tests added in `tests/test_model_builder.py` (`TestJsonDefaultRegistryDispatch`).

### Task 2: Base-Class Hooks — `_model_config_formatting()` `__type__` detection ✅

Updated `_model_config_formatting` in `pymc_marketing/model_builder.py` to check for `__type__` keys and route through `registry.deserialize()`, falling back to the legacy list→tuple/array heuristic for plain dicts. Tests added in `tests/test_model_builder.py` (`TestModelConfigFormattingTypeDetection`).

### Task 3: Orchestrator Save Side ✅

Replaced duck-typing with `registry.serialize()` in `_serializable_model_config` and `create_idata_attrs` in `multidimensional.py`. Added `__serialization_version__ = "1"` attr. **Critical learning:** save-side and load-side must be updated together — dual-format support was added to the load side in the same task. Tests in `tests/mmm/test_multidimensional.py` (`TestSerializationVersion`).

### Task 4: Supplementary Data Write — `df_events` into InferenceData ✅

Overrode `save()` on `MMM` in `multidimensional.py` to write `EventAdditiveEffect.df_events` as supplementary xarray groups in InferenceData before calling `super().save()`. Tests in `tests/mmm/test_multidimensional.py` (`TestSupplementaryDataWrite`).

### Task 5: Orchestrator Load Side — `attrs_to_init_kwargs` with registry ✅

`attrs_to_init_kwargs` and `build_from_idata` now have dual-format support: `__type__` dicts go through `registry.deserialize()`, legacy dicts fall back to `adstock_from_dict()` / `saturation_from_dict()` / `hsgp_from_dict()`. The `except Exception` catch-all in `build_from_idata` was removed; deserialization failures now propagate as `SerializationError`. Tests in `tests/mmm/test_multidimensional.py` (`TestRegistryDeserialization`).

### Task 6: Migration Tool — `serialization_migration.py` ✅

Created `pymc_marketing/serialization_migration.py` with `migrate_idata()` that converts v0 attrs to v1 format (rewrites `lookup_name` → `__type__`, `hsgp_class` → `__type__`, mu_effect `class` → `__type__`, drops stale `id`, sets `__serialization_version__`). Includes CLI entry point (`python -m pymc_marketing.serialization_migration`). Auto-migration wired into `load_from_idata()` on the MMM class with a `DeprecationWarning`. Tests in `tests/test_serialization_migration.py`.

### Key Learnings from Tasks 1–6

1. **Correct `MMM` class for tests:** `tests/mmm/test_mmm.py` imports the old deprecated `MMM`. All orchestrator serialization code is in `multidimensional.py`, so new serialization tests must use `tests/mmm/test_multidimensional.py` with the `simple_fitted_mmm` fixture.

2. **Save/load sides must be updated together:** Changing the save-side format immediately breaks existing roundtrip tests if the load-side still expects old keys. Dual-format support on the load side must be added in the same task.

3. **Phase 1's `to_dict()` changes propagate:** Phase 1 added `__type__` to `to_dict()` output and `from_dict()` converts `dims` to tuples. Tests that compare `to_dict()` output with strict equality may need updating.

4. **`build_from_idata` still has legacy fallback branches** (`else adstock_from_dict(...)`, etc.) — these are kept temporarily for dual-format support. Task 7 removes them now that the migration tool is in place.

5. **`_model_config_formatting()` override exists in `multidimensional.py`** — the base-class version (Task 2) handles `__type__` detection, but the override remains for now. Task 7 removes it.

---

## Task 7: Remove Old MuEffect Machinery from `multidimensional.py`

**Files:**
- Modify: `pymc_marketing/mmm/multidimensional.py`
- Test: existing tests

Once the orchestrator uses `registry.serialize()`/`registry.deserialize()` (Tasks 3–5), the old singledispatch and manual deserializer machinery can be removed.

> **Current state after Batch 1:** `attrs_to_init_kwargs` and `build_from_idata` already have dual-format support (registry for `__type__`, legacy for old format). The save side is fully migrated. Once the migration tool (Task 6) is in place, all loaded data will have `__type__` keys, so the legacy fallbacks can be safely removed.

**Step 1: Remove the following from `multidimensional.py`**

| What | Lines (approx.) | Description |
|------|-----------------|-------------|
| `_serialize_mu_effect` singledispatch | 236–255 | Default handler |
| `_deserialize_mu_effect` | 258–266 | Manual deserializer |
| `_MUEFFECT_DESERIALIZERS` | 270 | Empty dict |
| `_register_mu_effect_handlers()` | 275–371 | Registers 3 serializers + 3 deserializers |
| `_register_mu_effect_handlers()` call | 371 | Module-level invocation |

Also remove:
- The `_model_config_formatting()` override (lines ~1078–1110) — the base-class version (updated in Task 2) now handles `__type__` detection plus the legacy heuristic. **Important:** The base-class version must also handle the `format_nested_dict` logic (list→tuple for `dims`, list→np.array otherwise). Verify the base-class version covers this before removing the override.
- Legacy fallbacks in `attrs_to_init_kwargs()` — remove the `else adstock_from_dict(...)` / `saturation_from_dict(...)` / `hsgp_from_dict(...)` branches. Since the migration tool converts v0→v1 before deserialization, all data will have `__type__` keys.
- Legacy fallback in `build_from_idata()` — remove the `else _deserialize_mu_effect(effect_data)` branch and the `except Exception` catch-all (if not already done in Task 5).

**Step 2: Update imports**

Remove unused imports from the top of `multidimensional.py`:
- `from functools import singledispatch` (if no other singledispatch remains)
- `from pymc_marketing.mmm.components.adstock import adstock_from_dict`
- `from pymc_marketing.mmm.components.saturation import saturation_from_dict`
- `from pymc_marketing.mmm.hsgp import hsgp_from_dict`

**Step 3: Simplify `attrs_to_init_kwargs`**

Now that migration handles v0→v1 conversion, all components have `__type__` keys:

```python
    @classmethod
    def attrs_to_init_kwargs(cls, attrs: dict[str, str]) -> dict[str, Any]:
        """Convert the idata attributes to the model initialization kwargs."""
        from pymc_marketing.serialization import registry

        def _deser(raw: str, fallback=None):
            data = json.loads(raw)
            if isinstance(data, dict) and "__type__" in data:
                return registry.deserialize(data)
            return fallback if fallback is not None else data

        tvi_raw = attrs.get("time_varying_intercept", "false")
        tvm_raw = attrs.get("time_varying_media", "false")
        tvi_data = json.loads(tvi_raw)
        tvm_data = json.loads(tvm_raw)

        return {
            "model_config": cls._model_config_formatting(
                json.loads(attrs["model_config"])
            ),
            "date_column": attrs["date_column"],
            "control_columns": json.loads(attrs["control_columns"]),
            "channel_columns": json.loads(attrs["channel_columns"]),
            "adstock": _deser(attrs["adstock"]),
            "saturation": _deser(attrs["saturation"]),
            "adstock_first": json.loads(attrs.get("adstock_first", "true")),
            "yearly_seasonality": json.loads(attrs["yearly_seasonality"]),
            "time_varying_intercept": (
                registry.deserialize(tvi_data)
                if isinstance(tvi_data, dict) and "__type__" in tvi_data
                else tvi_data
            ),
            "target_column": attrs["target_column"],
            "time_varying_media": (
                registry.deserialize(tvm_data)
                if isinstance(tvm_data, dict) and "__type__" in tvm_data
                else tvm_data
            ),
            "sampler_config": json.loads(attrs["sampler_config"]),
            "dims": tuple(json.loads(attrs.get("dims", "[]"))),
            "scaling": _deser(attrs.get("scaling", "null")),
            "dag": json.loads(attrs.get("dag", "null")),
            "treatment_nodes": json.loads(attrs.get("treatment_nodes", "null")),
            "outcome_node": json.loads(attrs.get("outcome_node", "null")),
            "cost_per_unit": (
                _deserialize_cost_per_unit(attrs["cost_per_unit"])
                if attrs.get("cost_per_unit") and attrs["cost_per_unit"] != "null"
                else None
            ),
        }
```

**Step 4: Simplify `build_from_idata`**

```python
    def build_from_idata(self, idata: az.InferenceData) -> None:
        """Rebuild the model from an InferenceData object."""
        from pymc_marketing.serialization import DeserializationContext, registry

        if "mu_effects" in idata.attrs:
            mu_effects_data = json.loads(idata.attrs["mu_effects"])
            ctx = DeserializationContext(idata=idata)
            self.mu_effects = [
                registry.deserialize(effect_data, context=ctx)
                for effect_data in mu_effects_data
            ]

        dataset = idata.fit_data.to_dataframe()
        if isinstance(dataset.index, (pd.MultiIndex, pd.DatetimeIndex)):
            dataset = dataset.reset_index()
        X = dataset.drop(columns=[self.target_column])
        y = dataset[self.target_column]
        self.build_model(X, y)
```

**Step 5: Run tests**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_mmm.py -x -q --timeout=300`
Expected: PASS

**Step 6: Commit**

```bash
git add pymc_marketing/mmm/multidimensional.py
git commit -m "refactor(serialization): remove singledispatch, MUEFFECT_DESERIALIZERS, and _model_config_formatting override"
```

---

## Task 8 — Legacy MMM Compatibility Goal

> **Goal:** Throughout Tasks 8a and 8b, the deprecated `pymc_marketing.mmm.mmm.MMM` class (Legacy MMM) must continue to work — including its save/load path. The Legacy MMM is deprecated and scheduled for removal in 0.20.0, but it must not break while it exists.
>
> **Why this matters:** The Legacy MMM's `attrs_to_init_kwargs` calls `adstock_from_dict()` and `saturation_from_dict()`, which depend on the `ADSTOCK_TRANSFORMATIONS` / `SATURATION_TRANSFORMATIONS` lookup dicts. These dicts are currently populated by the metaclasses we're removing. Additionally, the Legacy MMM does **not** have the auto-migration wired into its `load_from_idata` — only the multidimensional MMM does. So we must keep the minimal machinery that the Legacy MMM's load path relies on.
>
> **Strategy (Option 1 — minimal carve-out):** Keep `adstock_from_dict()`, `saturation_from_dict()`, and their backing lookup dicts alive but deprecated with `FutureWarning`. Replace the metaclass-populated dicts with hardcoded dicts. Remove everything else (metaclasses, `RegistrationMeta`, `create_registration_meta`, `basis_from_dict`, `hsgp_from_dict`, all `register_deserialization` calls, etc.).

> **Critical (applies to both 8a and 8b):** Do NOT remove the 5 `register_deserialization` calls in `pymc_marketing/prior.py` (1) and `pymc_marketing/special_priors.py` (4). These feed the `pymc_extras` global deserializer used by `parse_model_config()` across all model families.

---

## Task 8a: Remove Legacy Infrastructure — Base Metaclass, Events, HSGP, Fourier, MediaTransformation, HSGPKwargs

This sub-task removes legacy code from files that the Legacy MMM does **not** depend on.

**Files:**
- Modify: `pymc_marketing/mmm/components/base.py` — remove `RegistrationMeta`, `create_registration_meta`, `DuplicatedTransformationError`
- Modify: `pymc_marketing/mmm/events.py` — remove `BASIS_TRANSFORMATIONS`, `BasisMeta`, `basis_from_dict`, `_is_basis`, `register_deserialization` calls
- Modify: `pymc_marketing/mmm/hsgp.py` — remove `hsgp_from_dict`, `_is_hsgp`, `register_deserialization` call
- Modify: `pymc_marketing/mmm/fourier.py` — remove 3 `register_deserialization` calls + `_is_*` guards
- Modify: `pymc_marketing/mmm/media_transformation.py` — remove 2 `register_deserialization` calls + `_is_*` guards
- Modify: `pymc_marketing/mmm/hsgp_kwargs.py` — remove `register_deserialization` call + `_is_hsgp_kwargs`
- Test: run all mmm tests

**Step 1: Remove `RegistrationMeta` infrastructure from `base.py`**

In `pymc_marketing/mmm/components/base.py`:
- Delete `DuplicatedTransformationError` class (lines 685–691)
- Delete `create_registration_meta()` function (lines 694–726)
- Remove `lookup_name` check in `_has_all_attributes()` (lines 307–308)

> **Do NOT remove yet:** `lookup_name: str` from the `Transformation` class (line 130) or from `to_dict()` output (line 180). These are removed in Task 8b after the adstock/saturation metaclasses are replaced.

**Step 2: Remove metaclass and lookup dict from `events.py`**

In `pymc_marketing/mmm/events.py`:
- Delete `BASIS_TRANSFORMATIONS` dict (line 120)
- Delete `BasisMeta` (line 121)
- Change `Basis` metaclass (line 124) — remove `metaclass=BasisMeta`
- Delete `basis_from_dict()`, `_is_basis()`, and the basis `register_deserialization()` call (lines 188–208)
- Delete the EventEffect `register_deserialization()` call (lines 276–279) plus its `_is_event_effect()` guard
- Remove unused `register_deserialization` import

**Step 3: Remove `hsgp_from_dict` from `hsgp.py`**

In `pymc_marketing/mmm/hsgp.py`:
- Delete `hsgp_from_dict()` function (lines 1488–1502)
- Delete `_is_hsgp()` function (lines 1504–1509)
- Delete `register_deserialization(...)` call (line 1512)
- Remove unused `register_deserialization` import

**Step 4: Remove `register_deserialization` calls from `fourier.py`**

In `pymc_marketing/mmm/fourier.py`:
- Delete 3 `register_deserialization(...)` calls (lines 1011–1023) and their `_is_*` guard functions
- Remove unused `register_deserialization` import

**Step 5: Remove `register_deserialization` calls from `media_transformation.py`**

In `pymc_marketing/mmm/media_transformation.py`:
- Delete 2 `register_deserialization(...)` calls (lines 279–282, 545–548) and their `_is_*` guard functions
- Remove unused `register_deserialization` import

**Step 6: Remove `register_deserialization` call from `hsgp_kwargs.py`**

In `pymc_marketing/mmm/hsgp_kwargs.py`:
- Delete `register_deserialization(...)` call (lines 122–125) and its `_is_hsgp_kwargs` guard
- Remove unused `register_deserialization` import

**Step 7: Clean up imports across all modified files**

Remove `from pymc_extras.deserialize import register_deserialization` (and `deserialize` if now unused) from all files where the calls were removed. Verify no other code in each file depends on these imports.

**Step 8: Run tests**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/ -x -q --timeout=600`
Expected: PASS

> **If tests fail:** The most likely cause is a remaining import or reference to the removed code. Check import statements in test files, conftest.py, and multidimensional.py.

**Step 9: Commit**

```bash
git add -A
git commit -m "refactor(serialization): remove RegistrationMeta, basis_from_dict, hsgp_from_dict, and register_deserialization calls"
```

### Task 8a Outcome ✅

Committed as `55c4af2a` — 12 files changed, 22 insertions(+), 556 deletions(-).

**Deferred to 8b:** `create_registration_meta()` and `DuplicatedTransformationError` in `base.py` were kept because `adstock.py` and `saturation.py` still import them. Must be removed in Task 8b after the metaclass references are eliminated.

**Additional changes not in original plan:**
- `EventEffect.from_dict` updated to use `registry.deserialize()` for `basis` field (old `pymc_extras.deserialize` no longer has Basis registrations)
- `test_events.py` updated to use `registry.serialize()`/`registry.deserialize()` instead of removed `basis_from_dict()`
- `test_fourier.py` `test_fourier_deserialization` updated to call `cls.from_dict(data)` directly instead of `pymc_extras.deserialize()`
- `test_media_transformation.py` `test_media_transformation_deserialize` and `test_media_config_list_deserialize` updated to use direct `from_dict()` calls; removed unused `deserialize` import
- `test_additive_effect.py` fixed 2 pre-existing assertion mismatches (`supplementary_data_custom_events` vs `supplementary_data/custom_events`)
- `test_base.py` removed obsolete `test_new_transformation_missing_lookup_name` (the `lookup_name` check it tested was removed)
- `tests/mmm/test_serialization_issues.py` deleted early (was planned for Task 10 but had broken imports from Task 7 removals)

---

## Task 8b: Remove Adstock/Saturation Metaclasses While Preserving Legacy MMM

This sub-task removes the metaclasses from `adstock.py` and `saturation.py` but keeps `adstock_from_dict()` and `saturation_from_dict()` alive (deprecated) so the Legacy MMM's load path continues to work.

> **Note from Task 8a:** `create_registration_meta()` and `DuplicatedTransformationError` were **not** removed from `base.py` in Task 8a because `adstock.py` and `saturation.py` still depend on them. They must be removed in this task after the metaclass references are eliminated from adstock/saturation.

**Files:**
- Modify: `pymc_marketing/mmm/components/base.py` — remove `lookup_name` from `Transformation`, `to_dict()`, **and also remove `create_registration_meta()` and `DuplicatedTransformationError`** (deferred from 8a)
- Modify: `pymc_marketing/mmm/components/adstock.py` — replace metaclass with hardcoded dict, deprecate `adstock_from_dict`
- Modify: `pymc_marketing/mmm/components/saturation.py` — same pattern
- Modify: `pymc_marketing/mmm/__init__.py` — keep but deprecate re-exports
- Test: run all mmm tests **including `test_mmm.py`** (Legacy MMM tests)

**Step 1: Remove `lookup_name` and legacy metaclass infrastructure from `base.py`**

In `pymc_marketing/mmm/components/base.py`:
- Remove `lookup_name: str` from the `Transformation` class (line 130)
- Remove `"lookup_name": self.lookup_name,` from `to_dict()` output (line 180)
- Delete `DuplicatedTransformationError` class (lines 682–688) — deferred from 8a
- Delete `create_registration_meta()` function (lines 691–719) — deferred from 8a

> **Important ordering:** Remove the `create_registration_meta` import from `adstock.py` and `saturation.py` (Step 2a/3) **before** deleting the function from `base.py`, or do all in the same commit. Otherwise imports will break.

**Step 2: Replace metaclass with hardcoded dict in `adstock.py`**

In `pymc_marketing/mmm/components/adstock.py`:

2a. Delete `AdstockRegistrationMeta` (line 84) and remove `metaclass=AdstockRegistrationMeta` from `AdstockTransformation` (line 87).

2b. Remove `lookup_name` from each concrete class: `BinomialAdstock`, `GeometricAdstock`, `DelayedAdstock`, `WeibullPDFAdstock`, `WeibullCDFAdstock`, `NoAdstock`.

2c. Keep `data.pop("lookup_name", None)` in `from_dict()` (line 150) — old serialized data may still contain this key and it must be stripped before passing to `cls(**data)`.

2d. Replace the metaclass-populated `ADSTOCK_TRANSFORMATIONS` dict (line 82) with a hardcoded dict **at the bottom of the file** (after all classes are defined):

```python
ADSTOCK_TRANSFORMATIONS: dict[str, type[AdstockTransformation]] = {
    "geometric": GeometricAdstock,
    "delayed": DelayedAdstock,
    "weibull_cdf": WeibullCDFAdstock,
    "weibull_pdf": WeibullPDFAdstock,
    "binomial": BinomialAdstock,
    "no_adstock": NoAdstock,
}
```

> **Note:** `GammaCDFAdstock` and `GammaPDFAdstock` may or may not have had `lookup_name` values — check before finalizing this dict. Only include classes that were previously auto-registered via the metaclass.

2e. Deprecate `adstock_from_dict()` — add a `FutureWarning` at the top of the function:

```python
def adstock_from_dict(data: dict) -> AdstockTransformation:
    """Create an adstock transformation from a dictionary.

    .. deprecated:: 0.18.2
        `adstock_from_dict` is deprecated and will be removed in 0.20.0.
        Use ``from pymc_marketing.serialization import registry; registry.deserialize(data)`` instead.
    """
    warnings.warn(
        "adstock_from_dict is deprecated and will be removed in 0.20.0. "
        "Use `from pymc_marketing.serialization import registry; "
        "registry.deserialize(data)` instead.",
        FutureWarning,
        stacklevel=2,
    )
    data = data.copy()
    data.pop("__type__", None)
    lookup_name = data.pop("lookup_name")
    cls = ADSTOCK_TRANSFORMATIONS[lookup_name]

    if "priors" in data:
        data["priors"] = {k: deserialize(v) for k, v in data["priors"].items()}

    return cls(**data)
```

2f. Delete `_is_adstock()` function and `register_deserialization(...)` call. Remove unused `register_deserialization` import.

**Step 3: Same pattern for `saturation.py`**

Mirror Step 2 for `pymc_marketing/mmm/components/saturation.py`:
- Delete `SaturationRegistrationMeta`, remove metaclass from `SaturationTransformation`
- Remove `lookup_name` from each concrete class
- Keep `data.pop("lookup_name", None)` in `from_dict()`
- Replace metaclass-populated `SATURATION_TRANSFORMATIONS` with hardcoded dict at bottom of file
- Deprecate `saturation_from_dict()` with `FutureWarning` (same pattern as adstock)
- Delete `_is_saturation()` and `register_deserialization(...)` call

**Step 4: Update `pymc_marketing/mmm/__init__.py`**

`adstock_from_dict` and `saturation_from_dict` are currently re-exported in `__init__.py` (lines 26, 39) and in `__all__` (lines 112, 123). **Keep these exports** — they are part of the public API and removing them would be a separate breaking change. The `FutureWarning` inside the functions is sufficient to signal deprecation.

**Step 5: Run tests (including Legacy MMM)**

Run both test suites:
```bash
/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_mmm.py -x -q --timeout=300
/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/ -x -q --timeout=600
```
Expected: PASS for both.

> **What to verify:** The Legacy MMM's save/load roundtrip must still work. If `test_mmm.py` has a save/load test, confirm it passes. If it does not have one, consider adding a minimal test that exercises `MMM.load()` to guard against regressions.

**Step 6: Verify removal is complete**

Run these greps — expected results noted:

```bash
# Should return 0 results (RegistrationMeta removed in 8a, create_registration_meta removed in 8b):
rg "RegistrationMeta|create_registration_meta|DuplicatedTransformationError" pymc_marketing/ --type py
rg "BASIS_TRANSFORMATIONS" pymc_marketing/ --type py
rg "basis_from_dict|hsgp_from_dict" pymc_marketing/ --type py

# Should return ONLY adstock.py and saturation.py (kept, deprecated):
rg "ADSTOCK_TRANSFORMATIONS|SATURATION_TRANSFORMATIONS" pymc_marketing/ --type py
rg "adstock_from_dict|saturation_from_dict" pymc_marketing/ --type py

# Should return 0 results (removed in Task 7):
rg "_serialize_mu_effect|_MUEFFECT_DESERIALIZERS|_deserialize_mu_effect" pymc_marketing/ --type py

# lookup_name should only appear in from_dict() pop calls and the hardcoded dicts:
rg "lookup_name" pymc_marketing/mmm/components/ --type py
```

**Step 7: Commit**

```bash
git add -A
git commit -m "refactor(serialization): remove adstock/saturation metaclasses, deprecate adstock_from_dict and saturation_from_dict (removal in 0.20.0)"
```

---

## Task 9: Remove `pymc_marketing/deserialize.py`

**Files:**
- Delete: `pymc_marketing/deserialize.py`
- Modify: any files that import from it

**Step 1: Check for imports**

```bash
rg "from pymc_marketing.deserialize import|from pymc_marketing import deserialize|import pymc_marketing.deserialize" pymc_marketing/ tests/
```

If any imports exist, update them to import directly from `pymc_extras.deserialize`.

**Step 2: Delete the file**

Delete `pymc_marketing/deserialize.py`.

**Step 3: Run tests**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/ -x -q --timeout=600`
Expected: PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor(serialization): remove deprecated pymc_marketing.deserialize shim"
```

---

## Task 10: Test Cleanup — Consolidate TypeRegistry Tests

> **Note from Task 8a:** `tests/mmm/test_serialization_issues.py` was already deleted in Task 8a (it had broken imports due to removed functions). Step 1 below is no longer needed.

**Files:**
- ~~Delete: `tests/mmm/test_serialization_issues.py`~~ (already deleted in 8a)
- Modify: `tests/mmm/test_serialization_roundtrips.py` (add missing unique tests)
- Modify: `tests/mmm/components/test_adstock.py` (remove `TestAdstockTypeRegistry`)
- Modify: `tests/mmm/components/test_saturation.py` (remove `TestSaturationTypeRegistry`)
- Modify: `tests/mmm/test_hsgp.py` (remove `TestHSGPTypeRegistry`, `TestDeferredFactoryInHSGP`, `TestHSGPKwargsTypeRegistry`)
- Modify: `tests/mmm/test_fourier.py` (remove `TestFourierTypeRegistry`)
- Modify: `tests/mmm/test_events.py` (remove `TestEventsTypeRegistry`)
- Modify: `tests/mmm/test_media_transformation.py` (remove `TestMediaTransformationTypeRegistry`)
- Modify: `tests/mmm/test_additive_effect.py` (remove `TestMuEffectTypeRegistry`, `TestLinearTrendEffectSerialization`, `TestFourierEffectSerialization`, `TestEventAdditiveEffectSerialization`)
- Modify: `tests/mmm/test_linear_trend.py` (remove `TestLinearTrendTypeRegistry`, `TestScalingTypeRegistry`)

**Step 1: ~~Delete `test_serialization_issues.py`~~ (DONE in Task 8a)**

~~This file is marked "THIS FILE SHOULD NOT BE MERGED TO THE MAIN REPO." Remove it entirely.~~ Already deleted.

**Step 2: Move unique tests to `test_serialization_roundtrips.py`**

The following test classes from per-component files have **no equivalent** in `test_serialization_roundtrips.py` and should be **moved** (not deleted):

| Source file | Class to move | New home in roundtrips file |
|---|---|---|
| `test_hsgp.py` | `TestHSGPKwargsTypeRegistry` (lines 997–1039) | Add as `TestHSGPKwargsRoundtrips` |
| `test_events.py` | `TestEventsTypeRegistry` (lines 953–1055) | Add as `TestBasisAndEventEffectRoundtrips` |
| `test_additive_effect.py` | `TestEventAdditiveEffectSerialization` (lines 412–503) | Add as `TestEventAdditiveEffectRoundtrips` |
| `test_linear_trend.py` | `TestLinearTrendTypeRegistry` (lines 150–189) | Add as `TestLinearTrendRoundtrips` |
| `test_linear_trend.py` | `TestScalingTypeRegistry` (lines 192–226) | Add as `TestScalingRoundtrips` |

Also enhance existing tests in `test_serialization_roundtrips.py`:
- `TestHSGPRoundtrips`: add an explicit SoftPlusHSGP round-trip test (currently missing).
- `TestMuEffectRoundtrips.test_fourier_effect_all_fourier_types`: parametrize over all 3 Fourier types (currently only MonthlyFourier).

Additionally, add a compact registration-validation test that covers all registered types:

```python
class TestRegistrationValidation:
    """Verify all built-in types are registered and have __type__ in to_dict()."""

    @pytest.mark.parametrize("cls", [
        GeometricAdstock, DelayedAdstock, WeibullCDFAdstock,
        LogisticSaturation, TanhSaturation, HillSaturation,
        HSGP, HSGPPeriodic, SoftPlusHSGP,
        YearlyFourier, MonthlyFourier, WeeklyFourier,
        # ... all registered types
    ])
    def test_type_registered_and_emits_type_key(self, cls):
        from pymc_marketing.serialization import registry
        type_key = f"{cls.__module__}.{cls.__qualname__}"
        assert type_key in registry._registry
```

**Step 3: Remove duplicate test classes from per-component files**

Remove these classes (they're fully duplicated in `test_serialization_roundtrips.py`):

| File | Class to remove | Lines |
|---|---|---|
| `test_adstock.py` | `TestAdstockTypeRegistry` | 269–323 |
| `test_saturation.py` | `TestSaturationTypeRegistry` | 341–389 |
| `test_hsgp.py` | `TestHSGPTypeRegistry` | 822–944 |
| `test_hsgp.py` | `TestDeferredFactoryInHSGP` | 947–994 |
| `test_fourier.py` | `TestFourierTypeRegistry` | 710–757 |
| `test_media_transformation.py` | `TestMediaTransformationTypeRegistry` | 234–330 |
| `test_additive_effect.py` | `TestMuEffectTypeRegistry` | 304–340 |
| `test_additive_effect.py` | `TestLinearTrendEffectSerialization` | 343–377 |
| `test_additive_effect.py` | `TestFourierEffectSerialization` | 380–409 |

**Step 4: Run tests**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/ -x -q --timeout=600`
Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "test(serialization): remove diagnostic file and consolidate TypeRegistry tests"
```

---

## Task 11: Integration Tests — End-to-End Save/Load

**Files:**
- Modify: `tests/mmm/test_multidimensional.py`
- Test: `tests/mmm/test_multidimensional.py`

Add end-to-end tests that exercise the **full** orchestrator path: create model → fit → save → load → verify. Use the multidimensional `MMM` class and fixtures from `tests/mmm/conftest.py`.

**Step 1: Add integration test class**

```python
class TestSerializationIntegration:
    """End-to-end save/load tests using the new TypeRegistry-based system."""

    def test_full_roundtrip_basic(self, simple_fitted_mmm, tmp_path):
        """Basic MMM save/load roundtrip with identical predictions."""
        fname = tmp_path / "model.nc"
        simple_fitted_mmm.save(str(fname))
        loaded = type(simple_fitted_mmm).load(str(fname))

        # Verify key attributes match
        assert type(loaded.adstock) == type(simple_fitted_mmm.adstock)
        assert type(loaded.saturation) == type(simple_fitted_mmm.saturation)
        assert loaded.channel_columns == simple_fitted_mmm.channel_columns
        assert loaded.date_column == simple_fitted_mmm.date_column

    def test_roundtrip_with_tvp(self, tmp_path):
        """Save/load with time-varying parameters (HSGP)."""
        # Build a model with time_varying_intercept=HSGP(...)
        # Fit, save, load, verify HSGP type is preserved
        pass  # Implementer builds minimal fixture

    def test_roundtrip_with_fourier_effect(self, tmp_path):
        """Save/load with FourierEffect mu_effect."""
        # Build model with mu_effects=[FourierEffect(...)]
        # Fit, save, load, verify FourierEffect is preserved
        pass  # Implementer builds minimal fixture

    def test_roundtrip_with_event_additive_effect(self, tmp_path):
        """Save/load with EventAdditiveEffect — df_events roundtrips via supplementary data."""
        # This is the critical test: EventAdditiveEffect was completely broken before
        # Build model with EventAdditiveEffect, fit, save, load
        # Verify df_events is intact and effect config matches
        pass  # Implementer builds minimal fixture

    def test_roundtrip_with_custom_mu_effect(self, tmp_path):
        """A user-defined MuEffect subclass saves and loads correctly."""
        from pymc_marketing.serialization import registry
        from pymc_marketing.mmm.additive_effect import MuEffect

        @registry.register
        class TestCustomEffect(MuEffect):
            my_param: float = 1.0
            prefix: str = "custom"
            date_dim_name: str = "date"

            def create_data(self, X): ...
            def create_effect(self, model, X, prefix): ...
            def set_data(self, X): ...

        # Build model with [TestCustomEffect(my_param=42.0)]
        # Fit, save, load, verify custom effect roundtrips
        pass  # Implementer builds minimal fixture

    def test_roundtrip_with_media_transformation(self, tmp_path):
        """Save/load with MediaTransformation (custom per-channel configs)."""
        pass  # Implementer builds minimal fixture
```

> **Note to implementer:** Several of these tests require minimal fitted models. Reuse the `simple_fitted_mmm` fixture from `tests/mmm/conftest.py` where possible. For tests that need specific component types, build the lightest possible MMM — small date range (20 rows), 1–2 channels, few MCMC samples (`draws=10, tune=10`). The goal is to verify the serialization path, not model quality.

**Step 2: Run tests**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_multidimensional.py::TestSerializationIntegration -v --timeout=600`
Expected: PASS

**Step 3: Run full test suite**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/ -x -q --timeout=600`
Expected: PASS

**Step 4: Commit**

```bash
git add tests/mmm/test_multidimensional.py
git commit -m "test(serialization): add end-to-end save/load integration tests"
```

---

## Task 12: Final Verification

**Step 1: Run the full test suite**

```bash
/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/ -x --timeout=600
```

Expected: All tests pass.

**Step 2: Verify old code removal**

```bash
# Should return 0 results (fully removed):
rg "RegistrationMeta|create_registration_meta" pymc_marketing/ --type py
rg "BASIS_TRANSFORMATIONS" pymc_marketing/ --type py
rg "basis_from_dict|hsgp_from_dict" pymc_marketing/ --type py
rg "_serialize_mu_effect|_MUEFFECT_DESERIALIZERS|_deserialize_mu_effect" pymc_marketing/ --type py

# Should return ONLY adstock.py, saturation.py, mmm.py, __init__.py (kept, deprecated):
rg "ADSTOCK_TRANSFORMATIONS|SATURATION_TRANSFORMATIONS" pymc_marketing/ --type py
rg "adstock_from_dict|saturation_from_dict" pymc_marketing/ --type py

# lookup_name should only appear in from_dict() pop calls:
rg "lookup_name" pymc_marketing/mmm/components/ --type py
```

**Step 3: Verify non-MMM models still work**

```bash
/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/clv/ -x -q --timeout=120
/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/customer_choice/ -x -q --timeout=120
```

Expected: All pass with zero changes to CLV/Customer Choice code.

**Step 4: Run pre-commit**

```bash
pre-commit run --all-files
```

Fix any formatting issues.

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup and verification for serialization Phase 2"
```

---

## Success Criteria Checklist

- [ ] **Full roundtrip:** MultidimensionalMMM with all component types saves/loads with identical predictions
- [ ] **EventAdditiveEffect roundtrip:** df_events intact in the .nc file via supplementary data
- [ ] **Custom MuEffect roundtrip:** User-defined MuEffect subclass saves/loads correctly
- [ ] **Migration:** Old-format (v0) .nc file loads successfully with deprecation warning
- [ ] **No silent failures:** Deserialization errors raise SerializationError, not warnings
- [ ] **Legacy code removal:** All metaclasses, `RegistrationMeta`, `basis_from_dict`, `hsgp_from_dict` are gone (verified by grep)
- [ ] **Legacy MMM still works:** `test_mmm.py` passes — the deprecated MMM's save/load path is functional
- [ ] **Deprecated functions emit warnings:** `adstock_from_dict` and `saturation_from_dict` emit `FutureWarning` (removal in 0.20.0)
- [ ] **Non-MMM models unaffected:** CLV and Customer Choice test suites pass without changes
- [ ] **No redundant tests:** test_serialization_issues.py removed; per-component TypeRegistry classes consolidated
- [ ] **__serialization_version__** attr present in all new saves

## Risk Notes

- **ArviZ supplementary data group naming:** Verify that ArviZ supports the chosen naming convention (`supplementary_data/prefix` or `supplementary_data_prefix`) for both saving and loading. Test with both `netcdf4` and `h5netcdf` engines.
- **Migration map completeness:** The `_ADSTOCK_TYPE_MAP` and `_SATURATION_TYPE_MAP` in `serialization_migration.py` must include every concrete subclass that has ever been saved. Cross-reference with the `lookup_name` values in each concrete class.
- **Backward compatibility during transition:** Tasks 3–5 add fallback branches (`if "__type__" in data: ... else: legacy_from_dict(...)`) that handle both old and new formats. These fallbacks are removed in Task 7 once the migration tool is in place. Verify tests pass at each step.
- **Import ordering:** Phase 1 added `from pymc_marketing.serialization import registry` to many module files. These imports happen at module load time. Verify no circular import issues arise when `model_builder.py` also imports from `serialization.py`.
- **`parse_model_config()` interaction:** The updated `_model_config_formatting()` may deserialize Priors via `registry.deserialize()` before `parse_model_config()` processes them. Verify that `parse_model_config()` handles already-deserialized Prior objects gracefully (skips them or returns as-is).
- **Legacy MMM save/load interaction with Task 1:** Task 1 updated `_json_default` to try `registry.serialize()` first, which emits `__type__` keys without `lookup_name`. The deprecated MMM uses this same base-class `_json_default` for saving, but its `adstock_from_dict` expects `lookup_name` on load. Verify that the Legacy MMM's save→load roundtrip still works after Tasks 1–6 — if `registry.serialize()` produces output without `lookup_name`, the load side may need to handle both formats.
