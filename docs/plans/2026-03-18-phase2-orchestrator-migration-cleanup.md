# Phase 2: Serialization Overhaul — Orchestrator, Migration & Cleanup

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the serialization overhaul by switching the MMM orchestrator to use the TypeRegistry, adding base-class hooks, creating a migration tool, removing all legacy serialization code, and cleaning up tests.

**Architecture:** Phase 1 added the `TypeRegistry`, `SerializableMixin`, and `DeferredFactory`, and updated all ~40 serializable classes to emit `__type__` keys and have `from_dict()` classmethods. Phase 2 wires the MMM orchestrator to use this new system for save/load, replaces silent failures with explicit errors, provides backward compatibility via a migration tool, and removes all old code paths.

**Tech Stack:** Python 3.10+, PyMC, ArviZ (InferenceData), Pydantic v2, xarray, pytest

**Parent design:** [Serialization Overhaul Design](2026-03-05-serialization-overhaul-design.md)
**Phase 1 recap:** [Phase 2 Scope Guide](2026-03-18-phase2-scope-guide.md)

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

## Task 1: Base-Class Hooks — `_json_default()` registry-first dispatch  ✅ DONE

**Files:**
- Modify: `pymc_marketing/model_builder.py:248-264` (`_json_default` inside `create_idata_attrs`)
- Test: `tests/test_model_builder.py` (add new test class)

**Step 1: Write the failing test**

Add to the bottom of `tests/test_model_builder.py`:

```python
class TestJsonDefaultRegistryDispatch:
    """Verify _json_default tries registry.serialize() before duck-typing."""

    def test_registered_type_serialized_via_registry(self):
        """A type registered in TypeRegistry should serialize with __type__ key."""
        from pymc_marketing.serialization import registry

        @registry.register
        class _DummyRegistered:
            def to_dict(self):
                return {
                    "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
                    "value": 42,
                }

            @classmethod
            def from_dict(cls, data):
                return cls()

        class FakeModel:
            _model_type = "test"
            version = "0.0.1"
            sampler_config = {}

            @property
            def _serializable_model_config(self):
                return {"component": _DummyRegistered()}

        model = FakeModel()
        model.id  # unused, just need the class shape
        attrs = ModelIO.create_idata_attrs(model)
        config = json.loads(attrs["model_config"])
        assert "__type__" in config["component"]

    def test_unregistered_type_falls_through(self):
        """An unregistered type with to_dict() still works via duck-typing fallback."""

        class _UnregisteredWithToDict:
            def to_dict(self):
                return {"value": 99}

        class FakeModel:
            _model_type = "test"
            version = "0.0.1"
            sampler_config = {}

            @property
            def _serializable_model_config(self):
                return {"component": _UnregisteredWithToDict()}

        model = FakeModel()
        attrs = ModelIO.create_idata_attrs(model)
        config = json.loads(attrs["model_config"])
        assert config["component"] == {"value": 99}
        assert "__type__" not in config["component"]
```

**Step 2: Run test to verify it fails**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/test_model_builder.py::TestJsonDefaultRegistryDispatch -v`
Expected: FAIL — `__type__` not in serialized output because `_json_default` doesn't try registry yet.

**Step 3: Write minimal implementation**

In `pymc_marketing/model_builder.py`, inside `create_idata_attrs()`, update the `_json_default` local function (lines 248–264):

```python
        def _json_default(obj):
            """Handle objects that aren't JSON serializable by default."""
            from pymc_marketing.serialization import registry

            try:
                return registry.serialize(obj)
            except (KeyError, TypeError):
                pass

            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "model_dump"):
                return obj.model_dump(mode="json")
            if hasattr(obj, "__dict__"):
                return {
                    k: v
                    for k, v in obj.__dict__.items()
                    if not callable(v) and not k.startswith("_")
                }
            return str(obj)
```

**Step 4: Run test to verify it passes**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/test_model_builder.py::TestJsonDefaultRegistryDispatch -v`
Expected: PASS

**Step 5: Run non-MMM model tests to verify no regression**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/clv/ -x -q --timeout=120`
Expected: PASS (base-class hooks are transparent to non-MMM models)

**Step 6: Commit**

```bash
git add pymc_marketing/model_builder.py tests/test_model_builder.py
git commit -m "feat(serialization): add registry-first dispatch to _json_default in ModelBuilder"
```

> **Batch 1 outcome:** Task 1 test passed *before* the implementation change because the existing `to_dict()` fallback already emitted `__type__`. The implementation change is still architecturally correct (registry-first ensures custom serializers are used), but the test doesn't prove the registry path is taken. A more rigorous test would use a registered type with a custom serializer that differs from `to_dict()`.

---

## Task 2: Base-Class Hooks — `_model_config_formatting()` `__type__` detection  ✅ DONE

**Files:**
- Modify: `pymc_marketing/model_builder.py:401-422` (`_model_config_formatting`)
- Test: `tests/test_model_builder.py`

**Step 1: Write the failing test**

Add to `tests/test_model_builder.py`:

```python
class TestModelConfigFormattingTypeDetection:
    """Verify _model_config_formatting routes __type__ dicts through registry."""

    def test_type_key_deserialized_via_registry(self):
        """A dict with __type__ key should be deserialized via registry."""
        from pymc_marketing.serialization import registry

        @registry.register
        class _DummyConfigType:
            def __init__(self, value=42):
                self.value = value

            def to_dict(self):
                return {
                    "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
                    "value": self.value,
                }

            @classmethod
            def from_dict(cls, data):
                return cls(value=data.get("value", 42))

        model_config = {
            "component": {
                "__type__": f"{_DummyConfigType.__module__}.{_DummyConfigType.__qualname__}",
                "value": 42,
            }
        }
        result = ModelIO._model_config_formatting(model_config)
        assert isinstance(result["component"], _DummyConfigType)
        assert result["component"].value == 42

    def test_legacy_dict_without_type_key_unchanged(self):
        """A dict without __type__ follows the existing list→tuple/array heuristic."""
        model_config = {
            "intercept": {"mu": 0, "sigma": 1, "dims": ["date"]},
        }
        result = ModelIO._model_config_formatting(model_config)
        assert isinstance(result["intercept"]["dims"], tuple)
        assert result["intercept"]["dims"] == ("date",)

    def test_plain_scalars_unchanged(self):
        """Non-dict values pass through unchanged."""
        model_config = {"n_obs": 100, "name": "test"}
        result = ModelIO._model_config_formatting(model_config)
        assert result == {"n_obs": 100, "name": "test"}
```

**Step 2: Run test to verify it fails**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/test_model_builder.py::TestModelConfigFormattingTypeDetection -v`
Expected: FAIL — `__type__` dicts are not deserialized (treated as regular dicts).

**Step 3: Write minimal implementation**

In `pymc_marketing/model_builder.py`, update `_model_config_formatting()` (lines 401–422):

```python
    @classmethod
    def _model_config_formatting(cls, model_config: dict) -> dict:
        """Format the model configuration.

        If a value is a dict with a ``__type__`` key, it is deserialized via the
        TypeRegistry. Otherwise the existing ``list→tuple`` / ``list→np.array``
        heuristic is applied for backward compatibility.
        """
        from pymc_marketing.serialization import registry

        for key in model_config:
            value = model_config[key]
            if isinstance(value, dict) and "__type__" in value:
                model_config[key] = registry.deserialize(value)
            elif isinstance(value, dict):
                for sub_key in value:
                    if isinstance(value[sub_key], list):
                        if sub_key == "dims":
                            value[sub_key] = tuple(value[sub_key])
                        else:
                            value[sub_key] = np.array(value[sub_key])
        return model_config
```

**Step 4: Run test to verify it passes**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/test_model_builder.py::TestModelConfigFormattingTypeDetection -v`
Expected: PASS

**Step 5: Run non-MMM model tests to verify no regression**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/clv/ -x -q --timeout=120`
Expected: PASS

**Step 6: Commit**

```bash
git add pymc_marketing/model_builder.py tests/test_model_builder.py
git commit -m "feat(serialization): add __type__ detection to _model_config_formatting in ModelBuilder"
```

> **Batch 1 outcome:** Task 2 base-class `_model_config_formatting` was updated successfully. Note: `multidimensional.py` has its own `_model_config_formatting` override that does **not** include `__type__` detection — it will be removed in Task 7 once the base-class version handles everything.

---

## Task 3: Orchestrator Save Side — `_serializable_model_config` and `create_idata_attrs`  ✅ DONE

**Files:**
- Modify: `pymc_marketing/mmm/multidimensional.py:997-1077` (`_serializable_model_config`, `create_idata_attrs`)
- Test: `tests/mmm/test_multidimensional.py` (NOT `test_mmm.py` — see class location note above)

This task replaces duck-typing with `registry.serialize()` and adds `__serialization_version__`.

**Step 1: Write the failing test**

Add to `tests/mmm/test_multidimensional.py` (uses the `simple_fitted_mmm` fixture from `tests/mmm/conftest.py`):

```python
class TestSerializationVersion:
    """Verify __serialization_version__ attr is written during save."""

    def test_idata_has_serialization_version(self, simple_fitted_mmm):
        """create_idata_attrs must include __serialization_version__."""
        attrs = simple_fitted_mmm.create_idata_attrs()
        assert "__serialization_version__" in attrs
        assert attrs["__serialization_version__"] == "1"

    def test_adstock_attr_has_type_key(self, simple_fitted_mmm):
        """Adstock attr should contain __type__ key after serialization."""
        attrs = simple_fitted_mmm.create_idata_attrs()
        adstock_data = json.loads(attrs["adstock"])
        assert "__type__" in adstock_data

    def test_saturation_attr_has_type_key(self, simple_fitted_mmm):
        """Saturation attr should contain __type__ key after serialization."""
        attrs = simple_fitted_mmm.create_idata_attrs()
        saturation_data = json.loads(attrs["saturation"])
        assert "__type__" in saturation_data

    def test_scaling_attr_has_type_key(self, simple_fitted_mmm):
        """Scaling attr should contain __type__ key if not null."""
        attrs = simple_fitted_mmm.create_idata_attrs()
        scaling_data = json.loads(attrs["scaling"])
        if scaling_data is not None:
            assert "__type__" in scaling_data
```

> **Note to implementer:** Use the `simple_fitted_mmm` fixture from `tests/mmm/conftest.py` (multidimensional MMM). Do NOT use `mmm_fitted` from `test_mmm.py` — that is the old deprecated class and won't hit the updated code.

**Step 2: Run test to verify it fails**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_multidimensional.py::TestSerializationVersion -v --timeout=120`
Expected: FAIL — `__serialization_version__` not in attrs, `scaling` missing `__type__`. (Note: `adstock` and `saturation` already have `__type__` from Phase 1 `to_dict()` updates.)

**Step 3: Implement `_serializable_model_config` with registry**

In `pymc_marketing/mmm/multidimensional.py`, replace lines 997–1020:

```python
    @property
    def _serializable_model_config(self) -> dict[str, Any]:
        from pymc_marketing.serialization import registry

        def serialize_value(value):
            """Recursively serialize values to JSON-compatible types."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            else:
                try:
                    return registry.serialize(value)
                except (KeyError, TypeError):
                    pass
                if hasattr(value, "to_dict"):
                    return value.to_dict()
                return value

        serializable_config = {}
        for key, value in self.model_config.items():
            serializable_config[key] = serialize_value(value)

        return serializable_config
```

**Step 4: Implement `create_idata_attrs` with registry**

In `pymc_marketing/mmm/multidimensional.py`, replace `create_idata_attrs()` (lines 1022–1077):

```python
    def create_idata_attrs(self) -> dict[str, str]:
        """Return the idata attributes for the model."""
        from pymc_marketing.serialization import registry

        attrs = super().create_idata_attrs()
        attrs["__serialization_version__"] = "1"
        attrs["dims"] = json.dumps(self.dims)
        attrs["date_column"] = self.date_column
        attrs["adstock"] = json.dumps(registry.serialize(self.adstock))
        attrs["saturation"] = json.dumps(registry.serialize(self.saturation))
        attrs["adstock_first"] = json.dumps(self.adstock_first)
        attrs["control_columns"] = json.dumps(self.control_columns)
        attrs["channel_columns"] = json.dumps(self.channel_columns)
        attrs["yearly_seasonality"] = json.dumps(self.yearly_seasonality)
        attrs["time_varying_intercept"] = json.dumps(
            registry.serialize(self.time_varying_intercept)
            if isinstance(self.time_varying_intercept, HSGPBase)
            else self.time_varying_intercept
        )
        attrs["time_varying_media"] = json.dumps(
            registry.serialize(self.time_varying_media)
            if isinstance(self.time_varying_media, HSGPBase)
            else self.time_varying_media
        )
        attrs["target_column"] = self.target_column
        attrs["scaling"] = json.dumps(registry.serialize(self.scaling))
        attrs["dag"] = json.dumps(getattr(self, "dag", None))
        attrs["treatment_nodes"] = json.dumps(getattr(self, "treatment_nodes", None))
        attrs["outcome_node"] = json.dumps(getattr(self, "outcome_node", None))

        mu_effects_list = [registry.serialize(effect) for effect in self.mu_effects]
        attrs["mu_effects"] = json.dumps(mu_effects_list)

        # cost_per_unit handling — keep as-is (DataFrame→JSON, not a registry type)
        cost_per_unit = getattr(self, "cost_per_unit", None)
        if cost_per_unit is not None and isinstance(cost_per_unit, pd.DataFrame):
            attrs["cost_per_unit"] = cost_per_unit.to_json(orient="split")
        else:
            attrs["cost_per_unit"] = json.dumps(None)

        return attrs
```

> **Critical:** The `hsgp_class` key injection that previously existed for `time_varying_intercept`/`time_varying_media` is now gone — HSGP's own `to_dict()` (updated in Phase 1) already emits `__type__`.

**Step 5: Run test to verify it passes**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_multidimensional.py::TestSerializationVersion -v --timeout=120`
Expected: PASS

**Step 6: Update load-side for dual-format compatibility (pulled forward from Task 5)**

> **Critical learning from Batch 1:** The save-side and load-side **cannot** be updated independently. The moment `create_idata_attrs` emits `__type__` keys instead of `"class"` / `"hsgp_class"` keys, existing roundtrip tests break because the load-side still expects the old format. You must update `attrs_to_init_kwargs` and `build_from_idata` to handle both formats in this same task.

Update `attrs_to_init_kwargs` to check for `__type__` and use `registry.deserialize()` when present, falling back to legacy `adstock_from_dict()` / `saturation_from_dict()` / `hsgp_from_dict()` otherwise. Update `build_from_idata` to check `__type__` on mu_effects and route through `registry.deserialize()` with a fallback to `_deserialize_mu_effect()`.

See Task 5 for the exact implementation (which is now partially done).

**Step 7: Fix existing test assertions for `to_dict()` changes**

Phase 1 changed `to_dict()` to include `__type__` and `from_dict()` now converts `dims` from `list` to `tuple`. Three existing tests in `test_multidimensional.py` that compare `loaded.time_varying_media.to_dict() == original_data` break because the roundtripped object's `to_dict()` includes `__type__` and has `dims` as a tuple instead of a list. Update these assertions to account for the new format.

**Step 8: Run existing MMM tests to verify no regression**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/ -x -q --timeout=300`
Expected: PASS

**Step 9: Commit**

```bash
git add pymc_marketing/mmm/multidimensional.py tests/mmm/test_multidimensional.py
git commit -m "feat(serialization): switch orchestrator save side to registry.serialize()"
```

---

## Batch 1 Learnings

The following learnings emerged during execution of Tasks 1–3 and should be considered for all remaining tasks:

1. **Correct `MMM` class for tests:** `tests/mmm/test_mmm.py` imports the old deprecated `MMM` from `pymc_marketing.mmm.mmm`. All orchestrator serialization code is in `multidimensional.py`, so new serialization tests must use `tests/mmm/test_multidimensional.py` with the `simple_fitted_mmm` fixture from `tests/mmm/conftest.py`.

2. **Save/load sides must be updated together:** Changing the save-side format (emitting `__type__` instead of `"class"` / `"hsgp_class"`) immediately breaks existing roundtrip tests if the load-side still expects the old keys. Dual-format support on the load side must be added in the same task as the save-side change.

3. **Phase 1's `to_dict()` changes propagate:** Phase 1 added `__type__` to `to_dict()` output and `from_dict()` converts `dims` to tuples. Any existing tests that compare `to_dict()` output with strict equality against the original input dict will break after a roundtrip. Expect to update such assertions.

4. **Task 1 tests don't prove registry path:** The `_json_default` test passed before implementation because `to_dict()` already includes `__type__`. If stronger proof is needed later, test with a type whose registry serializer differs from its `to_dict()`.

5. **`attrs_to_init_kwargs` and `build_from_idata` now have dual-format support** (registry for `__type__` dicts, legacy `*_from_dict()` functions otherwise). This partial Task 5 work is already merged.

6. **`build_from_idata` still has `except Exception` catch-all** on mu_effect deserialization (line ~3230). Task 5 should remove this and replace with explicit `SerializationError`.

---

## Task 4: Supplementary Data Write — `df_events` into InferenceData

**Files:**
- Modify: `pymc_marketing/mmm/multidimensional.py` (override `save()`)
- Test: `tests/mmm/test_multidimensional.py`

The `EventAdditiveEffect.to_dict()` stores a `df_events_group` key pointing to an idata group (e.g., `"supplementary_data/events"`). The deserializer reads from `context.idata[group_name]`. But **nothing currently writes** that group. This task adds the write side.

**Step 1: Write the failing test**

Add to `tests/mmm/test_multidimensional.py`:

```python
class TestSupplementaryDataWrite:
    """Verify EventAdditiveEffect's df_events is stored in idata during save."""

    def test_event_supplementary_data_written_to_idata(self, tmp_path):
        """Save a model with EventAdditiveEffect → idata contains supplementary group."""
        from pymc_marketing.mmm.events import EventEffect, GaussianBasis
        from pymc_marketing.mmm.additive_effect import EventAdditiveEffect
        from pymc_marketing.mmm.components.adstock import GeometricAdstock
        from pymc_marketing.mmm.components.saturation import LogisticSaturation

        df_events = pd.DataFrame({
            "name": ["event_a", "event_b"],
            "date": pd.to_datetime(["2023-06-15", "2023-09-01"]),
            "duration": [7, 14],
        })
        effect = EventEffect(basis=GaussianBasis())
        event_effect = EventAdditiveEffect(
            df_events=df_events,
            effect=effect,
            prefix="events",
        )

        # Build a minimal MMM with the event effect
        # (Use the simplest constructor that accepts mu_effects)
        mmm = MMM(
            date_column="date",
            channel_columns=["x1"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            mu_effects=[event_effect],
        )

        # Create minimal fit data and fit (or mock idata)
        # ... (implementer: use the lightest possible approach to get idata)
        # After fitting, save and reload the idata:
        fname = tmp_path / "model.nc"
        mmm.save(str(fname))

        loaded_idata = az.from_netcdf(fname)
        # The supplementary data group should exist
        assert hasattr(loaded_idata, "supplementary_data_events") or \
               "supplementary_data/events" in loaded_idata.groups()
```

> **Note to implementer:** This test requires a fitted model. Use the lightest fixture available or mock idata minimally. The exact group name depends on how ArviZ handles the naming — adjust the assertion accordingly.

**Step 2: Implement supplementary data write**

Override `save()` on the MMM class in `pymc_marketing/mmm/multidimensional.py`:

```python
    def save(self, fname: str, **kwargs) -> None:
        """Save the model, including supplementary data for MuEffects."""
        import xarray as xr
        from pymc_marketing.mmm.additive_effect import EventAdditiveEffect

        if self.idata is None or "posterior" not in self.idata:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")

        # Write supplementary data groups for MuEffects that need them
        for effect in self.mu_effects:
            if isinstance(effect, EventAdditiveEffect):
                group_name = f"supplementary_data/{effect.prefix}"
                ds = xr.Dataset.from_dataframe(
                    effect.df_events.set_index(
                        effect.df_events.columns[0]
                    )
                )
                self.idata.add_groups({group_name: ds})

        super().save(fname, **kwargs)
```

> **Implementation note:** ArviZ `add_groups()` with `/` in the name may require special handling depending on the ArviZ version. If ArviZ rejects `/`, use `_` as separator (e.g., `supplementary_data_events`) and update `EventAdditiveEffect.to_dict()` and the custom deserializer to match. **Test both naming conventions** and pick whichever ArviZ supports.

**Step 3: Run test to verify it passes**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_multidimensional.py::TestSupplementaryDataWrite -v --timeout=300`
Expected: PASS

**Step 4: Commit**

```bash
git add pymc_marketing/mmm/multidimensional.py tests/mmm/test_multidimensional.py
git commit -m "feat(serialization): write EventAdditiveEffect supplementary data to idata during save"
```

---

## Task 5: Orchestrator Load Side — `attrs_to_init_kwargs` with registry  ⚡ PARTIALLY DONE

**Files:**
- Modify: `pymc_marketing/mmm/multidimensional.py` (`attrs_to_init_kwargs`, `build_from_idata`)
- Test: `tests/mmm/test_multidimensional.py`

> **Already done (from Task 3 pull-forward):**
> - `attrs_to_init_kwargs` already has dual-format support: checks `__type__` → `registry.deserialize()`, else falls back to `adstock_from_dict()` / `saturation_from_dict()` / `hsgp_from_dict()`.
> - `build_from_idata` already routes `__type__` mu_effects through `registry.deserialize()` with fallback to `_deserialize_mu_effect()`.
>
> **What remains for this task:**
> 1. Remove the `except Exception` catch-all in `build_from_idata` (line ~3230) and replace with explicit `SerializationError` propagation.
> 2. Write the `SerializationError` test (below).
> 3. Create the `_make_minimal_mmm_attrs` helper for use in this and later tests.

**Step 1: Write the failing test**

Add to `tests/mmm/test_multidimensional.py`:

```python
class TestRegistryDeserialization:
    """Verify the load side uses registry.deserialize() for components."""

    def test_deserialization_error_raises_not_warns(self):
        """MuEffect deserialization failures should raise SerializationError."""
        from pymc_marketing.serialization import SerializationError

        # Create idata with an invalid mu_effect entry
        attrs = _make_minimal_mmm_attrs()  # helper that builds valid base attrs
        attrs["mu_effects"] = json.dumps([{
            "__type__": "nonexistent.module.FakeEffect",
            "value": 1,
        }])

        # Loading should raise SerializationError, not silently warn
        with pytest.raises(SerializationError):
            # Trigger deserialization via the orchestrator load path
            ...
```

> **Note to implementer:** This test needs a helper that produces valid MMM idata attrs with the new `__serialization_version__` = 1 format. Build this helper once and reuse across tests in this file. The key assertion is that `SerializationError` is raised (not swallowed by `except Exception`).

**Step 2: Remove the `except Exception` catch-all and propagate `SerializationError`**

> `attrs_to_init_kwargs` is already updated (done in Task 3). The remaining work is on `build_from_idata`.

In `pymc_marketing/mmm/multidimensional.py`, update the `build_from_idata()` mu_effects block to remove the `except Exception` catch-all (currently at line ~3230) and let deserialization errors propagate:

```python
        if "mu_effects" in idata.attrs:
            from pymc_marketing.serialization import (
                DeserializationContext,
                registry,
            )

            mu_effects_data = json.loads(idata.attrs["mu_effects"])
            ctx = DeserializationContext(idata=idata)
            self.mu_effects = []
            for effect_data in mu_effects_data:
                if isinstance(effect_data, dict) and "__type__" in effect_data:
                    effect = registry.deserialize(effect_data, context=ctx)
                else:
                    effect = _deserialize_mu_effect(effect_data)
                self.mu_effects.append(effect)
```

> **Critical change:** The `except Exception` catch-all is removed. Deserialization failures now propagate as `SerializationError`. The legacy `_deserialize_mu_effect` fallback is kept for v0 files during the transition.

**Step 3: Run tests**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/test_multidimensional.py -x -q --timeout=300`
Expected: PASS

**Step 4: Commit**

```bash
git add pymc_marketing/mmm/multidimensional.py tests/mmm/test_multidimensional.py
git commit -m "feat(serialization): remove except-Exception catch-all, propagate SerializationError"
```

---

## Task 6: Migration Tool — `migrate.py`

**Files:**
- Create: `pymc_marketing/migrate.py`
- Modify: `pymc_marketing/mmm/multidimensional.py` (wire auto-migration into load)
- Test: `tests/test_migration.py`

**Step 1: Write the failing test**

Create `tests/test_migration.py`:

```python
"""Tests for the migration tool (v0 → v1 format conversion)."""

import json

import arviz as az
import numpy as np
import pytest
import xarray as xr

from pymc_marketing.migrate import migrate_idata, CURRENT_VERSION


class TestMigrateIdataV0ToV1:
    """Verify v0 → v1 migration rewrites attrs correctly."""

    def _make_v0_idata(self):
        """Create a minimal InferenceData with v0 (old-format) attrs."""
        posterior = xr.Dataset({"x": xr.DataArray(np.random.randn(4, 100))})
        idata = az.InferenceData(posterior=posterior)
        idata.attrs["id"] = "abc123"
        idata.attrs["model_type"] = "MMM"
        idata.attrs["version"] = "0.0.1"
        idata.attrs["sampler_config"] = json.dumps({})
        idata.attrs["model_config"] = json.dumps({})
        idata.attrs["adstock"] = json.dumps({
            "lookup_name": "geometric",
            "l_max": 4,
            "normalize": True,
            "mode": "After",
            "priors": {"alpha": {"distribution": "Beta", "dims": ["channel"], "mu": 0.5}},
        })
        idata.attrs["saturation"] = json.dumps({
            "lookup_name": "logistic",
            "priors": {"lam": {"distribution": "Gamma", "mu": 1.0}},
        })
        idata.attrs["time_varying_intercept"] = json.dumps({
            "hsgp_class": "HSGP",
            "m": 200,
            "L": None,
            "eta": 1.0,
            "ls": 5.0,
            "dims": ["date"],
        })
        idata.attrs["mu_effects"] = json.dumps([
            {"class": "FourierEffect", "fourier": {"__type__": "pymc_marketing.mmm.fourier.YearlyFourier", "n_order": 2}},
            {"class": "LinearTrendEffect", "trend": {"n_changepoints": 5}, "prefix": "trend"},
        ])
        # No __serialization_version__
        return idata

    def test_migration_adds_serialization_version(self):
        idata = self._make_v0_idata()
        migrated = migrate_idata(idata)
        assert migrated.attrs["__serialization_version__"] == str(CURRENT_VERSION)

    def test_migration_converts_adstock_lookup_name_to_type(self):
        idata = self._make_v0_idata()
        migrated = migrate_idata(idata)
        adstock = json.loads(migrated.attrs["adstock"])
        assert "__type__" in adstock
        assert "lookup_name" not in adstock

    def test_migration_converts_hsgp_class_to_type(self):
        idata = self._make_v0_idata()
        migrated = migrate_idata(idata)
        tvi = json.loads(migrated.attrs["time_varying_intercept"])
        assert "__type__" in tvi
        assert "hsgp_class" not in tvi

    def test_migration_converts_mu_effect_class_to_type(self):
        idata = self._make_v0_idata()
        migrated = migrate_idata(idata)
        effects = json.loads(migrated.attrs["mu_effects"])
        for effect in effects:
            assert "__type__" in effect
            assert "class" not in effect

    def test_migration_drops_stale_id(self):
        idata = self._make_v0_idata()
        migrated = migrate_idata(idata)
        assert "id" not in migrated.attrs

    def test_already_v1_is_noop(self):
        idata = self._make_v0_idata()
        idata.attrs["__serialization_version__"] = str(CURRENT_VERSION)
        migrated = migrate_idata(idata)
        assert migrated.attrs == idata.attrs


class TestMigrateIdataCLI:
    """Verify the CLI entry point."""

    def test_cli_migrates_file(self, tmp_path):
        """python -m pymc_marketing.migrate <file> should migrate in-place."""
        # Create and save a v0 idata
        posterior = xr.Dataset({"x": xr.DataArray(np.random.randn(4, 100))})
        idata = az.InferenceData(posterior=posterior)
        idata.attrs["model_type"] = "MMM"
        idata.attrs["version"] = "0.0.1"
        idata.attrs["adstock"] = json.dumps({"lookup_name": "geometric", "l_max": 4})
        fname = tmp_path / "model.nc"
        idata.to_netcdf(str(fname))

        import subprocess
        result = subprocess.run(
            ["/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python", "-m", "pymc_marketing.migrate", str(fname)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

        reloaded = az.from_netcdf(fname)
        assert "__serialization_version__" in reloaded.attrs
```

**Step 2: Run test to verify it fails**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/test_migration.py -v`
Expected: FAIL — `pymc_marketing.migrate` module doesn't exist.

**Step 3: Create `pymc_marketing/migrate.py`**

```python
"""Migration tool for old-format serialized models.

Converts v0 (pre-TypeRegistry) idata attrs to v1 (__type__-based) format.

Usage:
    python -m pymc_marketing.migrate model.nc
"""

from __future__ import annotations

import json
import sys
import warnings
from typing import Any

import arviz as az

CURRENT_VERSION = 1

# Maps old lookup_name → fully-qualified __type__
_ADSTOCK_TYPE_MAP = {
    "geometric": "pymc_marketing.mmm.components.adstock.GeometricAdstock",
    "delayed": "pymc_marketing.mmm.components.adstock.DelayedAdstock",
    "weibull_cdf": "pymc_marketing.mmm.components.adstock.WeibullCDFAdstock",
    "weibull_pdf": "pymc_marketing.mmm.components.adstock.WeibullPDFAdstock",
    "gamma_cdf": "pymc_marketing.mmm.components.adstock.GammaCDFAdstock",
    "gamma_pdf": "pymc_marketing.mmm.components.adstock.GammaPDFAdstock",
}

_SATURATION_TYPE_MAP = {
    "logistic": "pymc_marketing.mmm.components.saturation.LogisticSaturation",
    "tanh": "pymc_marketing.mmm.components.saturation.TanhSaturation",
    "tanh_scaled": "pymc_marketing.mmm.components.saturation.TanhSaturationBaselined",
    "michaelis_menten": "pymc_marketing.mmm.components.saturation.MichaelisMentenSaturation",
    "hill": "pymc_marketing.mmm.components.saturation.HillSaturation",
    "hill_adbudg": "pymc_marketing.mmm.components.saturation.HillSaturationSigmoid",
    "inverse_scaled_logistic": "pymc_marketing.mmm.components.saturation.InverseScaledLogisticSaturation",
    "root": "pymc_marketing.mmm.components.saturation.RootSaturation",
    "power": "pymc_marketing.mmm.components.saturation.PowerSaturation",
}

_HSGP_CLASS_MAP = {
    "HSGP": "pymc_marketing.mmm.hsgp.HSGP",
    "SoftPlusHSGP": "pymc_marketing.mmm.hsgp.SoftPlusHSGP",
    "HSGPPeriodic": "pymc_marketing.mmm.hsgp.HSGPPeriodic",
}

_MUEFFECT_CLASS_MAP = {
    "FourierEffect": "pymc_marketing.mmm.additive_effect.FourierEffect",
    "LinearTrendEffect": "pymc_marketing.mmm.additive_effect.LinearTrendEffect",
    "EventAdditiveEffect": "pymc_marketing.mmm.additive_effect.EventAdditiveEffect",
}


def _migrate_v0_to_v1(attrs: dict[str, Any]) -> dict[str, Any]:
    """Rewrite v0 attrs to v1 format."""
    attrs = dict(attrs)

    # Migrate adstock: lookup_name → __type__
    if "adstock" in attrs:
        adstock = json.loads(attrs["adstock"])
        if isinstance(adstock, dict) and "lookup_name" in adstock and "__type__" not in adstock:
            lookup = adstock.pop("lookup_name")
            adstock["__type__"] = _ADSTOCK_TYPE_MAP.get(lookup, lookup)
            attrs["adstock"] = json.dumps(adstock)

    # Migrate saturation: lookup_name → __type__
    if "saturation" in attrs:
        sat = json.loads(attrs["saturation"])
        if isinstance(sat, dict) and "lookup_name" in sat and "__type__" not in sat:
            lookup = sat.pop("lookup_name")
            sat["__type__"] = _SATURATION_TYPE_MAP.get(lookup, lookup)
            attrs["saturation"] = json.dumps(sat)

    # Migrate time_varying_intercept: hsgp_class → __type__
    for key in ("time_varying_intercept", "time_varying_media"):
        if key in attrs:
            data = json.loads(attrs[key])
            if isinstance(data, dict) and "hsgp_class" in data and "__type__" not in data:
                cls_name = data.pop("hsgp_class")
                data["__type__"] = _HSGP_CLASS_MAP.get(cls_name, cls_name)
                attrs[key] = json.dumps(data)

    # Migrate mu_effects: class → __type__
    if "mu_effects" in attrs:
        effects = json.loads(attrs["mu_effects"])
        for effect in effects:
            if isinstance(effect, dict) and "class" in effect and "__type__" not in effect:
                cls_name = effect.pop("class")
                effect["__type__"] = _MUEFFECT_CLASS_MAP.get(cls_name, cls_name)
        attrs["mu_effects"] = json.dumps(effects)

    # Drop stale id (hash changes with new to_dict() output)
    attrs.pop("id", None)

    # Set version
    attrs["__serialization_version__"] = str(CURRENT_VERSION)

    return attrs


def migrate_idata(idata: az.InferenceData) -> az.InferenceData:
    """Migrate InferenceData attrs from old format to current version.

    Parameters
    ----------
    idata : az.InferenceData
        The InferenceData to migrate. Modified in-place and returned.

    Returns
    -------
    az.InferenceData
        The same object, with attrs updated to current version.
    """
    version_str = idata.attrs.get("__serialization_version__", "0")
    version = int(version_str)

    if version >= CURRENT_VERSION:
        return idata

    migrations = {0: _migrate_v0_to_v1}

    while version < CURRENT_VERSION:
        if version not in migrations:
            raise ValueError(
                f"No migration path from version {version} to {version + 1}"
            )
        idata.attrs = migrations[version](idata.attrs)
        version += 1

    return idata


def main() -> None:
    """CLI entry point: python -m pymc_marketing.migrate <file.nc>"""
    if len(sys.argv) != 2:
        print("Usage: python -m pymc_marketing.migrate <model.nc>")
        sys.exit(1)

    fname = sys.argv[1]
    print(f"Loading {fname}...")
    idata = az.from_netcdf(fname)

    version = idata.attrs.get("__serialization_version__", "0")
    if int(version) >= CURRENT_VERSION:
        print(f"Already at version {version}, nothing to do.")
        return

    print(f"Migrating from v{version} to v{CURRENT_VERSION}...")
    migrate_idata(idata)
    idata.to_netcdf(fname)
    print(f"Done. Saved migrated model to {fname}")


if __name__ == "__main__":
    main()
```

> **Important:** The `_ADSTOCK_TYPE_MAP` and `_SATURATION_TYPE_MAP` values must match exactly what each class's `to_dict()` emits as `__type__`. Verify by checking a few classes: e.g., `GeometricAdstock().to_dict()["__type__"]` should equal `"pymc_marketing.mmm.components.adstock.GeometricAdstock"`.

**Step 4: Run test to verify it passes**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/test_migration.py -v`
Expected: PASS

**Step 5: Wire auto-migration into load**

Override `load_from_idata()` on the MMM class in `pymc_marketing/mmm/multidimensional.py`:

```python
    @classmethod
    def load_from_idata(cls, idata: az.InferenceData, check: bool = True) -> "MMM":
        """Load from InferenceData, auto-migrating old formats."""
        from pymc_marketing.migrate import CURRENT_VERSION, migrate_idata

        version = int(idata.attrs.get("__serialization_version__", "0"))
        if version < CURRENT_VERSION:
            warnings.warn(
                f"Loading a model saved with serialization format v{version}. "
                f"Migrating to v{CURRENT_VERSION}. Re-save the model to avoid "
                "this warning in the future.",
                DeprecationWarning,
                stacklevel=2,
            )
            migrate_idata(idata)

        return super().load_from_idata(idata, check=check)
```

**Step 6: Write test for auto-migration on load**

Add to `tests/test_migration.py`:

```python
class TestAutoMigrationOnLoad:
    """Verify that MMM.load() auto-migrates old-format files."""

    def test_load_v0_emits_deprecation_warning(self, tmp_path):
        """Loading a v0 file should emit a DeprecationWarning."""
        # Create a v0 idata file
        # (implementer: build minimal valid v0 idata and save to tmp_path)
        # Then:
        # with pytest.warns(DeprecationWarning, match="Migrating to"):
        #     model = MMM.load(str(fname))
        pass  # Implementer fills in with real fixture
```

> **Note to implementer:** This test may require a fairly heavy fixture (a full MMM save file). Consider using pytest's `tmp_path` and building the idata programmatically with all required attrs. See the `_make_v0_idata()` helper from the earlier test class for the attrs structure, but you'll also need `fit_data`, `posterior`, etc. groups to make `build_from_idata()` succeed.

**Step 7: Run tests**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/test_migration.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add pymc_marketing/migrate.py pymc_marketing/mmm/multidimensional.py tests/test_migration.py
git commit -m "feat(serialization): add migration tool for v0→v1 format conversion"
```

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

## Task 8: Old Code Removal — Component Metaclasses, Lookup Dicts, and `register_deserialization` Calls

**Files:**
- Modify: `pymc_marketing/mmm/components/base.py` — remove `RegistrationMeta`, `create_registration_meta`, `DuplicatedTransformationError`, `lookup_name`
- Modify: `pymc_marketing/mmm/components/adstock.py` — remove `ADSTOCK_TRANSFORMATIONS`, `AdstockRegistrationMeta`, `adstock_from_dict`, `_is_adstock`, `register_deserialization` call, `lookup_name` from classes
- Modify: `pymc_marketing/mmm/components/saturation.py` — same pattern
- Modify: `pymc_marketing/mmm/events.py` — remove `BASIS_TRANSFORMATIONS`, `BasisMeta`, `basis_from_dict`, `_is_basis`, `register_deserialization` calls
- Modify: `pymc_marketing/mmm/hsgp.py` — remove `hsgp_from_dict`, `_is_hsgp`, `register_deserialization` call
- Modify: `pymc_marketing/mmm/fourier.py` — remove 3 `register_deserialization` calls + `_is_*` guards
- Modify: `pymc_marketing/mmm/media_transformation.py` — remove 2 `register_deserialization` calls + `_is_*` guards
- Modify: `pymc_marketing/mmm/hsgp_kwargs.py` — remove `register_deserialization` call + `_is_hsgp_kwargs`
- Test: run all mmm tests

> **Critical:** Do NOT remove the 5 `register_deserialization` calls in `pymc_marketing/prior.py` (1) and `pymc_marketing/special_priors.py` (4). These feed the `pymc_extras` global deserializer used by `parse_model_config()` across all model families.

**Step 1: Remove `RegistrationMeta` infrastructure from `base.py`**

In `pymc_marketing/mmm/components/base.py`:
- Delete `DuplicatedTransformationError` class (lines 685–691)
- Delete `create_registration_meta()` function (lines 694–726)
- Remove `lookup_name: str` from `Transformation` class (line 130)
- Remove `lookup_name` from `to_dict()` output (line 180)
- Remove `lookup_name` check in `_has_all_attributes()` (lines 307–308)

**Step 2: Remove metaclasses and lookup dicts from `adstock.py`**

In `pymc_marketing/mmm/components/adstock.py`:
- Delete `ADSTOCK_TRANSFORMATIONS` dict (line 82)
- Delete `AdstockRegistrationMeta` (line 84)
- Change `AdstockTransformation` metaclass from `metaclass=AdstockRegistrationMeta` to no metaclass (line 87)
- Remove `lookup_name` from each concrete class: `GeometricAdstock` (line 219), `DelayedAdstock` (line 258), `WeibullCDFAdstock` (line 297), `WeibullPDFAdstock` (line 340), `GammaCDFAdstock` (line 384), `GammaPDFAdstock` (line 409)
- Remove `data.pop("lookup_name", None)` from `from_dict()` (line 150)
- Delete `adstock_from_dict()` function (lines 423–432)
- Delete `_is_adstock()` function (lines 435–436)
- Delete `register_deserialization(...)` call (lines 438–441)
- Remove `from pymc_extras.deserialize import register_deserialization` import if unused

**Step 3: Remove metaclasses and lookup dicts from `saturation.py`**

Same pattern as adstock:
- Delete `SATURATION_TRANSFORMATIONS` dict (line 102)
- Delete `SaturationRegistrationMeta` (line 104)
- Change `SaturationTransformation` metaclass (line 107)
- Remove `lookup_name` from each concrete class
- Remove `data.pop("lookup_name", None)` from `from_dict()` (line 162)
- Delete `saturation_from_dict()`, `_is_saturation()`, `register_deserialization()` call
- Remove unused imports

**Step 4: Remove metaclass and lookup dict from `events.py`**

In `pymc_marketing/mmm/events.py`:
- Delete `BASIS_TRANSFORMATIONS` dict (line 120)
- Delete `BasisMeta` (line 121)
- Change `Basis` metaclass (line 124)
- Delete `basis_from_dict()`, `_is_basis()`, and the basis `register_deserialization()` call (lines 188–208)
- Delete the EventEffect `register_deserialization()` call (lines 276–279) plus its `_is_event_effect()` guard

**Step 5: Remove `hsgp_from_dict` from `hsgp.py`**

In `pymc_marketing/mmm/hsgp.py`:
- Delete `hsgp_from_dict()` function (lines 1488–1502)
- Delete `_is_hsgp()` function (lines 1504–1509)
- Delete `register_deserialization(...)` call (line 1512)

**Step 6: Remove `register_deserialization` calls from `fourier.py`**

In `pymc_marketing/mmm/fourier.py`:
- Delete 3 `register_deserialization(...)` calls (lines 1011–1023) and their `_is_*` guard functions

**Step 7: Remove `register_deserialization` calls from `media_transformation.py`**

In `pymc_marketing/mmm/media_transformation.py`:
- Delete 2 `register_deserialization(...)` calls (lines 279–282, 545–548) and their `_is_*` guard functions

**Step 8: Remove `register_deserialization` call from `hsgp_kwargs.py`**

In `pymc_marketing/mmm/hsgp_kwargs.py`:
- Delete `register_deserialization(...)` call (lines 122–125) and its `_is_hsgp_kwargs` guard

**Step 9: Clean up imports across all modified files**

Remove `from pymc_extras.deserialize import register_deserialization` (and `deserialize` if now unused) from all files where the calls were removed. Verify no other code in each file depends on these imports.

**Step 10: Run tests**

Run: `/Users/imrisofer/miniconda3/envs/pymc-dev-2379/bin/python -m pytest tests/mmm/ -x -q --timeout=600`
Expected: PASS

> **If tests fail:** The most likely cause is a remaining import or reference to the removed code. Check import statements in test files, conftest.py, and multidimensional.py.

**Step 11: Verify removal is complete**

Run these greps to confirm zero hits:

```bash
rg "RegistrationMeta|create_registration_meta" pymc_marketing/
rg "lookup_name" pymc_marketing/mmm/components/ pymc_marketing/mmm/events.py
rg "ADSTOCK_TRANSFORMATIONS|SATURATION_TRANSFORMATIONS|BASIS_TRANSFORMATIONS" pymc_marketing/
rg "_serialize_mu_effect|_MUEFFECT_DESERIALIZERS|_deserialize_mu_effect" pymc_marketing/
rg "adstock_from_dict|saturation_from_dict|basis_from_dict|hsgp_from_dict" pymc_marketing/
```

All should return 0 results (or only docs/plans references).

**Step 12: Commit**

```bash
git add -A
git commit -m "refactor(serialization): remove all legacy metaclasses, lookup dicts, and register_deserialization calls"
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

## Task 10: Test Cleanup — Remove Diagnostic File and Consolidate TypeRegistry Tests

**Files:**
- Delete: `tests/mmm/test_serialization_issues.py`
- Modify: `tests/mmm/test_serialization_roundtrips.py` (add missing unique tests)
- Modify: `tests/mmm/components/test_adstock.py` (remove `TestAdstockTypeRegistry`)
- Modify: `tests/mmm/components/test_saturation.py` (remove `TestSaturationTypeRegistry`)
- Modify: `tests/mmm/test_hsgp.py` (remove `TestHSGPTypeRegistry`, `TestDeferredFactoryInHSGP`, `TestHSGPKwargsTypeRegistry`)
- Modify: `tests/mmm/test_fourier.py` (remove `TestFourierTypeRegistry`)
- Modify: `tests/mmm/test_events.py` (remove `TestEventsTypeRegistry`)
- Modify: `tests/mmm/test_media_transformation.py` (remove `TestMediaTransformationTypeRegistry`)
- Modify: `tests/mmm/test_additive_effect.py` (remove `TestMuEffectTypeRegistry`, `TestLinearTrendEffectSerialization`, `TestFourierEffectSerialization`, `TestEventAdditiveEffectSerialization`)
- Modify: `tests/mmm/test_linear_trend.py` (remove `TestLinearTrendTypeRegistry`, `TestScalingTypeRegistry`)

**Step 1: Delete `test_serialization_issues.py`**

This file is marked "THIS FILE SHOULD NOT BE MERGED TO THE MAIN REPO." Remove it entirely.

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

**Step 2: Verify all old code is gone**

```bash
rg "RegistrationMeta|create_registration_meta" pymc_marketing/ --type py
rg "lookup_name" pymc_marketing/mmm/components/ pymc_marketing/mmm/events.py --type py
rg "ADSTOCK_TRANSFORMATIONS|SATURATION_TRANSFORMATIONS|BASIS_TRANSFORMATIONS" pymc_marketing/ --type py
rg "_serialize_mu_effect|_MUEFFECT_DESERIALIZERS|_deserialize_mu_effect" pymc_marketing/ --type py
rg "adstock_from_dict|saturation_from_dict|basis_from_dict|hsgp_from_dict" pymc_marketing/ --type py
```

All should return 0 results.

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
- [ ] **No old code:** All items in Task 8 removal list are gone (verified by grep)
- [ ] **Non-MMM models unaffected:** CLV and Customer Choice test suites pass without changes
- [ ] **No redundant tests:** test_serialization_issues.py removed; per-component TypeRegistry classes consolidated
- [ ] **__serialization_version__** attr present in all new saves

## Risk Notes

- **ArviZ supplementary data group naming:** Verify that ArviZ supports the chosen naming convention (`supplementary_data/prefix` or `supplementary_data_prefix`) for both saving and loading. Test with both `netcdf4` and `h5netcdf` engines.
- **Migration map completeness:** The `_ADSTOCK_TYPE_MAP` and `_SATURATION_TYPE_MAP` in `migrate.py` must include every concrete subclass that has ever been saved. Cross-reference with the `lookup_name` values in each concrete class.
- **Backward compatibility during transition:** Tasks 3–5 add fallback branches (`if "__type__" in data: ... else: legacy_from_dict(...)`) that handle both old and new formats. These fallbacks are removed in Task 7 once the migration tool is in place. Verify tests pass at each step.
- **Import ordering:** Phase 1 added `from pymc_marketing.serialization import registry` to many module files. These imports happen at module load time. Verify no circular import issues arise when `model_builder.py` also imports from `serialization.py`.
- **`parse_model_config()` interaction:** The updated `_model_config_formatting()` may deserialize Priors via `registry.deserialize()` before `parse_model_config()` processes them. Verify that `parse_model_config()` handles already-deserialized Prior objects gracefully (skips them or returns as-is).
