
## Serialization Workflow of the MMM Model

The serialization is built on a **layered architecture**: `ModelIO` (mixin) → `ModelBuilder` → `RegressionModelBuilder` → `MMM`. Each layer adds its own data to a single carrier — an ArviZ `InferenceData` object stored as a NetCDF file.

### Saving (`mmm.save("model.nc")`)

The flow has three phases:

**Phase 1 — Build the attribute dictionary** (`create_idata_attrs`)

The base `ModelIO.create_idata_attrs` creates a dict with five core keys:

```278:343:pymc_marketing/model_builder.py
    def set_idata_attrs(
        self, idata: az.InferenceData | None = None
    ) -> az.InferenceData:
        # ...
        attrs = self.create_idata_attrs()
        # ...
        idata.attrs = attrs
        return idata
```

The base method serializes:
- `id` — a SHA-256 hash of the model config + version + model type
- `model_type`, `version`
- `sampler_config` — JSON-dumped
- `model_config` — JSON-dumped via `_serializable_model_config`

Then `MMM.create_idata_attrs` **extends** this with all MMM-specific constructor args:

```1018:1073:pymc_marketing/mmm/multidimensional.py
    def create_idata_attrs(self) -> dict[str, str]:
        """Return the idata attributes for the model."""
        attrs = super().create_idata_attrs()
        attrs["dims"] = json.dumps(self.dims)
        attrs["date_column"] = self.date_column
        attrs["adstock"] = json.dumps(self.adstock.to_dict())
        attrs["saturation"] = json.dumps(self.saturation.to_dict())
        attrs["adstock_first"] = json.dumps(self.adstock_first)
        attrs["control_columns"] = json.dumps(self.control_columns)
        attrs["channel_columns"] = json.dumps(self.channel_columns)
        attrs["yearly_seasonality"] = json.dumps(self.yearly_seasonality)
        # ... time_varying_intercept, time_varying_media, scaling, dag, etc.
        attrs["mu_effects"] = json.dumps(mu_effects_list)
        # ... cost_per_unit as DataFrame→JSON (split orient)
        return attrs
```

Every value in the dict is a **JSON string** — this is because NetCDF attributes only support scalars and strings.

**Phase 2 — Stamp onto InferenceData** (`set_idata_attrs`)

`set_idata_attrs` takes the dict from Phase 1 and assigns it to `idata.attrs`. It also validates that every `__init__` parameter has a corresponding key in the attrs, ensuring nothing is lost on round-trip.

**Phase 3 — Write to disk** (`save`)

```345:398:pymc_marketing/model_builder.py
    def save(self, fname: str, **kwargs) -> None:
        # ...
        if self.idata is not None and "posterior" in self.idata:
            file = Path(str(fname))
            self.idata.to_netcdf(str(file), **kwargs)
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
```

This writes the entire `InferenceData` (posterior samples, fit data, predictions, **and** the attrs dict) as a single `.nc` file using ArviZ's NetCDF backend.

### Loading (`MMM.load("model.nc")`)

The load path is essentially the reverse:

```449:563:pymc_marketing/model_builder.py
    @classmethod
    def load(cls, fname: str, check: bool = True):
        # ...
        idata = from_netcdf(filepath)
        return cls.load_from_idata(idata, check=check)

    @classmethod
    def load_from_idata(cls, idata, check=True):
        init_kwargs = cls.idata_to_init_kwargs(idata)   # Step 1: attrs → kwargs
        model = cls(**init_kwargs)                       # Step 2: re-instantiate
        model.idata = idata                              # Step 3: attach idata
        model.build_from_idata(idata)                    # Step 4: rebuild PyMC model graph
        # Step 5: verify id match
```

**Step 1 — Deserialize attrs to kwargs** (`attrs_to_init_kwargs`)

This is where `MMM` does the heavy lifting of reversing the serialization:

```1110:1142:pymc_marketing/mmm/multidimensional.py
    @classmethod
    def attrs_to_init_kwargs(cls, attrs: dict[str, str]) -> dict[str, Any]:
        return {
            "model_config": cls._model_config_formatting(json.loads(attrs["model_config"])),
            "date_column": attrs["date_column"],
            "adstock": adstock_from_dict(json.loads(attrs["adstock"])),
            "saturation": saturation_from_dict(json.loads(attrs["saturation"])),
            # ... all other MMM constructor args ...
            "cost_per_unit": _deserialize_cost_per_unit(attrs["cost_per_unit"]),
        }
```

Each JSON string is parsed back into its native Python type — `adstock_from_dict` reconstructs the adstock transformation object, `saturation_from_dict` the saturation, `hsgp_from_dict` the HSGP objects, etc.

**Step 2 — Reconstruct the model** (`cls(**init_kwargs)`)

A fresh `MMM` instance is created with the deserialized kwargs — identical to how the user originally constructed it.

**Step 3 & 4 — Reattach idata + rebuild PyMC graph** (`build_from_idata`)

The `RegressionModelBuilder.build_from_idata` extracts `X` and `y` from `idata.fit_data` and calls `build_model(X, y)` to rebuild the PyMC model graph (needed for posterior predictive sampling, etc.).

**Step 5 — Integrity check**

The freshly built model's `id` hash is compared against the `id` stored in `idata.attrs`. If they don't match, a `DifferentModelError` is raised — this catches cases where the code has changed since the model was saved.

### Special serialization for complex objects

A few object types have dedicated serialize/deserialize handlers:

- **`MuEffect` subclasses** (FourierEffect, LinearTrendEffect, EventAdditiveEffect) — handled by the `@singledispatch` pattern in `_serialize_mu_effect` / `_deserialize_mu_effect` with a registry of type-specific handlers
- **Adstock / Saturation transformations** — use their own `to_dict()` / `from_dict()` protocol
- **HSGP objects** — serialized via `to_dict()` with an added `hsgp_class` discriminator key
- **Priors** — use `Prior.to_dict()` / `deserialize()`
- **`cost_per_unit`** — serialized as a DataFrame JSON (split orient), with a warning if timezone-aware dates are present
- **`model_config`** values — recursively serialized in `_serializable_model_config`, converting numpy arrays to lists, Priors via `.to_dict()`, Pydantic models via `.model_dump()`

### Summary diagram

```
SAVE:  MMM → create_idata_attrs() → {JSON strings} → idata.attrs → .to_netcdf("model.nc")
LOAD:  .nc file → from_netcdf() → idata.attrs → attrs_to_init_kwargs() → MMM(**kwargs) → build_from_idata() → verify id
```

The key design principle is that **everything needed to reconstruct the model is stored as JSON strings in the InferenceData attributes**, and the NetCDF file is the single artifact containing both the MCMC samples and the full model specification.
