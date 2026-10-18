# Consideration intercept migration

## Release note

`ConsiderationSetMixedLogit` now names its optional consideration intercept `gamma_consider_0`, in both the PyMC model and `model_config`.
This fixes the duplicate-variable error when `consideration_intercept=True` is combined with `instrumental_vars` for price endogeneity correction.
The price intercept keeps the name `gamma_0`.

## Custom priors

Use `model_config={"gamma_consider_0": prior}` when specifying a consideration intercept prior.
The old key `gamma_0` is still accepted with a `DeprecationWarning` and mapped to the new key, preserving the custom prior.
If both keys are supplied, `gamma_consider_0` takes precedence.
The caller's configuration dictionary is not modified.

## Reusing fitted posteriors

For a fit created before this rename with `consideration_intercept=True`, rename the posterior variable before attaching the saved inference data to the rebuilt model:

```python
idata["posterior"] = (
    idata["posterior"]
    .to_dataset()
    .rename_vars({"gamma_0": "gamma_consider_0"})
)
```

Without this migration, PyMC cannot match the saved consideration intercept to `gamma_consider_0` and may draw it from the prior during posterior predictive sampling.
This is a manual migration of the posterior variable; it does not add save/load support to this class.

Do not rename `gamma_0` in fits that used only `instrumental_vars`: that variable is the price intercept and is unchanged.
Models without `consideration_intercept=True` do not need a consideration-intercept migration.
