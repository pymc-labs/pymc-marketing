# API

## Modules
```{eval-rst}
.. currentmodule:: pymc_marketing
.. autosummary::
  :recursive:
  :toctree: generated/

  bass
  causal_utils
  clv
  customer_choice
  data
  experimental
  hsgp_kwargs
  metrics
  mlflow
  mmm
  model_builder
  model_config
  model_graph
  paths
  pie
  plot
  pytensor_utils
  r2d2
  serialization
  serialization_migration
  special_priors
  terms
  terms_gp
  utils
```

## Experimental graph-first terms

`pymc_marketing.experimental.terms` provides graph-first Structural Causal Models (SCMs) and term-first optimization.
Shared primitives remain in `pymc_marketing.terms`, and MMM components remain in `pymc_marketing.mmm`.
These APIs are under active development and do not replace the stable `MMM` optimizer.
See the [term-first optimization guide](../guide/mmm/experimental_optimizer.md) for the `Problem` → `evaluate` → `solve` workflow.

```{eval-rst}
.. currentmodule:: pymc_marketing.experimental.terms
.. autosummary::
  :toctree: generated/

  SCM
  Data
  Equation
  MediaTransform
  OptimizationResult
  Problem
```
