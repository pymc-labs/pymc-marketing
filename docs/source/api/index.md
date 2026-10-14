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

## Experimental term-first optimization

These APIs are under active development and do not replace the stable `MMM` optimizer.
See the [term-first optimization guide](../guide/mmm/experimental_optimizer.md) for the `Problem` → `evaluate` → `solve` workflow.

```{eval-rst}
.. currentmodule:: pymc_marketing.mmm.experimental
.. autosummary::
  :toctree: generated/

  Problem
  OptimizationResult
```
