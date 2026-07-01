#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Shared numerical primitives for customer-choice models.

These helpers are intentionally framework-light so that:

- The same numerically stable softmax is used by every choice model in this
  package, removing the drift between models that previously re-implemented
  the subtract-max trick (or, in some cases, omitted it).
- Non-centered Normal reparameterisation has a single, named entry point
  for the funnel-prone hierarchical blocks used by Mixed Logit and BLP.
- Quasi-Monte-Carlo draws used to integrate over consumer heterogeneity are
  produced from one canonical generator (Owen-scrambled Halton mapped through
  the standard-normal inverse CDF), so simulation noise is comparable across
  models and runs.
"""

from collections.abc import Sequence

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
from scipy.stats import norm, qmc


def stable_softmax(
    logits: TensorVariable,
    axis: int = -1,
    *,
    name: str | None = None,
    dims: str | Sequence[str] | None = None,
) -> TensorVariable:
    """Numerically stable softmax along ``axis``.

    Subtracts the per-slice max before exponentiating to bound the exponent
    away from overflow. If ``name`` is given, the result is wrapped in a
    ``pm.Deterministic`` with the supplied ``dims`` (so it appears in the
    InferenceData posterior group).
    """
    centered = logits - pt.max(logits, axis=axis, keepdims=True)
    probs = pm.math.softmax(centered, axis=axis)
    if name is not None:
        return pm.Deterministic(name, probs, dims=dims)
    return probs


def non_centered_normal(
    name: str,
    mu: TensorVariable | float,
    sigma: TensorVariable | float,
    dims: str | Sequence[str] | None = None,
    *,
    raw_suffix: str = "_raw",
) -> TensorVariable:
    """Non-centered Normal reparameterisation: ``mu + sigma * z``, ``z ~ N(0, 1)``.

    Creates the standard-normal raw variable as ``f"{name}{raw_suffix}"`` and
    returns the realised value as a ``pm.Deterministic`` named ``name`` (so
    the trace contains the value the user reasons about, not the raw).
    """
    z = pm.Normal(f"{name}{raw_suffix}", 0.0, 1.0, dims=dims)
    return pm.Deterministic(name, mu + sigma * z, dims=dims)


def halton_draws(
    n_draws: int,
    n_dims: int,
    *,
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
    skip: int = 0,
) -> np.ndarray:
    """Owen-scrambled Halton draws mapped to standard normals.

    Returns an ``(n_draws, n_dims)`` array of approximately N(0, 1) variates
    with low-discrepancy joint coverage. Owen scrambling fixes the well-known
    correlation pathology of plain Halton in dimensions ``>= 7``.

    Parameters
    ----------
    n_draws
        Number of consumer-heterogeneity draws.
    n_dims
        Dimension of the integration (typically the number of random
        coefficients).
    scramble
        Whether to Owen-scramble. Defaults to ``True``; only set ``False`` for
        bit-exact reproduction of legacy Halton results.
    seed
        Seed or ``np.random.Generator`` for the scrambling permutation.
    skip
        Number of leading Halton points to discard. Useful when
        ``n_dims`` is large to avoid the highly-correlated low-index region.
    """
    sampler = qmc.Halton(d=n_dims, scramble=scramble, seed=seed)
    if skip > 0:
        sampler.fast_forward(skip)
    u = sampler.random(n_draws)
    eps = np.finfo(np.float64).tiny
    u = np.clip(u, eps, 1.0 - eps)
    return norm.ppf(u)
