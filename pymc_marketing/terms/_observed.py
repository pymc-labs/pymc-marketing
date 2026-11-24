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
"""Family-friendly constructor for observed equations.

:func:`Observed` is a thin factory over :class:`~pymc_marketing.terms.Equation`:
every call returns an ``Equation`` and no separate type exists downstream, so
serialization, ``SCM``, and prediction all see one class. It exists so a
likelihood statement reads the way it is written on a whiteboard:

``converted ~ Binomial(entered, p)`` becomes
``Observed("Binomial", observed="converted", n="entered", p=<expression>)``.

Keyword arguments are the distribution's parameters, so the constructor is not
specialized to the Binomial family: ``Observed("Poisson", observed="sales",
mu=<expression>)`` takes the same code path. Family-specific algebra (for
example ``alpha = kappa * p`` for a BetaBinomial parameterized by a rate and a
concentration) belongs to the builder that wants it, expressed as ordinary
keyword arguments.

Validation is delegated, not reinvented: the family name resolves through
``pymc_extras.prior``'s distribution lookup, parameter names are checked
against the ``pymc.dims`` distribution's own signature (the same spirit as
``Prior._parameters_are_at_least_subset_of_pymc``), parameter overlap with the
likelihood recipe is checked by ``Equation``, and data references resolve
against the dataset at build time via ``Data`` terms.
"""

from __future__ import annotations

import inspect
from typing import Any

from pymc_extras.prior import (
    Prior,
    _get_pymc_dim_distribution,
)

from pymc_marketing.terms._graph import Data, Equation

__all__ = ["Observed"]


def _distribution_parameters(family: str) -> set[str]:
    """Parameter names of the ``pymc.dims`` distribution ``family``."""
    pmd_distribution = _get_pymc_dim_distribution(family)
    signature = inspect.signature(pmd_distribution.dist)
    return set(signature.parameters) - {"args", "kwargs"}


def Observed(
    dist: str | Prior,
    *,
    observed: str,
    **params: Any,
) -> Equation:
    """Declare an observed equation with family-friendly slots.

    Parameters
    ----------
    dist : str or pymc_extras.prior.Prior
        Likelihood family name, or a ``Prior`` recipe for explicit
        hyperpriors. A string resolves through ``pymc_extras.prior``'s
        distribution lookup, so registered custom distributions work here
        too.
    observed : str
        Dataset observation variable.
    **params : Any
        The distribution's parameters. A string value names a dataset
        variable (a ``Data`` reference); anything else is built as a model
        expression, so priors, terms, composed recipes, and literals all
        work.

    Returns
    -------
    pymc_marketing.terms.Equation
        The equation. There is no ``Observed`` type: the returned object is
        an ``Equation`` in every downstream path.

    Raises
    ------
    TypeError
        If ``observed`` is not a nonempty string or ``dist`` is neither a
        family name nor a ``Prior``.
    ValueError
        If a keyword argument is not a parameter of the family.
    UnsupportedDistributionError
        If the family is unknown to ``pymc_extras.prior``.

    Examples
    --------
    A funnel step conversion, ``converted ~ Binomial(entered, p)``:

    .. code-block:: python

        from pymc_marketing.terms import (
            Intercept,
            Observed,
            Parameter,
        )

        p = Transform(
            Parameter("step_logit", Prior("Normal", dims="step")),
            sigmoid,
        )
        equation = Observed(
            "Binomial",
            observed="converted",
            n="entered",
            p=p,
        )

    A count outcome with the same mechanism:

    .. code-block:: python

        Observed("Poisson", observed="sales", mu=Data("exposure") * rate)
    """
    if not isinstance(observed, str) or not observed:
        raise TypeError("observed must be a nonempty dataset variable name.")
    if isinstance(dist, Prior):
        likelihood = dist
        family = dist.distribution
    elif isinstance(dist, str):
        likelihood = Prior(dist)
        family = dist
    else:
        raise TypeError(
            f"dist must be a family name or a Prior recipe, got {type(dist).__name__}."
        )
    valid = _distribution_parameters(family)
    unknown = set(params) - valid
    if unknown:
        raise ValueError(
            f"{family!r} has no parameters {sorted(unknown)}; "
            f"its parameters are {sorted(valid)}."
        )
    resolved = {
        key: Data(value) if isinstance(value, str) else value
        for key, value in params.items()
    }
    return Equation(observed=observed, likelihood=likelihood, parameters=resolved)
