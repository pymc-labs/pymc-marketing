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
``Prior._parameters_are_at_least_subset_of_pymc``), required parameters without
defaults are checked at construction instead of at fit, parameter overlap with
the likelihood recipe is checked by ``Equation``, and data references resolve
against the dataset at build time via ``Data`` terms.
"""

from __future__ import annotations

import inspect
from typing import Any

from pymc_extras.prior import (
    Prior,
    UnsupportedDistributionError,
    _get_pymc_dim_distribution,
)

from pymc_marketing.terms._graph import Data, Equation

__all__ = ["Observed"]

# Keyword-like arguments of some ``dist`` signatures that are not distribution
# parameters: they belong on the ``Prior`` recipe, not on the equation.
_NON_PARAMETERS = frozenset({"core_dims", "lower"})


def _distribution_parameters(family: str) -> tuple[set[str], set[str]]:
    """Parameter names of the ``pymc.dims`` distribution ``family``.

    Returns the accepted parameter names and the subset that has no default
    and therefore must be supplied somewhere.
    """
    pmd_distribution = _get_pymc_dim_distribution(family)
    signature = inspect.signature(pmd_distribution.dist)
    names: set[str] = set()
    required: set[str] = set()
    for key, value in signature.parameters.items():
        if key in {"args", "kwargs"} | _NON_PARAMETERS:
            continue
        names.add(key)
        if value.default is inspect.Parameter.empty:
            required.add(key)
    return names, required


def _resolve_likelihood(dist: str | Prior) -> Prior:
    """Turn a family name into a ``Prior``, rejecting non-distributions."""
    if not isinstance(dist, str):
        return dist
    try:
        return Prior(dist)
    except AttributeError as error:
        # Names that exist on pymc but are not distributions, like "math".
        raise UnsupportedDistributionError(
            f"{dist!r} is not a distribution PyMC knows."
        ) from error


def Observed(
    dist: str | Prior,
    *,
    observed: str,
    name: str | None = None,
    dims: Any = None,
    **params: Any,
) -> Equation:
    """Declare an observed equation with family-friendly slots.

    Parameters
    ----------
    dist : str or pymc_extras.prior.Prior
        Likelihood family name, or a ``Prior`` recipe for explicit
        hyperpriors. A transformed or non-centered ``Prior`` is rejected:
        an observed equation always conditions on data.
    observed : str
        Dataset observation variable.
    name : str, optional
        Model variable name of the equation. Defaults to ``observed``.
        Needed when one equation's observation is the next one's data and
        the default names would collide with a ``Data`` reference.
    dims : tuple of str, optional
        Output dimensions of the equation. When omitted, the dimensions of
        the observation variable are used.
    **params : Any
        The distribution's parameters. A string value names a dataset
        variable (a ``Data`` reference); anything else is built as a model
        expression, so priors, terms, composed recipes, and literals all
        work. Distribution keyword-like arguments such as ``core_dims``
        belong on the ``Prior`` recipe instead.

    Returns
    -------
    pymc_marketing.terms.Equation
        The equation. There is no ``Observed`` type: the returned object is
        an ``Equation`` in every downstream path.

    Raises
    ------
    TypeError
        If ``observed`` is not a string or ``dist`` is neither a family name
        nor a ``Prior``.
    ValueError
        If a keyword argument is not a parameter of the family, a required
        parameter is missing, or the family is unknown.

    Notes
    -----
    A ``mu``-only call leaves the remaining parameters at the ``pymc.dims``
    defaults. For example, ``Observed("Normal", observed="y", mu=mu)`` fixes
    ``sigma`` at 1; pass ``sigma=Prior("HalfNormal")`` to learn a scale.

    Examples
    --------
    A funnel step conversion, ``converted ~ Binomial(entered, p)``:

    .. code-block:: python

        import pytensor.xtensor as ptx
        from pymc_extras.prior import Prior
        from pymc_marketing.terms import (
            Data,
            Intercept,
            Observed,
            Parameter,
            Transform,
        )

        p = Transform(
            Parameter("step_logit", Prior("Normal", dims="step")),
            ptx.math.sigmoid,
        )
        equation = Observed(
            "Binomial",
            observed="converted",
            n="entered",
            p=p,
        )

    A count outcome with the same mechanism:

    .. code-block:: python

        rate = Parameter("rate", Prior("Gamma", alpha=2, beta=2, dims="store"))
        equation = Observed(
            "Poisson",
            observed="sales",
            name="sales_obs",
            mu=Data("exposure") * rate,
        )

    A Normal outcome with an explicit scale:

    .. code-block:: python

        equation = Observed(
            "Normal",
            observed="revenue",
            mu=Intercept() + response,
            sigma=Prior("HalfNormal", sigma=2),
        )
    """
    if not isinstance(observed, str):
        raise TypeError("observed must be a dataset variable name.")
    if not observed:
        raise ValueError("observed must be a nonempty dataset variable name.")
    if isinstance(dist, Prior):
        if dist.transform or not dist.centered:
            raise ValueError(
                "Observed equations do not support transformed or "
                "noncentered likelihood recipes."
            )
        likelihood = dist
        family = dist.distribution
    elif isinstance(dist, str):
        likelihood = Prior(dist)
        family = dist
    else:
        raise TypeError(
            f"dist must be a family name or a Prior recipe, got {type(dist).__name__}."
        )
    accepted, required = _distribution_parameters(family)
    unknown = set(params) - accepted
    if unknown:
        raise ValueError(
            f"{family!r} has no parameters {sorted(unknown)}; "
            f"its parameters are {sorted(accepted)}."
        )
    missing = required - set(params) - set(likelihood.parameters)
    if missing:
        raise ValueError(
            f"{family!r} needs parameters {sorted(missing)}; pass them as "
            "keyword arguments or set them on the Prior recipe."
        )
    resolved = {
        key: Data(value) if isinstance(value, str) else value
        for key, value in params.items()
    }
    return Equation(
        observed=observed,
        name=name,
        dims=dims,
        likelihood=likelihood,
        parameters=resolved,
    )
