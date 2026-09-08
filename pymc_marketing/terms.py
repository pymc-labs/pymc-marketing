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

"""Composable model terms for building ``pymc.dims`` subgraphs from xarray data.

``pymc_marketing.terms`` provides a small set of built-in pieces
(``Parameter`` / ``Intercept``, ``Dot``, ``Transform``) that compose with
``+``, ``*``, and ``-`` into predictors.  **This module is not a full
modeling toolkit** --- it does not ship complete hierarchical,
domain-specific, or observation-model wrappers.  It is a **thin, expressive
core** that lets you build what you need by subclassing ``ModelTerm`` and
reusing the same helpers and operators that built-ins use.  If a term is
missing from this package, that is the intended path: write a custom term,
compose it with ``+`` / ``*`` / ``-``, and the framework handles coords,
data registration, tensor construction, and prediction updates.

The framework is ``pymc.dims``-native.  ``create_variable()`` returns an
``XTensorVariable`` (dimensional tensor).  Built-in terms use ``pmd.Data``
for shared data and ``pmd.Normal`` with ``xdist=True`` for variables.
Custom terms should use ``pymc.dims`` and ``pytensor.xtensor`` throughout
to ensure compatibility during composition.

Extending
---------
This module is designed so that **the parts you do not find here are the
parts you are meant to write.**  ``ModelTerm`` is the extension base;
subclass it, implement five lifecycle methods, and the result composes
with the built-ins via ``+`` / ``*`` / ``-`` --- no framework changes
required.

Lifecycle (per term)
^^^^^^^^^^^^^^^^^^^^
- ``get_coords(ds)`` -- return coordinates from data variables (outside model)
- ``add_coords(ds)`` -- add model-only coordinates, if not in dataset (inside model context)
- ``register_data(ds)`` -- register ``pmd.Data`` shared variables (inside model)
- ``create_variable()`` -- build a dimensional tensor **inside** a ``pm.Model`` context
- ``set_data(ds, model)`` -- update shared variables for prediction

``register_data`` calls ``add_coords`` before ``register_data`` on each
``ModelTerm``, so dynamic coordinates are handled automatically.

Rules
^^^^^
- ``get_coords`` should be **pure** (no model context).
- Guard ``pmd.Data`` registration with ``if name not in model`` to share
  data between terms.
- ``create_variable`` must return a dims-compatible (xtensor) tensor.
- Each RV name must be unique across the full expression tree.
- ``build_param`` accepts ``VariableFactory``, ``ModelTerm``, ``xr.DataArray``,
  or numeric literal.  ``Prior.create_variable`` is called with ``xdist=True``.

``Parameter`` and ``Intercept``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These are the same kind of term: a named free parameter whose shape
comes from the prior (scalar when no ``dims``; dimensional when the prior
carries ``dims``).  ``Intercept`` is a convenience alias with ``name``
defaulting to ``"intercept"``; use ``Parameter`` when the role is not an
intercept (e.g. a cohort scale or product effect).  When the prior
carries ``dims``, coordinates are automatically extracted from the dataset
during ``collect_coords``.

Example: custom gather term
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Copy this into your project as a starting point for cohort / index /
group gathers:

.. code-block:: python

    @dataclass
    class Indexed(ModelTerm):
        base: Any
        index_var: str

        def get_coords(self, ds):
            return get_coords(self.base, ds)

        def register_data(self, ds):
            register_data(self.base, ds=ds)
            model = pm.modelcontext(None)
            if self.index_var not in model:
                pmd.Data(
                    self.index_var, ds[self.index_var], dims=ds[self.index_var].dims
                )

        def create_variable(self):
            base_val = build_param(self.base)
            model = pm.modelcontext(None)
            idx = model[self.index_var]
            return base_val[idx]

        def set_data(self, ds, model=None):
            set_data(self.base, ds=ds, model=model)
            if self.index_var in ds:
                pm.set_data({self.index_var: ds[self.index_var].values}, model=model)

Usage (custom + built-ins composed in one expression):

.. code-block:: python

    import pytensor.xtensor as ptx
    from pymc_extras.prior import Prior
    from pymc_marketing.terms import (
        Dot,
        Parameter,
        Transform,
        build_param,
        collect_coords,
        get_coords,
        register_data,
        set_data,
    )

    mu = Indexed(
        Parameter("alpha_scale", prior=Prior("HalfNormal", dims="cohort")),
        "cohort_idx",
    ) * Transform(
        -Dot(var_name="covars", name="coef_alpha", prior=Prior("Normal", dims="covar")),
        ptx.math.exp,
    )
    coords = collect_coords(mu, ds=ds)
    with pm.Model(coords=coords) as model:
        register_data(mu, ds=ds)
        alpha = build_param(mu)

Examples
--------
Built-ins cover the common pieces.  The explicit four-step lifecycle
(``collect_coords`` / ``register_data`` / ``build_param`` / ``set_data``)
works the same for built-ins and custom terms --- no hidden helpers.

.. code-block:: python

    import pymc.dims as pmd
    import pytensor.xtensor as ptx
    from pymc_extras.prior import Prior
    from pymc_marketing.terms import (
        Dot,
        Intercept,
        Transform,
        build_param,
        collect_coords,
        register_data,
    )

    mu = Intercept(name="mu") + Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    sigma = Transform(Intercept(name="sigma"), func=ptx.math.exp)

    coords = collect_coords(mu, sigma, ds=ds)

    with pm.Model(coords=coords) as model:
        register_data(mu, ds=ds)
        register_data(sigma, ds=ds)
        pmd.Normal(
            "y_obs", mu=build_param(mu), sigma=build_param(sigma), observed=ds["y"]
        )

.. code-block:: python

    from pymc_marketing.terms import (
        Parameter,
        Dot,
        collect_coords,
        register_data,
        build_param,
    )

    mu = Parameter("baseline", prior=Prior("Normal", dims="product")) + Dot(
        var_name="x", prior=Prior("Normal", dims="feature")
    )
    coords = collect_coords(mu, ds=ds)
    with pm.Model(coords=coords) as model:
        register_data(mu, ds=ds)
        build_param(mu)

.. code-block:: python

    from pymc_marketing.terms import (
        Dot,
        Intercept,
        collect_coords,
        register_data,
        build_param,
    )

    alpha_branch = Dot(
        var_name="x", name="alpha_coef", prior=Prior("Normal", dims="feature")
    )
    beta_branch = Dot(
        var_name="x", name="beta_coef", prior=Prior("Normal", dims="feature")
    )
    mu = alpha_branch + Intercept(name="offset")
    # beta_branch can be used in a separate expression tree

Gotchas
-------
- ``create_variable()`` should return a dimensional tensor
  (via ``pmd.Normal`` with ``xdist=True``, or ``pytensor.xtensor``).
  Mixing non-xtensor output (e.g., regular ``pm.Normal``) with xtensor
  output in a ``Sum`` will fail when ``build_param`` composes them.
- Two ``Dot`` terms referencing the same ``var_name`` will share the
  ``pmd.Data`` variable automatically (duplicate registration is skipped).
- Two ``Dot`` terms with the same ``var_name`` will also create identically
  named beta variables (``{var_name}_beta``), causing a PyMC "already
  exists" error. Pass a distinct ``name`` to each ``Dot`` to give the
  coefficients different variable names while still sharing the same
  ``pmd.Data`` (e.g. separate alpha and beta coefficient branches on the
  same covariate matrix).
- ``build_param`` is not idempotent --- call it once to create PyMC
  variables, then sample. Calling it a second time tries to create
  variables with the same name, causing a PyMC error.
- Changing the number of observations in ``set_data`` requires rebuilding
  the model. This is a ``pmd.Data`` constraint (dimensional shared variables
  have fixed dimension sizes), not a framework limitation.
- Model-internal coordinates that are not in the dataset can be added
  by overriding ``add_coords``:

  .. code-block:: python

      def add_coords(self, ds):
          model = pm.modelcontext(None)
          unique = list(dict.fromkeys(ds[self.data_source].values))
          model.add_coords({self.data_source: unique})
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import xarray as xr
from pymc_extras.prior import Prior, VariableFactory
from pytensor.xtensor.type import as_xtensor

__all__ = [
    "Dot",
    "Intercept",
    "ModelTerm",
    "Parameter",
    "Product",
    "Sum",
    "Transform",
    "build_param",
    "collect_coords",
    "collect_terms",
    "get_coords",
    "register_data",
    "set_data",
]


@dataclass
class ModelTerm:
    """Base class for composable model terms.

    Subclass ``ModelTerm`` to define reusable PyMC subgraph recipes.
    Custom terms compose with the built-ins (``Parameter``, ``Intercept``,
    ``Dot``, ``Transform``) via ``+``, ``*``, and ``-`` and use the same
    helpers (``collect_coords``, ``register_data``, ``build_param``,
    ``set_data``) --- no framework changes required.  See the module
    docstring **Extending** section for the full contract and an example.
    """

    def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
        """Return coordinates needed for this term."""
        return {}

    def add_coords(self, ds: xr.Dataset) -> None:
        """Add coordinates to the model during ``register_data``.

        Called automatically by the ``register_data`` helper. The active
        model is available via ``pm.modelcontext(None)``. Override this
        to add coordinates that are not discoverable from the dataset
        alone (e.g., unique group labels from a data column).

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the data variables.
        """

    def register_data(self, ds: xr.Dataset) -> None:
        """Register shared data variables as ``pmd.Data``."""

    def create_variable(self) -> pt.TensorVariable | None:
        """Build the term's tensor contribution.

        Must be called within a ``pm.Model`` context. Override this to
        produce a tensor. Not all terms need to produce a variable —
        data reference terms may only implement ``register_data`` and
        ``set_data``.

        Returns
        -------
        pt.TensorVariable or None
        """
        return None

    def set_data(self, ds: xr.Dataset, model: pm.Model | None = None) -> None:
        """Update shared data variables for out-of-sample prediction.

        Parameters
        ----------
        ds : xr.Dataset
            New dataset containing updated values.
        model : pm.Model, optional
            The PyMC model to update.
        """

    def __add__(self, other: Any) -> Sum:
        """Compose additively with another term."""
        return Sum([self, other])

    def __radd__(self, other: Any) -> Sum | ModelTerm:
        """Compose additively from the left; ``0 + term`` returns the term."""
        if isinstance(other, int) and other == 0:
            return self
        return Sum([other, self])

    def __mul__(self, other: Any) -> Product:
        """Compose multiplicatively with another term."""
        return Product(self, other)

    def __rmul__(self, other: Any) -> Product:
        """Compose multiplicatively from the left."""
        return Product(other, self)

    def __sub__(self, other: Any) -> Sum:
        """Compose additively with the negation of another term."""
        return Sum([self, -other])

    def __rsub__(self, other: Any) -> Sum:
        """Compose additively from the left with the negation of this term."""
        return Sum([other, -self])

    def __neg__(self) -> Product:
        """Return the negation of this term."""
        return Product(-1, self)


@dataclass
class Sum:
    """Container for additive composition via ``+``.

    Created when terms are composed with the ``+`` operator.

    Parameters
    ----------
    terms : list
        The terms to sum together.
    """

    terms: list[Any]

    def __add__(self, other: Any) -> Sum:
        """Compose additively with another term or sum."""
        if isinstance(other, Sum):
            return Sum(self.terms + other.terms)
        return Sum([*self.terms, other])

    def __radd__(self, other: Any) -> Sum:
        """Compose additively from the left; ``0 + sum`` returns the sum."""
        if isinstance(other, int) and other == 0:
            return self
        return Sum([other, *self.terms])

    def __mul__(self, other: Any) -> Product:
        """Compose multiplicatively with another term."""
        return Product(self, other)

    def __rmul__(self, other: Any) -> Product:
        """Compose multiplicatively from the left."""
        return Product(other, self)

    def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
        """Collect coordinates from all terms."""
        coords: dict[str, Any] = {}
        for term in self.terms:
            coords.update(get_coords(term, ds))
        return coords

    def register_data(self, ds: xr.Dataset) -> None:
        """Register shared data for all terms.

        Delegates to the module-level ``register_data`` helper which
        calls ``add_coords`` on each term before registering data.
        """
        register_data(self, ds=ds)

    def set_data(self, ds: xr.Dataset, model: pm.Model | None = None) -> None:
        """Update shared data for all terms."""
        for term in self.terms:
            set_data(term, ds=ds, model=model)


@dataclass
class Product:
    """Container for multiplicative composition via ``*``.

    Created when terms are composed with the ``*`` operator.

    Parameters
    ----------
    left : Any
        Left operand.
    right : Any
        Right operand.
    """

    left: Any
    right: Any

    def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
        """Collect coordinates from both operands."""
        return {
            **get_coords(self.left, ds),
            **get_coords(self.right, ds),
        }

    def __mul__(self, other: Any) -> Product:
        """Compose multiplicatively with another term."""
        return Product(self, other)

    def __rmul__(self, other: Any) -> Product:
        """Compose multiplicatively from the left."""
        return Product(other, self)

    def register_data(self, ds: xr.Dataset) -> None:
        """Register shared data for both operands."""
        register_data(self, ds=ds)

    def set_data(self, ds: xr.Dataset, model: pm.Model | None = None) -> None:
        """Update shared data for both operands."""
        set_data(self.left, ds=ds, model=model)
        set_data(self.right, ds=ds, model=model)


@dataclass
class Parameter(ModelTerm):
    """Named free parameter (scalar or dimensional via ``prior.dims``).

    ``Parameter`` is the general-purpose leaf for free random variables.
    ``Intercept`` is a convenience alias that defaults ``name`` to
    ``"intercept"``; both share the same lifecycle.

    Each parameter in the same ``pm.Model`` must have a unique ``name``.
    Two parameters with the same name (even in different expression trees,
    e.g., a ``mu`` expression and a ``sigma`` expression) will clash when
    ``build_param`` creates the variables.

    Parameters
    ----------
    name : str
        Name for the PyMC variable (required).
    prior : VariableFactory
        Prior distribution for the parameter. Any ``VariableFactory``
        (``Prior``, ``Censored``, ``Scaled``, custom) is accepted.
        When the prior carries ``dims`` (e.g. ``Prior(..., dims="cohort")``),
        ``get_coords`` will extract those coordinates from the dataset.

    Examples
    --------
    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.terms import (
            Parameter,
            Dot,
            Intercept,
            collect_coords,
            register_data,
            build_param,
        )

        # scalar
        mu = Parameter("mu") + Dot(var_name="x", prior=Prior("Normal", dims="feature"))

        # dimensional (cohort)
        alpha = Parameter("alpha_scale", prior=Prior("HalfNormal", dims="cohort"))

        # same concept, GLM sugar
        intercept = Intercept("intercept")  # defaults to name="intercept"
    """

    name: str
    prior: VariableFactory = field(default_factory=lambda: Prior("Normal"))

    def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
        """Collect coordinates from ``prior.dims`` present in the dataset."""
        dims = getattr(self.prior, "dims", None)
        if dims is None:
            return {}
        if isinstance(dims, str):
            dims = (dims,)
        dims = cast("tuple[str, ...]", dims)
        return {d: ds.coords[d].values.tolist() for d in dims if d in ds.coords}

    def create_variable(self) -> pt.TensorVariable:
        """Build a free parameter variable."""
        return self.prior.create_variable(self.name, xdist=True)


@dataclass
class Intercept(Parameter):
    """GLM-style intercept term; same as :class:`Parameter` with a default name.

    ``Intercept`` is provided for familiar GLM notation.  It is identical
    to ``Parameter`` in behavior and lifecycle.  Use ``Parameter`` when the
    role is not an intercept (e.g. a cohort scale or product effect).

    Parameters
    ----------
    name : str
        Name for the PyMC variable.  Defaults to ``"intercept"``.
    prior : VariableFactory
        Prior distribution.  Any ``VariableFactory`` is accepted.
    """

    name: str = "intercept"
    prior: VariableFactory = field(default_factory=lambda: Prior("Normal"))


@dataclass(kw_only=True)
class Dot(ModelTerm):
    """Linear predictor term: ``data @ beta``.

    Registers the data variable as ``pmd.Data`` for out-of-sample prediction.

    Parameters
    ----------
    var_name : str
        Name of the data variable in the dataset.
    prior : VariableFactory
        Prior for the beta coefficients. Any ``VariableFactory`` is
        accepted. Must have dims matching the last dimension of the
        data variable.
    name : str, optional
        Name for the coefficient variable. Defaults to ``{var_name}_beta``.
        Provide a distinct ``name`` to reference the same ``var_name`` from
        multiple terms (e.g. separate alpha and beta coefficient branches)
        without colliding on the coefficient variable name.

    Examples
    --------
    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.terms import (
            Dot,
            Intercept,
            collect_coords,
            register_data,
            build_param,
        )

        # basic
        mu = Intercept("mu") + Dot(var_name="x", prior=Prior("Normal", dims="feature"))

        # two branches on the same data, distinct coefficient names
        alpha_branch = Dot(
            var_name="x", name="alpha_coef", prior=Prior("Normal", dims="feature")
        )
        beta_branch = Dot(
            var_name="x", name="beta_coef", prior=Prior("Normal", dims="feature")
        )
    """

    var_name: str
    prior: VariableFactory
    name: str | None = None

    def __post_init__(self):
        """Set the default coefficient name to ``{var_name}_beta``."""
        if self.name is None:
            self.name = f"{self.var_name}_beta"

    def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
        """Extract coordinates from the data variable."""
        return {
            cast("str", k): v.values.tolist()
            for k, v in ds[self.var_name].coords.items()
        }

    def register_data(self, ds: xr.Dataset) -> None:
        """Register the data variable as ``pmd.Data``."""
        model = pm.modelcontext(None)
        if self.var_name not in model:
            pmd.Data(self.var_name, ds[self.var_name])

    def set_data(self, ds: xr.Dataset, model: pm.Model | None = None) -> None:
        """Update ``pmd.Data`` for prediction."""
        if self.var_name in ds:
            da = ds[self.var_name]
            coords = {dim: ds[dim].values for dim in da.dims if dim in ds.coords}
            pm.set_data({self.var_name: da.values}, model=model, coords=coords)

    def create_variable(self) -> pt.TensorVariable:
        """Build ``data @ beta`` tensor."""
        model = pm.modelcontext(None)
        data = model[self.var_name]
        beta = self.prior.create_variable(self.name, xdist=True)
        return data @ beta


@dataclass
class Transform(ModelTerm):
    """Apply a pytensor function to a term's output.

    Delegates coordinates, data registration, and data updating to the
    inner expression. Covers link functions and arbitrary transformations
    without a separate link system.

    Parameters
    ----------
    inner : Any
        Inner expression accepted by ``build_param``
        (``ModelTerm``, ``Sum``, ``float``, ``Prior``, etc.).
    func : Callable
        A pytensor function applied to ``build_param(inner)``.
        E.g., ``pytensor.xtensor.math.exp``, ``pt.math.sigmoid``.

    Examples
    --------
    .. code-block:: python

        import pytensor.xtensor as ptx

        sigma = Transform(Intercept(name="sigma"), func=ptx.math.exp)

    ``inner`` may be any expression, including custom ``ModelTerm``
    subclasses and composed trees:

    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.terms import Dot, Transform

        effect = Transform(
            Dot(var_name="x", prior=Prior("Normal", dims="feature")),
            func=ptx.math.softplus,
        )
    """

    inner: Any
    func: Callable

    def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
        """Collect coordinates from the inner expression."""
        return get_coords(self.inner, ds)

    def register_data(self, ds: xr.Dataset) -> None:
        """Register shared data for the inner expression."""
        register_data(self.inner, ds=ds)

    def create_variable(self) -> pt.TensorVariable:
        """Build ``func(build_param(inner))``."""
        return self.func(build_param(self.inner))

    def set_data(self, ds: xr.Dataset, model: pm.Model | None = None) -> None:
        """Update shared data for the inner expression."""
        set_data(self.inner, ds=ds, model=model)


def get_coords(param: Any, ds: xr.Dataset) -> dict[str, Any]:
    """Extract coordinates from a term or term composition.

    Parameters
    ----------
    param : ModelTerm, Sum, Product, int, or float
        The term(s) to extract coordinates from.
    ds : xr.Dataset
        Dataset containing the data variables.

    Returns
    -------
    dict[str, Any]
        Coordinates dict for ``pm.Model`` (or ``pmd`` distributions).
    """
    if isinstance(param, (int, float)):
        return {}
    if isinstance(param, ModelTerm):
        return param.get_coords(ds)
    if isinstance(param, Sum):
        return param.get_coords(ds)
    if isinstance(param, Product):
        return param.get_coords(ds)
    return {}


def register_data(param: Any, *, ds: xr.Dataset) -> None:
    """Register shared data variables as ``pmd.Data`` in the active model.

    Calls ``add_coords`` before ``register_data`` on each ``ModelTerm``.

    Parameters
    ----------
    param : ModelTerm, Sum, Product, int, or float
        The term(s) containing data variables to register.
    ds : xr.Dataset
        Dataset containing the data values.
    """
    if isinstance(param, (int, float)):
        return
    if isinstance(param, ModelTerm):
        param.add_coords(ds)
        param.register_data(ds)
    if isinstance(param, Sum):
        for term in param.terms:
            register_data(term, ds=ds)
    if isinstance(param, Product):
        register_data(param.left, ds=ds)
        register_data(param.right, ds=ds)


def build_param(param: Any, name: str = "param") -> pt.TensorVariable | int | float:
    """Build a tensor parameter from a term, composition, variable factory, or constant.

    Must be called within a ``pm.Model`` context.

    Parameters
    ----------
    param : ModelTerm | Sum | Product | VariableFactory | int | float | xr.DataArray
        The term(s) or variable factory to build into a tensor.
    name : str
        Variable name used when building standalone ``VariableFactory``
        objects. Terms handle their own naming internally.

    Returns
    -------
    pt.TensorVariable or float
        The parameter tensor (``XTensorVariable`` when dimensional).
    """
    if isinstance(param, (int, float)):
        return param
    if isinstance(param, xr.DataArray):
        return as_xtensor(param.values, dims=cast("Sequence[str]", param.dims))
    if isinstance(param, ModelTerm):
        return cast("pt.TensorVariable", param.create_variable())
    if isinstance(param, Sum):
        result: pt.TensorVariable | int | float = 0
        for term in param.terms:
            result = result + build_param(term)
        return result
    if isinstance(param, Product):
        return build_param(param.left) * build_param(param.right)
    if isinstance(param, VariableFactory):
        return param.create_variable(name, xdist=True)
    raise TypeError(f"Cannot build param from {type(param)}")


def set_data(param: Any, *, ds: xr.Dataset, model: pm.Model) -> None:
    """Update shared data variables for out-of-sample prediction.

    Parameters
    ----------
    param : ModelTerm, Sum, Product, int, or float
        The term(s) to update.
    ds : xr.Dataset
        New dataset for prediction.
    model : pm.Model
        The PyMC model to update.
    """
    if isinstance(param, (int, float)):
        return
    if isinstance(param, ModelTerm):
        param.set_data(ds, model=model)
    if isinstance(param, Sum):
        param.set_data(ds, model=model)
    if isinstance(param, Product):
        param.set_data(ds, model=model)


def collect_terms(params: list[Any]) -> list[ModelTerm]:
    """Collect all ``ModelTerm`` instances from a list of parameter specifications.

    Parameters
    ----------
    params : list
        List of parameter specs (terms, term lists, compositions, constants).

    Returns
    -------
    list[ModelTerm]
        Flattened list of all individual term instances.
    """
    result: list[ModelTerm] = []
    for p in params:
        if isinstance(p, ModelTerm):
            result.append(p)
        elif isinstance(p, Sum):
            result.extend(collect_terms(p.terms))
        elif isinstance(p, Product):
            result.extend(collect_terms([p.left, p.right]))
    return result


def collect_coords(*params: Any, ds: xr.Dataset) -> dict[str, Any]:
    """Collect coordinates from multiple parameter expressions.

    Convenience wrapper for collecting coordinates from several independent
    term trees (e.g., a ``mu`` expression and a ``sigma`` expression).

    Parameters
    ----------
    *params : Any
        One or more ``ModelTerm``, ``Sum``, ``Product``, or compositions.
    ds : xr.Dataset
        Dataset containing the data variables.

    Returns
    -------
    dict[str, Any]
        Combined coordinates dict for ``pm.Model``.

    Examples
    --------
    .. code-block:: python

        coords = collect_coords(mu, sigma, ds=ds)
        coords["obs"] = ds.coords["obs"].values.tolist()

        with pm.Model(coords=coords) as model:
            ...
    """
    coords: dict[str, Any] = {}
    for param in params:
        coords.update(get_coords(param, ds))
    return coords
