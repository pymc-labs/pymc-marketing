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
"""BART term for the ``pymc_marketing.terms`` framework.

``pymc_marketing.bart`` exposes :class:`Bart`, a :class:`~pymc_marketing.terms.ModelTerm`
that wraps :func:`pymc_bart.pymc_bart.BART` so Bayesian Additive Regression Trees
can be composed like any other term in the ``pymc_marketing.terms`` vocabulary,
e.g. interchangeably with :class:`~pymc_marketing.terms.Dot`.

``pymc_bart`` is an optional dependency (``pip install 'pymc-marketing[pie]'``).
Importing this module never fails when ``pymc_bart`` is missing; constructing
the tensor raises a ``ImportError`` with an install hint instead. This keeps
models that swap the BART term for a linear :class:`~pymc_marketing.terms.Dot`
term runnable without ``pymc_bart`` installed.

Because ``pymc_bart`` does not yet operate on ``pymc.dims`` variables, the term
bridges the two worlds following the standard recipe: the dimensional shared
data variable is lowered to a plain tensor with ``.values`` before being handed
to ``pymc_bart``, and the BART output is lifted back into the dimensional graph
with ``pytensor.xtensor.type.as_xtensor``. See
`pymc-labs/pymc-marketing#2017 <https://github.com/pymc-labs/pymc-marketing/issues/2017>`_.

Examples
--------
Swap a BART mean function for a linear dot product:

.. code-block:: python

    import pymc.dims as pmd
    import xarray as xr
    from pymc_extras.prior import Prior

    from pymc_marketing.bart import Bart
    from pymc_marketing.terms import (
        build_param,
        collect_coords,
        register_data,
    )

    ds = xr.Dataset(
        {
            "X": (("obs", "feature"), X),
            "y_obs": (("obs",), y),
        },
    )
    mu = Bart(var_name="X", m=200)
    # ... or the plain linear alternative:
    # mu = Dot(var_name="X", prior=Prior("Normal", dims="feature"))

    coords = collect_coords(mu, ds=ds)
    with pm.Model(coords=coords) as model:
        register_data(mu, ds=ds)
        model = pm.modelcontext(None)
        sigma = Prior("HalfNormal", sigma=1.0).create_variable("sigma", xdist=True)
        pmd.Normal(
            "y",
            mu=build_param(mu),
            sigma=sigma,
            observed=model["y_obs"],
            dims="obs",
        )

Terms compose, so BART can also be an ingredient of a larger expression:

.. code-block:: python

    from pymc_marketing.terms import Intercept

    mu = Intercept(name="intercept") + Bart(var_name="X", m=200)

The term serializes through ``pymc_marketing.serialization`` (``to_dict`` /
``from_dict``), so it can be stored in ``model_config`` and survive
``save()`` / ``load()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import xarray as xr
from pytensor.xtensor.type import as_xtensor

from pymc_marketing.serialization import serialization
from pymc_marketing.terms import ModelTerm

try:
    import pymc_bart as pmb
except ImportError:  # pragma: no cover
    pmb = None  # type: ignore[assignment]

__all__ = ["Bart"]

_DEFAULT_M: int = 200
_DEFAULT_ALPHA: float = 0.95
_DEFAULT_BETA: float = 2.0
_DEFAULT_RESPONSE: str = "constant"


def _serialize_split_rules(rules: list[Any] | None) -> list[str] | None:
    """Serialize split rule instances to class names."""
    if rules is None:
        return None
    return [rule.__class__.__name__ for rule in rules]


def _load_split_rules(names: list[str] | None) -> list[Any] | None:
    """Rebuild split rule instances from serialized class names."""
    if names is None:
        return None
    if pmb is None:
        raise ImportError(
            "pymc-bart is required to rebuild split rules for a serialized "
            "Bart term. Install it with: pip install 'pymc-marketing[pie]'"
        )
    from pymc_bart import split_rules

    rules = []
    for name in names:
        if not hasattr(split_rules, name):
            raise ValueError(
                f"Unknown serialized split rule {name!r}. Split rules are "
                "resolved by class name against pymc_bart.split_rules."
            )
        rules.append(getattr(split_rules, name)())
    return rules


@serialization.register
@dataclass(kw_only=True)
class Bart(ModelTerm):
    """Bayesian Additive Regression Trees term.

    Wraps :func:`pymc_bart.pymc_bart.BART` as a composable
    :class:`~pymc_marketing.terms.ModelTerm`. Interchangeable with
    :class:`~pymc_marketing.terms.Dot` --- swapping the two changes the model
    graph but nothing else in the expression or lifecycle.

    The data variable ``var_name`` is registered as ``pymc.dims`` shared data
    and the target ``y_name`` must also be registered (BART uses ``Y`` to seed
    the leaf-value prior with residual quantiles). Whoever registers the
    observed data wins: registration is guarded by ``if name not in model``.

    Parameters
    ----------
    var_name : str
        Name of the feature variable in the dataset, with shape
        ``(..., feature)``. Registered as ``pmd.Data`` for out-of-sample
        prediction.
    y_name : str, optional
        Name of the target variable in the model. Used only to seed the
        leaf-value prior at build time. Defaults to ``"y_obs"``.
    m : int, optional
        Number of trees. Defaults to 200.
    alpha : float, optional
        Tree prior parameter controlling depth. Defaults to 0.95.
    beta : float, optional
        Tree prior parameter controlling depth. Defaults to 2.0.
    response : str, optional
        Leaf response form, one of ``"constant"``, ``"linear"``, or ``"mix"``.
        Defaults to ``"constant"``.
    split_rules : list, optional
        Split rules, one per feature column. Defaults to ``None``.
    name : str, optional
        Name for the PyMC variable. Defaults to ``"bart"``.

    Examples
    --------
    .. code-block:: python

        from pymc_extras.prior import Prior

        from pymc_marketing.bart import Bart
        from pymc_marketing.terms import Dot

        bart_mean = Bart(var_name="X", m=200, name="bart")
        linear_mean = Dot(
            var_name="X", name="mu_coef", prior=Prior("Normal", dims="feature")
        )
    """

    var_name: str
    y_name: str = "y_obs"
    m: int = _DEFAULT_M
    alpha: float = _DEFAULT_ALPHA
    beta: float = _DEFAULT_BETA
    response: str = _DEFAULT_RESPONSE
    split_rules: list[Any] | None = None
    name: str = "bart"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.response not in ("constant", "linear", "mix"):
            raise ValueError(
                "Bart response must be 'constant', 'linear', or 'mix', "
                f"got {self.response!r}."
            )
        if self.m < 1:
            raise ValueError(f"Bart m must be >= 1, got {self.m}.")

    def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
        """Extract coordinates from the feature data variable."""
        return {
            cast("str", dim): da.coords[dim].values.tolist()
            for dim, da in ds[self.var_name].coords.items()
        }

    def register_data(self, ds: xr.Dataset) -> None:
        """Register the feature and target data variables as ``pmd.Data``."""
        model = pm.modelcontext(None)
        if self.var_name not in model:
            pmd.Data(self.var_name, ds[self.var_name])
        if self.y_name in ds and self.y_name not in model:
            pmd.Data(self.y_name, ds[self.y_name])

    def create_variable(self) -> pt.TensorVariable:
        """Build the BART ensemble tensor.

        Returns
        -------
        pt.TensorVariable
            Dimensional tensor over the observation dimensions.
        """
        if pmb is None:
            raise ImportError(
                "pymc-bart is required to build a Bart term. "
                "Install it with: pip install 'pymc-marketing[pie]'"
            )
        model = pm.modelcontext(None)
        X = model[self.var_name]
        if self.y_name not in model:
            raise ValueError(
                f"Bart term requires target data '{self.y_name}' in the model. "
                "Register it as pmd.Data before building the term."
            )
        obs_dims = cast("tuple[str, ...]", X.dims)[:-1]
        mu_plain = pmb.BART(
            self.name,
            X=X.values,
            Y=model[self.y_name].values,
            m=self.m,
            alpha=self.alpha,
            beta=self.beta,
            response=self.response,
            split_rules=self.split_rules,
            dims=obs_dims,
        )
        return as_xtensor(mu_plain, dims=obs_dims)

    def set_data(self, ds: xr.Dataset, model: pm.Model | None = None) -> None:
        """Update shared data variables for out-of-sample prediction."""
        if self.var_name in ds:
            da = ds[self.var_name]
            coords = {dim: ds[dim].values for dim in da.dims if dim in ds.coords}
            pm.set_data({self.var_name: da.values}, model=model, coords=coords)
        if self.y_name in ds:
            da = ds[self.y_name]
            coords = {dim: ds[dim].values for dim in da.dims if dim in ds.coords}
            pm.set_data({self.y_name: da.values}, model=model, coords=coords)

    @property
    def sample_vars(self) -> list[str]:
        """Unobserved variables that must be re-sampled in predictive mode.

        The BART ensemble is conditioned on registered data, so when the data
        are swapped for out-of-sample prediction the likelihood contribution
        has to be recomputed rather than frozen from the trace.
        """
        return [self.name]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the term configuration.

        Returns
        -------
        dict[str, Any]
            JSON-safe configuration. ``split_rules`` serialize as class names.
        """
        return {
            "var_name": self.var_name,
            "y_name": self.y_name,
            "m": self.m,
            "alpha": self.alpha,
            "beta": self.beta,
            "response": self.response,
            "split_rules": _serialize_split_rules(self.split_rules),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Bart:
        """Reconstruct a ``Bart`` term from its serialized configuration."""
        return cls(
            var_name=data["var_name"],
            y_name=data.get("y_name", "y_obs"),
            m=data.get("m", _DEFAULT_M),
            alpha=data.get("alpha", _DEFAULT_ALPHA),
            beta=data.get("beta", _DEFAULT_BETA),
            response=data.get("response", _DEFAULT_RESPONSE),
            split_rules=_load_split_rules(data.get("split_rules")),
            name=data.get("name", "bart"),
        )
