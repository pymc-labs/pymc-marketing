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

"""Gaussian process model terms.

``pymc_marketing.terms_gp`` exposes the Hilbert Space Gaussian Process (HSGP)
components from :mod:`pymc_marketing.mmm.hsgp` as composable
:class:`~pymc_marketing.terms.ModelTerm` pieces: :class:`HSGPTerm`,
:class:`HSGPPeriodicTerm`, and :class:`SoftPlusHSGPTerm`. They subclass the
``ModelTerm`` lifecycle (``get_coords`` / ``add_coords`` / ``register_data`` /
``create_variable`` / ``set_data``) so they compose with the other built-ins
via ``+`` / ``*`` / ``-`` --- no framework changes required.

The terms are fully deferred. Constructing one touches no data and computes
nothing: ``m``, ``L``, ``X_mid``, and the default ``eta`` / ``ls`` priors are
resolved from the dataset inside the model context (cached on the instance
once resolved, so repeated builds reuse them). Explicit values always win
over the deferred resolution.

Defaults follow the assumptions of the existing HSGP classes:
``eta_mass=0.05``, ``eta_upper=1.0``, ``ls_lower=1.0``, ``ls_upper=None``,
``ls_mass=0.9``, ``cov_func="expquad"``, ``centered=False``,
``drop_first=True``, and ``demeaned_basis=False``.

Rules
^^^^^
- ``var_name`` names the time reference in the dataset and is read with
  ``ds[var_name]``, which works for both **coordinates** (the common case,
  e.g. ``coords={"date": ...}``) and data variables. Datetimes are
  converted to **days since the anchored first training date**, divided by
  ``time_resolution`` (the same convention as
  :func:`pymc_marketing.mmm.tvp.infer_time_index`), and registered as
  ``pmd.Data`` under ``{var_name}_index``.
- ``register_data`` must run before ``create_variable``. The module-level
  :func:`~pymc_marketing.terms.register_data` helper handles this.
- ``X_mid`` is frozen at the first registration so out-of-sample predictions
  via ``set_data`` stay centered on the training data. It is excluded from
  serialization and re-derived from the data on rebuild.
- Each random variable is prefixed with ``name`` (``{name}_eta``,
  ``{name}_ls``, ``{name}_hsgp_coefs``) and the basis coordinate is
  ``{name}_m``. Each term in an expression tree needs a unique ``name``;
  two no-argument terms collide on every variable.
- ``eta`` / ``ls`` priors must be scalar. Higher-dimensional coefficients
  (e.g. one GP curve per group) are configured with ``dims``, whose
  coordinates are collected from the dataset like
  :class:`~pymc_marketing.terms.Parameter` does.

Time-varying media
^^^^^^^^^^^^^^^^^^
The canonical time-varying media multiplier is :class:`SoftPlusHSGPTerm`,
matching the time-varying parameters of the stable MMM classes. The GP
output is one-dimensional over the time dimension, so it broadcasts across
the channel dimension of a media term through ``*``:

.. code-block:: python

    tvp_media = SoftPlusHSGPTerm() * media

Examples
--------
HSGP term with deferred defaults, composed into an outcome equation:

.. code-block:: python

    from pymc_extras.prior import Prior
    from pymc_marketing.terms import (
        Intercept,
        Dot,
        collect_coords,
        register_data,
        build_param,
    )
    from pymc_marketing.terms_gp import HSGPTerm

    trend = HSGPTerm(var_name="time", name="trend")
    mu = (
        Intercept("intercept")
        + trend
        + Dot(var_name="x", prior=Prior("Normal", dims="feature"))
    )

    coords = collect_coords(mu, ds=ds)
    with pm.Model(coords=coords) as model:
        register_data(mu, ds=ds)
        mu_value = build_param(mu)

No-argument construction for a one-dimensional GP over the ``"date"``
coordinate, then multiplied by a media term (dims ``(date, channel)``):

.. code-block:: python

    from pymc_marketing.terms_gp import SoftPlusHSGPTerm

    tvp_media = SoftPlusHSGPTerm() * media

Higher-dimensional coefficients, one GP curve per channel:

.. code-block:: python

    from pymc_marketing.terms_gp import HSGPTerm

    trend = HSGPTerm(var_name="time", name="trend", dims="channel")

Periodic seasonality term:

.. code-block:: python

    from pymc_extras.prior import Prior
    from pymc_marketing.terms_gp import HSGPPeriodicTerm

    seasonality = HSGPPeriodicTerm(
        var_name="time",
        name="seasonality",
        scale=Prior("HalfNormal", sigma=1),
        ls=Prior("InverseGamma", alpha=2, beta=1),
        period=52,
        m=20,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import xarray as xr
from pymc_extras.prior import Prior, VariableFactory
from pytensor.tensor import as_tensor
from pytensor.xtensor.type import as_xtensor

from pymc_marketing.hsgp_kwargs import CovFunc
from pymc_marketing.mmm.hsgp import (
    create_complexity_penalizing_prior,
    create_constrained_inverse_gamma_prior,
    create_eta_prior,
    create_m_and_L_recommendations,
)
from pymc_marketing.serialization import serialization
from pymc_marketing.terms import (
    ModelTerm,
    _deserialize_child,
    _serialize_child,
    build_param,
)

__all__ = ["HSGPPeriodicTerm", "HSGPTerm", "SoftPlusHSGPTerm"]

_GP_COV_FUNCS = {
    "expquad": pm.gp.cov.ExpQuad,
    "matern52": pm.gp.cov.Matern52,
    "matern32": pm.gp.cov.Matern32,
}


def _serialize_optional(value: Any) -> Any:
    """Serialize an optional term field, passing ``None`` through."""
    return None if value is None else _serialize_child(value)


def _deserialize_optional(value: Any) -> Any:
    """Deserialize an optional term field, passing ``None`` through."""
    return None if value is None else _deserialize_child(value)


def _normalize_dims(dims: str | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize a dims argument to a tuple."""
    if dims is None:
        return ()
    if isinstance(dims, str):
        return (dims,)
    return tuple(dims)


def _dims_to_list(dims: str | tuple[str, ...] | None) -> list[str] | None:
    """Serialize dims as a JSON-safe list."""
    normalized = _normalize_dims(dims)
    return list(normalized) or None


def _check_scalar(label: str, value: Any) -> None:
    """Require a scalar prior for a GP hyperparameter."""
    if getattr(value, "dims", None):
        raise ValueError(f"The {label} prior must be a scalar random variable.")


@dataclass(kw_only=True)
class GPDataTerm(ModelTerm):
    """Shared data lifecycle for GP terms.

    Resolves the time reference ``var_name`` to a numeric index, registers
    it as ``pmd.Data``, freezes the centering value (``X_mid``) at the
    first registration, and collects coordinates for the time dimension
    and any extra coefficient dims.

    ``var_name`` names the time reference in the dataset and is read with
    ``ds[var_name]``, which works for both coordinates and data variables.
    The numeric index is always registered as ``pmd.Data`` under
    ``{var_name}_index``; use :attr:`index_var` for that name.
    """

    var_name: str = "date"
    name: str = "hsgp"
    X_mid: float | None = None
    dims: str | tuple[str, ...] | None = None
    demeaned_basis: bool = False
    time_resolution: int = 1
    time_dim: str | None = field(default=None, init=False, repr=False)
    first_date: Any = field(default=None, init=False, repr=False)

    @property
    def index_var(self) -> str:
        """Name of the registered numeric index data variable."""
        return f"{self.var_name}_index"

    @property
    def extra_dims(self) -> tuple[str, ...]:
        """Dims beyond the time dimension."""
        return _normalize_dims(self.dims)

    def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
        """Collect coordinates from the time reference and extra dims."""
        da = ds[self.var_name]
        coords = {cast("str", k): v.values.tolist() for k, v in da.coords.items()}
        for dim in self.extra_dims:
            if dim in ds.coords and dim not in coords:
                coords[dim] = ds.coords[dim].values.tolist()
        return coords

    def _time_values(self, da: xr.DataArray) -> np.ndarray:
        """Convert a time reference to numeric index values.

        Datetimes become days since the anchored first training date,
        divided by ``time_resolution``. Numeric values are passed through
        as floats.
        """
        values = np.asarray(da.values)
        if np.issubdtype(values.dtype, np.datetime64):
            anchor = self.first_date if self.first_date is not None else values[0]
            values = (values - anchor) / np.timedelta64(1, "D")
            values = values / self.time_resolution
        return np.asarray(values, dtype=float)

    def _time_index(self, da: xr.DataArray) -> xr.DataArray:
        """Numeric time index as a DataArray, preserving dims and coords."""
        return xr.DataArray(self._time_values(da), dims=da.dims, coords=da.coords)

    def register_data(self, ds: xr.Dataset) -> None:
        """Register the numeric time index as ``pmd.Data`` and freeze ``X_mid``."""
        model = pm.modelcontext(None)
        da = ds[self.var_name]
        values = np.asarray(da.values)
        if np.issubdtype(values.dtype, np.datetime64) and self.first_date is None:
            self.first_date = values[0]
        if self.index_var not in model:
            pmd.Data(self.index_var, self._time_index(da))
        if self.X_mid is None:
            self.X_mid = float(self._time_values(da).mean())
        if self.time_dim is None:
            self.time_dim = cast("str", da.dims[0])

    def set_data(self, ds: xr.Dataset, model: pm.Model | None = None) -> None:
        """Update the shared time index for out-of-sample prediction."""
        if self.var_name not in ds:
            return
        if self.time_dim is None:
            raise ValueError(
                f"Nothing registered for {self.var_name!r}. "
                "Call `register_data` before `set_data`."
            )
        da = ds[self.var_name]
        coords = {dim: ds[dim].values for dim in da.dims if dim in ds.coords}
        pm.set_data({self.index_var: self._time_values(da)}, model=model, coords=coords)

    def _prepare_X(self) -> tuple[pt.TensorVariable, str]:
        """Return the registered time-index tensor and time dim."""
        model = pm.modelcontext(None)
        if self.X_mid is None or self.time_dim is None:
            raise ValueError(
                "The data must be registered before creating a variable. "
                f"Call `register_data` with a dataset containing {self.var_name!r}."
            )
        X = model[self.index_var]
        X_tensor = as_tensor(X, allow_xtensor_conversion=True)
        return X_tensor, self.time_dim


@serialization.register
@dataclass(kw_only=True)
class HSGPTerm(GPDataTerm):
    """Hilbert Space Gaussian Process term.

    A one-dimensional GP over the time reference ``var_name`` with optional
    higher-dimensional coefficients via ``dims``. Composes with the other
    terms via ``+`` / ``*`` / ``-``.

    All of ``m``, ``L``, ``X_mid``, ``eta``, and ``ls`` are deferred: when
    ``None``, they are resolved from the dataset inside the model context
    with the same heuristics and prior recommendations as
    :meth:`pymc_marketing.mmm.hsgp.HSGP.parameterize_from_data`, then cached
    on the instance. Explicit values always win.

    Parameters
    ----------
    var_name : str, optional
        Name of the time reference in the dataset, read with
        ``ds[var_name]`` (works for both coordinates and data variables).
        Datetimes are converted to days since the first date, divided by
        ``time_resolution``, and registered as ``pmd.Data`` under
        ``{var_name}_index``. Defaults to ``"date"``.
    name : str, optional
        Prefix for the variables and the output deterministic. Defaults to
        ``"hsgp"``.
    eta : VariableFactory or float, optional
        Prior for the GP variance. Defaults to the Exponential prior from
        :func:`pymc_marketing.mmm.hsgp.create_eta_prior`.
    ls : VariableFactory or float, optional
        Prior for the lengthscale. Defaults to the complexity-penalizing
        Weibull prior, or a constrained InverseGamma when ``ls_upper`` is
        set.
    m : int, optional
        Number of basis functions. Defaults to the Ruitort-Mayol et al.
        recommendation from the data.
    L : float, optional
        Extent of the basis functions. Defaults to the recommendation from
        the data.
    dims : str or tuple of str, optional
        Extra dims for the coefficients beyond the time dim, e.g.
        ``"channel"`` for one GP curve per channel. Coordinates are
        collected from the dataset.
    time_resolution : int
        Divisor applied to the day offsets of a datetime time reference.
        Default ``1`` (plain day offsets). For weekly data pass ``7`` (the
        stable MMM time-varying default uses ``5``) so the deferred ``m``
        and ``L`` heuristics see weekly-scale lengthscales.
    centered : bool
        Whether the coefficient prior is centered. Default ``False``.
    drop_first : bool
        Whether to drop the first (constant) basis function. Default ``True``.
    demeaned_basis : bool
        Whether each basis has its mean subtracted. Default ``False``.
    eta_mass, eta_upper, ls_lower, ls_upper, ls_mass : float
        Knobs for the deferred prior resolution. Defaults mirror
        ``HSGP.parameterize_from_data`` (``0.05``, ``1.0``, ``1.0``,
        ``None``, ``0.9``).
    cov_func : CovFunc
        Covariance function. Default ``CovFunc.ExpQuad``.

    Examples
    --------
    Deferred defaults composed into an outcome equation:

    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.terms import (
            Intercept,
            Dot,
            collect_coords,
            register_data,
            build_param,
        )
        from pymc_marketing.terms_gp import HSGPTerm

        trend = HSGPTerm(var_name="time", name="trend")
        mu = (
            Intercept("intercept")
            + trend
            + Dot(var_name="x", prior=Prior("Normal", dims="feature"))
        )

        coords = collect_coords(mu, ds=ds)
        with pm.Model(coords=coords) as model:
            register_data(mu, ds=ds)
            mu_value = build_param(mu)

    Explicit configuration:

    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.terms_gp import HSGPTerm

        trend = HSGPTerm(
            var_name="time",
            name="trend",
            eta=Prior("Exponential", lam=1),
            ls=Prior("InverseGamma", alpha=2, beta=1),
            m=20,
            L=150,
        )
    """

    eta: VariableFactory | float | None = None
    ls: VariableFactory | float | None = None
    m: int | None = None
    L: float | None = None
    centered: bool = False
    drop_first: bool = True
    eta_mass: float = 0.05
    eta_upper: float = 1.0
    ls_lower: float = 1.0
    ls_upper: float | None = None
    ls_mass: float = 0.9
    cov_func: CovFunc = CovFunc.ExpQuad

    def _resolve(self, ds: xr.Dataset) -> None:
        """Resolve deferred hyperparameters from the dataset (fill-if-None)."""
        X = self._time_values(ds[self.var_name])
        if self.X_mid is None:
            self.X_mid = float(X.mean())
        if self.m is None or self.L is None:
            m, L = create_m_and_L_recommendations(
                X,
                self.X_mid,
                ls_lower=self.ls_lower,
                ls_upper=self.ls_upper,
                cov_func=self.cov_func,
            )
            self.m = self.m if self.m is not None else m
            self.L = self.L if self.L is not None else L
        if self.eta is None:
            self.eta = create_eta_prior(mass=self.eta_mass, upper=self.eta_upper)
        if self.ls is None:
            if self.ls_upper is None:
                self.ls = create_complexity_penalizing_prior(
                    lower=self.ls_lower,
                    alpha=self.ls_mass,
                )
            else:
                self.ls = create_constrained_inverse_gamma_prior(
                    lower=self.ls_lower,
                    upper=self.ls_upper,
                    mass=self.ls_mass,
                )

    def add_coords(self, ds: xr.Dataset) -> None:
        """Resolve deferred values, then add the basis coordinate."""
        self._resolve(ds)
        m = cast("int", self.m)
        model = pm.modelcontext(None)
        model.add_coords({f"{self.name}_m": np.arange(m - 1 if self.drop_first else m)})

    def _build_gp(self, prefix: str) -> pt.TensorVariable:
        """Build the linearized GP expression under a variable-name prefix."""
        X_tensor, time_dim = self._prepare_X()
        _check_scalar("eta", self.eta)
        _check_scalar("ls", self.ls)

        eta = as_tensor(
            build_param(self.eta, name=f"{prefix}_eta"),
            allow_xtensor_conversion=True,
        )
        ls = as_tensor(
            build_param(self.ls, name=f"{prefix}_ls"),
            allow_xtensor_conversion=True,
        )

        cov_func = eta**2 * _GP_COV_FUNCS[self.cov_func.value](input_dim=1, ls=ls)
        gp = pm.gp.HSGP(m=[self.m], L=[self.L], cov_func=cov_func)
        phi, sqrt_psd = gp.prior_linearized(X_tensor[:, None] - self.X_mid)

        if self.drop_first:
            phi = phi[:, 1:]
            sqrt_psd = sqrt_psd[1:]

        if self.demeaned_basis:
            phi = phi - phi.mean(axis=0).eval()

        coord_name = f"{prefix}_m"
        phi_x = as_xtensor(phi, dims=(time_dim, coord_name))
        sqrt_psd_x = as_xtensor(sqrt_psd, dims=(coord_name,))

        hsgp_coefs = Prior(
            "Normal",
            mu=0,
            sigma=sqrt_psd_x,
            dims=(*self.extra_dims, coord_name),
            centered=self.centered,
        ).create_variable(f"{prefix}_hsgp_coefs", xdist=True)

        return phi_x.dot(hsgp_coefs)

    def create_variable(self) -> pt.TensorVariable:
        """Build the GP curve as a named deterministic."""
        f = self._build_gp(self.name)
        return pmd.Deterministic(self.name, f, dims=(self.time_dim, *self.extra_dims))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the term recipe.

        The frozen ``X_mid`` is excluded; it is re-derived from the data on
        rebuild.
        """
        return {
            "var_name": self.var_name,
            "name": self.name,
            "eta": _serialize_optional(self.eta),
            "ls": _serialize_optional(self.ls),
            "m": self.m,
            "L": self.L,
            "dims": _dims_to_list(self.dims),
            "centered": self.centered,
            "drop_first": self.drop_first,
            "demeaned_basis": self.demeaned_basis,
            "time_resolution": self.time_resolution,
            "eta_mass": self.eta_mass,
            "eta_upper": self.eta_upper,
            "ls_lower": self.ls_lower,
            "ls_upper": self.ls_upper,
            "ls_mass": self.ls_mass,
            "cov_func": self.cov_func.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HSGPTerm:
        """Reconstruct a term from its serialized form."""
        dims = data.get("dims")
        if isinstance(dims, list):
            dims = tuple(dims)
        return cls(
            var_name=data["var_name"],
            name=data["name"],
            eta=_deserialize_optional(data.get("eta")),
            ls=_deserialize_optional(data.get("ls")),
            m=data.get("m"),
            L=data.get("L"),
            dims=dims,
            centered=data["centered"],
            drop_first=data["drop_first"],
            demeaned_basis=data["demeaned_basis"],
            time_resolution=data["time_resolution"],
            eta_mass=data["eta_mass"],
            eta_upper=data["eta_upper"],
            ls_lower=data["ls_lower"],
            ls_upper=data.get("ls_upper"),
            ls_mass=data["ls_mass"],
            cov_func=CovFunc(data["cov_func"]),
        )


@serialization.register
@dataclass(kw_only=True)
class SoftPlusHSGPTerm(HSGPTerm):
    """HSGP term with softplus transformation and mean-one centering.

    Maps the latent GP to strictly positive values and normalizes
    multiplicatively by the time-mean, so the resulting multiplier has mean
    one over the time dimension while remaining strictly positive. This is
    the canonical time-varying multiplier for media effects, matching the
    time-varying parameters of the stable MMM classes.

    The output broadcasts across a media term's channel dimension through
    ``*``:

    .. code-block:: python

        tvp_media = SoftPlusHSGPTerm() * media

    For out-of-sample prediction with
    :func:`pymc_marketing.model_graph.deterministics_to_flat`, the
    ``{name}_f_mean`` deterministic is the one to replace:

    .. code-block:: python

        SoftPlusHSGPTerm.deterministics_to_replace("tvp")
        # ['tvp_f_mean']
    """

    name: str = "tvp"

    def add_coords(self, ds: xr.Dataset) -> None:
        """Resolve deferred values, then add the raw basis coordinate."""
        self._resolve(ds)
        m = cast("int", self.m)
        model = pm.modelcontext(None)
        model.add_coords(
            {f"{self.name}_raw_m": np.arange(m - 1 if self.drop_first else m)}
        )

    @staticmethod
    def deterministics_to_replace(name: str) -> list[str]:
        """Deterministics to replace with ``pm.Flat`` for out-of-sample.

        Required so out-of-sample predictions keep the training time-mean
        of one. Without this, the training and test curves are not
        continuous.

        """
        return [f"{name}_f_mean"]

    def create_variable(self) -> pt.TensorVariable:
        """Build the positive, mean-one GP multiplier."""
        f = self._build_gp(f"{self.name}_raw")
        f = pmd.math.softplus(f)

        f_mean = pmd.Deterministic(f"{self.name}_f_mean", f.mean(dim=self.time_dim))

        centered_f = f / f_mean
        return pmd.Deterministic(self.name, centered_f)


@serialization.register
@dataclass(kw_only=True)
class HSGPPeriodicTerm(GPDataTerm):
    """Periodic HSGP term.

    A one-dimensional periodic GP over the time reference ``var_name``, for
    seasonality with a fixed ``period``. Mirrors
    :class:`pymc_marketing.mmm.hsgp.HSGPPeriodic`: ``m``, ``scale``, ``ls``,
    and ``period`` are required (there is no data-driven recommendation for
    the periodic basis).

    Parameters
    ----------
    var_name : str, optional
        Name of the time reference in the dataset, read with
        ``ds[var_name]`` (works for both coordinates and data variables).
        Datetimes are converted to days since the first date, divided by
        ``time_resolution``, and registered as ``pmd.Data`` under
        ``{var_name}_index``. Defaults to ``"date"``.
    name : str, optional
        Prefix for the variables and the output deterministic. Defaults to
        ``"hsgp_periodic"``.
    scale : VariableFactory or float
        Prior for the scale of the periodic GP.
    ls : VariableFactory or float
        Prior for the lengthscale of the periodic GP.
    period : float
        The period of the function, in units of the time index.
    m : int
        Number of basis functions.
    dims : str or tuple of str, optional
        Extra dims for the coefficients beyond the time dim.
    demeaned_basis : bool
        Whether each basis has its mean subtracted. Default ``False``.

    Examples
    --------
    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.terms_gp import HSGPPeriodicTerm

        seasonality = HSGPPeriodicTerm(
            var_name="time",
            name="seasonality",
            scale=Prior("HalfNormal", sigma=1),
            ls=Prior("InverseGamma", alpha=2, beta=1),
            period=52,
            m=20,
        )
    """

    var_name: str = "date"
    name: str = "hsgp_periodic"
    scale: VariableFactory | float
    ls: VariableFactory | float
    period: float
    m: int

    def add_coords(self, ds: xr.Dataset) -> None:
        """Freeze the centering value, then add the basis coordinate."""
        if self.X_mid is None:
            self.X_mid = float(self._time_values(ds[self.var_name]).mean())
        model = pm.modelcontext(None)
        model.add_coords({f"{self.name}_m": np.arange((self.m * 2) - 1)})

    def _build_gp(self, prefix: str) -> pt.TensorVariable:
        """Build the linearized periodic GP expression under a prefix."""
        X_tensor, time_dim = self._prepare_X()
        _check_scalar("scale", self.scale)
        _check_scalar("ls", self.ls)

        scale = as_tensor(
            build_param(self.scale, name=f"{prefix}_scale"),
            allow_xtensor_conversion=True,
        )
        ls = as_tensor(
            build_param(self.ls, name=f"{prefix}_ls"),
            allow_xtensor_conversion=True,
        )

        cov_func = pm.gp.cov.Periodic(1, period=self.period, ls=ls)
        gp = pm.gp.HSGPPeriodic(m=self.m, scale=scale, cov_func=cov_func)
        (phi_cos, phi_sin), psd = gp.prior_linearized(X_tensor[:, None] - self.X_mid)

        if self.demeaned_basis:
            phi_cos = phi_cos - phi_cos.mean(axis=0).eval()
            phi_sin = phi_sin - phi_sin.mean(axis=0).eval()

        coord_name = f"{prefix}_m"
        phi_cos_x = as_xtensor(phi_cos, dims=(time_dim, coord_name))
        phi_sin_x = as_xtensor(phi_sin, dims=(time_dim, coord_name))
        psd_x = as_xtensor(psd, dims=(coord_name,))
        slice_idx = {coord_name: slice(1, None)}

        sigma = ptx.concat([psd_x, psd_x.isel(slice_idx)], dim=coord_name)
        hsgp_coefs = Prior(
            "Normal",
            mu=0,
            sigma=sigma,
            dims=(*self.extra_dims, coord_name),
            centered=False,
        ).create_variable(f"{prefix}_hsgp_coefs", xdist=True)

        phi = ptx.concat([phi_cos_x, phi_sin_x.isel(slice_idx)], dim=coord_name)
        return phi.dot(hsgp_coefs)

    def create_variable(self) -> pt.TensorVariable:
        """Build the periodic GP curve as a named deterministic."""
        f = self._build_gp(self.name)
        return pmd.Deterministic(self.name, f, dims=(self.time_dim, *self.extra_dims))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the term recipe."""
        return {
            "var_name": self.var_name,
            "name": self.name,
            "scale": _serialize_optional(self.scale),
            "ls": _serialize_optional(self.ls),
            "period": self.period,
            "m": self.m,
            "dims": _dims_to_list(self.dims),
            "demeaned_basis": self.demeaned_basis,
            "time_resolution": self.time_resolution,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HSGPPeriodicTerm:
        """Reconstruct a term from its serialized form."""
        dims = data.get("dims")
        if isinstance(dims, list):
            dims = tuple(dims)
        return cls(
            var_name=data["var_name"],
            name=data["name"],
            scale=_deserialize_optional(data.get("scale")),
            ls=_deserialize_optional(data.get("ls")),
            period=data["period"],
            m=data["m"],
            dims=dims,
            demeaned_basis=data["demeaned_basis"],
            time_resolution=data["time_resolution"],
        )
