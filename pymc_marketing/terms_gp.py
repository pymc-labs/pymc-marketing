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

The GP graph itself is **delegated** to the existing
:mod:`pymc_marketing.mmm.hsgp` classes: at ``create_variable`` time the term
builds a transient :class:`~pymc_marketing.mmm.hsgp.HSGP` (or ``HSGPPeriodic``
/ ``SoftPlusHSGP``) spec from its resolved fields and delegates variable
creation to it, so the two implementations cannot drift. These terms own the
modeling lifecycle: time-reference resolution (datetime coordinates welcome),
shared data registration, frozen centering, and the ``terms`` composition and
serialization contract.

Defaults follow the assumptions of the existing HSGP classes:
``eta_mass=0.05``, ``eta_upper=1.0``, ``ls_lower=1.0``, ``ls_upper=None``,
``ls_mass=0.9``, ``cov_func="expquad"``, ``centered=False``,
``drop_first=True``, and ``demeaned_basis=False``.

Rules
^^^^^
- ``var_name`` names the time reference in the dataset and is read with
  ``ds[var_name]``, which works for both **coordinates** (the common case,
  e.g. ``coords={"date": ...}``) and data variables. Datetimes are converted
  to **observation periods since the anchored first training date**, using
  ``time_resolution`` days per period, and registered as ``pmd.Data`` under
  ``{name}_index``.
- ``time_resolution`` defaults to ``None``, which infers the number of days
  per period from the observed date spacing -- the same convention as
  :func:`pymc_marketing.mmm.tvp.infer_time_index` and
  :class:`pymc_marketing.mmm.MMM`, so the numeric index counts periods rather
  than days and the deferred ``m`` / ``L`` / lengthscale heuristics see the
  same axis as the rest of the library. Pass it explicitly only to override
  the inferred unit. Numeric (non-datetime) references are passed through
  unchanged with a resolution of 1.
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
  :class:`~pymc_marketing.terms.Parameter` does, and recorded on first
  registration. A curve is fit per coordinate **in order**, so ``set_data``
  refuses a window whose coordinates for those dims were reordered, dropped,
  or added rather than reporting each curve under the wrong label. The time
  index itself stays one-dimensional and is shared across those dims.

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
import xarray as xr
from pymc_extras.prior import VariableFactory

from pymc_marketing.hsgp_kwargs import CovFunc
from pymc_marketing.mmm.hsgp import (
    HSGP,
    HSGPBase,
    HSGPPeriodic,
    SoftPlusHSGP,
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
)

__all__ = ["HSGPPeriodicTerm", "HSGPTerm", "SoftPlusHSGPTerm"]


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


def _serialize_coord(values: list[Any]) -> list[Any]:
    """Serialize an extra-dim coordinate to JSON-safe Python scalars.

    Datetime values become ISO strings rather than ``.item()`` integers: the
    naive ``.item()`` turns a ``datetime64[ns]`` coordinate into a count of
    nanoseconds since the epoch, which ``_check_extra_coords`` would then
    reject as a mismatch against the identical coordinate after a reload.
    """
    out: list[Any] = []
    for value in values:
        if isinstance(value, np.datetime64):
            out.append(str(value.astype("datetime64[us]")))
        elif hasattr(value, "item"):
            out.append(value.item())
        else:
            out.append(value)
    return out


def _deserialize_coord(values: list[Any], *, datetime64: bool = False) -> list[Any]:
    """Deserialize a JSON-safe coordinate back to a list.

    Coordinates tagged ``datetime64`` at serialize time are restored as
    ``datetime64`` from their ISO strings. Everything else is passed
    through as-is: string coordinates are never re-typed by guesswork,
    since ``np.datetime64`` parses strings like ``"12"`` or ``"NaT"``.
    """
    if datetime64:
        return [np.datetime64(value) for value in values]
    return list(values)


def _coord_is_datetime(values: list[Any]) -> bool:
    """Whether a serialized extra-dim coordinate holds datetime values."""
    return bool(values) and np.issubdtype(np.asarray(values).dtype, np.datetime64)


def _serialize_date(value: Any) -> str | None:
    """Serialize a date anchor as an ISO string (``None`` passes through)."""
    if value is None:
        return None
    return str(np.datetime64(value, "ns").astype("datetime64[us]"))


def _deserialize_date(value: str | None) -> Any:
    """Deserialize an ISO string back to a ``np.datetime64`` anchor."""
    if value is None:
        return None
    return np.datetime64(value)


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
    ``{name}_index``; use :attr:`index_var` for that name.
    """

    var_name: str = "date"
    name: str = "hsgp"
    X_mid: float | None = None
    dims: str | tuple[str, ...] | None = None
    demeaned_basis: bool = False
    time_resolution: int | None = None
    time_dim: str | None = field(default=None, init=False, repr=False)
    first_date: Any = field(default=None, init=False, repr=False)
    last_date: Any = field(default=None, init=False, repr=False)
    first_index: float | None = field(default=None, init=False, repr=False)
    last_index: float | None = field(default=None, init=False, repr=False)
    extra_coords: dict[str, list[Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def index_var(self) -> str:
        """Name of this term's registered numeric index data variable.

        Per term, not shared: two terms referencing the same time reference
        resolve their own ``time_resolution`` and anchor, so a shared index
        would force one term's basis onto the other's units.
        """
        return f"{self.name}_index"

    def __post_init__(self) -> None:
        """Normalize ``dims`` to a tuple for stable serialization round-trips."""
        if isinstance(self.dims, str):
            self.dims = (self.dims,)
        elif self.dims is not None:
            self.dims = tuple(self.dims)

    def _infer_time_resolution(self, da: xr.DataArray) -> int:
        """Infer the time resolution (days per period) from a datetime ref.

        Mirrors the convention in :class:`pymc_marketing.mmm.MMM`, which sets
        ``(dates[1] - dates[0]).days`` so the numeric time index is expressed
        in observation periods. Numeric references are passed through unchanged
        and keep a resolution of 1.
        """
        values = np.asarray(da.values)
        if not np.issubdtype(values.dtype, np.datetime64):
            return 1
        if len(values) < 2:
            return 1
        delta = (values[1] - values[0]) / np.timedelta64(1, "D")
        return max(round(float(delta)), 1)

    def _resolve_time_resolution(self, da: xr.DataArray) -> int:
        """Resolve ``time_resolution`` once, inferring it from the data if unset.

        Explicit values always win; inference happens on first use so the term
        stays constructible without data.
        """
        if self.time_resolution is None:
            self.time_resolution = self._infer_time_resolution(da)
        return self.time_resolution

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

        Datetimes become (whole) periods since the anchored first training
        date: the day offset divided by ``time_resolution`` (inferred from the
        observed spacing when not given, so the index is in observation
        periods like the rest of the library). Numeric values are passed
        through as floats.
        """
        values = np.asarray(da.values)
        self._resolve_time_resolution(da)
        if np.issubdtype(values.dtype, np.datetime64):
            anchor = self.first_date if self.first_date is not None else values[0]
            values = (values - anchor) / np.timedelta64(1, "D")
            values = values / self.time_resolution
        return np.asarray(values, dtype=float)

    def _time_index(self, da: xr.DataArray) -> xr.DataArray:
        """Numeric time index as a DataArray, preserving dims and coords."""
        return xr.DataArray(self._time_values(da), dims=da.dims, coords=da.coords)

    def _check_window_start(self, values: np.ndarray) -> None:
        """Refuse a window that begins before the recorded training anchor.

        The anchor (and the frozen basis) are set by the training data. A
        window whose earliest date precedes the anchor would place the time
        index before zero, outside the learned domain, so it is rejected
        rather than silently extrapolated.
        """
        if self.first_date is None or not np.issubdtype(values.dtype, np.datetime64):
            return
        earliest = values.min()
        if earliest < self.first_date:
            raise ValueError(
                f"The time reference {self.var_name!r} starts at {earliest}, before "
                f"the training anchor {self.first_date}. The GP basis and centering "
                "are frozen on the training data, so a window that begins earlier "
                "cannot be placed on the learned time axis. Pass data covering the "
                "training range (or starting at/after it) instead."
            )

    def _check_training_window(self, index: xr.DataArray) -> None:
        """Refuse to re-register a fitted term on any window but the training one.

        A rebuilt basis must be centered and sized on the training data itself:
        ``X_mid``, ``m``, and ``L`` were resolved from it and are frozen. A
        later registration on a different window (a future-only window, or
        train + future) would build the basis against data the term was never
        fit with and silently produce a different GP. The reload contract is:
        rebuild on the training window, then use ``set_data`` for prediction
        windows.
        """
        if self.first_index is None:
            return
        lo = float(index.min())
        hi = float(index.max())
        if lo == self.first_index and hi == self.last_index:
            return
        raise ValueError(
            f"The GP term {self.name!r} was fitted on the training window "
            f"[{self.first_index}, {self.last_index}] of {self.index_var!r}, but "
            f"the data given to `register_data` covers [{lo}, {hi}]. Rebuild on "
            "the training window and use `set_data` for prediction windows."
        )

    def register_data(self, ds: xr.Dataset) -> None:
        """Register the numeric time index as ``pmd.Data`` and freeze ``X_mid``."""
        model = pm.modelcontext(None)
        da = ds[self.var_name]
        values = np.asarray(da.values)
        is_datetime = np.issubdtype(values.dtype, np.datetime64)
        if is_datetime:
            self._check_window_start(values)
            if self.first_date is None:
                self.first_date = values.min()
            if self.last_date is None:
                self.last_date = values.max()
        index = self._time_index(da)
        self._check_training_window(index)
        if self.extra_coords:
            # rebuilt term: the same label order the curves were fit with
            self._check_extra_coords(ds)
        if self.index_var not in model:
            pmd.Data(self.index_var, index)
        if self.X_mid is None:
            self.X_mid = float(self._time_values(da).mean())
        if self.first_index is None:
            self.first_index = float(index.min())
            self.last_index = float(index.max())
        if self.time_dim is None:
            self.time_dim = cast("str", da.dims[0])
        if not self.extra_coords:
            self.extra_coords = {
                dim: list(ds.coords[dim].values)
                for dim in self.extra_dims
                if dim in ds.coords
            }

    def _check_extra_coords(self, ds: xr.Dataset) -> None:
        """Refuse a prediction window whose extra dims do not match training.

        Each extra dim gets one GP curve per coordinate, positioned by order.
        A window that reorders, drops, or adds a coordinate would silently
        report the wrong curve under the wrong label, so it is rejected.
        """
        for dim, expected in self.extra_coords.items():
            if dim not in ds.coords:
                raise ValueError(
                    f"The GP term {self.name!r} has one curve per {dim!r} but the "
                    f"dataset passed to `set_data` has no {dim!r} coordinate."
                )
            actual = list(ds.coords[dim].values)
            if actual != expected:
                raise ValueError(
                    f"The {dim!r} coordinate of the GP term {self.name!r} does not "
                    f"match the training data. Expected {expected!r}, got {actual!r}. "
                    "One GP curve is fit per coordinate, in order, so a different "
                    "or reordered set would report each curve under the wrong label."
                )

    def set_data(self, ds: xr.Dataset, model: pm.Model | None = None) -> None:
        """Update the shared time index for out-of-sample prediction."""
        if self.var_name not in ds:
            return
        if self.time_dim is None:
            if self.X_mid is not None:
                raise ValueError(
                    f"The GP term {self.name!r} was restored from a recipe that does "
                    "not record its time dimension, so the registered time index is "
                    "unknown. Re-serialize the term, or call `register_data` with the "
                    "training data before `set_data`."
                )
            raise ValueError(
                f"Nothing registered for {self.var_name!r}. "
                "Call `register_data` before `set_data`."
            )
        self._check_extra_coords(ds)
        da = ds[self.var_name]
        # The anchor applies here too, not only in register_data: a prediction
        # window starting before it would place the time index before zero,
        # outside the frozen basis, and extrapolate silently.
        self._check_window_start(np.asarray(da.values))
        coords = {dim: ds[dim].values for dim in da.dims if dim in ds.coords}
        values = self._time_values(da)
        pm.set_data({self.index_var: values}, model=model, coords=coords)

    def _check_registered(self) -> pm.Model:
        """Return the active model, raising if the time reference is unregistered."""
        model = pm.modelcontext(None)
        if self.X_mid is None or self.time_dim is None:
            raise ValueError(
                "The data must be registered before creating a variable. "
                f"Call `register_data` with a dataset containing {self.var_name!r}."
            )
        return model

    def _spec(self) -> HSGPBase:
        """Build the wrapped HSGP spec from the resolved fields."""
        raise NotImplementedError

    def create_variable(self) -> pt.TensorVariable:
        """Build the GP curve through the wrapped HSGP class.

        The wrapped spec owns the basis coordinate, hyperparameter variables,
        and the final deterministic; this term feeds it the frozen centering
        value and the shared time index.

        Raises
        ------
        ValueError
            If another term in the model already created a variable with this
            term's ``name``. Each GP term's ``name`` is used as the prefix for
            its variables (``{name}_eta``, ``{name}_ls``, ``{name}_hsgp_coefs``),
            so two terms cannot share one.
        """
        model = self._check_registered()
        spec = self._spec()
        spec.X_mid = self.X_mid
        try:
            spec.register_data(model[self.index_var])
            return spec.create_variable(self.name, xdist=True)
        except ValueError as err:
            if f"{self.name}_" in str(err):
                raise ValueError(
                    f"A variable for the GP term named {self.name!r} already exists "
                    f"in this model: {err} Each GP term's `name` is the prefix for "
                    "its variables, so give each GP term a distinct `name=`."
                ) from err
            raise


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
        ``{name}_index``. Defaults to ``"date"``.
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
        collected from the dataset. The coordinates of these dims are
        recorded on the first registration, and :meth:`set_data` refuses a
        window whose coordinates differ (reordered, dropped, or added),
        since each curve is fit per coordinate in order and a mismatch would
        report it under the wrong label.
    time_resolution : int, optional
        Number of days per observation period, dividing the day offsets of a
        datetime time reference. Default ``None``, which infers it from the
        observed date spacing so the numeric index counts periods (the same
        convention as :class:`pymc_marketing.mmm.MMM`). Pass an explicit value
        only to override the inferred unit.
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

    def __post_init__(self) -> None:
        """Normalize dims and coerce a string ``cov_func`` to the enum."""
        super().__post_init__()
        if isinstance(self.cov_func, str):
            # HSGP accepts a covariance name as a string; coerce so the
            # recipe serializes the same way regardless of how it was built.
            self.cov_func = CovFunc(self.cov_func)

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
        """Resolve deferred values.

        The basis coordinate is added by the wrapped HSGP class at build
        time, so this only triggers the deferred hyperparameter resolution.
        """
        self._resolve(ds)

    def _spec_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the wrapped HSGP spec."""
        dims = (cast("str", self.time_dim), *self.extra_dims)
        return {
            "ls": self.ls,
            "eta": self.eta,
            "m": cast("int", self.m),
            "L": cast("float", self.L),
            "dims": dims,
            "centered": self.centered,
            "drop_first": self.drop_first,
            "cov_func": self.cov_func,
            "demeaned_basis": self.demeaned_basis,
        }

    def _spec(self) -> HSGPBase:
        """Build the wrapped :class:`~pymc_marketing.mmm.hsgp.HSGP` spec."""
        _check_scalar("eta", self.eta)
        _check_scalar("ls", self.ls)
        return HSGP(**self._spec_kwargs())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the term recipe.

        The frozen training state (``X_mid`` and the date anchors) is carried
        through so a reloaded recipe indexes new data on the same time axis and
        refuses windows that fall before the training anchor.
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
            "X_mid": self.X_mid,
            "first_date": _serialize_date(self.first_date),
            "last_date": _serialize_date(self.last_date),
            "first_index": self.first_index,
            "last_index": self.last_index,
            "time_dim": self.time_dim,
            "extra_coords": {
                k: _serialize_coord(v) for k, v in self.extra_coords.items()
            },
            "extra_coord_dtypes": {
                k: "datetime64"
                for k, v in self.extra_coords.items()
                if _coord_is_datetime(v)
            },
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
        term = cls(
            var_name=data["var_name"],
            name=data["name"],
            eta=_deserialize_optional(data.get("eta")),
            ls=_deserialize_optional(data.get("ls")),
            m=data.get("m"),
            L=data.get("L"),
            dims=data.get("dims"),
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
        term.X_mid = data.get("X_mid")
        term.first_date = _deserialize_date(data.get("first_date"))
        term.last_date = _deserialize_date(data.get("last_date"))
        term.first_index = data.get("first_index")
        term.last_index = data.get("last_index")
        term.time_dim = data.get("time_dim")
        dtypes = data.get("extra_coord_dtypes") or {}
        term.extra_coords = {
            k: _deserialize_coord(v, datetime64=dtypes.get(k) == "datetime64")
            for k, v in (data.get("extra_coords") or {}).items()
        }
        return term


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

    def _spec(self) -> SoftPlusHSGP:
        """Build the wrapped :class:`~pymc_marketing.mmm.hsgp.SoftPlusHSGP`."""
        return SoftPlusHSGP(**self._spec_kwargs())

    @staticmethod
    def deterministics_to_replace(name: str) -> list[str]:
        """Deterministics to replace with ``pm.Flat`` for out-of-sample.

        Required so out-of-sample predictions keep the training time-mean
        of one. Without this, the training and test curves are not
        continuous.

        """
        return SoftPlusHSGP.deterministics_to_replace(name)


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
        ``{name}_index``. Defaults to ``"date"``.
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

    name: str = "hsgp_periodic"
    scale: VariableFactory | float
    ls: VariableFactory | float
    period: float
    m: int

    def add_coords(self, ds: xr.Dataset) -> None:
        """Freeze the centering value.

        The basis coordinate is added by the wrapped HSGPPeriodic class at
        build time.
        """
        if self.X_mid is None:
            self.X_mid = float(self._time_values(ds[self.var_name]).mean())

    def _spec(self) -> HSGPBase:
        """Build the wrapped :class:`~pymc_marketing.mmm.hsgp.HSGPPeriodic`."""
        _check_scalar("scale", self.scale)
        _check_scalar("ls", self.ls)
        dims = (cast("str", self.time_dim), *self.extra_dims)
        return HSGPPeriodic(
            scale=self.scale,
            ls=self.ls,
            period=self.period,
            m=self.m,
            dims=dims,
            demeaned_basis=self.demeaned_basis,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the term recipe, carrying the frozen training state."""
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
            "X_mid": self.X_mid,
            "first_date": _serialize_date(self.first_date),
            "last_date": _serialize_date(self.last_date),
            "first_index": self.first_index,
            "last_index": self.last_index,
            "time_dim": self.time_dim,
            "extra_coords": {
                k: _serialize_coord(v) for k, v in self.extra_coords.items()
            },
            "extra_coord_dtypes": {
                k: "datetime64"
                for k, v in self.extra_coords.items()
                if _coord_is_datetime(v)
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HSGPPeriodicTerm:
        """Reconstruct a term from its serialized form."""
        term = cls(
            var_name=data["var_name"],
            name=data["name"],
            scale=_deserialize_optional(data.get("scale")),
            ls=_deserialize_optional(data.get("ls")),
            period=data["period"],
            m=data["m"],
            dims=data.get("dims"),
            demeaned_basis=data["demeaned_basis"],
            time_resolution=data["time_resolution"],
        )
        term.X_mid = data.get("X_mid")
        term.first_date = _deserialize_date(data.get("first_date"))
        term.last_date = _deserialize_date(data.get("last_date"))
        term.first_index = data.get("first_index")
        term.last_index = data.get("last_index")
        term.time_dim = data.get("time_dim")
        dtypes = data.get("extra_coord_dtypes") or {}
        term.extra_coords = {
            k: _deserialize_coord(v, datetime64=dtypes.get(k) == "datetime64")
            for k, v in (data.get("extra_coords") or {}).items()
        }
        return term
