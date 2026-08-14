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
"""Optimization variables for budget optimization.

The decision vector handed to ``scipy.optimize.minimize`` is a flat 1-D array.
Everything the optimizer knows about *what that vector means* — which model
variable each segment substitutes, how a segment maps to model-space tensors
(the forward map), and how a solution maps back to labelled ``DataArray``s
(the inverse map) — lives here, stated once per variable.

- :class:`OptimizationVariable` is the per-variable protocol: a named, contiguous
  segment of the flat vector with a forward map (``to_model``), an inverse
  (``unpack``) and its exact inverse (``pack``), plus defaults for the initial
  guess and bounds.
- :class:`MediaVariable` implements the media-budget path: mask scatter,
  channel scaling, temporal distribution, cost-per-unit conversion, and
  adstock carry-over padding.
- :class:`OptimizationVariables` owns the flat symbolic input and the variable layout,
  and produces the single substitution dict for ``pymc.do``.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor as ptx
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable
from xarray import DataArray

__all__ = [
    "MediaVariable",
    "OptimizationVariable",
    "OptimizationVariables",
]


class OptimizationVariable(ABC):
    """One named, contiguous segment of the flat decision vector.

    A variable describes a single model variable that the optimizer controls:
    how its slice of the flat vector becomes the model-space tensor
    substituted into the graph (:meth:`to_model`), how a solution slice
    becomes a labelled ``DataArray`` (:meth:`unpack`), and the exact inverse
    of that labelling (:meth:`pack`).

    Attributes
    ----------
    name : str
        Name of the model variable this variable substitutes.
    dims : tuple[str, ...]
        Dimension names of the variable's labelled representation.
    coords : dict[str, list]
        Coordinates for ``dims``.
    flat_dim : str
        Name of the flat decision vector's dimension. Owned by the containing
        :class:`OptimizationVariables`, which assigns it at construction.
    """

    name: str
    dims: tuple[str, ...]
    coords: dict[str, list]
    flat_dim: str

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of entries this variable occupies in the flat vector."""

    @abstractmethod
    def to_model(self, z: XTensorVariable) -> XTensorVariable:
        """Map this variable's slice of the flat vector to its model-space tensor.

        Parameters
        ----------
        z : XTensorVariable
            The variable's contiguous slice of the flat decision vector, with
            a single flat dimension of length :attr:`size`.

        Returns
        -------
        XTensorVariable
            The tensor to substitute for :attr:`name` in the model graph.
        """

    @abstractmethod
    def unpack(self, x: np.ndarray) -> DataArray:
        """Map a solution slice to a labelled ``DataArray``.

        Parameters
        ----------
        x : np.ndarray
            1-D array of length :attr:`size`.

        Returns
        -------
        DataArray
            Labelled with :attr:`dims` and :attr:`coords`.
        """

    @abstractmethod
    def pack(self, da: DataArray) -> np.ndarray:
        """Exact inverse of :meth:`unpack`.

        Parameters
        ----------
        da : DataArray
            Labelled values covering this variable's coordinates.

        Returns
        -------
        np.ndarray
            1-D array of length :attr:`size`, in flat-vector order.
        """

    @abstractmethod
    def default_x0(self, total_budget: float) -> np.ndarray:
        """Default initial guess for this variable's slice."""

    @abstractmethod
    def default_bounds(
        self, total_budget: float
    ) -> list[tuple[float | None, float | None]]:
        """Default ``(low, high)`` bounds per flat entry."""


class MediaVariable(OptimizationVariable):
    """The media-budget decision variable.

    Owns the forward map from flat monetary budgets to the model's channel
    data tensor: scatter through the optimization mask, divide by
    ``channel_scales``, spread over ``num_periods`` (uniformly or via a fixed
    temporal distribution), convert monetary units via ``cost_per_unit``, and
    zero-pad ``adstock_periods`` for carry-over — and the inverse map from a
    flat solution back to a labelled monetary ``DataArray``.

    Parameters
    ----------
    name : str
        Name of the model's channel data variable.
    mask : DataArray
        Boolean mask over the budget dims selecting cells to optimize, already
        reindexed and transposed to the model's coordinate and dimension
        order. That alignment is load-bearing: the mask is consumed
        positionally by the forward map (scatter into the model's tensor
        layout) *and* supplies the labels for the inverse map, so a mask in a
        different coordinate order would silently attribute each value to the
        wrong cell.
    num_periods : int
        Number of periods budget is allocated over.
    adstock_periods : int
        Number of zero-padded carry-over periods appended after the
        allocation window.
    channel_scales : float or np.ndarray
        Per-channel scale factors converting monetary budgets to model units.
    dtype : str
        dtype of the model's channel data variable. Must be a float dtype.
    date_dim : str
        Name of the date dimension.
    budget_distribution_over_period_tensor : XTensorVariable or None
        Pre-processed masked temporal distribution factors with dims
        ``(date_dim, flat_dim)``, or None for uniform spread.
    cost_per_unit_tensor : XTensorVariable or None
        Pre-processed cost-per-unit tensor with dims
        ``(date_dim, *mask.dims)``, or None for no conversion.
    flat_dim : str
        Name of the flat dimension of the decision vector.
    """

    def __init__(
        self,
        name: str,
        mask: DataArray,
        num_periods: int,
        adstock_periods: int,
        channel_scales: float | np.ndarray,
        dtype: str,
        date_dim: str = "date",
        budget_distribution_over_period_tensor: XTensorVariable | None = None,
        cost_per_unit_tensor: XTensorVariable | None = None,
        flat_dim: str = "budgets_flat",
    ) -> None:
        if np.dtype(dtype).kind != "f":
            raise ValueError(
                f"Optimization requires channel data of float type, got {dtype}"
            )
        self.name = name
        self.mask = mask.astype(bool)
        self.dims = tuple(mask.dims)
        self.coords = {dim: list(mask.coords[dim].values) for dim in self.dims}
        self.num_periods = num_periods
        self.adstock_periods = adstock_periods
        self.channel_scales = channel_scales
        self.dtype = dtype
        self.date_dim = date_dim
        self.budget_distribution_over_period_tensor = (
            budget_distribution_over_period_tensor
        )
        self.cost_per_unit_tensor = cost_per_unit_tensor
        self.flat_dim = flat_dim
        self._bool_mask = np.asarray(self.mask.values).astype(bool)

    @property
    def size(self) -> int:
        """Number of optimized budget cells."""
        return int(self._bool_mask.sum())

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the full (unmasked) budget tensor."""
        return self._bool_mask.shape

    def scattered(self, z: XTensorVariable) -> XTensorVariable:
        """Scatter the flat slice into the full budget-dims tensor.

        Non-optimized cells are zero. Values remain in monetary units.
        """
        budgets_zeros = pt.zeros(self.shape)
        budgets_zeros.name = "budgets_zeros"
        return as_xtensor(
            budgets_zeros[self._bool_mask].set(z.values),
            dims=self.dims,
        )

    def _apply_budget_distribution_over_period(
        self, budgets: XTensorVariable, z: XTensorVariable
    ) -> XTensorVariable:
        """Distribute each cell's budget across periods with fixed factors.

        The distribution factors are pre-masked to the optimized cells, so
        the product is computed on the flat slice and scattered back per
        period.
        """
        repeated_budgets_flat = (
            z * self.budget_distribution_over_period_tensor
        ).transpose(self.date_dim, self.flat_dim)

        repeated_budgets_x = ptx.zeros_like(budgets).expand_dims(
            **{self.date_dim: self.num_periods}, axis=0
        )
        repeated_budgets = repeated_budgets_x.values[:, self._bool_mask].set(
            repeated_budgets_flat.values
        )
        repeated_budgets = as_xtensor(repeated_budgets, dims=repeated_budgets_x.dims)

        # Factors are fractions of the per-period budget level; rescale so a
        # uniform distribution reproduces the default (repeat) behaviour.
        repeated_budgets *= self.num_periods

        return repeated_budgets

    def to_model(self, z: XTensorVariable) -> XTensorVariable:
        """Build the channel-data substitution tensor from the flat slice."""
        budgets = self.scattered(z)
        budgets /= as_xtensor(
            self.channel_scales,
            dims=() if np.ndim(self.channel_scales) == 0 else ("channel",),
        )

        # Repeat budgets over num_periods (still in monetary units)
        if self.budget_distribution_over_period_tensor is not None:
            repeated_budgets = self._apply_budget_distribution_over_period(budgets, z)
        else:
            repeated_budgets = budgets.expand_dims(**{self.date_dim: self.num_periods})

        # Convert from monetary units to original units using date-specific
        # rates. Applied AFTER time distribution so each period uses its own
        # cost rate.
        if self.cost_per_unit_tensor is not None:
            repeated_budgets = repeated_budgets / self.cost_per_unit_tensor

        repeated_budgets.name = "repeated_budgets"

        repeated_budgets_with_carry_over = ptx.concat(
            [
                repeated_budgets.astype(self.dtype),
                ptx.as_xtensor(
                    pt.zeros(self.adstock_periods, dtype=self.dtype),
                    dims=(self.date_dim,),
                ),
            ],
            dim=self.date_dim,
        )
        repeated_budgets_with_carry_over.name = "repeated_budgets_with_carry_over"
        return repeated_budgets_with_carry_over

    def unpack(self, x: np.ndarray) -> DataArray:
        """Scatter a flat solution back into a labelled monetary ``DataArray``."""
        full = np.zeros(self.shape, dtype=float)
        full[self._bool_mask] = x
        return DataArray(full, dims=self.dims, coords=self.coords)

    def pack(self, da: DataArray) -> np.ndarray:
        """Flatten a labelled allocation into this variable's flat-vector order."""
        missing = set(self.dims) - set(da.dims)
        if missing:
            raise ValueError(
                f"{self.name}: DataArray is missing required dims {sorted(missing)}"
            )
        extra = set(da.dims) - set(self.dims)
        if extra:
            raise ValueError(
                f"{self.name}: DataArray has unexpected dims {sorted(extra)}; "
                f"expected exactly {list(self.dims)}"
            )
        aligned = da.reindex(self.coords).transpose(*self.dims)
        values = aligned.values[self._bool_mask]
        if np.isnan(values).any():
            raise ValueError(
                f"{self.name}: values missing (NaN after reindex) for optimized "
                "cells; provide a value for every cell in the optimization mask"
            )
        return values

    def default_x0(self, total_budget: float) -> np.ndarray:
        """Spread the total budget uniformly over the optimized cells."""
        return np.full(self.size, total_budget / self.size)

    def default_bounds(
        self, total_budget: float
    ) -> list[tuple[float | None, float | None]]:
        """Default ``[0, total_budget]`` bounds per optimized cell."""
        return [(0.0, float(total_budget))] * self.size


class OptimizationVariables:
    """The complete decision vector: an ordered list of variables.

    Owns the single flat symbolic input, the contiguous slice layout, and the
    forward/inverse maps between the flat vector and per-variable labelled
    ``DataArray``s. The slice layout tiles ``[0, size)`` exactly: variables are
    laid out in order with no gaps or overlaps.

    Parameters
    ----------
    variables : list[OptimizationVariable]
        Variables in flat-vector order.
    flat_name : str
        Name of the flat symbolic variable.
    flat_dim : str
        Dimension name of the flat symbolic variable.
    """

    def __init__(
        self,
        variables: list[OptimizationVariable],
        flat_name: str = "budgets_flat",
        flat_dim: str = "budgets_flat",
    ) -> None:
        if not variables:
            raise ValueError("OptimizationVariables requires at least one variable")
        names = [variable.name for variable in variables]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate variable names: {names}")
        self.variables = list(variables)
        self.flat_dim = flat_dim
        # Reject a variable naming a different flat dimension rather than
        # realigning it: a variable whose own tensors are keyed on its name is
        # internally coherent, and silently renaming it produces a graph that
        # is not. The optimizer always builds both with the same name, so this
        # only ever fires on a hand-built inconsistency.
        for variable in self.variables:
            variable_flat_dim = getattr(variable, "flat_dim", None)
            if variable_flat_dim is None:
                variable.flat_dim = flat_dim
            elif variable_flat_dim != flat_dim:
                raise ValueError(
                    f"{variable.name}: flat_dim {variable_flat_dim!r} does not "
                    f"match the container's {flat_dim!r}. Build the variable "
                    "with the same flat dimension name."
                )

        self.slices: dict[str, slice] = {}
        start = 0
        for variable in self.variables:
            self.slices[variable.name] = slice(start, start + variable.size)
            start += variable.size
        self.size = start

        self.flat: XTensorVariable = ptx.xtensor(
            flat_name, shape=(self.size,), dims=(flat_dim,)
        )

    def __getitem__(self, name: str) -> OptimizationVariable:
        """Return the variable registered under ``name``."""
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise KeyError(name)

    def variable_slice(self, name: str) -> XTensorVariable:
        """Return the symbolic slice of the flat vector for variable ``name``."""
        flat_slice = self.slices[name]
        if flat_slice == slice(0, self.size):
            return self.flat
        return self.flat.isel({self.flat_dim: flat_slice})

    def substitutions(self) -> dict[str, XTensorVariable]:
        """Model substitution dict: one entry per variable, one joint graph."""
        return {
            variable.name: variable.to_model(self.variable_slice(variable.name))
            for variable in self.variables
        }

    def pack(self, values: DataArray | Mapping[str, DataArray]) -> np.ndarray:
        """Assemble a flat vector from labelled per-variable values.

        Parameters
        ----------
        values : DataArray or Mapping[str, DataArray]
            A mapping from variable name to labelled values. A bare
            ``DataArray`` is accepted when there is a single variable.

        Returns
        -------
        np.ndarray
            Flat vector of length :attr:`size`.
        """
        if isinstance(values, DataArray):
            if len(self.variables) != 1:
                raise ValueError(
                    "A bare DataArray is ambiguous when there are multiple variables; "
                    f"pass a dict with keys {[b.name for b in self.variables]}"
                )
            values = {self.variables[0].name: values}

        missing = {variable.name for variable in self.variables} - set(values)
        if missing:
            raise ValueError(f"pack() missing values for variables: {sorted(missing)}")

        x = np.empty(self.size, dtype=float)
        for variable in self.variables:
            x[self.slices[variable.name]] = variable.pack(values[variable.name])
        return x

    def unpack(self, x: np.ndarray) -> dict[str, DataArray]:
        """Split a flat solution into labelled per-variable ``DataArray``s."""
        x = np.asarray(x)
        if x.shape != (self.size,):
            raise ValueError(f"expected shape ({self.size},), got {x.shape}")
        return {
            variable.name: variable.unpack(x[self.slices[variable.name]])
            for variable in self.variables
        }

    def x0(self, total_budget: float) -> np.ndarray:
        """Default initial guess: each variable's ``default_x0`` concatenated."""
        return np.concatenate(
            [variable.default_x0(total_budget) for variable in self.variables]
        )

    def bounds(
        self,
        total_budget: float,
        overrides: Mapping[str, list[tuple[float | None, float | None]]] | None = None,
    ) -> list[tuple[float | None, float | None]]:
        """Assemble per-entry bounds, variable by variable.

        Parameters
        ----------
        total_budget : float
            Total budget, used by variables' default bounds.
        overrides : Mapping[str, list[tuple]] or None
            Explicit per-entry bounds for specific variables, replacing that
            variable's defaults. Each list must have one ``(low, high)`` pair
            per flat entry of the variable.

        Returns
        -------
        list[tuple]
            One ``(low, high)`` pair per flat entry, in flat-vector order.
        """
        overrides = dict(overrides or {})
        unknown = set(overrides) - {variable.name for variable in self.variables}
        if unknown:
            raise ValueError(
                f"bounds overrides for unknown variables: {sorted(unknown)}"
            )

        bounds: list[tuple[float | None, float | None]] = []
        for variable in self.variables:
            variable_bounds = overrides.get(
                variable.name, variable.default_bounds(total_budget)
            )
            if len(variable_bounds) != variable.size:
                raise ValueError(
                    f"{variable.name}: expected {variable.size} bounds pairs, "
                    f"got {len(variable_bounds)}"
                )
            bounds.extend(variable_bounds)
        return bounds
