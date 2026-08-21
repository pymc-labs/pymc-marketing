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
Everything the optimizer knows about *what that vector means*, which model
variable each segment substitutes, how a segment maps to model-space tensors
(the forward map), and how a solution maps back to labelled ``DataArray`` objects
(the inverse map), lives here, stated once per variable.

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
from collections.abc import Mapping, Sequence

import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor as ptx
from pymc import Model
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable
from xarray import DataArray

__all__ = [
    "FLAT_DIM",
    "LeverVariable",
    "MediaVariable",
    "OptimizationVariable",
    "OptimizationVariables",
    "align_to_model_coords",
]

#: Name of the flat decision vector's dimension, shared by every variable and
#: by the container so a rename cannot desynchronise them.
FLAT_DIM = "budgets_flat"


def align_to_model_coords(
    da: DataArray,
    coords: Mapping[str, list],
    *,
    label: str,
) -> DataArray:
    """Reindex a labelled input onto the model's budget coordinates.

    Every coordinate-bearing input the optimizer accepts (the optimization
    mask, the temporal distribution, cost per unit, budget bounds) is
    ultimately consumed positionally against the model's tensor layout, so all
    of them have to be expressed in the model's coordinate order before use.
    Reindexing alone is not enough to make that safe: it silently drops labels
    the model does not have and silently fills missing ones with NaN, so both
    are rejected here.

    Parameters
    ----------
    da : DataArray
        User-supplied input carrying some or all of the budget dims.
    coords : Mapping[str, list]
        The model's coordinates for the budget dims.
    label : str
        Name of the input, used in error messages.

    Returns
    -------
    DataArray
        ``da`` reindexed onto ``coords``. Dims outside ``coords`` (a date or
        bound dim, say) are left untouched, as are dims carrying no
        coordinates, which reindex labels positionally.

    Raises
    ------
    ValueError
        If the input carries coordinates the model does not have, or does not
        cover every coordinate the model does.
    """
    unknown = {
        dim: sorted(
            set(np.asarray(da.coords[dim].values).ravel().tolist()) - set(values)
        )
        for dim, values in coords.items()
        if dim in da.coords
    }
    unknown = {dim: labels for dim, labels in unknown.items() if labels}
    if unknown:
        raise ValueError(
            f"{label} has coordinates the model does not have: {unknown}. "
            "They would be dropped silently, so the input is rejected instead."
        )

    aligned = da.reindex(coords)
    if bool(aligned.isnull().any()):
        missing = {
            dim: sorted(
                set(values) - set(np.asarray(da.coords[dim].values).ravel().tolist())
            )
            for dim, values in coords.items()
            if dim in da.coords
        }
        missing = {dim: labels for dim, labels in missing.items() if labels}
        raise ValueError(
            f"{label} does not cover every model coordinate; missing {missing}."
            if missing
            else f"{label} contains missing values (NaN) after alignment."
        )
    return aligned


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

    def budget_contribution(self, z: XTensorVariable) -> XTensorVariable | None:
        """Monetary amount this variable draws from the shared budget.

        Returns None by default, for a variable optimized in its own units: a
        discount depth is not money. A variable that spends from the pot
        returns that spend, and the default budget-sum constraint totals every
        such contribution.

        The amount must follow the same convention as the media budgets, which
        are per-period rates rather than totals over the horizon, since the
        constraint sums the contributions directly.
        """
        return None


class MediaVariable(OptimizationVariable):
    """The media-budget decision variable.

    Owns the forward map from flat monetary budgets to the model's channel
    data tensor: scatter through the optimization mask, spread over
    ``num_periods`` (uniformly or via a fixed temporal distribution), divide by
    ``channel_scales``, convert monetary units via ``cost_per_unit``, and
    zero-pad ``adstock_periods`` for carry-over, and the inverse map from a
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
        scales_dims: tuple[str, ...] = ("channel",),
        date_dim: str = "date",
        budget_distribution_over_period_tensor: XTensorVariable | None = None,
        cost_per_unit_tensor: XTensorVariable | None = None,
        carry_in_values: np.ndarray | None = None,
        flat_dim: str = FLAT_DIM,
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
        self.scales_dims = tuple(scales_dims)
        self.dtype = dtype
        self.date_dim = date_dim
        self.budget_distribution_over_period_tensor = (
            budget_distribution_over_period_tensor
        )
        self.cost_per_unit_tensor = cost_per_unit_tensor
        self.flat_dim = flat_dim
        self.carry_in_values = (
            None if carry_in_values is None else np.asarray(carry_in_values)
        )
        if self.carry_in_values is not None:
            expected = (self.carry_in_values.shape[0], *tuple(mask.shape))
            if self.carry_in_values.shape != expected:
                raise ValueError(
                    f"{name}: carry_in_values has shape "
                    f"{self.carry_in_values.shape}, expected "
                    f"{expected} -- one leading period per carry-in date, over "
                    "the same cells as the mask."
                )
        self._bool_mask = np.asarray(self.mask.values).astype(bool)
        self._size = int(self._bool_mask.sum())
        if self._size == 0:
            raise ValueError(
                f"{name}: the optimization mask selects no cells, so there is "
                "nothing to optimize. Check budgets_to_optimize, or the "
                "posterior contributions it is auto-detected from."
            )

    @property
    def size(self) -> int:
        """Number of optimized budget cells."""
        return self._size

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the full (unmasked) budget tensor."""
        return self._bool_mask.shape

    @classmethod
    def from_model(
        cls,
        model: Model,
        name: str,
        *,
        num_periods: int,
        adstock_periods: int,
        mask: DataArray | None = None,
        carry_in_values: np.ndarray | None = None,
        scales: float | np.ndarray = 1.0,
        date_dim: str = "date",
        **kwargs,
    ) -> "MediaVariable":
        """Build a monetary variable by reading a named node's dims off a model.

        The media budgets are one instance of a monetary decision; a
        lower-funnel spend, or the reach an impression budget buys, is another.
        What distinguishes them is which node they substitute and over which
        dims they are decided, and both already live in the model, so callers
        name the node and this reads the rest.

        Parameters
        ----------
        model : Model
            The model holding the node.
        name : str
            Name of the monetary node to optimize. Must carry named dims,
            including *date_dim*.
        num_periods, adstock_periods : int
            Decision and carry-over period counts for the window.
        mask : DataArray or None
            Which cells to optimize, over the node's non-date dims. Defaults to
            all of them.
        carry_in_values : np.ndarray or None
            Spend already made before the window, one leading period per row.
        scales : float or np.ndarray
            Scale factors converting monetary amounts into the node's units.
        date_dim : str
            The model's date dimension, which the node must vary over.
        **kwargs
            Passed through to the constructor.

        Returns
        -------
        MediaVariable
            A monetary variable over ``name``.

        Raises
        ------
        ValueError
            If ``name`` has no named dims, or does not vary over *date_dim*.
        """
        if name not in model.named_vars_to_dims:
            raise ValueError(
                f"spend variable '{name}' is not a variable with named dims in "
                "the model, so the cells to optimize cannot be determined."
            )
        dims = tuple(model.named_vars_to_dims[name])
        if date_dim not in dims:
            raise ValueError(
                f"spend variable '{name}' has dims {dims}, which do not include "
                f"{date_dim!r}. A budget is spent over time; a quantity that "
                "does not vary by date is a lever, not a spend."
            )
        cell_dims = tuple(dim for dim in dims if dim != date_dim)
        if mask is None:
            shape = tuple(len(model.coords[dim]) for dim in cell_dims)
            mask = DataArray(
                np.ones(shape, dtype=bool),
                dims=cell_dims,
                coords={dim: list(model.coords[dim]) for dim in cell_dims},
            )
        return cls(
            name=name,
            mask=mask,
            num_periods=num_periods,
            adstock_periods=adstock_periods,
            carry_in_values=carry_in_values,
            channel_scales=scales,
            dtype=model[name].dtype,
            scales_dims=cell_dims,
            date_dim=date_dim,
            **kwargs,
        )

    def scattered(self, z: XTensorVariable) -> XTensorVariable:
        """Scatter the flat slice into the full budget-dims tensor.

        Non-optimized cells are zero. Values remain in monetary units.
        """
        if not self.dims:
            # One budget, decided over nothing but date -- a single national
            # spend. There are no cells to scatter into, the mask is 0-d, and
            # pytensor cannot index with a scalar boolean. The constructor has
            # already refused a mask that selects nothing, so the one cell here
            # is always optimized and reshaping is the whole of the scatter.
            return as_xtensor(z.values.reshape(()), dims=())
        budgets_zeros = pt.zeros(self.shape)
        budgets_zeros.name = "budgets_zeros"
        return as_xtensor(
            budgets_zeros[self._bool_mask].set(z.values),
            dims=self.dims,
        )

    def _apply_budget_distribution_over_period(
        self, z: XTensorVariable
    ) -> XTensorVariable:
        """Distribute each cell's budget across periods with fixed factors.

        The distribution factors are pre-masked to the optimized cells, so the
        product is computed on the flat slice and scattered back per period.
        Values stay in monetary units; :meth:`to_model` scales them.
        """
        repeated_budgets_flat = (
            z * self.budget_distribution_over_period_tensor
        ).transpose(self.date_dim, self.flat_dim)

        dims = (self.date_dim, *self.dims)
        zeros = pt.zeros((self.num_periods, *self.shape))
        repeated_budgets = as_xtensor(
            zeros[:, self._bool_mask].set(repeated_budgets_flat.values), dims=dims
        )

        # Factors are fractions of the per-period budget level; rescale so a
        # uniform distribution reproduces the default (repeat) behaviour.
        repeated_budgets *= self.num_periods

        return repeated_budgets

    def to_model(self, z: XTensorVariable) -> XTensorVariable:
        """Build the channel-data substitution tensor from the flat slice."""
        # Spread the monetary budgets over the periods, then scale. Scaling is
        # elementwise per channel and spreading is elementwise per period, so
        # the two commute; doing it after the branch means both the uniform and
        # the temporal path are scaled by construction, rather than only the
        # one that remembers to.
        if self.budget_distribution_over_period_tensor is not None:
            repeated_budgets = self._apply_budget_distribution_over_period(z)
        else:
            repeated_budgets = self.scattered(z).expand_dims(
                **{self.date_dim: self.num_periods}
            )

        repeated_budgets = repeated_budgets / as_xtensor(
            self.channel_scales,
            dims=() if np.ndim(self.channel_scales) == 0 else self.scales_dims,
        )

        # Convert from monetary units to original units using date-specific
        # rates. Applied AFTER time distribution so each period uses its own
        # cost rate.
        if self.cost_per_unit_tensor is not None:
            repeated_budgets = repeated_budgets / self.cost_per_unit_tensor

        repeated_budgets.name = "repeated_budgets"

        # The date axis is three blocks and only the middle one is decided:
        # spend already made before the window (fixed, so the adstock does not
        # start cold), the decisions, and zero-spend periods that catch the
        # carry-over the decisions produce after the window closes.
        blocks = []
        if self.carry_in_values is not None:
            blocks.append(
                ptx.as_xtensor(
                    pt.as_tensor_variable(self.carry_in_values).astype(self.dtype),
                    dims=(self.date_dim, *self.dims),
                )
            )
        blocks.append(repeated_budgets.astype(self.dtype))
        blocks.append(
            ptx.as_xtensor(
                pt.zeros(self.adstock_periods, dtype=self.dtype),
                dims=(self.date_dim,),
            )
        )
        repeated_budgets_with_carry_over = ptx.concat(blocks, dim=self.date_dim)
        repeated_budgets_with_carry_over.name = "repeated_budgets_with_carry_over"
        return repeated_budgets_with_carry_over

    def budget_contribution(self, z: XTensorVariable) -> XTensorVariable:
        """Media spends its whole slice, already in monetary units."""
        return self.scattered(z)

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


class LeverVariable(OptimizationVariable):
    """A non-media decision variable: one ``pm.Data`` node in native units.

    Where :class:`MediaVariable` converts monetary budgets into channel data,
    a lever is optimized directly in whatever units the model already uses for
    it (a discount fraction, a price index), so the forward map is the identity
    up to relabelling the flat dimension.

    A lever only moves if the optimizer's ``response_variable`` can reach it.
    The default ``total_media_contribution_original_scale`` is built from the
    channel contribution alone, so a lever acting through a ``MuEffect`` is
    invisible to it; score against ``total_response_original_scale``, which an
    :class:`~pymc_marketing.mmm.mmm.MMM` with mu effects registers.

    Levers do not participate in the default budget-sum constraint, which sums
    the media tensor alone: a discount depth is not money drawn from a shared
    pool. To constrain one, write a custom
    :class:`~pymc_marketing.mmm.constraints.Constraint` and reach its segment
    of the flat vector through
    ``optimizer.optimization_variables.variable_slice(name)``.

    Parameters
    ----------
    name : str
        Name of the model's ``pm.Data`` node this lever substitutes.
    dim : str
        The node's single dimension, which must not be the date dim.
    coords : list
        Coordinates of ``dim``, in the model's order.
    bounds : Sequence[tuple] or None
        Declared ``(low, high)`` bounds per entry in the lever's native units,
        or None to leave it unbounded.
    initial_value : np.ndarray
        The node's current value in the model, used as the warm start. It is
        read once, when the optimizer is constructed, which matches the rest of
        the optimizer: the objective graph is built and frozen at construction
        too, and this node is substituted out of it entirely. Updating the
        ``pm.Data`` node afterwards therefore changes neither the objective nor
        the warm start; rebuild the optimizer to pick up a new value.
    flat_dim : str
        Name of the flat decision vector's dimension.
    """

    def __init__(
        self,
        name: str,
        dim: str,
        coords: list,
        bounds: Sequence[tuple[float | None, float | None]] | None,
        initial_value: np.ndarray,
        flat_dim: str = FLAT_DIM,
    ) -> None:
        self.name = name
        self.dim = dim
        self.dims = (dim,)
        self.coords = {dim: list(coords)}
        self.bounds = list(bounds) if bounds is not None else None
        self.initial_value = np.ravel(np.asarray(initial_value, dtype=float))
        self.flat_dim = flat_dim
        if self.initial_value.shape != (self.size,):
            raise ValueError(
                f"{name}: initial value has {self.initial_value.size} entries, "
                f"expected {self.size} (the length of dim {dim!r})"
            )
        if self.bounds is not None and len(self.bounds) != self.size:
            raise ValueError(
                f"{name}: {len(self.bounds)} bounds pairs for {self.size} "
                f"entries of dim {dim!r}"
            )

    @property
    def size(self) -> int:
        """Number of lever entries, the length of its dim."""
        return len(self.coords[self.dim])

    def to_model(self, z: XTensorVariable) -> XTensorVariable:
        """Relabel the flat slice with the lever's own dim."""
        return z.rename({self.flat_dim: self.dim})

    def unpack(self, x: np.ndarray) -> DataArray:
        """Label a flat solution slice with the lever's dim and coords."""
        return DataArray(np.asarray(x, dtype=float), dims=self.dims, coords=self.coords)

    def pack(self, da: DataArray) -> np.ndarray:
        """Flatten labelled lever values into flat-vector order."""
        if set(da.dims) != set(self.dims):
            raise ValueError(
                f"{self.name}: expected dims {list(self.dims)}, got {list(da.dims)}"
            )
        return (
            align_to_model_coords(da, self.coords, label=self.name)
            .transpose(*self.dims)
            .values
        )

    @classmethod
    def from_model(
        cls,
        model: Model,
        name: str,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        *,
        date_dim: str = "date",
        flat_dim: str = FLAT_DIM,
    ) -> "LeverVariable":
        """Build a lever by reading a named node's dim, coords and value off a model.

        Everything a lever needs besides its bounds already lives in the model,
        so callers name the node and this reads the rest, raising if the node
        cannot serve as a lever.

        Parameters
        ----------
        model : Model
            The model holding the node.
        name : str
            Name of the node to optimize. Must carry named dims.
        bounds : Sequence[tuple] or None
            Declared ``(low, high)`` bounds per entry, in native units.
        date_dim : str
            The model's date dimension, which a lever may not vary over.
        flat_dim : str
            Name of the flat decision vector's dimension.

        Returns
        -------
        LeverVariable
            A lever over ``name``.

        Raises
        ------
        ValueError
            If ``name`` has no named dims in the model, or does not have
            exactly one dim other than ``date_dim``.
        """
        if name not in model.named_vars_to_dims:
            raise ValueError(
                f"optimizable_vars entry '{name}' is not a variable "
                "with named dims in the model."
            )
        dims = tuple(model.named_vars_to_dims[name])
        if len(dims) != 1 or dims[0] == date_dim:
            raise ValueError(
                f"optimizable_vars entry '{name}' must have exactly "
                f"one dim, and not the {date_dim!r} dim; got "
                f"{dims}. Date-varying optimizable variables "
                "are not supported."
            )
        return cls(
            name=name,
            dim=dims[0],
            coords=list(model.coords[dims[0]]),
            bounds=bounds,
            initial_value=model[name].get_value(),
            flat_dim=flat_dim,
        )

    def default_x0(self, total_budget: float) -> np.ndarray:
        """Warm start at the lever's current model value, clipped to bounds."""
        x0 = self.initial_value.copy()
        if self.bounds is not None:
            lows = np.array([-np.inf if lo is None else lo for lo, _ in self.bounds])
            highs = np.array([np.inf if hi is None else hi for _, hi in self.bounds])
            x0 = np.clip(x0, lows, highs)
        return x0

    def default_bounds(
        self, total_budget: float
    ) -> list[tuple[float | None, float | None]]:
        """Return the declared native-unit bounds, or unbounded."""
        if self.bounds is not None:
            return list(self.bounds)
        return [(None, None)] * self.size


class OptimizationVariables:
    """The complete decision vector: an ordered list of variables.

    Owns the single flat symbolic input, the contiguous slice layout, and the
    forward/inverse maps between the flat vector and per-variable labelled
    ``DataArray`` objects. The slice layout tiles ``[0, size)`` exactly: variables are
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
        flat_name: str = FLAT_DIM,
        flat_dim: str = FLAT_DIM,
    ) -> None:
        if not variables:
            raise ValueError("OptimizationVariables requires at least one variable")
        names = [variable.name for variable in variables]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate variable names: {names}")
        # A tuple, not a list: the slice layout is computed once below, so a
        # variable appended later would silently desynchronise it from the
        # flat input. The container is exposed publicly for constraints.
        self.variables: tuple[OptimizationVariable, ...] = tuple(variables)
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

        # Which variables spend is fixed once the layout is, and answering it
        # builds a throwaway symbolic tensor per variable. Settled here so
        # repeated `x0` calls -- a warm start, a sweep over budgets -- do not
        # rebuild the same graphs to reach the same answer.
        self._spending: tuple[OptimizationVariable, ...] = tuple(
            variable
            for variable in self.variables
            if variable.budget_contribution(self.variable_slice(variable.name))
            is not None
        )

    def variable_slice(self, name: str) -> XTensorVariable:
        """Return the symbolic slice of the flat vector for variable ``name``.

        This is a node in the PyTensor graph, not a buffer: it carries no
        values and nothing is copied or mutated. A variable spanning the whole
        vector gets the flat input itself; otherwise an ``isel`` node over its
        contiguous segment.
        """
        flat_slice = self.slices[name]
        if flat_slice == slice(0, self.size):
            return self.flat
        return self.flat.isel({self.flat_dim: flat_slice})

    def budget_contributions(self) -> list[XTensorVariable]:
        """Monetary contributions of every variable that spends from the pot."""
        contributions = []
        for variable in self.variables:
            contribution = variable.budget_contribution(
                self.variable_slice(variable.name)
            )
            if contribution is not None:
                contributions.append(contribution)
        return contributions

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

        known = {variable.name for variable in self.variables}
        missing = known - set(values)
        if missing:
            raise ValueError(f"pack() missing values for variables: {sorted(missing)}")
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f"pack() got values for unknown variables: {sorted(unknown)}; "
                f"expected {sorted(known)}"
            )

        x = np.empty(self.size, dtype=float)
        for variable in self.variables:
            x[self.slices[variable.name]] = variable.pack(values[variable.name])
        return x

    def unpack(self, x: np.ndarray) -> dict[str, DataArray]:
        """Split a flat solution into labelled per-variable ``DataArray`` objects."""
        x = np.asarray(x)
        if x.shape != (self.size,):
            raise ValueError(f"expected shape ({self.size},), got {x.shape}")
        return {
            variable.name: variable.unpack(x[self.slices[variable.name]])
            for variable in self.variables
        }

    def spending_variables(self) -> list[OptimizationVariable]:
        """Return the variables that draw from the shared budget.

        Read off ``budget_contribution`` rather than a second flag, so there is
        one answer to "does this spend?" and it cannot disagree with itself.
        Settled once at construction, since the layout it depends on is fixed
        there too.
        """
        return list(self._spending)

    def x0(self, total_budget: float) -> np.ndarray:
        """Default initial guess: each variable's ``default_x0`` concatenated.

        The budget is shared out across the *cells* of every spending variable,
        so the guess sums to ``total_budget`` however many of them there are and
        every spending cell starts at the same amount. With one spender this is
        exactly ``default_x0(total_budget)``; with two, splitting matters --
        giving each the whole budget would start the solver at twice it, outside
        the equality constraint before the first step.

        A variable that does not spend is handed the total untouched: it is in
        its own units, and its default ignores the figure anyway.
        """
        spenders = self.spending_variables()
        cells = sum(variable.size for variable in spenders)
        shares = (
            {
                variable.name: total_budget * variable.size / cells
                for variable in spenders
            }
            if cells
            else {}
        )
        return np.concatenate(
            [
                variable.default_x0(shares.get(variable.name, total_budget))
                for variable in self.variables
            ]
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
