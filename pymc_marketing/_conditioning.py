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

"""Helpers behind :class:`~pymc_marketing.pytensor_utils.ConditionedModel`.

Private: the posterior layout checks, the traversal that finds the posterior
variables a graph reads, and the reorder to a model's coordinates, shared by
:mod:`pymc_marketing.pytensor_utils` and the MMM consumers of a conditioned
model.  None of it is public API.
"""

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import pandas as pd
import xarray as xr
from pymc.model.core import Model
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors


class VariableLayout(NamedTuple):
    """The non-sample layout a posterior variable was constructed with."""

    dims: tuple[str, ...]
    coords: dict[str, pd.Index]
    shape: tuple[int, ...]


def variable_layout(
    posterior_da: xr.DataArray, *, sample_dims: tuple[str, ...]
) -> VariableLayout:
    """Record the non-sample dims, labels and shape of ``posterior_da``."""
    dims = tuple(str(dim) for dim in posterior_da.dims if str(dim) not in sample_dims)
    return VariableLayout(
        dims=dims,
        coords={
            dim: posterior_da.indexes[dim]
            for dim in dims
            if dim in posterior_da.indexes
        },
        shape=tuple(posterior_da.sizes[dim] for dim in dims),
    )


def non_sample_dims(
    posterior_da: xr.DataArray, *, sample_dims: tuple[str, ...]
) -> tuple[str, ...]:
    """Return the non-sample dims of ``posterior_da`` in storage order.

    Parameters
    ----------
    posterior_da : xr.DataArray
        A posterior variable.
    sample_dims : tuple of str
        The dims that index draws, such as ``("chain", "draw")`` or ``("sample",)``.

    Returns
    -------
    tuple of str
        Every other dim of ``posterior_da``, in the order it is stored in.
    """
    return tuple(str(dim) for dim in posterior_da.dims if str(dim) not in sample_dims)


def check_aligned_dims(
    name: str, dims: tuple[str, ...], layout: VariableLayout
) -> None:
    """Raise if ``dims`` is not the dim set the variable was constructed with.

    The order is free here: :func:`align_posterior_dim` and the final transpose
    put the axes back in the layout's order.

    Parameters
    ----------
    name : str
        The posterior variable, named in the error.
    dims : tuple of str
        Its non-sample dims, as returned by :func:`non_sample_dims`.
    layout : VariableLayout
        The layout the variable was constructed with.

    Raises
    ------
    ValueError
        If ``dims`` and ``layout.dims`` are different sets.
    """
    if set(dims) != set(layout.dims):
        raise ValueError(
            f"Posterior variable {name!r} has dims {dims}, expected {layout.dims}."
        )


def align_posterior_dim(
    name: str,
    posterior_da: xr.DataArray,
    dim: str,
    bound: pd.Index | None,
) -> xr.DataArray:
    """Check ``dim``'s labels against ``bound`` and reorder the axis to them.

    A dimension labeled on one side only cannot be checked, and a duplicated
    layout label could not be realigned by ``sel`` -- it would silently pick
    one column twice -- so both are refused.  An unlabeled dimension, or one
    whose labels already match, keeps its labels in the bound order.

    Parameters
    ----------
    name : str
        The posterior variable, named in the error.
    posterior_da : xr.DataArray
        The variable to align.
    dim : str
        The dimension to check and reorder.
    bound : pd.Index or None
        The labels ``dim`` was constructed with, or ``None`` when it had none.

    Returns
    -------
    xr.DataArray
        ``posterior_da`` with ``dim`` in the order of ``bound``, or unchanged
        when ``dim`` is unlabeled on both sides.

    Raises
    ------
    ValueError
        If ``dim`` is labeled on one side only, ``bound`` holds a duplicated
        label, or the two label sets differ in length or content.
    """
    given = posterior_da.indexes.get(dim)
    if (bound is None) != (given is None):
        raise ValueError(
            f"Posterior variable {name!r}: dimension {dim!r} is labeled on "
            "one side only. Label both sides or neither; positions cannot "
            "be checked against labels."
        )
    # After the check above both are labeled or neither is.
    if bound is None or given is None:
        return posterior_da
    # A duplicated layout label could not be realigned by ``sel``: it would
    # silently pick one column twice, so the layout has to be a set.
    if not bound.is_unique:
        raise ValueError(
            f"Posterior variable {name!r} has duplicated {dim} labels: "
            f"{bound[bound.duplicated()].unique().tolist()}."
        )
    # pd.Index compares datetimes by value across storage units.  The bound
    # index is unique, so equal lengths plus an empty difference means the same
    # set of labels, with any duplicate on the given side surfacing as a
    # missing label.  No sorting, so a mixed-type object axis is fine.
    if len(bound) != len(given) or not bound.difference(given).empty:
        extra, missing = given.difference(bound), bound.difference(given)
        raise ValueError(
            f"Posterior variable {name!r} has different {dim} labels: "
            f"{len(extra)} not bound, {len(missing)} missing "
            f"(e.g. {list(extra[:3])} / {list(missing[:3])})."
        )
    return posterior_da.sel({dim: bound})


def check_aligned_shape(
    name: str, shape: tuple[int, ...], layout: VariableLayout
) -> None:
    """Raise if the non-sample ``shape`` differs from the constructed one.

    Parameters
    ----------
    name : str
        The posterior variable, named in the error.
    shape : tuple of int
        Its non-sample shape, in the layout's dim order.
    layout : VariableLayout
        The layout the variable was constructed with.

    Raises
    ------
    ValueError
        If ``shape`` differs from ``layout.shape``.  Only the sample dimension
        may change length.
    """
    if shape != layout.shape:
        raise ValueError(
            f"Posterior variable {name!r} has shape {shape} over dims "
            f"{layout.dims}, expected {layout.shape}. Only the sample "
            "dimension may change length."
        )


def aligned_posterior(
    name: str,
    posterior_da: xr.DataArray,
    layout: VariableLayout,
    *,
    sample_dims: tuple[str, ...],
) -> xr.DataArray:
    """Return ``posterior_da`` laid out and checked against ``layout``.

    The layout comes from the posterior a graph was constructed with, so a
    variable is validated against its own construction geometry, not against
    the whole posterior.  The returned array has its non-sample dims in the
    layout's order and the incoming labels reordered to the layout's.

    Raises
    ------
    ValueError
        If the dims differ, a dimension is labeled on one side only, a label
        set differs (or the layout's labels are duplicated), or a non-sample
        length differs.  The sample dimension may change length freely.
    """
    check_aligned_dims(
        name, non_sample_dims(posterior_da, sample_dims=sample_dims), layout
    )
    for dim in layout.dims:
        posterior_da = align_posterior_dim(
            name, posterior_da, dim, layout.coords.get(dim)
        )
    posterior_da = posterior_da.transpose(*sample_dims, *layout.dims)
    check_aligned_shape(name, posterior_da.shape[len(sample_dims) :], layout)
    return posterior_da


def needed_posterior_names(
    pymc_model: Model,
    response_vars: Sequence[Variable],
    frozen_deterministics: Sequence[str] = (),
) -> list[str]:
    """Return the posterior variables ``response_vars`` read.

    The free RVs and the frozen deterministics are the blockers of the
    traversal: an extraction substitutes their posterior draws instead of
    traversing their subgraphs.  The order is the graph traversal's, so it is
    stable for a given response but says nothing about the posterior layout.

    Parameters
    ----------
    pymc_model : pm.Model
        The model ``response_vars`` belong to.
    response_vars : sequence of Variable
        The response graphs to traverse.
    frozen_deterministics : sequence of str, default ()
        Deterministics held at their posterior values; a name the model does
        not have is ignored.

    Returns
    -------
    list of str
        The names of the free RVs and frozen deterministics the responses
        depend on, each once.
    """
    blockers = set(pymc_model.free_RVs)
    blockers.update(
        pymc_model[name]
        for name in frozen_deterministics
        if name in pymc_model.named_vars
    )
    names: list[str] = []
    for rv in ancestors(response_vars, blockers=blockers):
        if rv in blockers and rv.name is not None:
            names.append(rv.name)
    return names


def reorder_to_model_coords(
    posterior: xr.Dataset, model_coords: Mapping[str, Any]
) -> xr.Dataset:
    """Return ``posterior`` with each labeled non-sample dim in the model's order.

    A graph reads draws and data by position in the model's coord order, so the
    model is the reference layout.  :class:`ConditionedModel` applies this to
    every posterior it receives, and ``Incrementality`` to the channel data its
    baseline is evaluated on.  A dim whose labels differ from the model's in
    value or length -- a refit window, a thinned posterior -- is left as given,
    for whatever reads it to validate.  ``chain`` and ``draw`` are never
    reordered.

    Parameters
    ----------
    posterior : xr.Dataset
        The draws, or any dataset over the model's dims, to reorder.
    model_coords : mapping
        The coordinates of the model, such as ``pm.Model.coords``.

    Returns
    -------
    xr.Dataset
        ``posterior`` in the model's label order on every dim whose labels
        equal the model's (unique) labels as a set; other dims keep their order.
    """
    indexers: dict[str, pd.Index] = {}
    for name in posterior.sizes:
        dim = str(name)
        if dim in ("chain", "draw"):
            continue
        index = posterior.indexes.get(dim)
        if index is None:
            continue
        coords = model_coords.get(dim)
        if coords is None:
            continue
        reference = pd.Index(coords)
        # A duplicated reference label would make ``sel`` drop a column, so
        # that dim is stored as given and refused later by whatever reads it.
        if not reference.is_unique or index.equals(reference):
            continue
        # Compared by value as ``aligned_posterior`` does: pd.Index handles
        # datetime storage units and ``datetime.date`` labels, unlike ``set``.
        if len(reference) != len(index) or not reference.difference(index).empty:
            continue
        indexers[dim] = reference
    return posterior.sel(indexers) if indexers else posterior


def transpose_to_model_dims(
    posterior: xr.Dataset,
    model_dims: Mapping[str, Sequence[str | None]],
    *,
    sample_dims: tuple[str, ...] = ("chain", "draw"),
) -> xr.Dataset:
    """Return ``posterior`` with each variable's dims in the order the model declares.

    PyMC's posterior predictive reads a variable's values in storage order
    against the dims its model declares, so a variable stored as ``(channel,
    geo)`` for a model that declares ``(geo, channel)`` is read transposed, and
    silently so when the two axes have the same length.  A variable whose
    non-sample dims are the declared dims in another order is transposed.  Any
    other variable (one the model does not declare, or one over another set of
    dims) is left as given.

    Parameters
    ----------
    posterior : xr.Dataset
        The draws to transpose.
    model_dims : mapping
        The dims of each model variable, such as ``pm.Model.named_vars_to_dims``.
    sample_dims : tuple of str, default ("chain", "draw")
        The dims that index draws.  They stay in front, in their stored order.

    Returns
    -------
    xr.Dataset
        ``posterior`` with every such variable in the model's dim order, or
        ``posterior`` itself when every variable already is.
    """
    transposed: dict[str, xr.DataArray] = {}
    for name, posterior_da in posterior.data_vars.items():
        declared = tuple(str(dim) for dim in model_dims.get(str(name)) or ())
        stored = non_sample_dims(posterior_da, sample_dims=sample_dims)
        # Only the same dims in another order are moved.
        if stored == declared or sorted(stored) != sorted(declared):
            continue
        leading = [dim for dim in posterior_da.dims if dim in sample_dims]
        transposed[str(name)] = posterior_da.transpose(*leading, *declared)
    return posterior.assign(transposed) if transposed else posterior
