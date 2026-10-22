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

"""PyTensor utility functions."""

import contextlib
import warnings
from collections import Counter, OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, NamedTuple, cast, overload

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pymc import do
from pymc.model.core import Model
from pymc.model.fgraph import (
    ModelValuedVar,
    ModelVar,
    clone_model,
    fgraph_from_model,
    model_from_fgraph,
)
from pymc.pytensorf import rvs_in_graph, toposort_replace
from pytensor.compile.sharedvalue import SharedVariable, shared
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting import rewrite_graph
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.op import RNGConsumerOp
from pytensor.xtensor import xtensor_constant
from pytensor.xtensor.type import XTensorVariable, as_xtensor
from pytensor.xtensor.vectorization import vectorize_graph

__all__ = [
    "ConditionedModel",
    "ModelSamplerEstimator",
    "SharedPosterior",
    "VariableLayout",
    "align_posterior_dim",
    "aligned_posterior",
    "check_aligned_dims",
    "check_aligned_shape",
    "extract_response_distribution",
    "merge_models",
    "needed_posterior_names",
    "non_sample_dims",
    "reorder_to_model_coords",
    "transpose_to_model_dims",
    "validate_unique_value_vars",
    "variable_layout",
]

_GRAPH_CACHE_SIZE = 32
"""Number of extractions a :class:`ConditionedModel` caches before evicting the least recent."""


def _rebuild_model_var(var, *, name: str, dims: tuple[str, ...]):
    """Return a copy of a ``ModelVar``-wrapped variable with new ``name``/``dims`` props."""
    op = var.owner.op
    if isinstance(op, ModelValuedVar):
        new_op = type(op)(name=name, dims=dims, transform=op.transform)
    else:
        new_op = type(op)(name=name, dims=dims)
    return new_op(*var.owner.inputs)


def _prefix_model(f2, prefix: str, exclude_vars: set | None = None):
    """Prefix variable and dimension names in a FunctionGraph.

    Variables listed in ``exclude_vars`` (e.g., a shared variable like ``channel_data``)
    are kept unprefixed, and their dims/coords are also preserved without prefix.
    """
    if exclude_vars is None:
        exclude_vars = set()

    model_nodes = [node for node in f2.toposort() if isinstance(node.op, ModelVar)]

    # Dimensions that belong to excluded variables are never prefixed
    exclude_dims: set[str] = {
        dim
        for node in model_nodes
        if node.op.name in exclude_vars
        for dim in node.op.dims
    }

    def _prefixed(name: str) -> str:
        return name if name in exclude_dims else f"{prefix}_{name}"

    # Rebuild every ModelVar op with the prefixed name/dims (they are Op props).
    replacements = []
    for node in model_nodes:
        op = node.op
        old_name = op.name
        new_name = old_name if old_name in exclude_vars else f"{prefix}_{old_name}"
        new_dims = tuple(_prefixed(dim) for dim in op.dims)
        if new_name == old_name and new_dims == op.dims:
            continue
        if isinstance(op, ModelValuedVar) and new_name != old_name:
            # The value var of a free RV is a graph input named "<name>" or
            # "<name>_<transform>__". Observed value vars are either Data (a
            # ModelNamed node, rebuilt in this loop) or unnamed constants.
            value = node.inputs[1]
            if isinstance(value.name, str) and value.name and value.owner is None:
                value.name = f"{prefix}_{value.name}"
        replacements.append(
            (
                node.outputs[0],
                _rebuild_model_var(node.outputs[0], name=new_name, dims=new_dims),
            )
        )
    # Consumers before producers, so no stale ModelVar node is re-imported
    toposort_replace(f2, tuple(replacements), reverse=True)

    f2._coords = {_prefixed(k): v for k, v in f2._coords.items()}  # type: ignore[attr-defined]
    f2._dim_lengths = {  # type: ignore[attr-defined]
        _prefixed(k): v
        for k, v in f2._dim_lengths.items()  # type: ignore[attr-defined]
    }

    return f2


def merge_models(
    models: list[Model],
    *,
    prefixes: list[str] | None = None,
    merge_on: str | None = None,
) -> Model:
    """Merge multiple PyMC models into a single model.

    Parameters
    ----------
    models : list of pm.Model
        List of models to merge.
    prefixes : list of str or None
        List of prefixes for each model. If None, will auto-generate as 'model1', 'model2', ...
    merge_on : str or None
        Variable name to merge on (shared across all models) - this variable will NOT be prefixed.

    Returns
    -------
    pm.Model
        Merged model.
    """
    if len(models) < 2:
        raise ValueError("Need at least 2 models to merge")

    # Auto-generate prefixes if not provided
    if prefixes is None:
        prefixes = [f"model{i + 1}" for i in range(len(models))]
    elif len(prefixes) != len(models):
        raise ValueError(
            f"Number of prefixes ({len(prefixes)}) must match number of models ({len(models)})"
        )

    # Variables to exclude from prefixing
    exclude_vars = {merge_on} if merge_on else set()

    # Convert all models to FunctionGraphs and apply prefixes
    fgraphs: list[FunctionGraph] = []
    for model, prefix in zip(models, prefixes, strict=False):
        fg, _ = fgraph_from_model(model, inlined_views=True)
        if prefix is not None:
            fg = _prefix_model(fg, prefix=prefix, exclude_vars=exclude_vars)
        fgraphs.append(fg)

    # Handle merge_on logic
    if merge_on is not None:
        # Find the merge variable in the first model (unprefixed)
        first_merge_vars = [out for out in fgraphs[0].outputs if out.name == merge_on]
        if not first_merge_vars:
            raise ValueError(f"Variable '{merge_on}' not found in first model")
        first_merge_var = first_merge_vars[0]

        # Replace the merge variable in all other models with the one from the first model
        for i in range(1, len(fgraphs)):
            merge_vars = [out for out in fgraphs[i].outputs if out.name == merge_on]
            if not merge_vars:
                raise ValueError(f"Variable '{merge_on}' not found in model {i + 1}")
            fgraphs[i].replace(merge_vars[0], first_merge_var, import_missing=True)
            # Shared dims of the merge variable must use a single dim-length variable
            first_dim_lengths = fgraphs[0]._dim_lengths  # type: ignore[attr-defined]
            for dim, length in list(fgraphs[i]._dim_lengths.items()):  # type: ignore[attr-defined]
                first_length = first_dim_lengths.get(dim)
                if first_length is None or first_length is length:
                    continue
                if length in fgraphs[i].variables:
                    fgraphs[i].replace(length, first_length, import_missing=True)
                fgraphs[i]._dim_lengths[dim] = first_length  # type: ignore[attr-defined]

    # Combine all outputs
    all_outputs = []
    for fg in fgraphs:
        all_outputs.extend(fg.outputs)

    # Create merged FunctionGraph
    f = FunctionGraph(outputs=all_outputs, clone=False)

    # Merge coordinates and dim lengths from all models (first model wins on shared dims)
    merged_coords: dict = {}
    merged_dim_lengths: dict = {}
    for fg in fgraphs:
        merged_coords.update(fg._coords)  # type: ignore[attr-defined]
        for dim, length in fg._dim_lengths.items():  # type: ignore[attr-defined]
            merged_dim_lengths.setdefault(dim, length)
    f._coords = merged_coords  # type: ignore[attr-defined]
    f._dim_lengths = merged_dim_lengths  # type: ignore[attr-defined]

    return model_from_fgraph(f, mutate_fgraph=True)


def validate_unique_value_vars(model: Model) -> None:
    """Validate that a model has unique, non-null value var names and 1:1 mappings.

    This checks that:
    - All entries in ``model.value_vars`` have unique, non-empty names
    - Keys of ``model.values_to_rvs`` (value vars) also have unique names
    - ``model.rvs_to_values`` mapping is consistent (bijection by names)
    """
    # Check value_vars names are unique and non-empty
    value_vars = list(getattr(model, "value_vars", []))
    value_var_names = [getattr(v, "name", None) for v in value_vars]
    if any(n is None or n == "" for n in value_var_names):
        raise ValueError("Found unnamed value variables in model.value_vars")
    dup_vnames = [n for n, c in Counter(value_var_names).items() if c > 1]
    if dup_vnames:
        raise ValueError(f"Duplicate value variable names: {dup_vnames}")

    # Check values_to_rvs keys are unique by name
    v2r = getattr(model, "values_to_rvs", {})
    v2r_value_vars = list(v2r.keys())
    v2r_value_names = [getattr(v, "name", None) for v in v2r_value_vars]
    if any(n is None or n == "" for n in v2r_value_names):
        raise ValueError("Found unnamed value variables in values_to_rvs")
    # Some observed/deterministic value-vars may legitimately share names across merged models
    # if they were intentionally merged on (e.g., merge_on) or are non-free and identical.
    # Only enforce uniqueness among value vars that correspond to free RVs.
    _ = {
        getattr(v2r[v], "name", None)
        for v in v2r_value_vars
        if v in getattr(model, "value_vars", [])
        and v2r.get(v) in getattr(model, "free_RVs", [])
    }
    # Map back to the value-var names for those free RVs
    free_value_var_names = [
        getattr(model.rvs_to_values[rv], "name", None) for rv in model.free_RVs
    ]
    dup_map_names = [n for n, c in Counter(free_value_var_names).items() if n and c > 1]
    if dup_map_names:
        raise ValueError("Duplicate value variable names for free RVs: {dup_map_names}")

    # Check consistency of reverse mapping by names
    r2v = getattr(model, "rvs_to_values", {})
    # Names on the value side of both dicts should align set-wise
    r2v_value_names = [getattr(v, "name", None) for v in r2v.values()]
    if set(r2v_value_names) != set(v2r_value_names):
        raise ValueError(
            "Mismatch between values_to_rvs and rvs_to_values by value var names"
        )


# Helpers behind SharedPosterior and ConditionedModel: the layout a posterior
# variable is checked against, the posterior variables a graph reads, and the
# reorder and transpose of a posterior to a model's coordinates and dims.


class VariableLayout(NamedTuple):
    """The non-sample layout a posterior variable was constructed with.

    :class:`SharedPosterior` and :class:`ConditionedModel` record one per
    posterior variable and check every later posterior against it.

    Attributes
    ----------
    dims : tuple of str
        The non-sample dims, in storage order.
    coords : dict of str to pd.Index
        The labels of each labeled dim; an unlabeled dim has no entry.
    shape : tuple of int
        The length of each dim in ``dims``.
    """

    dims: tuple[str, ...]
    coords: dict[str, pd.Index]
    shape: tuple[int, ...]


def variable_layout(
    posterior_da: xr.DataArray, *, sample_dims: tuple[str, ...]
) -> VariableLayout:
    """Record the non-sample dims, labels and shape of ``posterior_da``.

    Parameters
    ----------
    posterior_da : xr.DataArray
        A posterior variable.
    sample_dims : tuple of str
        The dims that index draws, such as ``("chain", "draw")`` or ``("sample",)``.

    Returns
    -------
    VariableLayout
        Its non-sample dims in storage order, the labels of the labeled ones
        and their lengths.
    """
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
    the whole posterior.

    Parameters
    ----------
    name : str
        The posterior variable, named in the error.
    posterior_da : xr.DataArray
        The variable to lay out.
    layout : VariableLayout
        The layout it was constructed with.
    sample_dims : tuple of str
        The dims that index draws, such as ``("chain", "draw")`` or ``("sample",)``.

    Returns
    -------
    xr.DataArray
        ``posterior_da`` with the sample dims first, then the layout's dims in
        the layout's order, each labeled dim in the layout's label order.

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


def _posterior_sample_major(idata: xr.DataTree | xr.Dataset) -> xr.Dataset:
    """Stack ``chain`` and ``draw`` into a leading ``sample`` dimension."""
    extracted = az.extract(idata)
    # A single-variable posterior comes back squeezed to a DataArray.
    if isinstance(extracted, xr.DataArray):
        extracted = extracted.to_dataset()
    return extracted.transpose("sample", ...)  # type: ignore


def _array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """``np.array_equal`` that treats NaN as equal to itself for float arrays."""
    equal_nan = np.issubdtype(a.dtype, np.inexact) and np.issubdtype(
        b.dtype, np.inexact
    )
    return bool(np.array_equal(a, b, equal_nan=equal_nan))


class SharedPosterior:
    """Posterior draws held in PyTensor shared variables.

    :func:`extract_response_distribution` conditions a graph on posterior draws
    by substituting them as constants, which welds every graph -- and every
    function compiled from it -- to one posterior.  Passing a
    ``SharedPosterior`` instead binds the draws through shared variables, so
    :meth:`set_posterior` can point an already compiled function at a new set
    of draws without extracting or compiling again.  That is what makes a
    loop over many posteriors (a resampled or updated posterior per
    iteration) affordable: the compile happens once.

    One instance can back several graphs.  A variable is created the first
    time a graph needs it and reused by every later extraction that passes the
    same instance, so a single :meth:`set_posterior` call rebinds all of them
    at once.  A later extraction must therefore be conditioned on the draws
    the instance already holds; handing it a different posterior raises,
    since the graph would otherwise silently read the bound draws.

    The contract for a rebind: every bound variable is present with the
    dtype it was bound with; every non-sample dimension keeps its name and
    its set of labels, compared by value (so a reordered ``channel`` axis is
    realigned, and a datetime axis may change storage unit); a dimension
    bound without labels must arrive without labels; the number of draws is
    free.  Anything else is refused, and a rebind is atomic: every variable
    is validated and laid out before any is written.
    Only the variables bound to this instance are checked, which is why a
    posterior variable no graph reads may change freely.
    An extraction is atomic too, so a failure leaves the instance exactly as it was.
    A later rebind that satisfies every graph already handed out is then accepted.

    Examples
    --------
    .. code-block:: python

        shared_posterior = SharedPosterior()
        graph = extract_response_distribution(
            model, idata, "channel_contribution", shared_posterior=shared_posterior
        )
        fn = function([budgets], graph)  # compiled once

        fn(x)  # under ``idata``
        shared_posterior.set_posterior(other_idata)
        fn(x)  # under ``other_idata``, no recompile
    """

    def __init__(self) -> None:
        self._variables: dict[str, SharedVariable] = {}
        self._layouts: dict[str, VariableLayout] = {}

    def __repr__(self) -> str:
        """Show each bound variable with its current ``(sample, ...)`` shape."""
        bound = ", ".join(
            f"{name}{tuple(self._variables[name].get_value(borrow=True).shape)}"
            for name in self._variables
        )
        return f"SharedPosterior({bound})"

    @property
    def variables(self) -> dict[str, SharedVariable]:
        """The shared variables, keyed by posterior variable name."""
        return dict(self._variables)

    @property
    def dims(self) -> dict[str, tuple[str, ...]]:
        """The dimension order each variable is stored in, ``sample`` first."""
        return {
            name: ("sample", *layout.dims) for name, layout in self._layouts.items()
        }

    def _aligned_values(self, name: str, posterior_da: xr.DataArray) -> np.ndarray:
        """Return ``posterior_da`` laid out and typed like the bound variable ``name``.

        Raises ``ValueError`` on any mismatch with the contract in the class
        docstring; writes nothing.  The returned array is a fresh allocation
        the shared variable can own outright, which keeps the write phase of
        :meth:`set_posterior` infallible.
        """
        aligned = aligned_posterior(
            name, posterior_da, self._layouts[name], sample_dims=("sample",)
        )
        var = self._variables[name]
        # astype always allocates, so the shared variable owns the array and
        # borrow=True cannot alias the caller's data.
        return aligned.values.astype(var.type.dtype)

    def _get_or_create(self, posterior_da: xr.DataArray, *, name: str, dtype: str):
        """Return the xtensor view of ``name``'s shared variable, creating it if needed.

        On reuse the draws handed in must be the ones already bound: the graph
        will read the shared variable, so a different posterior here would be
        silently ignored.  A new variable is validated before it is inserted,
        so a refusal leaves the registries untouched.
        """
        if name not in self._variables:
            layout = variable_layout(posterior_da, sample_dims=("sample",))
            # Labels are a set: a duplicated label could not be realigned by
            # ``sel`` later, which would silently pick one column twice.
            for dim, index in layout.coords.items():
                if not index.is_unique:
                    raise ValueError(
                        f"Posterior variable {name!r} has duplicated {dim} labels: "
                        f"{index[index.duplicated()].unique().tolist()}."
                    )
            self._variables[name] = shared(posterior_da.values.astype(dtype), name=name)
            self._layouts[name] = layout
            dims = ("sample", *layout.dims)
        else:
            var = self._variables[name]
            if np.dtype(dtype) != np.dtype(var.type.dtype):
                raise ValueError(
                    f"Posterior variable {name!r} is bound as {var.type.dtype}, "
                    f"but this graph wants {dtype}."
                )
            dims = ("sample", *self._layouts[name].dims)
            values = self._aligned_values(name, posterior_da)
            if not _array_equal(values, var.get_value(borrow=True)):
                raise ValueError(
                    f"This SharedPosterior already binds {name!r} to different draws. "
                    "Extract every graph from the same posterior, then call "
                    "set_posterior() to move all of them together."
                )
        return as_xtensor(self._variables[name], dims=dims)

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[None]:
        """Undo every registry change if the wrapped extraction raises.

        An extraction inserts variables as it walks the placeholders, so a
        refusal part-way would otherwise leave the instance holding some of the
        new graph's variables.  Restoring the copies makes the failure leave no
        trace, which keeps a later rebind that satisfies every graph already
        handed out acceptable.
        """
        snapshot = (self._variables.copy(), self._layouts.copy())
        try:
            yield
        except BaseException:
            self._variables, self._layouts = snapshot
            raise

    def set_posterior(self, idata: xr.DataTree | xr.Dataset) -> None:
        """Rebind every variable to the draws in ``idata``.

        Parameters
        ----------
        idata : xr.DataTree or xr.Dataset
            Inference data with a ``posterior`` group (or the posterior
            dataset itself) holding every variable this instance owns.

        Raises
        ------
        KeyError
            If the posterior lacks a variable this instance owns.
        ValueError
            If a variable's non-sample dimensions differ, in name, length or
            coordinate labels, from the ones it was created with.  Nothing is
            written in that case, and a write cut short by an interrupt puts
            the previous draws back.
        """
        # The input is read as ``az.extract`` reads it, so the narrowing below
        # accepts and refuses the same inputs.
        posterior = az.convert_to_dataset(idata, group="posterior")
        missing = sorted(set(self._variables) - set(posterior.data_vars))
        if missing:
            raise KeyError(
                f"Posterior is missing variables bound by this SharedPosterior: {missing}"
            )
        # Stack only the bound variables.  A posterior also carries every
        # deterministic (``channel_contribution``, ``mu`` and the rest), which
        # stacking would copy on every rebind of a lazily loaded posterior, and
        # a variable no graph reads may hold anything, even a layout that does
        # not stack.  The draws must still be indexed by ``chain`` and
        # ``draw``, as ``az.extract`` requires, even with nothing bound.
        bound = list(self._variables)
        narrowed = posterior[bound] if bound else posterior
        absent = [dim for dim in ("chain", "draw") if dim not in narrowed.dims]
        if absent:
            raise ValueError(
                f"Posterior draws must be indexed by 'chain' and 'draw'; {absent} "
                f"missing from dims {tuple(narrowed.dims)}."
            )
        if not bound:
            return
        posterior = _posterior_sample_major(narrowed)
        # Validate, align and cast everything first so a failure leaves the
        # binding untouched rather than half-swapped between two posteriors.
        # Every fallible step lives in _aligned_values; the write loop only
        # hands over arrays that already have the right dtype and shape.
        prepared = {
            name: self._aligned_values(name, posterior[name])
            for name in self._variables
        }
        # The arrays held now, by reference: a write cut short by an interrupt
        # puts them back, so no compiled function reads a mix of two posteriors.
        previous = {
            name: var.get_value(borrow=True) for name, var in self._variables.items()
        }
        try:
            for name, values in prepared.items():
                self._variables[name].set_value(values, borrow=True)
        except BaseException:
            for name, values in previous.items():
                self._variables[name].set_value(values, borrow=True)
            raise


@overload
def extract_response_distribution(
    pymc_model: Model,
    idata: xr.DataTree | xr.Dataset,
    response_variable: str | Variable,
    frozen_deterministics: list[str] | None = ...,
    *,
    shared_posterior: SharedPosterior | None = ...,
) -> Variable: ...


@overload
def extract_response_distribution(
    pymc_model: Model,
    idata: xr.DataTree | xr.Dataset,
    response_variable: Sequence[str | Variable],
    frozen_deterministics: list[str] | None = ...,
    *,
    shared_posterior: SharedPosterior | None = ...,
) -> list[Variable]: ...


def extract_response_distribution(
    pymc_model: Model,
    idata: xr.DataTree | xr.Dataset,
    response_variable: str | Variable | Sequence[str | Variable],
    frozen_deterministics: list[str] | None = None,
    *,
    shared_posterior: SharedPosterior | None = None,
) -> Variable | list[Variable]:
    """Extract the response distribution graph, conditioned on posterior parameters.

    Parameters
    ----------
    pymc_model : Model
        The PyMC model to extract the response distribution from.
    idata : xr.DataTree or xr.Dataset
        The inference data containing posterior samples.
        A tree's ``posterior`` group is used; a plain dataset is taken as the posterior itself.
    response_variable : str, Variable, or sequence of either
        The response variable to extract, by name or as the graph node itself.
        A node is accepted because not every interesting quantity is a named
        variable of the model: an MMM's linear predictor is only registered as a
        ``Deterministic`` under a log link, and is an anonymous intermediate
        otherwise.  A sequence extracts several variables in a single pass, so
        any subgraph they share is conditioned, rewritten and vectorized once
        and stays shared in the result -- which is what makes it cheap to
        evaluate, say, ``channel_contribution`` alongside a mediated effect that
        reads the same spend data.
    frozen_deterministics : list of str, optional
        Names of Deterministic variables to freeze at their posterior values instead of recomputing from the graph.
        Some models (e.g, those containing HSGP) need this to to obtain a valid conditional posterior graph.
    shared_posterior : SharedPosterior, optional
        Bind the posterior draws through this object's shared variables instead
        of baking them into the graph as constants.  Its
        :meth:`SharedPosterior.set_posterior` then swaps the draws under any
        function compiled from the result without recompiling.  Variables the
        object already holds are reused, so several graphs extracted with the
        same instance are rebound together.  An extraction that raises leaves
        the object exactly as it was.

    Returns
    -------
    pt.TensorVariable or list of pt.TensorVariable
        The response distribution graph.  A list -- matching
        ``response_variable`` element-wise -- when a sequence was passed, a
        single variable otherwise.

    Examples
    --------
    `extract_response_distribution(model, idata, "channel_contribution")`
    returns a graph that computes `"channel_contribution"` as a function of both
    the newly introduced budgets and the posterior of model parameters.
    """
    # Convert DataTree to a sample-major xarray
    posterior = _posterior_sample_major(idata)

    # A single name keeps the historical scalar return type; a sequence opts
    # into the list form.  Everything in between is list-shaped.
    single = isinstance(response_variable, str | Variable)
    requested = [response_variable] if single else list(response_variable)
    if not requested:
        raise ValueError("'response_variable' must name at least one variable.")

    # The PyMC variables to extract
    response_vars = [
        pymc_model[var] if isinstance(var, str) else var for var in requested
    ]

    # Identify which free RVs are needed to compute `response_var`.
    # Frozen deterministics are treated as additional blockers so their
    # subgraphs are not traversed — their posterior values are substituted
    # directly, just like free RVs.
    needed_names = needed_posterior_names(
        pymc_model, response_vars, frozen_deterministics or ()
    )
    placeholder_replace_dict = {
        pymc_model[name]: pymc_model[name].clone() for name in needed_names
    }

    response_vars = list(
        clone_replace(
            response_vars,
            replace=placeholder_replace_dict,
        )
    )

    if rvs_in_graph(response_vars):
        raise RuntimeError("RVs found in the extracted graph, this is likely a bug")

    # Cleanup graph.  rewrite_graph mirrors the shape of what it is given, so a
    # list of outputs comes back as a list.
    response_vars = list(
        cast(
            Sequence[Variable],
            rewrite_graph(response_vars, include=("canonicalize", "ShapeOpt")),
        )
    )

    # Replace placeholders with actual posterior samples: constants by default,
    # shared variables when the caller wants to rebind them later.  A shared
    # extraction that raises leaves the SharedPosterior as it was.
    transaction = (
        shared_posterior._transaction()
        if shared_posterior is not None
        else contextlib.nullcontext()
    )
    with transaction:
        replace_dict = {}
        # Bind in the order the posterior lists its variables.  A caller that
        # passes a construction-ordered subset (ConditionedModel) controls the
        # order in which shared variables are created.
        order = {name: index for index, name in enumerate(posterior.data_vars)}
        for placeholder in sorted(
            placeholder_replace_dict.values(),
            key=lambda var: order.get(var.name, len(order)),
        ):
            posterior_da = posterior[placeholder.name].astype(placeholder.dtype)
            if shared_posterior is not None:
                replace_dict[placeholder] = shared_posterior._get_or_create(
                    posterior_da, name=placeholder.name, dtype=placeholder.dtype
                )
            else:
                replace_dict[placeholder] = xtensor_constant(
                    posterior_da.values,
                    name=placeholder.name,
                    dims=posterior_da.dims,
                )

        # Vectorize across samples
        response_distribution = list(
            vectorize_graph(response_vars, replace=replace_dict)
        )

        # Final cleanup
        response_distribution = list(
            cast(
                Sequence[Variable],
                rewrite_graph(
                    response_distribution,
                    include=(
                        "useless",
                        "local_eager_useless_unbatched_blockwise",
                        "local_useless_unbatched_blockwise",
                    ),
                ),
            )
        )

    return response_distribution[0] if single else response_distribution


def _posterior_dataset(idata: xr.DataTree | xr.Dataset) -> xr.Dataset:
    """Return the ``posterior`` group of ``idata`` as a plain dataset.

    A tree may carry other groups (``constant_data``, ``observed_data``,
    ``prior``); conditioning only reads the posterior, so it is taken out once
    here and the rest of the object works with the dataset alone.  An
    ``arviz.InferenceData`` is not accepted; pass its ``posterior`` dataset or
    a tree holding it.

    Parameters
    ----------
    idata : xr.DataTree or xr.Dataset
        The posterior dataset itself, or a tree carrying it under
        ``posterior``.

    Returns
    -------
    xr.Dataset
        The posterior draws.

    Raises
    ------
    TypeError
        If ``idata`` is neither an ``xarray.Dataset`` nor an
        ``xarray.DataTree``.
    """
    if isinstance(idata, xr.Dataset):
        return idata
    if isinstance(idata, xr.DataTree):
        return idata["/posterior"].to_dataset()
    raise TypeError(
        "Expected the posterior as an xarray.Dataset or xarray.DataTree, "
        f"got {type(idata).__name__}."
    )


def _apply_intervention(
    model: Model,
    intervention: Mapping[str, Variable | np.ndarray | float] | None,
) -> Model:
    """Return a clone of ``model`` with ``intervention`` applied.

    Parameters
    ----------
    model : pm.Model
        The model to transform.  Never modified.
    intervention : mapping or None
        A mapping of :func:`pymc.do` arguments -- variable name to intervention
        value.  ``None`` clones the model unchanged.

    Returns
    -------
    pm.Model
        The intervened clone, whose data no later ``pm.set_data`` on ``model``
        reaches.
    """
    if intervention is None:
        return clone_model(model)
    with warnings.catch_warnings():
        # The ``scale`` intervention references the target on purpose:
        # ``target * mask`` is what holds the factual value in the graph.
        # pymc warns about the self-reference and registers the intervened node
        # under a ``do_`` name.
        warnings.filterwarnings(
            "ignore",
            message="Intervention expression references the variable "
            "that is being intervened",
        )
        return do(model, dict(intervention))


def _model_nodes(model: Model) -> frozenset[Variable]:
    """Return every node of ``model``: its named variables, its dim lengths and their ancestors."""
    return frozenset(
        ancestors([*model.named_vars.values(), *model.dim_lengths.values()])
    )


class ConditionedModel:
    """A model after an intervention, and the one owner of its posterior binding.

    A caller hands over a model, a posterior and, optionally, an intervention --
    a mapping of :func:`pymc.do` arguments, which this object applies itself.
    It hands out response graphs conditioned on the posterior and rebinds all
    of them when the posterior moves.

    Every graph reads the draws through one :class:`SharedPosterior`, so
    :meth:`set_posterior` points every graph handed out so far -- and every
    function compiled from one -- at the new draws without extracting or
    compiling again, whatever the number of draws.

    Extraction is cached by response names in a bounded LRU: a repeated
    :meth:`extract` of a cached key returns the identical graph, and a new
    derived node per call evicts the least recently used entry.  Eviction only
    drops the cache entry: an evicted graph still reads the shared variables
    and keeps following every rebind.  A new node per call costs an extraction
    and a cache entry every time, so a caller that derives one should keep the
    graph it gets back and reuse it.

    A posterior is reordered to the model's coords per dimension on the way in.
    A variable is validated when a graph first reads it, against its layout in
    the construction posterior, and on every rebind after that; a variable no
    graph reads is not validated.

    The model is always cloned, by :func:`pymc.do` when an intervention is
    given, so its data are fixed at construction: a later ``pm.set_data`` on
    the model does not reach any graph handed out here.  For the same reason a
    graph that reads a node of the source model, or a shared or random
    variable from outside the clone, is refused.

    Parameters
    ----------
    model : pm.Model
        The model to intervene on.  It is cloned, never modified.
    idata : xr.DataTree or xr.Dataset
        Posterior draws to condition on.  A tree's ``posterior`` group is used.
    intervention : mapping, optional
        A mapping of ``pm.do`` arguments.  ``None`` clones the model unchanged.
    frozen_deterministics : list of str, optional
        Names of Deterministic variables to freeze at their posterior values.

    Raises
    ------
    NotImplementedError, ValueError
        If PyMC cannot clone ``model``, as for a model with custom initial
        values or nested sub-models.

    Notes
    -----
    Cloning copies every input of the model, including a free symbolic input
    that is not a model variable, so a graph extracted here reads the clone's
    copy of it and a function compiled against the original input fails to
    compile.  Pass such an input through ``intervention`` instead:
    :func:`pymc.do` keeps the inputs of an intervention expression.

    Examples
    --------
    .. code-block:: python

        spend = {"channel_data": new_spend}
        conditioned = ConditionedModel(model, idata, intervention=spend)
        graph = conditioned.extract("channel_contribution")
        fn = pytensor.function([], graph.values)
        fn()  # under ``idata``
        conditioned.set_posterior(updated_idata)
        fn()  # under ``updated_idata``, no recompile
    """

    def __init__(
        self,
        model: Model,
        idata: xr.DataTree | xr.Dataset,
        *,
        intervention: Mapping[str, Variable | np.ndarray | float] | None = None,
        frozen_deterministics: list[str] | None = None,
    ) -> None:
        self.frozen_deterministics: tuple[str, ...] = tuple(frozen_deterministics or ())
        self._model = _apply_intervention(model, intervention)
        # Every node of the clone, so an input read from outside it can be told
        # from one of the clone's own inputs by identity.  The dim lengths are
        # walked too: a dim no variable is sized by is an ancestor of nothing.
        # Taken again when the clone gains a variable or a dim.
        self._clone_nodes = _model_nodes(self._model)
        self._clone_size = (len(self._model.named_vars), len(self._model.dim_lengths))
        # Every node of the source model, by identity.  The clone holds copies
        # of them, so a graph that reads one reads the source's live data and,
        # after an intervention, its pre-intervention values.
        self._source_nodes = _model_nodes(model)
        # The source's dim lengths, so a refusal names one as a dim.
        self._source_dim_lengths: dict[Variable, str] = {
            length: dim
            for dim, length in model.dim_lengths.items()
            if not isinstance(length, Constant)
        }
        self._posterior = reorder_to_model_coords(
            _posterior_dataset(idata), self._model.coords
        )
        # The layout each variable is validated against when a graph reads it,
        # taken once here.  It is metadata only; a variable no graph reads is
        # never checked, so a refit over a new date window is accepted.
        self._layouts: dict[str, VariableLayout] = {
            str(name): variable_layout(var, sample_dims=("chain", "draw"))
            for name, var in self._posterior.data_vars.items()
        }
        self._shared_posterior = SharedPosterior()
        self._cache: OrderedDict[tuple[str | Variable, ...], list[XTensorVariable]] = (
            OrderedDict()
        )

    @property
    def model(self) -> Model:
        """The clone every response is resolved against, intervened when an intervention was given."""
        return self._model

    @property
    def posterior(self) -> xr.Dataset:
        """The draws the graphs are currently conditioned on."""
        return self._posterior

    @property
    def shared_posterior(self) -> SharedPosterior:
        """The :class:`SharedPosterior` every graph reads its draws from."""
        return self._shared_posterior

    @overload
    def extract(self, response_vars: str | Variable) -> XTensorVariable: ...

    @overload
    def extract(self, response_vars: Sequence[Variable]) -> list[XTensorVariable]: ...

    def extract(
        self, response_vars: str | Variable | Sequence[str | Variable]
    ) -> XTensorVariable | list[XTensorVariable]:
        """Return response graphs conditioned on the current posterior.

        Every extraction is cached by response names: a repeated call returns
        the identical graph, and a full cache evicts the least recently used
        key.  Every graph reads its draws through :attr:`shared_posterior`, so
        it follows every later :meth:`set_posterior`, evicted or not, and every
        variable it reads must be in every later posterior.  A failure leaves
        the cache and the shared posterior as they were.

        Parameters
        ----------
        response_vars : str, Variable, or sequence of either
            The response variable(s) to extract.  A name resolves against the
            intervened model, a ``Variable`` may be any graph over that clone,
            and a sequence extracts all in one pass.  A graph that reads a node
            of the source model is refused, since that input is not the
            clone's: it holds the source's live data and no intervention.  So
            is a graph that reads a shared or random variable from outside the
            clone: a shared variable is read live, and a random variable is not
            conditioned on the posterior.  A free symbolic input the model does
            not bind is accepted as a function argument.

        Returns
        -------
        XTensorVariable or list of XTensorVariable
            A single graph when one variable was requested, a list otherwise,
            matching ``extract_response_distribution``.

        Raises
        ------
        KeyError
            If a requested name is not a variable of the intervened model, or
            if the current posterior lacks a variable the graph reads.
        ValueError
            If a ``Variable`` reads an input of the source model or a shared or
            random variable from outside the clone, if no
            response variable is given, or if a variable the graph
            reads has different non-sample dimensions, length or labels from
            its construction layout (or is labeled on one side only).
        """
        if isinstance(response_vars, str | Variable):
            single = True
            requested = [response_vars]
        else:
            single = False
            requested = list(response_vars)
        if not requested:
            raise ValueError("'response_vars' must name at least one variable.")

        resolved = [self._resolve(var) for var in requested]
        cache_key = self._names(resolved)
        graphs = self._cache.get(cache_key)
        if graphs is None:
            graphs = self._extract(resolved)
            self._cache[cache_key] = graphs
            # Eviction only drops the entry: an evicted graph still reads the
            # shared variables, which every rebind writes.
            while len(self._cache) > _GRAPH_CACHE_SIZE:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(cache_key)
        # Hand out a fresh list every call: the cached one is shared by every
        # consumer, so a caller mutating what it got back must not corrupt it.
        # The graph elements themselves stay identical.
        return graphs[0] if single else list(graphs)

    def set_posterior(self, idata: xr.DataTree | xr.Dataset) -> None:
        """Point every graph handed out so far at the draws in ``idata``.

        The draws are swapped in place in the shared variables, so a function
        compiled from any of the graphs follows without recompiling, and the
        number of draws may change.  Every variable a graph has read is checked
        against its construction layout before any is written: it must be
        present, with the same non-sample dimensions and the same set of labels
        on each, which are realigned by value.  A posterior variable no graph
        has read may change freely; it is checked when a graph first reads it.

        Parameters
        ----------
        idata : xr.DataTree or xr.Dataset
            A tree with a ``posterior`` group, or the posterior dataset itself;
            an ``arviz.InferenceData`` is not accepted.

        Raises
        ------
        KeyError
            If the posterior lacks a variable a graph reads.
        ValueError
            If a variable a graph reads differs from its construction layout in
            its non-sample dimensions, their lengths or their labels.

        Notes
        -----
        A rebind is atomic: after any error, an interrupt included, every graph
        reads the previous draws and :attr:`posterior` is unchanged.
        """
        posterior = reorder_to_model_coords(
            _posterior_dataset(idata), self._model.coords
        )
        previous = self._posterior
        self._posterior = posterior
        try:
            # Validates every bound variable and writes nothing unless all of
            # them pass; a write cut short puts the previous draws back.
            self._shared_posterior.set_posterior(posterior)
        except BaseException:
            self._posterior = previous
            raise

    def _extract(self, resolved: list[Variable]) -> list[XTensorVariable]:
        """Extract ``resolved`` conditioned on the variables they read.

        The posterior is narrowed to the read names, each validated and
        realigned against its own construction layout.  Only the variables a
        graph reads are checked: unread data may change freely.
        """
        read_names = needed_posterior_names(
            self._model, resolved, self.frozen_deterministics
        )
        # Every conditioned graph is an xtensor expression, so the sequence is safely narrowed.
        return cast(
            list[XTensorVariable],
            extract_response_distribution(
                pymc_model=self._model,
                idata=self._read_posterior(read_names, self._posterior),
                response_variable=resolved,
                frozen_deterministics=list(self.frozen_deterministics),
                shared_posterior=self._shared_posterior,
            ),
        )

    def _read_posterior(
        self, read_names: Sequence[str], posterior: xr.Dataset
    ) -> xr.Dataset:
        """Return the posterior subset the graphs read, in construction order.

        Each variable is validated and realigned against its construction
        layout here, at the read.  A variable absent from the construction
        posterior was thinned away, so its layout is unknown and it is taken as
        given.  The subset's order is the construction order, so a shared
        variable first created after a rebind is still created in that order.
        """
        if not read_names:
            return posterior
        missing = sorted(set(read_names) - set(posterior.data_vars))
        if missing:
            raise KeyError(
                f"Posterior is missing variables this graph reads: {missing}"
            )
        ordered = [name for name in self._layouts if name in read_names]
        ordered += [name for name in read_names if name not in self._layouts]
        selected: dict[str, xr.DataArray] = {}
        for name in ordered:
            posterior_da = posterior[name]
            layout = self._layouts.get(name)
            if layout is not None:
                posterior_da = aligned_posterior(
                    name,
                    posterior_da,
                    layout,
                    sample_dims=("chain", "draw"),
                )
            selected[name] = posterior_da
        return xr.Dataset(selected)

    def _resolve(self, var: str | Variable) -> Variable:
        """Resolve a response against the intervened clone.

        A node the clone registers, and any graph built over it, is accepted,
        and so is a free symbolic input, as a function argument.  A graph that
        reads a node of the source model is refused: the clone holds its own
        copy of every such input, so the graph would read the source's live
        data and carry no intervention.  A graph that reads a shared variable
        or a random variable from outside the clone is refused too: the shared
        variable is embedded and read live, so a later change to it reaches
        every compiled function, and the random variable is not conditioned on
        the posterior.  The source refusal comes first, since it says what to
        pass instead.  A refusal names those inputs, falling back to their type
        when they are anonymous.  Constants are never refused, a frozen model's
        data among them.
        """
        if isinstance(var, str):
            if var not in self._model.named_vars:
                raise KeyError(
                    f"Response variable {var!r} is not a variable of the "
                    f"intervened model, which has {sorted(self._model.named_vars)}."
                )
            return self._model[var]
        source_inputs, outside_inputs = self._outside_inputs(var)
        if source_inputs:
            described = self._describe_outside_inputs(
                source_inputs, dim_lengths=self._source_dim_lengths
            )
            raise ValueError(
                f"{var!r} reads inputs of the source model: {described}. Pass a "
                "name or a node of the model this object holds (its .model); an "
                "input of the source model is not the clone's, so it would "
                "carry neither the intervention nor the clone's data."
            )
        # Every outside input is a root or a random variable from outside both
        # models; only a free symbolic root is a function argument.
        foreign_inputs = [
            node
            for node in outside_inputs
            if isinstance(node, SharedVariable) or node.owner is not None
        ]
        if foreign_inputs:
            raise ValueError(
                f"{var!r} reads variables from outside the model this object "
                f"holds: {self._describe_outside_inputs(foreign_inputs)}. A "
                "shared variable is read live, so a later change to it "
                "(pm.set_data on the model that owns it, or set_value) would "
                "reach every function compiled from the graph, and a random "
                "variable outside the model is not conditioned on the posterior. "
                "Pass a name or a node of the model this object holds (its "
                ".model), and pass an outside value as an array or as a "
                "symbolic input."
            )
        return var

    def _outside_inputs(self, var: Variable) -> tuple[list[Variable], list[Variable]]:
        """Return the inputs ``var`` reads from the source model, and from elsewhere outside the clone.

        An input is a root or a random variable.  The walk stops at the clone's
        own nodes, at constants and at the source's inputs.  A random variable
        is an input in its own right, so the RNG and the dim lengths that size
        it are not listed.  Past an input from outside both models, such as a
        random variable built over the source, the walk goes on only to find
        the source's inputs behind it.  Both lists are in the order a
        depth-first walk meets them.
        """
        inside = self._current_clone_nodes()
        source_inputs: list[Variable] = []
        outside_inputs: list[Variable] = []
        # Each entry carries whether an outside input found there is listed:
        # past an input, only the source's inputs are.
        seen: set[tuple[Variable, bool]] = set()
        stack = [(var, True)]
        while stack:
            node, listed = stack.pop()
            if (node, listed) in seen or node in inside or isinstance(node, Constant):
                continue
            seen.add((node, listed))
            owner = node.owner
            is_input = owner is None or isinstance(owner.op, RNGConsumerOp)
            if is_input and node in self._source_nodes:
                if node not in source_inputs:
                    source_inputs.append(node)
                continue
            if is_input and listed:
                outside_inputs.append(node)
            if owner is not None:
                stack.extend(
                    (parent, listed and not is_input)
                    for parent in reversed(owner.inputs)
                )
        return source_inputs, outside_inputs

    def _current_clone_nodes(self) -> frozenset[Variable]:
        """Return every node of the clone, taken again if it gained a variable or a dim.

        A variable added to the clone after construction, and any graph built
        over it, is the clone's own, so it must not be taken for an outside
        input.
        """
        size = (len(self._model.named_vars), len(self._model.dim_lengths))
        if size != self._clone_size:
            self._clone_nodes, self._clone_size = _model_nodes(self._model), size
        return self._clone_nodes

    @staticmethod
    def _describe_outside_inputs(
        outside_inputs: list[Variable],
        dim_lengths: Mapping[Variable, str] | None = None,
    ) -> str:
        """List outside inputs by name, or by type when anonymous, at most five.

        A variable in ``dim_lengths`` is named as the length of its dim, so it
        does not read as a data variable.
        """
        dim_lengths = dim_lengths or {}
        named = [
            f"the length of dim {dim_lengths[node]!r}"
            if node in dim_lengths
            else repr(node.name)
            for node in outside_inputs
            if node in dim_lengths or node.name is not None
        ]
        # An anonymous root is named by its type, never by a repr that
        # may carry a memory address.
        labels = list(dict.fromkeys(named)) or list(
            dict.fromkeys(f"<{type(node).__name__}>" for node in outside_inputs)
        )
        listed = ", ".join(labels[:5])
        if len(labels) > 5:
            listed += f" and {len(labels) - 5} more"
        return listed

    def _names(self, resolved: list[Variable]) -> tuple[str | Variable, ...]:
        """Return the cache-key names of resolved responses.

        A variable keys by its name only when that name is the one the
        intervened model registers for it; any other node keys by the node
        itself, so a name two nodes happen to share cannot collide in the cache.
        """
        return tuple(
            var.name
            if var.name is not None and self._model.named_vars.get(var.name) is var
            else var
            for var in resolved
        )


class ModelSamplerEstimator:
    """Estimate computational characteristics of a PyMC model using JAX/NumPyro.

    This utility measures the average evaluation time of the model's logp and gradients
    and estimates the number of integrator steps taken by NUTS during warmup + sampling.
    It then compiles the information into a single-row pandas DataFrame with helpful
    metadata to guide planning and benchmarking.

    Parameters
    ----------
    tune : int, default 1000
        Number of warmup iterations to use when estimating NUTS steps.
    draws : int, default 1000
        Number of sampling iterations to use when estimating NUTS steps.
    chains : int, default 1
        Intended number of chains (metadata only; not used in JAX runs here).
    sequential_chains : int, default 1
        Number of chains expected to run sequentially on the target environment.
        Used to scale the wall-clock time estimate.
    seed : int | None, default None
        Random seed used for the step estimation runs.

    Examples
    --------
    .. code-block:: python

        est = ModelSamplerEstimator(
            tune=1000, draws=1000, chains=4, sequential_chains=1, seed=1
        )
        df = est.run(model)
        print(df)
    """

    def __init__(
        self,
        *,
        tune: int = 1000,
        draws: int = 1000,
        chains: int = 1,
        sequential_chains: int = 1,
        seed: int | None = None,
    ) -> None:
        self.tune = int(tune)
        self.draws = int(draws)
        self.chains = int(chains)
        self.sequential_chains = int(sequential_chains)
        self.seed = seed

    @property
    def default_nuts_kwargs(self) -> dict:
        """Default keyword arguments for a NumPyro NUTS kernel.

        Mirrors the current hard-coded defaults used in this estimator.
        """
        return {
            "target_accept_prob": 0.8,
            "adapt_step_size": True,
            "adapt_mass_matrix": True,
            "dense_mass": False,
        }

    @property
    def default_mcmc_kwargs(self) -> dict:
        """Default keyword arguments for a NumPyro MCMC runner.

        Parameters that depend on the run size (``num_warmup`` and ``num_samples``)
        are intentionally excluded and provided explicitly by the estimator.
        """
        return {
            "num_chains": 1,
            "postprocess_fn": None,
            "chain_method": "sequential",
            "progress_bar": False,
        }

    def estimate_model_eval_time(
        self, model: Model, num_evaluations: int | None = None
    ) -> float:
        """Estimate average evaluation time (seconds) of logp+dlogp using JAX.

        Parameters
        ----------
        model : Model
            PyMC model whose logp and gradients are jitted and evaluated.
        num_evaluations : int | None, optional
            Number of repeated evaluations to average over. If ``None``, a value
            is chosen to take roughly 5 seconds in total for a stable estimate.

        Returns
        -------
        float
            Average evaluation time in seconds.
        """
        from time import perf_counter_ns

        import numpy as np

        try:
            import jax
            from pymc.sampling.jax import get_jaxified_logp
        except Exception as err:  # pragma: no cover - environment specific
            raise RuntimeError(
                "JAX backend is required for ModelSamplerEstimator."
            ) from err

        initial_point = list(model.initial_point().values())
        logp_fn = get_jaxified_logp(model)
        logp_dlogp_fn = jax.jit(jax.value_and_grad(logp_fn, argnums=0))
        logp_res, grad_res = logp_dlogp_fn(initial_point)
        for val in (logp_res, *grad_res):
            if not np.isfinite(val).all():
                raise RuntimeError(
                    "logp or gradients are not finite at the model initial point; the model may be misspecified"
                )

        if num_evaluations is None:
            start = perf_counter_ns()
            jax.block_until_ready(logp_dlogp_fn(initial_point))
            end = perf_counter_ns()
            num_evaluations = max(5, int(5e9 / max(end - start, 1)))

        start = perf_counter_ns()
        for _ in range(num_evaluations):
            jax.block_until_ready(logp_dlogp_fn(initial_point))
        end = perf_counter_ns()
        eval_time = (end - start) / num_evaluations * 1e-9
        return float(eval_time)

    def estimate_num_steps_sampling(
        self,
        model: Model,
        *,
        tune: int | None = None,
        draws: int | None = None,
        seed: int | None = None,
        nuts_kwargs: dict | None = None,
        mcmc_kwargs: dict | None = None,
    ) -> int:
        """Estimate total number of NUTS steps during warmup + sampling using NumPyro.

        Parameters
        ----------
        model : Model
            PyMC model to estimate steps for using a JAX/NumPyro NUTS kernel.
        tune : int | None, optional
            Warmup iterations. Defaults to the estimator setting if ``None``.
        draws : int | None, optional
            Sampling iterations. Defaults to the estimator setting if ``None``.
        seed : int | None, optional
            Random seed for the JAX run. Defaults to the estimator setting if ``None``.
        nuts_kwargs : dict | None, optional
            Additional keyword arguments passed to ``numpyro.infer.NUTS``. If not provided,
            the estimator's ``default_nuts_kwargs`` are used. Provided values override
            the defaults.
        mcmc_kwargs : dict | None, optional
            Additional keyword arguments passed to ``numpyro.infer.MCMC`` (excluding
            ``num_warmup`` and ``num_samples``, which are set by ``tune``/``draws``). If not
            provided, the estimator's ``default_mcmc_kwargs`` are used. Provided values
            override the defaults.

        Returns
        -------
        int
            Total number of leapfrog steps across warmup + sampling.
        """
        import numpy as np

        try:
            import jax
            from numpyro.infer import MCMC, NUTS
            from pymc.sampling.jax import get_jaxified_logp
        except Exception as err:  # pragma: no cover - environment specific
            raise RuntimeError(
                "JAX and NumPyro are required for ModelSamplerEstimator."
            ) from err

        num_warmup = int(self.tune if tune is None else tune)
        num_samples = int(self.draws if draws is None else draws)

        initial_point = list(model.initial_point().values())
        logp_fn = get_jaxified_logp(model, negative_logp=False)
        merged_nuts_kwargs = {**self.default_nuts_kwargs, **(nuts_kwargs or {})}
        nuts_kernel = NUTS(
            potential_fn=logp_fn,
            **merged_nuts_kwargs,
        )

        merged_mcmc_kwargs = {**self.default_mcmc_kwargs, **(mcmc_kwargs or {})}
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            **merged_mcmc_kwargs,
        )

        if seed is None:
            rng_seed = int(np.random.default_rng().integers(2**32))
        else:
            rng_seed = int(seed)

        tune_rng, sample_rng = jax.random.split(jax.random.PRNGKey(int(rng_seed)), 2)
        mcmc.warmup(
            tune_rng,
            init_params=initial_point,
            extra_fields=("num_steps",),
            collect_warmup=True,
        )
        warmup_steps = int(mcmc.get_extra_fields()["num_steps"].sum())
        mcmc.run(sample_rng, extra_fields=("num_steps",))
        sample_steps = int(mcmc.get_extra_fields()["num_steps"].sum())
        return int(warmup_steps + sample_steps)

    def run(self, model: Model) -> pd.DataFrame:
        """Execute the estimation pipeline and return a single-row DataFrame.

        Parameters
        ----------
        model : Model
            PyMC model to evaluate.

        Returns
        -------
        pandas.DataFrame
            Single-row DataFrame with columns including ``num_steps``, ``eval_time_seconds``,
            ``sequential_chains``, and estimated sampling wall-clock time in seconds,
            minutes, and hours, along with metadata such as ``tune``, ``draws``, ``chains``,
            ``seed``, ``timestamp``, and ``model_name``.

        Examples
        --------
        .. code-block:: python

            df = ModelSamplerEstimator().run(model)
            df[
                [
                    "num_steps",
                    "eval_time_seconds",
                    "estimated_sampling_time_minutes",
                ]
            ]
        """
        import time

        steps = self.estimate_num_steps_sampling(
            model, tune=self.tune, draws=self.draws, seed=self.seed
        )
        eval_time_s = self.estimate_model_eval_time(model)

        sampling_time_seconds = float(
            eval_time_s * steps * max(self.sequential_chains, 1)
        )
        data = {
            "model_name": getattr(model, "name", "PyMCModel"),
            "num_steps": int(steps),
            "eval_time_seconds": float(eval_time_s),
            "sequential_chains": int(self.sequential_chains),
            "estimated_sampling_time_seconds": sampling_time_seconds,
            "estimated_sampling_time_minutes": sampling_time_seconds / 60.0,
            "estimated_sampling_time_hours": sampling_time_seconds / 3600.0,
            "tune": int(self.tune),
            "draws": int(self.draws),
            "chains": int(self.chains),
            "seed": int(self.seed) if self.seed is not None else None,
            "timestamp": pd.Timestamp.utcfromtimestamp(int(time.time())),
        }
        return pd.DataFrame([data])
