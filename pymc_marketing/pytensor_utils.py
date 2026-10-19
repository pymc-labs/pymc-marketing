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
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import NamedTuple, cast, overload

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pymc import do
from pymc.model.core import Model
from pymc.model.fgraph import (
    ModelValuedVar,
    ModelVar,
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
from pytensor.xtensor import xtensor_constant
from pytensor.xtensor.type import XTensorVariable, as_xtensor
from pytensor.xtensor.vectorization import vectorize_graph


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
        self._dims: dict[str, tuple[str, ...]] = {}
        self._coords: dict[str, dict[str, pd.Index]] = {}

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
        return dict(self._dims)

    def _aligned_values(self, name: str, posterior_da: xr.DataArray) -> np.ndarray:
        """Return ``posterior_da`` laid out and typed like the bound variable ``name``.

        Raises ``ValueError`` on any mismatch with the contract in the class
        docstring; writes nothing.  The returned array is a fresh allocation
        the shared variable can own outright, which keeps the write phase of
        :meth:`set_posterior` infallible.
        """
        dims = self._dims[name]
        if set(posterior_da.dims) != set(dims):
            raise ValueError(
                f"Posterior variable {name!r} has dims {tuple(posterior_da.dims)}, "
                f"expected {dims}."
            )
        for dim in dims[1:]:
            bound = self._coords[name].get(dim)
            given = posterior_da.indexes.get(dim)
            if (bound is None) != (given is None):
                raise ValueError(
                    f"Posterior variable {name!r}: dimension {dim!r} is labelled on "
                    "one side only. Label both sides or neither; positions cannot "
                    "be checked against labels."
                )
            if bound is None or given is None:
                continue
            # pd.Index compares datetimes by value across storage units.  The
            # bound index is unique (checked at bind time), so equal lengths
            # plus an empty difference means the same set of labels, with any
            # duplicate on the given side surfacing as a missing label.  No
            # sorting, so a mixed-type object axis is fine.
            if len(bound) != len(given) or not bound.difference(given).empty:
                extra, missing = given.difference(bound), bound.difference(given)
                raise ValueError(
                    f"Posterior variable {name!r} has different {dim} labels: "
                    f"{len(extra)} not bound, {len(missing)} missing "
                    f"(e.g. {list(extra[:3])} / {list(missing[:3])})."
                )
            posterior_da = posterior_da.sel({dim: bound})
        var = self._variables[name]
        # astype always allocates, so the shared variable owns the array and
        # borrow=True cannot alias the caller's data.
        values = posterior_da.transpose(*dims).values.astype(var.type.dtype)
        bound_shape = var.get_value(borrow=True).shape[1:]
        if values.shape[1:] != bound_shape:
            raise ValueError(
                f"Posterior variable {name!r} has shape {values.shape[1:]} over "
                f"dims {dims[1:]}, expected {bound_shape}. Only the sample "
                "dimension may change length between rebinds."
            )
        return values

    def _get_or_create(self, posterior_da: xr.DataArray, *, name: str, dtype: str):
        """Return the xtensor view of ``name``'s shared variable, creating it if needed.

        On reuse the draws handed in must be the ones already bound: the graph
        will read the shared variable, so a different posterior here would be
        silently ignored.  A new variable is validated before it is inserted,
        so a refusal leaves the three registries untouched.
        """
        if name not in self._variables:
            dims = tuple(str(dim) for dim in posterior_da.dims)
            coords = {
                dim: posterior_da.indexes[dim]
                for dim in dims[1:]
                if dim in posterior_da.indexes
            }
            # Labels are a set: a duplicated label could not be realigned by
            # ``sel`` later, which would silently pick one column twice.
            for dim, index in coords.items():
                if not index.is_unique:
                    raise ValueError(
                        f"Posterior variable {name!r} has duplicated {dim} labels: "
                        f"{index[index.duplicated()].unique().tolist()}."
                    )
            self._variables[name] = shared(posterior_da.values.astype(dtype), name=name)
            self._dims[name] = dims
            self._coords[name] = coords
        else:
            var = self._variables[name]
            if np.dtype(dtype) != np.dtype(var.type.dtype):
                raise ValueError(
                    f"Posterior variable {name!r} is bound as {var.type.dtype}, "
                    f"but this graph wants {dtype}."
                )
            values = self._aligned_values(name, posterior_da)
            if not _array_equal(values, var.get_value(borrow=True)):
                raise ValueError(
                    f"This SharedPosterior already binds {name!r} to different draws. "
                    "Extract every graph from the same posterior, then call "
                    "set_posterior() to move all of them together."
                )
        return as_xtensor(self._variables[name], dims=self._dims[name])

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[None]:
        """Undo every registry change if the wrapped extraction raises.

        An extraction inserts variables as it walks the placeholders, so a
        refusal part-way would otherwise leave the instance holding some of the
        new graph's variables.  Restoring the copies makes the failure leave no
        trace, which keeps a later rebind that satisfies every graph already
        handed out acceptable.
        """
        snapshot = (self._variables.copy(), self._dims.copy(), self._coords.copy())
        try:
            yield
        except BaseException:
            self._variables, self._dims, self._coords = snapshot
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
            written in that case.
        """
        posterior = _posterior_sample_major(idata)
        missing = sorted(set(self._variables) - set(posterior.data_vars))
        if missing:
            raise KeyError(
                f"Posterior is missing variables bound by this SharedPosterior: {missing}"
            )
        # Validate, align and cast everything first so a failure leaves the
        # binding untouched rather than half-swapped between two posteriors.
        # Every fallible step lives in _aligned_values; the write loop only
        # hands over arrays that already have the right dtype and shape.
        prepared = {
            name: self._aligned_values(name, posterior[name])
            for name in self._variables
        }
        for name, values in prepared.items():
            self._variables[name].set_value(values, borrow=True)


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
    free_rvs = set(pymc_model.free_RVs)
    frozen_vars: set = set()
    if frozen_deterministics:
        for name in frozen_deterministics:
            if name in pymc_model.named_vars:
                frozen_vars.add(pymc_model[name])

    blockers = free_rvs | frozen_vars
    needed_rvs = [
        rv for rv in ancestors(response_vars, blockers=blockers) if rv in blockers
    ]
    placeholder_replace_dict = {pymc_model[rv.name]: rv.clone() for rv in needed_rvs}

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
        for placeholder in placeholder_replace_dict.values():
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
    """Return ``model`` with ``intervention`` applied, or ``model`` unchanged.

    Parameters
    ----------
    model : pm.Model
        The model to transform.  Never modified: ``do`` clones it.
    intervention : mapping or None
        A mapping of :func:`pymc.do` arguments -- variable name to intervention
        value.  ``None`` leaves the model as it is.

    Returns
    -------
    pm.Model
        The intervened model.
    """
    if intervention is None:
        return model
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


class _ExtractKey(NamedTuple):
    """Cache key of a conditioned extraction."""

    names: tuple[str | Variable, ...]
    frozen_deterministics: tuple[str, ...]


class _CachedGraphs(NamedTuple):
    """A cached extraction: the resolved inputs and the graphs handed out."""

    resolved: list[Variable]
    graphs: list[XTensorVariable]


class ConditionedModel:
    """A model after an intervention, and the one owner of its posterior binding.

    A caller hands over a model, a posterior and, optionally, an intervention --
    a mapping of :func:`pymc.do` arguments, which this object applies itself.
    It then hands out response graphs conditioned on the posterior and keeps
    rebinding them as the posterior moves, so the "intervene, condition, extract,
    compile" dance is owned here instead of by each consumer.

    Extraction is cached by ``(response names, frozen deterministics)``.
    Repeated :meth:`extract` of the same key returns the identical graph.
    By default the draws enter as constants until the first :meth:`set_posterior`, keeping constant folding.
    With ``share_posterior=True`` they enter through a :class:`SharedPosterior` from the first :meth:`extract`.
    No rebind then re-extracts; even the first only swaps draws.

    A model built without ``share_posterior`` starts with the draws baked in as constants.
    Once a graph has been cached, its first :meth:`set_posterior` drops the cache.
    It then re-extracts each key through a new :class:`SharedPosterior`.
    That rebind fires the registered callbacks so consumers recompile.
    A :meth:`set_posterior` with nothing cached yet still checks the labels.
    It then only switches to the shared path and fires nothing.
    An uncached ``cache=False`` graph is untouched by it.
    The posterior given at construction defines the label layout every rebind is realigned to.
    Every later rebind only swaps the draws in place, which is atomic and free in the number of draws.
    A compiled function therefore follows the newer posterior without being recompiled.

    A response is resolved against the intervened clone.
    A :class:`~pytensor.graph.basic.Variable` may be any graph built over that clone, including a derived expression.
    A graph that reads an input from outside the intervened model is refused.
    That covers a node of the pre-intervention model as well as one from an unrelated model.
    Such a graph would silently carry no intervention into the result.
    With ``intervention=None`` the clone is the model itself, so a node of the source model is accepted.

    Parameters
    ----------
    model : pm.Model
        The model to intervene on.
    idata : xr.DataTree or xr.Dataset
        Posterior draws to condition on.  A tree's ``posterior`` group is used.
    intervention : mapping, optional
        A mapping of ``pm.do`` arguments.  ``None`` uses the model unchanged.
    frozen_deterministics : list of str, optional
        Names of Deterministic variables to freeze at their posterior values
        instead of recomputing from the graph.
    share_posterior : bool, default False
        Start on the shared path: bind the draws through a :class:`SharedPosterior` from the first extraction.
        No rebind then re-extracts, and the first :meth:`set_posterior` only swaps draws.

    Examples
    --------
    .. code-block:: python

        conditioned = ConditionedModel(
            model, idata, intervention={"channel_data": new_spend}
        )
        graph = conditioned.extract("channel_contribution")
        conditioned.register(lambda: recompile(graph))
        conditioned.set_posterior(updated_idata)  # recompiles once
        conditioned.set_posterior(another_idata)  # no recompile
    """

    def __init__(
        self,
        model: Model,
        idata: xr.DataTree | xr.Dataset,
        *,
        intervention: Mapping[str, Variable | np.ndarray | float] | None = None,
        frozen_deterministics: list[str] | None = None,
        share_posterior: bool = False,
    ) -> None:
        self.frozen_deterministics: tuple[str, ...] = tuple(frozen_deterministics or ())
        self._model = _apply_intervention(model, intervention)
        # Every node of the clone, so an input read from outside it can be told
        # from one of the clone's own inputs by identity.
        self._clone_nodes = frozenset(ancestors(list(self._model.named_vars.values())))
        self._posterior = _posterior_dataset(idata)
        # The label layout every rebind is realigned to, taken once here.
        # A thinned posterior can drop a dimension, so it cannot be read from the last stored posterior.
        self._label_layout: dict[str, pd.Index] = {
            dim: index
            for dim, index in self._posterior.indexes.items()
            if dim not in ("chain", "draw")
        }
        self._shared_posterior: SharedPosterior | None = (
            SharedPosterior() if share_posterior else None
        )
        self._cache: dict[_ExtractKey, _CachedGraphs] = {}
        self._callbacks: list[Callable[[], None]] = []

    @property
    def model(self) -> Model:
        """The intervened clone every response is resolved against."""
        return self._model

    @property
    def posterior(self) -> xr.Dataset:
        """The draws the graphs are currently conditioned on."""
        return self._posterior

    @property
    def shared_posterior(self) -> SharedPosterior | None:
        """The shared posterior binding the graphs, or ``None`` while the draws are constant.

        Returns
        -------
        SharedPosterior or None
            ``None`` while the draws are baked into the graphs as constants.
            Once it exists, it is the :class:`SharedPosterior` every rebind swaps draws in.
            A model built with ``share_posterior=True`` has it from construction.
        """
        return self._shared_posterior

    @overload
    def extract(
        self,
        response_vars: str | Variable,
        *,
        cache: bool = ...,
    ) -> XTensorVariable: ...

    @overload
    def extract(
        self,
        response_vars: Sequence[Variable],
        *,
        cache: bool = ...,
    ) -> list[XTensorVariable]: ...

    def extract(
        self,
        response_vars: str | Variable | Sequence[str | Variable],
        *,
        cache: bool = True,
    ) -> XTensorVariable | list[XTensorVariable]:
        """Return response graphs conditioned on the current posterior.

        A failure leaves the cache and the shared posterior as they were.
        A later rebind that satisfies every graph already handed out is therefore accepted.

        Parameters
        ----------
        response_vars : str, Variable, or sequence of either
            The response variable(s) to extract.
            A name resolves against the intervened model.
            A ``Variable`` may be any graph built over that clone, including a derived expression over its nodes.
            A graph that reads an input from outside the intervened model is refused.
            A sequence extracts every variable in one pass, so a subgraph they share stays shared in the result.
        cache : bool, default True
            Whether to keep the extraction for :meth:`set_posterior` to re-extract through the shared posterior.
            A cached graph is the long-lived kind: after the first rebind it reads shared variables.
            It therefore follows every later rebind.
            Its variables must be present in every posterior handed to :meth:`set_posterior`.
            A caller that passes ``cache=False`` gets a graph conditioned on the current draws as constants instead.
            It registers nothing on the shared posterior.
            A later rebind to a posterior without those variables therefore still succeeds.
            A caller that builds a fresh derived node per call should pass ``cache=False``.
            Nothing tracks that graph across a rebind, so a consumer re-extracts it after a rebind.

        Returns
        -------
        XTensorVariable or list of XTensorVariable
            A single graph when one variable was requested, a list otherwise,
            matching ``extract_response_distribution``.
            The single-graph overload is typed ``XTensorVariable``.
            That holds for a name and for any xtensor expression.

        Raises
        ------
        KeyError
            If a requested name is not a variable of the intervened model.
        ValueError
            If a ``Variable`` reads an input from outside the intervened model, or if no response variable is given.
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
        if not cache:
            # A one-shot graph is conditioned on constants, never on the shared
            # posterior, so it registers no variable a later rebind would need.
            graphs = self._extract(resolved, None, self._posterior)
            return graphs[0] if single else list(graphs)
        cache_key = _ExtractKey(self._names(resolved), self.frozen_deterministics)
        if cache_key not in self._cache:
            graphs = self._extract(resolved, self._shared_posterior, self._posterior)
            self._cache[cache_key] = _CachedGraphs(resolved, graphs)
        graphs = self._cache[cache_key].graphs
        # Hand out a fresh list every call: the cached one is shared by every
        # consumer, so a caller mutating what it got back must not corrupt it.
        # The graph elements themselves stay identical.
        return graphs[0] if single else list(graphs)

    def set_posterior(self, idata: xr.DataTree | xr.Dataset) -> None:
        """Point the handed-out graphs at the draws in ``idata``.

        The posterior given at construction defines the label layout every rebind aligns to.
        ``idata`` is realigned to it once, when it arrives, before any graph is bound.
        Without ``share_posterior``, the first call after a graph has been cached rebuilds those keys.
        It rebuilds them on a shared posterior.
        That call is validated, label realigned and length checked exactly like every later one.
        It re-extracted the graphs, so it fires the registered callbacks and consumers recompile from them.
        A call with nothing cached still checks the labels and switches to the shared path.
        There is no cached graph to rebuild, so it fires no callback.
        An uncached ``cache=False`` graph is untouched by it.
        Every later call only swaps the draws, which needs no recompile and accepts a different number of draws.

        Parameters
        ----------
        idata : xr.DataTree or xr.Dataset
            A tree with a ``posterior`` group, or the posterior dataset itself;
            an ``arviz.InferenceData`` is not accepted.  It must hold every
            variable the graphs read, with the same non-sample dimensions and
            coordinate labels; a reordered axis is realigned and the number of
            draws is free.

        Raises
        ------
        KeyError
            If the posterior lacks a variable the graphs read.
        ValueError
            If a variable's non-sample dimensions, lengths or labels differ from the ones it was bound with.
            Also if a dimension is labelled on one side only.

        Notes
        -----
        A call before anything was cached still checks ``idata``'s labels at the door.
        It binds no variable in that case.
        After either error, and when a registered callback raises, the previous binding is left in place.

        A callback that extracts a new key during a rebind adds it to the cache.
        If a first rebind fails and is rolled back, the cache returns to its previous state and drops those entries.
        """
        posterior = _posterior_dataset(idata)
        rollback = self._apply_posterior(self._realigned(posterior))
        if rollback is None:
            # Either nothing is cached yet, so only the mode switched.
            # Or this was a later rebind that only swapped the draws.
            # Both leave no cached graph for a consumer to rebuild, so no callback fires.
            return
        try:
            self._fire_callbacks()
        except BaseException:
            rollback()
            raise

    def _realigned(self, posterior: xr.Dataset) -> xr.Dataset:
        """Return ``posterior`` laid out along the labels of the construction posterior.

        A dimension absent from ``posterior`` is skipped, so a thinned posterior that drops a variable is accepted.
        A dimension present without labels, while the reference draws have them, is refused.
        Positions cannot be checked against labels, so both sides must agree or both go without.
        Reordering needs unique reference labels; a duplicated one would make ``sel`` drop a column.
        """
        # The construction layout is the reference, even when a thinned posterior dropped a dimension.
        # A dimension unlabelled at construction keeps the old fallback.
        # It is checked against the replaced draws, so the one-sided check still fires after a rebind introduced labels.
        references = {**self._posterior.indexes, **self._label_layout}
        indexers: dict[str, pd.Index] = {}
        for dim, bound in references.items():
            if dim in ("chain", "draw"):
                continue
            if dim not in posterior.sizes:
                continue
            given = posterior.indexes.get(dim)
            if given is None:
                raise ValueError(
                    f"Posterior dimension {dim!r} is labelled on one side only. "
                    "Label both sides or neither; positions cannot be checked "
                    "against labels."
                )
            if given.equals(bound):
                continue
            if not bound.is_unique:
                raise ValueError(
                    f"Posterior has duplicated {dim} labels in the construction "
                    f"posterior: {bound[bound.duplicated()].unique().tolist()}."
                )
            if len(given) != len(bound) or not bound.difference(given).empty:
                raise ValueError(
                    f"Posterior has different {dim} labels from the construction posterior."
                )
            indexers[dim] = bound
        return posterior.sel(indexers) if indexers else posterior

    def _apply_posterior(self, posterior: xr.Dataset) -> Callable[[], None] | None:
        """Bind the graphs to ``posterior``, returning a rollback when that re-extracted the graphs.

        See :meth:`set_posterior` for the rebind and rollback contract.
        """
        if not self._cache:
            # Nothing is cached yet, so there is no cached graph to re-extract.
            # ``set_posterior`` already realigned and checked the labels at the door.
            # Uncached ``cache=False`` graphs are untouched by this.
            # Only switch to the shared path and record the new draws.
            # An instance built with ``share_posterior=True`` is kept, not replaced.
            if self._shared_posterior is None:
                self._shared_posterior = SharedPosterior()
            self._posterior = posterior
            return None

        if self._shared_posterior is not None:
            # ``SharedPosterior.set_posterior`` validates every variable before writing any.
            # A failure therefore leaves the previous draws bound, with nothing to roll back.
            self._shared_posterior.set_posterior(posterior)
            self._posterior = posterior
            return None

        previous_cache = self._cache
        previous_posterior = self._posterior
        shared, cache = self._extract_through_shared(previous_posterior)
        shared.set_posterior(posterior)
        self._shared_posterior = shared
        self._posterior = posterior
        self._cache = cache

        def rollback() -> None:
            self._shared_posterior = None
            self._posterior = previous_posterior
            self._cache = previous_cache

        return rollback

    def register(self, callback: Callable[[], None]) -> None:
        """Register ``callback`` to run after a rebind that re-extracted the graphs.

        That is the first :meth:`set_posterior` of a model built without ``share_posterior``.
        It only re-extracts once a graph has been cached.
        A rebind with nothing cached only switches to the shared path, and later rebinds only swap the draws.
        Neither fires a callback, and uncached ``cache=False`` graphs are untouched.
        A consumer recompiles what it compiled from the graphs here.
        The callback takes no arguments and calls :meth:`extract` to read the refreshed graphs.
        Callbacks run in registration order, and one that raises stops the rest.
        This object rolls back only its own binding; each consumer is responsible for rolling back its own state.

        Parameters
        ----------
        callback : callable
            A no-argument callable run after a rebind that re-extracted the graphs, in registration order.
        """
        self._callbacks.append(callback)

    def _extract(
        self,
        resolved: list[Variable],
        shared_posterior: SharedPosterior | None,
        posterior: xr.Dataset,
    ) -> list[XTensorVariable]:
        """Extract ``resolved`` conditioned on ``posterior``, in a single pass."""
        # Every conditioned graph is an xtensor expression, so the sequence is safely narrowed.
        return cast(
            list[XTensorVariable],
            extract_response_distribution(
                pymc_model=self._model,
                idata=posterior,
                response_variable=resolved,
                frozen_deterministics=list(self.frozen_deterministics),
                shared_posterior=shared_posterior,
            ),
        )

    def _extract_through_shared(
        self, posterior: xr.Dataset
    ) -> tuple[SharedPosterior, dict[_ExtractKey, _CachedGraphs]]:
        """Re-extract every handed-out key through one shared posterior.

        Returns the shared posterior and a fresh cache, committing nothing
        until the caller installs them, so a failure part-way leaves the
        constant graphs in place.
        """
        shared = SharedPosterior()
        cache = {
            key: _CachedGraphs(resolved, self._extract(resolved, shared, posterior))
            for key, (resolved, _) in self._cache.items()
        }
        return shared, cache

    def _resolve(self, var: str | Variable) -> Variable:
        """Resolve a response against the intervened clone, refusing a graph that reads an outsider.

        A node the clone registers, and any graph built over it, is accepted.
        A graph whose leaves include an input from outside the clone is refused.
        A node of the pre-intervention model, or of an unrelated one, is such an input.
        The refusal names those inputs, falling back to their type when they are anonymous.
        With ``intervention=None`` the clone is the model itself, so a node of the source model is accepted.
        Constants are never treated as an outside input.
        """
        if isinstance(var, str):
            if var not in self._model.named_vars:
                raise KeyError(
                    f"Response variable {var!r} is not a variable of the "
                    f"intervened model, which has {sorted(self._model.named_vars)}."
                )
            return self._model[var]
        outside_inputs = [
            node
            for node in ancestors([var])
            if node.owner is None
            and not isinstance(node, Constant)
            and node not in self._clone_nodes
        ]
        if outside_inputs:
            raise ValueError(
                f"{var!r} reads inputs outside the intervened model: "
                f"{self._describe_outside_inputs(outside_inputs)}. Pass a "
                "name or a node of the model this object holds (its .model); "
                "a node of the pre-intervention model is refused because it "
                "would carry no intervention into the graph."
            )
        return var

    @staticmethod
    def _describe_outside_inputs(outside_inputs: list[Variable]) -> str:
        """List outside inputs by name, or by type when anonymous, at most five."""
        named = [repr(node.name) for node in outside_inputs if node.name is not None]
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

    def _fire_callbacks(self) -> None:
        """Run every registered callback in order."""
        for callback in self._callbacks:
            callback()


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
