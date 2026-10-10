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
"""Bind selected fitted terms to deterministic, labeled optimization inputs."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from functools import cache
from numbers import Integral
from typing import Any, cast

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import xarray as xr
from pymc.pytensorf import rvs_in_graph
from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant, Variable, clone_get_equiv
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting import rewrite_graph
from pytensor.graph.traversal import ancestors, truncated_graph_inputs
from pytensor.tensor.basic import Nonzero
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.shape import Shape, Shape_i
from pytensor.tensor.type_other import MakeSlice
from pytensor.xtensor.basic import Rename, TensorFromXTensor, XTensorFromTensor
from pytensor.xtensor.indexing import Index
from pytensor.xtensor.reduction import XReduce
from pytensor.xtensor.shape import Transpose
from pytensor.xtensor.type import XTensorSharedVariable, XTensorVariable, as_xtensor
from pytensor.xtensor.vectorization import XBlockwise, XElemwise

from pymc_marketing.mmm.components.adstock import (
    BinomialAdstock,
    DelayedAdstock,
    GeometricAdstock,
    WeibullCDFAdstock,
    WeibullPDFAdstock,
)
from pymc_marketing.mmm.experimental._data import (
    _align_labels,
    _dates,
    validate_dataset,
)
from pymc_marketing.mmm.experimental._gam import GAM
from pymc_marketing.mmm.experimental._graph import (
    BuildContext,
    Data,
    Equation,
    walk,
)
from pymc_marketing.mmm.experimental._terms import MediaTransform
from pymc_marketing.pytensor_utils import (
    _posterior_sample_major,
    extract_response_distribution,
)
from pymc_marketing.terms import (
    Dot,
    ModelTerm,
    Parameter,
    Ref,
)

_LOWER = ("lower_xtensor", "canonicalize", "stabilize")


def _input_provenance(
    context: BuildContext, declared: Collection[str]
) -> dict[Variable, str]:
    """Keep authoritative container ownership before inferring declared shared aliases."""
    model = context.model
    result = {model[internal]: raw for raw, internal in context.data_variables.items()}
    for variable, raw in context._registered_inputs.items():
        result.setdefault(variable, raw)
    for raw in declared:
        if raw in model.named_vars and isinstance(model[raw], XTensorSharedVariable):
            result.setdefault(model[raw], raw)
    return result


def _fitted_inputs(gam: GAM) -> dict[Variable, str]:
    """Recover fitted raw input provenance, including shared-term data aliases."""
    return _input_provenance(
        cast(BuildContext, gam._context),
        _data_dependencies({"equations": gam.equations}),
    )


def _fitted_graphs(terms: Mapping[str, Any], gam: GAM) -> list[Variable]:
    """Resolve fitted identities and explicit references without rebuilding equations."""
    context = cast(BuildContext, gam._context)
    result = []
    for node in walk(
        terms, stop=lambda node: id(node) in context.variables or isinstance(node, Ref)
    ):
        if id(node) in context.variables:
            value = context.variables[id(node)]
        elif isinstance(node, Ref):
            if node.name not in context.model.named_vars:
                raise ValueError(
                    f"Ref({node.name!r}) must reference a variable in the fitted model."
                )
            value = context.model[node.name]
        else:
            continue
        if isinstance(value, Variable):
            result.append(value)
    return result


def _data_dependencies(terms: Mapping[str, Any], gam: GAM | None = None) -> set[str]:
    """Find every declared input and, when fitted, inputs hidden behind named references."""
    context = None
    if gam is not None:
        gam._check_fitted()
        context = cast(BuildContext, gam._context)
    required: set[str] = set()
    for node in walk(
        terms,
        stop=lambda node: (
            context is not None
            and (id(node) in context.variables or isinstance(node, Ref))
        ),
    ):
        # A fitted boundary reads its actual graph, not the prior or lifecycle recipe.
        if context is not None and (
            id(node) in context.variables or isinstance(node, Ref)
        ):
            continue
        if isinstance(node, Data):
            required.add(node.var_name)
        elif isinstance(node, ModelTerm):
            required.update(node.data_vars)
    if gam is not None and context is not None:
        inputs = _fitted_inputs(gam)
        blockers = set(context.model.free_RVs) | set(context.model.observed_RVs)
        required.update(
            inputs[node]
            for node in ancestors(_fitted_graphs(terms, gam), blockers=blockers)
            if node in inputs
        )
    return required


def _scenario_data(
    gam: GAM, required: set[str], data: xr.Dataset
) -> tuple[xr.Dataset, dict[str, xr.DataArray]]:
    """Align scenario inputs, retaining fitted dimension and nondate label order."""
    if not isinstance(data, xr.Dataset):
        raise TypeError("Optimizer data must be an xarray.Dataset.")
    missing = required.difference(data.data_vars)
    if missing:
        raise ValueError(f"Required optimizer data are missing: {sorted(missing)}.")
    data = gam._prediction_data(
        xr.Dataset({name: data[name] for name in sorted(required)}, coords=data.coords)
    )
    training = cast(xr.Dataset, gam._training_data)
    templates = {}
    for name in sorted(required):
        value = data[name]
        dims = (
            tuple(training[name].dims)
            if name in training
            else tuple(sorted(value.dims))
        )
        if set(value.dims) != set(dims):
            raise ValueError(
                f"Input {name!r} must have fitted dimensions {dims}, got {value.dims}."
            )
        if "sample" in dims:
            raise ValueError(
                "Input dimension 'sample' is reserved for posterior draws."
            )
        templates[name] = BuildContext._canonical_input(value.transpose(*dims), name)
    return xr.Dataset(templates, coords=data.coords), templates


def _prepend_history(
    scenario: xr.Dataset,
    templates: Mapping[str, xr.DataArray],
    history: xr.Dataset | None,
) -> tuple[xr.Dataset, dict[str, XTensorVariable], int]:
    """Prepend explicitly supplied, fixed history to date-indexed inputs only."""
    if history is None:
        return scenario, {}, 0
    if not isinstance(history, xr.Dataset):
        raise TypeError("History must be an xarray.Dataset.")
    dated = [name for name, value in templates.items() if "date" in value.dims]
    if not dated:
        return scenario, {}, 0
    missing = set(dated).difference(history.data_vars)
    if missing:
        raise ValueError(f"History is missing date-indexed inputs: {sorted(missing)}.")
    history = validate_dataset(history[dated])
    if "date" not in history.dims:
        raise ValueError("History for date-indexed inputs must contain labeled dates.")
    if history.get_index("date")[-1] >= scenario.get_index("date")[0]:
        raise ValueError("History dates must strictly precede scenario dates.")
    arrays = dict(templates)
    fixed = {}
    for name in dated:
        template = templates[name]
        value = history[name]
        if set(value.dims) != set(template.dims):
            raise ValueError(
                f"History input {name!r} must have dimensions {template.dims}."
            )
        value = _align_labels(
            value, template, dims=[dim for dim in template.dims if dim != "date"]
        ).transpose(*template.dims)
        # Own the historical values so caller mutations cannot change the bound history.
        value = BuildContext._canonical_input(value, name, copy=True)
        fixed[name] = as_xtensor(value)
        arrays[name] = xr.concat(
            [value, template],
            dim="date",
            coords="minimal",
            compat="override",
            join="exact",
        )
    coords = dict(scenario.coords)
    coords["date"] = np.concatenate(
        [history.coords["date"].values, scenario.coords["date"].values]
    )
    return xr.Dataset(arrays, coords=coords), fixed, history.sizes["date"]


def _dated_constant_replacements(
    context: BuildContext,
    built: Mapping[str, XTensorVariable],
    blockers: set[Variable],
) -> dict[Variable, Variable]:
    """Validate selected deterministic constant labels, selecting covered scenario dates."""
    selected = set(ancestors(list(built.values()), blockers=blockers))
    needed = {
        variable: value
        for variable, value in context._labeled_constants.items()
        if variable in selected
    }
    if not needed:
        return {}
    reference = xr.Dataset(
        coords=_coordinates(
            {
                dim: pd.Index(labels, name=dim, tupleize_cols=False)
                for dim, labels in context.model.coords.items()
                if labels is not None
            }
        )
    )
    replacements = {}
    for variable, value in needed.items():
        for dim in value.dims:
            if dim not in value.coords or value.coords[dim].dims != (dim,):
                raise ValueError(
                    f"Deterministic constant dimension {dim!r} must label its own dimension."
                )
        value = _align_labels(
            value, reference, dims=[dim for dim in value.dims if dim != "date"]
        )
        if "date" in value.dims:
            replacements[variable] = as_xtensor(context._select_constant_dates(value))
    return replacements


def _constant_index_value(index: Variable) -> Any:
    """Read index constants through only the known conversion and Boolean nonzero ops."""
    if isinstance(index, Constant):
        return index.data
    owner = index.owner
    if owner is None:
        return None
    if isinstance(owner.op, MakeSlice) and all(
        isinstance(item, Constant) for item in owner.inputs
    ):
        return slice(*(cast(Constant, item).data for item in owner.inputs))
    if isinstance(owner.op, (XTensorFromTensor, TensorFromXTensor)):
        return _constant_index_value(owner.inputs[0])
    if isinstance(owner.op, Nonzero):
        value = _constant_index_value(owner.inputs[0])
        if value is not None:
            value = np.asarray(value)
            if value.ndim == 1 and value.dtype.kind == "b":
                return np.flatnonzero(value)
    return None


def _full_date_index(index: Variable, length: int) -> bool:
    """Prove identity using the index's Boolean, integer, or slice semantics."""
    value = _constant_index_value(index)
    if isinstance(value, slice):
        return value.indices(length) == (0, length, 1)
    if value is None:
        return False
    value = np.asarray(value)
    if value.shape != (length,):
        return False
    if value.dtype.kind == "b":
        return bool(value.all())
    positions = np.arange(length)
    if value.dtype.kind == "i":
        return bool(np.all((value == positions) | (value == positions - length)))
    if value.dtype.kind == "u":
        return np.array_equal(value, positions)
    return False


def _fixed_shape_proof(boundaries: Collection[Variable]) -> Callable[[Variable], bool]:
    """Share the conservative shape-only exemption across both history proofs."""

    @cache
    def has_fixed_shape(value: Variable) -> bool:
        if value in boundaries:
            return True
        owner = value.owner
        if owner is not None and isinstance(owner.op, Index):
            original, *indexers = owner.inputs
            return has_fixed_shape(original) and all(
                _constant_index_value(index) is not None for index in indexers
            )
        shape = getattr(value.type, "shape", None)
        if shape is not None and all(size is not None for size in shape):
            return True
        if value.owner is None or not isinstance(
            value.owner.op,
            (
                DeepCopyOp,
                ViewOp,
                Elemwise,
                DimShuffle,
                XElemwise,
                XReduce,
                Rename,
                Transpose,
                TensorFromXTensor,
                XTensorFromTensor,
            ),
        ):
            return False
        return all(has_fixed_shape(item) for item in value.owner.inputs)

    return has_fixed_shape


def _check_history_dates(
    gam: GAM,
    terms: Mapping[str, Any],
    context: BuildContext,
    built: Mapping[str, XTensorVariable],
    inputs: Mapping[Variable, str],
    templates: Mapping[str, xr.DataArray],
    dated_constants: Mapping[Variable, Variable],
) -> None:
    """Prove full combined date provenance before removing a positional history prefix."""
    fitted_model = cast(BuildContext, gam._context).model
    blockers = set(fitted_model.free_RVs) | set(fitted_model.observed_RVs)
    sources = {
        value
        for value, raw in inputs.items()
        if raw in templates and "date" in templates[raw].dims
    }
    sources.update(
        rv
        for rv in fitted_model.free_RVs
        if "date"
        in getattr(rv, "dims", fitted_model.named_vars_to_dims.get(rv.name, ()))
    )
    sources.update(dated_constants)
    fourier_outputs = {
        value
        for _, term in context._fourier_terms.values()
        if isinstance(value := context.variables.get(id(term)), XTensorVariable)
    }
    sources.update(fourier_outputs)
    adstock_inputs: dict[XTensorVariable, XTensorVariable] = {}
    for node in walk([terms, gam.equations]):
        if not isinstance(node, MediaTransform) or type(node.transformation) not in (
            BinomialAdstock,
            DelayedAdstock,
            GeometricAdstock,
            WeibullCDFAdstock,
            WeibullPDFAdstock,
        ):
            continue
        value = context.variables.get(id(node))
        media = context.variables.get(id(node.media), node.media)
        if isinstance(value, XTensorVariable) and isinstance(media, XTensorVariable):
            # Built-in adstock retains the input's date axis after its internal padding/convolution.
            adstock_inputs[value] = media

    has_fixed_shape = _fixed_shape_proof(set(inputs) | blockers | fourier_outputs)

    @cache
    def depends_on_dates(value: Variable) -> bool:
        if value in sources:
            return True
        owner = value.owner
        if value in blockers or owner is None:
            return False
        # Match the temporal-history proof: only value-independent shapes cut dated ancestry.
        if isinstance(owner.op, (Shape, Shape_i)) and has_fixed_shape(owner.inputs[0]):
            return False
        return any(depends_on_dates(item) for item in owner.inputs)

    @cache
    def preserves_dates(value: Variable) -> bool:
        if value in sources:
            return True
        if not isinstance(value, XTensorVariable) or "date" not in value.dims:
            return False
        if value in adstock_inputs:
            return preserves_dates(adstock_inputs[value])
        owner = value.owner
        if owner is None:
            return False
        dated = [
            item
            for item in owner.inputs
            if isinstance(item, XTensorVariable) and "date" in item.dims
        ]
        if not dated:
            return False
        op = owner.op
        if isinstance(op, Index):
            original, *indexers = owner.inputs
            original = cast(XTensorVariable, original)
            for dim, index in zip(original.dims, indexers, strict=False):
                if dim == "date":
                    if not _full_date_index(index, context.ds.sizes["date"]):
                        return False
                elif "date" in getattr(index, "dims", ()) or depends_on_dates(index):
                    return False
            return preserves_dates(original)
        if isinstance(op, XReduce):
            safe = "date" not in op.dims
        elif isinstance(op, Rename):
            original = cast(XTensorVariable, owner.inputs[0])
            safe = original.dims.index("date") == value.dims.index("date")
        elif isinstance(op, XBlockwise):
            safe = all("date" not in dims for group in op.core_dims for dims in group)
        else:
            safe = isinstance(op, (XElemwise, Transpose, ViewOp, DeepCopyOp))
        return (
            safe
            and all(preserves_dates(item) for item in dated)
            and all(
                not depends_on_dates(item)
                for item in owner.inputs
                if not isinstance(item, XTensorVariable) or "date" not in item.dims
            )
        )

    for name, output in built.items():
        if "date" not in output.dims:
            if depends_on_dates(output):
                raise ValueError(
                    f"Selected term {name!r} reduces date before history can be removed; "
                    "select a date-indexed term and reduce scenario dates in the callback instead."
                )
        elif not preserves_dates(output):
            raise ValueError(
                f"Selected term {name!r} must retain the full combined date axis in its original order "
                "without a dependency that reduces date before history can be removed; "
                "perform date slicing and scoring inside the callback instead."
            )


def _check_history_inputs(
    gam: GAM,
    terms: Mapping[str, Any],
    context: BuildContext,
    built: Mapping[str, XTensorVariable],
    templates: Mapping[str, xr.DataArray],
    dated_constants: Mapping[Variable, Variable],
) -> None:
    """Validate date provenance, immutable temporal history, and fitted cadence."""
    fitted = cast(BuildContext, gam._context)
    blockers = set(fitted.model.free_RVs) | set(fitted.model.observed_RVs)
    selected = set(ancestors(list(built.values()), blockers=blockers))
    inputs = _fitted_inputs(gam)
    inputs.update(_input_provenance(context, templates))

    fourier_outputs = {
        value
        for _, term in context._fourier_terms.values()
        if isinstance(value := context.variables.get(id(term)), XTensorVariable)
    }
    has_fixed_shape = _fixed_shape_proof(set(inputs) | blockers | fourier_outputs)

    # A fixed input shape is harmless; a data-dependent cardinality is not.
    value_blockers = blockers | {
        value
        for value in selected
        if value.owner is not None
        and isinstance(value.owner.op, (Shape, Shape_i))
        and has_fixed_shape(value.owner.inputs[0])
    }
    _check_history_dates(gam, terms, context, built, inputs, templates, dated_constants)
    dated_inputs = {
        value
        for value, raw in inputs.items()
        if raw in templates and "date" in templates[raw].dims
    }

    @cache
    def depends_on_dated_decisions(value: Variable) -> bool:
        if value in dated_inputs:
            return True
        if value in value_blockers or value.owner is None:
            return False
        return any(depends_on_dated_decisions(item) for item in value.owner.inputs)

    temporal = False
    for node in walk([terms, gam.equations]):
        value = context.variables.get(id(node))
        if not isinstance(value, Variable) or value not in selected:
            continue
        lookback = getattr(node, "required_history", 0)
        if (
            isinstance(lookback, bool)
            or not isinstance(lookback, Integral)
            or lookback < 0
        ):
            raise ValueError("Term required_history must be a nonnegative integer.")
        if not lookback:
            continue
        temporal = True
        upstream_nodes = set(ancestors([value], blockers=value_blockers))
        upstream = {
            inputs[ancestor] for ancestor in upstream_nodes if ancestor in inputs
        }
        static = sorted(
            raw
            for raw in upstream
            if raw in templates and "date" not in templates[raw].dims
        )
        if static:
            raise ValueError(
                f"Static inputs {static} occur upstream of a history-dependent term; "
                "scenario decisions must not change measured history. "
                "Place static decision inputs downstream of temporal terms instead."
            )
        if any(
            isinstance(ancestor, XTensorVariable)
            and "date" not in ancestor.dims
            and depends_on_dated_decisions(ancestor)
            for ancestor in upstream_nodes
        ):
            raise ValueError(
                "A decision-derived intermediate reduces date upstream of a history-dependent term; "
                "scenario decisions must not change measured history. "
                "Place date reductions downstream of temporal terms or score them inside the callback instead."
            )
    if not temporal:
        return
    training = cast(xr.Dataset, gam._training_data)
    dates = _dates(training.coords["date"].values)
    if len(dates) < 2:
        raise ValueError(
            "Temporal history requires at least two training dates to infer the fitted cadence."
        )
    frequency = pd.infer_freq(dates) if len(dates) >= 3 else dates[1] - dates[0]
    if frequency is None:
        raise ValueError("Temporal history requires regularly spaced training dates.")
    combined = _dates(context.ds.coords["date"].values).as_unit("us")
    expected = pd.date_range(
        combined[0], periods=len(combined), freq=frequency
    ).as_unit("us")
    if not combined.equals(expected):
        raise ValueError(
            "Temporal history and scenario dates must be consecutive at the fitted cadence."
        )


def _freeze_posterior_shapes(
    gam: GAM,
    context: BuildContext,
    built: dict[str, XTensorVariable],
) -> dict[str, XTensorVariable]:
    """Freeze proven shape metadata before adding a posterior sample axis."""
    fitted = cast(BuildContext, gam._context)
    free = set(fitted.model.free_RVs)
    blockers = free | set(fitted.model.observed_RVs)
    nodes = set(ancestors(list(built.values()), blockers=blockers))
    inputs = _fitted_inputs(gam)
    inputs.update(_input_provenance(context, context.data_variables))
    fourier = {
        value
        for _, term in context._fourier_terms.values()
        if isinstance(value := context.variables.get(id(term)), XTensorVariable)
    }
    fixed_shape = _fixed_shape_proof(set(inputs) | blockers | fourier)
    shapes = [
        node
        for node in nodes
        if node.owner is not None
        and isinstance(node.owner.op, (Shape, Shape_i))
        and fixed_shape(node.owner.inputs[0])
        and free.intersection(ancestors([node], blockers=blockers))
    ]
    if not shapes:
        return built
    shape_nodes = set(ancestors(shapes, blockers=blockers))
    replacements: dict[Variable, Variable] = {
        original: context.model[context.data_variables[raw]]
        for original, raw in inputs.items()
        if raw in context.data_variables
    }
    # These are value-free shape stand-ins, never substitute posterior values or priors.
    for value in shape_nodes.intersection(free | fourier):
        dims = getattr(
            value, "dims", fitted.model.named_vars_to_dims.get(value.name, ())
        )
        sizes = []
        for dim in dims:
            if dim in context.ds.sizes:
                sizes.append(context.ds.sizes[dim])
            else:
                labels = context.model.coords.get(dim)
                if labels is None:
                    raise ValueError(
                        f"Posterior shape metadata requires labeled coordinates for {dim!r}."
                    )
                sizes.append(len(labels))
        replacements[value] = as_xtensor(pt.zeros(sizes, dtype=value.dtype), dims=dims)
    rewritten = clone_replace(shapes, replace=replacements, rebuild_strict=False)
    values = pytensor.function([], rewritten)()
    outputs = list(built.values())
    boundaries = {
        **{
            node: pt.as_tensor_variable(value)
            for node, value in zip(shapes, values, strict=True)
        },
        **{rv: rv for rv in blockers},
    }
    memo = clone_get_equiv(
        list(truncated_graph_inputs(outputs, boundaries)),
        outputs,
        copy_inputs=False,
        copy_orphans=False,
        memo=boundaries,
        strict=False,
    )
    return {name: cast(XTensorVariable, memo[value]) for name, value in built.items()}


def _condition_outputs(
    gam: GAM,
    built: dict[str, XTensorVariable],
    data: xr.Dataset,
    shape_context: BuildContext,
) -> tuple[dict[str, XTensorVariable], pd.Index | None]:
    """Condition actual fitted RV identities, rejecting stochastic outcomes and foreign RVs."""
    context = cast(BuildContext, gam._context)
    model = context.model
    free = set(model.free_RVs)
    blockers = free | set(model.observed_RVs)
    nodes = set(ancestors(list(built.values()), blockers=blockers))
    equations = {model[name] for name in context.equation_names.values()}
    if nodes.intersection(equations):
        raise ValueError(
            "Optimizer targets and dependencies must be deterministic terms, not stochastic Equation outputs."
        )
    placeholders = clone_replace(
        list(built.values()), replace={rv: rv.clone() for rv in free}
    )
    if rvs_in_graph(placeholders):
        raise ValueError(
            "Selected terms contain stochastic variables not bound to this fitted model; "
            "use its actual fitted parameter objects, not new parameters with matching names."
        )
    built = _freeze_posterior_shapes(gam, shape_context, built)
    nodes = set(ancestors(list(built.values()), blockers=blockers))
    needed = nodes.intersection(free)
    if not needed:
        return built, None
    posterior = cast(xr.DataTree, gam.idata)["posterior"].to_dataset()
    indexers = {}
    for rv in needed:
        if rv.name not in posterior:
            raise ValueError(f"The fitted posterior is missing parameter {rv.name!r}.")
        value = posterior[rv.name]
        dims = getattr(rv, "dims", model.named_vars_to_dims.get(rv.name, ()))
        for dim in dims:
            if dim == "sample":
                raise ValueError(
                    "Parameter dimension 'sample' is reserved for posterior draws."
                )
            if dim not in data.coords:
                raise ValueError(
                    f"Posterior parameter {rv.name!r} requires scenario labels for {dim!r}."
                )
            labels = data.get_index(dim)
            given = value.get_index(dim)
            if dim == "date":
                matches = given.is_unique and labels.isin(given).all()
            else:
                matches = (
                    given.is_unique
                    and len(labels) == len(given)
                    and labels.isin(given).all()
                )
            if not matches:
                raise ValueError(
                    f"Posterior parameter {rv.name!r} has no fitted draws for the requested {dim!r} labels."
                )
            indexers[dim] = data.coords[dim]
    # This is a labeled, read-only view of the actual joint draws, not an assignment to the GAM.
    posterior = posterior[sorted(cast(str, rv.name) for rv in needed)].sel(indexers)
    idata = xr.DataTree.from_dict({"posterior": posterior})
    sampled = [
        name
        for name, value in built.items()
        if free.intersection(ancestors([value], blockers=free))
    ]
    conditioned = extract_response_distribution(
        model, idata, [built[name] for name in sampled]
    )
    result = dict(built)
    result.update(zip(sampled, cast(list[XTensorVariable], conditioned), strict=True))
    return result, _posterior_sample_major(idata).get_index("sample")


def _coordinates(coords: Mapping[str, pd.Index]) -> xr.Coordinates:
    """Construct labeled coordinates without implicit MultiIndex promotion."""
    result = xr.Coordinates(
        {
            dim: labels
            for dim, labels in coords.items()
            if not isinstance(labels, pd.MultiIndex)
        }
    )
    for dim, labels in coords.items():
        if isinstance(labels, pd.MultiIndex):
            result.update(xr.Coordinates.from_pandas_multiindex(labels, dim))
    return result


class _Evaluator:
    """Evaluate selected deterministic terms on original-unit scenario decisions."""

    def __init__(
        self,
        outputs: dict[str, XTensorVariable],
        containers: dict[str, XTensorSharedVariable],
        templates: dict[str, xr.DataArray],
        coords: dict[str, pd.Index],
        history: dict[str, XTensorVariable],
    ) -> None:
        self.outputs = outputs
        self.containers = containers
        self.templates = templates
        self.coords = coords
        self._history = history
        self._numeric: Callable[..., list[np.ndarray]] | None = None

    def __call__(self, u: Mapping[str, XTensorVariable]) -> dict[str, XTensorVariable]:
        """Substitute each input once, prepending fixed history before model computation."""
        missing = set(self.containers).difference(u)
        extra = set(u).difference(self.containers)
        if missing or extra:
            raise ValueError(
                f"evaluate requires exactly {sorted(self.containers)}; "
                f"missing {sorted(missing)}, extra {sorted(extra)}."
            )
        replacements = {}
        for name, container in self.containers.items():
            value = u[name]
            if not isinstance(value, XTensorVariable):
                raise TypeError(f"Input {name!r} must be an XTensorVariable.")
            if set(value.dims) != set(container.dims):
                raise ValueError(
                    f"Input {name!r} must have dimensions {container.dims}, got {value.dims}."
                )
            value = value.transpose(*container.dims)
            if name in self._history:
                value = ptx.concat([self._history[name], value], dim="date")
            replacements[container] = value
        values = clone_replace(
            list(self.outputs.values()), replace=replacements, rebuild_strict=False
        )
        return dict(zip(self.outputs, cast(list[XTensorVariable], values), strict=True))

    def constant(self, value: xr.DataArray) -> XTensorVariable:
        """Align a labeled original-unit constant to the evaluator's coordinates.

        Parameters
        ----------
        value : xarray.DataArray
            Scalar or labeled array over a subset of the scenario and posterior dimensions.

        Returns
        -------
        XTensorVariable
            Constant with each dimension's labels in evaluator order.
        """
        if not isinstance(value, xr.DataArray):
            raise TypeError("constant expects a labeled xarray.DataArray.")
        for dim in value.dims:
            if dim not in value.coords or value.coords[dim].dims != (dim,):
                raise ValueError(
                    f"Constant dimension {dim!r} must label its own dimension."
                )
        reference = xr.Dataset(coords=_coordinates(self.coords))
        return as_xtensor(_align_labels(value, reference).astype(float))

    def evaluate(self, data: xr.Dataset) -> xr.Dataset:
        """Numerically evaluate the bound scenario, for example to check posterior fidelity.

        Parameters
        ----------
        data : xarray.Dataset
            Selected inputs with the bound scenario's labels, in any dimension or label order.
            Unrelated variables are ignored.

        Returns
        -------
        xarray.Dataset
            Selected outputs in original units. Only posterior-dependent outputs carry ``sample``.
        """
        if not isinstance(data, xr.Dataset):
            raise TypeError("Evaluator data must be an xarray.Dataset.")
        missing = set(self.templates).difference(data.data_vars)
        if missing:
            raise ValueError(f"Required evaluator data are missing: {sorted(missing)}.")
        data = validate_dataset(data[sorted(self.templates)])
        arrays = []
        for name, template in self.templates.items():
            value = data[name]
            if set(value.dims) != set(template.dims):
                raise ValueError(
                    f"Input {name!r} must have dimensions {template.dims}."
                )
            value = BuildContext._canonical_input(value, name)
            arrays.append(
                _align_labels(value, template).transpose(*template.dims).values
            )
        if self._numeric is None:
            tensors = {
                name: pt.tensor(name=name, dtype="float64", shape=template.shape)
                for name, template in self.templates.items()
            }
            values = self(
                {
                    name: as_xtensor(tensors[name], dims=template.dims)
                    for name, template in self.templates.items()
                }
            )
            outputs = rewrite_graph(
                [value.values for value in values.values()], include=_LOWER
            )
            self._numeric = pytensor.function(
                list(tensors.values()),
                outputs,
                on_unused_input="ignore",
            )
        numeric = self._numeric(*arrays)
        return xr.Dataset(
            {
                name: xr.DataArray(
                    value,
                    dims=output.dims,
                    coords=_coordinates({dim: self.coords[dim] for dim in output.dims}),
                )
                for (name, output), value in zip(
                    self.outputs.items(), numeric, strict=True
                )
            }
        )


def _bind_terms(
    gam: GAM,
    terms: Mapping[str, Any],
    data: xr.Dataset,
    history: xr.Dataset | None = None,
) -> _Evaluator:
    """Reuse fitted term provenance, freeze the joint posterior, and bind only selected inputs."""
    gam._check_fitted()
    fitted = cast(BuildContext, gam._context)
    admitted = tuple(walk(terms, stop=lambda node: id(node) in fitted.variables))
    if any(isinstance(node, Equation) for node in admitted):
        raise ValueError(
            "Optimizer targets and dependencies must be deterministic terms, not stochastic Equations."
        )
    required = _data_dependencies(terms, gam)
    scenario, templates = _scenario_data(gam, required, data)
    combined, fixed, history_length = _prepend_history(scenario, templates, history)
    for node in admitted:
        if isinstance(node, (Parameter, Dot)) and id(node) not in fitted.variables:
            raise ValueError(
                "Selected terms contain a stochastic parameter not bound to this fitted model; "
                "use its actual fitted parameter objects, not new parameters with matching names."
            )
    with pm.Model(
        coords={
            dim: coord.values
            for dim, coord in combined.coords.items()
            if dim in combined.dims
        }
    ) as model:
        context = BuildContext(
            combined,
            bindings=gam._bindings,
            prediction=True,
            allow_constant_date_subset=True,
        )
        # Reuse the fitted deterministic graph and its actual random-variable identities.
        context.variables.update(fitted.variables)
        for node in admitted:
            if isinstance(node, Ref):
                context.variables[id(node)] = fitted.model[node.name]
        context._labeled_constants.update(fitted._labeled_constants)
        context._suggested_names.update(fitted._suggested_names)
        context._fourier_terms.update(fitted._fourier_terms)
        context._names.update(fitted._names)
        containers = {
            name: cast(XTensorSharedVariable, context.data(name)) for name in templates
        }
        built = {name: as_xtensor(context.build(term)) for name, term in terms.items()}
    blockers = set(fitted.model.free_RVs) | set(fitted.model.observed_RVs)
    selected = set(ancestors(list(built.values()), blockers=blockers))
    legacy_inputs = sorted(
        f"{raw!r} ({getattr(value, 'dtype', None)})"
        for value, raw in _fitted_inputs(gam).items()
        if value in selected and getattr(value, "dtype", None) != "float64"
    )
    if legacy_inputs:
        raise ValueError(
            f"Selected fitted inputs {legacy_inputs} do not have canonical float64 graph dtype. "
            "Rebuild and refit the model, or save/reload it and verify that the reloaded fitted input "
            "graphs use float64 before optimizing."
        )
    dated_constants = _dated_constant_replacements(context, built, blockers)
    if history_length:
        _check_history_inputs(gam, terms, context, built, templates, dated_constants)
    if dated_constants:
        # Rebind before posterior graph rewrites can fold or replace a recorded leaf.
        # Stop at actual RV identities; clone_replace would clone their owned graphs again.
        constant_outputs = list(built.values())
        dated_replacements = {**dated_constants, **{rv: rv for rv in blockers}}
        memo = clone_get_equiv(
            list(truncated_graph_inputs(constant_outputs, dated_replacements)),
            constant_outputs,
            copy_inputs=False,
            copy_orphans=False,
            memo=dated_replacements,
            strict=False,
        )
        built = {
            name: cast(XTensorVariable, memo[value]) for name, value in built.items()
        }
        # Shape proofs must follow the Fourier output that actually survives rebinding.
        for _, term in context._fourier_terms.values():
            key = id(term)
            value = context.variables.get(key)
            if isinstance(value, XTensorVariable) and value in memo:
                context.variables[key] = memo[value]
    built, samples = _condition_outputs(gam, built, combined, context)
    replacements: dict[Variable, Variable] = {
        original: containers[raw]
        for original, raw in _fitted_inputs(gam).items()
        if raw in containers
    }
    replacements.update(
        {
            original: containers[raw]
            for original, raw in _input_provenance(context, templates).items()
        }
    )
    for dim, length in fitted.model.dim_lengths.items():
        if dim in combined.dims:
            replacements[length] = pt.as_tensor_variable(np.int64(combined.sizes[dim]))
    graph_nodes = set(ancestors(list(built.values())))
    for fourier, _ in fitted._fourier_terms.values():
        name = f"_{fourier.prefix}_dayofperiod"
        original = fitted.model[name]
        if original in graph_nodes:
            if "date" not in combined.dims:
                raise ValueError(
                    "Selected Fourier terms require scenario date coordinates."
                )
            days = fourier._get_days_in_period(_dates(combined.coords["date"].values))
            replacements[original] = as_xtensor(days.to_numpy(), dims=("date",))
    values = clone_replace(
        list(built.values()), replace=replacements, rebuild_strict=False
    )
    outputs = dict(zip(built, cast(list[XTensorVariable], values), strict=True))
    if history_length:
        outputs = {
            name: output.isel(date=slice(history_length, None))
            if "date" in output.dims
            else output
            for name, output in outputs.items()
        }
    if rvs_in_graph(list(outputs.values())):
        raise ValueError(
            "Selected outputs still contain stochastic variables after posterior conditioning."
        )
    allowed = set(containers.values())
    unresolved = {
        node.name
        for node in ancestors(list(outputs.values()))
        if isinstance(node, SharedVariable) and node not in allowed
    }
    if unresolved:
        raise ValueError(
            f"Selected terms contain unresolved shared inputs {sorted(unresolved, key=str)}; "
            "declare their raw data dependencies through Data or ModelTerm.data_vars."
        )
    coords = {
        dim: scenario.get_index(dim)
        if dim in scenario.dims
        else pd.Index(labels, name=dim, tupleize_cols=False)
        for dim, labels in model.coords.items()
        if labels is not None
    }
    if samples is not None:
        if "sample" in coords and not coords["sample"].equals(samples):
            raise ValueError(
                "Selected term sample labels must match the fitted joint posterior sample labels."
            )
        coords["sample"] = samples
    missing = {
        dim for output in outputs.values() for dim in output.dims if dim not in coords
    }
    if missing:
        raise ValueError(
            f"Selected outputs require labeled coordinates for dimensions {sorted(missing)}."
        )
    return _Evaluator(outputs, containers, templates, coords, fixed)
