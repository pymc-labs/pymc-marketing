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

"""The compiled sweep graph and the structure it is keyed by.

A sweep grid is a function argument, so one compiled function serves every grid
of a given structure.  :class:`SweepCacheKey` names that structure and
:class:`SweepEntry` owns the compiled function and the model clone it reads.
The cache that stores entries lives in :mod:`pymc_marketing.mmm.sweep_cache`;
this module holds no cache state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import xarray as xr
from pymc import Model
from pytensor import function
from pytensor.graph.basic import Variable
from pytensor.tensor import TensorVariable
from pytensor.xtensor.type import XTensorVariable, as_xtensor
from pytensor.xtensor.vectorization import vectorize_graph

from pymc_marketing.mmm.leased_cache import buffers_replaced, watched_source_buffers
from pymc_marketing.pytensor_utils import ConditionedModel, posterior_in_model_coords

SWEEP_DIM = "sweep"
"""Name of the dimension the sweep grid is vectorized over."""

SWEEP_NAME = "sweep"
"""Name of the symbolic sweep input every compiled entry takes."""

MASK_NAME = "mask"
"""Name of the symbolic response-mask input an entry with a mask takes."""


class SweepCacheKey(NamedTuple):
    """Structure of a sweep question, which is what a compiled entry is keyed by.

    The sweep values and the mask values are function arguments, so they are
    not part of the key: a new grid or a new mask only recompiles when it
    changes the shape of the graph.  The source model is checked separately
    because a cache belongs to one model.
    """

    var_input: str
    var_names: str
    sweep_type: str
    mask_shape: tuple[int, ...] | None


def apply_response_mask(
    response: XTensorVariable, mask: TensorVariable
) -> XTensorVariable:
    """Zero the response where a symbolic boolean mask is false.

    The mask is broadcast over the whole response graph, its sample batch
    included, because a ``set_subtensor`` with slices cannot be vectorized.
    Broadcasting the mask rather than the response keeps the mask a plain
    function argument of the compiled sweep.

    Parameters
    ----------
    response : XTensorVariable
        The extracted response graph, with a leading ``sample`` dimension.
    mask : TensorVariable
        Boolean symbolic input with the response's non-sample shape.

    Returns
    -------
    XTensorVariable
        The response with every masked-out element set to zero.
    """
    mask_bcast, _ = pt.broadcast_arrays(mask, response.values)
    return as_xtensor(response.values[~mask_bcast].set(0), dims=response.dims)


def build_sweep_response(
    base_value: XTensorVariable,
    response: XTensorVariable,
    sweep: XTensorVariable,
    sweep_type: str,
) -> XTensorVariable:
    """Compose the sweep graph: input combination, vectorization and date sum.

    Parameters
    ----------
    base_value : XTensorVariable
        The input the sweep replaces; the clone's own input in a cached entry.
    response : XTensorVariable
        The extracted response graph, already masked when a mask is used.
    sweep : XTensorVariable
        The symbolic sweep grid.
    sweep_type : str
        One of ``"multiplicative"``, ``"additive"`` or ``"absolute"``.

    Returns
    -------
    XTensorVariable
        The response summed over ``date``, transposed to ``sample, sweep, ...``.

    Raises
    ------
    ValueError
        If ``sweep_type`` is not one of the three supported types.
    """
    match sweep_type:
        case "multiplicative":
            sweep_combinations = sweep * base_value
        case "additive":
            sweep_combinations = sweep + base_value
        case "absolute":
            sweep_combinations = sweep.broadcast_like(base_value)
        case _:
            raise ValueError(f"Unknown sweep_type {sweep_type!r}")

    sweep_response = vectorize_graph(response, replace={base_value: sweep_combinations})
    return sweep_response.sum(dim="date").transpose("sample", "sweep", ...)


class SweepEntry:
    """One compiled sweep graph, conditioned on its own clone of the model.

    The entry holds the clone, so a rebind cannot touch another entry's draws,
    and the symbolic ``sweep`` and ``mask`` inputs, because the compiled
    function only accepts the handles it was built with.  The sweep values and
    the mask values stay plain function arguments.

    Parameters
    ----------
    source_model : Model
        The model the clone was taken from.
    conditioned : ConditionedModel
        The clone, which owns the posterior binding of the compiled function.
    compiled : callable
        The compiled sweep function, taking the sweep grid and, when present,
        the mask.
    sweep_input : XTensorVariable
        The symbolic sweep grid the compiled function was built with.
    mask_input : Variable or None
        The symbolic mask the compiled function was built with, when one is used.
    output_dims : tuple of str
        Dimension order of the compiled output.
    watched_buffers : dict of str to np.ndarray
        The source data buffers the compiled graph reads.
    """

    def __init__(
        self,
        *,
        source_model: Model,
        conditioned: ConditionedModel,
        compiled: Callable,
        sweep_input: XTensorVariable,
        mask_input: Variable | None,
        output_dims: tuple[str, ...],
        watched_buffers: dict[str, np.ndarray],
    ) -> None:
        self._source_model = source_model
        self._conditioned = conditioned
        self._compiled: Callable[..., np.ndarray] = compiled
        self._sweep_input = sweep_input
        self._mask_input = mask_input
        self.output_dims = output_dims
        self._watched_buffers = watched_buffers
        # The labels each read variable was built on, by dimension.  A rebind
        # must land on these, not merely on the same set of labels: realigning
        # a permutation of them would read the new draws by position while an
        # entry built on them read by position too, and the two orders differ.
        self._built_on_labels = {
            name: {
                str(dim): index
                for dim, index in conditioned.posterior[name].indexes.items()
                if dim not in ("chain", "draw")
            }
            for name in conditioned.shared_posterior.variables
        }

    @classmethod
    def build(
        cls, model: Model, idata: xr.DataTree | xr.Dataset, key: SweepCacheKey
    ) -> SweepEntry:
        """Clone ``model``, compose the sweep graph for ``key`` and compile it.

        The graph is built on the clone's own input, because an input of the
        source model is refused by
        :meth:`~pymc_marketing.pytensor_utils.ConditionedModel.extract`.

        Parameters
        ----------
        model : Model
            The source model, cloned here and never modified.
        idata : xr.DataTree or xr.Dataset
            Posterior draws the clone is conditioned on.
        key : SweepCacheKey
            The structure of the sweep to build.

        Returns
        -------
        SweepEntry
            The compiled entry.

        Raises
        ------
        ModelCloneError
            If PyMC cannot clone ``model``, as for a model with custom initial
            values or a nested sub-model.  The caller falls back to compiling
            per call, which raises any error that was really about ``idata``.
        """
        conditioned = ConditionedModel(model, idata)
        sweep = ptx.xtensor(SWEEP_NAME, dims=(SWEEP_DIM,))
        mask = (
            None
            if key.mask_shape is None
            else pt.tensor(MASK_NAME, shape=key.mask_shape, dtype=bool)
        )
        response = conditioned.extract(key.var_names)
        if mask is not None:
            response = apply_response_mask(response, mask)
        base_value: XTensorVariable = conditioned.model[key.var_input]
        graph = build_sweep_response(base_value, response, sweep, key.sweep_type)
        inputs = [sweep] if mask is None else [sweep, mask]
        return cls(
            source_model=model,
            conditioned=conditioned,
            compiled=function(inputs, graph),
            sweep_input=sweep,
            mask_input=mask,
            output_dims=graph.dims,
            watched_buffers=watched_source_buffers(model, [graph]),
        )

    def is_stale(self, source_model: Model) -> bool:
        """Whether the entry no longer matches the model data it was built on.

        Each watched variable costs one borrowed-buffer identity check and no
        value comparison.  ``pm.set_data`` replaces the buffer, which is what
        this sees; an in-place write to the buffer itself is not detected.  A
        buffer replaced by equal values still counts as a change, and so does a
        different source model.

        Parameters
        ----------
        source_model : Model
            The model a call is about to use.

        Returns
        -------
        bool
            ``True`` when the entry must be rebuilt rather than leased.
        """
        if source_model is not self._source_model:
            return True
        return buffers_replaced(self._source_model, self._watched_buffers)

    def set_posterior(self, idata: xr.DataTree | xr.Dataset) -> None:
        """Rebind the compiled function to the draws in ``idata``.

        The shared draws are swapped in place, so the number of draws may
        change and the function is not recompiled.  Everything the graphs read
        is validated against the construction layout, with coordinates
        realigned by label, and each variable must keep the labels this entry
        was built on.  A posterior whose labels are renamed, reordered or
        dropped on one side is refused rather than bound by position, so the
        caller rebuilds and answers what an empty cache would.

        Parameters
        ----------
        idata : xr.DataTree or xr.Dataset
            The posterior draws to evaluate under.

        Raises
        ------
        KeyError
            If the posterior lacks a variable the compiled graph reads.
        ValueError
            If a variable the compiled graph reads differs in its non-sample
            dimensions, their lengths or their labels from the ones this entry
            was built on.
        """
        posterior = posterior_in_model_coords(idata, self._conditioned.model.coords)
        self._raise_if_relabelled(posterior)
        self._conditioned.set_posterior(posterior)

    def _raise_if_relabelled(self, posterior: xr.Dataset) -> None:
        """Refuse a posterior that reorders or renames a label of a read variable.

        A variable missing from ``posterior`` is skipped, so
        :meth:`ConditionedModel.set_posterior` reports it in its own words.
        """
        for name, built_on in self._built_on_labels.items():
            if name not in posterior.data_vars:
                continue
            for dim, labels in built_on.items():
                given = posterior[name].indexes.get(dim)
                if given is not None and not given.equals(labels):
                    raise ValueError(
                        f"Posterior variable {name!r}: dimension {dim!r} labels "
                        "differ from the ones this entry was built on."
                    )

    def evaluate(
        self, sweep_values: np.ndarray, mask_values: np.ndarray | None
    ) -> np.ndarray:
        """Evaluate the compiled sweep under the currently bound posterior.

        Parameters
        ----------
        sweep_values : np.ndarray
            The sweep grid, a function argument so no recompile is needed.
        mask_values : np.ndarray or None
            The response mask, passed only when the entry was built with one.

        Returns
        -------
        np.ndarray
            The sweep result, with the entry's ``output_dims`` layout.
        """
        if self._mask_input is None:
            return self._compiled(sweep_values)
        return self._compiled(sweep_values, mask_values)
