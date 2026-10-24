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

"""The compiled-sweep cache behind :meth:`SensitivityAnalysis.run_sweep`.

A sweep grid is a function argument, so one compiled function serves every
grid of a given structure.  A :class:`SweepCache` keeps those functions, keyed
by the structure of the question, and a call leases one entry for its duration
so concurrent callers never rebind each other's posterior draws.

An entry owns its own :class:`~pymc_marketing.pytensor_utils.ConditionedModel`
clone of the source model, because the sweep graph must read the clone's own
data variables.  The clone freezes its data at construction, so the entry also
watches the source model's shared data buffers and is rebuilt when
``pm.set_data`` replaces one, which keeps the result identical to the
uncached path.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import xarray as xr
from pymc import Model
from pytensor import function
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors
from pytensor.tensor import TensorVariable
from pytensor.xtensor.type import XTensorVariable, as_xtensor
from pytensor.xtensor.vectorization import vectorize_graph

from pymc_marketing.pytensor_utils import ConditionedModel

SWEEP_DIM = "sweep"
"""Name of the dimension the sweep grid is vectorized over."""

SWEEP_NAME = "sweep"
"""Name of the symbolic sweep input every compiled entry takes."""

MASK_NAME = "mask"
"""Name of the symbolic response-mask input an entry with a mask takes."""

DEFAULT_MAX_ENTRIES = 4
"""Number of compiled sweeps a cache keeps before evicting the least recent."""


class SweepCloneError(Exception):
    """The source model cannot be cloned into a cache entry.

    :class:`~pymc_marketing.pytensor_utils.ConditionedModel` raises
    ``NotImplementedError`` or ``ValueError`` when PyMC cannot clone a model,
    as for custom initial values or nested sub-models.  Only that step is
    wrapped, so the caller can fall back to compiling per call without also
    catching errors raised while binding or evaluating an entry.
    """


class SweepCacheKey(NamedTuple):
    """Structure of a sweep question, which is what a compiled entry is keyed by.

    The sweep values and the mask values are function arguments, so they are
    not part of the key: a new grid or a new mask only recompiles when it
    changes the shape of the graph.  ``dims_order`` is kept for readability of
    the key; it is already implied by ``var_input``.
    """

    var_input: str
    var_names: str
    sweep_type: str
    mask_shape: tuple[int, ...] | None
    response_dims: tuple[str, ...]
    dims_order: tuple[str, ...]


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


def watched_source_buffers(model: Model, graph: Variable) -> dict[str, np.ndarray]:
    """Record the source-model data buffers ``graph`` reads, keyed by name.

    :class:`~pymc_marketing.pytensor_utils.ConditionedModel` clones the model,
    so the graph reads the clone's copies of the data variables, which never
    change again.  The source model, which ``pm.set_data`` mutates, is matched
    by name.  A graph node with no source-model counterpart, such as the shared
    posterior variables or an anonymous symbolic input, is not recorded.

    Parameters
    ----------
    model : Model
        The source model a caller hands to the analysis.
    graph : Variable
        The composed sweep graph to watch.

    Returns
    -------
    dict of str to np.ndarray
        The borrowed buffer of each source data variable the graph reads.
    """
    watched: dict[str, np.ndarray] = {}
    for node in ancestors([graph]):
        name = node.name
        if name is None or name in watched:
            continue
        source_var = model.named_vars.get(name)
        if isinstance(source_var, SharedVariable):
            watched[name] = source_var.get_value(borrow=True)
    return watched


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
        SweepCloneError
            If PyMC cannot clone ``model``, as for a model with custom initial
            values or nested sub-models.  The caller falls back to compiling
            per call, which raises any error that was really about ``idata``.
        """
        try:
            conditioned = ConditionedModel(model, idata)
        except (NotImplementedError, ValueError) as err:
            raise SweepCloneError(str(err)) from err
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
            watched_buffers=watched_source_buffers(model, graph),
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
        for name, buffer in self._watched_buffers.items():
            source_var = self._source_model.named_vars.get(name)
            if not isinstance(source_var, SharedVariable):
                return True
            if source_var.get_value(borrow=True) is not buffer:
                return True
        return False

    def set_posterior(self, idata: xr.DataTree | xr.Dataset) -> None:
        """Rebind the compiled function to the draws in ``idata``.

        The shared draws are swapped in place, so the number of draws may
        change and the function is not recompiled.  Everything the graphs read
        is validated against the construction layout, with coordinates
        realigned by label.

        Parameters
        ----------
        idata : xr.DataTree or xr.Dataset
            The posterior draws to evaluate under.
        """
        self._conditioned.set_posterior(idata)

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


class SweepLease:
    """One call's hold on a compiled sweep, taken out of the cache.

    A leased entry belongs to the call until :meth:`release`: nothing else can
    rebind the shared draws under it, so a concurrent call on the same key
    misses the cache and builds its own entry.  The entry is returned to the
    cache when the call is done, also when it raises.

    Parameters
    ----------
    cache : SweepCache
        The cache it was taken from and is returned to.
    key : SweepCacheKey
        The identity it is cached under.
    entry : SweepEntry
        The compiled sweep.
    cached : bool
        Whether it was taken from the cache rather than built for this call.
    """

    def __init__(
        self,
        cache: SweepCache,
        key: SweepCacheKey,
        entry: SweepEntry,
        *,
        cached: bool,
    ) -> None:
        self._cache = cache
        self._key = key
        self._entry = entry
        self.cached = cached
        self._released = False

    @property
    def entry(self) -> SweepEntry:
        """The compiled sweep this lease holds."""
        return self._entry

    def release(self) -> None:
        """Return the entry to the cache, once.

        Releasing twice is a no-op, so both a failure path and the caller's
        ``finally`` may release.  If the cache already holds an entry for this
        key -- another call finished first -- one is kept and the LRU stays
        bounded.
        """
        if self._released:
            return
        self._released = True
        self._cache.put(self._key, self._entry)


class SweepCache:
    """Bounded LRU cache of compiled sweeps, leased for the duration of a call.

    A :class:`~pymc_marketing.mmm.sensitivity_analysis.SensitivityAnalysis`
    keeps one, so repeated calls over the same structure pay one compile.  The
    cache is bounded so a model cannot pin an unbounded number of compiled
    functions, and it is never serialized because compiled functions only make
    sense in-process.

    A call leases an entry for its duration: :meth:`lease` takes the entry out
    of the cache and the call returns it in a ``finally``.  Concurrent calls on
    one cache never share the draws an entry is rebound to -- one of them
    misses the cache and builds its own -- and no call waits on another.  A
    lock guards only the dictionary bookkeeping, never an evaluation.  An entry
    whose compiled graph reads a source input replaced since its build is
    rebuilt rather than leased.

    An entry is only reused for the source model it was built on.
    Analyses on different models that share one cache and one key structure
    therefore rebuild each other's entry on every call, which gives correct
    results but no reuse.

    Parameters
    ----------
    max_entries : int, default 4
        Number of compiled sweeps kept before the least recently used is
        evicted.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[SweepCacheKey, SweepEntry] = OrderedDict()
        # Guards only the dictionary bookkeeping, never a call: a release must
        # not fail because another thread evicted its key between two dict
        # operations.
        self._lock = threading.Lock()

    def put(self, key: SweepCacheKey, entry: SweepEntry) -> None:
        """Store ``entry`` under ``key``, evicting the least recent entry.

        An entry already cached under ``key`` is replaced, so the cache never
        holds more than ``max_entries`` entries.

        Parameters
        ----------
        key : SweepCacheKey
            Identity to store it under.
        entry : SweepEntry
            The compiled sweep.
        """
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def lease(
        self,
        key: SweepCacheKey,
        source_model: Model,
        factory: Callable[[], SweepEntry],
    ) -> SweepLease:
        """Take the entry cached under ``key``, building one on a miss.

        The entry is removed from the cache and stays out until the lease is
        released, so a second call for the same key does not find it and
        builds its own instead of sharing draws with the first.  An entry whose
        compiled graph reads a source buffer replaced since it was built is
        treated as a miss, because the clone holds the old buffer and the
        factory rebuilds from the live data.

        Parameters
        ----------
        key : SweepCacheKey
            Identity to look up.
        source_model : Model
            The model the call is about to use; an entry built on another
            model is never leased.
        factory : callable
            Builds the entry when the key is absent or stale.

        Returns
        -------
        SweepLease
            The entry and whether it came from the cache.  A caller that
            rebinds a cached entry to new draws needs to know the difference: a
            freshly built one is already conditioned on the draws it was asked
            for.

        Raises
        ------
        SweepCloneError
            If the factory cannot clone ``source_model``.
        """
        with self._lock:
            entry = self._entries.pop(key, None)
        if entry is not None and not entry.is_stale(source_model):
            return SweepLease(self, key, entry, cached=True)
        return SweepLease(self, key, factory(), cached=False)

    def __len__(self) -> int:
        """Return the number of cached entries."""
        with self._lock:
            return len(self._entries)

    def __getstate__(self) -> dict[str, int]:
        """Skip the compiled sweeps when the owning analysis is pickled."""
        return {"_max_entries": self._max_entries}

    def __setstate__(self, state: dict[str, int]) -> None:
        """Restore ``max_entries`` with an empty cache."""
        self._max_entries = state["_max_entries"]
        self._entries = OrderedDict()
        self._lock = threading.Lock()
