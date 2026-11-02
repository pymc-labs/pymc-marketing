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

"""Storage shared by the caches whose entries a request leases.

The compiled-evaluator cache and the compiled-sweep cache differ in what they
lease and in how they decide an entry is stale, so each keeps its own
:meth:`lease` signature and lease class.  They share the storage in
:class:`LeasedLRUCache` and the buffer checks a leased entry uses to notice that
the model data it was built on was replaced.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Collection, Iterable, Mapping

import numpy as np
from pymc.model.core import Model
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors


class LeasedLRUCache[K, V]:
    """Bounded LRU storage for entries a request holds for its duration.

    A cache takes an entry out with its own ``lease`` method and the request
    returns it in a ``finally``, so this class only stores entries and bounds
    how many are kept.  The lock guards the dictionary bookkeeping, never an
    evaluation.  A cache is never serialized with its entries, because compiled
    functions only make sense in-process; the bound is kept across a pickle or
    a deepcopy.

    Parameters
    ----------
    max_entries : int, default 4
        Number of entries kept before the least recently used is evicted.

    Raises
    ------
    ValueError
        If ``max_entries`` is negative.
    """

    def __init__(self, max_entries: int = 4) -> None:
        if max_entries < 0:
            raise ValueError(f"max_entries must be non-negative, got {max_entries}.")
        self._max_entries = max_entries
        self._entries: OrderedDict[K, V] = OrderedDict()
        # Guards only the dictionary bookkeeping, never a request: a release must
        # not fail because another thread evicted its key between two dict
        # operations.
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        """Return the entry cached under ``key``, marking it most recent.

        Parameters
        ----------
        key : K
            Identity to look up.

        Returns
        -------
        V or None
            The cached entry, which stays in the cache, or ``None`` when
            ``key`` is absent.  A request takes an entry out with its cache's
            ``lease`` method instead.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key: K, entry: V) -> None:
        """Store ``entry`` under ``key``, evicting the least recent entry.

        An entry already cached under ``key`` is replaced, so the cache never
        holds more than ``max_entries`` entries.

        Parameters
        ----------
        key : K
            Identity to store it under.
        entry : V
            The entry to cache.
        """
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def _take(self, key: K) -> V | None:
        """Remove and return the entry cached under ``key``, if any.

        The caller leases the returned entry, so it stays out of the cache until
        its lease is released.
        """
        with self._lock:
            return self._entries.pop(key, None)

    def __len__(self) -> int:
        """Return the number of cached entries."""
        with self._lock:
            return len(self._entries)

    def __getstate__(self) -> dict[str, int]:
        """Skip the cached entries when the owning object is pickled."""
        return {"_max_entries": self._max_entries}

    def __setstate__(self, state: dict[str, int]) -> None:
        """Restore ``max_entries`` with an empty cache."""
        self._max_entries = state["_max_entries"]
        self._entries = OrderedDict()
        self._lock = threading.Lock()


def watched_source_buffers(
    source_model: Model,
    graphs: Iterable[Variable],
    *,
    exclude: Collection[str] = (),
) -> dict[str, np.ndarray]:
    """Record the source-model data buffers ``graphs`` read, keyed by name.

    :class:`~pymc_marketing.pytensor_utils.ConditionedModel` clones the model,
    so the graphs read the clone's copies of the data variables, which never
    change again.  The source model, which ``pm.set_data`` mutates, is matched
    by name.  A graph node with no source-model counterpart, such as the shared
    posterior variables or an anonymous symbolic input, is not recorded.

    Parameters
    ----------
    source_model : Model
        The model the graphs were cloned from.
    graphs : iterable of Variable
        Graph outputs whose ancestors are scanned.
    exclude : collection of str, default ()
        Names never to record, for inputs a caller reads another way.

    Returns
    -------
    dict of str to np.ndarray
        The borrowed buffer of each source data variable the graphs read.
    """
    watched: dict[str, np.ndarray] = {}
    for node in ancestors(graphs):
        name = node.name
        if name is None or name in watched or name in exclude:
            continue
        source_var = source_model.named_vars.get(name)
        if isinstance(source_var, SharedVariable):
            watched[name] = source_var.get_value(borrow=True)
    return watched


def buffers_replaced(source_model: Model, watched: Mapping[str, np.ndarray]) -> bool:
    """Whether a source-model data buffer changed since it was recorded.

    Each watched variable costs one borrowed-buffer identity check and no value
    comparison.  ``pm.set_data`` replaces the buffer, which is what this sees;
    an in-place write to the buffer itself is not detected.  A buffer replaced
    by equal values still counts as a change, and a variable that is no longer
    a shared input counts as one too.

    Parameters
    ----------
    source_model : Model
        The model a call is about to use.
    watched : dict of str to np.ndarray
        Buffers recorded by :func:`watched_source_buffers`.

    Returns
    -------
    bool
        ``True`` when a watched source variable no longer holds the buffer
        recorded at build time, so the compiled graphs read stale values.
    """
    for name, buffer in watched.items():
        source_var = source_model.named_vars.get(name)
        if not isinstance(source_var, SharedVariable):
            return True
        if source_var.get_value(borrow=True) is not buffer:
            return True
    return False
