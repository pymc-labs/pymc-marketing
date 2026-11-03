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
:meth:`lease` signature.  They share the storage in :class:`LeasedLRUCache`
and the lease a request holds in :class:`Lease`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

DEFAULT_MAX_ENTRIES = 4
"""Number of entries a leased cache keeps before evicting the least recent."""


class Lease[V]:
    """One request's hold on a cached entry, taken out of the cache.

    A leased entry belongs to the request until :meth:`release`: nothing else
    can use it while it is out, so a concurrent request on the same key misses
    the cache and builds or leases a copy of its own.  The entry is returned to
    the cache when the request is done, also when it raises, unless the request
    drops it with :meth:`discard`.

    Parameters
    ----------
    cache : LeasedLRUCache
        The cache it was taken from and is returned to.
    key : object
        The identity it is cached under.
    value : V
        The cached entry it holds.
    cached : bool
        Whether it was taken from the cache rather than built for this request.
    """

    def __init__(
        self,
        cache: LeasedLRUCache[Any, V],
        key: Any,
        value: V,
        *,
        cached: bool,
    ) -> None:
        self._cache = cache
        self._key = key
        self._value = value
        self.cached = cached
        self._released = False

    @property
    def value(self) -> V:
        """The cached entry this lease holds."""
        return self._value

    def release(self) -> None:
        """Return the entry to the cache, once.

        Releasing twice is a no-op, so both a failure path and the caller's
        ``finally`` may release.  If the cache already holds an entry for this
        key -- another request finished first -- one is kept and the LRU stays
        bounded.
        """
        if self._released:
            return
        self._released = True
        self._cache.put(self._key, self._value)

    def discard(self) -> None:
        """Drop the entry instead of returning it to the cache.

        An entry that cannot answer this request was built for another
        structure, so keeping it would make every later request fail the same
        way.  A later :meth:`release` is a no-op; the caller builds a fresh
        entry in place of this one.
        """
        self._released = True


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

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
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
